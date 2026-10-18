#include "ultima/server/http_server.hpp"

// httplib pulls in <windows.h> on Windows; keep our headers first to avoid
// symbol pollution creeping into other TUs. Do NOT define
// CPPHTTPLIB_OPENSSL_SUPPORT here — the header uses `#ifdef` so any
// definition (even to 0) enables OpenSSL includes we don't want in v0.1.
#include <httplib.h>

#include <nlohmann/json.hpp>

#include "ultima/kv_cache/kv_cache.hpp"
#include "ultima/model/imodel.hpp"
#include "ultima/sampler/sampler.hpp"
#include "ultima/tokenizer/chat_template.hpp"
#include "ultima/server/permissions.hpp"
#include "ultima/server/session.hpp"
#include "ultima/server/skills.hpp"
#include "ultima/server/tools.hpp"
#include "ultima/server/workspace.hpp"
#include "ultima/tokenizer/tokenizer.hpp"

#include <atomic>
#include <condition_variable>
#include <optional>
#include <unordered_map>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace ultima::server {

namespace {

using core::ErrorCode;
using core::Failure;
using core::Result;
using core::fail;
using json = nlohmann::json;

// ---- Request bodies --------------------------------------------------------

std::vector<tokenizer::ChatMessage>
parse_messages(const json& j) {
    std::vector<tokenizer::ChatMessage> msgs;
    if (!j.is_array()) return msgs;
    msgs.reserve(j.size());
    for (const auto& m : j) {
        tokenizer::ChatMessage cm;
        cm.role    = m.value("role", "user");
        cm.content = m.value("content", "");
        cm.name    = m.value("name", "");
        msgs.push_back(std::move(cm));
    }
    return msgs;
}

sampler::SamplerParams parse_sampler(const json& body) {
    sampler::SamplerParams p;
    if (body.contains("temperature"))       p.temperature       = body["temperature"].get<float>();
    if (body.contains("top_p"))             p.top_p             = body["top_p"].get<float>();
    if (body.contains("top_k"))             p.top_k             = body["top_k"].get<std::size_t>();
    if (body.contains("min_p"))             p.min_p             = body["min_p"].get<float>();
    if (body.contains("repetition_penalty"))p.rep_penalty       = body["repetition_penalty"].get<float>();
    if (body.contains("frequency_penalty")) p.frequency_penalty = body["frequency_penalty"].get<float>();
    if (body.contains("presence_penalty"))  p.presence_penalty  = body["presence_penalty"].get<float>();
    if (body.contains("seed")) {
        // JSON number -> uint64. Some clients send negative -> treat as random.
        auto n = body["seed"];
        if (n.is_number_integer()) {
            const auto v = n.get<std::int64_t>();
            if (v >= 0) p.seed = static_cast<std::uint64_t>(v);
        }
    }
    return p;
}

// ---- Chat completion driver -----------------------------------------------

struct ChatResult {
    std::string           text;
    std::size_t           prompt_tokens;
    std::size_t           completion_tokens;
    std::uint64_t         seed_used;
    double                first_token_ms;
    double                total_ms;
};

// Core generation loop. `on_delta` receives the newly produced UTF-8 piece
// after each emitted token (multi-byte codepoints split across two tokens
// come out correctly because we diff cumulative-decode outputs). Pass a
// no-op callback for non-streaming — the accumulated text is available in
// the returned ChatResult either way.
Result<ChatResult>
run_chat_stream(RuntimeContext& ctx,
                const std::vector<tokenizer::ChatMessage>& msgs,
                const sampler::SamplerParams& user_params,
                std::size_t max_tokens,
                const std::function<void(std::string_view)>& on_delta,
                const std::function<bool()>& should_abort = {}) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    // 1. Render + tokenize.
    const std::string prompt = tokenizer::render_chatml(ctx.model.arch(), msgs, true);
    auto prompt_ids = ctx.tokenizer.encode(prompt, /*allow_special=*/true);
    if (prompt_ids.empty()) {
        return fail(ErrorCode::InvalidModel, "empty prompt after tokenization", "server");
    }

    // 2. Acquire slot + prefix reuse.
    static std::atomic<std::uint64_t> session_counter{1};
    const auto sid = session_counter.fetch_add(1);
    auto slot = ctx.kv.acquire(sid);

    // A hybrid model's recurrent layers cannot resume from a truncated
    // prefix — their state has no per-position structure to keep. Such a
    // model re-prefills from zero instead; reuse_prefix would leave the
    // recurrence holding tokens the KV cache no longer agrees with.
    const bool reuse = ctx.model.supports_prefix_reuse();
    if (reuse) {
        ctx.kv.reuse_prefix(slot, prompt_ids);
    } else {
        ctx.kv.release(slot);
        slot = ctx.kv.acquire(sid);
    }
    const std::size_t already = reuse ? ctx.kv.pos(slot) : 0;

    std::vector<float> logits(ctx.model.dims().vocab, 0.0f);

    // 3. Prefill any remaining prompt tokens.
    if (already < prompt_ids.size()) {
        std::span<const std::int32_t> tail(
            prompt_ids.data() + already, prompt_ids.size() - already);
        if (auto r = ctx.model.prefill(ctx.kv, slot, tail, logits.data()); !r) {
            ctx.kv.release(slot);
            return Failure{r.error()};
        }
    } else {
        // Every prompt token is already cached, so rewind by one and decode
        // the last. Only reachable when prefix reuse applies at all.
        const auto last = prompt_ids.back();
        ctx.kv.reuse_prefix(slot, std::span<const std::int32_t>(prompt_ids.data(),
                                                                prompt_ids.size() - 1));
        if (auto r = ctx.model.decode(ctx.kv, slot, last, logits.data()); !r) {
            ctx.kv.release(slot);
            return Failure{r.error()};
        }
    }

    const auto t_first = clock::now();

    // 4. Sampler + decode loop.
    sampler::SamplerParams params = user_params;
    sampler::SamplerContext sctx =
        params.seed ? sampler::SamplerContext{*params.seed}
                    : sampler::SamplerContext{};
    sctx.set_recent(std::span<const std::int32_t>(prompt_ids.data(),
                                                  prompt_ids.size()));

    std::vector<std::int32_t> out_ids;
    out_ids.reserve(max_tokens);
    std::string accumulated;
    accumulated.reserve(max_tokens * 4);

    const auto eos = ctx.tokenizer.eos_id();
    const auto im_end = ctx.tokenizer.im_end_id();

    for (std::size_t i = 0; i < max_tokens; ++i) {
        // Checked every token so a hung-up client stops the work immediately
        // rather than after the whole budget. The model mutex is held for the
        // duration of a turn, so a wasted generation blocks every later request.
        if (should_abort && should_abort()) break;
        const std::int32_t next = sampler::sample(std::span<float>(logits), params, sctx);
        if (next < 0) break;
        if (next == eos || next == im_end) break;
        out_ids.push_back(next);
        sctx.push_recent(next);

        // Re-decode the full sequence and emit only the new UTF-8 suffix.
        // Cheap relative to a model forward pass; ensures multi-byte
        // codepoints that span two BPE tokens still come out clean.
        const std::string full = ctx.tokenizer.decode(out_ids, /*skip_special=*/true);
        if (full.size() > accumulated.size()) {
            std::string_view delta{full.data() + accumulated.size(),
                                   full.size() - accumulated.size()};
            if (on_delta) on_delta(delta);
            accumulated.assign(full);
        }

        if (i + 1 == max_tokens) break;
        if (auto r = ctx.model.decode(ctx.kv, slot, next, logits.data()); !r) {
            ctx.kv.release(slot);
            return Failure{r.error()};
        }
    }
    ctx.kv.release(slot);

    const auto t_end = clock::now();
    const double first_ms = std::chrono::duration<double, std::milli>(t_first - t0).count();
    const double total_ms = std::chrono::duration<double, std::milli>(t_end   - t0).count();

    ChatResult res;
    res.text              = std::move(accumulated);
    res.prompt_tokens     = prompt_ids.size();
    res.completion_tokens = out_ids.size();
    res.seed_used         = sctx.seed();
    res.first_token_ms    = first_ms;
    res.total_ms          = total_ms;
    return res;
}

// Non-streaming shim: preserves the old signature for the JSON handler.
Result<ChatResult>
run_chat_completion(RuntimeContext& ctx,
                    const std::vector<tokenizer::ChatMessage>& msgs,
                    const sampler::SamplerParams& user_params,
                    std::size_t max_tokens) {
    return run_chat_stream(ctx, msgs, user_params, max_tokens, {});
}

// ---- JSON response builders -----------------------------------------------

// ---- SSE chunk helpers -----------------------------------------------------
// OpenAI streaming shape: repeated `data: {...}\n\n` chunks, terminated by
// `data: [DONE]\n\n`. Each JSON chunk carries a `delta` object; the first
// chunk sets role=assistant, subsequent chunks append content, and the
// final chunk sets finish_reason=stop.

std::string sse_frame(const json& j) {
    return "data: " + j.dump() + "\n\n";
}

json sse_role_chunk(const std::string& model_name, std::int64_t created_ts) {
    return {
        {"id",      "chatcmpl-ultima"},
        {"object",  "chat.completion.chunk"},
        {"created", created_ts},
        {"model",   model_name},
        {"choices", json::array({
            { {"index", 0}, {"delta", {{"role", "assistant"}}}, {"finish_reason", nullptr} }
        })}
    };
}

json sse_content_chunk(const std::string& model_name, std::int64_t created_ts,
                       std::string_view delta) {
    return {
        {"id",      "chatcmpl-ultima"},
        {"object",  "chat.completion.chunk"},
        {"created", created_ts},
        {"model",   model_name},
        {"choices", json::array({
            { {"index", 0}, {"delta", {{"content", std::string(delta)}}}, {"finish_reason", nullptr} }
        })}
    };
}

json sse_stop_chunk(const std::string& model_name, std::int64_t created_ts,
                    const ChatResult& cr) {
    return {
        {"id",      "chatcmpl-ultima"},
        {"object",  "chat.completion.chunk"},
        {"created", created_ts},
        {"model",   model_name},
        {"choices", json::array({
            { {"index", 0}, {"delta", json::object()}, {"finish_reason", "stop"} }
        })},
        {"usage", {
            {"prompt_tokens",     cr.prompt_tokens},
            {"completion_tokens", cr.completion_tokens},
            {"total_tokens",      cr.prompt_tokens + cr.completion_tokens}
        }},
        {"ultima", {
            {"seed",           cr.seed_used},
            {"first_token_ms", cr.first_token_ms},
            {"total_ms",       cr.total_ms},
            {"decode_toks",    cr.completion_tokens},
            {"decode_tps",     cr.total_ms > 0.0
                                   ? (1000.0 * static_cast<double>(cr.completion_tokens) / cr.total_ms)
                                   : 0.0}
        }}
    };
}

json openai_chat_response(const std::string& model_name,
                          const ChatResult&  cr) {
    const std::int64_t now = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    return {
        {"id",      "chatcmpl-ultima"},
        {"object",  "chat.completion"},
        {"created", now},
        {"model",   model_name},
        {"choices", json::array({
            {
                {"index", 0},
                {"message", {
                    {"role", "assistant"},
                    {"content", cr.text}
                }},
                {"finish_reason", "stop"}
            }
        })},
        {"usage", {
            {"prompt_tokens",     cr.prompt_tokens},
            {"completion_tokens", cr.completion_tokens},
            {"total_tokens",      cr.prompt_tokens + cr.completion_tokens}
        }},
        {"ultima", {
            {"seed",           cr.seed_used},
            {"first_token_ms", cr.first_token_ms},
            {"total_ms",       cr.total_ms},
            {"decode_toks",    cr.completion_tokens},
            {"decode_tps",     cr.total_ms > 0.0
                                   ? (1000.0 * static_cast<double>(cr.completion_tokens) / cr.total_ms)
                                   : 0.0}
        }}
    };
}

} // namespace

// ---- Testpad bundle --------------------------------------------------------
// A tiny vanilla-JS single-file testpad. Decision 14 §14.5 — chat pane,
// sampler panel with coding defaults, live tok/s. Kept intentionally
// minimal in v0.1 so the server ships without a build-time UI toolchain.
// Defined in the CMake-generated testpad_bundle_generated.cpp (or the
// fallback placeholder in testpad_bundle.cpp).
extern const std::string_view kTestpadHtml;

// Request helpers.
//
// These are free functions on purpose. They were previously lambdas local to
// the HttpServer constructor, captured by reference into each route handler —
// but handlers run long after the constructor returns, so those references
// dangled. It worked by luck until the stack layout changed.

// Bearer check. An empty configured token disables auth (local-only default).
bool check_auth_token(const std::string& token,
                      const httplib::Request& req, httplib::Response& res) {
    if (token.empty()) return true;
    auto it = req.headers.find("Authorization");
    const std::string expected = "Bearer " + token;
    if (it == req.headers.end() || it->second != expected) {
        res.status = 401;
        res.set_content(json{{"error", "unauthorized"}}.dump(), "application/json");
        return false;
    }
    return true;
}

json plan_to_json(const Plan& plan) {
    json phases = json::array();
    for (const auto& p : plan.phases) {
        json todos = json::array();
        for (const auto& t : p.todos) {
            todos.push_back({{"id", t.id}, {"text", t.text}, {"done", t.done}});
        }
        phases.push_back({{"number", p.number}, {"text", p.text},
                          {"done", p.done}, {"todos", todos}});
    }
    const Phase* act = plan.active();
    return {
        {"title",     plan.title},
        {"requested", plan.requested},
        {"started",   plan.started},
        {"phases",    phases},
        {"active",    act ? json(act->number) : json(nullptr)},
        {"done",      plan.done_count()},
    };
}

// Map a workspace failure onto an HTTP status: a refused or malformed path is
// the caller's fault, a missing file is 404, anything else is ours.
int fs_status_for(const core::Error& e) {
    switch (e.code) {
    case ErrorCode::InvalidArgument: return 400;
    case ErrorCode::NotFound:
    case ErrorCode::FileNotFound:    return 404;
    default:                         return 500;
    }
}

void fs_send_error(httplib::Response& res, const core::Error& e) {
    res.status = fs_status_for(e);
    res.set_content(json{{"error", e.message}}.dump(), "application/json");
}

// The workspace is opened at listen time; a request that arrives before that,
// or after a failed open, gets a clear 503 instead of dereferencing nothing.
Workspace* fs_ready_or_503(std::optional<Workspace>& ws, httplib::Response& res) {
    if (!ws) {
        res.status = 503;
        res.set_content(json{{"error", "no workspace is open"}}.dump(),
                        "application/json");
        return nullptr;
    }
    return &*ws;
}

struct HttpServer::Impl {
    Impl(RuntimeContext c, ServerConfig g) : ctx{std::move(c)}, cfg{std::move(g)} {}

    RuntimeContext ctx;
    ServerConfig   cfg;
    httplib::Server svr;
    std::mutex     model_mutex;   // serializes chat requests (one slot pool in v0.1)
    std::atomic<bool> stopping{false};

    // Opened lazily in listen_blocking() so a bad --workspace is reported as a
    // startup error rather than thrown from a constructor.
    std::optional<Workspace> ws;
    SkillIndex     skills;        // discovered once at listen time
    std::mutex     ws_mutex;      // file ops are not otherwise serialized

    // ---- Permissions (PERMISSIONS.md) ----
    // A prompt is emitted on the agent's SSE stream and answered by a separate
    // POST, so the generating thread parks on `perm_cv` until the answer lands
    // or the wait times out. Answers live only for the process.
    PermissionMode          perm_mode = PermissionMode::Auto;
    std::vector<Risk>       perm_session_allowed;
    std::mutex              perm_mutex;
    std::condition_variable perm_cv;
    std::unordered_map<std::string, std::string> perm_answers;  // id -> allow|always|deny
    std::uint64_t           perm_next_id = 1;
};

HttpServer::HttpServer(RuntimeContext ctx, ServerConfig cfg)
    : impl_{std::make_unique<Impl>(std::move(ctx), std::move(cfg))} {
    auto& svr = impl_->svr;
    auto& state = *impl_;


    // ---- CORS (localhost dev) ----
    svr.set_pre_routing_handler([&](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin",  "*");
        res.set_header("Access-Control-Allow-Methods",
                       "GET, POST, PUT, PATCH, DELETE, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        if (req.method == "OPTIONS") {
            res.status = 204;
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    // ---- /api/health ----
    svr.Get("/api/health", [&](const httplib::Request& req, httplib::Response& res) {
        if (!check_auth_token(state.cfg.auth_token, req, res)) return;
        json j = {
            {"status",   "ok"},
            {"model",    state.ctx.model_name},
            {"arch",     state.ctx.model.arch()},
            {"kv_slots", {
                {"total",   state.ctx.kv.config().n_slots},
                {"in_use",  state.ctx.kv.n_active_slots()}
            }}
        };
        res.set_content(j.dump(), "application/json");
    });

    // ---- /v1/models ----
    svr.Get("/v1/models", [&](const httplib::Request& req, httplib::Response& res) {
        if (!check_auth_token(state.cfg.auth_token, req, res)) return;
        const std::int64_t now = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        json j = {
            {"object", "list"},
            {"data", json::array({
                {
                    {"id",       state.ctx.model_name},
                    {"object",   "model"},
                    {"created",  now},
                    {"owned_by", "ultima"}
                }
            })}
        };
        res.set_content(j.dump(), "application/json");
    });

    // ---- /api/fs/* — Coder IDE file operations (Decision 18 §18.2) --------
    //
    // Every path in every handler below goes through Workspace::resolve(),
    // which is the one place a client string becomes a filesystem path. A
    // refusal is a 400 naming the reason; nothing is ever clamped into range.

    svr.Get("/api/fs/tree", [&](const httplib::Request& req, httplib::Response& res) {
        if (!check_auth_token(state.cfg.auth_token, req, res)) return;
        std::lock_guard<std::mutex> lk{state.ws_mutex};
        auto* ws = fs_ready_or_503(state.ws, res);
        if (!ws) return;

        const bool recursive = req.get_param_value("recursive") == "1";
        const std::string path = req.get_param_value("path");

        auto entries = recursive ? ws->list_recursive() : ws->list(path);
        if (!entries) { fs_send_error(res, entries.error()); return; }

        json arr = json::array();
        for (const auto& e : *entries) {
            arr.push_back({{"path", e.path}, {"is_dir", e.is_dir},
                           {"size", e.size}, {"mtime", e.mtime}});
        }
        res.set_content(json{{"root", ws->root().generic_string()},
                             {"entries", arr}}.dump(), "application/json");
    });

    svr.Get("/api/fs/file", [&](const httplib::Request& req, httplib::Response& res) {
        if (!check_auth_token(state.cfg.auth_token, req, res)) return;
        std::lock_guard<std::mutex> lk{state.ws_mutex};
        auto* ws = fs_ready_or_503(state.ws, res);
        if (!ws) return;

        const std::string path = req.get_param_value("path");
        auto got = ws->read(path);
        if (!got) { fs_send_error(res, got.error()); return; }
        res.set_content(json{{"path", path},
                             {"content", got->content},
                             {"truncated", got->truncated}}.dump(),
                        "application/json");
    });

    svr.Put("/api/fs/file", [&](const httplib::Request& req, httplib::Response& res) {
        if (!check_auth_token(state.cfg.auth_token, req, res)) return;
        std::lock_guard<std::mutex> lk{state.ws_mutex};
        auto* ws = fs_ready_or_503(state.ws, res);
        if (!ws) return;
        try {
            const auto body = json::parse(req.body);
            const std::string path    = body.at("path").get<std::string>();
            const std::string content = body.value("content", std::string{});
            if (auto w = ws->write(path, content); !w) { fs_send_error(res, w.error()); return; }
            res.set_content(json{{"path", path}, {"bytes", content.size()}}.dump(),
                            "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    svr.Patch("/api/fs/file", [&](const httplib::Request& req, httplib::Response& res) {
        if (!check_auth_token(state.cfg.auth_token, req, res)) return;
        std::lock_guard<std::mutex> lk{state.ws_mutex};
        auto* ws = fs_ready_or_503(state.ws, res);
        if (!ws) return;
        try {
            const auto body = json::parse(req.body);
            const std::string path = body.at("path").get<std::string>();
            std::vector<Workspace::Edit> edits;
            for (const auto& e : body.at("edits")) {
                edits.push_back({e.at("find").get<std::string>(),
                                 e.value("replace", std::string{}),
                                 e.value("all", false)});
            }
            auto n = ws->edit(path, edits);
            if (!n) { fs_send_error(res, n.error()); return; }
            res.set_content(json{{"path", path}, {"replacements", *n}}.dump(),
                            "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    svr.Post("/api/fs/rename", [&](const httplib::Request& req, httplib::Response& res) {
        if (!check_auth_token(state.cfg.auth_token, req, res)) return;
        std::lock_guard<std::mutex> lk{state.ws_mutex};
        auto* ws = fs_ready_or_503(state.ws, res);
        if (!ws) return;
        try {
            const auto body = json::parse(req.body);
            const std::string from = body.at("from").get<std::string>();
            const std::string to   = body.at("to").get<std::string>();
            if (auto r = ws->rename(from, to); !r) { fs_send_error(res, r.error()); return; }
            res.set_content(json{{"from", from}, {"to", to}}.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    svr.Delete("/api/fs/file", [&](const httplib::Request& req, httplib::Response& res) {
        if (!check_auth_token(state.cfg.auth_token, req, res)) return;
        std::lock_guard<std::mutex> lk{state.ws_mutex};
        auto* ws = fs_ready_or_503(state.ws, res);
        if (!ws) return;

        const std::string path = req.get_param_value("path");
        if (auto r = ws->remove(path); !r) { fs_send_error(res, r.error()); return; }
        res.set_content(json{{"path", path}, {"deleted", true}}.dump(),
                        "application/json");
    });

    svr.Post("/api/fs/mkdir", [&](const httplib::Request& req, httplib::Response& res) {
        if (!check_auth_token(state.cfg.auth_token, req, res)) return;
        std::lock_guard<std::mutex> lk{state.ws_mutex};
        auto* ws = fs_ready_or_503(state.ws, res);
        if (!ws) return;
        try {
            const auto body = json::parse(req.body);
            const std::string path = body.at("path").get<std::string>();
            if (auto r = ws->mkdir(path); !r) { fs_send_error(res, r.error()); return; }
            res.set_content(json{{"path", path}}.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // ---- /api/plan — the phased-planning state (Decision 18 §18.5) -------

    svr.Get("/api/plan", [&](const httplib::Request& req, httplib::Response& res) {
        if (!check_auth_token(state.cfg.auth_token, req, res)) return;
        std::lock_guard<std::mutex> lk{state.ws_mutex};
        auto* ws = fs_ready_or_503(state.ws, res);
        if (!ws) return;

        auto plan = load_plan(*ws);
        if (!plan) { fs_send_error(res, plan.error()); return; }
        res.set_content(plan_to_json(*plan).dump(), "application/json");
    });

    // Check a phase off (or back on) from the UI. The agent does the same thing
    // by rewriting plan.md; this is the hand-editing path.
    svr.Post("/api/plan/phase", [&](const httplib::Request& req, httplib::Response& res) {
        if (!check_auth_token(state.cfg.auth_token, req, res)) return;
        std::lock_guard<std::mutex> lk{state.ws_mutex};
        auto* ws = fs_ready_or_503(state.ws, res);
        if (!ws) return;
        try {
            const auto body = json::parse(req.body);
            const auto number = body.at("phase").get<std::size_t>();
            const bool done   = body.value("done", true);

            auto plan = load_plan(*ws);
            if (!plan) { fs_send_error(res, plan.error()); return; }
            if (number < 1 || number > plan->phases.size()) {
                res.status = 400;
                res.set_content(json{{"error", "no such phase"}}.dump(), "application/json");
                return;
            }
            plan->phases[number - 1].done = done;
            if (auto w = save_plan(*ws, *plan); !w) { fs_send_error(res, w.error()); return; }
            if (done) {
                append_changelog(*ws, "phase " + std::to_string(number) + " done - " +
                                      plan->phases[number - 1].text);
            }
            res.set_content(plan_to_json(*plan).dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // ---- /api/permission — answer a parked prompt -----------------------
    svr.Post("/api/permission", [&](const httplib::Request& req, httplib::Response& res) {
        if (!check_auth_token(state.cfg.auth_token, req, res)) return;
        try {
            const auto body = json::parse(req.body);
            const auto id  = body.at("id").get<std::string>();
            auto decision  = body.value("decision", std::string{"deny"});
            if (decision != "allow" && decision != "always") decision = "deny";
            {
                std::lock_guard<std::mutex> lk{state.perm_mutex};
                state.perm_answers[id] = decision;
            }
            state.perm_cv.notify_all();
            res.set_content(json{{"id", id}, {"decision", decision}}.dump(),
                            "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // ---- /api/permissions — the current mode ----------------------------
    svr.Get("/api/permissions", [&](const httplib::Request& req, httplib::Response& res) {
        if (!check_auth_token(state.cfg.auth_token, req, res)) return;
        std::lock_guard<std::mutex> lk{state.perm_mutex};
        json allowed = json::array();
        for (auto r : state.perm_session_allowed) allowed.push_back(to_string(r));
        res.set_content(json{{"mode", to_string(state.perm_mode)},
                             {"session_allowed", allowed}}.dump(), "application/json");
    });

    svr.Post("/api/permissions", [&](const httplib::Request& req, httplib::Response& res) {
        if (!check_auth_token(state.cfg.auth_token, req, res)) return;
        try {
            const auto body = json::parse(req.body);
            const auto mode = parse_permission_mode(body.value("mode", std::string{"auto"}));
            {
                std::lock_guard<std::mutex> lk{state.perm_mutex};
                state.perm_mode = mode;
                // Changing the mode clears blanket approvals: they were granted
                // under the old policy and should not survive a tightening.
                state.perm_session_allowed.clear();
            }
            res.set_content(json{{"mode", to_string(mode)}}.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // ---- /api/slots ----
    svr.Get("/api/slots", [&](const httplib::Request& req, httplib::Response& res) {
        if (!check_auth_token(state.cfg.auth_token, req, res)) return;
        json j = {
            {"total",  state.ctx.kv.config().n_slots},
            {"in_use", state.ctx.kv.n_active_slots()},
            {"free",   state.ctx.kv.n_free_slots()},
            {"n_ctx",  state.ctx.kv.config().n_ctx}
        };
        res.set_content(j.dump(), "application/json");
    });

    // ---- /api/tokenize + /api/detokenize ----
    svr.Post("/api/tokenize", [&](const httplib::Request& req, httplib::Response& res) {
        if (!check_auth_token(state.cfg.auth_token, req, res)) return;
        try {
            auto body = json::parse(req.body);
            const std::string text = body.value("text", "");
            const bool special     = body.value("allow_special", false);
            auto ids = state.ctx.tokenizer.encode(text, special);
            res.set_content(json{{"tokens", ids}, {"count", ids.size()}}.dump(),
                            "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });
    svr.Post("/api/detokenize", [&](const httplib::Request& req, httplib::Response& res) {
        if (!check_auth_token(state.cfg.auth_token, req, res)) return;
        try {
            auto body = json::parse(req.body);
            std::vector<std::int32_t> ids;
            if (body.contains("tokens") && body["tokens"].is_array()) {
                for (const auto& x : body["tokens"]) ids.push_back(x.get<std::int32_t>());
            }
            const bool skip_special = body.value("skip_special", true);
            const std::string t = state.ctx.tokenizer.decode(ids, skip_special);
            res.set_content(json{{"text", t}}.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // ---- /v1/chat/completions (streaming + non-streaming) ----
    svr.Post("/v1/chat/completions",
        [&](const httplib::Request& req, httplib::Response& res) {
        if (!check_auth_token(state.cfg.auth_token, req, res)) return;

        json body;
        try { body = json::parse(req.body); }
        catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
            return;
        }

        const auto msgs   = parse_messages(body.value("messages", json::array()));
        const auto params = parse_sampler(body);
        const std::size_t max_tokens = body.contains("max_tokens")
                                       && body["max_tokens"].is_number_integer()
            ? std::min<std::size_t>(body["max_tokens"].get<std::size_t>(),
                                    state.cfg.max_gen_tokens)
            : state.cfg.max_gen_tokens;
        const bool want_stream = body.value("stream", false);

        if (!want_stream) {
            std::lock_guard<std::mutex> lk(state.model_mutex);
            auto r = run_chat_completion(state.ctx, msgs, params, max_tokens);
            if (!r) {
                res.status = 500;
                res.set_content(json{{"error", r.error().message}}.dump(),
                                "application/json");
                return;
            }
            res.set_content(openai_chat_response(state.ctx.model_name, *r).dump(),
                            "application/json");
            return;
        }

        // ---- SSE streaming path ----
        // Defeat reverse-proxy buffering per Decision 14 §14.3.
        res.set_header("Cache-Control",     "no-cache");
        res.set_header("Connection",        "keep-alive");
        res.set_header("X-Accel-Buffering", "no");

        // The provider closure captures by value into a shared_ptr so the
        // state survives past this handler's stack frame while httplib
        // streams asynchronously.
        auto req_state = std::make_shared<std::tuple<
            std::vector<tokenizer::ChatMessage>,
            sampler::SamplerParams,
            std::size_t
        >>(msgs, params, max_tokens);

        res.set_chunked_content_provider("text/event-stream",
            [this_state_ptr = &state, req_state]
            (std::size_t /*offset*/, httplib::DataSink& sink) -> bool {
                auto& s = *this_state_ptr;
                const std::int64_t created =
                    std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count();

                // 1. Role chunk (matches OpenAI client expectations).
                {
                    const auto f = sse_frame(sse_role_chunk(s.ctx.model_name, created));
                    if (!sink.write(f.data(), f.size())) return false;
                }

                // 2. Serialize the model behind the same mutex the JSON path
                // uses so concurrent streaming clients don't interleave into
                // the single KV slot.
                std::lock_guard<std::mutex> lk(s.model_mutex);
                auto r = run_chat_stream(
                    s.ctx,
                    std::get<0>(*req_state),
                    std::get<1>(*req_state),
                    std::get<2>(*req_state),
                    [&](std::string_view piece) {
                        const auto f = sse_frame(
                            sse_content_chunk(s.ctx.model_name, created, piece));
                        sink.write(f.data(), f.size());
                    });

                if (!r) {
                    // Report the error inline as a synthetic delta then stop.
                    json err = {
                        {"error", r.error().message},
                        {"component", r.error().component}
                    };
                    const auto f = "event: error\ndata: " + err.dump() + "\n\n";
                    sink.write(f.data(), f.size());
                } else {
                    const auto f = sse_frame(
                        sse_stop_chunk(s.ctx.model_name, created, *r));
                    sink.write(f.data(), f.size());
                }
                const std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
                sink.done();
                return true;
            });
    });

    // ---- /api/agent/chat — the coding session (Decision 18 §18.3) --------
    //
    // A turn is a loop, not a single generation: the model replies, any tool
    // calls in that reply are executed, the results are appended as `tool`
    // messages, and it generates again. The loop ends when a reply contains no
    // tool calls, or when the round budget runs out.
    //
    // SSE events: delta | tool_call | tool_result | file | done | error.
    svr.Post("/api/agent/chat", [&](const httplib::Request& req, httplib::Response& res) {
        if (!check_auth_token(state.cfg.auth_token, req, res)) return;

        json body;
        try { body = json::parse(req.body); }
        catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
            return;
        }

        auto msgs = parse_messages(body.value("messages", json::array()));
        if (msgs.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "messages is empty"}}.dump(),
                            "application/json");
            return;
        }

        // "plan"  — produce a plan and nothing else.
        // "work"  — execute, with the active phase injected.
        // "auto"  — work, planning first only if asked to.
        const std::string mode = body.value("mode", std::string{"auto"});

        // Read the plan up front: it shapes the system prompt, and re-reading
        // it every turn is what lets the user hand-edit plan.md between turns.
        Plan plan;
        {
            std::lock_guard<std::mutex> lk{state.ws_mutex};
            if (state.ws) {
                if (auto p = load_plan(*state.ws); p) plan = std::move(*p);
            }
        }
        const bool planning = (mode == "plan");

        // Append the tool protocol to the system message, creating one if the
        // caller did not send it. The tools have to be described inside the
        // system turn or the model will not use them.
        {
            std::string block = render_tools_system_prompt();
            {
                std::lock_guard<std::mutex> lk{state.ws_mutex};
                block += render_skills_prompt(state.skills);
            }
            if (planning) {
                block += render_planning_instructions(*state.ws);
            } else {
                block += render_plan_for_prompt(plan);
            }
            if (!msgs.empty() && msgs.front().role == "system") {
                msgs.front().content += block;
            } else {
                tokenizer::ChatMessage sys;
                sys.role = "system";
                sys.content =
                    "You are Ultima's coding assistant. You work inside the user's "
                    "project and change it by calling tools." + block;
                msgs.insert(msgs.begin(), std::move(sys));
            }
        }

        const auto params = parse_sampler(body);
        const std::size_t max_tokens = body.contains("max_tokens")
                                       && body["max_tokens"].is_number_integer()
            ? std::min<std::size_t>(body["max_tokens"].get<std::size_t>(),
                                    state.cfg.max_gen_tokens)
            : state.cfg.max_gen_tokens;
        // Enough rounds to read a couple of files and then write them back;
        // low enough that a model stuck in a loop stops costing time.
        const std::size_t max_rounds = std::min<std::size_t>(
            body.value("max_rounds", std::size_t{6}), 16);

        res.set_header("Cache-Control",     "no-cache");
        res.set_header("Connection",        "keep-alive");
        res.set_header("X-Accel-Buffering", "no");

        auto req_state = std::make_shared<std::tuple<
            std::vector<tokenizer::ChatMessage>, sampler::SamplerParams,
            std::size_t, std::size_t, bool>>(std::move(msgs), params, max_tokens,
                                             // Planning is a single round by
                                             // construction: write the plan, stop.
                                             planning ? std::size_t{2} : max_rounds,
                                             planning);

        res.set_chunked_content_provider("text/event-stream",
            [this_state_ptr = &state, req_state]
            (std::size_t /*offset*/, httplib::DataSink& sink) -> bool {
                auto& s = *this_state_ptr;
                auto& conv       = std::get<0>(*req_state);
                const auto& parm = std::get<1>(*req_state);
                const auto maxtk = std::get<2>(*req_state);
                const auto rounds_budget = std::get<3>(*req_state);
                const bool planning      = std::get<4>(*req_state);

                // On Windows a closed socket is not reliably visible to
                // is_writable() until a write actually fails, so disconnect is
                // detected from the writes themselves.
                bool sink_dead = false;
                auto emit = [&](const char* event, const json& payload) {
                    if (sink_dead) return false;
                    const std::string f =
                        std::string("event: ") + event + "\ndata: " + payload.dump() + "\n\n";
                    if (!sink.write(f.data(), f.size())) { sink_dead = true; return false; }
                    return true;
                };

                std::lock_guard<std::mutex> lk(s.model_mutex);
                Workspace* ws = s.ws ? &*s.ws : nullptr;
                if (!ws) {
                    emit("error", json{{"error", "no workspace is open"}});
                    sink.done();
                    return true;
                }

                std::size_t round = 0;
                std::size_t total_calls = 0;
                bool plan_written = false;

                // A client that hangs up should stop the work. Otherwise the
                // turn generates into a dead sink while holding model_mutex,
                // and every later request waits behind it.
                // Detecting a hung-up client is unreliable on its own: a small
                // write into a socket whose peer has vanished succeeds locally
                // until the TCP stack notices, which can take minutes. So the
                // write-failure check is backed by a wall-clock cap, which is
                // crude but always fires. Either way the point is the same —
                // the model mutex is held for a whole turn, so a turn nobody is
                // listening to blocks every later request.
                const auto turn_start = std::chrono::steady_clock::now();
                const auto turn_budget = std::chrono::seconds(s.cfg.max_turn_seconds);
                auto client_gone = [&] {
                    if (sink_dead || !sink.is_writable()) return true;
                    return s.cfg.max_turn_seconds > 0 &&
                           (std::chrono::steady_clock::now() - turn_start) > turn_budget;
                };
                for (; round < rounds_budget; ++round) {
                    if (client_gone()) return true;
                    // Only what is outside a <tool_call> block reaches the user.
                    std::string acc;
                    std::size_t shown = 0;
                    std::size_t since_probe = 0;

                    auto r = run_chat_stream(s.ctx, conv, parm, maxtk,
                        [&](std::string_view piece) {
                            acc.append(piece);
                            const std::string vis = visible_prose_prefix(acc);
                            if (vis.size() > shown) {
                                emit("delta", json{{"text", vis.substr(shown)}});
                                shown = vis.size();
                                since_probe = 0;
                            } else if (++since_probe >= 24) {
                                // An SSE comment: clients ignore it, but the
                                // write fails once the client has gone. Inside
                                // a tool call there is no prose to emit, so
                                // without this nothing would notice.
                                since_probe = 0;
                                static constexpr std::string_view ping = ": ping\n\n";
                                if (!sink_dead && !sink.write(ping.data(), ping.size())) {
                                    sink_dead = true;
                                }
                            }
                        },
                        client_gone);

                    if (!r) {
                        emit("error", json{{"error", r.error().message},
                                           {"component", r.error().component}});
                        sink.done();
                        return true;
                    }

                    if (client_gone()) return true;

                    const auto calls = parse_tool_calls(r->text);

                    // A round that yields neither prose nor a tool call is a
                    // dead end the user would otherwise see as nothing at all
                    // happening. Surface the raw text so the failure is
                    // diagnosable instead of silent.
                    if (calls.empty() && shown == 0) {
                        std::string head = r->text.substr(0, 400);
                        emit("warning", json{
                            {"reason", "the model produced no visible text and no tool call"},
                            {"raw_head", head},
                            {"raw_len", r->text.size()},
                            {"round", round}});
                    }

                    if (calls.empty()) {
                        // No tool calls: this reply is the answer, so the turn
                        // is over.
                        emit("done", json{{"rounds", round + 1},
                                          {"tool_calls", total_calls},
                                          {"stop", "complete"}});
                        sink.done();
                        return true;
                    }

                    // Record what the model said, verbatim, so the next round
                    // sees its own calls alongside their results.
                    tokenizer::ChatMessage assistant;
                    assistant.role = "assistant";
                    assistant.content = r->text;
                    conv.push_back(std::move(assistant));

                    for (const auto& c : calls) {
                        if (client_gone()) return true;
                        ++total_calls;
                        emit("tool_call", json{{"name", c.name},
                                               {"arguments", c.args},
                                               {"round", round}});

                        // Permission gate (PERMISSIONS.md). Reads are never
                        // gated; ordinary writes pass in auto; the always-ask
                        // set parks here until the user answers.
                        ToolResult tr;
                        bool refused = false;
                        {
                            Verdict verdict;
                            {
                                std::lock_guard<std::mutex> plk{s.perm_mutex};
                                verdict = evaluate(s.perm_mode, c, s.perm_session_allowed);
                            }
                            if (verdict.needs_prompt) {
                                std::string pid;
                                {
                                    std::lock_guard<std::mutex> plk{s.perm_mutex};
                                    pid = "perm-" + std::to_string(s.perm_next_id++);
                                }
                                emit("permission_request", json{
                                    {"id", pid},
                                    {"tool", c.name},
                                    {"risk", to_string(verdict.risk)},
                                    {"reason", verdict.reason},
                                    {"subject", verdict.subject},
                                    {"arguments", c.args}});

                                std::string answer = "deny";
                                {
                                    std::unique_lock<std::mutex> plk{s.perm_mutex};
                                    // Silence is never consent: an unanswered
                                    // prompt times out as a denial.
                                    const bool got = s.perm_cv.wait_for(
                                        plk, std::chrono::seconds(120),
                                        [&] { return s.perm_answers.count(pid) > 0; });
                                    if (got) {
                                        answer = s.perm_answers[pid];
                                        s.perm_answers.erase(pid);
                                    }
                                    if (answer == "always") {
                                        s.perm_session_allowed.push_back(verdict.risk);
                                    }
                                }
                                emit("permission_result", json{{"id", pid}, {"decision", answer}});

                                if (answer == "deny") {
                                    tr.name = c.name;
                                    tr.ok = false;
                                    tr.content = "Permission denied by the user. Do not retry "
                                                 "this call; try a different approach or stop "
                                                 "and ask what they would prefer.";
                                    refused = true;
                                }
                            }
                        }

                        // In planning mode the only legal act is writing the
                        // plan. Refusing anything else here is what stops a
                        // model from "helpfully" starting the work as well.
                        const std::string target = c.args.value("path", std::string{});
                        const bool writes_plan =
                            target.find("plan.md") != std::string::npos;
                        if (refused) {
                            // already filled in above
                        } else if (planning && !writes_plan) {
                            tr.name = c.name;
                            tr.ok = false;
                            tr.content =
                                "Not now. This turn is for writing the plan only. "
                                "Call write_project_file with path \"" +
                                std::string{k_plan_path} + "\" and the phase list, "
                                "then stop.";
                        } else {
                            tr = dispatch_tool(*ws, c, &s.skills);
                        }
                        emit("tool_result", json{{"name", tr.name},
                                                 {"ok", tr.ok},
                                                 {"content", tr.content},
                                                 {"round", round}});
                        // Decision 16 §16.4: a written file opens its own tab.
                        if (!tr.changed_path.empty() && tr.ok) {
                            emit("file", json{{"path", tr.changed_path},
                                              {"tab", tr.tab_label}});
                            // The plan drives the UI panel and the next turn's
                            // system prompt, so push it the moment it changes.
                            if (tr.changed_path.find("plan.md") != std::string::npos ||
                                tr.changed_path.find("todos.json") != std::string::npos) {
                                if (auto p2 = load_plan(*ws); p2) {
                                    // A plan written this turn cannot have
                                    // finished phases. Observed live: the model
                                    // checked off Phase 4 as it wrote the list,
                                    // which would have made the runtime skip it.
                                    if (planning) {
                                        bool fixed = false;
                                        for (auto& ph : p2->phases) {
                                            if (ph.done) { ph.done = false; fixed = true; }
                                        }
                                        if (fixed) save_plan(*ws, *p2);
                                    }
                                    emit("plan", plan_to_json(*p2));
                                    plan_written = true;
                                }
                            }
                        }

                        tokenizer::ChatMessage tool_msg;
                        tool_msg.role = "tool";
                        tool_msg.name = tr.name;
                        tool_msg.content = tr.ok
                            ? tr.content
                            : ("ERROR: " + tr.content);
                        conv.push_back(std::move(tool_msg));
                    }
                    if (planning && plan_written) {
                        emit("done", json{{"rounds", round + 1},
                                          {"tool_calls", total_calls},
                                          {"stop", "plan_written"}});
                        sink.done();
                        return true;
                    }
                }

                // Budget exhausted. Say so rather than letting the UI think the
                // model simply finished.
                emit("done", json{{"rounds", round},
                                  {"tool_calls", total_calls},
                                  {"stop", "max_rounds"}});
                sink.done();
                return true;
            });
    });

    // ---- Testpad ----
    svr.Get("/", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(std::string(kTestpadHtml), "text/html; charset=utf-8");
    });
}

HttpServer::~HttpServer() { stop(); }

Result<void> HttpServer::listen_blocking() {
    // Open the workspace before binding: a bad --workspace should fail loudly
    // at startup, not on the first file request.
    {
        auto ws = Workspace::open(impl_->cfg.workspace_root);
        if (!ws) return Failure{ws.error()};
        impl_->ws.emplace(std::move(*ws));
        impl_->perm_mode = parse_permission_mode(impl_->cfg.permission_mode);
        impl_->skills = SkillIndex::discover(
            default_skill_roots(impl_->ws->root()));
    }

    if (!impl_->svr.listen(impl_->cfg.bind_host.c_str(),
                           static_cast<int>(impl_->cfg.bind_port))) {
        return fail(ErrorCode::InvalidModel,
                    "httplib failed to bind — port in use?",
                    "server");
    }
    return {};
}

void HttpServer::stop() {
    impl_->stopping.store(true);
    impl_->svr.stop();
}

} // namespace ultima::server
