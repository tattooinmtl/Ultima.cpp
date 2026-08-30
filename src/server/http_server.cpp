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
#include "ultima/tokenizer/tokenizer.hpp"

#include <atomic>
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
                const std::function<void(std::string_view)>& on_delta) {
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
    ctx.kv.reuse_prefix(slot, prompt_ids);
    const std::size_t already = ctx.kv.pos(slot);

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

struct HttpServer::Impl {
    Impl(RuntimeContext c, ServerConfig g) : ctx{std::move(c)}, cfg{std::move(g)} {}

    RuntimeContext ctx;
    ServerConfig   cfg;
    httplib::Server svr;
    std::mutex     model_mutex;   // serializes chat requests (one slot pool in v0.1)
    std::atomic<bool> stopping{false};
};

HttpServer::HttpServer(RuntimeContext ctx, ServerConfig cfg)
    : impl_{std::make_unique<Impl>(std::move(ctx), std::move(cfg))} {
    auto& svr = impl_->svr;
    auto& state = *impl_;

    // ---- Auth middleware (local bypass, LAN requires bearer) ----
    auto check_auth = [&](const httplib::Request& req, httplib::Response& res) -> bool {
        if (state.cfg.auth_token.empty()) return true;      // auth disabled
        auto it = req.headers.find("Authorization");
        const std::string expected = "Bearer " + state.cfg.auth_token;
        if (it == req.headers.end() || it->second != expected) {
            res.status = 401;
            res.set_content(json{{"error", "unauthorized"}}.dump(), "application/json");
            return false;
        }
        return true;
    };

    // ---- CORS (localhost dev) ----
    svr.set_pre_routing_handler([&](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin",  "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        if (req.method == "OPTIONS") {
            res.status = 204;
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    // ---- /api/health ----
    svr.Get("/api/health", [&](const httplib::Request& req, httplib::Response& res) {
        if (!check_auth(req, res)) return;
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
        if (!check_auth(req, res)) return;
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

    // ---- /api/slots ----
    svr.Get("/api/slots", [&](const httplib::Request& req, httplib::Response& res) {
        if (!check_auth(req, res)) return;
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
        if (!check_auth(req, res)) return;
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
        if (!check_auth(req, res)) return;
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
        if (!check_auth(req, res)) return;

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
                return false;
            });
    });

    // ---- Testpad ----
    svr.Get("/", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(std::string(kTestpadHtml), "text/html; charset=utf-8");
    });
}

HttpServer::~HttpServer() { stop(); }

Result<void> HttpServer::listen_blocking() {
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
