#pragma once

#include "ultima/core/error.hpp"
#include "ultima/kv_cache/kv_cache.hpp"
#include "ultima/model/imodel.hpp"
#include "ultima/sampler/sampler.hpp"
#include "ultima/tokenizer/tokenizer.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace ultima::runtime { class ThreadPool; }

namespace ultima::server {

// All runtime dependencies the server needs. Owned by the caller; the
// HttpServer only holds references.
struct RuntimeContext {
    model::IModel&                model;
    tokenizer::BpeTokenizer&      tokenizer;
    kv_cache::KVCache&            kv;
    runtime::ThreadPool&          pool;
    std::string                   model_name = "ultima";   // shown in /v1/models
};

struct ServerConfig {
    std::string     bind_host      = "127.0.0.1";
    std::uint16_t   bind_port      = 11434;         // Ollama-mnemonic per Decision 14
    std::string     auth_token;                     // empty => auth disabled for local traffic
    std::size_t     max_gen_tokens = 512;           // safety cap for a single request
};

class HttpServer {
public:
    HttpServer(RuntimeContext ctx, ServerConfig cfg);
    ~HttpServer();

    HttpServer(const HttpServer&)            = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    // Start listening. Blocks until stop() is called from another thread or
    // the process is killed. Returns Ok on graceful shutdown.
    core::Result<void> listen_blocking();

    // Request the server to stop from another thread. Safe to call before
    // listen_blocking() returns.
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ultima::server
