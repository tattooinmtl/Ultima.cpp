#pragma once

#include "ultima/core/error.hpp"
#include "ultima/model/imodel.hpp"
#include "ultima/model/loaded_model.hpp"

#include <string>

namespace ultima::model {

// Parse ModelDims + RopeConfig from a LoadedModel's metadata using the
// architecture-specific key prefix ("qwen2" or "qwen3").
core::Result<void>
parse_qwen_config(const LoadedModel& gguf,
                  const std::string& arch_prefix,   // "qwen2" or "qwen3"
                  ModelDims&  out_dims,
                  RopeConfig& out_rope);

} // namespace ultima::model
