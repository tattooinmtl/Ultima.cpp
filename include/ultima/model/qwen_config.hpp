#pragma once

#include "ultima/core/error.hpp"
#include "ultima/model/arch_registry.hpp"
#include "ultima/model/imodel.hpp"
#include "ultima/model/loaded_model.hpp"

#include <string>

namespace ultima::model {

// Parse ModelDims, RopeConfig and ArchParams from a LoadedModel's metadata.
//
// `arch` is the GGUF `general.architecture` string, used as the metadata key
// prefix; `spec` is its registry entry, which supplies the structural choices
// (FFN kind, QKV kind, RoPE convention) that metadata does not record.
//
// Reads tensors as well as metadata: Phi-3 stores its LongRoPE frequency
// factors as tensors rather than KV pairs.
core::Result<void>
parse_arch_config(const LoadedModel& gguf,
                  const ArchSpec&    spec,
                  const std::string& arch,
                  ModelDims&  out_dims,
                  RopeConfig& out_rope,
                  ArchParams& out_params);

// Dims + RoPE only, for callers that do not care about the structural
// parameters (config inspection, tests).
core::Result<void>
parse_qwen_config(const LoadedModel& gguf,
                  const std::string& arch_prefix,
                  ModelDims&  out_dims,
                  RopeConfig& out_rope);

} // namespace ultima::model
