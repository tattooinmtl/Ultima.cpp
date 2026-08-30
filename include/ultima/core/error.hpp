#pragma once

#include <nonstd/expected.hpp>

#include <string>
#include <string_view>

namespace ultima::core {

enum class ErrorCode {
    // Generic
    Unknown,
    InvalidArgument,
    NotFound,
    NotImplemented,

    // I/O
    FileNotFound,
    FileTooSmall,
    IoFailure,
    MmapFailure,

    // Model loading
    InvalidModel,
    UnsupportedVersion,
    UnsupportedArchitecture,
    UnsupportedDataType,
    MissingRequiredMetadata,
    CorruptTensorDirectory,
    MultiFileNotSupported,
};

struct Error {
    ErrorCode   code;
    std::string message;
    std::string component;   // e.g. "gguf_loader", "mmap"
};

const char* to_string(ErrorCode code) noexcept;

template <typename T>
using Result = nonstd::expected<T, Error>;

using Failure = nonstd::unexpected<Error>;

inline Failure fail(ErrorCode code, std::string message, std::string component = {}) {
    return Failure{Error{code, std::move(message), std::move(component)}};
}

} // namespace ultima::core
