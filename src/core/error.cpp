#include "ultima/core/error.hpp"

namespace ultima::core {

const char* to_string(ErrorCode code) noexcept {
    switch (code) {
    case ErrorCode::Unknown:                 return "Unknown";
    case ErrorCode::InvalidArgument:         return "InvalidArgument";
    case ErrorCode::NotFound:                return "NotFound";
    case ErrorCode::NotImplemented:          return "NotImplemented";
    case ErrorCode::FileNotFound:            return "FileNotFound";
    case ErrorCode::FileTooSmall:            return "FileTooSmall";
    case ErrorCode::IoFailure:               return "IoFailure";
    case ErrorCode::MmapFailure:             return "MmapFailure";
    case ErrorCode::InvalidModel:            return "InvalidModel";
    case ErrorCode::UnsupportedVersion:      return "UnsupportedVersion";
    case ErrorCode::UnsupportedArchitecture: return "UnsupportedArchitecture";
    case ErrorCode::UnsupportedDataType:     return "UnsupportedDataType";
    case ErrorCode::MissingRequiredMetadata: return "MissingRequiredMetadata";
    case ErrorCode::CorruptTensorDirectory:  return "CorruptTensorDirectory";
    case ErrorCode::MultiFileNotSupported:   return "MultiFileNotSupported";
    }
    return "?";
}

} // namespace ultima::core
