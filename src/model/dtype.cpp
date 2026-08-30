#include "ultima/model/dtype.hpp"

namespace ultima::model {

const char* to_string(DataType t) noexcept {
    switch (t) {
    case DataType::F32:  return "F32";
    case DataType::F16:  return "F16";
    case DataType::Q8_0: return "Q8_0";
    case DataType::Q4_K: return "Q4_K";
    case DataType::Q6_K: return "Q6_K";
    case DataType::Q4_0: return "Q4_0";
    case DataType::Q4_1: return "Q4_1";
    case DataType::Q5_0: return "Q5_0";
    case DataType::Q5_1: return "Q5_1";
    case DataType::Q2_K: return "Q2_K";
    case DataType::Q3_K: return "Q3_K";
    case DataType::Q5_K: return "Q5_K";
    case DataType::Q8_K: return "Q8_K";
    case DataType::BF16: return "BF16";
    case DataType::Unknown: return "Unknown";
    }
    return "?";
}

bool is_active(DataType t) noexcept {
    switch (t) {
    case DataType::F32:
    case DataType::F16:
    case DataType::Q8_0:
    case DataType::Q4_K:
    case DataType::Q6_K:
        return true;
    default:
        return false;
    }
}

std::uint32_t block_size(DataType t) noexcept {
    switch (t) {
    case DataType::F32:
    case DataType::F16:
    case DataType::BF16:
        return 1;
    case DataType::Q4_0:
    case DataType::Q4_1:
    case DataType::Q5_0:
    case DataType::Q5_1:
    case DataType::Q8_0:
        return 32;
    case DataType::Q2_K:
    case DataType::Q3_K:
    case DataType::Q4_K:
    case DataType::Q5_K:
    case DataType::Q6_K:
    case DataType::Q8_K:
        return 256;
    default:
        return 0;
    }
}

std::uint32_t bytes_per_block(DataType t) noexcept {
    switch (t) {
    case DataType::F32:  return 4;
    case DataType::F16:  return 2;
    case DataType::BF16: return 2;

    case DataType::Q4_0: return 18;
    case DataType::Q4_1: return 20;
    case DataType::Q5_0: return 22;
    case DataType::Q5_1: return 24;
    case DataType::Q8_0: return 34;

    case DataType::Q2_K: return 84;
    case DataType::Q3_K: return 110;
    case DataType::Q4_K: return 144;
    case DataType::Q5_K: return 176;
    case DataType::Q6_K: return 210;
    case DataType::Q8_K: return 292;

    default: return 0;
    }
}

} // namespace ultima::model
