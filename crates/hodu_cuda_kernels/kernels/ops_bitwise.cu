#include "utils.cuh"
#include <stdint.h>

// ============================================================================
// Binary Bitwise Operations
// ============================================================================
// Performs element-wise bitwise operations on tensors.
// Only integer types are supported (u8, u16, u32, u64, i8, i16, i32, i64).
//
// Binary Bitwise Metadata Layout:
// - metadata[0]: num_els (total number of output elements)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: lhs_shape
// - metadata[2+num_dims..2+2*num_dims]: rhs_shape
// - metadata[2+2*num_dims..2+3*num_dims]: lhs_strides
// - metadata[2+3*num_dims..2+4*num_dims]: rhs_strides
// - metadata[2+4*num_dims]: lhs_offset
// - metadata[2+4*num_dims+1]: rhs_offset

#define BINARY_BITWISE_OP(TYPENAME, FN_NAME, FUNC)                                                 \
    extern "C" __global__ void hodu_cuda_##FN_NAME(const TYPENAME *lhs, const TYPENAME *rhs,       \
                                                   TYPENAME *out, const size_t *metadata) {        \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *lhs_shape = metadata + 2;                                                    \
        const size_t *rhs_shape = metadata + 2 + num_dims;                                         \
        const size_t *lhs_strides = metadata + 2 + 2 * num_dims;                                   \
        const size_t *rhs_strides = metadata + 2 + 3 * num_dims;                                   \
        const size_t lhs_offset = metadata[2 + 4 * num_dims];                                      \
        const size_t rhs_offset = metadata[2 + 4 * num_dims + 1];                                  \
        bool lhs_cont = is_contiguous(num_dims, lhs_shape, lhs_strides);                           \
        bool rhs_cont = is_contiguous(num_dims, rhs_shape, rhs_strides);                           \
        if (lhs_cont && rhs_cont) {                                                                \
            for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_els;                  \
                 i += blockDim.x * gridDim.x) {                                                    \
                TYPENAME x = lhs[lhs_offset + i];                                                  \
                TYPENAME y = rhs[rhs_offset + i];                                                  \
                out[i] = FUNC;                                                                     \
            }                                                                                      \
        } else if (lhs_cont) {                                                                     \
            for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_els;                  \
                 i += blockDim.x * gridDim.x) {                                                    \
                uint32_t rhs_i = get_strided_index(i, num_dims, rhs_shape, rhs_strides);           \
                TYPENAME x = lhs[lhs_offset + i];                                                  \
                TYPENAME y = rhs[rhs_offset + rhs_i];                                              \
                out[i] = FUNC;                                                                     \
            }                                                                                      \
        } else if (rhs_cont) {                                                                     \
            for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_els;                  \
                 i += blockDim.x * gridDim.x) {                                                    \
                uint32_t lhs_i = get_strided_index(i, num_dims, lhs_shape, lhs_strides);           \
                TYPENAME x = lhs[lhs_offset + lhs_i];                                              \
                TYPENAME y = rhs[rhs_offset + i];                                                  \
                out[i] = FUNC;                                                                     \
            }                                                                                      \
        } else {                                                                                   \
            for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_els;                  \
                 i += blockDim.x * gridDim.x) {                                                    \
                uint32_t lhs_i = get_strided_index(i, num_dims, lhs_shape, lhs_strides);           \
                uint32_t rhs_i = get_strided_index(i, num_dims, rhs_shape, rhs_strides);           \
                TYPENAME x = lhs[lhs_offset + lhs_i];                                              \
                TYPENAME y = rhs[rhs_offset + rhs_i];                                              \
                out[i] = FUNC;                                                                     \
            }                                                                                      \
        }                                                                                          \
    }

// ============================================================================
// Unary Bitwise Operations
// ============================================================================
// Metadata Layout:
// - metadata[0]: num_els (total number of elements)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: shape
// - metadata[2+num_dims..2+2*num_dims]: strides
// - metadata[2+2*num_dims]: offset

#define UNARY_BITWISE_OP(TYPENAME, FN_NAME, FUNC)                                                  \
    extern "C" __global__ void hodu_cuda_##FN_NAME(const TYPENAME *input, TYPENAME *output,        \
                                                   const size_t *metadata) {                       \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *shape = metadata + 2;                                                        \
        const size_t *strides = metadata + 2 + num_dims;                                           \
        const size_t offset = metadata[2 + 2 * num_dims];                                          \
        bool is_cont = is_contiguous(num_dims, shape, strides);                                    \
        if (is_cont) {                                                                             \
            for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_els;                  \
                 i += blockDim.x * gridDim.x) {                                                    \
                TYPENAME x = input[offset + i];                                                    \
                output[i] = FUNC;                                                                  \
            }                                                                                      \
        } else {                                                                                   \
            for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_els;                  \
                 i += blockDim.x * gridDim.x) {                                                    \
                uint32_t src_i = get_strided_index(i, num_dims, shape, strides);                   \
                TYPENAME x = input[offset + src_i];                                                \
                output[i] = FUNC;                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

// ============================================================================
// Binary Bitwise Operations - Unsigned Integers
// ============================================================================

// u8
BINARY_BITWISE_OP(uint8_t, shl_u8, x << y)
BINARY_BITWISE_OP(uint8_t, shr_u8, x >> y)
BINARY_BITWISE_OP(uint8_t, bitwise_and_u8, x &y)
BINARY_BITWISE_OP(uint8_t, bitwise_or_u8, x | y)
BINARY_BITWISE_OP(uint8_t, bitwise_xor_u8, x ^ y)

// u16
BINARY_BITWISE_OP(uint16_t, shl_u16, x << y)
BINARY_BITWISE_OP(uint16_t, shr_u16, x >> y)
BINARY_BITWISE_OP(uint16_t, bitwise_and_u16, x &y)
BINARY_BITWISE_OP(uint16_t, bitwise_or_u16, x | y)
BINARY_BITWISE_OP(uint16_t, bitwise_xor_u16, x ^ y)

// u32
BINARY_BITWISE_OP(uint32_t, shl_u32, x << y)
BINARY_BITWISE_OP(uint32_t, shr_u32, x >> y)
BINARY_BITWISE_OP(uint32_t, bitwise_and_u32, x &y)
BINARY_BITWISE_OP(uint32_t, bitwise_or_u32, x | y)
BINARY_BITWISE_OP(uint32_t, bitwise_xor_u32, x ^ y)

// u64
BINARY_BITWISE_OP(uint64_t, shl_u64, x << y)
BINARY_BITWISE_OP(uint64_t, shr_u64, x >> y)
BINARY_BITWISE_OP(uint64_t, bitwise_and_u64, x &y)
BINARY_BITWISE_OP(uint64_t, bitwise_or_u64, x | y)
BINARY_BITWISE_OP(uint64_t, bitwise_xor_u64, x ^ y)

// ============================================================================
// Binary Bitwise Operations - Signed Integers
// ============================================================================

// i8
BINARY_BITWISE_OP(int8_t, shl_i8, x << y)
BINARY_BITWISE_OP(int8_t, shr_i8, x >> y)
BINARY_BITWISE_OP(int8_t, bitwise_and_i8, x &y)
BINARY_BITWISE_OP(int8_t, bitwise_or_i8, x | y)
BINARY_BITWISE_OP(int8_t, bitwise_xor_i8, x ^ y)

// i16
BINARY_BITWISE_OP(int16_t, shl_i16, x << y)
BINARY_BITWISE_OP(int16_t, shr_i16, x >> y)
BINARY_BITWISE_OP(int16_t, bitwise_and_i16, x &y)
BINARY_BITWISE_OP(int16_t, bitwise_or_i16, x | y)
BINARY_BITWISE_OP(int16_t, bitwise_xor_i16, x ^ y)

// i32
BINARY_BITWISE_OP(int32_t, shl_i32, x << y)
BINARY_BITWISE_OP(int32_t, shr_i32, x >> y)
BINARY_BITWISE_OP(int32_t, bitwise_and_i32, x &y)
BINARY_BITWISE_OP(int32_t, bitwise_or_i32, x | y)
BINARY_BITWISE_OP(int32_t, bitwise_xor_i32, x ^ y)

// i64
BINARY_BITWISE_OP(int64_t, shl_i64, x << y)
BINARY_BITWISE_OP(int64_t, shr_i64, x >> y)
BINARY_BITWISE_OP(int64_t, bitwise_and_i64, x &y)
BINARY_BITWISE_OP(int64_t, bitwise_or_i64, x | y)
BINARY_BITWISE_OP(int64_t, bitwise_xor_i64, x ^ y)

// ============================================================================
// Unary Bitwise Operations - Unsigned Integers
// ============================================================================

UNARY_BITWISE_OP(uint8_t, bitwise_not_u8, ~x)
UNARY_BITWISE_OP(uint16_t, bitwise_not_u16, ~x)
UNARY_BITWISE_OP(uint32_t, bitwise_not_u32, ~x)
UNARY_BITWISE_OP(uint64_t, bitwise_not_u64, ~x)

// ============================================================================
// Unary Bitwise Operations - Signed Integers
// ============================================================================

UNARY_BITWISE_OP(int8_t, bitwise_not_i8, ~x)
UNARY_BITWISE_OP(int16_t, bitwise_not_i16, ~x)
UNARY_BITWISE_OP(int32_t, bitwise_not_i32, ~x)
UNARY_BITWISE_OP(int64_t, bitwise_not_i64, ~x)

// ============================================================================
// Scalar Shift Operations
// ============================================================================
// Shift all elements by a constant scalar amount.
// Metadata layout is same as unary ops.

#define SCALAR_SHIFT_OP(TYPENAME, FN_NAME, FUNC)                                                   \
    extern "C" __global__ void hodu_cuda_##FN_NAME(const TYPENAME *input, TYPENAME *output,        \
                                                   const size_t *metadata, uint32_t shift) {       \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *shape = metadata + 2;                                                        \
        const size_t *strides = metadata + 2 + num_dims;                                           \
        const size_t offset = metadata[2 + 2 * num_dims];                                          \
        bool is_cont = is_contiguous(num_dims, shape, strides);                                    \
        if (is_cont) {                                                                             \
            for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_els;                  \
                 i += blockDim.x * gridDim.x) {                                                    \
                TYPENAME x = input[offset + i];                                                    \
                output[i] = FUNC;                                                                  \
            }                                                                                      \
        } else {                                                                                   \
            for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_els;                  \
                 i += blockDim.x * gridDim.x) {                                                    \
                uint32_t src_i = get_strided_index(i, num_dims, shape, strides);                   \
                TYPENAME x = input[offset + src_i];                                                \
                output[i] = FUNC;                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

// ============================================================================
// Scalar Shift Operations - Unsigned Integers
// ============================================================================

SCALAR_SHIFT_OP(uint8_t, shl_scalar_u8, x << shift)
SCALAR_SHIFT_OP(uint8_t, shr_scalar_u8, x >> shift)
SCALAR_SHIFT_OP(uint16_t, shl_scalar_u16, x << shift)
SCALAR_SHIFT_OP(uint16_t, shr_scalar_u16, x >> shift)
SCALAR_SHIFT_OP(uint32_t, shl_scalar_u32, x << shift)
SCALAR_SHIFT_OP(uint32_t, shr_scalar_u32, x >> shift)
SCALAR_SHIFT_OP(uint64_t, shl_scalar_u64, x << shift)
SCALAR_SHIFT_OP(uint64_t, shr_scalar_u64, x >> shift)

// ============================================================================
// Scalar Shift Operations - Signed Integers
// ============================================================================

SCALAR_SHIFT_OP(int8_t, shl_scalar_i8, x << shift)
SCALAR_SHIFT_OP(int8_t, shr_scalar_i8, x >> shift)
SCALAR_SHIFT_OP(int16_t, shl_scalar_i16, x << shift)
SCALAR_SHIFT_OP(int16_t, shr_scalar_i16, x >> shift)
SCALAR_SHIFT_OP(int32_t, shl_scalar_i32, x << shift)
SCALAR_SHIFT_OP(int32_t, shr_scalar_i32, x >> shift)
SCALAR_SHIFT_OP(int64_t, shl_scalar_i64, x << shift)
SCALAR_SHIFT_OP(int64_t, shr_scalar_i64, x >> shift)
