#include "./headers/utils.metal"
#include <metal_stdlib>

using namespace metal;

// Bitwise Operations
// ==================
// Performs element-wise bitwise operations on tensors.
// Only integer types are supported (u8, u16, u32, u64, i8, i16, i32, i64).
//
// Binary Bitwise Metadata Layout (same as ops_binary):
// - metadata[0]: num_els (total number of output elements)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: lhs_shape
// - metadata[2+num_dims..2+2*num_dims]: rhs_shape
// - metadata[2+2*num_dims..2+3*num_dims]: lhs_strides
// - metadata[2+3*num_dims..2+4*num_dims]: rhs_strides
// - metadata[2+4*num_dims]: lhs_offset
// - metadata[2+4*num_dims+1]: rhs_offset

#define BINARY_BITWISE_OP(TYPENAME, FN_NAME, FUNC)                                                 \
    kernel void hodu_metal_##FN_NAME(                                                              \
        const device TYPENAME *lhs [[buffer(0)]], const device TYPENAME *rhs [[buffer(1)]],        \
        device TYPENAME *out [[buffer(2)]], constant size_t *metadata [[buffer(3)]],               \
        uint id [[thread_position_in_grid]]) {                                                     \
        const size_t num_els = metadata[0];                                                        \
        if (id >= num_els)                                                                         \
            return;                                                                                \
                                                                                                   \
        const size_t num_dims = metadata[1];                                                       \
        const constant size_t *lhs_shape = metadata + 2;                                           \
        const constant size_t *rhs_shape = metadata + 2 + num_dims;                                \
        const constant size_t *lhs_strides = metadata + 2 + 2 * num_dims;                          \
        const constant size_t *rhs_strides = metadata + 2 + 3 * num_dims;                          \
        const size_t lhs_offset = metadata[2 + 4 * num_dims];                                      \
        const size_t rhs_offset = metadata[2 + 4 * num_dims + 1];                                  \
                                                                                                   \
        bool lhs_cont = is_contiguous(num_dims, lhs_shape, lhs_strides);                           \
        bool rhs_cont = is_contiguous(num_dims, rhs_shape, rhs_strides);                           \
                                                                                                   \
        if (lhs_cont && rhs_cont) {                                                                \
            TYPENAME x = lhs[lhs_offset + id];                                                     \
            TYPENAME y = rhs[rhs_offset + id];                                                     \
            out[id] = FUNC;                                                                        \
        } else if (lhs_cont) {                                                                     \
            unsigned int tmp_i = id;                                                               \
            unsigned int rhs_i = 0;                                                                \
            for (int d = num_dims - 1; d >= 0; d--) {                                              \
                unsigned int i_dim = tmp_i % rhs_shape[d];                                         \
                rhs_i += i_dim * rhs_strides[d];                                                   \
                tmp_i /= rhs_shape[d];                                                             \
            }                                                                                      \
            TYPENAME x = lhs[lhs_offset + id];                                                     \
            TYPENAME y = rhs[rhs_offset + rhs_i];                                                  \
            out[id] = FUNC;                                                                        \
        } else if (rhs_cont) {                                                                     \
            unsigned int tmp_i = id;                                                               \
            unsigned int lhs_i = 0;                                                                \
            for (int d = num_dims - 1; d >= 0; d--) {                                              \
                unsigned int i_dim = tmp_i % lhs_shape[d];                                         \
                lhs_i += i_dim * lhs_strides[d];                                                   \
                tmp_i /= lhs_shape[d];                                                             \
            }                                                                                      \
            TYPENAME x = lhs[lhs_offset + lhs_i];                                                  \
            TYPENAME y = rhs[rhs_offset + id];                                                     \
            out[id] = FUNC;                                                                        \
        } else {                                                                                   \
            unsigned int tmp_i = id;                                                               \
            unsigned int lhs_i = 0;                                                                \
            unsigned int rhs_i = 0;                                                                \
            for (int d = num_dims - 1; d >= 0; d--) {                                              \
                unsigned int i_dim = tmp_i % lhs_shape[d];                                         \
                lhs_i += i_dim * lhs_strides[d];                                                   \
                rhs_i += i_dim * rhs_strides[d];                                                   \
                tmp_i /= lhs_shape[d];                                                             \
            }                                                                                      \
            TYPENAME x = lhs[lhs_offset + lhs_i];                                                  \
            TYPENAME y = rhs[rhs_offset + rhs_i];                                                  \
            out[id] = FUNC;                                                                        \
        }                                                                                          \
    }

// Unary Bitwise Metadata Layout (same as ops_unary):
// - metadata[0]: num_els (total number of elements)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: shape
// - metadata[2+num_dims..2+2*num_dims]: strides
// - metadata[2+2*num_dims]: offset

#define UNARY_BITWISE_OP(TYPENAME, FN_NAME, FUNC)                                                  \
    kernel void hodu_metal_##FN_NAME(                                                              \
        const device TYPENAME *input [[buffer(0)]], device TYPENAME *output [[buffer(1)]],         \
        constant size_t *metadata [[buffer(2)]], uint id [[thread_position_in_grid]]) {            \
        const size_t num_els = metadata[0];                                                        \
        if (id >= num_els)                                                                         \
            return;                                                                                \
                                                                                                   \
        const size_t num_dims = metadata[1];                                                       \
        const constant size_t *shape = metadata + 2;                                               \
        const constant size_t *strides = metadata + 2 + num_dims;                                  \
        const size_t offset = metadata[2 + 2 * num_dims];                                          \
                                                                                                   \
        bool is_cont = is_contiguous(num_dims, shape, strides);                                    \
                                                                                                   \
        TYPENAME x;                                                                                \
        if (is_cont) {                                                                             \
            x = input[offset + id];                                                                \
        } else {                                                                                   \
            unsigned int tmp_i = id;                                                               \
            unsigned int src_i = 0;                                                                \
            for (int d = num_dims - 1; d >= 0; d--) {                                              \
                unsigned int i_dim = tmp_i % shape[d];                                             \
                src_i += i_dim * strides[d];                                                       \
                tmp_i /= shape[d];                                                                 \
            }                                                                                      \
            x = input[offset + src_i];                                                             \
        }                                                                                          \
        output[id] = FUNC;                                                                         \
    }

// ============================================================================
// Binary Bitwise Operations - Unsigned Integers
// ============================================================================

// u8 (uchar)
BINARY_BITWISE_OP(uchar, shl_u8, x << y)
BINARY_BITWISE_OP(uchar, shr_u8, x >> y)
BINARY_BITWISE_OP(uchar, bitwise_and_u8, x &y)
BINARY_BITWISE_OP(uchar, bitwise_or_u8, x | y)
BINARY_BITWISE_OP(uchar, bitwise_xor_u8, x ^ y)

// u16 (ushort)
BINARY_BITWISE_OP(ushort, shl_u16, x << y)
BINARY_BITWISE_OP(ushort, shr_u16, x >> y)
BINARY_BITWISE_OP(ushort, bitwise_and_u16, x &y)
BINARY_BITWISE_OP(ushort, bitwise_or_u16, x | y)
BINARY_BITWISE_OP(ushort, bitwise_xor_u16, x ^ y)

// u32 (uint)
BINARY_BITWISE_OP(uint, shl_u32, x << y)
BINARY_BITWISE_OP(uint, shr_u32, x >> y)
BINARY_BITWISE_OP(uint, bitwise_and_u32, x &y)
BINARY_BITWISE_OP(uint, bitwise_or_u32, x | y)
BINARY_BITWISE_OP(uint, bitwise_xor_u32, x ^ y)

// u64 (ulong)
BINARY_BITWISE_OP(ulong, shl_u64, x << y)
BINARY_BITWISE_OP(ulong, shr_u64, x >> y)
BINARY_BITWISE_OP(ulong, bitwise_and_u64, x &y)
BINARY_BITWISE_OP(ulong, bitwise_or_u64, x | y)
BINARY_BITWISE_OP(ulong, bitwise_xor_u64, x ^ y)

// ============================================================================
// Binary Bitwise Operations - Signed Integers
// ============================================================================

// i8 (char)
BINARY_BITWISE_OP(char, shl_i8, x << y)
BINARY_BITWISE_OP(char, shr_i8, x >> y)
BINARY_BITWISE_OP(char, bitwise_and_i8, x &y)
BINARY_BITWISE_OP(char, bitwise_or_i8, x | y)
BINARY_BITWISE_OP(char, bitwise_xor_i8, x ^ y)

// i16 (short)
BINARY_BITWISE_OP(short, shl_i16, x << y)
BINARY_BITWISE_OP(short, shr_i16, x >> y)
BINARY_BITWISE_OP(short, bitwise_and_i16, x &y)
BINARY_BITWISE_OP(short, bitwise_or_i16, x | y)
BINARY_BITWISE_OP(short, bitwise_xor_i16, x ^ y)

// i32 (int)
BINARY_BITWISE_OP(int, shl_i32, x << y)
BINARY_BITWISE_OP(int, shr_i32, x >> y)
BINARY_BITWISE_OP(int, bitwise_and_i32, x &y)
BINARY_BITWISE_OP(int, bitwise_or_i32, x | y)
BINARY_BITWISE_OP(int, bitwise_xor_i32, x ^ y)

// i64 (long)
BINARY_BITWISE_OP(long, shl_i64, x << y)
BINARY_BITWISE_OP(long, shr_i64, x >> y)
BINARY_BITWISE_OP(long, bitwise_and_i64, x &y)
BINARY_BITWISE_OP(long, bitwise_or_i64, x | y)
BINARY_BITWISE_OP(long, bitwise_xor_i64, x ^ y)

// ============================================================================
// Unary Bitwise Operations - Unsigned Integers
// ============================================================================

UNARY_BITWISE_OP(uchar, bitwise_not_u8, ~x)
UNARY_BITWISE_OP(ushort, bitwise_not_u16, ~x)
UNARY_BITWISE_OP(uint, bitwise_not_u32, ~x)
UNARY_BITWISE_OP(ulong, bitwise_not_u64, ~x)

// ============================================================================
// Unary Bitwise Operations - Signed Integers
// ============================================================================

UNARY_BITWISE_OP(char, bitwise_not_i8, ~x)
UNARY_BITWISE_OP(short, bitwise_not_i16, ~x)
UNARY_BITWISE_OP(int, bitwise_not_i32, ~x)
UNARY_BITWISE_OP(long, bitwise_not_i64, ~x)

// ============================================================================
// Scalar Shift Operations
// ============================================================================
// Shift all elements by a constant scalar amount.
// Metadata layout is same as unary ops.
// Shift amount is passed via buffer(3).

#define SCALAR_SHIFT_OP(TYPENAME, FN_NAME, FUNC)                                                   \
    kernel void hodu_metal_##FN_NAME(                                                              \
        const device TYPENAME *input [[buffer(0)]], device TYPENAME *output [[buffer(1)]],         \
        constant size_t *metadata [[buffer(2)]], constant uint *shift_ptr [[buffer(3)]],           \
        uint id [[thread_position_in_grid]]) {                                                     \
        const size_t num_els = metadata[0];                                                        \
        if (id >= num_els)                                                                         \
            return;                                                                                \
                                                                                                   \
        const size_t num_dims = metadata[1];                                                       \
        const constant size_t *shape = metadata + 2;                                               \
        const constant size_t *strides = metadata + 2 + num_dims;                                  \
        const size_t offset = metadata[2 + 2 * num_dims];                                          \
        const uint shift = *shift_ptr;                                                             \
                                                                                                   \
        bool is_cont = is_contiguous(num_dims, shape, strides);                                    \
                                                                                                   \
        TYPENAME x;                                                                                \
        if (is_cont) {                                                                             \
            x = input[offset + id];                                                                \
        } else {                                                                                   \
            unsigned int tmp_i = id;                                                               \
            unsigned int src_i = 0;                                                                \
            for (int d = num_dims - 1; d >= 0; d--) {                                              \
                unsigned int i_dim = tmp_i % shape[d];                                             \
                src_i += i_dim * strides[d];                                                       \
                tmp_i /= shape[d];                                                                 \
            }                                                                                      \
            x = input[offset + src_i];                                                             \
        }                                                                                          \
        output[id] = FUNC;                                                                         \
    }

// ============================================================================
// Scalar Shift Operations - Unsigned Integers
// ============================================================================

SCALAR_SHIFT_OP(uchar, shl_scalar_u8, x << shift)
SCALAR_SHIFT_OP(uchar, shr_scalar_u8, x >> shift)
SCALAR_SHIFT_OP(ushort, shl_scalar_u16, x << shift)
SCALAR_SHIFT_OP(ushort, shr_scalar_u16, x >> shift)
SCALAR_SHIFT_OP(uint, shl_scalar_u32, x << shift)
SCALAR_SHIFT_OP(uint, shr_scalar_u32, x >> shift)
SCALAR_SHIFT_OP(ulong, shl_scalar_u64, x << shift)
SCALAR_SHIFT_OP(ulong, shr_scalar_u64, x >> shift)

// ============================================================================
// Scalar Shift Operations - Signed Integers
// ============================================================================

SCALAR_SHIFT_OP(char, shl_scalar_i8, x << shift)
SCALAR_SHIFT_OP(char, shr_scalar_i8, x >> shift)
SCALAR_SHIFT_OP(short, shl_scalar_i16, x << shift)
SCALAR_SHIFT_OP(short, shr_scalar_i16, x >> shift)
SCALAR_SHIFT_OP(int, shl_scalar_i32, x << shift)
SCALAR_SHIFT_OP(int, shr_scalar_i32, x >> shift)
SCALAR_SHIFT_OP(long, shl_scalar_i64, x << shift)
SCALAR_SHIFT_OP(long, shr_scalar_i64, x >> shift)
