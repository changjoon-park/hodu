/**
 * @file ops_bitwise.h
 * @brief Bitwise tensor operations header
 *
 * Declares all element-wise bitwise operations for tensors including:
 * - Binary bitwise operations (shl, shr, bitwise_and, bitwise_or, bitwise_xor)
 * - Unary bitwise operations (bitwise_not)
 *
 * All operations support integer types only (u8, u16, u32, u64, i8, i16, i32, i64).
 * Float and bool types are not supported.
 */

#ifndef HODU_CPU_KERNELS_OPS_BITWISE_H
#define HODU_CPU_KERNELS_OPS_BITWISE_H

#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// BINARY BITWISE OPERATION FUNCTION SIGNATURES
// ============================================================================
//
// Binary bitwise operations follow the same signature as other binary ops:
//   void hodu_cpu_op_type(const void *lhs, const void *rhs, void *output, const size_t *metadata)
//
// Metadata layout (same as ops_binary):
// - metadata[0]: num_els (total number of elements)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: lhs_shape
// - metadata[2+num_dims..2+2*num_dims]: rhs_shape
// - metadata[2+2*num_dims..2+3*num_dims]: lhs_strides
// - metadata[2+3*num_dims..2+4*num_dims]: rhs_strides
// - metadata[2+4*num_dims]: lhs_offset
// - metadata[2+4*num_dims+1]: rhs_offset

/// Macro to declare binary bitwise operations for a given type
/// Declares: shl, shr, bitwise_and, bitwise_or, bitwise_xor
#define DECLARE_BINARY_BITWISE_OP(TYPE_SUFFIX)                                                     \
    void hodu_cpu_shl_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output,                \
                                    const size_t *metadata);                                       \
    void hodu_cpu_shr_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output,                \
                                    const size_t *metadata);                                       \
    void hodu_cpu_bitwise_and_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output,        \
                                            const size_t *metadata);                               \
    void hodu_cpu_bitwise_or_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output,         \
                                           const size_t *metadata);                                \
    void hodu_cpu_bitwise_xor_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output,        \
                                            const size_t *metadata);

// Unsigned integer types
DECLARE_BINARY_BITWISE_OP(u8)
DECLARE_BINARY_BITWISE_OP(u16)
DECLARE_BINARY_BITWISE_OP(u32)
DECLARE_BINARY_BITWISE_OP(u64)

// Signed integer types
DECLARE_BINARY_BITWISE_OP(i8)
DECLARE_BINARY_BITWISE_OP(i16)
DECLARE_BINARY_BITWISE_OP(i32)
DECLARE_BINARY_BITWISE_OP(i64)

// ============================================================================
// UNARY BITWISE OPERATION FUNCTION SIGNATURES
// ============================================================================
//
// Unary bitwise operations follow the same signature as other unary ops:
//   void hodu_cpu_op_type(const void *input, void *output, const size_t *metadata)
//
// Metadata layout (same as ops_unary):
// - metadata[0]: num_els (total number of elements)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: shape
// - metadata[2+num_dims..2+2*num_dims]: strides
// - metadata[2+2*num_dims]: offset

/// Macro to declare unary bitwise operations for a given type
/// Declares: bitwise_not
#define DECLARE_UNARY_BITWISE_OP(TYPE_SUFFIX)                                                      \
    void hodu_cpu_bitwise_not_##TYPE_SUFFIX(const void *input, void *output,                       \
                                            const size_t *metadata);

// Unsigned integer types
DECLARE_UNARY_BITWISE_OP(u8)
DECLARE_UNARY_BITWISE_OP(u16)
DECLARE_UNARY_BITWISE_OP(u32)
DECLARE_UNARY_BITWISE_OP(u64)

// Signed integer types
DECLARE_UNARY_BITWISE_OP(i8)
DECLARE_UNARY_BITWISE_OP(i16)
DECLARE_UNARY_BITWISE_OP(i32)
DECLARE_UNARY_BITWISE_OP(i64)

// ============================================================================
// SCALAR SHIFT OPERATION FUNCTION SIGNATURES
// ============================================================================
//
// Scalar shift operations apply a uniform shift amount to all tensor elements:
//   void hodu_cpu_op_type(const void *input, void *output, const size_t *metadata, uint32_t shift)
//
// Metadata layout (same as ops_unary):
// - metadata[0]: num_els (total number of elements)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: shape
// - metadata[2+num_dims..2+2*num_dims]: strides
// - metadata[2+2*num_dims]: offset

/// Macro to declare scalar shift operations for a given type
/// Declares: shl_scalar, shr_scalar
#define DECLARE_SCALAR_SHIFT_OP(TYPE_SUFFIX)                                                       \
    void hodu_cpu_shl_scalar_##TYPE_SUFFIX(const void *input, void *output,                        \
                                           const size_t *metadata, u32_t shift);                   \
    void hodu_cpu_shr_scalar_##TYPE_SUFFIX(const void *input, void *output,                        \
                                           const size_t *metadata, u32_t shift);

// Unsigned integer types
DECLARE_SCALAR_SHIFT_OP(u8)
DECLARE_SCALAR_SHIFT_OP(u16)
DECLARE_SCALAR_SHIFT_OP(u32)
DECLARE_SCALAR_SHIFT_OP(u64)

// Signed integer types
DECLARE_SCALAR_SHIFT_OP(i8)
DECLARE_SCALAR_SHIFT_OP(i16)
DECLARE_SCALAR_SHIFT_OP(i32)
DECLARE_SCALAR_SHIFT_OP(i64)

#ifdef __cplusplus
}
#endif

#endif // HODU_CPU_KERNELS_OPS_BITWISE_H
