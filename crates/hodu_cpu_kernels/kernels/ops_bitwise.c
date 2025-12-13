#include "ops_bitwise.h"
#include "thread_utils.h"
#include "types.h"

// ============================================================================
// BINARY BITWISE OPERATION IMPLEMENTATION MACROS
// ============================================================================

#define IMPL_BINARY_BITWISE_OP(TYPE, TYPE_SUFFIX, OP_NAME, FUNC)                                   \
    typedef struct {                                                                               \
        const TYPE *lhs;                                                                           \
        const TYPE *rhs;                                                                           \
        TYPE *output;                                                                              \
        size_t start;                                                                              \
        size_t end;                                                                                \
        size_t lhs_offset;                                                                         \
        size_t rhs_offset;                                                                         \
    } bitwise_##OP_NAME##_##TYPE_SUFFIX##_args_t;                                                  \
                                                                                                   \
    static void *bitwise_##OP_NAME##_##TYPE_SUFFIX##_worker(void *arg) {                           \
        bitwise_##OP_NAME##_##TYPE_SUFFIX##_args_t *args =                                         \
            (bitwise_##OP_NAME##_##TYPE_SUFFIX##_args_t *)arg;                                     \
        for (size_t i = args->start; i < args->end; i++) {                                         \
            TYPE x = args->lhs[args->lhs_offset + i];                                              \
            TYPE y = args->rhs[args->rhs_offset + i];                                              \
            args->output[i] = FUNC;                                                                \
        }                                                                                          \
        return NULL;                                                                               \
    }                                                                                              \
                                                                                                   \
    void hodu_cpu_##OP_NAME##_##TYPE_SUFFIX(const void *lhs, const void *rhs, void *output,        \
                                            const size_t *metadata) {                              \
        const TYPE *l = (const TYPE *)lhs;                                                         \
        const TYPE *r = (const TYPE *)rhs;                                                         \
        TYPE *out = (TYPE *)output;                                                                \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *lhs_shape = metadata + 2;                                                    \
        const size_t *rhs_shape = metadata + 2 + num_dims;                                         \
        const size_t *lhs_strides = metadata + 2 + 2 * num_dims;                                   \
        const size_t *rhs_strides = metadata + 2 + 3 * num_dims;                                   \
        const size_t lhs_offset = metadata[2 + 4 * num_dims];                                      \
        const size_t rhs_offset = metadata[2 + 4 * num_dims + 1];                                  \
                                                                                                   \
        bool lhs_cont = is_contiguous(num_dims, lhs_shape, lhs_strides);                           \
        bool rhs_cont = is_contiguous(num_dims, rhs_shape, rhs_strides);                           \
                                                                                                   \
        if (lhs_cont && rhs_cont) {                                                                \
            const size_t min_work_per_thread = 100000;                                             \
            size_t num_threads = get_optimal_threads(num_els, min_work_per_thread);                \
                                                                                                   \
            if (num_threads > 1) {                                                                 \
                thread_t threads[num_threads];                                                     \
                bitwise_##OP_NAME##_##TYPE_SUFFIX##_args_t args[num_threads];                      \
                                                                                                   \
                size_t chunk_size = num_els / num_threads;                                         \
                size_t remaining = num_els % num_threads;                                          \
                                                                                                   \
                for (size_t t = 0; t < num_threads; t++) {                                         \
                    args[t].lhs = l;                                                               \
                    args[t].rhs = r;                                                               \
                    args[t].output = out;                                                          \
                    args[t].lhs_offset = lhs_offset;                                               \
                    args[t].rhs_offset = rhs_offset;                                               \
                    args[t].start = t * chunk_size;                                                \
                    args[t].end = (t == num_threads - 1) ? (t + 1) * chunk_size + remaining        \
                                                         : (t + 1) * chunk_size;                   \
                    thread_create(&threads[t], bitwise_##OP_NAME##_##TYPE_SUFFIX##_worker,         \
                                  &args[t]);                                                       \
                }                                                                                  \
                for (size_t t = 0; t < num_threads; t++) {                                         \
                    thread_join(threads[t]);                                                       \
                }                                                                                  \
            } else {                                                                               \
                for (size_t i = 0; i < num_els; i++) {                                             \
                    TYPE x = l[lhs_offset + i];                                                    \
                    TYPE y = r[rhs_offset + i];                                                    \
                    out[i] = FUNC;                                                                 \
                }                                                                                  \
            }                                                                                      \
        } else {                                                                                   \
            for (size_t i = 0; i < num_els; i++) {                                                 \
                size_t lhs_idx = lhs_offset;                                                       \
                size_t rhs_idx = rhs_offset;                                                       \
                size_t idx = i;                                                                    \
                for (int d = num_dims - 1; d >= 0; d--) {                                          \
                    size_t coord = idx % lhs_shape[d];                                             \
                    idx /= lhs_shape[d];                                                           \
                    lhs_idx += coord * lhs_strides[d];                                             \
                    rhs_idx += coord * rhs_strides[d];                                             \
                }                                                                                  \
                TYPE x = l[lhs_idx];                                                               \
                TYPE y = r[rhs_idx];                                                               \
                out[i] = FUNC;                                                                     \
            }                                                                                      \
        }                                                                                          \
    }

// ============================================================================
// UNARY BITWISE OPERATION IMPLEMENTATION MACROS
// ============================================================================

#define IMPL_UNARY_BITWISE_OP(TYPE, TYPE_SUFFIX, OP_NAME, FUNC)                                    \
    typedef struct {                                                                               \
        const TYPE *input;                                                                         \
        TYPE *output;                                                                              \
        size_t start;                                                                              \
        size_t end;                                                                                \
        size_t offset;                                                                             \
    } unary_bitwise_##OP_NAME##_##TYPE_SUFFIX##_args_t;                                            \
                                                                                                   \
    static void *unary_bitwise_##OP_NAME##_##TYPE_SUFFIX##_worker(void *arg) {                     \
        unary_bitwise_##OP_NAME##_##TYPE_SUFFIX##_args_t *args =                                   \
            (unary_bitwise_##OP_NAME##_##TYPE_SUFFIX##_args_t *)arg;                               \
        for (size_t i = args->start; i < args->end; i++) {                                         \
            TYPE x = args->input[args->offset + i];                                                \
            args->output[i] = FUNC;                                                                \
        }                                                                                          \
        return NULL;                                                                               \
    }                                                                                              \
                                                                                                   \
    void hodu_cpu_##OP_NAME##_##TYPE_SUFFIX(const void *input, void *output,                       \
                                            const size_t *metadata) {                              \
        const TYPE *in = (const TYPE *)input;                                                      \
        TYPE *out = (TYPE *)output;                                                                \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *shape = metadata + 2;                                                        \
        const size_t *strides = metadata + 2 + num_dims;                                           \
        const size_t offset = metadata[2 + 2 * num_dims];                                          \
                                                                                                   \
        bool is_cont = is_contiguous(num_dims, shape, strides);                                    \
                                                                                                   \
        if (is_cont) {                                                                             \
            const size_t min_work_per_thread = 100000;                                             \
            size_t num_threads = get_optimal_threads(num_els, min_work_per_thread);                \
                                                                                                   \
            if (num_threads > 1) {                                                                 \
                thread_t threads[num_threads];                                                     \
                unary_bitwise_##OP_NAME##_##TYPE_SUFFIX##_args_t args[num_threads];                \
                                                                                                   \
                size_t chunk_size = num_els / num_threads;                                         \
                size_t remaining = num_els % num_threads;                                          \
                                                                                                   \
                for (size_t t = 0; t < num_threads; t++) {                                         \
                    args[t].input = in;                                                            \
                    args[t].output = out;                                                          \
                    args[t].offset = offset;                                                       \
                    args[t].start = t * chunk_size;                                                \
                    args[t].end = (t == num_threads - 1) ? (t + 1) * chunk_size + remaining        \
                                                         : (t + 1) * chunk_size;                   \
                    thread_create(&threads[t], unary_bitwise_##OP_NAME##_##TYPE_SUFFIX##_worker,   \
                                  &args[t]);                                                       \
                }                                                                                  \
                for (size_t t = 0; t < num_threads; t++) {                                         \
                    thread_join(threads[t]);                                                       \
                }                                                                                  \
            } else {                                                                               \
                for (size_t i = 0; i < num_els; i++) {                                             \
                    TYPE x = in[offset + i];                                                       \
                    out[i] = FUNC;                                                                 \
                }                                                                                  \
            }                                                                                      \
        } else {                                                                                   \
            for (size_t i = 0; i < num_els; i++) {                                                 \
                size_t src_idx = offset;                                                           \
                size_t idx = i;                                                                    \
                for (int d = num_dims - 1; d >= 0; d--) {                                          \
                    size_t coord = idx % shape[d];                                                 \
                    idx /= shape[d];                                                               \
                    src_idx += coord * strides[d];                                                 \
                }                                                                                  \
                TYPE x = in[src_idx];                                                              \
                out[i] = FUNC;                                                                     \
            }                                                                                      \
        }                                                                                          \
    }

// ============================================================================
// SCALAR SHIFT OPERATION IMPLEMENTATION MACROS
// ============================================================================

#define IMPL_SCALAR_SHIFT_OP(TYPE, TYPE_SUFFIX, OP_NAME, FUNC)                                     \
    typedef struct {                                                                               \
        const TYPE *input;                                                                         \
        TYPE *output;                                                                              \
        size_t start;                                                                              \
        size_t end;                                                                                \
        size_t offset;                                                                             \
        u32_t shift;                                                                               \
    } scalar_shift_##OP_NAME##_##TYPE_SUFFIX##_args_t;                                             \
                                                                                                   \
    static void *scalar_shift_##OP_NAME##_##TYPE_SUFFIX##_worker(void *arg) {                      \
        scalar_shift_##OP_NAME##_##TYPE_SUFFIX##_args_t *args =                                    \
            (scalar_shift_##OP_NAME##_##TYPE_SUFFIX##_args_t *)arg;                                \
        u32_t shift = args->shift;                                                                 \
        for (size_t i = args->start; i < args->end; i++) {                                         \
            TYPE x = args->input[args->offset + i];                                                \
            args->output[i] = FUNC;                                                                \
        }                                                                                          \
        return NULL;                                                                               \
    }                                                                                              \
                                                                                                   \
    void hodu_cpu_##OP_NAME##_##TYPE_SUFFIX(const void *input, void *output,                       \
                                            const size_t *metadata, u32_t shift) {                 \
        const TYPE *in = (const TYPE *)input;                                                      \
        TYPE *out = (TYPE *)output;                                                                \
                                                                                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *shape = metadata + 2;                                                        \
        const size_t *strides = metadata + 2 + num_dims;                                           \
        const size_t offset = metadata[2 + 2 * num_dims];                                          \
                                                                                                   \
        bool is_cont = is_contiguous(num_dims, shape, strides);                                    \
                                                                                                   \
        if (is_cont) {                                                                             \
            const size_t min_work_per_thread = 100000;                                             \
            size_t num_threads = get_optimal_threads(num_els, min_work_per_thread);                \
                                                                                                   \
            if (num_threads > 1) {                                                                 \
                thread_t threads[num_threads];                                                     \
                scalar_shift_##OP_NAME##_##TYPE_SUFFIX##_args_t args[num_threads];                 \
                                                                                                   \
                size_t chunk_size = num_els / num_threads;                                         \
                size_t remaining = num_els % num_threads;                                          \
                                                                                                   \
                for (size_t t = 0; t < num_threads; t++) {                                         \
                    args[t].input = in;                                                            \
                    args[t].output = out;                                                          \
                    args[t].offset = offset;                                                       \
                    args[t].shift = shift;                                                         \
                    args[t].start = t * chunk_size;                                                \
                    args[t].end = (t == num_threads - 1) ? (t + 1) * chunk_size + remaining        \
                                                         : (t + 1) * chunk_size;                   \
                    thread_create(&threads[t], scalar_shift_##OP_NAME##_##TYPE_SUFFIX##_worker,    \
                                  &args[t]);                                                       \
                }                                                                                  \
                for (size_t t = 0; t < num_threads; t++) {                                         \
                    thread_join(threads[t]);                                                       \
                }                                                                                  \
            } else {                                                                               \
                for (size_t i = 0; i < num_els; i++) {                                             \
                    TYPE x = in[offset + i];                                                       \
                    out[i] = FUNC;                                                                 \
                }                                                                                  \
            }                                                                                      \
        } else {                                                                                   \
            for (size_t i = 0; i < num_els; i++) {                                                 \
                size_t src_idx = offset;                                                           \
                size_t idx = i;                                                                    \
                for (int d = num_dims - 1; d >= 0; d--) {                                          \
                    size_t coord = idx % shape[d];                                                 \
                    idx /= shape[d];                                                               \
                    src_idx += coord * strides[d];                                                 \
                }                                                                                  \
                TYPE x = in[src_idx];                                                              \
                out[i] = FUNC;                                                                     \
            }                                                                                      \
        }                                                                                          \
    }

// ============================================================================
// UNSIGNED INTEGER IMPLEMENTATIONS
// ============================================================================

// u8
IMPL_BINARY_BITWISE_OP(u8_t, u8, shl, x << y)
IMPL_BINARY_BITWISE_OP(u8_t, u8, shr, x >> y)
IMPL_BINARY_BITWISE_OP(u8_t, u8, bitwise_and, x &y)
IMPL_BINARY_BITWISE_OP(u8_t, u8, bitwise_or, x | y)
IMPL_BINARY_BITWISE_OP(u8_t, u8, bitwise_xor, x ^ y)
IMPL_UNARY_BITWISE_OP(u8_t, u8, bitwise_not, ~x)

// u16
IMPL_BINARY_BITWISE_OP(u16_t, u16, shl, x << y)
IMPL_BINARY_BITWISE_OP(u16_t, u16, shr, x >> y)
IMPL_BINARY_BITWISE_OP(u16_t, u16, bitwise_and, x &y)
IMPL_BINARY_BITWISE_OP(u16_t, u16, bitwise_or, x | y)
IMPL_BINARY_BITWISE_OP(u16_t, u16, bitwise_xor, x ^ y)
IMPL_UNARY_BITWISE_OP(u16_t, u16, bitwise_not, ~x)

// u32
IMPL_BINARY_BITWISE_OP(u32_t, u32, shl, x << y)
IMPL_BINARY_BITWISE_OP(u32_t, u32, shr, x >> y)
IMPL_BINARY_BITWISE_OP(u32_t, u32, bitwise_and, x &y)
IMPL_BINARY_BITWISE_OP(u32_t, u32, bitwise_or, x | y)
IMPL_BINARY_BITWISE_OP(u32_t, u32, bitwise_xor, x ^ y)
IMPL_UNARY_BITWISE_OP(u32_t, u32, bitwise_not, ~x)

// u64
IMPL_BINARY_BITWISE_OP(u64_t, u64, shl, x << y)
IMPL_BINARY_BITWISE_OP(u64_t, u64, shr, x >> y)
IMPL_BINARY_BITWISE_OP(u64_t, u64, bitwise_and, x &y)
IMPL_BINARY_BITWISE_OP(u64_t, u64, bitwise_or, x | y)
IMPL_BINARY_BITWISE_OP(u64_t, u64, bitwise_xor, x ^ y)
IMPL_UNARY_BITWISE_OP(u64_t, u64, bitwise_not, ~x)

// ============================================================================
// SIGNED INTEGER IMPLEMENTATIONS
// ============================================================================

// i8
IMPL_BINARY_BITWISE_OP(i8_t, i8, shl, x << y)
IMPL_BINARY_BITWISE_OP(i8_t, i8, shr, x >> y)
IMPL_BINARY_BITWISE_OP(i8_t, i8, bitwise_and, x &y)
IMPL_BINARY_BITWISE_OP(i8_t, i8, bitwise_or, x | y)
IMPL_BINARY_BITWISE_OP(i8_t, i8, bitwise_xor, x ^ y)
IMPL_UNARY_BITWISE_OP(i8_t, i8, bitwise_not, ~x)

// i16
IMPL_BINARY_BITWISE_OP(i16_t, i16, shl, x << y)
IMPL_BINARY_BITWISE_OP(i16_t, i16, shr, x >> y)
IMPL_BINARY_BITWISE_OP(i16_t, i16, bitwise_and, x &y)
IMPL_BINARY_BITWISE_OP(i16_t, i16, bitwise_or, x | y)
IMPL_BINARY_BITWISE_OP(i16_t, i16, bitwise_xor, x ^ y)
IMPL_UNARY_BITWISE_OP(i16_t, i16, bitwise_not, ~x)

// i32
IMPL_BINARY_BITWISE_OP(i32_t, i32, shl, x << y)
IMPL_BINARY_BITWISE_OP(i32_t, i32, shr, x >> y)
IMPL_BINARY_BITWISE_OP(i32_t, i32, bitwise_and, x &y)
IMPL_BINARY_BITWISE_OP(i32_t, i32, bitwise_or, x | y)
IMPL_BINARY_BITWISE_OP(i32_t, i32, bitwise_xor, x ^ y)
IMPL_UNARY_BITWISE_OP(i32_t, i32, bitwise_not, ~x)

// i64
IMPL_BINARY_BITWISE_OP(i64_t, i64, shl, x << y)
IMPL_BINARY_BITWISE_OP(i64_t, i64, shr, x >> y)
IMPL_BINARY_BITWISE_OP(i64_t, i64, bitwise_and, x &y)
IMPL_BINARY_BITWISE_OP(i64_t, i64, bitwise_or, x | y)
IMPL_BINARY_BITWISE_OP(i64_t, i64, bitwise_xor, x ^ y)
IMPL_UNARY_BITWISE_OP(i64_t, i64, bitwise_not, ~x)

// ============================================================================
// SCALAR SHIFT IMPLEMENTATIONS
// ============================================================================

// u8
IMPL_SCALAR_SHIFT_OP(u8_t, u8, shl_scalar, x << shift)
IMPL_SCALAR_SHIFT_OP(u8_t, u8, shr_scalar, x >> shift)

// u16
IMPL_SCALAR_SHIFT_OP(u16_t, u16, shl_scalar, x << shift)
IMPL_SCALAR_SHIFT_OP(u16_t, u16, shr_scalar, x >> shift)

// u32
IMPL_SCALAR_SHIFT_OP(u32_t, u32, shl_scalar, x << shift)
IMPL_SCALAR_SHIFT_OP(u32_t, u32, shr_scalar, x >> shift)

// u64
IMPL_SCALAR_SHIFT_OP(u64_t, u64, shl_scalar, x << shift)
IMPL_SCALAR_SHIFT_OP(u64_t, u64, shr_scalar, x >> shift)

// i8
IMPL_SCALAR_SHIFT_OP(i8_t, i8, shl_scalar, x << shift)
IMPL_SCALAR_SHIFT_OP(i8_t, i8, shr_scalar, x >> shift)

// i16
IMPL_SCALAR_SHIFT_OP(i16_t, i16, shl_scalar, x << shift)
IMPL_SCALAR_SHIFT_OP(i16_t, i16, shr_scalar, x >> shift)

// i32
IMPL_SCALAR_SHIFT_OP(i32_t, i32, shl_scalar, x << shift)
IMPL_SCALAR_SHIFT_OP(i32_t, i32, shr_scalar, x >> shift)

// i64
IMPL_SCALAR_SHIFT_OP(i64_t, i64, shl_scalar, x << shift)
IMPL_SCALAR_SHIFT_OP(i64_t, i64, shr_scalar, x >> shift)
