//! Bitwise operations for tensor element-wise computations
//!
//! This module provides element-wise bitwise operations including:
//! - Binary: shl, shr, bitwise_and, bitwise_or, bitwise_xor
//! - Unary: bitwise_not
//! - Scalar shift: shl_scalar, shr_scalar
//!
//! All operations support integer types only (u8-u64, i8-i64).
//! Float and bool types are not supported.

use crate::{error::Result, kernels::macros::ops};
use core::ffi::c_void;

// Define all bitwise operations using the macro
ops!(
    shl,
    shr,
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    bitwise_not,
    shl_scalar,
    shr_scalar
);

/// Execute a binary bitwise operation on two tensors
///
/// Performs element-wise binary bitwise operations on tensors with arbitrary shapes and strides.
/// Only integer types are supported.
///
/// # Arguments
/// * `kernel_name` - The bitwise operation to perform (e.g., shl::U32, bitwise_and::I64)
/// * `lhs` - Pointer to left-hand side tensor data
/// * `rhs` - Pointer to right-hand side tensor data
/// * `output` - Pointer to output tensor buffer
/// * `metadata` - Tensor metadata array (same layout as binary ops)
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of elements to process)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: lhs_shape
/// - metadata[2+num_dims..2+2*num_dims]: rhs_shape
/// - metadata[2+2*num_dims..2+3*num_dims]: lhs_strides
/// - metadata[2+3*num_dims..2+4*num_dims]: rhs_strides
/// - metadata[2+4*num_dims]: lhs_offset
/// - metadata[2+4*num_dims+1]: rhs_offset
pub fn call_ops_binary_bitwise(
    kernel_name: crate::kernels::macros::Kernel,
    lhs: *const c_void,
    rhs: *const c_void,
    output: *mut c_void,
    metadata: &[usize],
) -> Result<()> {
    unsafe {
        dispatch_binary_bitwise(kernel_name.0, lhs, rhs, output, metadata.as_ptr());
    }

    Ok(())
}

/// Execute a unary bitwise operation on a tensor
///
/// Performs element-wise unary bitwise operations (currently only bitwise_not).
/// Only integer types are supported.
///
/// # Arguments
/// * `kernel_name` - The bitwise operation to perform (e.g., bitwise_not::U32)
/// * `input` - Pointer to input tensor data
/// * `output` - Pointer to output tensor buffer
/// * `metadata` - Tensor metadata array (same layout as unary ops)
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of elements)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: shape
/// - metadata[2+num_dims..2+2*num_dims]: strides
/// - metadata[2+2*num_dims]: offset
pub fn call_ops_unary_bitwise(
    kernel_name: crate::kernels::macros::Kernel,
    input: *const c_void,
    output: *mut c_void,
    metadata: &[usize],
) -> Result<()> {
    unsafe {
        dispatch_unary_bitwise(kernel_name.0, input, output, metadata.as_ptr());
    }

    Ok(())
}

/// Macro to generate extern C declarations and dispatch logic for binary bitwise operations
/// Only integer types are supported (u8, u16, u32, u64, i8, i16, i32, i64)
macro_rules! declare_and_dispatch_binary_bitwise {
    ($($op:ident),* $(,)?) => {
        paste::paste! {
            // Extern C declarations for all operations and integer types
            extern "C" {
                $(
                    fn [<hodu_cpu_ $op _u8>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _u16>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _u32>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _u64>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _i8>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _i16>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _i32>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _i64>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, metadata: *const usize);
                )*
            }

            // Dispatch function
            unsafe fn dispatch_binary_bitwise(
                name: &str,
                lhs: *const c_void,
                rhs: *const c_void,
                output: *mut c_void,
                metadata: *const usize,
            ) {
                match name {
                    $(
                        concat!("hodu_cpu_", stringify!($op), "_u8") => [<hodu_cpu_ $op _u8>](lhs, rhs, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_u16") => [<hodu_cpu_ $op _u16>](lhs, rhs, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_u32") => [<hodu_cpu_ $op _u32>](lhs, rhs, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_u64") => [<hodu_cpu_ $op _u64>](lhs, rhs, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_i8") => [<hodu_cpu_ $op _i8>](lhs, rhs, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_i16") => [<hodu_cpu_ $op _i16>](lhs, rhs, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_i32") => [<hodu_cpu_ $op _i32>](lhs, rhs, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_i64") => [<hodu_cpu_ $op _i64>](lhs, rhs, output, metadata),
                    )*
                    _ => panic!("Unsupported binary bitwise kernel: {}", name),
                }
            }
        }
    };
}

/// Macro to generate extern C declarations and dispatch logic for unary bitwise operations
/// Only integer types are supported (u8, u16, u32, u64, i8, i16, i32, i64)
macro_rules! declare_and_dispatch_unary_bitwise {
    ($($op:ident),* $(,)?) => {
        paste::paste! {
            // Extern C declarations for all operations and integer types
            extern "C" {
                $(
                    fn [<hodu_cpu_ $op _u8>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _u16>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _u32>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _u64>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _i8>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _i16>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _i32>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _i64>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                )*
            }

            // Dispatch function
            unsafe fn dispatch_unary_bitwise(
                name: &str,
                input: *const c_void,
                output: *mut c_void,
                metadata: *const usize,
            ) {
                match name {
                    $(
                        concat!("hodu_cpu_", stringify!($op), "_u8") => [<hodu_cpu_ $op _u8>](input, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_u16") => [<hodu_cpu_ $op _u16>](input, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_u32") => [<hodu_cpu_ $op _u32>](input, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_u64") => [<hodu_cpu_ $op _u64>](input, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_i8") => [<hodu_cpu_ $op _i8>](input, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_i16") => [<hodu_cpu_ $op _i16>](input, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_i32") => [<hodu_cpu_ $op _i32>](input, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_i64") => [<hodu_cpu_ $op _i64>](input, output, metadata),
                    )*
                    _ => panic!("Unsupported unary bitwise kernel: {}", name),
                }
            }
        }
    };
}

// Declare binary bitwise operations
declare_and_dispatch_binary_bitwise!(shl, shr, bitwise_and, bitwise_or, bitwise_xor);

// Declare unary bitwise operations
declare_and_dispatch_unary_bitwise!(bitwise_not);

/// Execute a scalar shift operation on a tensor
///
/// Performs element-wise shift by a scalar amount.
/// Only integer types are supported.
///
/// # Arguments
/// * `kernel_name` - The shift operation to perform (e.g., shl_scalar::U32, shr_scalar::I64)
/// * `input` - Pointer to input tensor data
/// * `output` - Pointer to output tensor buffer
/// * `metadata` - Tensor metadata array (same layout as unary ops)
/// * `shift` - Shift amount
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of elements)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: shape
/// - metadata[2+num_dims..2+2*num_dims]: strides
/// - metadata[2+2*num_dims]: offset
pub fn call_ops_scalar_shift(
    kernel_name: crate::kernels::macros::Kernel,
    input: *const c_void,
    output: *mut c_void,
    metadata: &[usize],
    shift: u32,
) -> Result<()> {
    unsafe {
        dispatch_scalar_shift(kernel_name.0, input, output, metadata.as_ptr(), shift);
    }

    Ok(())
}

/// Macro to generate extern C declarations and dispatch logic for scalar shift operations
/// Only integer types are supported (u8, u16, u32, u64, i8, i16, i32, i64)
macro_rules! declare_and_dispatch_scalar_shift {
    ($($op:ident),* $(,)?) => {
        paste::paste! {
            // Extern C declarations for all operations and integer types
            extern "C" {
                $(
                    fn [<hodu_cpu_ $op _u8>](input: *const c_void, output: *mut c_void, metadata: *const usize, shift: u32);
                    fn [<hodu_cpu_ $op _u16>](input: *const c_void, output: *mut c_void, metadata: *const usize, shift: u32);
                    fn [<hodu_cpu_ $op _u32>](input: *const c_void, output: *mut c_void, metadata: *const usize, shift: u32);
                    fn [<hodu_cpu_ $op _u64>](input: *const c_void, output: *mut c_void, metadata: *const usize, shift: u32);
                    fn [<hodu_cpu_ $op _i8>](input: *const c_void, output: *mut c_void, metadata: *const usize, shift: u32);
                    fn [<hodu_cpu_ $op _i16>](input: *const c_void, output: *mut c_void, metadata: *const usize, shift: u32);
                    fn [<hodu_cpu_ $op _i32>](input: *const c_void, output: *mut c_void, metadata: *const usize, shift: u32);
                    fn [<hodu_cpu_ $op _i64>](input: *const c_void, output: *mut c_void, metadata: *const usize, shift: u32);
                )*
            }

            // Dispatch function
            unsafe fn dispatch_scalar_shift(
                name: &str,
                input: *const c_void,
                output: *mut c_void,
                metadata: *const usize,
                shift: u32,
            ) {
                match name {
                    $(
                        concat!("hodu_cpu_", stringify!($op), "_u8") => [<hodu_cpu_ $op _u8>](input, output, metadata, shift),
                        concat!("hodu_cpu_", stringify!($op), "_u16") => [<hodu_cpu_ $op _u16>](input, output, metadata, shift),
                        concat!("hodu_cpu_", stringify!($op), "_u32") => [<hodu_cpu_ $op _u32>](input, output, metadata, shift),
                        concat!("hodu_cpu_", stringify!($op), "_u64") => [<hodu_cpu_ $op _u64>](input, output, metadata, shift),
                        concat!("hodu_cpu_", stringify!($op), "_i8") => [<hodu_cpu_ $op _i8>](input, output, metadata, shift),
                        concat!("hodu_cpu_", stringify!($op), "_i16") => [<hodu_cpu_ $op _i16>](input, output, metadata, shift),
                        concat!("hodu_cpu_", stringify!($op), "_i32") => [<hodu_cpu_ $op _i32>](input, output, metadata, shift),
                        concat!("hodu_cpu_", stringify!($op), "_i64") => [<hodu_cpu_ $op _i64>](input, output, metadata, shift),
                    )*
                    _ => panic!("Unsupported scalar shift kernel: {}", name),
                }
            }
        }
    };
}

// Declare scalar shift operations
declare_and_dispatch_scalar_shift!(shl_scalar, shr_scalar);
