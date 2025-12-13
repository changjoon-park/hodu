use crate::{
    cuda::*,
    error::{CudaKernelError, Result},
    kernel::Kernels,
    kernels::macros::ops,
    source::Source,
};

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
/// Only integer types are supported (u8, u16, u32, u64, i8, i16, i32, i64).
///
/// # Arguments
/// * `kernel` - The binary bitwise operation to perform (e.g., "shl::U32", "bitwise_and::I64")
/// * `kernels` - Kernel cache for managing compiled kernels
/// * `context` - CUDA context to execute on
/// * `lhs` - Left-hand side tensor device slice
/// * `rhs` - Right-hand side tensor device slice
/// * `output` - Output tensor device slice
/// * `metadata` - Device slice containing metadata describing tensor layout
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of elements to process)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: lhs_shape (shape of left tensor)
/// - metadata[2+num_dims..2+2*num_dims]: rhs_shape (shape of right tensor)
/// - metadata[2+2*num_dims..2+3*num_dims]: lhs_strides (stride of left tensor)
/// - metadata[2+3*num_dims..2+4*num_dims]: rhs_strides (stride of right tensor)
/// - metadata[2+4*num_dims]: lhs_offset (starting offset in left tensor)
/// - metadata[2+4*num_dims+1]: rhs_offset (starting offset in right tensor)
///
/// Total metadata length: `2 + num_dims * 4 + 2`
pub fn call_ops_binary_bitwise<T>(
    kernel: crate::kernels::macros::Kernel,
    kernels: &Kernels,
    context: &Arc<CudaContext>,
    lhs: &CudaSlice<T>,
    rhs: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    metadata: &[usize],
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr,
{
    let func = kernels.load_function(context, Source::OpsBitwise, kernel.0)?;

    let num_els = metadata[0];
    let block_size = 256u32;
    let grid_size = (num_els as u32).div_ceil(block_size).max(1);

    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let stream = context.default_stream();
    let metadata_dev = stream
        .memcpy_stod(metadata)
        .map_err(|e| CudaKernelError::MemoryError(format!("Failed to copy metadata: {:?}", e)))?;

    unsafe {
        func.launch(&stream, cfg, |args| {
            args.arg(lhs).arg(rhs).arg(output).arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}

/// Execute a unary bitwise operation on a tensor
///
/// Performs element-wise unary bitwise operations (bitwise NOT) on tensors.
/// Only integer types are supported (u8, u16, u32, u64, i8, i16, i32, i64).
///
/// # Arguments
/// * `kernel` - The unary bitwise operation to perform (e.g., "bitwise_not::U32")
/// * `kernels` - Kernel cache for managing compiled kernels
/// * `context` - CUDA context to execute on
/// * `input` - Input tensor device slice
/// * `output` - Output tensor device slice
/// * `metadata` - Device slice containing metadata describing tensor layout
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of elements to process)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: shape (shape of tensor)
/// - metadata[2+num_dims..2+2*num_dims]: strides (stride of tensor)
/// - metadata[2+2*num_dims]: offset (starting offset in tensor)
///
/// Total metadata length: `2 + num_dims * 2 + 1`
pub fn call_ops_unary_bitwise<T>(
    kernel: crate::kernels::macros::Kernel,
    kernels: &Kernels,
    context: &Arc<CudaContext>,
    input: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    metadata: &[usize],
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr,
{
    let func = kernels.load_function(context, Source::OpsBitwise, kernel.0)?;

    let num_els = metadata[0];
    let block_size = 256u32;
    let grid_size = (num_els as u32).div_ceil(block_size).max(1);

    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let stream = context.default_stream();
    let metadata_dev = stream
        .memcpy_stod(metadata)
        .map_err(|e| CudaKernelError::MemoryError(format!("Failed to copy metadata: {:?}", e)))?;

    unsafe {
        func.launch(&stream, cfg, |args| {
            args.arg(input).arg(output).arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}

/// Execute a scalar shift operation on a tensor
///
/// Shifts all elements by a constant scalar amount.
/// Only integer types are supported (u8, u16, u32, u64, i8, i16, i32, i64).
///
/// # Arguments
/// * `kernel` - The scalar shift operation to perform (e.g., "shl_scalar::U32", "shr_scalar::I64")
/// * `kernels` - Kernel cache for managing compiled kernels
/// * `context` - CUDA context to execute on
/// * `input` - Input tensor device slice
/// * `output` - Output tensor device slice
/// * `metadata` - Device slice containing metadata describing tensor layout
/// * `shift` - Shift amount
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of elements to process)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: shape (shape of tensor)
/// - metadata[2+num_dims..2+2*num_dims]: strides (stride of tensor)
/// - metadata[2+2*num_dims]: offset (starting offset in tensor)
///
/// Total metadata length: `2 + num_dims * 2 + 1`
pub fn call_ops_scalar_shift<T>(
    kernel: crate::kernels::macros::Kernel,
    kernels: &Kernels,
    context: &Arc<CudaContext>,
    input: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    metadata: &[usize],
    shift: u32,
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr,
{
    let func = kernels.load_function(context, Source::OpsBitwise, kernel.0)?;

    let num_els = metadata[0];
    let block_size = 256u32;
    let grid_size = (num_els as u32).div_ceil(block_size).max(1);

    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let stream = context.default_stream();
    let metadata_dev = stream
        .memcpy_stod(metadata)
        .map_err(|e| CudaKernelError::MemoryError(format!("Failed to copy metadata: {:?}", e)))?;

    unsafe {
        func.launch(&stream, cfg, |args| {
            args.arg(input).arg(output).arg(&metadata_dev).arg(&shift);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}
