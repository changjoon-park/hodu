use crate::{
    error::MetalKernelError,
    kernel::Kernels,
    kernels::macros::ops,
    metal::{Buffer, ComputeCommandEncoder, Device},
    set_params,
    source::Source,
    utils::{linear_split, BufferOffset, EncoderProvider},
};
use objc2_metal::MTLResourceUsage;

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

/// Executes a binary bitwise operation on two input tensors using Metal compute pipeline.
///
/// # Arguments
/// * `kernel` - Binary bitwise operation kernel to execute (shl, shr, bitwise_and, bitwise_or, bitwise_xor)
/// * `kernels` - Kernel cache
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
/// * `left_input` - Left input tensor buffer with offset
/// * `right_input` - Right input tensor buffer with offset
/// * `output` - Output buffer
/// * `metadata` - Metadata describing tensor shapes and strides
///
/// # Metadata Layout
/// The metadata buffer must contain the following elements in order:
/// - `metadata[0]`: num_els (total number of output elements)
/// - `metadata[1]`: num_dims (number of dimensions)
/// - `metadata[2..2+num_dims]`: lhs_shape (shape of left input)
/// - `metadata[2+num_dims..2+2*num_dims]`: rhs_shape (shape of right input)
/// - `metadata[2+2*num_dims..2+3*num_dims]`: lhs_strides (strides of left input)
/// - `metadata[2+3*num_dims..2+4*num_dims]`: rhs_strides (strides of right input)
/// - `metadata[2+4*num_dims]`: lhs_offset (starting offset in left input buffer)
/// - `metadata[2+4*num_dims+1]`: rhs_offset (starting offset in right input buffer)
///
/// Total metadata length: `2 + num_dims * 4 + 2`
///
/// # Supported Types
/// Only integer types are supported: U8, U16, U32, U64, I8, I16, I32, I64
#[allow(clippy::too_many_arguments)]
pub fn call_ops_binary_bitwise(
    kernel: Kernel,
    kernels: &Kernels,
    device: &Device,
    ep: impl EncoderProvider,
    left_input: BufferOffset,
    right_input: BufferOffset,
    output: &Buffer,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Bitwise, kernel.0)?;

    let num_els = metadata[0];

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Metal kernel signature:
    // buffer(0): lhs input
    // buffer(1): rhs input
    // buffer(2): output
    // buffer(3): metadata
    set_params!(encoder, (&left_input, &right_input, output, metadata));

    encoder.use_resource(left_input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(right_input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}

/// Executes a unary bitwise operation on a tensor using Metal compute pipeline.
///
/// # Arguments
/// * `kernel` - Unary bitwise operation kernel (bitwise_not)
/// * `kernels` - Kernel cache
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
/// * `input` - Input tensor buffer
/// * `output` - Output buffer
/// * `metadata` - Metadata describing tensor shape, strides, and offset
///
/// # Metadata Layout
/// Total metadata length: `2 + num_dims * 2 + 1`
///
/// - `metadata[0]`: num_els (total number of elements)
/// - `metadata[1]`: num_dims (number of dimensions)
/// - `metadata[2..2+num_dims]`: shape (dimensions of the tensor)
/// - `metadata[2+num_dims..2+2*num_dims]`: strides (stride for each dimension)
/// - `metadata[2+2*num_dims]`: offset (starting offset in input buffer)
///
/// # Supported Types
/// Only integer types are supported: U8, U16, U32, U64, I8, I16, I32, I64
pub fn call_ops_unary_bitwise(
    kernel: Kernel,
    kernels: &Kernels,
    device: &Device,
    ep: impl EncoderProvider,
    input: BufferOffset,
    output: &Buffer,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Bitwise, kernel.0)?;

    let num_els = metadata[0];

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Metal kernel signature:
    // buffer(0): input
    // buffer(1): output
    // buffer(2): metadata
    set_params!(encoder, (&input, output, metadata));

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}

/// Executes a scalar shift operation on a tensor using Metal compute pipeline.
///
/// # Arguments
/// * `kernel` - Scalar shift operation kernel (shl_scalar, shr_scalar)
/// * `kernels` - Kernel cache
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
/// * `input` - Input tensor buffer
/// * `output` - Output buffer
/// * `metadata` - Metadata describing tensor shape, strides, and offset
/// * `shift` - Shift amount
///
/// # Metadata Layout
/// Total metadata length: `2 + num_dims * 2 + 1`
///
/// - `metadata[0]`: num_els (total number of elements)
/// - `metadata[1]`: num_dims (number of dimensions)
/// - `metadata[2..2+num_dims]`: shape (dimensions of the tensor)
/// - `metadata[2+num_dims..2+2*num_dims]`: strides (stride for each dimension)
/// - `metadata[2+2*num_dims]`: offset (starting offset in input buffer)
///
/// # Supported Types
/// Only integer types are supported: U8, U16, U32, U64, I8, I16, I32, I64
#[allow(clippy::too_many_arguments)]
pub fn call_ops_scalar_shift(
    kernel: Kernel,
    kernels: &Kernels,
    device: &Device,
    ep: impl EncoderProvider,
    input: BufferOffset,
    output: &Buffer,
    metadata: &[usize],
    shift: u32,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Bitwise, kernel.0)?;

    let num_els = metadata[0];

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Metal kernel signature:
    // buffer(0): input
    // buffer(1): output
    // buffer(2): metadata
    // buffer(3): shift
    set_params!(encoder, (&input, output, metadata, shift));

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}
