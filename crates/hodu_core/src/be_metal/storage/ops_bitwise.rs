use crate::{
    be::storage::BackendStorageT,
    be_metal::storage::MetalStorage,
    error::{HoduError, HoduResult},
    ops::Op,
    types::Layout,
};
use hodu_metal_kernels::{kernels, utils::BufferOffset};

pub fn call_ops_bitwise_binary(
    lhs_storage: &MetalStorage,
    rhs_storage: &MetalStorage,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    op: Op,
) -> HoduResult<MetalStorage> {
    // Extract bitwise binary op
    let bitwise_op = match op {
        Op::BitwiseBinary(bitwise_op) => bitwise_op,
        _ => {
            return Err(HoduError::BackendError(
                "call_ops_bitwise_binary expects bitwise binary op".to_string(),
            ))
        },
    };

    let output_layout = lhs_layout.clone();
    let metadata = crate::op_metadatas::binary_metadata(lhs_layout, rhs_layout, &output_layout);

    let num_els = lhs_layout.shape().size();
    let dtype = lhs_storage.dtype();
    let device = lhs_storage.backend_device();

    // Create output buffer (same dtype as input)
    let output_buffer = device.new_buffer(num_els, dtype, "bitwise_binary_output")?;

    // Get kernel name
    let kernel_name = format!("hodu_metal_{}_{}", bitwise_op, dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Create buffer offsets for inputs
    let lhs_offset = BufferOffset::zero_offset(lhs_storage.buffer());
    let rhs_offset = BufferOffset::zero_offset(rhs_storage.buffer());

    // Get command buffer and call kernel
    let command_buffer = device.command_buffer()?;
    kernels::call_ops_binary_bitwise(
        kernel,
        device.kernels(),
        device.device(),
        &command_buffer,
        lhs_offset,
        rhs_offset,
        &output_buffer,
        &metadata,
    )?;

    Ok(MetalStorage::new(output_buffer, device.clone(), num_els, dtype))
}

pub fn call_ops_bitwise_unary(storage: &MetalStorage, layout: &Layout, op: Op) -> HoduResult<MetalStorage> {
    // Extract bitwise unary op
    let bitwise_op = match op {
        Op::BitwiseUnary(bitwise_op) => bitwise_op,
        _ => {
            return Err(HoduError::BackendError(
                "call_ops_bitwise_unary expects bitwise unary op".to_string(),
            ))
        },
    };

    // Output layout is the same as input for bitwise ops
    let output_layout = layout.clone();
    let metadata = crate::op_metadatas::unary_metadata(layout, &output_layout);

    let num_els = layout.shape().size();
    let dtype = storage.dtype();
    let device = storage.backend_device();

    // Create output buffer (same dtype as input)
    let output_buffer = device.new_buffer(num_els, dtype, "bitwise_unary_output")?;

    // Get kernel name
    let kernel_name = format!("hodu_metal_{}_{}", bitwise_op, dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Create buffer offset for input
    let input_offset = BufferOffset::zero_offset(storage.buffer());

    // Get command buffer and call kernel
    let command_buffer = device.command_buffer()?;
    kernels::call_ops_unary_bitwise(
        kernel,
        device.kernels(),
        device.device(),
        &command_buffer,
        input_offset,
        &output_buffer,
        &metadata,
    )?;

    Ok(MetalStorage::new(output_buffer, device.clone(), num_els, dtype))
}

pub fn call_ops_bitwise_unary_scalar(
    storage: &MetalStorage,
    layout: &Layout,
    shift: u32,
    op: Op,
) -> HoduResult<MetalStorage> {
    // Extract bitwise unary scalar op
    let bitwise_op = match op {
        Op::BitwiseUnaryScalar(bitwise_op) => bitwise_op,
        _ => {
            return Err(HoduError::BackendError(
                "call_ops_bitwise_unary_scalar expects bitwise unary scalar op".to_string(),
            ))
        },
    };

    // Output layout is the same as input for bitwise ops
    let output_layout = layout.clone();
    let metadata = crate::op_metadatas::unary_metadata(layout, &output_layout);

    let num_els = layout.shape().size();
    let dtype = storage.dtype();
    let device = storage.backend_device();

    // Create output buffer (same dtype as input)
    let output_buffer = device.new_buffer(num_els, dtype, "bitwise_scalar_shift_output")?;

    // Get kernel name
    let kernel_name = format!("hodu_metal_{}_{}", bitwise_op, dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Create buffer offset for input
    let input_offset = BufferOffset::zero_offset(storage.buffer());

    // Get command buffer and call kernel
    let command_buffer = device.command_buffer()?;
    kernels::call_ops_scalar_shift(
        kernel,
        device.kernels(),
        device.device(),
        &command_buffer,
        input_offset,
        &output_buffer,
        &metadata,
        shift,
    )?;

    Ok(MetalStorage::new(output_buffer, device.clone(), num_els, dtype))
}
