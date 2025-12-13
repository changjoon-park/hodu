use crate::{
    be::{device::BackendDeviceT, storage::BackendStorageT},
    be_cpu::{device::CpuDevice, storage::CpuStorage},
    error::{HoduError, HoduResult},
    ops::Op,
    types::Layout,
};
use core::ffi::c_void;

pub fn call_ops_bitwise_binary(
    lhs_storage: &CpuStorage,
    rhs_storage: &CpuStorage,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    op: Op,
) -> HoduResult<CpuStorage> {
    // Extract bitwise binary op
    let bitwise_op = match op {
        Op::BitwiseBinary(bitwise_op) => bitwise_op,
        _ => {
            return Err(HoduError::BackendError(
                "call_ops_bitwise_binary expects bitwise binary op".to_string(),
            ))
        },
    };

    let lhs_shape = lhs_layout.shape();
    let num_els = lhs_shape.size();

    // Compute output layout for metadata generation
    let output_layout = lhs_layout.clone(); // Binary ops preserve layout

    // Generate metadata using centralized function
    let metadata = crate::op_metadatas::binary_metadata(lhs_layout, rhs_layout, &output_layout);

    // Use Display to get kernel name
    let kernel_name = format!("hodu_cpu_{}_{}", bitwise_op, lhs_storage.dtype());
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = hodu_cpu_kernels::macros::Kernel(kernel_name_static);

    // Create output storage (same dtype as input)
    let dtype = lhs_storage.dtype();
    let mut output = CpuDevice::allocate(num_els, dtype)?;

    // Get raw pointers and call kernel
    macro_rules! call_kernel {
        ($lhs_data:expr, $rhs_data:expr, $out_data:expr) => {{
            let lhs_ptr = $lhs_data.as_ptr() as *const c_void;
            let rhs_ptr = $rhs_data.as_ptr() as *const c_void;
            let out_ptr = $out_data.as_mut_ptr() as *mut c_void;

            hodu_cpu_kernels::call_ops_binary_bitwise(kernel, lhs_ptr, rhs_ptr, out_ptr, &metadata)?;
        }};
    }

    // Bitwise operations only support integer types
    match (lhs_storage, rhs_storage, &mut output) {
        (CpuStorage::U8(lhs), CpuStorage::U8(rhs), CpuStorage::U8(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        #[cfg(feature = "u16")]
        (CpuStorage::U16(lhs), CpuStorage::U16(rhs), CpuStorage::U16(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        (CpuStorage::U32(lhs), CpuStorage::U32(rhs), CpuStorage::U32(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        #[cfg(feature = "u64")]
        (CpuStorage::U64(lhs), CpuStorage::U64(rhs), CpuStorage::U64(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        (CpuStorage::I8(lhs), CpuStorage::I8(rhs), CpuStorage::I8(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        #[cfg(feature = "i16")]
        (CpuStorage::I16(lhs), CpuStorage::I16(rhs), CpuStorage::I16(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        (CpuStorage::I32(lhs), CpuStorage::I32(rhs), CpuStorage::I32(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        #[cfg(feature = "i64")]
        (CpuStorage::I64(lhs), CpuStorage::I64(rhs), CpuStorage::I64(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        _ => {
            return Err(HoduError::BackendError(
                "bitwise operations only support integer types".to_string(),
            ))
        },
    }

    Ok(output)
}

pub fn call_ops_bitwise_unary(storage: &CpuStorage, layout: &Layout, op: Op) -> HoduResult<CpuStorage> {
    // Extract bitwise unary op
    let bitwise_op = match op {
        Op::BitwiseUnary(bitwise_op) => bitwise_op,
        _ => {
            return Err(HoduError::BackendError(
                "call_ops_bitwise_unary expects bitwise unary op".to_string(),
            ))
        },
    };

    let shape = layout.shape();
    let num_els = shape.size();

    // Output layout is the same as input for bitwise ops
    let output_layout = layout.clone();

    // Generate metadata using centralized function
    let metadata = crate::op_metadatas::unary_metadata(layout, &output_layout);

    // Use Display to get kernel name
    let kernel_name = format!("hodu_cpu_{}_{}", bitwise_op, storage.dtype());
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = hodu_cpu_kernels::macros::Kernel(kernel_name_static);

    // Create output storage (same dtype as input)
    let dtype = storage.dtype();
    let mut output = CpuDevice::allocate(num_els, dtype)?;

    // Get raw pointers and call kernel
    macro_rules! call_kernel {
        ($in_data:expr, $out_data:expr) => {{
            let in_ptr = $in_data.as_ptr() as *const c_void;
            let out_ptr = $out_data.as_mut_ptr() as *mut c_void;

            hodu_cpu_kernels::call_ops_unary_bitwise(kernel, in_ptr, out_ptr, &metadata)?;
        }};
    }

    // Bitwise operations only support integer types
    match (storage, &mut output) {
        (CpuStorage::U8(input), CpuStorage::U8(out)) => {
            call_kernel!(input, out)
        },
        #[cfg(feature = "u16")]
        (CpuStorage::U16(input), CpuStorage::U16(out)) => {
            call_kernel!(input, out)
        },
        (CpuStorage::U32(input), CpuStorage::U32(out)) => {
            call_kernel!(input, out)
        },
        #[cfg(feature = "u64")]
        (CpuStorage::U64(input), CpuStorage::U64(out)) => {
            call_kernel!(input, out)
        },
        (CpuStorage::I8(input), CpuStorage::I8(out)) => {
            call_kernel!(input, out)
        },
        #[cfg(feature = "i16")]
        (CpuStorage::I16(input), CpuStorage::I16(out)) => {
            call_kernel!(input, out)
        },
        (CpuStorage::I32(input), CpuStorage::I32(out)) => {
            call_kernel!(input, out)
        },
        #[cfg(feature = "i64")]
        (CpuStorage::I64(input), CpuStorage::I64(out)) => {
            call_kernel!(input, out)
        },
        _ => {
            return Err(HoduError::BackendError(
                "bitwise operations only support integer types".to_string(),
            ))
        },
    }

    Ok(output)
}

pub fn call_ops_bitwise_unary_scalar(
    storage: &CpuStorage,
    layout: &Layout,
    shift: u32,
    op: Op,
) -> HoduResult<CpuStorage> {
    // Extract bitwise unary scalar op
    let bitwise_op = match op {
        Op::BitwiseUnaryScalar(bitwise_op) => bitwise_op,
        _ => {
            return Err(HoduError::BackendError(
                "call_ops_bitwise_unary_scalar expects bitwise unary scalar op".to_string(),
            ))
        },
    };

    let shape = layout.shape();
    let num_els = shape.size();

    // Output layout is the same as input for bitwise ops
    let output_layout = layout.clone();

    // Generate metadata using centralized function
    let metadata = crate::op_metadatas::unary_metadata(layout, &output_layout);

    // Use Display to get kernel name
    let kernel_name = format!("hodu_cpu_{}_{}", bitwise_op, storage.dtype());
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = hodu_cpu_kernels::macros::Kernel(kernel_name_static);

    // Create output storage (same dtype as input)
    let dtype = storage.dtype();
    let mut output = CpuDevice::allocate(num_els, dtype)?;

    // Get raw pointers and call kernel
    macro_rules! call_kernel {
        ($in_data:expr, $out_data:expr) => {{
            let in_ptr = $in_data.as_ptr() as *const c_void;
            let out_ptr = $out_data.as_mut_ptr() as *mut c_void;

            hodu_cpu_kernels::call_ops_scalar_shift(kernel, in_ptr, out_ptr, &metadata, shift)?;
        }};
    }

    // Bitwise operations only support integer types
    match (storage, &mut output) {
        (CpuStorage::U8(input), CpuStorage::U8(out)) => {
            call_kernel!(input, out)
        },
        #[cfg(feature = "u16")]
        (CpuStorage::U16(input), CpuStorage::U16(out)) => {
            call_kernel!(input, out)
        },
        (CpuStorage::U32(input), CpuStorage::U32(out)) => {
            call_kernel!(input, out)
        },
        #[cfg(feature = "u64")]
        (CpuStorage::U64(input), CpuStorage::U64(out)) => {
            call_kernel!(input, out)
        },
        (CpuStorage::I8(input), CpuStorage::I8(out)) => {
            call_kernel!(input, out)
        },
        #[cfg(feature = "i16")]
        (CpuStorage::I16(input), CpuStorage::I16(out)) => {
            call_kernel!(input, out)
        },
        (CpuStorage::I32(input), CpuStorage::I32(out)) => {
            call_kernel!(input, out)
        },
        #[cfg(feature = "i64")]
        (CpuStorage::I64(input), CpuStorage::I64(out)) => {
            call_kernel!(input, out)
        },
        _ => {
            return Err(HoduError::BackendError(
                "bitwise scalar shift operations only support integer types".to_string(),
            ))
        },
    }

    Ok(output)
}
