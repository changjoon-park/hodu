use crate::{
    be::storage::BackendStorageT,
    be_cuda::storage::{CudaStorage, CudaStorageData},
    error::{HoduError, HoduResult},
    ops::Op,
    types::Layout,
};
use hodu_cuda_kernels::{cuda::CudaSlice, kernels};
use std::sync::Arc;

pub fn call_ops_bitwise_binary(
    lhs_storage: &CudaStorage,
    rhs_storage: &CudaStorage,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    op: Op,
) -> HoduResult<CudaStorage> {
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

    let lhs_shape = lhs_layout.shape();
    let num_els = lhs_shape.size();
    let dtype = lhs_storage.dtype();
    let device = lhs_storage.get_device();

    let kernel_name = format!("hodu_cuda_{}_{}", bitwise_op, dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);
    let device_id = lhs_storage.device_id();
    let device_arc = Arc::clone(&lhs_storage.device);

    macro_rules! call_bitwise_binary {
        ($lhs:expr, $rhs:expr, $ty:ty) => {{
            let mut output: CudaSlice<$ty> = device.new_buffer(num_els as usize)?;
            kernels::call_ops_binary_bitwise(
                kernel,
                device.kernels(),
                device.context(),
                $lhs,
                $rhs,
                &mut output,
                &metadata,
            )?;
            output
        }};
    }

    // Bitwise operations only support integer types
    match (&lhs_storage.data, &rhs_storage.data) {
        (CudaStorageData::U8(lhs), CudaStorageData::U8(rhs)) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::U8(call_bitwise_binary!(lhs, rhs, u8)),
        )),
        #[cfg(feature = "u16")]
        (CudaStorageData::U16(lhs), CudaStorageData::U16(rhs)) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::U16(call_bitwise_binary!(lhs, rhs, u16)),
        )),
        (CudaStorageData::U32(lhs), CudaStorageData::U32(rhs)) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::U32(call_bitwise_binary!(lhs, rhs, u32)),
        )),
        #[cfg(feature = "u64")]
        (CudaStorageData::U64(lhs), CudaStorageData::U64(rhs)) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::U64(call_bitwise_binary!(lhs, rhs, u64)),
        )),
        (CudaStorageData::I8(lhs), CudaStorageData::I8(rhs)) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::I8(call_bitwise_binary!(lhs, rhs, i8)),
        )),
        #[cfg(feature = "i16")]
        (CudaStorageData::I16(lhs), CudaStorageData::I16(rhs)) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::I16(call_bitwise_binary!(lhs, rhs, i16)),
        )),
        (CudaStorageData::I32(lhs), CudaStorageData::I32(rhs)) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::I32(call_bitwise_binary!(lhs, rhs, i32)),
        )),
        #[cfg(feature = "i64")]
        (CudaStorageData::I64(lhs), CudaStorageData::I64(rhs)) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::I64(call_bitwise_binary!(lhs, rhs, i64)),
        )),
        _ => Err(HoduError::BackendError(
            "bitwise operations only support integer types".to_string(),
        )),
    }
}

pub fn call_ops_bitwise_unary(storage: &CudaStorage, layout: &Layout, op: Op) -> HoduResult<CudaStorage> {
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

    let shape = layout.shape();
    let num_els = shape.size();
    let dtype = storage.dtype();
    let device = storage.get_device();

    let kernel_name = format!("hodu_cuda_{}_{}", bitwise_op, dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);
    let device_id = storage.device_id();
    let device_arc = Arc::clone(&storage.device);

    macro_rules! call_bitwise_unary {
        ($input:expr, $ty:ty) => {{
            let mut output: CudaSlice<$ty> = device.new_buffer(num_els as usize)?;
            kernels::call_ops_unary_bitwise(
                kernel,
                device.kernels(),
                device.context(),
                $input,
                &mut output,
                &metadata,
            )?;
            output
        }};
    }

    // Bitwise operations only support integer types
    match &storage.data {
        CudaStorageData::U8(input) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::U8(call_bitwise_unary!(input, u8)),
        )),
        #[cfg(feature = "u16")]
        CudaStorageData::U16(input) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::U16(call_bitwise_unary!(input, u16)),
        )),
        CudaStorageData::U32(input) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::U32(call_bitwise_unary!(input, u32)),
        )),
        #[cfg(feature = "u64")]
        CudaStorageData::U64(input) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::U64(call_bitwise_unary!(input, u64)),
        )),
        CudaStorageData::I8(input) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::I8(call_bitwise_unary!(input, i8)),
        )),
        #[cfg(feature = "i16")]
        CudaStorageData::I16(input) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::I16(call_bitwise_unary!(input, i16)),
        )),
        CudaStorageData::I32(input) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::I32(call_bitwise_unary!(input, i32)),
        )),
        #[cfg(feature = "i64")]
        CudaStorageData::I64(input) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::I64(call_bitwise_unary!(input, i64)),
        )),
        _ => Err(HoduError::BackendError(
            "bitwise operations only support integer types".to_string(),
        )),
    }
}

pub fn call_ops_bitwise_unary_scalar(
    storage: &CudaStorage,
    layout: &Layout,
    shift: u32,
    op: Op,
) -> HoduResult<CudaStorage> {
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

    let shape = layout.shape();
    let num_els = shape.size();
    let dtype = storage.dtype();
    let device = storage.get_device();

    let kernel_name = format!("hodu_cuda_{}_{}", bitwise_op, dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);
    let device_id = storage.device_id();
    let device_arc = Arc::clone(&storage.device);

    macro_rules! call_scalar_shift {
        ($input:expr, $ty:ty) => {{
            let mut output: CudaSlice<$ty> = device.new_buffer(num_els as usize)?;
            kernels::call_ops_scalar_shift(
                kernel,
                device.kernels(),
                device.context(),
                $input,
                &mut output,
                &metadata,
                shift,
            )?;
            output
        }};
    }

    // Bitwise operations only support integer types
    match &storage.data {
        CudaStorageData::U8(input) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::U8(call_scalar_shift!(input, u8)),
        )),
        #[cfg(feature = "u16")]
        CudaStorageData::U16(input) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::U16(call_scalar_shift!(input, u16)),
        )),
        CudaStorageData::U32(input) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::U32(call_scalar_shift!(input, u32)),
        )),
        #[cfg(feature = "u64")]
        CudaStorageData::U64(input) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::U64(call_scalar_shift!(input, u64)),
        )),
        CudaStorageData::I8(input) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::I8(call_scalar_shift!(input, i8)),
        )),
        #[cfg(feature = "i16")]
        CudaStorageData::I16(input) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::I16(call_scalar_shift!(input, i16)),
        )),
        CudaStorageData::I32(input) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::I32(call_scalar_shift!(input, i32)),
        )),
        #[cfg(feature = "i64")]
        CudaStorageData::I64(input) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::I64(call_scalar_shift!(input, i64)),
        )),
        _ => Err(HoduError::BackendError(
            "bitwise scalar shift operations only support integer types".to_string(),
        )),
    }
}
