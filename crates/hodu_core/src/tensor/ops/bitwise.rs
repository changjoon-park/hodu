use crate::{
    error::HoduResult,
    op_params::BitwiseUnaryScalarParams,
    ops::{
        BitwiseBinaryOp, BitwiseBinaryParams, BitwiseUnaryOp, BitwiseUnaryParams, BitwiseUnaryScalarOp, Op, OpParams,
    },
    tensor::{create_builder_tensor, from_storage_with_context, utils::broadcast_tensors2, Tensor},
    utils::valid::{validate_dtype_for_device, validate_dtype_for_op, validate_same_device, validate_same_dtype},
};

macro_rules! bitwise_binary_op {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name(&self, rhs: &Self) -> HoduResult<Self> {
            validate_same_device(&[self, rhs], Op::BitwiseBinary(BitwiseBinaryOp::$op_name))?;
            validate_same_dtype(&[self, rhs], Op::BitwiseBinary(BitwiseBinaryOp::$op_name))?;
            validate_dtype_for_device(self.dtype(), self.device())?;
            validate_dtype_for_op(self.dtype(), Op::BitwiseBinary(BitwiseBinaryOp::$op_name))?;

            let (lhs, rhs) = broadcast_tensors2(self, rhs)?;

            if crate::snapshot::capture::is_active() {
                let lhs_layout = lhs.layout();
                let rhs_layout = rhs.layout();
                let result_layout = lhs_layout.clone();
                // Bitwise ops don't support gradient, requires_grad is always false
                let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), lhs.dtype(), false);

                crate::snapshot::capture::capture_operation(
                    Op::BitwiseBinary(BitwiseBinaryOp::$op_name),
                    Some(OpParams::BitwiseBinary(BitwiseBinaryParams)),
                    vec![lhs.id(), rhs.id()],
                    result_id,
                    vec![lhs_layout, rhs_layout],
                    result_layout,
                )?;

                Ok(result_tensor)
            } else {
                let lhs_layout = lhs.layout();
                let rhs_layout = rhs.layout();

                let storage = lhs.with_storage(|lhs_storage| {
                    rhs.with_storage(|rhs_storage| {
                        lhs_storage.call_ops_bitwise_binary(
                            rhs_storage,
                            &lhs_layout,
                            &rhs_layout,
                            Op::BitwiseBinary(BitwiseBinaryOp::$op_name),
                        )
                    })
                })?;

                // Bitwise ops don't support gradient
                let result = from_storage_with_context(storage, lhs_layout, true, false);

                Ok(result)
            }
        }
    };
}

macro_rules! bitwise_unary_op {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name(&self) -> HoduResult<Self> {
            validate_dtype_for_device(self.dtype(), self.device())?;
            validate_dtype_for_op(self.dtype(), Op::BitwiseUnary(BitwiseUnaryOp::$op_name))?;

            if crate::snapshot::capture::is_active() {
                let input_layout = self.layout();
                let result_layout = input_layout.clone();
                // Bitwise ops don't support gradient, requires_grad is always false
                let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), self.dtype(), false);

                crate::snapshot::capture::capture_operation(
                    Op::BitwiseUnary(BitwiseUnaryOp::$op_name),
                    Some(OpParams::BitwiseUnary(BitwiseUnaryParams)),
                    vec![self.id()],
                    result_id,
                    vec![input_layout],
                    result_layout,
                )?;

                Ok(result_tensor)
            } else {
                let input_layout = self.layout();

                let storage = self.with_storage(|storage| {
                    storage.call_ops_bitwise_unary(&input_layout, Op::BitwiseUnary(BitwiseUnaryOp::$op_name))
                })?;

                // Bitwise ops don't support gradient
                let result = from_storage_with_context(storage, input_layout, true, false);

                Ok(result)
            }
        }
    };
}

macro_rules! bitwise_scalar_shift_op {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name(&self, shift: u32) -> HoduResult<Self> {
            validate_dtype_for_device(self.dtype(), self.device())?;
            validate_dtype_for_op(
                self.dtype(),
                Op::BitwiseUnaryScalar(BitwiseUnaryScalarOp::$op_name),
            )?;

            if crate::snapshot::capture::is_active() {
                let input_layout = self.layout();
                let result_layout = input_layout.clone();
                // Bitwise ops don't support gradient, requires_grad is always false
                let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), self.dtype(), false);

                crate::snapshot::capture::capture_operation(
                    Op::BitwiseUnaryScalar(BitwiseUnaryScalarOp::$op_name),
                    Some(OpParams::BitwiseUnaryScalar(BitwiseUnaryScalarParams { shift })),
                    vec![self.id()],
                    result_id,
                    vec![input_layout],
                    result_layout,
                )?;

                Ok(result_tensor)
            } else {
                let input_layout = self.layout();

                let storage = self.with_storage(|storage| {
                    storage.call_ops_bitwise_unary_scalar(
                        &input_layout,
                        shift,
                        Op::BitwiseUnaryScalar(BitwiseUnaryScalarOp::$op_name),
                    )
                })?;

                // Bitwise ops don't support gradient
                let result = from_storage_with_context(storage, input_layout, true, false);

                Ok(result)
            }
        }
    };
}

// Bitwise operations (integer only)
impl Tensor {
    bitwise_binary_op!(shl, Shl);
    bitwise_binary_op!(shr, Shr);
    bitwise_binary_op!(bitwise_and, And);
    bitwise_binary_op!(bitwise_or, Or);
    bitwise_binary_op!(bitwise_xor, Xor);

    bitwise_unary_op!(bitwise_not, Not);

    bitwise_scalar_shift_op!(shl_scalar, ShlScalar);
    bitwise_scalar_shift_op!(shr_scalar, ShrScalar);
}
