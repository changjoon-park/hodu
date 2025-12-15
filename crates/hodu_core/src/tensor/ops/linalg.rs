use crate::{
    error::HoduResult,
    ops::{DetParams, InvParams, LinalgOp, Op, OpParams, TraceParams},
    scalar::Scalar,
    tensor::{create_builder_tensor, from_storage_with_context, gradient, Tensor},
    types::{Layout, Shape},
    utils::valid::{validate_dtype_for_device, validate_dtype_for_op, validate_requires_grad_for_op},
};

impl Tensor {
    /// Compute the determinant of a square matrix.
    ///
    /// # Input
    /// - Square matrix `[..., N, N]` (supports batched input)
    ///
    /// # Output
    /// - Scalar `[...]` (batch dimensions preserved)
    ///
    /// # Example
    /// ```ignore
    /// let matrix = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    /// let det = matrix.det()?; // Returns -2.0
    /// ```
    pub fn det(&self) -> HoduResult<Self> {
        let op = Op::Linalg(LinalgOp::Det);

        // Validate dtype for device and operation
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), op.clone())?;

        let shape = self.shape();
        let ndim = shape.ndim();

        // Validate shape - need at least 2D and square matrix
        if ndim < 2 {
            return Err(crate::error::HoduError::InvalidArgument(
                "det requires at least 2D tensor".to_string(),
            ));
        }

        let n = shape.dims()[ndim - 1];
        let m = shape.dims()[ndim - 2];

        if n != m {
            return Err(crate::error::HoduError::InvalidArgument(format!(
                "det requires square matrix, got {}×{}",
                m, n
            )));
        }

        // Compute output shape (batch dimensions only)
        let output_shape = if ndim == 2 {
            Shape::new(&[1])
        } else {
            Shape::new(&shape.dims()[..ndim - 2])
        };

        let result_layout = Layout::from_shape(&output_shape);
        let self_layout = self.layout();
        let validate_requires_grad = validate_requires_grad_for_op(op.clone());

        if crate::snapshot::capture::is_active() {
            let requires_grad = self.is_requires_grad() && validate_requires_grad;
            let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), self.dtype(), requires_grad);

            crate::snapshot::capture::capture_operation(
                op.clone(),
                Some(OpParams::Det(DetParams)),
                vec![self.id()],
                result_id,
                vec![self_layout],
                result_layout,
            )?;

            if requires_grad {
                gradient::record_operation(vec![self.id()], result_id, op, OpParams::Det(DetParams))?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| storage.call_ops_det(&self_layout))?;

            let requires_grad = self.is_requires_grad() && validate_requires_grad;
            let result = from_storage_with_context(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                gradient::record_operation(vec![self.id()], result.id(), op, OpParams::Det(DetParams))?;
            }

            Ok(result)
        }
    }

    /// Compute the inverse of a square matrix.
    ///
    /// # Input
    /// - Square matrix `[..., N, N]` (supports batched input)
    ///
    /// # Output
    /// - Inverse matrix `[..., N, N]` (same shape as input)
    ///
    /// # Example
    /// ```ignore
    /// let matrix = Tensor::from_slice(&[4.0, 7.0, 2.0, 6.0], &[2, 2])?;
    /// let inv = matrix.inv()?; // Returns [[0.6, -0.7], [-0.2, 0.4]]
    /// ```
    ///
    /// # Notes
    /// - For singular matrices, the result will contain inf/nan values.
    /// - Uses Gauss-Jordan elimination with partial pivoting for numerical stability.
    pub fn inv(&self) -> HoduResult<Self> {
        let op = Op::Linalg(LinalgOp::Inv);

        // Validate dtype for device and operation
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), op.clone())?;

        let shape = self.shape();
        let ndim = shape.ndim();

        // Validate shape - need at least 2D and square matrix
        if ndim < 2 {
            return Err(crate::error::HoduError::InvalidArgument(
                "inv requires at least 2D tensor".to_string(),
            ));
        }

        let n = shape.dims()[ndim - 1];
        let m = shape.dims()[ndim - 2];

        if n != m {
            return Err(crate::error::HoduError::InvalidArgument(format!(
                "inv requires square matrix, got {}×{}",
                m, n
            )));
        }

        // Output shape is same as input
        let output_shape = shape.clone();
        let result_layout = Layout::from_shape(&output_shape);
        let self_layout = self.layout();
        let validate_requires_grad = validate_requires_grad_for_op(op.clone());

        if crate::snapshot::capture::is_active() {
            let requires_grad = self.is_requires_grad() && validate_requires_grad;
            let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), self.dtype(), requires_grad);

            crate::snapshot::capture::capture_operation(
                op.clone(),
                Some(OpParams::Inv(InvParams)),
                vec![self.id()],
                result_id,
                vec![self_layout],
                result_layout,
            )?;

            if requires_grad {
                gradient::record_operation(vec![self.id()], result_id, op, OpParams::Inv(InvParams))?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| storage.call_ops_inv(&self_layout))?;

            let requires_grad = self.is_requires_grad() && validate_requires_grad;
            let result = from_storage_with_context(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                gradient::record_operation(vec![self.id()], result.id(), op, OpParams::Inv(InvParams))?;
            }

            Ok(result)
        }
    }

    /// Compute the trace of a square matrix (sum of diagonal elements).
    ///
    /// # Input
    /// - Square matrix `[..., N, N]` (supports batched input)
    ///
    /// # Output
    /// - Scalar `[...]` (batch dimensions preserved)
    ///
    /// # Example
    /// ```ignore
    /// let matrix = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    /// let tr = matrix.trace()?; // Returns 5.0 (1.0 + 4.0)
    /// ```
    pub fn trace(&self) -> HoduResult<Self> {
        let op = Op::Linalg(LinalgOp::Trace);

        // Validate dtype for device and operation
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), op.clone())?;

        let shape = self.shape();
        let ndim = shape.ndim();

        // Validate shape - need at least 2D and square matrix
        if ndim < 2 {
            return Err(crate::error::HoduError::InvalidArgument(
                "trace requires at least 2D tensor".to_string(),
            ));
        }

        let n = shape.dims()[ndim - 1];
        let m = shape.dims()[ndim - 2];

        if n != m {
            return Err(crate::error::HoduError::InvalidArgument(format!(
                "trace requires square matrix, got {}×{}",
                m, n
            )));
        }

        // Compute output shape (batch dimensions only)
        let output_shape = if ndim == 2 {
            Shape::new(&[1])
        } else {
            Shape::new(&shape.dims()[..ndim - 2])
        };

        let result_layout = Layout::from_shape(&output_shape);
        let self_layout = self.layout();
        let validate_requires_grad = validate_requires_grad_for_op(op.clone());

        if crate::snapshot::capture::is_active() {
            let requires_grad = self.is_requires_grad() && validate_requires_grad;
            let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), self.dtype(), requires_grad);

            crate::snapshot::capture::capture_operation(
                op.clone(),
                Some(OpParams::Trace(TraceParams)),
                vec![self.id()],
                result_id,
                vec![self_layout],
                result_layout,
            )?;

            if requires_grad {
                gradient::record_operation(vec![self.id()], result_id, op, OpParams::Trace(TraceParams))?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| storage.call_ops_trace(&self_layout))?;

            let requires_grad = self.is_requires_grad() && validate_requires_grad;
            let result = from_storage_with_context(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                gradient::record_operation(vec![self.id()], result.id(), op, OpParams::Trace(TraceParams))?;
            }

            Ok(result)
        }
    }

    /// Solve a linear system Ax = b.
    ///
    /// # Input
    /// - `self` (A): Square matrix `[..., N, N]` (coefficient matrix)
    /// - `b`: Vector or matrix `[..., N]` or `[..., N, K]` (right-hand side)
    ///
    /// # Output
    /// - Solution `x` with shape `[..., N]` or `[..., N, K]`
    ///
    /// # Example
    /// ```ignore
    /// let a = Tensor::from_slice(&[2.0, 1.0, 1.0, 3.0], &[2, 2])?;
    /// let b = Tensor::from_slice(&[4.0, 7.0], &[2])?;
    /// let x = a.solve(&b)?; // Solves Ax = b
    /// ```
    ///
    /// # Notes
    /// - Internally uses `inv(A) @ b` for simplicity.
    /// - For singular matrices, the result will contain inf/nan values.
    pub fn solve(&self, b: &Tensor) -> HoduResult<Self> {
        // solve(A, b) = inv(A) @ b
        let inv_a = self.inv()?;

        // For 1D b, unsqueeze -> matmul -> squeeze
        if b.ndim() == 1 {
            let b_col = b.unsqueeze(-1)?;
            let x_col = inv_a.matmul(&b_col)?;
            x_col.squeeze(&[-1])
        } else {
            inv_a.matmul(b)
        }
    }

    /// Returns the lower triangular part of the matrix, zeroing out elements above the k-th diagonal.
    ///
    /// # Input
    /// - Matrix `[..., N, M]` (supports batched input)
    /// - `diagonal`: The diagonal above which to zero elements.
    ///   - `k = 0` (default): main diagonal
    ///   - `k > 0`: above main diagonal
    ///   - `k < 0`: below main diagonal
    ///
    /// # Output
    /// - Lower triangular matrix `[..., N, M]` (same shape as input)
    ///
    /// # Example
    /// ```ignore
    /// let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3])?;
    /// let lower = x.tril(0)?;
    /// // [[1, 0, 0],
    /// //  [4, 5, 0],
    /// //  [7, 8, 9]]
    /// ```
    pub fn tril<D: Into<Scalar>>(&self, diagonal: D) -> HoduResult<Self> {
        let diagonal = diagonal.into().to_i32();
        let shape = self.shape();
        let ndim = shape.ndim();

        if ndim < 2 {
            return Err(crate::error::HoduError::InvalidArgument(
                "tril requires at least 2D tensor".to_string(),
            ));
        }

        let n = shape.dims()[ndim - 2];
        let m = shape.dims()[ndim - 1];

        // Create row indices [0, 1, 2, ...] with shape [N, 1]
        let row_indices = Self::arange(0i32, n as i32, 1i32)?
            .reshape([n, 1])?
            .to_device(self.device())?;

        // Create col indices [0, 1, 2, ...] with shape [1, M]
        let col_indices = Self::arange(0i32, m as i32, 1i32)?
            .reshape([1, m])?
            .to_device(self.device())?;

        // tril mask: row >= col - diagonal → row + diagonal >= col
        let row_plus_diag = row_indices.add_scalar(diagonal)?;
        let mask = row_plus_diag.ge(&col_indices)?;

        // Convert mask to input dtype and multiply
        let mask_typed = mask.to_dtype(self.dtype())?;
        self.mul(&mask_typed)
    }

    /// Returns the upper triangular part of the matrix, zeroing out elements below the k-th diagonal.
    ///
    /// # Input
    /// - Matrix `[..., N, M]` (supports batched input)
    /// - `diagonal`: The diagonal below which to zero elements.
    ///   - `k = 0` (default): main diagonal
    ///   - `k > 0`: above main diagonal
    ///   - `k < 0`: below main diagonal
    ///
    /// # Output
    /// - Upper triangular matrix `[..., N, M]` (same shape as input)
    ///
    /// # Example
    /// ```ignore
    /// let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3])?;
    /// let upper = x.triu(0)?;
    /// // [[1, 2, 3],
    /// //  [0, 5, 6],
    /// //  [0, 0, 9]]
    /// ```
    pub fn triu<D: Into<Scalar>>(&self, diagonal: D) -> HoduResult<Self> {
        let diagonal = diagonal.into().to_i32();
        let shape = self.shape();
        let ndim = shape.ndim();

        if ndim < 2 {
            return Err(crate::error::HoduError::InvalidArgument(
                "triu requires at least 2D tensor".to_string(),
            ));
        }

        let n = shape.dims()[ndim - 2];
        let m = shape.dims()[ndim - 1];

        // Create row indices [0, 1, 2, ...] with shape [N, 1]
        let row_indices = Self::arange(0i32, n as i32, 1i32)?
            .reshape([n, 1])?
            .to_device(self.device())?;

        // Create col indices [0, 1, 2, ...] with shape [1, M]
        let col_indices = Self::arange(0i32, m as i32, 1i32)?
            .reshape([1, m])?
            .to_device(self.device())?;

        // triu mask: col >= row + diagonal
        let row_plus_diag = row_indices.add_scalar(diagonal)?;
        let mask = col_indices.ge(&row_plus_diag)?;

        // Convert mask to input dtype and multiply
        let mask_typed = mask.to_dtype(self.dtype())?;
        self.mul(&mask_typed)
    }

    /// If input is 1D: creates a 2D diagonal matrix with input on the k-th diagonal.
    /// If input is 2D: extracts the k-th diagonal as a 1D tensor.
    ///
    /// # Arguments
    /// - `diagonal`: The diagonal offset.
    ///   - `k = 0`: main diagonal
    ///   - `k > 0`: above main diagonal
    ///   - `k < 0`: below main diagonal
    ///
    /// # Examples
    /// ```ignore
    /// // 1D -> 2D: create diagonal matrix
    /// let v = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3])?;
    /// let m = v.diag(0)?;
    /// // [[1, 0, 0],
    /// //  [0, 2, 0],
    /// //  [0, 0, 3]]
    ///
    /// // 2D -> 1D: extract diagonal
    /// let m = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3])?;
    /// let d = m.diag(0)?;  // [1, 5, 9]
    /// let d1 = m.diag(1)?; // [2, 6]
    /// ```
    pub fn diag<D: Into<Scalar>>(&self, diagonal: D) -> HoduResult<Self> {
        let k = diagonal.into().to_i32();
        let ndim = self.ndim();

        match ndim {
            1 => self.diag_1d_to_2d(k),
            2 => self.diag_2d_to_1d(k),
            _ => Err(crate::error::HoduError::InvalidArgument(
                "diag requires 1D or 2D tensor".to_string(),
            )),
        }
    }

    /// Creates a 2D diagonal matrix from a 1D tensor.
    fn diag_1d_to_2d(&self, k: i32) -> HoduResult<Self> {
        let n = self.shape().dims()[0];
        let abs_k = k.unsigned_abs() as usize;
        let size = n + abs_k;

        // Determine starting position based on diagonal offset
        let (row_start, col_start) = if k >= 0 { (0, k as usize) } else { (abs_k, 0) };

        // Create flat indices for diagonal positions
        // For position i: flat_idx = (row_start + i) * size + (col_start + i)
        //                         = row_start * size + col_start + i * (size + 1)
        let base = (row_start * size + col_start) as i32;
        let stride = (size + 1) as i32;

        let i_range = Self::arange(0i32, n as i32, 1i32)?.to_device(self.device())?;
        let indices = i_range
            .mul_scalar(stride)?
            .add_scalar(base)?
            .to_dtype(crate::types::DType::I32)?;

        // Create zeros matrix and flatten
        let zeros = Self::zeros([size, size], self.dtype())?.to_device(self.device())?;
        let flat_zeros = zeros.reshape([size * size])?;

        // Scatter input values to diagonal positions
        let result_flat = flat_zeros.scatter(0, &indices, self)?;
        result_flat.reshape([size, size])
    }

    /// Extracts the k-th diagonal from a 2D matrix as a 1D tensor.
    fn diag_2d_to_1d(&self, k: i32) -> HoduResult<Self> {
        let shape = self.shape();
        let n = shape.dims()[0];
        let m = shape.dims()[1];

        // Calculate diagonal length
        let diag_len = if k >= 0 {
            let k_usize = k as usize;
            if k_usize >= m {
                0
            } else {
                n.min(m - k_usize)
            }
        } else {
            let abs_k = (-k) as usize;
            if abs_k >= n {
                0
            } else {
                m.min(n - abs_k)
            }
        };

        if diag_len == 0 {
            return Self::zeros([0], self.dtype());
        }

        // Starting position
        let (row_start, col_start) = if k >= 0 {
            (0usize, k as usize)
        } else {
            ((-k) as usize, 0usize)
        };

        // Create indices for gathering
        // row_indices = [row_start, row_start+1, ..., row_start+diag_len-1]
        // col_indices = [col_start, col_start+1, ..., col_start+diag_len-1]
        let row_indices =
            Self::arange(row_start as i32, (row_start + diag_len) as i32, 1i32)?.to_device(self.device())?;
        let col_indices =
            Self::arange(col_start as i32, (col_start + diag_len) as i32, 1i32)?.to_device(self.device())?;

        // Use gather: first select rows, then gather from columns
        // self[row_indices, col_indices] for each i
        let rows_selected = self.index_select(0, &row_indices.to_dtype(crate::types::DType::I32)?)?;

        // Now rows_selected is [diag_len, m], we need element [i, col_indices[i]] for each i
        // Reshape col_indices to [diag_len, 1] for gather along dim 1
        let col_indices_reshaped = col_indices.reshape([diag_len, 1])?.to_dtype(crate::types::DType::I32)?;
        let result = rows_selected.gather(1, &col_indices_reshaped)?;

        // Result is [diag_len, 1], squeeze to [diag_len]
        result.reshape([diag_len])
    }

    /// Extracts diagonal elements from a tensor along specified dimensions.
    ///
    /// # Arguments
    /// - `offset`: Diagonal offset (k=0 main diagonal, k>0 above, k<0 below)
    /// - `dim1`: First dimension for the 2D sub-matrix
    /// - `dim2`: Second dimension for the 2D sub-matrix
    ///
    /// # Example
    /// ```ignore
    /// let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3])?;
    /// let d = x.diagonal(0, 0, 1)?; // [1, 5, 9]
    /// ```
    pub fn diagonal<D: Into<Scalar>>(&self, offset: D, dim1: i32, dim2: i32) -> HoduResult<Self> {
        let k = offset.into().to_i32();
        let ndim = self.ndim() as i32;

        if ndim < 2 {
            return Err(crate::error::HoduError::InvalidArgument(
                "diagonal requires at least 2D tensor".to_string(),
            ));
        }

        // Normalize negative dimensions
        let dim1_norm = if dim1 < 0 { ndim + dim1 } else { dim1 } as usize;
        let dim2_norm = if dim2 < 0 { ndim + dim2 } else { dim2 } as usize;

        if dim1_norm == dim2_norm {
            return Err(crate::error::HoduError::InvalidArgument(
                "diagonal dimensions must be different".to_string(),
            ));
        }

        if dim1_norm >= ndim as usize || dim2_norm >= ndim as usize {
            return Err(crate::error::HoduError::InvalidArgument(format!(
                "diagonal dimensions out of range for {}D tensor",
                ndim
            )));
        }

        // For 2D case, use diag directly
        if ndim == 2 && dim1_norm == 0 && dim2_norm == 1 {
            return self.diag(k);
        }

        // For higher dimensions: permute to move dim1, dim2 to last two dims, then extract diagonal
        let mut perm: Vec<usize> = (0..ndim as usize).collect();
        perm.retain(|&x| x != dim1_norm && x != dim2_norm);
        perm.push(dim1_norm);
        perm.push(dim2_norm);

        let permuted = self.permute(&perm.iter().map(|&x| x as i32).collect::<Vec<_>>())?;

        // Now extract diagonal from last two dimensions
        let perm_shape = permuted.shape();
        let perm_dims = perm_shape.dims();
        let batch_dims: Vec<usize> = perm_dims[..perm_dims.len() - 2].to_vec();
        let n = perm_dims[perm_dims.len() - 2];
        let m = perm_dims[perm_dims.len() - 1];

        // Calculate diagonal length
        let diag_len = if k >= 0 {
            let k_usize = k as usize;
            if k_usize >= m {
                0
            } else {
                n.min(m - k_usize)
            }
        } else {
            let abs_k = (-k) as usize;
            if abs_k >= n {
                0
            } else {
                m.min(n - abs_k)
            }
        };

        if diag_len == 0 {
            let mut out_shape = batch_dims.clone();
            out_shape.push(0);
            return Self::zeros(out_shape, self.dtype());
        }

        // Flatten batch dimensions
        let batch_size: usize = batch_dims.iter().product();
        let flattened = permuted.reshape([batch_size, n, m])?;

        // Extract diagonal for each batch
        let (row_start, col_start) = if k >= 0 {
            (0usize, k as usize)
        } else {
            ((-k) as usize, 0usize)
        };

        let row_indices =
            Self::arange(row_start as i32, (row_start + diag_len) as i32, 1i32)?.to_device(self.device())?;
        let col_indices =
            Self::arange(col_start as i32, (col_start + diag_len) as i32, 1i32)?.to_device(self.device())?;

        // For batched operation: select rows then gather columns
        let row_idx_i32 = row_indices.to_dtype(crate::types::DType::I32)?;
        let col_idx_i32 = col_indices.to_dtype(crate::types::DType::I32)?.reshape([diag_len, 1])?;

        let rows_selected = flattened.index_select(1, &row_idx_i32)?; // [batch, diag_len, m]
        let result = rows_selected.gather(2, &col_idx_i32.broadcast([batch_size, diag_len, 1])?)?; // [batch, diag_len, 1]

        // Reshape to [batch_dims..., diag_len]
        let mut out_shape = batch_dims;
        out_shape.push(diag_len);
        result.reshape(out_shape)
    }
}
