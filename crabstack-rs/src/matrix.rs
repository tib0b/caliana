//! Small dense row-major matrix used throughout the registration core.
//!
//! Mirrors the `matrix<double>` template from the C++ TurboReg port. Indexing is
//! `(row, col)` via the `Index`/`IndexMut` impls so the numeric code reads close
//! to the original.

use std::ops::{Index, IndexMut};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Mat {
    pub fn new(rows: usize, cols: usize) -> Self {
        Mat {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Build from a row-major slice.
    pub fn from_rows(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols, "data length must equal rows*cols");
        Mat { rows, cols, data }
    }

    #[inline]
    pub fn nrows(&self) -> usize {
        self.rows
    }

    #[inline]
    pub fn ncols(&self) -> usize {
        self.cols
    }

    #[inline]
    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    #[inline]
    pub fn row_mut(&mut self, i: usize) -> &mut [f64] {
        &mut self.data[i * self.cols..(i + 1) * self.cols]
    }
}

impl Index<(usize, usize)> for Mat {
    type Output = f64;
    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        &self.data[i * self.cols + j]
    }
}

impl IndexMut<(usize, usize)> for Mat {
    #[inline]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        &mut self.data[i * self.cols + j]
    }
}
