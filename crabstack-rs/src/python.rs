//! PyO3 bindings. `_register(ref, mov, transformation)` and `_transform(mov,
//! matrix)` keep the signatures pystackreg's `turboreg` extension had, so the
//! two stay swappable; `_register` additionally takes the optional per-frame
//! masks that pystackreg never exposed.
//!
//! Built by maturin into caliana's package tree as `caliana._crabstack`, which
//! is why the module is named for the file rather than for the crate.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{
    register_masked as core_register, transform as core_transform, Image2D, Mask, Mat,
    Transformation,
};

/// Build a row-major [`Image2D`] from a (possibly non-contiguous) numpy array.
/// `ndarray`'s logical iteration yields row-major order regardless of strides,
/// matching what pystackreg's `PyArray_FROM_OTF(..., IN_ARRAY)` copy produces.
fn image_from_array(a: &PyReadonlyArray2<f64>) -> Image2D {
    let view = a.as_array();
    let (height, width) = (view.shape()[0], view.shape()[1]);
    let data: Vec<f64> = view.iter().copied().collect();
    Image2D {
        width,
        height,
        data,
    }
}

/// Build a mask from a boolean array, checking it against the image it guards.
/// `which` names it in the error message.
fn mask_from_array(a: &PyReadonlyArray2<bool>, img: &Image2D, which: &str) -> PyResult<Mask> {
    let view = a.as_array();
    let (height, width) = (view.shape()[0], view.shape()[1]);
    if width != img.width || height != img.height {
        return Err(PyValueError::new_err(format!(
            "{which} mask has shape ({height}, {width}), expected ({}, {})",
            img.height, img.width
        )));
    }
    let mask = Mask::new(width, height, view.iter().copied().collect());
    if mask.active_count() == 0 {
        return Err(PyValueError::new_err(format!(
            "{which} mask excludes every pixel, leaving nothing to register on"
        )));
    }
    Ok(mask)
}

/// What `_register` hands back: `(matrix, ref_points, mov_points)`.
type RegistrationArrays<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
);

/// `_register(ref, mov, transformation, ref_mask=None, mov_mask=None)` →
/// `(matrix, ref_pts, mov_pts)`.
///
/// The masks are boolean arrays of the same shape as their image; `True` means
/// the pixel takes part in the fit. Omitting both is the pystackreg behaviour —
/// every pixel counts — and costs nothing extra.
#[pyfunction]
#[pyo3(name = "_register")]
#[pyo3(signature = (reference, moving, transformation, ref_mask=None, mov_mask=None))]
fn register_py<'py>(
    py: Python<'py>,
    reference: PyReadonlyArray2<'py, f64>,
    moving: PyReadonlyArray2<'py, f64>,
    transformation: i32,
    ref_mask: Option<PyReadonlyArray2<'py, bool>>,
    mov_mask: Option<PyReadonlyArray2<'py, bool>>,
) -> PyResult<RegistrationArrays<'py>> {
    let t = Transformation::from_code(transformation)
        .ok_or_else(|| PyValueError::new_err("Invalid transformation"))?;
    let ref_img = image_from_array(&reference);
    let mov_img = image_from_array(&moving);
    if ref_img.width != mov_img.width || ref_img.height != mov_img.height {
        return Err(PyValueError::new_err(
            "Input arrays must be of the same shape",
        ));
    }
    let ref_msk = ref_mask
        .map(|m| mask_from_array(&m, &ref_img, "reference"))
        .transpose()?;
    let mov_msk = mov_mask
        .map(|m| mask_from_array(&m, &mov_img, "moving"))
        .transpose()?;

    // Release the GIL for the numeric work.
    let reg =
        py.detach(|| core_register(&ref_img, ref_msk.as_ref(), &mov_img, mov_msk.as_ref(), t));

    Ok((
        mat_to_pyarray(py, &reg.matrix),
        mat_to_pyarray(py, &reg.ref_points),
        mat_to_pyarray(py, &reg.mov_points),
    ))
}

/// `turboreg._transform(mov, matrix)` → warped image.
#[pyfunction]
#[pyo3(name = "_transform")]
fn transform_py<'py>(
    py: Python<'py>,
    moving: PyReadonlyArray2<'py, f64>,
    matrix: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let mov_img = image_from_array(&moving);
    let mview = matrix.as_array();
    if mview.shape()[0] != 2 {
        return Err(PyValueError::new_err(
            "Transformation matrix must be of shape (2,1), (2,3) or (2,4)",
        ));
    }
    let cols = mview.shape()[1];
    let short = Mat::from_rows(2, cols, mview.iter().copied().collect());

    let out = py.detach(|| core_transform(&mov_img, &short));
    Ok(Array2::from_shape_vec((out.height, out.width), out.data)
        .expect("shape matches data")
        .into_pyarray(py))
}

/// `set_num_threads(n)` — `1` disables parallelism, `0` restores the default.
///
/// Results do not depend on this: the work split is fixed by image size, so any
/// thread count produces bit-identical output. Set it to 1 when driving
/// crabstack from your own worker pool.
#[pyfunction]
fn set_num_threads(n: usize) {
    crate::set_num_threads(n);
}

/// The thread count currently in effect.
#[pyfunction]
fn num_threads() -> usize {
    crate::num_threads()
}

fn mat_to_pyarray<'py>(py: Python<'py>, m: &Mat) -> Bound<'py, PyArray2<f64>> {
    Array2::from_shape_vec((m.rows, m.cols), m.data.clone())
        .expect("shape matches data")
        .into_pyarray(py)
}

#[pymodule]
#[pyo3(name = "_crabstack")]
fn crabstack(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(register_py, m)?)?;
    m.add_function(wrap_pyfunction!(transform_py, m)?)?;
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(num_threads, m)?)?;
    m.add("TRANSLATION", crate::transform_kind::TRANSLATION)?;
    m.add("RIGID_BODY", crate::transform_kind::RIGID_BODY)?;
    m.add("SCALED_ROTATION", crate::transform_kind::SCALED_ROTATION)?;
    m.add("AFFINE", crate::transform_kind::AFFINE)?;
    m.add("BILINEAR", crate::transform_kind::BILINEAR)?;
    Ok(())
}
