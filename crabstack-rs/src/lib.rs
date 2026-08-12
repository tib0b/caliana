//! # crabstack
//!
//! A dependency-free Rust implementation of subpixel image registration by the
//! Thévenaz–Ruttimann–Unser pyramid method, originally ported from
//! [pystackreg] / TurboReg.
//!
//! The numeric core (`register` / `transform`) implements the same algorithm as
//! the C++ TurboReg that ships with pystackreg, and the [`StackReg`] type
//! mirrors the Python `StackReg` orchestration layer. [`register_masked`] goes
//! beyond that surface: it restricts the fit to the pixels a [`Mask`] keeps,
//! which pystackreg's C++ could not express.
//!
//! [pystackreg]: https://github.com/glichtner/pystackreg
//!
//! ## Accuracy and determinism
//!
//! The per-pixel loops are vectorised and split across threads, which
//! reassociates floating-point sums, so results are **close to but not
//! bit-identical** with the reference: transformation matrices agree to ~1e-9
//! absolute, and warped images to within one `f32` ulp. See
//! [`set_num_threads`] for controlling parallelism — it does not affect
//! results, because the work split is fixed by image size rather than by thread
//! count.
//!
//! ## Example
//! ```no_run
//! use crabstack::{Image2D, StackReg, Transformation};
//! let reference = Image2D::new(64, 64, vec![0.0; 64 * 64]);
//! let moving = Image2D::new(64, 64, vec![0.0; 64 * 64]);
//! let mut sr = StackReg::new(Transformation::RigidBody);
//! let _matrix = sr.register(&reference, &moving);
//! let _aligned = sr.transform(&moving, None);
//! ```

// The degree-7 B-spline pole constants are copied verbatim from the C++ source
// (more digits than an f64 holds); both toolchains round the literal identically,
// so we keep the exact text for provenance. The manual-stride loop in the mask/
// dual reduction pairs two cursors that don't map onto a single iterator.
#![allow(clippy::excessive_precision, clippy::needless_range_loop)]

mod image;
mod kernel;
mod mask;
mod parallel;
mod point_handler;
mod transform;

pub mod matrix;
pub mod stack;

pub use parallel::{num_threads, set_num_threads, set_simd_enabled, simd_enabled};

#[cfg(feature = "python")]
mod python;

pub use matrix::Mat;
pub use stack::StackReg;

use image::TurboRegImage;
use mask::TurboRegMask;
use transform::TurboRegTransform;

/// Raw transformation codes shared across the port. These are the exact integer
/// values TurboReg uses (and relies on arithmetically, e.g. `code / 2`).
pub mod transform_kind {
    pub const TRANSLATION: i32 = 2;
    pub const RIGID_BODY: i32 = 3;
    pub const SCALED_ROTATION: i32 = 4;
    pub const AFFINE: i32 = 6;
    pub const BILINEAR: i32 = 8;
}
use transform_kind::*;

/// Round a value through `f32`, matching the `(float)` casts in the C++ pyramid
/// construction and warp output.
///
/// This truncation is part of the algorithm, not an artefact: it is what keeps
/// results interchangeable with pystackreg, and it costs essentially nothing
/// (a single convert pair that vectorises). Dropping it would make the port
/// slightly *more* accurate but no faster, and would move every output pixel.
#[inline]
pub(crate) fn f32_round(x: f64) -> f64 {
    x as f32 as f64
}

/// Minimal linear dimension of an image in the multiresolution pyramid.
pub const PYRAMID_MIN_SIZE: i32 = 12;

/// The five supported transformation models.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transformation {
    Translation,
    RigidBody,
    ScaledRotation,
    Affine,
    Bilinear,
}

impl Transformation {
    /// The raw TurboReg integer code.
    pub fn code(self) -> i32 {
        match self {
            Transformation::Translation => TRANSLATION,
            Transformation::RigidBody => RIGID_BODY,
            Transformation::ScaledRotation => SCALED_ROTATION,
            Transformation::Affine => AFFINE,
            Transformation::Bilinear => BILINEAR,
        }
    }

    pub fn from_code(code: i32) -> Option<Self> {
        Some(match code {
            TRANSLATION => Transformation::Translation,
            RIGID_BODY => Transformation::RigidBody,
            SCALED_ROTATION => Transformation::ScaledRotation,
            AFFINE => Transformation::Affine,
            BILINEAR => Transformation::Bilinear,
            _ => return None,
        })
    }
}

/// A row-major 2-D image. `width` is the number of columns, `height` the number
/// of rows; element `(row, col)` lives at `data[row * width + col]`.
///
/// This matches numpy C-order and TurboReg's internal layout (note that
/// pystackreg swaps its `Nx`/`Ny` when calling the native code — a numpy array of
/// shape `(rows, cols)` has `width = cols`, `height = rows`).
#[derive(Clone, Debug, PartialEq)]
pub struct Image2D {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f64>,
}

impl Image2D {
    pub fn new(width: usize, height: usize, data: Vec<f64>) -> Self {
        assert_eq!(
            data.len(),
            width * height,
            "data length must equal width*height"
        );
        Image2D {
            width,
            height,
            data,
        }
    }

    /// Drop the last row and last column, as pystackreg does (`img[:-1, :-1]`)
    /// before registration.
    pub(crate) fn crop_last_row_col(&self) -> Image2D {
        Image2D {
            width: self.width - 1,
            height: self.height - 1,
            data: cropped_last_row_col(&self.data, self.width, self.height),
        }
    }
}

/// Which pixels of an image take part in a registration: `true` contributes to
/// the fit, `false` is ignored entirely. Same row-major layout as [`Image2D`],
/// and must have the same dimensions as the image it masks.
///
/// Masking is an extension over pystackreg, whose C++ core always registered on
/// every pixel. Use it to keep static background, saturated regions, or a
/// neighbouring object from dragging the estimate around.
///
/// ## Leave a margin
///
/// Cut a mask a couple of pixels wider than the thing you are excluding. Both
/// the cubic B-spline interpolation and the pyramid decimation spread a feature
/// roughly two pixels past its own edge, so a mask cut exactly to a bright
/// artefact still leaves contaminated pixels in the fit — enough, in the
/// crate's own tests, to move a translation estimate by a third of a pixel.
/// Two pixels of surround removes the effect entirely.
///
/// Within the pyramid the mask is dilated for you, by one coarse pixel per
/// level: a coarse pixel survives if any fine pixel under it did. That errs
/// towards keeping boundary pixels, so a tight mask stays usable at the
/// coarsest scales rather than vanishing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mask {
    pub width: usize,
    pub height: usize,
    pub data: Vec<bool>,
}

impl Mask {
    pub fn new(width: usize, height: usize, data: Vec<bool>) -> Self {
        assert_eq!(
            data.len(),
            width * height,
            "mask length must equal width*height"
        );
        Mask {
            width,
            height,
            data,
        }
    }

    /// Number of participating pixels. A mask with none of them cannot
    /// constrain a fit — see [`register_masked`].
    pub fn active_count(&self) -> usize {
        self.data.iter().filter(|&&b| b).count()
    }

    /// Intersection with `other`: active only where both are. Used to derive a
    /// reference mask for an averaged reference frame, where a pixel is only
    /// trustworthy if every contributing frame kept it.
    pub fn intersect(&self, other: &Mask) -> Mask {
        assert!(
            self.width == other.width && self.height == other.height,
            "masks must have the same dimensions to intersect"
        );
        Mask {
            width: self.width,
            height: self.height,
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(&a, &b)| a && b)
                .collect(),
        }
    }

    /// Drop the last row and last column, matching [`Image2D::crop_last_row_col`]
    /// so a mask stays aligned with its image through [`StackReg`].
    pub(crate) fn crop_last_row_col(&self) -> Mask {
        Mask {
            width: self.width - 1,
            height: self.height - 1,
            data: cropped_last_row_col(&self.data, self.width, self.height),
        }
    }
}

/// Row-major copy of `data` without its last row and column.
fn cropped_last_row_col<T: Copy>(data: &[T], width: usize, height: usize) -> Vec<T> {
    let w = width - 1;
    let mut out = Vec::with_capacity(w * (height - 1));
    for row in 0..height - 1 {
        let base = row * width;
        out.extend_from_slice(&data[base..base + w]);
    }
    out
}

/// Result of a registration: the short-form TurboReg matrix plus the refined
/// landmark point pairs.
#[derive(Clone, Debug)]
pub struct Registration {
    /// Short-form transformation matrix (`2×1`, `2×3`, or `2×4`).
    pub matrix: Mat,
    /// Reference (target) landmark coordinates.
    pub ref_points: Mat,
    /// Moving (source) landmark coordinates after refinement.
    pub mov_points: Mat,
}

/// Number of pyramid levels for the given source/target dimensions.
/// Direct port of `getPyramidDepth`.
pub fn get_pyramid_depth(mut sw: i32, mut sh: i32, mut tw: i32, mut th: i32) -> i32 {
    let mut depth = 1;
    while 2 * PYRAMID_MIN_SIZE <= sw
        && 2 * PYRAMID_MIN_SIZE <= sh
        && 2 * PYRAMID_MIN_SIZE <= tw
        && 2 * PYRAMID_MIN_SIZE <= th
    {
        sw /= 2;
        sh /= 2;
        tw /= 2;
        th /= 2;
        depth += 1;
    }
    depth
}

fn transformation_from_ncols(ncols: usize) -> i32 {
    match ncols {
        1 => TRANSLATION,
        3 => AFFINE, // RIGID_BODY / SCALED_ROTATION share the same matrix shape
        4 => BILINEAR,
        _ => panic!("Invalid transformation matrix shape"),
    }
}

/// Low-level registration, equivalent to `turboreg._register`. `reference` and
/// `moving` must be the same size and are used as-is (no cropping — the caller,
/// e.g. [`StackReg`], is responsible for the pystackreg crop).
pub fn register(
    reference: &Image2D,
    moving: &Image2D,
    transformation: Transformation,
) -> Registration {
    register_masked(reference, None, moving, None, transformation)
}

/// Registration restricted to the masked-in pixels of either frame.
///
/// A pixel pair contributes to the fit only where both masks keep it: the
/// reference mask is tested at the interpolated source position, the moving
/// mask at the pixel being visited. Passing `None` for a mask keeps every pixel
/// on that side, so `register(a, b, t)` is exactly `register_masked(a, None, b,
/// None, t)` — and costs the same, since an absent mask allocates nothing and
/// leaves the per-pixel test compiled out of the loop.
///
/// Each mask must match its image's dimensions. A mask that keeps too little —
/// in the limit, nothing — leaves the optimiser underdetermined and can yield a
/// non-finite matrix; check [`Mask::active_count`] if the mask comes from a
/// segmentation that may come back empty.
pub fn register_masked(
    reference: &Image2D,
    reference_mask: Option<&Mask>,
    moving: &Image2D,
    moving_mask: Option<&Mask>,
    transformation: Transformation,
) -> Registration {
    assert_eq!(reference.width, moving.width);
    assert_eq!(reference.height, moving.height);
    let code = transformation.code();
    let (w, h) = (reference.width, reference.height);
    for (mask, which) in [(reference_mask, "reference"), (moving_mask, "moving")] {
        if let Some(m) = mask {
            assert!(
                m.width == w && m.height == h,
                "{which} mask is {}x{}, but its image is {w}x{h}",
                m.width,
                m.height
            );
        }
    }

    let mut ref_img = TurboRegImage::new(&reference.data, w, h, code, true);
    let mut mov_img = TurboRegImage::new(&moving.data, w, h, code, false);

    let ref_pts = point_handler::points_by_transformation(w, h, code);
    let mov_pts = point_handler::points_by_transformation(w, h, code);

    let mut ref_msk = reference_mask.map(TurboRegMask::new);
    let mut mov_msk = moving_mask.map(TurboRegMask::new);

    let depth = get_pyramid_depth(w as i32, h as i32, w as i32, h as i32);
    ref_img.set_pyramid_depth(depth);
    mov_img.set_pyramid_depth(depth);
    ref_img.init();
    mov_img.init();
    for msk in [ref_msk.as_mut(), mov_msk.as_mut()].into_iter().flatten() {
        msk.set_pyramid_depth(depth);
        msk.init();
    }

    let mut tform = TurboRegTransform::new(code, false);
    tform.source_point = mov_pts;
    tform.target_point = ref_pts;

    // Passed by `&mut` so the optimizer can move each pyramid level out as it
    // consumes it rather than cloning the whole set up front.
    tform.do_registration(
        &mut mov_img,
        mov_msk.as_mut(),
        &mut ref_img,
        ref_msk.as_mut(),
    );

    let matrix = tform.transformation_matrix();
    Registration {
        matrix,
        ref_points: tform.target_point.clone(),
        mov_points: tform.source_point.clone(),
    }
}

/// Low-level warp, equivalent to `turboreg._transform`. `matrix` is a short-form
/// matrix (`2×1`, `2×3`, or `2×4`).
pub fn transform(moving: &Image2D, matrix: &Mat) -> Image2D {
    assert_eq!(matrix.rows, 2, "transformation matrix must have 2 rows");
    let code = transformation_from_ncols(matrix.cols);
    let (w, h) = (moving.width, moving.height);

    // The warp only ever reads the full-resolution B-spline coefficients, so
    // the multiresolution pyramid and the xy-gradients that `init` would build
    // are pure waste here.
    let coefficient = TurboRegImage::coefficients_only(&moving.data, w, h);

    let mut tform = TurboRegTransform::new(code, false);
    let out = tform.do_final_transform_matrix(coefficient, w, h, matrix);
    Image2D {
        width: w,
        height: h,
        data: out,
    }
}
