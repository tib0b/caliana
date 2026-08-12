//! Port of `TurboRegImage`: B-spline coefficient / image / gradient pyramids.
//!
//! The original stores intermediate 2-D arrays as `double` but writes into them
//! through `(float)` casts (`putRow`/`putColumn`). That single-precision
//! rounding is part of the algorithm, so [`f32_round`] is applied in exactly the
//! same spots as the C++ code.
//!
//! Every pass here is separable — a sweep along rows then a sweep along
//! columns — and each line is independent of the others, so both sweeps run in
//! parallel over bands (see [`map_rows`] / [`map_columns_inplace`]). The column
//! sweeps additionally gather [`COL_BLOCK`] columns at a time; walking one
//! column at a time reads eight bytes per cache line and re-streams the whole
//! image once per column.

use crate::f32_round;
use crate::parallel;
use crate::transform_kind::*;

/// One level of the image pyramid.
#[derive(Clone, Debug, Default)]
pub struct ImageStackItem {
    pub half_img: Vec<f64>,
    pub x_gradient: Vec<f64>,
    pub y_gradient: Vec<f64>,
    pub half_width: usize,
    pub half_height: usize,
}

impl ImageStackItem {
    fn new(half_width: usize, half_height: usize, gradient: bool) -> Self {
        ImageStackItem {
            half_img: vec![0.0; half_width * half_height],
            x_gradient: if gradient {
                vec![0.0; half_width * half_height]
            } else {
                Vec::new()
            },
            y_gradient: if gradient {
                vec![0.0; half_width * half_height]
            } else {
                Vec::new()
            },
            half_width,
            half_height,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TurboRegImage {
    /// Pyramid levels, pushed coarser-last (index 0 = first/largest half level,
    /// last = smallest). Consumers pop from the end, matching `std::stack`.
    pub pyramid: Vec<ImageStackItem>,
    pub image: Vec<f64>,
    pub coefficient: Vec<f64>,
    pub x_gradient: Vec<f64>,
    pub y_gradient: Vec<f64>,
    pub width: usize,
    pub height: usize,
    pub pyramid_depth: i32,
    pub transformation: i32,
    pub is_target: bool,
}

impl TurboRegImage {
    pub fn new(
        img: &[f64],
        width: usize,
        height: usize,
        transformation: i32,
        is_target: bool,
    ) -> Self {
        TurboRegImage {
            pyramid: Vec::new(),
            image: img.to_vec(),
            coefficient: Vec::new(),
            x_gradient: Vec::new(),
            y_gradient: Vec::new(),
            width,
            height,
            pyramid_depth: 1,
            transformation,
            is_target,
        }
    }

    pub fn set_pyramid_depth(&mut self, d: i32) {
        self.pyramid_depth = d;
    }

    /// Just the full-resolution cubic B-spline coefficients — what the final
    /// warp needs, without the pyramid and gradients that [`Self::init`] also
    /// builds.
    pub fn coefficients_only(img: &[f64], width: usize, height: usize) -> Vec<f64> {
        get_basic_from_cardinal_2d(img, width, height, 3)
    }

    /// Whether this image's full-resolution coefficients are ever read.
    ///
    /// The optimizer samples the *target* coefficients for the affine family
    /// and the *source* coefficients for bilinear; the other side of the pair
    /// only ever supplies `image` and its gradients. Computing the coefficients
    /// unconditionally, as the C++ does, throws away a full two-pass B-spline
    /// solve on every registration.
    fn needs_coefficient(&self) -> bool {
        if self.transformation == BILINEAR {
            !self.is_target
        } else {
            self.is_target
        }
    }

    pub fn init(&mut self) {
        if self.needs_coefficient() {
            self.coefficient = Self::coefficients_only(&self.image, self.width, self.height);
        }
        match self.transformation {
            TRANSLATION | RIGID_BODY | SCALED_ROTATION | AFFINE => {
                if self.is_target {
                    self.build_coefficient_pyramid();
                } else {
                    self.image_to_xy_gradient_2d();
                    self.build_image_and_gradient_pyramid();
                }
            }
            BILINEAR => {
                if self.is_target {
                    self.build_image_pyramid();
                } else {
                    self.build_coefficient_pyramid();
                }
            }
            _ => {}
        }
    }

    fn image_to_xy_gradient_2d(&mut self) {
        let (w, h) = (self.width, self.height);
        let mut x_gradient = vec![0.0; w * h];
        let mut y_gradient = vec![0.0; w * h];
        let to_gradient = |src: &[f64], dst: &mut [f64], tmp: &mut [f64]| {
            tmp.copy_from_slice(src);
            samples_to_interpolation_coefficient_1d(tmp, 3, IIR_TOLERANCE);
            anti_symmetric_fir_mirror_off_bounds_1d(&GRADIENT_H, tmp, dst);
        };
        map_rows(&self.image, w, &mut x_gradient, w, h, to_gradient);
        map_columns_to(&self.image, h, &mut y_gradient, h, w, to_gradient);
        self.x_gradient = x_gradient;
        self.y_gradient = y_gradient;
    }

    fn build_coefficient_pyramid(&mut self) {
        let (w, h) = (self.width, self.height);
        let mut full_dual = vec![0.0; w * h];
        if self.pyramid_depth > 1 {
            basic_to_cardinal_2d(&self.coefficient, &mut full_dual, w, h, 7);
        }
        let (mut half_w, mut half_h) = (w, h);
        for _ in 1..self.pyramid_depth {
            let (full_w, full_h) = (half_w, half_h);
            half_w /= 2;
            half_h /= 2;
            let half_dual = get_half_dual_2d(&full_dual, full_w, full_h);
            let mut item = ImageStackItem::new(half_w, half_h, false);
            item.half_img = get_basic_from_cardinal_2d(&half_dual, half_w, half_h, 7);
            self.pyramid.push(item);
            full_dual = half_dual;
        }
    }

    fn build_image_and_gradient_pyramid(&mut self) {
        let (w, h) = (self.width, self.height);
        let mut full_dual = vec![0.0; w * h];
        if self.pyramid_depth > 1 {
            full_dual = cardinal_to_dual_2d(&self.image, w, h, 3);
        }
        let (mut half_w, mut half_h) = (w, h);
        for _ in 1..self.pyramid_depth {
            let (full_w, full_h) = (half_w, half_h);
            half_w /= 2;
            half_h /= 2;
            let mut item = ImageStackItem::new(half_w, half_h, true);
            let half_dual = get_half_dual_2d(&full_dual, full_w, full_h);
            item.half_img = get_basic_from_cardinal_2d(&half_dual, half_w, half_h, 7);
            coefficient_to_xy_gradient_2d(
                &item.half_img,
                &mut item.x_gradient,
                &mut item.y_gradient,
                half_w,
                half_h,
            );
            // basicToCardinal2D(halfImg, halfImg, ...) — in place in the original.
            let basic = item.half_img.clone();
            let mut cardinal = vec![0.0; half_w * half_h];
            basic_to_cardinal_2d(&basic, &mut cardinal, half_w, half_h, 3);
            item.half_img = cardinal;
            self.pyramid.push(item);
            full_dual = half_dual;
        }
    }

    fn build_image_pyramid(&mut self) {
        let (w, h) = (self.width, self.height);
        let mut full_dual = vec![0.0; w * h];
        if self.pyramid_depth > 1 {
            full_dual = cardinal_to_dual_2d(&self.image, w, h, 3);
        }
        let (mut half_w, mut half_h) = (w, h);
        for _ in 1..self.pyramid_depth {
            let (full_w, full_h) = (half_w, half_h);
            half_w /= 2;
            half_h /= 2;
            let mut item = ImageStackItem::new(half_w, half_h, true);
            let half_dual = get_half_dual_2d(&full_dual, full_w, full_h);
            item.half_img = dual_to_cardinal_2d(&half_dual, half_w, half_h, 3);
            self.pyramid.push(item);
            full_dual = half_dual;
        }
    }
}

/* ----- separable-pass drivers ----- */

/// Columns handled per gather in [`map_columns_core`].
///
/// The 1-D filters run along a column, so a naive pass reads one `f64` per
/// 64-byte cache line and re-streams the whole image once per column. Gathering
/// eight columns at a time makes every loaded line fully useful.
const COL_BLOCK: usize = 8;

/// Apply `op` to every row: row `y` of `src` in, row `y` of `dst` out. The
/// output is rounded through `f32`, matching the `(float)` casts in `putRow`.
///
/// Rows are independent, so bands own disjoint output rows.
fn map_rows(
    src: &[f64],
    src_w: usize,
    dst: &mut [f64],
    dst_w: usize,
    height: usize,
    op: impl Fn(&[f64], &mut [f64], &mut [f64]) + Sync,
) {
    let nbands = parallel::bands_for_grid(height, dst_w.max(src_w));
    parallel::for_each_band_mut(dst, dst_w, height, nbands, |y0, rows| {
        let mut out = vec![0.0; dst_w];
        let mut scratch = vec![0.0; src_w];
        for (r, dst_row) in rows.chunks_exact_mut(dst_w).enumerate() {
            let base = (y0 + r) * src_w;
            op(&src[base..base + src_w], &mut out, &mut scratch);
            for (d, &v) in dst_row.iter_mut().zip(out.iter()) {
                *d = f32_round(v);
            }
        }
    });
}

/// Apply `op` to every column: column `x` of `src` in, column `x` of `dst` out,
/// `f32`-rounded as `putColumn` does.
///
/// SAFETY: `src` and `dst` must each be valid for `width * src_h` / `width *
/// dst_h` elements. They may alias — a column block is fully gathered before
/// any of it is written back.
unsafe fn map_columns_core(
    src: *const f64,
    src_h: usize,
    dst: *mut f64,
    dst_h: usize,
    width: usize,
    op: impl Fn(&[f64], &mut [f64], &mut [f64]) + Sync,
) {
    let nbands = parallel::bands_for_grid(width, src_h.max(dst_h));
    let src = parallel::SyncPtr::new(src as *mut f64);
    let dst = parallel::SyncPtr::new(dst);
    parallel::for_each_band(nbands, &|b| {
        let (c0, c1) = parallel::band_range(b, nbands, width);
        // Padded so consecutive lanes do not land in the same L1 cache set;
        // an exact `src_h` stride is a power-of-two multiple for typical image
        // heights and makes all eight lanes alias.
        let lane_stride = src_h + 1;
        let mut lanes = vec![0.0f64; COL_BLOCK * lane_stride];
        let mut out = vec![0.0f64; dst_h];
        let mut scratch = vec![0.0f64; src_h];
        let mut x = c0;
        while x < c1 {
            let n = COL_BLOCK.min(c1 - x);
            // Gather: one sequential sweep, `n` columns at a time.
            for y in 0..src_h {
                for l in 0..n {
                    // SAFETY: `x + l < width` and `y < src_h`.
                    lanes[l * lane_stride + y] = unsafe { *src.at(y * width + x + l) };
                }
            }
            for l in 0..n {
                op(
                    &lanes[l * lane_stride..l * lane_stride + src_h],
                    &mut out,
                    &mut scratch,
                );
                // Scatter this lane before reusing `out`.
                for (y, &v) in out.iter().enumerate() {
                    // SAFETY: `x + l < width` and `y < dst_h`; bands own
                    // disjoint column ranges, so no two threads write here.
                    unsafe { *dst.at(y * width + x + l) = f32_round(v) };
                }
            }
            x += n;
        }
    });
}

/// Column pass that rewrites `buf` in place.
fn map_columns_inplace(
    buf: &mut [f64],
    width: usize,
    height: usize,
    op: impl Fn(&[f64], &mut [f64], &mut [f64]) + Sync,
) {
    let p = buf.as_mut_ptr();
    // SAFETY: `buf` is `width * height` long; aliasing src/dst is fine because
    // each column block is gathered before it is written.
    unsafe { map_columns_core(p, height, p, height, width, op) };
}

/// Column pass between two distinct buffers of the same width.
fn map_columns_to(
    src: &[f64],
    src_h: usize,
    dst: &mut [f64],
    dst_h: usize,
    width: usize,
    op: impl Fn(&[f64], &mut [f64], &mut [f64]) + Sync,
) {
    debug_assert_eq!(src.len(), width * src_h);
    debug_assert_eq!(dst.len(), width * dst_h);
    // SAFETY: both slices are valid for the stated extents.
    unsafe { map_columns_core(src.as_ptr(), src_h, dst.as_mut_ptr(), dst_h, width, op) };
}

/* ----- free-function signal-processing primitives ----- */

fn anti_symmetric_fir_mirror_off_bounds_1d(h: &[f64], c: &[f64], s: &mut [f64]) {
    if h.len() >= 2 {
        let n = s.len();
        s[0] = h[1] * (c[1] - c[0]);
        for i in 1..n - 1 {
            s[i] = h[1] * (c[i + 1] - c[i - 1]);
        }
        s[n - 1] = h[1] * (c[c.len() - 1] - c[c.len() - 2]);
    } else {
        s[0] = 0.0;
    }
}

fn symmetric_fir_mirror_off_bounds_1d(h: &[f64], c: &[f64], s: &mut [f64]) {
    let cl = c.len();
    match h.len() {
        2 => {
            if cl >= 2 {
                let n = s.len();
                s[0] = h[0] * c[0] + h[1] * (c[0] + c[1]);
                for i in 1..n - 1 {
                    s[i] = h[0] * c[i] + h[1] * (c[i - 1] + c[i + 1]);
                }
                s[n - 1] = h[0] * c[cl - 1] + h[1] * (c[cl - 2] + c[cl - 1]);
            } else {
                s[0] = (h[0] + 2.0 * h[1]) * c[0];
            }
        }
        4 => {
            if cl >= 6 {
                let n = s.len();
                s[0] = h[0] * c[0]
                    + h[1] * (c[0] + c[1])
                    + h[2] * (c[1] + c[2])
                    + h[3] * (c[2] + c[3]);
                s[1] = h[0] * c[1]
                    + h[1] * (c[0] + c[2])
                    + h[2] * (c[0] + c[3])
                    + h[3] * (c[1] + c[4]);
                s[2] = h[0] * c[2]
                    + h[1] * (c[1] + c[3])
                    + h[2] * (c[0] + c[4])
                    + h[3] * (c[0] + c[5]);
                for i in 3..n - 3 {
                    s[i] = h[0] * c[i]
                        + h[1] * (c[i - 1] + c[i + 1])
                        + h[2] * (c[i - 2] + c[i + 2])
                        + h[3] * (c[i - 3] + c[i + 3]);
                }
                s[n - 3] = h[0] * c[cl - 3]
                    + h[1] * (c[cl - 4] + c[cl - 2])
                    + h[2] * (c[cl - 5] + c[cl - 1])
                    + h[3] * (c[cl - 6] + c[cl - 1]);
                s[n - 2] = h[0] * c[cl - 2]
                    + h[1] * (c[cl - 3] + c[cl - 1])
                    + h[2] * (c[cl - 4] + c[cl - 1])
                    + h[3] * (c[cl - 5] + c[cl - 2]);
                s[n - 1] = h[0] * c[cl - 1]
                    + h[1] * (c[cl - 2] + c[cl - 1])
                    + h[2] * (c[cl - 3] + c[cl - 2])
                    + h[3] * (c[cl - 4] + c[cl - 3]);
            } else {
                match cl {
                    5 => {
                        s[0] = h[0] * c[0]
                            + h[1] * (c[0] + c[1])
                            + h[2] * (c[1] + c[2])
                            + h[3] * (c[2] + c[3]);
                        s[1] = h[0] * c[1]
                            + h[1] * (c[0] + c[2])
                            + h[2] * (c[0] + c[3])
                            + h[3] * (c[1] + c[4]);
                        s[2] = h[0] * c[2] + h[1] * (c[1] + c[3]) + (h[2] + h[3]) * (c[0] + c[4]);
                        s[3] = h[0] * c[3]
                            + h[1] * (c[2] + c[4])
                            + h[2] * (c[1] + c[4])
                            + h[3] * (c[0] + c[3]);
                        s[4] = h[0] * c[4]
                            + h[1] * (c[3] + c[4])
                            + h[2] * (c[2] + c[3])
                            + h[3] * (c[1] + c[2]);
                    }
                    4 => {
                        s[0] = h[0] * c[0]
                            + h[1] * (c[0] + c[1])
                            + h[2] * (c[1] + c[2])
                            + h[3] * (c[2] + c[3]);
                        s[1] = h[0] * c[1]
                            + h[1] * (c[0] + c[2])
                            + h[2] * (c[0] + c[3])
                            + h[3] * (c[1] + c[3]);
                        s[2] = h[0] * c[2]
                            + h[1] * (c[1] + c[3])
                            + h[2] * (c[0] + c[3])
                            + h[3] * (c[0] + c[2]);
                        s[3] = h[0] * c[3]
                            + h[1] * (c[2] + c[3])
                            + h[2] * (c[1] + c[2])
                            + h[3] * (c[0] + c[1]);
                    }
                    3 => {
                        s[0] = h[0] * c[0]
                            + h[1] * (c[0] + c[1])
                            + h[2] * (c[1] + c[2])
                            + 2.0 * h[3] * c[2];
                        s[1] = h[0] * c[1] + (h[1] + h[2]) * (c[0] + c[2]) + 2.0 * h[3] * c[1];
                        s[2] = h[0] * c[2]
                            + h[1] * (c[1] + c[2])
                            + h[2] * (c[0] + c[1])
                            + 2.0 * h[3] * c[0];
                    }
                    2 => {
                        s[0] = (h[0] + h[1] + h[3]) * c[0] + (h[1] + 2.0 * h[2] + h[3]) * c[1];
                        s[1] = (h[0] + h[1] + h[3]) * c[1] + (h[1] + 2.0 * h[2] + h[3]) * c[0];
                    }
                    1 => {
                        s[0] = (h[0] + 2.0 * (h[1] + h[2] + h[3])) * c[0];
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
}

fn reduce_dual_1d(c: &[f64], s: &mut [f64]) {
    let h = [6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
    let cl = c.len();
    let sl = s.len();
    if sl >= 2 {
        s[0] = h[0] * c[0] + h[1] * (c[0] + c[1]) + h[2] * (c[1] + c[2]);
        let mut i = 2;
        for j in 1..sl - 1 {
            s[j] = h[0] * c[i] + h[1] * (c[i - 1] + c[i + 1]) + h[2] * (c[i - 2] + c[i + 2]);
            i += 2;
        }
        if cl == 2 * sl {
            s[sl - 1] =
                h[0] * c[cl - 2] + h[1] * (c[cl - 3] + c[cl - 1]) + h[2] * (c[cl - 4] + c[cl - 1]);
        } else {
            s[sl - 1] =
                h[0] * c[cl - 3] + h[1] * (c[cl - 4] + c[cl - 2]) + h[2] * (c[cl - 5] + c[cl - 1]);
        }
    } else {
        match cl {
            3 => s[0] = h[0] * c[0] + h[1] * (c[0] + c[1]) + h[2] * (c[1] + c[2]),
            2 => s[0] = h[0] * c[0] + h[1] * (c[0] + c[1]) + 2.0 * h[2] * c[1],
            _ => {}
        }
    }
}

fn get_initial_anti_causal_coefficient_mirror_off_bounds(c: &[f64], z: f64, _tol: f64) -> f64 {
    z * c[c.len() - 1] / (z - 1.0)
}

/// Truncation tolerance for the causal-initialisation sum.
///
/// The C++ passes 0.0, which forces the sum to run the full length of the
/// line. Both series in it are bounded by `|z|^n`, so past `ln(eps)/ln(|z|)`
/// terms every contribution is below the f64 rounding of the partial sum —
/// for a 512-pixel line that is ~30 useful iterations out of 512.
const IIR_TOLERANCE: f64 = f64::EPSILON;

fn get_initial_causal_coefficient_mirror_off_bounds(c: &[f64], z: f64, tolerance: f64) -> f64 {
    let mut z1 = z;
    let mut zn = z.powf(c.len() as f64);
    let mut sum = (1.0 + z) * (c[0] + zn * c[c.len() - 1]);
    let mut horizon = c.len() as i32;
    if tolerance > 0.0 {
        horizon = 2 + (tolerance.ln() / (z.abs()).ln()) as i32;
        horizon = horizon.min(c.len() as i32);
    }
    zn *= zn;
    for n in 1..horizon - 1 {
        z1 *= z;
        zn /= z;
        sum += (z1 + zn) * c[n as usize];
    }
    sum / (1.0 - z.powf((2 * c.len()) as f64))
}

fn samples_to_interpolation_coefficient_1d(c: &mut [f64], degree: i32, tolerance: f64) {
    let z: &[f64] = match degree {
        3 => &[3.0_f64.sqrt() - 2.0],
        7 => &[
            -0.5352804307964381655424037816816460718339231523426924148812,
            -0.122554615192326690515272264359357343605486549427295558490763,
            -0.0091486948096082769285930216516478534156925639545994482648003,
        ],
        _ => &[],
    };
    if c.len() == 1 {
        return;
    }
    let mut lambda = 1.0;
    for &zk in z {
        lambda *= (1.0 - zk) * (1.0 - 1.0 / zk);
    }
    for v in c.iter_mut() {
        *v *= lambda;
    }
    for &zk in z {
        c[0] = get_initial_causal_coefficient_mirror_off_bounds(c, zk, tolerance);
        for n in 1..c.len() {
            c[n] += zk * c[n - 1];
        }
        let last = c.len() - 1;
        c[last] = get_initial_anti_causal_coefficient_mirror_off_bounds(c, zk, tolerance);
        for n in (0..last).rev() {
            c[n] = zk * (c[n + 1] - c[n]);
        }
    }
}

/// Filter taps of `coefficientToGradient1D`.
const GRADIENT_H: [f64; 2] = [0.0, 0.5];
/// Filter taps of `coefficientToSamples1D` (the cubic `spline_h`).
const SAMPLES_H: [f64; 2] = [2.0 / 3.0, 1.0 / 6.0];

fn spline_h(degree: i32) -> Vec<f64> {
    match degree {
        3 => vec![2.0 / 3.0, 1.0 / 6.0],
        7 => vec![151.0 / 315.0, 397.0 / 1680.0, 1.0 / 42.0, 1.0 / 5040.0],
        _ => vec![1.0],
    }
}

fn basic_to_cardinal_2d(
    basic: &[f64],
    cardinal: &mut [f64],
    width: usize,
    height: usize,
    degree: i32,
) {
    let h = spline_h(degree);
    let fir = |src: &[f64], dst: &mut [f64], _: &mut [f64]| {
        symmetric_fir_mirror_off_bounds_1d(&h, src, dst)
    };
    map_rows(basic, width, cardinal, width, height, fir);
    map_columns_inplace(cardinal, width, height, fir);
}

fn get_basic_from_cardinal_2d(
    cardinal: &[f64],
    width: usize,
    height: usize,
    degree: i32,
) -> Vec<f64> {
    let mut basic = vec![0.0; width * height];
    let to_coefficients = |src: &[f64], dst: &mut [f64], _: &mut [f64]| {
        dst.copy_from_slice(src);
        samples_to_interpolation_coefficient_1d(dst, degree, IIR_TOLERANCE);
    };
    map_rows(cardinal, width, &mut basic, width, height, to_coefficients);
    map_columns_inplace(&mut basic, width, height, to_coefficients);
    basic
}

fn get_half_dual_2d(full_dual: &[f64], full_width: usize, full_height: usize) -> Vec<f64> {
    let half_width = full_width / 2;
    let half_height = full_height / 2;
    let mut demi_dual = vec![0.0; half_width * full_height];
    let mut half_dual = vec![0.0; half_width * half_height];
    let reduce = |src: &[f64], dst: &mut [f64], _: &mut [f64]| reduce_dual_1d(src, dst);
    map_rows(
        full_dual,
        full_width,
        &mut demi_dual,
        half_width,
        full_height,
        reduce,
    );
    map_columns_to(
        &demi_dual,
        full_height,
        &mut half_dual,
        half_height,
        half_width,
        reduce,
    );
    half_dual
}

fn cardinal_to_dual_2d(cardinal: &[f64], width: usize, height: usize, degree: i32) -> Vec<f64> {
    let basic = get_basic_from_cardinal_2d(cardinal, width, height, degree);
    let mut dual = vec![0.0; width * height];
    basic_to_cardinal_2d(&basic, &mut dual, width, height, 2 * degree + 1);
    dual
}

fn dual_to_cardinal_2d(dual: &[f64], width: usize, height: usize, degree: i32) -> Vec<f64> {
    let basic = get_basic_from_cardinal_2d(dual, width, height, 2 * degree + 1);
    let mut cardinal = vec![0.0; width * height];
    basic_to_cardinal_2d(&basic, &mut cardinal, width, height, degree);
    cardinal
}

fn coefficient_to_xy_gradient_2d(
    basic: &[f64],
    x_gradient: &mut [f64],
    y_gradient: &mut [f64],
    width: usize,
    height: usize,
) {
    let gradient = |src: &[f64], dst: &mut [f64], _: &mut [f64]| {
        anti_symmetric_fir_mirror_off_bounds_1d(&GRADIENT_H, src, dst)
    };
    let samples = |src: &[f64], dst: &mut [f64], _: &mut [f64]| {
        symmetric_fir_mirror_off_bounds_1d(&SAMPLES_H, src, dst)
    };
    // x: differentiate along rows, resample along columns. y: the reverse.
    map_rows(basic, width, x_gradient, width, height, gradient);
    map_rows(basic, width, y_gradient, width, height, samples);
    map_columns_inplace(x_gradient, width, height, samples);
    map_columns_inplace(y_gradient, width, height, gradient);
}
