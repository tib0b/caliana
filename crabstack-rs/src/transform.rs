//! Port of `TurboRegTransform`: the Marquardt-Levenberg optimizer plus the
//! mean-squares / gradient / Hessian evaluators and the final image warp.
//!
//! The optimizer control flow follows the C++ original step for step. The
//! per-pixel evaluators do not: instead of walking one pixel at a time through
//! object scratch fields, each evaluator splits its output rows into bands
//! (see [`crate::parallel`]), evaluates them independently — in parallel and
//! with AVX2/FMA where available — and reduces the partials in band order.
//!
//! That reassociates the floating-point sums, so results are close to but not
//! bit-identical with the reference. The band split is fixed by problem size
//! alone, so the output is still deterministic and thread-count independent.

use crate::f32_round;
use crate::image::TurboRegImage;
use crate::kernel::{
    base_index, round_msk, spline_weights, spline_weights_d, split, Accum, Deriv, Sampler, MAX_T,
    MS, MS_G, MS_GH,
};
use crate::mask::TurboRegMask;
use crate::matrix::Mat;
use crate::parallel;
use crate::transform_kind::*;

const FEW_ITERATIONS: i32 = 5;
const FIRST_LAMBDA: f64 = 1.0;
const LAMBDA_MAGSTEP: f64 = 4.0;
const MANY_ITERATIONS: i32 = 10;
const PIXEL_HIGH_PRECISION: f64 = 0.001;
const PIXEL_LOW_PRECISION: f64 = 0.1;
const ITERATION_PROGRESSION: i32 = 2;

#[derive(Default)]
pub struct TurboRegTransform {
    accelerated: bool,
    pixel_precision: f64,
    target_jacobian: f64,
    pub source_point: Mat,
    pub target_point: Mat,
    in_img: Vec<f64>,
    /// Mask for the input grid, empty when the caller supplied none.
    in_msk: Vec<bool>,
    out_img: Vec<f64>,
    /// Mask for the output grid, empty when the caller supplied none.
    out_msk: Vec<bool>,
    x_gradient: Vec<f64>,
    y_gradient: Vec<f64>,
    in_nx: i32,
    in_ny: i32,
    iteration_power: i32,
    max_iterations: i32,
    out_nx: i32,
    out_ny: i32,
    pyramid_depth: i32,
    transformation: i32,
    /// Set once per pyramid level: true when no installed mask excludes
    /// anything, so every in-range pixel takes part. Always true when the
    /// caller supplied no masks, which lets the inner loop drop two loads and a
    /// test on the hot default path.
    all_active: bool,
}

impl TurboRegTransform {
    pub fn new(transformation: i32, accelerated: bool) -> Self {
        let (pixel_precision, max_iterations) = if accelerated {
            (PIXEL_LOW_PRECISION, FEW_ITERATIONS)
        } else {
            (PIXEL_HIGH_PRECISION, MANY_ITERATIONS)
        };
        TurboRegTransform {
            accelerated,
            transformation,
            pixel_precision,
            max_iterations,
            ..Default::default()
        }
    }

    fn t(&self) -> usize {
        self.transformation as usize
    }

    fn ctx(&self) -> EvalCtx<'_> {
        EvalCtx {
            in_img: &self.in_img,
            in_msk: &self.in_msk,
            out_img: &self.out_img,
            out_msk: &self.out_msk,
            x_gradient: &self.x_gradient,
            y_gradient: &self.y_gradient,
            in_nx: self.in_nx,
            in_ny: self.in_ny,
            out_nx: self.out_nx,
            all_active: self.all_active,
        }
    }

    /// Recompute [`Self::all_active`] for the buffers currently installed.
    /// O(pixels) once per pyramid level, against O(pixels × iterations) saved.
    /// An absent mask is an empty slice, which excludes nothing.
    fn refresh_mask_flag(&mut self) {
        let keeps_all = |m: &[bool]| m.iter().all(|&v| v);
        self.all_active = keeps_all(&self.in_msk) && keeps_all(&self.out_msk);
    }

    /* ---------- band driver ---------- */

    /// Evaluate `f` over every row band and reduce the partials in band order.
    fn accumulate<F>(&self, f: F) -> Accum
    where
        F: Fn(usize, usize) -> Accum + Sync,
    {
        let rows = self.out_ny as usize;
        let nbands = parallel::bands_for_grid(rows, self.out_nx as usize);
        if nbands <= 1 {
            return f(0, rows);
        }
        let parts = parallel::map_bands(nbands, |b| {
            let (v0, v1) = parallel::band_range(b, nbands, rows);
            f(v0, v1)
        });
        Accum::reduce(&parts)
    }

    /* ---------- translation mean squares ---------- */

    fn translation_eval<const MODE: u8>(&self, m: &Mat) -> Accum {
        let ctx = self.ctx();
        self.accumulate(|v0, v1| translation_band_dispatch::<MODE>(&ctx, m, v0, v1))
    }

    fn get_translation_mean_squares(&self, m: &Mat) -> f64 {
        let acc = self.translation_eval::<MS>(m);
        acc.ms / acc.area as f64
    }

    fn get_translation_mean_squares_g(&self, m: &Mat, gradient: &mut [f64]) -> f64 {
        let acc = self.translation_eval::<MS_G>(m);
        store_g(&acc, self.t(), gradient);
        acc.ms / acc.area as f64
    }

    fn get_translation_mean_squares_gh(
        &self,
        m: &Mat,
        hessian: &mut Mat,
        gradient: &mut [f64],
    ) -> f64 {
        let acc = self.translation_eval::<MS_GH>(m);
        store_gh(&acc, self.t(), hessian, gradient);
        acc.ms / acc.area as f64
    }

    /* ---------- rigid-body mean squares ---------- */

    fn rigid_body_eval<const MODE: u8>(&self, m: &Mat) -> Accum {
        let ctx = self.ctx();
        let d = RigidBodyDeriv;
        self.accumulate(|v0, v1| {
            affine_family_band_dispatch::<RigidBodyDeriv, MODE, 3>(&ctx, m, &d, v0, v1)
        })
    }

    fn get_rigid_body_mean_squares(&self, m: &Mat) -> f64 {
        let acc = self.rigid_body_eval::<MS>(m);
        acc.ms / acc.area as f64
    }

    fn get_rigid_body_mean_squares_g(&self, m: &Mat, gradient: &mut [f64]) -> f64 {
        let acc = self.rigid_body_eval::<MS_G>(m);
        store_g(&acc, self.t(), gradient);
        acc.ms / acc.area as f64
    }

    fn get_rigid_body_mean_squares_gh(
        &self,
        m: &Mat,
        hessian: &mut Mat,
        gradient: &mut [f64],
    ) -> f64 {
        let acc = self.rigid_body_eval::<MS_GH>(m);
        store_gh(&acc, self.t(), hessian, gradient);
        acc.ms / acc.area as f64
    }

    /* ---------- scaled-rotation mean squares ---------- */

    fn scaled_rotation_eval<const MODE: u8>(&self, sp: &Mat, m: &Mat) -> (Accum, f64) {
        let ctx = self.ctx();
        let sr = ScaledRotConstants::new(sp);
        let acc = self.accumulate(|v0, v1| {
            affine_family_band_dispatch::<ScaledRotConstants, MODE, 4>(&ctx, m, &sr, v0, v1)
        });
        (acc, sr.uv2)
    }

    fn get_scaled_rotation_mean_squares(&self, source_point: &Mat, m: &Mat) -> f64 {
        let (acc, uv2) = self.scaled_rotation_eval::<MS>(source_point, m);
        acc.ms / (acc.area as f64 * uv2 / self.target_jacobian)
    }

    fn get_scaled_rotation_mean_squares_g(
        &self,
        source_point: &Mat,
        m: &Mat,
        gradient: &mut [f64],
    ) -> f64 {
        let (acc, uv2) = self.scaled_rotation_eval::<MS_G>(source_point, m);
        store_g(&acc, self.t(), gradient);
        acc.ms / (acc.area as f64 * uv2 / self.target_jacobian)
    }

    fn get_scaled_rotation_mean_squares_gh(
        &self,
        source_point: &Mat,
        m: &Mat,
        hessian: &mut Mat,
        gradient: &mut [f64],
    ) -> f64 {
        let (acc, uv2) = self.scaled_rotation_eval::<MS_GH>(source_point, m);
        store_gh(&acc, self.t(), hessian, gradient);
        acc.ms / (acc.area as f64 * uv2 / self.target_jacobian)
    }

    /* ---------- affine mean squares ---------- */

    fn affine_eval<const MODE: u8>(&self, sp: &Mat, m: &Mat) -> (Accum, f64) {
        let ctx = self.ctx();
        let af = AffineConstants::new(sp);
        let acc = self.accumulate(|v0, v1| {
            affine_family_band_dispatch::<AffineConstants, MODE, 6>(&ctx, m, &af, v0, v1)
        });
        (acc, af.det)
    }

    fn get_affine_mean_squares(&self, source_point: &Mat, m: &Mat) -> f64 {
        let (acc, det) = self.affine_eval::<MS>(source_point, m);
        acc.ms / (acc.area as f64 * (det / self.target_jacobian).abs())
    }

    fn get_affine_mean_squares_g(&self, source_point: &Mat, m: &Mat, gradient: &mut [f64]) -> f64 {
        let (acc, det) = self.affine_eval::<MS_G>(source_point, m);
        store_g(&acc, self.t(), gradient);
        acc.ms / (acc.area as f64 * (det / self.target_jacobian).abs())
    }

    fn get_affine_mean_squares_gh(
        &self,
        source_point: &Mat,
        m: &Mat,
        hessian: &mut Mat,
        gradient: &mut [f64],
    ) -> f64 {
        let (acc, det) = self.affine_eval::<MS_GH>(source_point, m);
        store_gh(&acc, self.t(), hessian, gradient);
        acc.ms / (acc.area as f64 * (det / self.target_jacobian).abs())
    }

    /* ---------- bilinear mean squares ---------- */

    fn get_bilinear_mean_squares(&self, m: &Mat) -> f64 {
        let ctx = self.ctx();
        let acc = self.accumulate(|v0, v1| {
            bilinear_band_dispatch::<MS>(&ctx, m, &BilinearConstants::ZERO, v0, v1)
        });
        acc.ms / acc.area as f64
    }

    fn get_bilinear_mean_squares_gh(
        &self,
        m: &Mat,
        hessian: &mut Mat,
        gradient: &mut [f64],
    ) -> f64 {
        let ctx = self.ctx();
        let bc = BilinearConstants::new(&self.target_point);
        let acc = self.accumulate(|v0, v1| bilinear_band_dispatch::<MS_GH>(&ctx, m, &bc, v0, v1));
        store_gh(&acc, self.t(), hessian, gradient);
        acc.ms / acc.area as f64
    }

    /* ---------- final image warps ---------- */

    fn translation_transform(&mut self, m: &Mat) {
        let (dx0, dy0) = (m[(0, 0)], m[(1, 0)]);
        let wx = spline_weights(dx0 - dx0.floor());
        let wy = spline_weights(dy0 - dy0.floor());
        let (in_nx, in_ny) = (self.in_nx, self.in_ny);
        let nx = self.out_nx as usize;
        let accelerated = self.accelerated;
        let in_img = std::mem::take(&mut self.in_img);
        let mut out = std::mem::take(&mut self.out_img);

        self.warp_bands(&mut out, nx, |v0, rows| {
            let s = Sampler::new(&in_img, in_nx, in_ny);
            for (r, row) in rows.chunks_exact_mut(nx).enumerate() {
                let y = dy0 + (v0 + r) as f64;
                let y_msk = round_msk(y);
                if y_msk < 0 || y_msk >= in_ny {
                    row.fill(0.0);
                    continue;
                }
                let y_off = y_msk * in_nx;
                let yi = base_index(y);
                for (u, o) in row.iter_mut().enumerate() {
                    let x = dx0 + u as f64;
                    let x_msk = round_msk(x);
                    *o = if x_msk < 0 || x_msk >= in_nx {
                        0.0
                    } else if accelerated {
                        in_img[(x_msk + y_off) as usize]
                    } else {
                        f32_round(s.interpolate(base_index(x), &wx, yi, &wy))
                    };
                }
            }
        });

        self.in_img = in_img;
        self.out_img = out;
    }

    /// Shared body of the affine and bilinear warps. `m03`/`m13` are the
    /// bilinear cross terms; they are zero for the affine family.
    fn general_transform(&mut self, m: &Mat, bilinear: bool) {
        let (m00, m01, m02) = (m[(0, 0)], m[(0, 1)], m[(0, 2)]);
        let (m10, m11, m12) = (m[(1, 0)], m[(1, 1)], m[(1, 2)]);
        let (m03, m13) = if bilinear {
            (m[(0, 3)], m[(1, 3)])
        } else {
            (0.0, 0.0)
        };
        let (in_nx, in_ny) = (self.in_nx, self.in_ny);
        let nx = self.out_nx as usize;
        let accelerated = self.accelerated;
        let in_img = std::mem::take(&mut self.in_img);
        let mut out = std::mem::take(&mut self.out_img);

        self.warp_bands(&mut out, nx, |v0, rows| {
            let s = Sampler::new(&in_img, in_nx, in_ny);
            for (r, row) in rows.chunks_exact_mut(nx).enumerate() {
                let v = (v0 + r) as f64;
                // Recomputed per row rather than accumulated across rows, so a
                // band's result does not depend on where the band starts.
                let mut x0 = m00 + v * m02;
                let mut y0 = m10 + v * m12;
                let xstep = m01 + v * m03;
                let ystep = m11 + v * m13;
                for o in row.iter_mut() {
                    let (x, y) = (x0, y0);
                    x0 += xstep;
                    y0 += ystep;
                    let x_msk = round_msk(x);
                    let y_msk = round_msk(y);
                    *o = if x_msk < 0 || x_msk >= in_nx || y_msk < 0 || y_msk >= in_ny {
                        0.0
                    } else if accelerated {
                        in_img[(x_msk + y_msk * in_nx) as usize]
                    } else {
                        let (xi, fx) = split(x);
                        let (yi, fy) = split(y);
                        f32_round(s.interpolate(xi, &spline_weights(fx), yi, &spline_weights(fy)))
                    };
                }
            }
        });

        self.in_img = in_img;
        self.out_img = out;
    }

    fn warp_bands(&self, out: &mut [f64], nx: usize, f: impl Fn(usize, &mut [f64]) + Sync) {
        let rows = self.out_ny as usize;
        let nbands = parallel::bands_for_grid(rows, nx);
        parallel::for_each_band_mut(out, nx, rows, nbands, f);
    }

    /* ---------- transformation matrix from landmark pairs ---------- */

    pub fn get_transformation_matrix(&self, from: &Mat, to: &Mat) -> Mat {
        match self.transformation {
            TRANSLATION => {
                let mut m = Mat::new(2, 1);
                m[(0, 0)] = to[(0, 0)] - from[(0, 0)];
                m[(1, 0)] = to[(0, 1)] - from[(0, 1)];
                m
            }
            RIGID_BODY => {
                let angle = (from[(2, 0)] - from[(1, 0)]).atan2(from[(2, 1)] - from[(1, 1)])
                    - (to[(2, 0)] - to[(1, 0)]).atan2(to[(2, 1)] - to[(1, 1)]);
                let c = angle.cos();
                let s = angle.sin();
                let mut m = Mat::new(2, 3);
                m[(0, 0)] = to[(0, 0)] - c * from[(0, 0)] + s * from[(0, 1)];
                m[(0, 1)] = c;
                m[(0, 2)] = -s;
                m[(1, 0)] = to[(0, 1)] - s * from[(0, 0)] - c * from[(0, 1)];
                m[(1, 1)] = s;
                m[(1, 2)] = c;
                m
            }
            SCALED_ROTATION => {
                let mut m = Mat::new(2, 3);
                let mut a = Mat::new(3, 3);
                let mut v = [0.0; 3];
                a[(0, 0)] = 1.0;
                a[(0, 1)] = from[(0, 0)];
                a[(0, 2)] = from[(0, 1)];
                a[(1, 0)] = 1.0;
                a[(1, 1)] = from[(1, 0)];
                a[(1, 2)] = from[(1, 1)];
                a[(2, 0)] = 1.0;
                a[(2, 1)] = from[(0, 1)] - from[(1, 1)] + from[(1, 0)];
                a[(2, 2)] = from[(1, 0)] + from[(1, 1)] - from[(0, 0)];
                invert_gauss(&mut a);
                v[0] = to[(0, 0)];
                v[1] = to[(1, 0)];
                v[2] = to[(0, 1)] - to[(1, 1)] + to[(1, 0)];
                for i in 0..3 {
                    m[(0, i)] = 0.0;
                    for j in 0..3 {
                        m[(0, i)] += a[(i, j)] * v[j];
                    }
                }
                v[0] = to[(0, 1)];
                v[1] = to[(1, 1)];
                v[2] = to[(1, 0)] + to[(1, 1)] - to[(0, 0)];
                for i in 0..3 {
                    m[(1, i)] = 0.0;
                    for j in 0..3 {
                        m[(1, i)] += a[(i, j)] * v[j];
                    }
                }
                m
            }
            AFFINE => {
                let mut m = Mat::new(2, 3);
                let mut a = Mat::new(3, 3);
                let mut v = [0.0; 3];
                a[(0, 0)] = 1.0;
                a[(0, 1)] = from[(0, 0)];
                a[(0, 2)] = from[(0, 1)];
                a[(1, 0)] = 1.0;
                a[(1, 1)] = from[(1, 0)];
                a[(1, 2)] = from[(1, 1)];
                a[(2, 0)] = 1.0;
                a[(2, 1)] = from[(2, 0)];
                a[(2, 2)] = from[(2, 1)];
                invert_gauss(&mut a);
                v[0] = to[(0, 0)];
                v[1] = to[(1, 0)];
                v[2] = to[(2, 0)];
                for i in 0..3 {
                    m[(0, i)] = 0.0;
                    for j in 0..3 {
                        m[(0, i)] += a[(i, j)] * v[j];
                    }
                }
                v[0] = to[(0, 1)];
                v[1] = to[(1, 1)];
                v[2] = to[(2, 1)];
                for i in 0..3 {
                    m[(1, i)] = 0.0;
                    for j in 0..3 {
                        m[(1, i)] += a[(i, j)] * v[j];
                    }
                }
                m
            }
            BILINEAR => {
                let mut m = Mat::new(2, 4);
                let mut a = Mat::new(4, 4);
                let mut v = [0.0; 4];
                for r in 0..4 {
                    a[(r, 0)] = 1.0;
                    a[(r, 1)] = from[(r, 0)];
                    a[(r, 2)] = from[(r, 1)];
                    a[(r, 3)] = from[(r, 0)] * from[(r, 1)];
                }
                invert_gauss(&mut a);
                for (r, vr) in v.iter_mut().enumerate() {
                    *vr = to[(r, 0)];
                }
                for i in 0..4 {
                    m[(0, i)] = 0.0;
                    for j in 0..4 {
                        m[(0, i)] += a[(i, j)] * v[j];
                    }
                }
                for (r, vr) in v.iter_mut().enumerate() {
                    *vr = to[(r, 1)];
                }
                for i in 0..4 {
                    m[(1, i)] = 0.0;
                    for j in 0..4 {
                        m[(1, i)] += a[(i, j)] * v[j];
                    }
                }
                m
            }
            _ => Mat::new(0, 0),
        }
    }

    pub fn transformation_matrix(&self) -> Mat {
        self.get_transformation_matrix(&self.target_point, &self.source_point)
    }

    /* ---------- optimizers ---------- */

    fn inverse_marquardt_levenberg_optimization(&mut self) {
        let t = self.t();
        let half = t / 2;
        let mut attempt = Mat::new(half, 2);
        let mut hessian = Mat::new(t, t);
        let mut pseudo_hessian = Mat::new(t, t);
        let mut gradient = vec![0.0; t];
        let mut m = self.get_transformation_matrix(&self.source_point, &self.target_point);
        let mut update;
        let mut lambda = FIRST_LAMBDA;
        let mut iteration = 0;

        let sp = self.source_point.clone();
        let mut best_mean_squares = match self.transformation {
            TRANSLATION => self.get_translation_mean_squares_gh(&m, &mut hessian, &mut gradient),
            SCALED_ROTATION => {
                self.get_scaled_rotation_mean_squares_gh(&sp, &m, &mut hessian, &mut gradient)
            }
            AFFINE => self.get_affine_mean_squares_gh(&sp, &m, &mut hessian, &mut gradient),
            _ => 0.0,
        };
        iteration += 1;
        loop {
            for k in 0..t {
                pseudo_hessian[(k, k)] = (1.0 + lambda) * hessian[(k, k)];
            }
            invert_gauss(&mut pseudo_hessian);
            update = matrix_multiply(&pseudo_hessian, &gradient);
            let mut displacement = 0.0;
            for k in 0..half {
                attempt[(k, 0)] = self.source_point[(k, 0)] - update[2 * k];
                attempt[(k, 1)] = self.source_point[(k, 1)] - update[2 * k + 1];
                displacement +=
                    (update[2 * k] * update[2 * k] + update[2 * k + 1] * update[2 * k + 1]).sqrt();
            }
            displacement /= 0.5 * t as f64;
            m = self.get_transformation_matrix(&attempt, &self.target_point);
            let mean_squares = match self.transformation {
                TRANSLATION => {
                    if self.accelerated {
                        self.get_translation_mean_squares_g(&m, &mut gradient)
                    } else {
                        self.get_translation_mean_squares_gh(&m, &mut hessian, &mut gradient)
                    }
                }
                SCALED_ROTATION => {
                    if self.accelerated {
                        self.get_scaled_rotation_mean_squares_g(&attempt, &m, &mut gradient)
                    } else {
                        self.get_scaled_rotation_mean_squares_gh(
                            &attempt,
                            &m,
                            &mut hessian,
                            &mut gradient,
                        )
                    }
                }
                AFFINE => {
                    if self.accelerated {
                        self.get_affine_mean_squares_g(&attempt, &m, &mut gradient)
                    } else {
                        self.get_affine_mean_squares_gh(&attempt, &m, &mut hessian, &mut gradient)
                    }
                }
                _ => 0.0,
            };
            iteration += 1;
            if mean_squares < best_mean_squares {
                best_mean_squares = mean_squares;
                for k in 0..half {
                    self.source_point[(k, 0)] = attempt[(k, 0)];
                    self.source_point[(k, 1)] = attempt[(k, 1)];
                }
                lambda /= LAMBDA_MAGSTEP;
            } else {
                lambda *= LAMBDA_MAGSTEP;
            }
            if !(iteration < (self.max_iterations * self.iteration_power - 1)
                && self.pixel_precision <= displacement)
            {
                break;
            }
        }
        invert_gauss(&mut hessian);
        update = matrix_multiply(&hessian, &gradient);
        for k in 0..half {
            attempt[(k, 0)] = self.source_point[(k, 0)] - update[2 * k];
            attempt[(k, 1)] = self.source_point[(k, 1)] - update[2 * k + 1];
        }
        m = self.get_transformation_matrix(&attempt, &self.target_point);
        let mean_squares = match self.transformation {
            TRANSLATION => self.get_translation_mean_squares(&m),
            SCALED_ROTATION => self.get_scaled_rotation_mean_squares(&attempt, &m),
            AFFINE => self.get_affine_mean_squares(&attempt, &m),
            _ => 0.0,
        };
        if mean_squares < best_mean_squares {
            for k in 0..half {
                self.source_point[(k, 0)] = attempt[(k, 0)];
                self.source_point[(k, 1)] = attempt[(k, 1)];
            }
        }
    }

    fn inverse_marquardt_levenberg_rigid_body_optimization(&mut self) {
        let t = self.t();
        let mut attempt = Mat::new(2, 3);
        let mut hessian = Mat::new(t, t);
        let mut pseudo_hessian = Mat::new(t, t);
        let mut gradient = vec![0.0; t];
        let mut m = self.get_transformation_matrix(&self.target_point, &self.source_point);
        let mut update;
        let mut lambda = FIRST_LAMBDA;
        let mut iteration = 0;
        for k in 0..t {
            self.source_point[(k, 0)] = m[(0, 0)]
                + self.target_point[(k, 0)] * m[(0, 1)]
                + self.target_point[(k, 1)] * m[(0, 2)];
            self.source_point[(k, 1)] = m[(1, 0)]
                + self.target_point[(k, 0)] * m[(1, 1)]
                + self.target_point[(k, 1)] * m[(1, 2)];
        }
        m = self.get_transformation_matrix(&self.source_point, &self.target_point);
        let mut best_mean_squares =
            self.get_rigid_body_mean_squares_gh(&m, &mut hessian, &mut gradient);
        iteration += 1;
        loop {
            for k in 0..t {
                pseudo_hessian[(k, k)] = (1.0 + lambda) * hessian[(k, k)];
            }
            invert_gauss(&mut pseudo_hessian);
            update = matrix_multiply(&pseudo_hessian, &gradient);
            let angle = m[(0, 2)].atan2(m[(0, 1)]) - update[0];
            attempt[(0, 1)] = angle.cos();
            attempt[(0, 2)] = angle.sin();
            attempt[(1, 1)] = -attempt[(0, 2)];
            attempt[(1, 2)] = attempt[(0, 1)];
            let c = update[0].cos();
            let s = update[0].sin();
            attempt[(0, 0)] = (m[(0, 0)] + update[1]) * c - (m[(1, 0)] + update[2]) * s;
            attempt[(1, 0)] = (m[(0, 0)] + update[1]) * s + (m[(1, 0)] + update[2]) * c;
            let displacement = (update[1] * update[1] + update[2] * update[2]).sqrt()
                + 0.25
                    * ((self.in_nx * self.in_nx) as f64 + (self.in_ny * self.in_ny) as f64).sqrt()
                    * update[0].abs();
            let mean_squares = if self.accelerated {
                self.get_rigid_body_mean_squares_g(&attempt, &mut gradient)
            } else {
                self.get_rigid_body_mean_squares_gh(&attempt, &mut hessian, &mut gradient)
            };
            iteration += 1;
            if mean_squares < best_mean_squares {
                best_mean_squares = mean_squares;
                for i in 0..2 {
                    for j in 0..3 {
                        m[(i, j)] = attempt[(i, j)];
                    }
                }
                lambda /= LAMBDA_MAGSTEP;
            } else {
                lambda *= LAMBDA_MAGSTEP;
            }
            if !(iteration < (self.max_iterations * self.iteration_power - 1)
                && self.pixel_precision <= displacement)
            {
                break;
            }
        }
        invert_gauss(&mut hessian);
        update = matrix_multiply(&hessian, &gradient);
        let angle = m[(0, 2)].atan2(m[(0, 1)]) - update[0];
        attempt[(0, 1)] = angle.cos();
        attempt[(0, 2)] = angle.sin();
        attempt[(1, 1)] = -attempt[(0, 2)];
        attempt[(1, 2)] = attempt[(0, 1)];
        let c = update[0].cos();
        let s = update[0].sin();
        attempt[(0, 0)] = (m[(0, 0)] + update[1]) * c - (m[(1, 0)] + update[2]) * s;
        attempt[(1, 0)] = (m[(0, 0)] + update[1]) * s + (m[(1, 0)] + update[2]) * c;
        let mean_squares = self.get_rigid_body_mean_squares(&attempt);
        if mean_squares < best_mean_squares {
            for i in 0..2 {
                for j in 0..3 {
                    m[(i, j)] = attempt[(i, j)];
                }
            }
        }
        for k in 0..t {
            self.source_point[(k, 0)] = (self.target_point[(k, 0)] - m[(0, 0)]) * m[(0, 1)]
                + (self.target_point[(k, 1)] - m[(1, 0)]) * m[(1, 1)];
            self.source_point[(k, 1)] = (self.target_point[(k, 0)] - m[(0, 0)]) * m[(0, 2)]
                + (self.target_point[(k, 1)] - m[(1, 0)]) * m[(1, 2)];
        }
    }

    fn marquardt_levenberg_optimization(&mut self) {
        let t = self.t();
        let half = t / 2;
        let mut attempt = Mat::new(half, 2);
        let mut hessian = Mat::new(t, t);
        let mut pseudo_hessian = Mat::new(t, t);
        let mut gradient = vec![0.0; t];
        let mut m = self.get_transformation_matrix(&self.target_point, &self.source_point);
        let mut update;
        let mut lambda = FIRST_LAMBDA;
        let mut iteration = 0;
        let mut best_mean_squares =
            self.get_bilinear_mean_squares_gh(&m, &mut hessian, &mut gradient);
        iteration += 1;
        loop {
            for k in 0..t {
                pseudo_hessian[(k, k)] = (1.0 + lambda) * hessian[(k, k)];
            }
            invert_gauss(&mut pseudo_hessian);
            update = matrix_multiply(&pseudo_hessian, &gradient);
            let mut displacement = 0.0;
            for k in 0..half {
                attempt[(k, 0)] = self.source_point[(k, 0)] - update[2 * k];
                attempt[(k, 1)] = self.source_point[(k, 1)] - update[2 * k + 1];
                displacement +=
                    (update[2 * k] * update[2 * k] + update[2 * k + 1] * update[2 * k + 1]).sqrt();
            }
            displacement /= 0.5 * t as f64;
            m = self.get_transformation_matrix(&self.target_point, &attempt);
            let mean_squares = self.get_bilinear_mean_squares_gh(&m, &mut hessian, &mut gradient);
            iteration += 1;
            if mean_squares < best_mean_squares {
                best_mean_squares = mean_squares;
                for k in 0..half {
                    self.source_point[(k, 0)] = attempt[(k, 0)];
                    self.source_point[(k, 1)] = attempt[(k, 1)];
                }
                lambda /= LAMBDA_MAGSTEP;
            } else {
                lambda *= LAMBDA_MAGSTEP;
            }
            if !(iteration < (self.max_iterations * self.iteration_power - 1)
                && self.pixel_precision <= displacement)
            {
                break;
            }
        }
        invert_gauss(&mut hessian);
        update = matrix_multiply(&hessian, &gradient);
        for k in 0..half {
            attempt[(k, 0)] = self.source_point[(k, 0)] - update[2 * k];
            attempt[(k, 1)] = self.source_point[(k, 1)] - update[2 * k + 1];
        }
        m = self.get_transformation_matrix(&self.target_point, &attempt);
        let mean_squares = self.get_bilinear_mean_squares(&m);
        if mean_squares < best_mean_squares {
            for k in 0..half {
                self.source_point[(k, 0)] = attempt[(k, 0)];
                self.source_point[(k, 1)] = attempt[(k, 1)];
            }
        }
    }

    fn scale_bottom_down_landmarks(&mut self) {
        for _ in 1..self.pyramid_depth {
            let n = if self.transformation == RIGID_BODY {
                self.t()
            } else {
                self.t() / 2
            };
            for i in 0..n {
                self.source_point[(i, 0)] *= 0.5;
                self.source_point[(i, 1)] *= 0.5;
                self.target_point[(i, 0)] *= 0.5;
                self.target_point[(i, 1)] *= 0.5;
            }
        }
    }

    fn scale_up_landmarks(&mut self) {
        let n = if self.transformation == RIGID_BODY {
            self.t()
        } else {
            self.t() / 2
        };
        for i in 0..n {
            self.source_point[(i, 0)] *= 2.0;
            self.source_point[(i, 1)] *= 2.0;
            self.target_point[(i, 0)] *= 2.0;
            self.target_point[(i, 1)] *= 2.0;
        }
    }

    /* ---------- registration driver ---------- */

    /// Run the pyramid optimisation. The masks are optional; where one is
    /// `None` its pyramid pops yield empty slices, which the per-pixel test
    /// reads as "nothing excluded".
    pub fn do_registration(
        &mut self,
        source_img: &mut TurboRegImage,
        mut source_msk: Option<&mut TurboRegMask>,
        target_img: &mut TurboRegImage,
        mut target_msk: Option<&mut TurboRegMask>,
    ) {
        // The pyramids are consumed level by level and never read again, so
        // they are moved out rather than cloned — at 512×512 the clone alone
        // copied ~10 MB per registration.
        let mut src_img_pyr = std::mem::take(&mut source_img.pyramid);
        let mut tar_img_pyr = std::mem::take(&mut target_img.pyramid);
        let take_pyramid = |m: &mut Option<&mut TurboRegMask>| {
            m.as_mut()
                .map(|m| std::mem::take(&mut m.pyramid))
                .unwrap_or_default()
        };
        let mut src_msk_pyr = take_pyramid(&mut source_msk);
        let mut tar_msk_pyr = take_pyramid(&mut target_msk);

        self.pyramid_depth = target_img.pyramid_depth;
        self.iteration_power = ITERATION_PROGRESSION.pow(self.pyramid_depth as u32);

        self.scale_bottom_down_landmarks();

        while !tar_img_pyr.is_empty() {
            let src_img_item = src_img_pyr.pop().unwrap();
            let tar_img_item = tar_img_pyr.pop().unwrap();
            // Empty when that side has no mask: the pyramid is then empty too,
            // so every level pops `None`.
            let src_msk_item = src_msk_pyr.pop().unwrap_or_default();
            let tar_msk_item = tar_msk_pyr.pop().unwrap_or_default();

            self.iteration_power /= ITERATION_PROGRESSION;

            if self.transformation == BILINEAR {
                self.in_nx = src_img_item.half_width as i32;
                self.in_ny = src_img_item.half_height as i32;
                self.in_img = src_img_item.half_img;
                self.in_msk = src_msk_item;
                self.out_nx = tar_img_item.half_width as i32;
                self.out_ny = tar_img_item.half_height as i32;
                self.out_img = tar_img_item.half_img;
                self.out_msk = tar_msk_item;
            } else {
                self.in_nx = tar_img_item.half_width as i32;
                self.in_ny = tar_img_item.half_height as i32;
                self.in_img = tar_img_item.half_img;
                self.in_msk = tar_msk_item;
                self.out_nx = src_img_item.half_width as i32;
                self.out_ny = src_img_item.half_height as i32;
                self.out_img = src_img_item.half_img;
                self.x_gradient = src_img_item.x_gradient;
                self.y_gradient = src_img_item.y_gradient;
                self.out_msk = src_msk_item;
            }
            self.refresh_mask_flag();
            match self.transformation {
                TRANSLATION => {
                    self.target_jacobian = 1.0;
                    self.inverse_marquardt_levenberg_optimization();
                }
                RIGID_BODY => self.inverse_marquardt_levenberg_rigid_body_optimization(),
                SCALED_ROTATION => {
                    self.target_jacobian = (self.target_point[(0, 0)] - self.target_point[(1, 0)])
                        * (self.target_point[(0, 0)] - self.target_point[(1, 0)])
                        + (self.target_point[(0, 1)] - self.target_point[(1, 1)])
                            * (self.target_point[(0, 1)] - self.target_point[(1, 1)]);
                    self.inverse_marquardt_levenberg_optimization();
                }
                AFFINE => {
                    self.target_jacobian = (self.target_point[(1, 0)] - self.target_point[(2, 0)])
                        * self.target_point[(0, 1)]
                        + (self.target_point[(2, 0)] - self.target_point[(0, 0)])
                            * self.target_point[(1, 1)]
                        + (self.target_point[(0, 0)] - self.target_point[(1, 0)])
                            * self.target_point[(2, 1)];
                    self.inverse_marquardt_levenberg_optimization();
                }
                BILINEAR => self.marquardt_levenberg_optimization(),
                _ => {}
            }
            self.scale_up_landmarks();
        }

        self.iteration_power /= ITERATION_PROGRESSION;
        // The finest level likewise moves the full-resolution buffers out of
        // the image/mask objects instead of copying them.
        let src_mask = source_msk
            .map(|m| std::mem::take(&mut m.mask))
            .unwrap_or_default();
        let tar_mask = target_msk
            .map(|m| std::mem::take(&mut m.mask))
            .unwrap_or_default();
        if self.transformation == BILINEAR {
            self.in_nx = source_img.width as i32;
            self.in_ny = source_img.height as i32;
            self.in_img = std::mem::take(&mut source_img.coefficient);
            self.in_msk = src_mask;
            self.out_nx = target_img.width as i32;
            self.out_ny = target_img.height as i32;
            self.out_img = std::mem::take(&mut target_img.image);
            self.out_msk = tar_mask;
        } else {
            self.in_nx = target_img.width as i32;
            self.in_ny = target_img.height as i32;
            self.in_img = std::mem::take(&mut target_img.coefficient);
            self.in_msk = tar_mask;
            self.out_nx = source_img.width as i32;
            self.out_ny = source_img.height as i32;
            self.out_img = std::mem::take(&mut source_img.image);
            self.x_gradient = std::mem::take(&mut source_img.x_gradient);
            self.y_gradient = std::mem::take(&mut source_img.y_gradient);
            self.out_msk = src_mask;
        }
        self.refresh_mask_flag();

        match self.transformation {
            RIGID_BODY => self.inverse_marquardt_levenberg_rigid_body_optimization(),
            TRANSLATION | SCALED_ROTATION | AFFINE => {
                self.inverse_marquardt_levenberg_optimization()
            }
            BILINEAR => self.marquardt_levenberg_optimization(),
            _ => {}
        }
    }

    /// Warp `source` (whose B-spline coefficients are `coefficient`) by the short
    /// transformation matrix `m`. Mirrors `doFinalTransform(sourceImg, m)`.
    pub fn do_final_transform_matrix(
        &mut self,
        coefficient: Vec<f64>,
        width: usize,
        height: usize,
        m: &Mat,
    ) -> Vec<f64> {
        self.in_img = coefficient;
        self.in_nx = width as i32;
        self.in_ny = height as i32;
        self.out_nx = width as i32;
        self.out_ny = height as i32;
        self.out_img = vec![0.0; width * height];
        match self.transformation {
            TRANSLATION => self.translation_transform(m),
            RIGID_BODY | SCALED_ROTATION | AFFINE => self.general_transform(m, false),
            BILINEAR => self.general_transform(m, true),
            _ => {}
        }
        std::mem::take(&mut self.out_img)
    }
}

/* ---------------- per-band evaluators ---------------- */

/// Everything a band needs to evaluate its rows. All shared, so it crosses
/// thread boundaries freely.
struct EvalCtx<'a> {
    in_img: &'a [f64],
    in_msk: &'a [bool],
    out_img: &'a [f64],
    out_msk: &'a [bool],
    x_gradient: &'a [f64],
    y_gradient: &'a [f64],
    in_nx: i32,
    in_ny: i32,
    out_nx: i32,
    all_active: bool,
}

impl EvalCtx<'_> {
    #[inline(always)]
    fn sampler(&self) -> Sampler<'_> {
        Sampler::new(self.in_img, self.in_nx, self.in_ny)
    }

    /// Does this pixel pair take part in the fit? `k` indexes the output grid,
    /// `in_off` the input grid. Either mask may be absent (empty), in which case
    /// that side excludes nothing; when both are, `all_active` short-circuits
    /// before any of this runs.
    #[inline(always)]
    fn active(&self, k: usize, in_off: i32) -> bool {
        if self.all_active {
            return true;
        }
        // SAFETY: callers gate on `0 <= x_msk < in_nx` and `0 <= y_msk < in_ny`,
        // so `in_off` indexes inside `in_msk`; `k < out_msk.len()`.
        let m_in = self.in_msk.is_empty() || unsafe { *self.in_msk.get_unchecked(in_off as usize) };
        let m_out = self.out_msk.is_empty() || unsafe { *self.out_msk.get_unchecked(k) };
        m_in && m_out
    }
}

/// Accumulate the difference (and, per `MODE`, the gradient and Hessian) for a
/// single active pixel.
#[inline(always)]
fn accumulate_pixel<const MODE: u8, const T: usize>(
    acc: &mut Accum,
    difference: f64,
    d: &[f64; MAX_T],
) {
    acc.ms += difference * difference;
    if MODE >= MS_G {
        for i in 0..T {
            acc.grad[i] += difference * d[i];
        }
    }
    if MODE >= MS_GH {
        // Upper triangle only; the caller mirrors it. `T` is a constant here,
        // so this unrolls to a fixed rank-1 update.
        for i in 0..T {
            for j in i..T {
                acc.hess[i * MAX_T + j] += d[i] * d[j];
            }
        }
    }
}

/* ---------------- SIMD dispatch for the band kernels ---------------- */
//
// Each kernel below is `#[inline(always)]` and gets an AVX2+FMA twin that does
// nothing but call it, so the body really is compiled into a
// `#[target_feature]` function. Merely *calling* a kernel from inside such a
// wrapper does not work: LLVM leaves the call outstanding — in practice it
// tail-jumps to it — and the kernel keeps the crate's baseline features.
//
// What it buys per pixel: the 4x4 B-spline tap chain contracts into FMAs, and
// the four contiguous taps of an interior row fit a single 256-bit load.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn translation_band_avx2<const MODE: u8>(
    ctx: &EvalCtx,
    m: &Mat,
    v0: usize,
    v1: usize,
) -> Accum {
    translation_band::<MODE>(ctx, m, v0, v1)
}

#[inline]
fn translation_band_dispatch<const MODE: u8>(
    ctx: &EvalCtx,
    m: &Mat,
    v0: usize,
    v1: usize,
) -> Accum {
    #[cfg(target_arch = "x86_64")]
    if parallel::has_simd() {
        // SAFETY: guarded by the runtime feature check.
        return unsafe { translation_band_avx2::<MODE>(ctx, m, v0, v1) };
    }
    translation_band::<MODE>(ctx, m, v0, v1)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn affine_family_band_avx2<D: Deriv, const MODE: u8, const T: usize>(
    ctx: &EvalCtx,
    m: &Mat,
    deriv: &D,
    v0: usize,
    v1: usize,
) -> Accum {
    affine_family_band::<D, MODE, T>(ctx, m, deriv, v0, v1)
}

#[inline]
fn affine_family_band_dispatch<D: Deriv, const MODE: u8, const T: usize>(
    ctx: &EvalCtx,
    m: &Mat,
    deriv: &D,
    v0: usize,
    v1: usize,
) -> Accum {
    #[cfg(target_arch = "x86_64")]
    if parallel::has_simd() {
        // SAFETY: guarded by the runtime feature check.
        return unsafe { affine_family_band_avx2::<D, MODE, T>(ctx, m, deriv, v0, v1) };
    }
    affine_family_band::<D, MODE, T>(ctx, m, deriv, v0, v1)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn bilinear_band_avx2<const MODE: u8>(
    ctx: &EvalCtx,
    m: &Mat,
    bc: &BilinearConstants,
    v0: usize,
    v1: usize,
) -> Accum {
    bilinear_band::<MODE>(ctx, m, bc, v0, v1)
}

#[inline]
fn bilinear_band_dispatch<const MODE: u8>(
    ctx: &EvalCtx,
    m: &Mat,
    bc: &BilinearConstants,
    v0: usize,
    v1: usize,
) -> Accum {
    #[cfg(target_arch = "x86_64")]
    if parallel::has_simd() {
        // SAFETY: guarded by the runtime feature check.
        return unsafe { bilinear_band_avx2::<MODE>(ctx, m, bc, v0, v1) };
    }
    bilinear_band::<MODE>(ctx, m, bc, v0, v1)
}

/// Translation: the sample position advances by exactly one pixel per step, so
/// the in-cell coordinate — and therefore the spline weights — are constant for
/// the whole image and are hoisted out of both loops.
#[inline(always)]
fn translation_band<const MODE: u8>(ctx: &EvalCtx, m: &Mat, v0: usize, v1: usize) -> Accum {
    let (dx0, dy0) = (m[(0, 0)], m[(1, 0)]);
    let wx = spline_weights(dx0 - dx0.floor());
    let wy = spline_weights(dy0 - dy0.floor());
    let s = ctx.sampler();
    let nx = ctx.out_nx as usize;
    let mut acc = Accum::default();

    for v in v0..v1 {
        let y = dy0 + v as f64;
        let y_msk = round_msk(y);
        if y_msk < 0 || y_msk >= ctx.in_ny {
            continue;
        }
        let y_off = y_msk * ctx.in_nx;
        let yi = base_index(y);
        for (k, u) in (v * nx..).zip(0..nx) {
            let x = dx0 + u as f64;
            let x_msk = round_msk(x);
            if 0 <= x_msk && x_msk < ctx.in_nx && ctx.active(k, y_off + x_msk) {
                acc.area += 1;
                // SAFETY: `k < out_img.len()`, and the gradients are the same size.
                let difference = unsafe { *ctx.out_img.get_unchecked(k) }
                    - s.interpolate(base_index(x), &wx, yi, &wy);
                let d = if MODE >= MS_G {
                    let mut d = [0.0; MAX_T];
                    d[0] = unsafe { *ctx.x_gradient.get_unchecked(k) };
                    d[1] = unsafe { *ctx.y_gradient.get_unchecked(k) };
                    d
                } else {
                    [0.0; MAX_T]
                };
                accumulate_pixel::<MODE, 2>(&mut acc, difference, &d);
            }
        }
    }
    acc
}

/// Rigid body, scaled rotation and affine share this loop; they differ only in
/// how the per-pixel parameter derivatives are formed.
#[inline(always)]
fn affine_family_band<D: Deriv, const MODE: u8, const T: usize>(
    ctx: &EvalCtx,
    m: &Mat,
    deriv: &D,
    v0: usize,
    v1: usize,
) -> Accum {
    // `T` duplicates `D::T` because an associated const cannot be used as a
    // const-generic argument; keep them in step.
    debug_assert_eq!(D::T, T);
    let (m00, m01, m02) = (m[(0, 0)], m[(0, 1)], m[(0, 2)]);
    let (m10, m11, m12) = (m[(1, 0)], m[(1, 1)], m[(1, 2)]);
    let s = ctx.sampler();
    let nx = ctx.out_nx as usize;
    let mut acc = Accum::default();

    for v in v0..v1 {
        let vf = v as f64;
        // Row origins are computed from `v` rather than carried across rows, so
        // a band's values do not depend on which band it is.
        let mut x = m00 + vf * m02;
        let mut y = m10 + vf * m12;
        for (k, u) in (v * nx..).zip(0..nx) {
            let (px, py) = (x, y);
            x += m01;
            y += m11;
            let x_msk = round_msk(px);
            let y_msk = round_msk(py);
            if 0 <= x_msk
                && x_msk < ctx.in_nx
                && 0 <= y_msk
                && y_msk < ctx.in_ny
                && ctx.active(k, y_msk * ctx.in_nx + x_msk)
            {
                acc.area += 1;
                let (xi, fx) = split(px);
                let (yi, fy) = split(py);
                let val = s.interpolate(xi, &spline_weights(fx), yi, &spline_weights(fy));
                // SAFETY: `k` indexes the output image and gradients, all of
                // which are `out_nx * out_ny` long.
                let difference = unsafe { *ctx.out_img.get_unchecked(k) } - val;
                let d = if MODE >= MS_G {
                    deriv.d(
                        u as f64,
                        vf,
                        unsafe { *ctx.x_gradient.get_unchecked(k) },
                        unsafe { *ctx.y_gradient.get_unchecked(k) },
                    )
                } else {
                    [0.0; MAX_T]
                };
                accumulate_pixel::<MODE, T>(&mut acc, difference, &d);
            }
        }
    }
    acc
}

/// Bilinear: the sample position has a `u·v` cross term, the residual sign is
/// flipped relative to the other models, and the parameter derivatives need the
/// image gradient at the sample point — so it takes the combined
/// value-and-gradient tap.
#[inline(always)]
fn bilinear_band<const MODE: u8>(
    ctx: &EvalCtx,
    m: &Mat,
    bc: &BilinearConstants,
    v0: usize,
    v1: usize,
) -> Accum {
    let (m00, m01, m02, m03) = (m[(0, 0)], m[(0, 1)], m[(0, 2)], m[(0, 3)]);
    let (m10, m11, m12, m13) = (m[(1, 0)], m[(1, 1)], m[(1, 2)], m[(1, 3)]);
    let s = ctx.sampler();
    let nx = ctx.out_nx as usize;
    let mut acc = Accum::default();

    for v in v0..v1 {
        let vf = v as f64;
        let mut x = m00 + vf * m02;
        let mut y = m10 + vf * m12;
        let xstep = m01 + vf * m03;
        let ystep = m11 + vf * m13;
        for (k, u) in (v * nx..).zip(0..nx) {
            let (px, py) = (x, y);
            x += xstep;
            y += ystep;
            let x_msk = round_msk(px);
            let y_msk = round_msk(py);
            if 0 <= x_msk
                && x_msk < ctx.in_nx
                && 0 <= y_msk
                && y_msk < ctx.in_ny
                && ctx.active(k, y_msk * ctx.in_nx + x_msk)
            {
                acc.area += 1;
                let (xi, fx) = split(px);
                let (yi, fy) = split(py);
                // SAFETY: `k` indexes the output image, of length `out_nx * out_ny`.
                let target = unsafe { *ctx.out_img.get_unchecked(k) };
                let mut d = [0.0; MAX_T];
                let difference = if MODE >= MS_GH {
                    let (wx, dwx) = spline_weights_d(fx);
                    let (wy, dwy) = spline_weights_d(fy);
                    let (val, gx, gy) = s.interpolate_with_gradient(xi, &wx, &dwx, yi, &wy, &dwy);
                    let g = bc.weights(u as f64, vf);
                    d = [
                        gx * g[0],
                        gy * g[0],
                        gx * g[1],
                        gy * g[1],
                        gx * g[2],
                        gy * g[2],
                        gx * g[3],
                        gy * g[3],
                    ];
                    val - target
                } else {
                    s.interpolate(xi, &spline_weights(fx), yi, &spline_weights(fy)) - target
                };
                accumulate_pixel::<MODE, 8>(&mut acc, difference, &d);
            }
        }
    }
    acc
}

/* ---------------- reduction plumbing ---------------- */

fn store_g(acc: &Accum, t: usize, gradient: &mut [f64]) {
    gradient[..t].copy_from_slice(&acc.grad[..t]);
}

fn store_gh(acc: &Accum, t: usize, hessian: &mut Mat, gradient: &mut [f64]) {
    for i in 0..t {
        gradient[i] = acc.grad[i];
        for j in i..t {
            hessian[(i, j)] = acc.hess[i * MAX_T + j];
        }
    }
    for i in 1..t {
        for j in 0..i {
            hessian[(i, j)] = hessian[(j, i)];
        }
    }
}

/* ---------------- per-model derivative constants ---------------- */

/// Rigid body: rotation about the output origin plus a translation.
struct RigidBodyDeriv;

impl Deriv for RigidBodyDeriv {
    const T: usize = 3;

    #[inline(always)]
    fn d(&self, u: f64, v: f64, x_gradient: f64, y_gradient: f64) -> [f64; MAX_T] {
        let mut d = [0.0; MAX_T];
        d[0] = y_gradient * u - x_gradient * v;
        d[1] = x_gradient;
        d[2] = y_gradient;
        d
    }
}

/// Helper reproducing `getScaledRotationMeanSquares`'s pre-loop constants and the
/// per-pixel derivative block.
struct ScaledRotConstants {
    uv2: f64,
    c1: f64,
    c2: f64,
    c3: f64,
    c4: f64,
    c8: f64,
    c9: f64,
    c0: f64,
    c7: f64,
    dgxx0: f64,
    dgyx0: f64,
    dgxx1: f64,
    dgyy1: f64,
}

impl ScaledRotConstants {
    fn new(source_point: &Mat) -> Self {
        let u1 = source_point[(0, 0)];
        let u2 = source_point[(1, 0)];
        let v1 = source_point[(0, 1)];
        let v2 = source_point[(1, 1)];
        let u12 = u1 - u2;
        let v12 = v1 - v2;
        let uv2 = u12 * u12 + v12 * v12;
        let c = 0.5 * (u2 * v1 - u1 * v2) / uv2;
        let c1 = u12 / uv2;
        let c2 = v12 / uv2;
        let c3 = (uv2 - u12 * v12) / uv2;
        let c4 = (uv2 + u12 * v12) / uv2;
        let c5 = c + u1 * c1 + u2 * c2;
        let c6 = c * (u12 * u12 - v12 * v12) / uv2;
        let c7 = c1 * c4;
        let c8 = c1 - c2 - c1 * c2 * v12;
        let c9 = c1 + c2 - c1 * c2 * u12;
        let c0 = c2 * c3;
        ScaledRotConstants {
            uv2,
            c1,
            c2,
            c3,
            c4,
            c8,
            c9,
            c0,
            c7,
            dgxx0: c1 * u2 + c2 * v2,
            dgyx0: 2.0 * c,
            dgxx1: c5 + c6,
            dgyy1: c5 - c6,
        }
    }
}

impl Deriv for ScaledRotConstants {
    const T: usize = 4;

    #[inline(always)]
    fn d(&self, u: f64, v: f64, x_gradient: f64, y_gradient: f64) -> [f64; MAX_T] {
        let gxx0 = u * self.c1 + v * self.c2 - self.dgxx0;
        let gyx0 = v * self.c1 - u * self.c2 + self.dgyx0;
        let gxy0 = -gyx0;
        let gyy0 = gxx0;
        let gxx1 = v * self.c8 - u * self.c7 + self.dgxx1;
        let gyx1 = -self.c3 * gyx0;
        let gxy1 = self.c4 * gyx0;
        let gyy1 = self.dgyy1 - u * self.c9 - v * self.c0;
        let mut d = [0.0; MAX_T];
        d[0] = x_gradient * gxx0 + y_gradient * gyx0;
        d[1] = x_gradient * gxy0 + y_gradient * gyy0;
        d[2] = x_gradient * gxx1 + y_gradient * gyx1;
        d[3] = x_gradient * gxy1 + y_gradient * gyy1;
        d
    }
}

/// Helper reproducing `getAffineMeanSquares`'s pre-loop constants and per-pixel
/// derivative block (6 partials).
struct AffineConstants {
    det: f64,
    u12: f64,
    u23: f64,
    u31: f64,
    v12: f64,
    v23: f64,
    v31: f64,
    uv32: f64,
    uv21: f64,
    uv13: f64,
}

impl AffineConstants {
    fn new(source_point: &Mat) -> Self {
        let u1 = source_point[(0, 0)];
        let u2 = source_point[(1, 0)];
        let u3 = source_point[(2, 0)];
        let v1 = source_point[(0, 1)];
        let v2 = source_point[(1, 1)];
        let v3 = source_point[(2, 1)];
        let mut uv32 = u3 * v2 - u2 * v3;
        let mut uv21 = u2 * v1 - u1 * v2;
        let mut uv13 = u1 * v3 - u3 * v1;
        let det = uv32 + uv21 + uv13;
        let u12 = (u1 - u2) / det;
        let u23 = (u2 - u3) / det;
        let u31 = (u3 - u1) / det;
        let v12 = (v1 - v2) / det;
        let v23 = (v2 - v3) / det;
        let v31 = (v3 - v1) / det;
        uv32 /= det;
        uv21 /= det;
        uv13 /= det;
        AffineConstants {
            det,
            u12,
            u23,
            u31,
            v12,
            v23,
            v31,
            uv32,
            uv21,
            uv13,
        }
    }
}

impl Deriv for AffineConstants {
    const T: usize = 6;

    #[inline(always)]
    fn d(&self, u: f64, v: f64, x_gradient: f64, y_gradient: f64) -> [f64; MAX_T] {
        let g0 = self.u23 * v - self.v23 * u + self.uv32;
        let g1 = self.u31 * v - self.v31 * u + self.uv13;
        let g2 = self.u12 * v - self.v12 * u + self.uv21;
        let mut d = [0.0; MAX_T];
        d[0] = x_gradient * g0;
        d[1] = y_gradient * g0;
        d[2] = x_gradient * g1;
        d[3] = y_gradient * g1;
        d[4] = x_gradient * g2;
        d[5] = y_gradient * g2;
        d
    }
}

/// The four bilinear shape functions, from `computeBilinearGradientConstants`.
struct BilinearConstants {
    c: [f64; 4],
    cu: [f64; 4],
    cv: [f64; 4],
    cuv: [f64; 4],
}

impl BilinearConstants {
    const ZERO: BilinearConstants = BilinearConstants {
        c: [0.0; 4],
        cu: [0.0; 4],
        cv: [0.0; 4],
        cuv: [0.0; 4],
    };

    fn new(target_point: &Mat) -> Self {
        let u1 = target_point[(0, 0)];
        let u2 = target_point[(1, 0)];
        let u3 = target_point[(2, 0)];
        let u4 = target_point[(3, 0)];
        let v1 = target_point[(0, 1)];
        let v2 = target_point[(1, 1)];
        let v3 = target_point[(2, 1)];
        let v4 = target_point[(3, 1)];
        let v12 = v1 - v2;
        let v13 = v1 - v3;
        let v14 = v1 - v4;
        let v23 = v2 - v3;
        let v24 = v2 - v4;
        let v34 = v3 - v4;
        let uv12 = u1 * u2 * v12;
        let uv13 = u1 * u3 * v13;
        let uv14 = u1 * u4 * v14;
        let uv23 = u2 * u3 * v23;
        let uv24 = u2 * u4 * v24;
        let uv34 = u3 * u4 * v34;
        let det = uv12 * v34 - uv13 * v24 + uv14 * v23 + uv23 * v14 - uv24 * v13 + uv34 * v12;
        BilinearConstants {
            c: [
                (-uv34 * v2 + uv24 * v3 - uv23 * v4) / det,
                (uv34 * v1 - uv14 * v3 + uv13 * v4) / det,
                (-uv24 * v1 + uv14 * v2 - uv12 * v4) / det,
                (uv23 * v1 - uv13 * v2 + uv12 * v3) / det,
            ],
            cu: [
                (u3 * v3 * v24 - u2 * v2 * v34 - u4 * v4 * v23) / det,
                (-u3 * v3 * v14 + u1 * v1 * v34 + u4 * v4 * v13) / det,
                (u2 * v2 * v14 - u1 * v1 * v24 - u4 * v4 * v12) / det,
                (-u2 * v2 * v13 + u1 * v1 * v23 + u3 * v3 * v12) / det,
            ],
            cv: [
                (uv23 - uv24 + uv34) / det,
                (-uv13 + uv14 - uv34) / det,
                (uv12 - uv14 + uv24) / det,
                (-uv12 + uv13 - uv23) / det,
            ],
            cuv: [
                (u4 * v23 - u3 * v24 + u2 * v34) / det,
                (-u4 * v13 + u3 * v14 - u1 * v34) / det,
                (u4 * v12 - u2 * v14 + u1 * v24) / det,
                (-u3 * v1 + u2 * v13 + u3 * v2 - u1 * v23) / det,
            ],
        }
    }

    #[inline(always)]
    fn weights(&self, u: f64, v: f64) -> [f64; 4] {
        let uv = u * v;
        let mut g = [0.0; 4];
        for i in 0..4 {
            g[i] = self.cuv[i] * uv + self.cu[i] * u + self.cv[i] * v + self.c[i];
        }
        g
    }
}

/* ---------- free numeric helpers ---------- */

fn matrix_multiply(m: &Mat, v: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; m.nrows()];
    for i in 0..m.nrows() {
        let mut acc = 0.0;
        for (j, &vj) in v.iter().enumerate() {
            acc += m[(i, j)] * vj;
        }
        result[i] = acc;
    }
    result
}

fn invert_gauss(m: &mut Mat) {
    let n = m.nrows();
    let mut inverse = Mat::new(n, n);
    for i in 0..n {
        let mut max = m[(i, 0)];
        let mut abs_max = max.abs();
        for j in 0..n {
            inverse[(i, j)] = 0.0;
            if abs_max < m[(i, j)].abs() {
                max = m[(i, j)];
                abs_max = max.abs();
            }
        }
        inverse[(i, i)] = 1.0 / max;
        for j in 0..n {
            m[(i, j)] /= max;
        }
    }
    for j in 0..n {
        let mut max = m[(j, j)];
        let mut abs_max = max.abs();
        let mut k = j;
        for i in j + 1..n {
            if abs_max < m[(i, j)].abs() {
                max = m[(i, j)];
                abs_max = max.abs();
                k = i;
            }
        }
        if k != j {
            for col in j..n {
                m.data.swap(j * n + col, k * n + col);
            }
            for col in 0..n {
                inverse.data.swap(j * n + col, k * n + col);
            }
        }
        for kk in 0..=j {
            inverse[(j, kk)] /= max;
        }
        for kk in j + 1..n {
            m[(j, kk)] /= max;
            inverse[(j, kk)] /= max;
        }
        for i in j + 1..n {
            for kk in 0..=j {
                let d = m[(i, j)] * inverse[(j, kk)];
                inverse[(i, kk)] -= d;
            }
            for kk in j + 1..n {
                let mij = m[(i, j)];
                m[(i, kk)] -= mij * m[(j, kk)];
                inverse[(i, kk)] -= mij * inverse[(j, kk)];
            }
        }
    }
    for j in (1..n).rev() {
        for i in (0..j).rev() {
            for kk in 0..=j {
                let d = m[(i, j)] * inverse[(j, kk)];
                inverse[(i, kk)] -= d;
            }
            for kk in j + 1..n {
                let mij = m[(i, j)];
                m[(i, kk)] -= mij * m[(j, kk)];
                inverse[(i, kk)] -= mij * inverse[(j, kk)];
            }
        }
    }
    for i in 0..n {
        for j in 0..n {
            m[(i, j)] = inverse[(i, j)];
        }
    }
}
