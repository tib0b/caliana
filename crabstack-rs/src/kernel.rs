//! Cubic B-spline sampling kernel and the per-band accumulator.
//!
//! The C++ original (and the first Rust port) kept the sample position, the
//! four-tap weights and the four-tap indices in *object fields*, so every pixel
//! round-tripped them through memory and nothing could stay in a register. Here
//! they are plain locals and the tap gather is a free function, which lets LLVM
//! keep the 4×4 stencil in registers and vectorise it.
//!
//! The other change that matters is [`Sampler::interpolate`]'s interior fast
//! path. Away from the border no mirror-folding is needed, so the four taps of a
//! row are *contiguous* — four adjacent `f64` instead of four gathered loads —
//! and the stencil is accumulated column-wise so the reduction happens once
//! instead of once per row. See [`Sampler::interior_columns`].

/// Integer cell index of `v`, matching TurboReg's `xIndexes`/`yIndexes`.
///
/// This is `floor(v)` except at exact negative integers, where the original
/// truncation lands one cell lower (with a compensating fractional part of
/// 1.0). Index and fraction are derived from the same value here, so the pair
/// stays consistent exactly as it does in the C++.
#[inline(always)]
pub(crate) fn base_index(v: f64) -> i32 {
    let t = v as i32;
    if v >= 0.0 {
        t
    } else {
        t - 1
    }
}

/// Split `v` into the cell index and the in-cell coordinate used by the spline.
#[inline(always)]
pub(crate) fn split(v: f64) -> (i32, f64) {
    let i = base_index(v);
    (i, v - i as f64)
}

/// Nearest-integer mask coordinate (`round half away from zero`), as TurboReg does it.
#[inline(always)]
pub(crate) fn round_msk(v: f64) -> i32 {
    if v >= 0.0 {
        (v + 0.5) as i32
    } else {
        (v - 0.5) as i32
    }
}

/// Cubic B-spline weights for in-cell coordinate `t`.
///
/// `w[0]` belongs to cell `i+2` and `w[3]` to cell `i-1` — the same descending
/// tap order the original walks.
#[inline(always)]
pub(crate) fn spline_weights(t: f64) -> [f64; 4] {
    let s = 1.0 - t;
    let t2 = t * t;
    let w3 = s * s * s / 6.0;
    let w2 = 2.0 / 3.0 - 0.5 * t2 * (2.0 - t);
    let w0 = t2 * t / 6.0;
    [w0, 1.0 - w0 - w2 - w3, w2, w3]
}

/// Spline weights together with their derivatives, for the bilinear model.
#[inline(always)]
pub(crate) fn spline_weights_d(t: f64) -> ([f64; 4], [f64; 4]) {
    let s = 1.0 - t;
    let d0 = 0.5 * t * t;
    let d3 = -0.5 * s * s;
    let d = [d0, 1.0 - 2.0 * d0 + d3, 1.5 * t * (t - 4.0 / 3.0), d3];
    let w = [
        t * d0 / 3.0,
        2.0 / 3.0 + (1.0 + t) * d3,
        2.0 / 3.0 - (2.0 - t) * d0,
        s * d3 / -3.0,
    ];
    (w, d)
}

/// Mirror-fold a tap index into `[0, n)`, matching `xIndexes`/`yIndexes`.
#[inline(always)]
fn fold(p: i32, n: i32, twice_n: i32) -> i32 {
    let mut q = if p < 0 { -1 - p } else { p };
    if twice_n <= q {
        q -= twice_n * (q / twice_n);
    }
    if n <= q {
        twice_n - 1 - q
    } else {
        q
    }
}

/// The four mirror-folded tap indices starting at `p` and descending.
#[inline(always)]
fn fold4(p: i32, n: i32, twice_n: i32) -> [i32; 4] {
    [
        fold(p, n, twice_n),
        fold(p - 1, n, twice_n),
        fold(p - 2, n, twice_n),
        fold(p - 3, n, twice_n),
    ]
}

/// A B-spline coefficient image that can be sampled at arbitrary positions.
pub(crate) struct Sampler<'a> {
    img: &'a [f64],
    nx: i32,
    ny: i32,
    twice_nx: i32,
    twice_ny: i32,
}

impl<'a> Sampler<'a> {
    pub(crate) fn new(img: &'a [f64], nx: i32, ny: i32) -> Self {
        Sampler {
            img,
            nx,
            ny,
            twice_nx: 2 * nx,
            twice_ny: 2 * ny,
        }
    }

    /// True when the 4×4 stencil at cell `(xi, yi)` lies wholly inside the
    /// image, so no mirror-folding is needed and the taps are contiguous.
    #[inline(always)]
    fn interior(&self, xi: i32, yi: i32) -> bool {
        xi >= 1 && xi <= self.nx - 3 && yi >= 1 && yi <= self.ny - 3
    }

    /// Sample with precomputed weights (`wx[0]` ↔ cell `xi+2`, as in
    /// [`spline_weights`]).
    #[inline(always)]
    pub(crate) fn interpolate(&self, xi: i32, wx: &[f64; 4], yi: i32, wy: &[f64; 4]) -> f64 {
        if self.interior(xi, yi) {
            // SAFETY: `interior` bounds the whole 4×4 block inside the image.
            unsafe { self.interior_tap(xi, wx, yi, wy) }
        } else {
            self.folded_tap(xi, wx, yi, wy)
        }
    }

    /// Accumulate the 4x4 interior stencil *vertically*: `col[i]` ends up
    /// holding the row-weighted sum of column `xi-1+i`.
    ///
    /// Doing it this way — rather than reducing each row to a scalar and then
    /// combining — is what makes the stencil vectorise. Each step is four
    /// independent lane-wise multiply-accumulates (one 256-bit operation),
    /// and the single horizontal reduction is deferred to the caller. The
    /// row-first form needs four horizontal reductions instead, and the
    /// compiler cannot rewrite one into the other on its own because
    /// reassociating floating-point sums is not a legal optimisation.
    ///
    /// SAFETY: requires `self.interior(xi, yi)`.
    #[inline(always)]
    unsafe fn interior_columns(&self, xi: i32, yi: i32, wy: &[f64; 4]) -> [f64; 4] {
        let nx = self.nx as usize;
        let mut base = (yi - 1) as usize * nx + (xi - 1) as usize;
        let mut col = [0.0f64; 4];
        // Rows run `yi-1 ..= yi+2`, so they pair with the weights in reverse.
        for j in (0..4).rev() {
            let r = self.img.get_unchecked(base..base + 4);
            let w = wy[j];
            for i in 0..4 {
                col[i] += w * r[i];
            }
            base += nx;
        }
        col
    }

    /// Reduce the column sums against the x weights. `col[i]` belongs to column
    /// `xi-1+i`, so the weights are consumed in reverse.
    #[inline(always)]
    fn reduce_columns(col: &[f64; 4], wx: &[f64; 4]) -> f64 {
        col[0] * wx[3] + col[1] * wx[2] + col[2] * wx[1] + col[3] * wx[0]
    }

    /// Contiguous-tap sample.
    ///
    /// SAFETY: requires `self.interior(xi, yi)`.
    #[inline(always)]
    unsafe fn interior_tap(&self, xi: i32, wx: &[f64; 4], yi: i32, wy: &[f64; 4]) -> f64 {
        Self::reduce_columns(&self.interior_columns(xi, yi, wy), wx)
    }

    /// General sample: mirror-folds every tap. Used near the border.
    #[inline]
    fn folded_tap(&self, xi: i32, wx: &[f64; 4], yi: i32, wy: &[f64; 4]) -> f64 {
        let cols = fold4(xi + 2, self.nx, self.twice_nx);
        let rows = fold4(yi + 2, self.ny, self.twice_ny);
        let mut acc = 0.0;
        for j in 0..4 {
            let p = (rows[j] * self.nx) as usize;
            let mut s = 0.0;
            for i in 0..4 {
                // SAFETY: `fold` returns indices in `[0, nx)` / `[0, ny)`, so
                // `p + cols[i]` is inside the image — the same invariant the
                // C++ raw-pointer walk relies on.
                s += wx[i] * unsafe { *self.img.get_unchecked(p + cols[i] as usize) };
            }
            acc += wy[j] * s;
        }
        acc
    }

    /// Sample together with both partial derivatives, in a single pass over the
    /// 4×4 stencil. The bilinear optimiser needs all three, and the original
    /// walked the taps three separate times to get them.
    #[inline(always)]
    pub(crate) fn interpolate_with_gradient(
        &self,
        xi: i32,
        wx: &[f64; 4],
        dwx: &[f64; 4],
        yi: i32,
        wy: &[f64; 4],
        dwy: &[f64; 4],
    ) -> (f64, f64, f64) {
        if self.interior(xi, yi) {
            // Two vertical passes give all three results: the value and the
            // x-derivative share the `wy`-weighted columns and differ only in
            // which x weights reduce them, while the y-derivative is the same
            // reduction over `dwy`-weighted columns.
            // SAFETY: `interior` bounds the whole 4×4 block inside the image.
            let col = unsafe { self.interior_columns(xi, yi, wy) };
            let dcol = unsafe { self.interior_columns(xi, yi, dwy) };
            (
                Self::reduce_columns(&col, wx),
                Self::reduce_columns(&col, dwx),
                Self::reduce_columns(&dcol, wx),
            )
        } else {
            let cols = fold4(xi + 2, self.nx, self.twice_nx);
            let rows = fold4(yi + 2, self.ny, self.twice_ny);
            let (mut val, mut gx, mut gy) = (0.0, 0.0, 0.0);
            for j in 0..4 {
                let p = (rows[j] * self.nx) as usize;
                let (mut s, mut ds) = (0.0, 0.0);
                for i in 0..4 {
                    // SAFETY: see `folded_tap`.
                    let c = unsafe { *self.img.get_unchecked(p + cols[i] as usize) };
                    s += wx[i] * c;
                    ds += dwx[i] * c;
                }
                val += wy[j] * s;
                gx += wy[j] * ds;
                gy += dwy[j] * s;
            }
            (val, gx, gy)
        }
    }
}

/* ---------------- reduction accumulator ---------------- */

/// Maximum number of transformation parameters (bilinear).
pub(crate) const MAX_T: usize = 8;

/// Partial sums produced by one row band. Combined in band order so the result
/// does not depend on the thread count.
#[derive(Clone, Copy)]
pub(crate) struct Accum {
    pub ms: f64,
    pub area: i64,
    pub grad: [f64; MAX_T],
    /// Upper triangle of the Hessian, row-major in an `MAX_T × MAX_T` grid.
    pub hess: [f64; MAX_T * MAX_T],
}

impl Default for Accum {
    fn default() -> Self {
        Accum {
            ms: 0.0,
            area: 0,
            grad: [0.0; MAX_T],
            hess: [0.0; MAX_T * MAX_T],
        }
    }
}

impl Accum {
    #[inline]
    pub(crate) fn merge(&mut self, o: &Accum) {
        self.ms += o.ms;
        self.area += o.area;
        for i in 0..MAX_T {
            self.grad[i] += o.grad[i];
        }
        for i in 0..MAX_T * MAX_T {
            self.hess[i] += o.hess[i];
        }
    }

    /// Reduce band partials in band order.
    pub(crate) fn reduce(parts: &[Accum]) -> Accum {
        let mut total = Accum::default();
        for p in parts {
            total.merge(p);
        }
        total
    }
}

/* ---------------- evaluation modes ---------------- */

/// Mean squares only.
pub(crate) const MS: u8 = 0;
/// Mean squares and gradient.
pub(crate) const MS_G: u8 = 1;
/// Mean squares, gradient and Hessian.
pub(crate) const MS_GH: u8 = 2;

/// Per-pixel parameter derivatives for the models that share the affine-family
/// loop shape (rigid body, scaled rotation, affine).
pub(crate) trait Deriv: Sync {
    /// Number of transformation parameters.
    const T: usize;
    /// ∂(warped sample)/∂(parameter) at output pixel `(u, v)`.
    fn d(&self, u: f64, v: f64, x_gradient: f64, y_gradient: f64) -> [f64; MAX_T];
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The interior fast path and the general folded path must agree wherever
    /// both are valid — otherwise the warp would show a seam at the border.
    #[test]
    fn interior_and_folded_paths_agree() {
        let (nx, ny) = (17i32, 13i32);
        let img: Vec<f64> = (0..(nx * ny))
            .map(|i| ((i * 37) % 101) as f64 - 50.0)
            .collect();
        let s = Sampler::new(&img, nx, ny);
        for yi in 1..=(ny - 3) {
            for xi in 1..=(nx - 3) {
                for &t in &[0.0, 0.25, 0.5, 0.9] {
                    let wx = spline_weights(t);
                    let wy = spline_weights(1.0 - t);
                    // SAFETY: the loop bounds are exactly `interior`.
                    let fast = unsafe { s.interior_tap(xi, &wx, yi, &wy) };
                    let slow = s.folded_tap(xi, &wx, yi, &wy);
                    assert!(
                        (fast - slow).abs() < 1e-12 * slow.abs().max(1.0),
                        "xi={xi} yi={yi} t={t}: {fast} vs {slow}"
                    );
                }
            }
        }
    }

    #[test]
    fn weights_sum_to_one() {
        for i in 0..=20 {
            let t = i as f64 / 20.0;
            let w = spline_weights(t);
            assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-15);
            let (wd, d) = spline_weights_d(t);
            assert!((wd.iter().sum::<f64>() - 1.0).abs() < 1e-15);
            // The derivative weights must sum to zero (a constant image has
            // zero gradient).
            assert!(d.iter().sum::<f64>().abs() < 1e-15);
        }
    }

    #[test]
    fn base_index_matches_original_truncation() {
        // Positive and non-integer negatives behave like `floor`; exact
        // negative integers land one cell lower, as in the C++.
        assert_eq!(base_index(3.7), 3);
        assert_eq!(base_index(0.0), 0);
        assert_eq!(base_index(-1.5), -2);
        assert_eq!(base_index(-1.0), -2);
        let (i, f) = split(-1.0);
        assert_eq!((i, f), (-2, 1.0));
    }
}
