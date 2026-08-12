//! Registration masks and their multiresolution pyramid.
//!
//! A mask says which pixels take part in the fit. TurboReg carried masks as f64
//! images and only ever tested them against zero, so they are plain booleans
//! here: an eighth of the memory, and the pyramid reduction collapses from a
//! sum of absolute values to "was any contributing pixel active?" — which is
//! exactly what that sum was tested for (all terms are non-negative, so the sum
//! is non-zero iff some term is).

use crate::Mask;

/// A mask together with the decimated copies used at each pyramid level.
///
/// `pyramid` is ordered fine-to-coarse and consumed back-to-front by
/// [`crate::transform::TurboRegTransform::do_registration`], in lockstep with
/// the image pyramid; `mask` is the full-resolution level, used last.
#[derive(Clone, Debug)]
pub struct TurboRegMask {
    pub pyramid: Vec<Vec<bool>>,
    pub mask: Vec<bool>,
    width: usize,
    height: usize,
    pyramid_depth: i32,
}

impl TurboRegMask {
    pub fn new(mask: &Mask) -> Self {
        TurboRegMask {
            pyramid: Vec::new(),
            mask: mask.data.clone(),
            width: mask.width,
            height: mask.height,
            pyramid_depth: 1,
        }
    }

    pub fn set_pyramid_depth(&mut self, d: i32) {
        self.pyramid_depth = d;
    }

    /// Build the pyramid. Must run after [`Self::set_pyramid_depth`], and
    /// produces exactly `pyramid_depth - 1` levels so the pops in
    /// `do_registration` stay aligned with the image pyramid.
    pub fn init(&mut self) {
        let (mut w, mut h) = (self.width, self.height);
        for level in 1..self.pyramid_depth {
            let (full_w, full_h) = (w, h);
            w /= 2;
            h /= 2;
            // Each level reduces the previous one; read it in place rather than
            // keeping a second copy alive.
            let full: &[bool] = if level == 1 {
                &self.mask
            } else {
                self.pyramid.last().unwrap()
            };
            let half = half_mask(full, full_w, full_h);
            self.pyramid.push(half);
        }
    }
}

/// Decimate a mask by two: coarse pixel `(x, y)` is active when any of the 3×3
/// fine pixels centred on `(2x, 2y)` is, clamped at the low edges. A trailing
/// odd row/column contributes to nothing, matching the image pyramid, which
/// drops it too.
fn half_mask(full: &[bool], full_width: usize, full_height: usize) -> Vec<bool> {
    let (half_w, half_h) = (full_width / 2, full_height / 2);
    let mut half = vec![false; half_w * half_h];
    for y in 0..half_h {
        // 2*y+1 <= 2*half_h-1 < full_height, so only the low edge needs clamping.
        let rows = (2 * y).saturating_sub(1)..=(2 * y + 1);
        for (x, out) in half[y * half_w..][..half_w].iter_mut().enumerate() {
            let cols = (2 * x).saturating_sub(1)..=(2 * x + 1);
            *out = rows
                .clone()
                .any(|r| cols.clone().any(|c| full[r * full_width + c]));
        }
    }
    half
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reduction TurboReg shipped: a hand-unrolled pointer walk summing
    /// absolute values. Kept here as the reference for [`half_mask`], which must
    /// mark a coarse pixel active exactly where this produced a non-zero sum.
    fn legacy_half_mask_2d(full: &[f64], full_width: usize, full_height: usize) -> Vec<f64> {
        let half_w = full_width / 2;
        let half_h = full_height / 2;
        let odd_width = 2 * half_w != full_width;
        let mut half = vec![0.0; half_w * half_h];

        let mut k: usize = 0;
        let mut n: usize = 0;

        for _y in 0..half_h.saturating_sub(1) {
            for _x in 0..half_w.saturating_sub(1) {
                half[k] += full[n].abs();
                n += 1;
                half[k] += full[n].abs();
                k += 1;
                half[k] += full[n].abs();
                n += 1;
            }
            half[k] += full[n].abs();
            n += 1;
            half[k] += full[n].abs();
            k += 1;
            n += 1;
            if odd_width {
                n += 1;
            }
            for _x in 0..half_w.saturating_sub(1) {
                half[k - half_w] += full[n].abs();
                half[k] += full[n].abs();
                n += 1;
                half[k - half_w] += full[n].abs();
                half[k - half_w + 1] += full[n].abs();
                half[k] += full[n].abs();
                k += 1;
                half[k] += full[n].abs();
                n += 1;
            }
            half[k - half_w] += full[n].abs();
            half[k] += full[n].abs();
            n += 1;
            half[k - half_w] += full[n].abs();
            half[k] += full[n].abs();
            k += 1;
            n += 1;
            if odd_width {
                n += 1;
            }
            k -= half_w;
        }

        for _x in 0..half_w.saturating_sub(1) {
            half[k] += full[n].abs();
            n += 1;
            half[k] += full[n].abs();
            k += 1;
            half[k] += full[n].abs();
            n += 1;
        }
        half[k] += full[n].abs();
        n += 1;
        half[k] += full[n].abs();
        k += 1;
        n += 1;
        if odd_width {
            n += 1;
        }
        k -= half_w;
        for _x in 0..half_w.saturating_sub(1) {
            half[k] += full[n].abs();
            n += 1;
            half[k] += full[n].abs();
            k += 1;
            half[k] += full[n].abs();
            n += 1;
        }
        half[k] += full[n].abs();
        n += 1;
        half[k] += full[n].abs();

        half
    }

    /// Deterministic xorshift, so the sweep below is reproducible.
    fn pseudo_random_bits(n: usize, seed: u64, density: u64) -> Vec<bool> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                s % 100 < density
            })
            .collect()
    }

    #[test]
    fn half_mask_matches_legacy_reduction() {
        // Even/odd widths and heights, and the 24-49 range the pyramid actually
        // decimates; densities from sparse to nearly full.
        for (w, h) in [(24, 24), (25, 24), (24, 25), (25, 25), (49, 33), (33, 49)] {
            for density in [3, 25, 50, 97] {
                for seed in [1, 2, 3] {
                    let bits = pseudo_random_bits(w * h, seed * 7919 + density, density);
                    let as_f64: Vec<f64> =
                        bits.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();

                    let got = half_mask(&bits, w, h);
                    let legacy = legacy_half_mask_2d(&as_f64, w, h);

                    assert_eq!(got.len(), legacy.len());
                    for (i, (&g, &l)) in got.iter().zip(legacy.iter()).enumerate() {
                        assert_eq!(
                            g,
                            l != 0.0,
                            "level pixel {i} of {w}x{h} (density {density}, seed {seed})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn pyramid_has_one_level_per_halving() {
        let mask = Mask::new(64, 64, vec![true; 64 * 64]);
        let mut m = TurboRegMask::new(&mask);
        m.set_pyramid_depth(3);
        m.init();
        assert_eq!(m.pyramid.len(), 2);
        assert_eq!(m.pyramid[0].len(), 32 * 32);
        assert_eq!(m.pyramid[1].len(), 16 * 16);
        assert!(m.pyramid[1].iter().all(|&b| b));
    }

    #[test]
    fn inactive_regions_survive_decimation() {
        // A fully masked-out quadrant must stay masked out at the next level,
        // except along the boundary the 3x3 footprint dilates by one.
        let (w, h) = (32, 32);
        let mut data = vec![true; w * h];
        for row in 0..16 {
            for col in 0..16 {
                data[row * w + col] = false;
            }
        }
        let half = half_mask(&data, w, h);
        assert!(!half[0], "interior of the masked quadrant became active");
        assert!(!half[7 * 16 + 7], "just inside the boundary became active");
        assert!(half[8 * 16 + 8], "the unmasked quadrant was lost");
    }
}
