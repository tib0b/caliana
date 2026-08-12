//! The `StackReg` orchestration layer, after `pystackreg/pystackreg.py`.
//!
//! Covers single-image registration/transform plus stack (movie) registration
//! along the leading axis, with `previous` / `first` / `mean` reference modes and
//! moving-average pre-smoothing. The `_masked` variants extend it with per-frame
//! masks, which pystackreg has no equivalent of.
//!
//! Note that caliana does not go through this type: it drives the native
//! `_register` / `_transform` directly and runs its own frame loop in
//! `caliana/src/caliana/_stackreg.py`, so per-frame behaviour usually needs
//! changing in both places.

use crate::matrix::Mat;
use crate::transform_kind::*;
use crate::{register_masked, transform, Image2D, Mask, Transformation};

/// Reference frame strategy for stack registration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reference {
    /// Align each frame to the previous frame (transforms are composed).
    Previous,
    /// Align each frame to the mean of the first `n_frames` frames.
    First,
    /// Align each frame to the mean of all frames.
    Mean,
}

pub struct StackReg {
    transformation: Transformation,
    /// Current short-form matrix from the last registration / `set_matrix`.
    m: Option<Mat>,
    /// Per-frame long-form matrices from the last `register_stack`.
    tmats: Option<Vec<Mat>>,
    ref_points: Option<Mat>,
    mov_points: Option<Mat>,
    is_registered: bool,
}

impl StackReg {
    pub fn new(transformation: Transformation) -> Self {
        StackReg {
            transformation,
            m: None,
            tmats: None,
            ref_points: None,
            mov_points: None,
            is_registered: false,
        }
    }

    pub fn is_registered(&self) -> bool {
        self.is_registered
    }

    fn tmat_dim(&self) -> usize {
        if self.transformation == Transformation::Bilinear {
            4
        } else {
            3
        }
    }

    /// Register `moving` onto `reference` (cropping the last row/column exactly
    /// like pystackreg). Returns the long-form transformation matrix.
    pub fn register(&mut self, reference: &Image2D, moving: &Image2D) -> Mat {
        self.register_masked(reference, None, moving, None)
    }

    /// [`StackReg::register`], restricted to the pixels either mask keeps. Masks
    /// are cropped alongside their images, so they are given at full frame size.
    /// See [`crate::register_masked`] for the per-pixel semantics.
    pub fn register_masked(
        &mut self,
        reference: &Image2D,
        reference_mask: Option<&Mask>,
        moving: &Image2D,
        moving_mask: Option<&Mask>,
    ) -> Mat {
        let r = register_masked(
            &reference.crop_last_row_col(),
            reference_mask.map(Mask::crop_last_row_col).as_ref(),
            &moving.crop_last_row_col(),
            moving_mask.map(Mask::crop_last_row_col).as_ref(),
            self.transformation,
        );
        self.m = Some(r.matrix);
        self.ref_points = Some(r.ref_points);
        self.mov_points = Some(r.mov_points);
        self.is_registered = true;
        self.get_matrix()
    }

    /// Warp `moving`. If `tmat` (long form) is given it is used; otherwise the
    /// matrix from the previous `register` call is used.
    pub fn transform(&self, moving: &Image2D, tmat: Option<&Mat>) -> Image2D {
        let short = match tmat {
            Some(t) => self.matrix_long_to_short(t),
            None => self
                .m
                .clone()
                .expect("register() must be called before transform()"),
        };
        transform(moving, &short)
    }

    pub fn register_transform(&mut self, reference: &Image2D, moving: &Image2D) -> Image2D {
        self.register(reference, moving);
        self.transform(moving, None)
    }

    pub fn get_matrix(&self) -> Mat {
        self.matrix_short_to_long(self.m.as_ref().expect("no registration matrix yet"))
    }

    pub fn set_matrix(&mut self, mat: &Mat) {
        let exp = self.tmat_dim();
        assert!(
            mat.rows == exp && mat.cols == exp,
            "Invalid shape of transformation matrix: expected {exp}x{exp}"
        );
        self.m = Some(self.matrix_long_to_short(mat));
        self.is_registered = true;
    }

    pub fn get_points(&self) -> (Option<&Mat>, Option<&Mat>) {
        (self.ref_points.as_ref(), self.mov_points.as_ref())
    }

    /* ---------- short <-> long matrix conversions ---------- */

    fn matrix_short_to_long(&self, m: &Mat) -> Mat {
        match self.transformation.code() {
            TRANSLATION => {
                let mut mat = identity(3);
                mat[(0, 2)] = m[(0, 0)];
                mat[(1, 2)] = m[(1, 0)];
                mat
            }
            RIGID_BODY | SCALED_ROTATION | AFFINE => {
                let mut mat = identity(3);
                for i in 0..2 {
                    mat[(i, 0)] = m[(i, 1)];
                    mat[(i, 1)] = m[(i, 2)];
                    mat[(i, 2)] = m[(i, 0)];
                }
                mat
            }
            BILINEAR => {
                let mut mat = identity(4);
                for i in 0..2 {
                    mat[(i, 0)] = m[(i, 1)];
                    mat[(i, 1)] = m[(i, 2)];
                    mat[(i, 2)] = m[(i, 3)];
                    mat[(i, 3)] = m[(i, 0)];
                }
                mat
            }
            _ => unreachable!(),
        }
    }

    fn matrix_long_to_short(&self, mat: &Mat) -> Mat {
        match self.transformation.code() {
            TRANSLATION => {
                let mut m = Mat::new(2, 1);
                m[(0, 0)] = mat[(0, 2)];
                m[(1, 0)] = mat[(1, 2)];
                m
            }
            RIGID_BODY | SCALED_ROTATION | AFFINE => {
                let mut m = Mat::new(2, 3);
                for i in 0..2 {
                    m[(i, 0)] = mat[(i, 2)];
                    m[(i, 1)] = mat[(i, 0)];
                    m[(i, 2)] = mat[(i, 1)];
                }
                m
            }
            BILINEAR => {
                let mut m = Mat::new(2, 4);
                for i in 0..2 {
                    m[(i, 0)] = mat[(i, 3)];
                    m[(i, 1)] = mat[(i, 0)];
                    m[(i, 2)] = mat[(i, 1)];
                    m[(i, 3)] = mat[(i, 2)];
                }
                m
            }
            _ => unreachable!(),
        }
    }

    /* ---------- stack registration (frames along axis 0) ---------- */

    /// Register a movie. Returns one long-form matrix per frame and stores them
    /// for a subsequent [`StackReg::transform_stack`].
    pub fn register_stack(
        &mut self,
        frames: &[Image2D],
        reference: Reference,
        n_frames: usize,
        moving_average: usize,
    ) -> Vec<Mat> {
        self.register_stack_masked(frames, None, reference, n_frames, moving_average)
    }

    /// [`StackReg::register_stack`] with one mask per frame.
    ///
    /// Which mask guards the reference side follows what the reference *is*:
    /// under [`Reference::Previous`] it is frame `i-1`'s own mask, and under
    /// [`Reference::First`] / [`Reference::Mean`] — where the reference is an
    /// average — it is the intersection of the averaged frames' masks, since a
    /// pixel some frame masked out has already polluted that average.
    /// `moving_average` smoothing intersects over its window for the same
    /// reason.
    pub fn register_stack_masked(
        &mut self,
        frames: &[Image2D],
        masks: Option<&[Mask]>,
        reference: Reference,
        n_frames: usize,
        moving_average: usize,
    ) -> Vec<Mat> {
        if self.transformation == Transformation::Bilinear && reference == Reference::Previous {
            panic!("Bilinear stack registration is not supported with reference == Previous");
        }
        assert!(!frames.is_empty(), "stack must contain at least one frame");
        if let Some(masks) = masks {
            assert_eq!(
                masks.len(),
                frames.len(),
                "expected one mask per frame ({} frames, {} masks)",
                frames.len(),
                masks.len()
            );
        }

        let smoothed;
        let work: &[Image2D] = if moving_average > 1 {
            smoothed = running_mean(frames, moving_average);
            &smoothed
        } else {
            frames
        };
        let narrowed;
        let masks: Option<&[Mask]> = match (masks, moving_average > 1) {
            (Some(m), true) => {
                narrowed = running_intersection(m, moving_average);
                Some(&narrowed)
            }
            (m, _) => m,
        };
        let l = work.len();
        let dim = self.tmat_dim();
        let mut tmats: Vec<Mat> = (0..l).map(|_| identity(dim)).collect();

        let mut idx_start = if moving_average > 1 { 0 } else { 1 };
        let averaged = match reference {
            Reference::First => 0..n_frames.min(l),
            Reference::Mean => 0..l,
            Reference::Previous => 0..0,
        };
        let reference_frame: Option<Image2D> = match reference {
            Reference::Previous => None,
            Reference::Mean => {
                idx_start = 0;
                Some(mean_frames(&work[averaged.clone()]))
            }
            Reference::First => Some(mean_frames(&work[averaged.clone()])),
        };
        let reference_mask: Option<Mask> = masks.and_then(|m| intersect_all(&m[averaged.clone()]));

        for i in idx_start..l {
            let ref_idx = (i as isize - 1).rem_euclid(l as isize) as usize;
            let ref_frame: &Image2D = match reference {
                Reference::Previous => &work[ref_idx],
                _ => reference_frame.as_ref().unwrap(),
            };
            let ref_mask: Option<&Mask> = match reference {
                Reference::Previous => masks.map(|m| &m[ref_idx]),
                _ => reference_mask.as_ref(),
            };
            tmats[i] = self.register_masked(ref_frame, ref_mask, &work[i], masks.map(|m| &m[i]));
            if reference == Reference::Previous && i > 0 {
                tmats[i] = matmul_square(&tmats[i], &tmats[i - 1]);
            }
        }

        self.tmats = Some(tmats.clone());
        tmats
    }

    /// Warp every frame using stored (or supplied) per-frame matrices.
    pub fn transform_stack(&self, frames: &[Image2D], tmats: Option<&[Mat]>) -> Vec<Image2D> {
        let tmats = tmats
            .map(|t| t.to_vec())
            .or_else(|| self.tmats.clone())
            .expect("register_stack() must be called first or matrices supplied");
        assert_eq!(
            tmats.len(),
            frames.len(),
            "matrix count must match stack length"
        );
        frames
            .iter()
            .zip(tmats.iter())
            .map(|(f, t)| self.transform(f, Some(t)))
            .collect()
    }

    pub fn register_transform_stack(
        &mut self,
        frames: &[Image2D],
        reference: Reference,
        n_frames: usize,
        moving_average: usize,
    ) -> Vec<Image2D> {
        self.register_stack(frames, reference, n_frames, moving_average);
        self.transform_stack(frames, None)
    }
}

fn identity(n: usize) -> Mat {
    let mut m = Mat::new(n, n);
    for i in 0..n {
        m[(i, i)] = 1.0;
    }
    m
}

fn matmul_square(a: &Mat, b: &Mat) -> Mat {
    let n = a.rows;
    let mut c = Mat::new(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                acc += a[(i, k)] * b[(k, j)];
            }
            c[(i, j)] = acc;
        }
    }
    c
}

fn mean_frames(frames: &[Image2D]) -> Image2D {
    let w = frames[0].width;
    let h = frames[0].height;
    let mut data = vec![0.0; w * h];
    for f in frames {
        for (d, &v) in data.iter_mut().zip(f.data.iter()) {
            *d += v;
        }
    }
    let n = frames.len() as f64;
    for d in data.iter_mut() {
        *d /= n;
    }
    Image2D {
        width: w,
        height: h,
        data,
    }
}

/// The mask keeping only what every one of `masks` keeps, or `None` for an
/// empty slice (no frames were averaged, so nothing constrains the reference).
fn intersect_all(masks: &[Mask]) -> Option<Mask> {
    let (first, rest) = masks.split_first()?;
    Some(rest.iter().fold(first.clone(), |acc, m| acc.intersect(m)))
}

/// Per-frame mask intersection over the same edge-padded window [`running_mean`]
/// averages, so a smoothed frame is masked wherever any frame blended into it
/// was.
fn running_intersection(masks: &[Mask], n: usize) -> Vec<Mask> {
    let l = masks.len();
    let mut padded: Vec<&Mask> = Vec::with_capacity(l + n);
    padded.extend(std::iter::repeat_n(&masks[0], n.div_ceil(2)));
    padded.extend(masks.iter());
    padded.extend(std::iter::repeat_n(&masks[l - 1], n / 2));

    (0..l)
        .map(|i| {
            // Window `i+1 ..= i+n` of the padded sequence, matching the
            // prefix-sum span `running_mean` divides by `n`.
            let (first, rest) = padded[i + 1..=i + n].split_first().unwrap();
            rest.iter()
                .fold((*first).clone(), |acc, m| acc.intersect(m))
        })
        .collect()
}

/// Moving average across frames with edge padding, reproducing the semantics of
/// pystackreg's `running_mean` (output has the same length as the input).
fn running_mean(frames: &[Image2D], n: usize) -> Vec<Image2D> {
    let l = frames.len();
    let w = frames[0].width;
    let h = frames[0].height;
    let pad_front = n.div_ceil(2);
    let pad_back = n / 2;

    // Build the edge-padded sequence.
    let mut padded: Vec<&Image2D> = Vec::with_capacity(l + n);
    for _ in 0..pad_front {
        padded.push(&frames[0]);
    }
    for f in frames {
        padded.push(f);
    }
    for _ in 0..pad_back {
        padded.push(&frames[l - 1]);
    }

    // Prefix sums: cumsum[k] = sum(padded[0..=k]).
    let plen = padded.len();
    let mut cumsum: Vec<Vec<f64>> = Vec::with_capacity(plen);
    let mut acc = vec![0.0; w * h];
    for p in &padded {
        for (a, &v) in acc.iter_mut().zip(p.data.iter()) {
            *a += v;
        }
        cumsum.push(acc.clone());
    }

    let nf = n as f64;
    (0..l)
        .map(|i| {
            let hi = &cumsum[n + i];
            let lo = &cumsum[i];
            let data: Vec<f64> = hi
                .iter()
                .zip(lo.iter())
                .map(|(&a, &b)| (a - b) / nf)
                .collect();
            Image2D {
                width: w,
                height: h,
                data,
            }
        })
        .collect()
}
