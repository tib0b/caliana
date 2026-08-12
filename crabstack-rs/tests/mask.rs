//! Masked registration: only the pixels a mask keeps may constrain the fit.

use crabstack::{
    register, register_masked, stack::Reference, Image2D, Mask, StackReg, Transformation,
};

/// A smooth analytic image, shifted by (dx, dy) so a moving frame has real
/// sub-pixel structure to lock onto.
fn make_image(w: usize, h: usize, dx: f64, dy: f64) -> Image2D {
    let mut data = vec![0.0; w * h];
    let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
    for row in 0..h {
        for col in 0..w {
            let x = col as f64 - dx;
            let y = row as f64 - dy;
            let wave = 80.0 * (x * 0.15).sin() * (y * 0.11).cos();
            let r2 = (x - cx).powi(2) + (y - cy).powi(2);
            let blob = 60.0 * (-r2 / (2.0 * 9.0 * 9.0)).exp();
            data[row * w + col] = 128.0 + wave + blob;
        }
    }
    Image2D::new(w, h, data)
}

/// A high-contrast checkerboard painted at the *same* place in every frame it is
/// applied to — a stand-in for the static clutter (a rig edge, a label, a dead
/// sensor patch) that masking exists to ignore. Several times the amplitude of
/// the real signal, so an unmasked fit visibly anchors to it.
struct Block {
    x0: usize,
    y0: usize,
    size: usize,
}

impl Block {
    fn paint(&self, img: &mut Image2D) {
        for row in self.y0..self.y0 + self.size {
            for col in self.x0..self.x0 + self.size {
                let checker = if (row / 2 + col / 2) % 2 == 0 {
                    1.0
                } else {
                    -1.0
                };
                img.data[row * img.width + col] = 128.0 + 300.0 * checker;
            }
        }
    }

    /// A mask excluding this block plus `MARGIN` pixels of surround. The margin
    /// is not optional: the cubic B-spline interpolation and the pyramid
    /// decimation both spread a feature about two pixels past its edge, so a
    /// mask cut exactly to the block leaves contaminated pixels in the fit.
    fn mask(&self, w: usize, h: usize) -> Mask {
        const MARGIN: usize = 2;
        let mut data = vec![true; w * h];
        let rows = self.y0.saturating_sub(MARGIN)..(self.y0 + self.size + MARGIN).min(h);
        let cols = self.x0.saturating_sub(MARGIN)..(self.x0 + self.size + MARGIN).min(w);
        for row in rows {
            for col in cols.clone() {
                data[row * w + col] = false;
            }
        }
        Mask::new(w, h, data)
    }
}

fn all_true(w: usize, h: usize) -> Mask {
    Mask::new(w, h, vec![true; w * h])
}

fn translation(r: &crabstack::Registration) -> (f64, f64) {
    (r.matrix[(0, 0)], r.matrix[(1, 0)])
}

/// How far a recovered translation sits from the injected one.
fn err(got: (f64, f64), want: (f64, f64)) -> f64 {
    (got.0 - want.0).hypot(got.1 - want.1)
}

/// A mask that keeps every pixel must not change anything: it is the same fit
/// the unmasked path runs, just with the per-pixel test switched on.
#[test]
fn all_true_mask_matches_no_mask() {
    let (w, h) = (64, 64);
    let reference = make_image(w, h, 0.0, 0.0);
    let moving = make_image(w, h, 3.0, -2.0);
    let ones = all_true(w, h);

    for t in [
        Transformation::Translation,
        Transformation::RigidBody,
        Transformation::ScaledRotation,
        Transformation::Affine,
    ] {
        let plain = register(&reference, &moving, t);
        let masked = register_masked(&reference, Some(&ones), &moving, Some(&ones), t);
        for i in 0..plain.matrix.rows {
            for j in 0..plain.matrix.cols {
                assert_eq!(
                    plain.matrix[(i, j)],
                    masked.matrix[(i, j)],
                    "{t:?} matrix ({i},{j}) moved under an all-true mask"
                );
            }
        }
    }
}

/// Masking out clean pixels costs accuracy but must not bias the fit: with a
/// quarter of an undisturbed image excluded, the answer is still exact.
#[test]
fn excluding_clean_pixels_leaves_the_fit_intact() {
    let (w, h) = (64, 64);
    let want = (3.0, -2.0);
    let reference = make_image(w, h, 0.0, 0.0);
    let moving = make_image(w, h, want.0, want.1);

    for (x0, y0, size) in [(0, 0, 32), (40, 40, 20), (20, 4, 24)] {
        let hole = Block { x0, y0, size }.mask(w, h);
        for (rm, mm) in [
            (Some(&hole), Some(&hole)),
            (Some(&hole), None),
            (None, Some(&hole)),
        ] {
            let got = translation(&register_masked(
                &reference,
                rm,
                &moving,
                mm,
                Transformation::Translation,
            ));
            assert!(
                err(got, want) < 0.05,
                "hole at ({x0},{y0}) size {size} (ref={}, mov={}): got {got:?}, want {want:?}",
                rm.is_some(),
                mm.is_some()
            );
        }
    }
}

/// The point of the feature: static clutter drags an unmasked fit off the true
/// shift, and masking it out recovers it.
#[test]
fn masking_out_static_clutter_recovers_the_shift() {
    let (w, h) = (64, 64);
    let block = Block {
        x0: 40,
        y0: 40,
        size: 16,
    };
    let mask = block.mask(w, h);

    for want in [(3.0, -2.0), (2.0, -1.0), (4.0, -2.0)] {
        let mut reference = make_image(w, h, 0.0, 0.0);
        let mut moving = make_image(w, h, want.0, want.1);
        block.paint(&mut reference);
        block.paint(&mut moving);

        let unmasked = translation(&register(&reference, &moving, Transformation::Translation));
        let masked = translation(&register_masked(
            &reference,
            Some(&mask),
            &moving,
            Some(&mask),
            Transformation::Translation,
        ));

        assert!(
            err(masked, want) < 0.05,
            "masked fit missed the injected shift: got {masked:?}, want {want:?}"
        );
        assert!(
            err(unmasked, want) > 0.5,
            "the clutter did not bias the unmasked fit, so this proves nothing: \
             got {unmasked:?}, want {want:?}"
        );
    }
}

/// Clutter present in only one frame is excluded by that frame's mask alone —
/// the pair test drops a pixel when *either* side rejects it.
#[test]
fn a_single_sided_mask_excludes_clutter_in_its_own_frame() {
    let (w, h) = (64, 64);
    let want = (3.0, -2.0);
    let block = Block {
        x0: 40,
        y0: 40,
        size: 16,
    };
    let mask = block.mask(w, h);

    // Clutter in the moving frame, masked on the moving side (`out_msk`, indexed
    // by the pixel being visited).
    let reference = make_image(w, h, 0.0, 0.0);
    let mut moving = make_image(w, h, want.0, want.1);
    block.paint(&mut moving);
    let got = translation(&register_masked(
        &reference,
        None,
        &moving,
        Some(&mask),
        Transformation::Translation,
    ));
    assert!(err(got, want) < 0.05, "moving-side mask: got {got:?}");

    // Clutter in the reference frame, masked on the reference side (`in_msk`,
    // tested at the interpolated source position).
    let mut reference = make_image(w, h, 0.0, 0.0);
    let moving = make_image(w, h, want.0, want.1);
    block.paint(&mut reference);
    let got = translation(&register_masked(
        &reference,
        Some(&mask),
        &moving,
        None,
        Transformation::Translation,
    ));
    assert!(err(got, want) < 0.05, "reference-side mask: got {got:?}");
}

/// Masks reach the coarse pyramid levels, not just the finest one. A mask that
/// only bit at full resolution would let the coarse levels converge onto the
/// clutter, and the fine levels could not climb back out.
#[test]
fn masks_apply_at_every_pyramid_level() {
    // 160x160 gives depth 3, so two decimated mask levels are built and used.
    let (w, h) = (160, 160);
    let want = (3.0, -2.0);
    let block = Block {
        x0: 100,
        y0: 100,
        size: 40,
    };
    let mut reference = make_image(w, h, 0.0, 0.0);
    let mut moving = make_image(w, h, want.0, want.1);
    block.paint(&mut reference);
    block.paint(&mut moving);
    let mask = block.mask(w, h);

    let masked = translation(&register_masked(
        &reference,
        Some(&mask),
        &moving,
        Some(&mask),
        Transformation::Translation,
    ));
    let unmasked = translation(&register(&reference, &moving, Transformation::Translation));
    assert!(
        err(masked, want) < 0.05,
        "got {masked:?}, want {want:?} (unmasked {unmasked:?})"
    );
}

#[test]
#[should_panic(expected = "mask is 32x32, but its image is 64x64")]
fn mask_dimensions_must_match_the_image() {
    let (w, h) = (64, 64);
    let reference = make_image(w, h, 0.0, 0.0);
    let moving = make_image(w, h, 3.0, -2.0);
    let wrong = all_true(32, 32);
    register_masked(
        &reference,
        Some(&wrong),
        &moving,
        None,
        Transformation::Translation,
    );
}

/// `StackReg` crops the last row and column off every frame; masks must be
/// cropped with them or they would sit a pixel off, and silently so.
#[test]
fn stack_registration_accepts_per_frame_masks() {
    let (w, h) = (64, 64);
    let block = Block {
        x0: 40,
        y0: 40,
        size: 16,
    };
    let shifts = [(0.0, 0.0), (2.0, -1.0), (4.0, -2.0)];
    let frames: Vec<Image2D> = shifts
        .iter()
        .map(|&(dx, dy)| {
            let mut f = make_image(w, h, dx, dy);
            block.paint(&mut f);
            f
        })
        .collect();
    let masks: Vec<Mask> = shifts.iter().map(|_| block.mask(w, h)).collect();

    let mut sr = StackReg::new(Transformation::Translation);
    let tmats = sr.register_stack_masked(&frames, Some(&masks), Reference::First, 1, 1);

    assert_eq!(tmats.len(), frames.len());
    for (i, &want) in shifts.iter().enumerate() {
        let got = (tmats[i][(0, 2)], tmats[i][(1, 2)]);
        assert!(
            err(got, want) < 0.1,
            "frame {i}: got {got:?}, want {want:?}"
        );
    }
}
