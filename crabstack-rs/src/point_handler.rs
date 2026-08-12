//! Port of `TurboRegPointHandler::setPointsByTransformation` (default, non
//! `TURBOREG_MODE` branch — the one pystackreg compiles).

use crate::matrix::Mat;
use crate::transform_kind::*;

fn golden_ratio() -> f64 {
    0.5 * (5.0_f64.sqrt() - 1.0)
}

/// Initial landmark positions. `width`/`height` are the cropped image dims; the
/// original adds 1 to each because pystackreg passes images cropped by 1px.
pub fn points_by_transformation(width: usize, height: usize, transformation: i32) -> Mat {
    let width = (width + 1) as f64;
    let height = (height + 1) as f64;
    let gr = golden_ratio();

    match transformation {
        TRANSLATION => {
            let mut p = Mat::new(1, 2);
            p[(0, 0)] = (width / 2.0).floor();
            p[(0, 1)] = (height / 2.0).floor();
            p
        }
        RIGID_BODY => {
            let mut p = Mat::new(3, 2);
            p[(0, 0)] = (width / 2.0).floor();
            p[(0, 1)] = (height / 2.0).floor();
            p[(1, 0)] = (width / 2.0).floor();
            p[(1, 1)] = (height / 4.0).floor();
            p[(2, 0)] = (width / 2.0).floor();
            p[(2, 1)] = (3.0 * height / 4.0).floor();
            p
        }
        SCALED_ROTATION => {
            let mut p = Mat::new(2, 2);
            p[(0, 0)] = (width / 4.0).floor();
            p[(0, 1)] = (height / 2.0).floor();
            p[(1, 0)] = (3.0 * width / 4.0).floor();
            p[(1, 1)] = (height / 2.0).floor();
            p
        }
        AFFINE => {
            let mut p = Mat::new(3, 2);
            p[(0, 0)] = (width / 2.0).floor();
            p[(0, 1)] = (height / 4.0).floor();
            p[(1, 0)] = (width / 4.0).floor();
            p[(1, 1)] = (3.0 * height / 4.0).floor();
            p[(2, 0)] = (3.0 * width / 4.0).floor();
            p[(2, 1)] = (3.0 * height / 4.0).floor();
            p
        }
        BILINEAR => {
            let mut p = Mat::new(4, 2);
            p[(0, 0)] = (0.25 * gr * width).floor();
            p[(0, 1)] = (0.25 * gr * height).floor();
            p[(1, 0)] = (0.25 * gr * width).floor();
            p[(1, 1)] = height - (0.25 * gr * height).ceil();
            p[(2, 0)] = width - (0.25 * gr * width).ceil();
            p[(2, 1)] = (0.25 * gr * height).floor();
            p[(3, 0)] = width - (0.25 * gr * width).ceil();
            p[(3, 1)] = height - (0.25 * gr * height).ceil();
            p
        }
        _ => Mat::new(0, 0),
    }
}
