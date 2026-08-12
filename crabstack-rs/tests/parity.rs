//! Differential tests: run the same inputs through the Rust port and the
//! original C++ TurboReg core (`oracle/oracle`) and compare.
//!
//! The oracle is built by `tests/build_oracle.sh` (or the command in
//! `oracle/oracle.cpp`). If the binary is missing the differential tests are
//! skipped with a warning so `cargo test` still exercises the pure-Rust checks.

use crabstack::{
    register, set_simd_enabled, simd_enabled, transform, Image2D, Mat, Transformation,
};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;

/// The oracle comparisons run down both kernel paths, because `has_simd()`
/// chooses one per CPU and the other then never executes — while the wheels we
/// publish run on both. Whichever path this machine would not have taken is the
/// one shipping untested, and the bug it would hide is not a rounding
/// difference: it is a wrong lane or a mishandled remainder in the tail
/// elements, which lands far above these tolerances.
const KERNEL_PATHS: &[(&str, bool)] = &[("simd", true), ("scalar", false)];

/// `set_simd_enabled` is process-wide, so the two tests that sweep the paths
/// must not interleave — one would flip the flag out from under the other and
/// silently test the same path twice. Poisoning is ignored deliberately: if a
/// path assertion has already failed, that failure is the useful message.
static PATH_SWEEP: Mutex<()> = Mutex::new(());

fn sweep_paths(mut body: impl FnMut(&str)) {
    let _guard = PATH_SWEEP.lock().unwrap_or_else(|e| e.into_inner());
    for &(name, simd) in KERNEL_PATHS {
        set_simd_enabled(simd);
        // On a CPU without AVX2 both entries select the scalar kernels; running
        // it twice would just be slower, and would report coverage that is not
        // there.
        if simd && !simd_enabled() {
            eprintln!("no AVX2+FMA on this CPU: the scalar path is the only one");
            continue;
        }
        body(name);
    }
    set_simd_enabled(true);
}

fn oracle_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("oracle/oracle")
}

fn tmp(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}

fn write_f64(path: &PathBuf, data: &[f64]) {
    let mut f = std::fs::File::create(path).unwrap();
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    f.write_all(&bytes).unwrap();
}

fn read_f64(path: &PathBuf, n: usize) -> Vec<f64> {
    let mut f = std::fs::File::open(path).unwrap();
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();
    assert_eq!(buf.len(), n * 8, "unexpected oracle output size");
    buf.chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

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

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn oracle_available() -> bool {
    if oracle_path().exists() {
        return true;
    }
    eprintln!(
        "SKIP differential test: oracle binary not found at {}. \
         Build it with tests/build_oracle.sh",
        oracle_path().display()
    );
    false
}

fn run_oracle(args: &[String]) {
    let status = Command::new(oracle_path()).args(args).status().unwrap();
    assert!(status.success(), "oracle failed: {args:?}");
}

fn oracle_register(reference: &Image2D, moving: &Image2D, code: i32, ncols: usize) -> Mat {
    let refp = tmp("ref.bin");
    let movp = tmp("mov.bin");
    let outp = tmp("mat.bin");
    write_f64(&refp, &reference.data);
    write_f64(&movp, &moving.data);
    run_oracle(&[
        "register".into(),
        reference.width.to_string(),
        reference.height.to_string(),
        code.to_string(),
        refp.to_string_lossy().into(),
        movp.to_string_lossy().into(),
        outp.to_string_lossy().into(),
    ]);
    Mat::from_rows(2, ncols, read_f64(&outp, 2 * ncols))
}

fn oracle_transform(moving: &Image2D, m: &Mat) -> Image2D {
    let movp = tmp("tmov.bin");
    let matp = tmp("tmat.bin");
    let outp = tmp("timg.bin");
    write_f64(&movp, &moving.data);
    write_f64(&matp, &m.data);
    run_oracle(&[
        "transform".into(),
        moving.width.to_string(),
        moving.height.to_string(),
        m.cols.to_string(),
        movp.to_string_lossy().into(),
        matp.to_string_lossy().into(),
        outp.to_string_lossy().into(),
    ]);
    Image2D::new(
        moving.width,
        moving.height,
        read_f64(&outp, moving.width * moving.height),
    )
}

const CASES: &[(Transformation, i32, usize)] = &[
    (Transformation::Translation, 2, 1),
    (Transformation::RigidBody, 3, 3),
    (Transformation::ScaledRotation, 4, 3),
    (Transformation::Affine, 6, 3),
    (Transformation::Bilinear, 8, 4),
];

// A spread of sizes: odd/non-square (odd-width mask path, depth 2), a large one
// (depth 3), and a square power-of-two-ish case.
const SIZES: &[(usize, usize)] = &[(57, 43), (130, 100), (96, 96)];

#[test]
fn register_matches_cpp_oracle() {
    if !oracle_available() {
        return;
    }
    sweep_paths(|path| {
        for &(w, h) in SIZES {
            let reference = make_image(w, h, 0.0, 0.0);
            let moving = make_image(w, h, 1.7, -1.2);
            for &(t, code, ncols) in CASES {
                let rust = register(&reference, &moving, t);
                let cpp = oracle_register(&reference, &moving, code, ncols);
                let d = max_abs_diff(&rust.matrix.data, &cpp.data);
                println!("register {t:?} {w}x{h} [{path}]: max|Δmatrix| = {d:.3e}");
                assert!(
                    d < 1e-6,
                    "register matrix diverged for {t:?} {w}x{h} on the {path} path: {d:.3e}"
                );
            }
        }
    });
}

#[test]
fn transform_matches_cpp_oracle() {
    if !oracle_available() {
        return;
    }
    sweep_paths(|path| {
        for &(w, h) in SIZES {
            let reference = make_image(w, h, 0.0, 0.0);
            let moving = make_image(w, h, 1.7, -1.2);
            for &(t, _code, _ncols) in CASES {
                // Use the Rust-derived matrix as the (identical) input to both warps,
                // isolating interpolation parity from optimizer parity.
                let m = register(&reference, &moving, t).matrix;
                let rust_img = transform(&moving, &m);
                let cpp_img = oracle_transform(&moving, &m);
                let d = max_abs_diff(&rust_img.data, &cpp_img.data);
                println!("transform {t:?} {w}x{h} [{path}]: max|Δpixel| = {d:.3e}");
                assert!(
                    d < 1e-6,
                    "transform image diverged for {t:?} {w}x{h} on the {path} path: {d:.3e}"
                );
            }
        }
    });
}

/* ---------- pure-Rust sanity (run even without the oracle) ---------- */

#[test]
fn identity_registration_is_near_zero_translation() {
    let (w, h) = (48, 48);
    let img = make_image(w, h, 0.0, 0.0);
    let r = register(&img, &img, Transformation::Translation);
    // Registering an image to itself should yield ~zero displacement.
    assert!(r.matrix[(0, 0)].abs() < 1e-3, "dx = {}", r.matrix[(0, 0)]);
    assert!(r.matrix[(1, 0)].abs() < 1e-3, "dy = {}", r.matrix[(1, 0)]);
}

#[test]
fn recovers_known_translation() {
    let (w, h) = (64, 64);
    let reference = make_image(w, h, 0.0, 0.0);
    let moving = make_image(w, h, 3.0, -2.0);
    let r = register(&reference, &moving, Transformation::Translation);
    // The short translation matrix maps target->source landmark offset; it should
    // be close to the injected (3, -2) shift.
    assert!(
        (r.matrix[(0, 0)] - 3.0).abs() < 0.25,
        "dx = {}",
        r.matrix[(0, 0)]
    );
    assert!(
        (r.matrix[(1, 0)] + 2.0).abs() < 0.25,
        "dy = {}",
        r.matrix[(1, 0)]
    );
}

#[test]
fn transform_with_identity_returns_input_interior() {
    let (w, h) = (48, 48);
    let img = make_image(w, h, 0.0, 0.0);
    // Identity translation (short 2x1 of zeros).
    let m = Mat::from_rows(2, 1, vec![0.0, 0.0]);
    let out = transform(&img, &m);
    // Interior pixels (away from the 2px spline border) should round-trip closely.
    let mut worst = 0.0f64;
    for row in 3..h - 3 {
        for col in 3..w - 3 {
            let d = (out.data[row * w + col] - img.data[row * w + col]).abs();
            worst = worst.max(d);
        }
    }
    assert!(worst < 1e-3, "interior round-trip error {worst}");
}

/// The work split is fixed by image size, never by thread count, and band
/// partials are reduced in band order — so serial and parallel execution must
/// agree *exactly*, not just closely. This is what makes results reproducible
/// across machines.
#[test]
fn results_are_independent_of_thread_count() {
    let (w, h) = (130, 100);
    let reference = make_image(w, h, 0.0, 0.0);
    let moving = make_image(w, h, 1.7, -1.2);

    for &(t, _code, _ncols) in CASES {
        crabstack::set_num_threads(0); // default: parallel
        let parallel = register(&reference, &moving, t);
        let warp_parallel = transform(&moving, &parallel.matrix);

        crabstack::set_num_threads(1); // serial
        let serial = register(&reference, &moving, t);
        let warp_serial = transform(&moving, &serial.matrix);

        crabstack::set_num_threads(0);

        assert_eq!(
            parallel.matrix.data, serial.matrix.data,
            "{t:?}: matrix differs between serial and parallel execution"
        );
        assert_eq!(
            warp_parallel.data, warp_serial.data,
            "{t:?}: warp differs between serial and parallel execution"
        );
    }
}

#[test]
fn stack_registration_aligns_frames_to_reference() {
    use crabstack::stack::Reference;
    use crabstack::StackReg;

    // A 5-frame movie where each frame is progressively shifted.
    let (w, h) = (64, 64);
    let frames: Vec<Image2D> = (0..5)
        .map(|i| make_image(w, h, i as f64 * 0.8, -(i as f64) * 0.5))
        .collect();
    let reference = &frames[0];

    let mut sr = StackReg::new(Transformation::Translation);
    let aligned = sr.register_transform_stack(&frames, Reference::First, 1, 1);
    assert_eq!(aligned.len(), frames.len());

    // Each shifted frame should match the reference far better after alignment;
    // frame 0 is the reference itself, so it should merely stay near-identity.
    for (i, out) in aligned.iter().enumerate() {
        let mut aligned_err = 0.0f64;
        let mut raw_err = 0.0f64;
        for row in 6..h - 6 {
            for col in 6..w - 6 {
                let idx = row * w + col;
                aligned_err = aligned_err.max((out.data[idx] - reference.data[idx]).abs());
                raw_err = raw_err.max((frames[i].data[idx] - reference.data[idx]).abs());
            }
        }
        if i == 0 {
            assert!(
                aligned_err < 1.0,
                "frame 0 round-trip error too large: {aligned_err}"
            );
        } else {
            assert!(
                aligned_err < raw_err * 0.5,
                "frame {i}: alignment did not clearly improve match (aligned {aligned_err}, raw {raw_err})"
            );
        }
    }
}
