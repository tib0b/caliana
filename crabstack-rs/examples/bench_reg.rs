//! Fast, Python-free harness: times `register`, `transform` and stack
//! registration per model/size so hot-path changes can be measured without
//! rebuilding the Python extension. Correctness is gated separately by
//! `cargo test --test parity`, which diffs against the C++ oracle.
//!
//! For the head-to-head against pystackreg itself, use `bench/bench.py`.
//!
//! Run: `cargo run --release --example bench_reg`

use std::time::Instant;

use crabstack::{register, transform, Image2D, Transformation};

fn make_pair(w: usize, h: usize) -> (Image2D, Image2D) {
    // Deterministic, pystackreg-bench-like synthetic pair (no numpy needed).
    let mut base = vec![0.0f64; w * h];
    let mut seed = 0x9e3779b9u64;
    let mut rng = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    for y in 0..h {
        for x in 0..w {
            let xx = x as f64;
            let yy = y as f64;
            base[y * w + x] = 128.0
                + 60.0 * (xx * 0.08).sin() * (yy * 0.06).cos()
                + 40.0
                    * (-(((xx - w as f64 / 2.0).powi(2) + (yy - h as f64 / 2.0).powi(2))
                        / (2.0 * (w as f64 / 6.0).powi(2))))
                    .exp()
                + rng() * 8.0;
        }
    }
    // Shift by (3, -2) and add noise, like np.roll in the python bench.
    let mut mov = vec![0.0f64; w * h];
    for y in 0..h {
        for x in 0..w {
            let sy = (y + h - 3) % h;
            let sx = (x + 2) % w;
            mov[y * w + x] = base[sy * w + sx] + rng() * 4.0;
        }
    }
    (Image2D::new(w, h, base), Image2D::new(w, h, mov))
}

/// A movie whose frames each drift by one more pixel, so every registration has
/// real work to do. (Reusing `make_pair` here would hand the optimizer twenty
/// identical frames, which converge on the first iteration and make the stack
/// timing meaningless.)
fn make_movie(w: usize, h: usize, frames: usize) -> Vec<Image2D> {
    let (base, _) = make_pair(w, h);
    (0..frames)
        .map(|i| {
            let mut f = vec![0.0f64; w * h];
            for y in 0..h {
                for x in 0..w {
                    let sy = (y + h - i) % h;
                    let sx = (x + 2 * i) % w;
                    f[y * w + x] = base.data[sy * w + sx];
                }
            }
            Image2D::new(w, h, f)
        })
        .collect()
}

/// Mirror pystackreg's `img[:-1, :-1]` crop (the register inputs).
fn crop(img: &Image2D) -> Image2D {
    let (w, h) = (img.width - 1, img.height - 1);
    let mut data = Vec::with_capacity(w * h);
    for row in 0..h {
        let base = row * img.width;
        data.extend_from_slice(&img.data[base..base + w]);
    }
    Image2D::new(w, h, data)
}

fn timeit(mut f: impl FnMut(), repeats: u32) -> f64 {
    f(); // warm-up
    let mut best = f64::INFINITY;
    for _ in 0..repeats {
        let t0 = Instant::now();
        f();
        best = best.min(t0.elapsed().as_secs_f64());
    }
    best
}

fn main() {
    let models = [
        ("TRANSLATION", Transformation::Translation),
        ("RIGID_BODY", Transformation::RigidBody),
        ("SCALED_ROTATION", Transformation::ScaledRotation),
        ("AFFINE", Transformation::Affine),
        ("BILINEAR", Transformation::Bilinear),
    ];

    println!("== register (min of N) ==");
    println!("{:16} {:>10} {:>12}", "model", "size", "register");
    for &(w, h) in &[(128usize, 128usize), (256, 256), (512, 512)] {
        let (r, m) = make_pair(w, h);
        let rc = crop(&r);
        let mc = crop(&m);
        let reps = if w >= 512 { 3 } else { 7 };
        for (name, t) in models {
            let dt = timeit(
                || {
                    let _ = register(&rc, &mc, t);
                },
                reps,
            );
            println!(
                "{:16} {:>10} {:>10.3}ms",
                name,
                format!("{w}x{h}"),
                dt * 1e3
            );
        }
    }

    println!("\n== register_transform_stack (min of N) ==");
    {
        use crabstack::stack::Reference;
        use crabstack::StackReg;
        let (w, h) = (256usize, 256usize);
        let frames = make_movie(w, h, 20);
        // pystackreg (and this port) reject bilinear with `Previous`, since
        // bilinear transforms do not compose.
        for (name, t) in models
            .into_iter()
            .filter(|&(_, t)| t != Transformation::Bilinear)
        {
            let dt = timeit(
                || {
                    let mut sr = StackReg::new(t);
                    let _ = sr.register_transform_stack(&frames, Reference::Previous, 1, 1);
                },
                2,
            );
            println!("{:16} {:>10} {:>10.3}ms", name, "20x256x256", dt * 1e3);
        }
    }

    println!("\n== transform (min of N) ==");
    println!("{:16} {:>10} {:>12}", "model", "size", "transform");
    for &(w, h) in &[(256usize, 256usize), (512, 512)] {
        let (r, m) = make_pair(w, h);
        let rc = crop(&r);
        let mc = crop(&m);
        let reps = if w >= 512 { 4 } else { 8 };
        for (name, t) in models {
            let reg = register(&rc, &mc, t);
            let dt = timeit(
                || {
                    let _ = transform(&m, &reg.matrix);
                },
                reps,
            );
            println!(
                "{:16} {:>10} {:>10.3}ms",
                name,
                format!("{w}x{h}"),
                dt * 1e3
            );
        }
    }
}
