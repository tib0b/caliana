# crabstack

A dependency-free Rust implementation of subpixel image registration by the
Thévenaz–Ruttimann–Unser pyramid method (IEEE Trans. Image Processing, 1998),
originally ported from [pystackreg](https://github.com/glichtner/pystackreg) /
**TurboReg**.

It implements the same algorithm as the C++ TurboReg core that pystackreg wraps,
including its single-precision (`float`) rounding inside the B-spline pyramids
and warp output — but the per-pixel work is vectorised and multi-threaded, so it
runs roughly **5× faster on registration and 9-17× faster on warping** than the
C++ (see [Benchmark](#benchmark)).

### Relationship to pystackreg

crabstack began as a faithful transcription and is verified against the original
C++ (see [Status](#status)), but it is no longer bound by it: it is the
registration backend for [caliana], and its API follows what caliana needs.
Where the two diverge — [masking](#masking) is the first case — the parity tests
still pin the unmasked path pystackreg exercised, so existing results and stored
matrices stay valid.

crabstack is not a separate distribution: this crate lives inside the caliana
repository and is compiled into that package as `caliana._crabstack`, so the two
version and ship together. The crate itself stays a normal Rust crate — `cargo
test`, `cargo bench` and the C++ oracle all run from this directory.

[caliana]: ..

## Status

Verified against the original C++, built from the unmodified pystackreg sources
by the same toolchain, across image sizes `57×43`, `130×100`, `96×96` (pyramid
depths 2–3):

| Model            | register (matrix) | transform (image) |
|------------------|-------------------|-------------------|
| Translation      | ≤ 1e-13           | ✅ bit-exact      |
| Rigid body       | ≤ 1e-13           | ✅ bit-exact      |
| Scaled rotation  | ≤ 1e-13           | ✅ bit-exact      |
| Affine           | ≤ 1e-13           | ✅ bit-exact      |
| Bilinear         | ≤ 1e-13           | ✅ bit-exact      |

Warps are bit-identical to the reference. Registration matrices are not: the
optimizer's reductions are summed per row band rather than in one sequential
pass, which reassociates the sums. The observed divergence (`max|Δmatrix|`) is
at most `3e-13` — far below the `1e-6` the differential tests enforce, and some
nine orders of magnitude below the sub-pixel precision the optimizer targets.

### Determinism

Results do **not** depend on the thread count or the machine. Work is split into
a fixed number of row bands derived from the image size alone, and band partials
are always reduced in band order, so a 4-core and a 64-core run produce
bit-identical output — as does `crabstack::set_num_threads(1)`. This is pinned
by the `results_are_independent_of_thread_count` test.

## API

```rust
use crabstack::{Image2D, StackReg, Transformation};

// Low-level, mirrors turboreg._register / ._transform:
let reg = crabstack::register(&reference, &moving, Transformation::RigidBody);
let aligned = crabstack::transform(&moving, &reg.matrix);

// High-level, mirrors the Python StackReg class:
let mut sr = StackReg::new(Transformation::Affine);
let long_matrix = sr.register(&reference, &moving);   // 3×3 (4×4 for bilinear)
let aligned = sr.transform(&moving, None);

// Movie/stack registration (frames along the leading axis):
use crabstack::stack::Reference;
let aligned_stack = sr.register_transform_stack(&frames, Reference::Previous, 1, 1);
```

`Image2D` is row-major with `width` columns and `height` rows
(`data[row * width + col]`), matching numpy C-order.

## Masking

A `Mask` says which pixels may take part in a fit — `true` participates. This is
an extension over pystackreg, whose C++ core always registered on every pixel
(`TurboRegMask` existed, but `clearMask()` filled it with ones and nothing could
change that). Use it to stop static background, saturated regions, or a
neighbouring object from dragging the estimate around.

```rust
use crabstack::{register_masked, Mask, Transformation};

let mask = Mask::new(w, h, keep);          // Vec<bool>, one per pixel
let reg = register_masked(&reference, Some(&mask), &moving, Some(&mask),
                          Transformation::Affine);
```

- Either side may be `None`, which keeps every pixel there. A pixel counts only
  where **both** frames keep it, so masking one side is enough to drop it — mask
  the frame the artefact is actually in.
- `register(a, b, t)` is exactly `register_masked(a, None, b, None, t)`, and
  costs the same: an absent mask allocates nothing and the per-pixel test stays
  out of the loop.
- **Leave a margin.** Cut masks a couple of pixels wider than the thing you are
  excluding. The cubic B-spline interpolation and the pyramid decimation each
  spread a feature about two pixels past its edge, so a mask cut exactly to a
  bright artefact still leaves contaminated pixels in the fit — worth about a
  third of a pixel of translation error in `tests/mask.rs`. Two pixels of
  surround removes it.
- Inside the pyramid the mask is dilated for you, one coarse pixel per level, so
  a tight mask still exists at the coarsest scales rather than vanishing.
- `StackReg::register_masked` and `register_stack_masked` take masks at full
  frame size and crop them alongside their frames. For an averaged reference
  (`Reference::First` / `Mean`) the reference mask is the intersection of the
  averaged frames' masks, since a pixel some frame masked out has already
  polluted that average.

## Layout

| File                    | Ports from pystackreg            |
|-------------------------|----------------------------------|
| `src/matrix.rs`         | `inc/matrix.h`                   |
| `src/image.rs`          | `TurboRegImage.{h,cpp}`          |
| `src/mask.rs`           | `TurboRegMask.{h,cpp}` (rewritten — booleans, and actually reachable) |
| `src/point_handler.rs`  | `TurboRegPointHandler.{h,cpp}`   |
| `src/transform.rs`      | `TurboRegTransform.{h,cpp}`      |
| `src/lib.rs`            | `TurboReg.{h,cpp}` + `pymain.cpp` boundary |
| `src/stack.rs`          | `pystackreg/pystackreg.py`       |
| `src/kernel.rs`         | — (B-spline sampling kernel)     |
| `src/parallel.rs`       | — (worker pool, SIMD dispatch)   |

## Testing / parity harness

Pure-Rust tests run with a plain `cargo test`. The differential tests
additionally build a small C++ oracle (`oracle/oracle.cpp`) from the original
pystackreg sources — no Python or numpy required:

```sh
./tests/build_oracle.sh   # produces oracle/oracle
cargo test --release      # runs pure-Rust + differential tests
```

If the oracle binary is absent, the differential tests skip themselves (with a
notice) so the pure-Rust checks still run.

## Python module

The crate also builds a Python extension (via [maturin]) exposing the same native
API as pystackreg's `turboreg` C++ extension — `_register(ref, mov, transformation)`
and `_transform(mov, matrix)` — so it is a drop-in backend, plus the optional
mask arguments pystackreg never had:

```sh
pip install maturin numpy
maturin develop --release          # builds & installs `crabstack`
```

```python
import numpy as np, crabstack
mat, ref_pts, mov_pts = crabstack._register(ref[:-1, :-1], mov[:-1, :-1], crabstack.RIGID_BODY)
aligned = crabstack._transform(mov, mat)

# Restricted to the pixels a boolean mask keeps (either may be None):
mat, _, _ = crabstack._register(
    ref[:-1, :-1], mov[:-1, :-1], crabstack.RIGID_BODY,
    ref_mask[:-1, :-1], mov_mask[:-1, :-1],
)

# Or reuse pystackreg's StackReg orchestration with the Rust backend:
import pystackreg.pystackreg as psr
psr.turboreg = crabstack
from pystackreg import StackReg
aligned_stack = StackReg(StackReg.RIGID_BODY).register_transform_stack(stack, reference="first")
```

### Equivalence vs the pystackreg wheel

Measured against the distributed pystackreg wheel at 256×256 and 512×512:

- **Warped images are bit-identical** (`0` of 262 144 pixels differ) when both
  backends are given the same matrix.
- **Registration matrices** agree to `≤ 1e-8` absolute.
- **End to end** — each backend registering *and* warping with its own
  matrix — outputs differ by at most one `f32` ulp (`1.5e-5` on pixel values
  around 128, i.e. `0.00002%` of the data range), and for several models not at
  all.

### Benchmark

`python bench/bench.py` (numpy 2.5, CPython 3.14, release build, AMD Ryzen 5
PRO 4650U — 6 cores / 12 threads). Median wall time; speedup = pystackreg ÷
crabstack:

| Operation | Size | pystackreg (C++) | crabstack (Rust) | speedup |
|---|---|---|---|---|
| `_register` (translation)  | 512×512 |  89.1 ms | 22.0 ms |  4.06× |
| `_register` (rigid body)   | 512×512 | 168.8 ms | 32.6 ms |  5.18× |
| `_register` (scaled rot.)  | 512×512 | 164.3 ms | 34.3 ms |  4.79× |
| `_register` (affine)       | 512×512 | 143.8 ms | 29.5 ms |  4.88× |
| `_register` (bilinear)     | 512×512 | 198.7 ms | 31.8 ms |  6.24× |
| `_register` (rigid body)   | 256×256 |  51.5 ms | 21.2 ms |  2.43× |
| `_register` (rigid body)   | 128×128 |  12.1 ms |  9.3 ms |  1.30× |
| `_transform` (warp)        | 512×512 |  58.3 ms |  3.5 ms | 16.61× |
| `_transform` (warp)        | 256×256 |  11.2 ms |  1.2 ms |  9.09× |
| `register_transform_stack` (20×256×256, rigid)  | — | 1067 ms | 332 ms | 3.22× |
| `register_transform_stack` (20×256×256, affine) | — | 1140 ms | 342 ms | 3.33× |

Where the time goes, and why the speedup varies:

- **Warping** gains the most (9–17×). It is embarrassingly parallel, and
  `transform` no longer builds the multiresolution pyramid and xy-gradients that
  the C++ constructs and then never reads.
- **Registration** gains 4–6× at 512×512. The optimizer's mean-squares /
  gradient / Hessian passes parallelise cleanly, but the coarse pyramid levels
  are too small to distribute and run serially — classic Amdahl, which is also
  why 128×128 only reaches ~1.3×.
- **Stack processing** gains ~3.2×; the remainder is pystackreg's own Python and
  numpy orchestration, which is unchanged.

[maturin]: https://github.com/PyO3/maturin

## Threading

Parallelism is on by default, sized from `std::thread::available_parallelism()`.
To change it:

```rust
crabstack::set_num_threads(1);  // serial; 0 restores the default
```

or set `CRABSTACK_NUM_THREADS` in the environment. The same call is exposed to
Python as `crabstack.set_num_threads(n)`. Use `1` when driving crabstack from
your own worker pool — output is unaffected either way.

Nested and concurrent use is safe: a second thread that starts a registration
while one is already running simply executes its own work inline rather than
queueing behind it.

## Notes on fidelity

- The `(float)` casts in `putRow`/`putColumn` and warp output are preserved as
  `x as f32 as f64` (`f32_round`). They cost essentially nothing and are what
  keeps warp output interchangeable with pystackreg.
- Landmark placement uses the default (non-`TURBOREG_MODE`) golden-ratio branch,
  which is the one pystackreg compiles.
- Floating-point sums are reassociated relative to the C++: the per-pixel loops
  accumulate per row band and are contracted into FMAs where the CPU supports
  them. This is the source of the `1e-13` matrix divergence, and it makes the
  intermediate arithmetic slightly *more* accurate, not less.
- The C++ index/fraction quirk at exact negative coordinates (`x as i32`
  truncation rather than `floor`) is reproduced deliberately — see
  `kernel::base_index`.
- The mask pyramid's sum-of-absolute-values reduction is now a boolean "any",
  which is exactly equivalent — every term is non-negative, so the sum is
  non-zero precisely when some term is, and non-zero is all it was ever tested
  for. `mask::tests::half_mask_matches_legacy_reduction` pins the new reduction
  against a copy of the original pointer walk.

## License

The algorithm derives from Philippe Thévenaz's TurboReg and pystackreg's C++
port (BSD-3-Clause). See the upstream projects for citation requirements.
