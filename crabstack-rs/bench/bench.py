"""
Benchmark: crabstack (Rust) vs pystackreg (C++) for TurboReg registration.

Both backends expose the same native API (`_register`, `_transform`), so this
compares the native compute directly and also end-to-end through pystackreg's
own `StackReg` orchestration (by swapping the backend module).

Run inside a venv with numpy, pystackreg and crabstack installed:
    python bench.py
"""

import statistics
import time

import numpy as np
from pystackreg import turboreg as cpp
import crabstack as rs
import pystackreg.pystackreg as psr
from pystackreg import StackReg

TF = {
    "TRANSLATION": 2,
    "RIGID_BODY": 3,
    "SCALED_ROTATION": 4,
    "AFFINE": 6,
    "BILINEAR": 8,
}


def make_pair(w, h, seed=0):
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    base = (
        128.0
        + 60.0 * np.sin(xx * 0.08) * np.cos(yy * 0.06)
        + 40.0 * np.exp(-(((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (2 * (w / 6) ** 2)))
        + rng.rand(h, w) * 8.0
    )
    mov = np.roll(np.roll(base, 3, axis=0), -2, axis=1) + rng.rand(h, w) * 4.0
    return base, mov


def timeit(fn, repeats=7):
    fn()  # warm-up
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def bench_register():
    print("\n== _register (single image, median of 7) ==")
    print(f"{'transform':16s} {'size':>10s} {'pystackreg':>12s} {'crabstack':>12s} {'speedup':>9s}")
    for (w, h) in [(128, 128), (256, 256), (512, 512)]:
        ref, mov = make_pair(w, h)
        rc, mc = ref[:-1, :-1].copy(), mov[:-1, :-1].copy()
        for name, code in TF.items():
            t_cpp = timeit(lambda: cpp._register(rc, mc, code))
            t_rs = timeit(lambda: rs._register(rc, mc, code))
            print(
                f"{name:16s} {f'{w}x{h}':>10s} {t_cpp*1e3:10.2f}ms {t_rs*1e3:10.2f}ms {t_cpp/t_rs:8.2f}x"
            )


def bench_transform():
    print("\n== _transform (single image, median of 7) ==")
    print(f"{'transform':16s} {'size':>10s} {'pystackreg':>12s} {'crabstack':>12s} {'speedup':>9s}")
    for (w, h) in [(256, 256), (512, 512)]:
        ref, mov = make_pair(w, h)
        for name, code in TF.items():
            m, _, _ = cpp._register(ref[:-1, :-1].copy(), mov[:-1, :-1].copy(), code)
            t_cpp = timeit(lambda: cpp._transform(mov, m))
            t_rs = timeit(lambda: rs._transform(mov, m))
            print(
                f"{name:16s} {f'{w}x{h}':>10s} {t_cpp*1e3:10.2f}ms {t_rs*1e3:10.2f}ms {t_cpp/t_rs:8.2f}x"
            )


def bench_stack():
    print("\n== StackReg.register_transform_stack end-to-end (identical Python, swapped backend) ==")
    n_frames, w, h = 20, 256, 256
    rng = np.random.RandomState(7)
    base, _ = make_pair(w, h)
    stack = np.stack(
        [np.roll(np.roll(base, int(i * 0.7), 0), -int(i * 0.5), 1) + rng.rand(h, w) * 4 for i in range(n_frames)]
    )
    print(f"{'transform':16s} {'stack':>12s} {'pystackreg':>12s} {'crabstack':>12s} {'speedup':>9s}")
    for name in ["TRANSLATION", "RIGID_BODY", "AFFINE"]:
        code = getattr(StackReg, name)

        def run_cpp():
            psr.turboreg = cpp
            StackReg(code).register_transform_stack(stack, reference="first")

        def run_rs():
            psr.turboreg = rs
            StackReg(code).register_transform_stack(stack, reference="first")

        t_cpp = timeit(run_cpp, repeats=3)
        t_rs = timeit(run_rs, repeats=3)
        psr.turboreg = cpp
        print(
            f"{name:16s} {f'{n_frames}x{w}x{h}':>12s} {t_cpp*1e3:10.1f}ms {t_rs*1e3:10.1f}ms {t_cpp/t_rs:8.2f}x"
        )


if __name__ == "__main__":
    print("numpy", np.__version__)
    bench_register()
    bench_transform()
    bench_stack()
