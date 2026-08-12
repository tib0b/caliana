//! Data-parallel execution for the pixel loops, plus runtime SIMD dispatch.
//!
//! The crate stays dependency-free, so this is a small persistent worker pool
//! built on `std` only. A pool (rather than `thread::scope` per loop) matters
//! because a single 512×512 registration runs ~600 full-image passes; paying a
//! thread spawn for each would cost more than the work itself.
//!
//! ## Determinism
//!
//! Work is split into a **fixed** number of row bands ([`bands_for_grid`]) that does
//! not depend on the thread count, and floating-point reductions always combine
//! band partials in band order. Two consequences:
//!
//! * results are identical on a 4-core and a 64-core machine;
//! * `set_num_threads(1)` reproduces the multi-threaded result exactly — it
//!   changes only *where* the bands run, never how they are summed.
//!
//! The band split does reassociate sums relative to a plain sequential loop, so
//! results differ from the C++ reference in the last few ulps.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

/// Fixed number of row bands a parallel loop is cut into.
///
/// Chosen to (a) balance well up to ~32 threads, and (b) be a constant so the
/// reduction order — and therefore the output — is machine-independent.
const BANDS: usize = 32;

/// Minimum elements a band must own to be worth dispatching.
///
/// Gating on *area* rather than row count matters: the coarse pyramid levels
/// are as small as 12×12 but are evaluated hundreds of times, so splitting them
/// costs far more in synchronisation than it saves in compute.
const MIN_WORK_PER_BAND: usize = 1024;

/// How many bands a `rows × cols` grid should be split into, dividing along
/// rows. Returns 1 when the work is too small to be worth distributing.
pub(crate) fn bands_for_grid(rows: usize, cols: usize) -> usize {
    let work = rows.saturating_mul(cols.max(1));
    BANDS.min(work / MIN_WORK_PER_BAND).min(rows).max(1)
}

/// Row range `[start, end)` covered by band `b` of `nbands` over `rows` rows.
#[inline]
pub(crate) fn band_range(b: usize, nbands: usize, rows: usize) -> (usize, usize) {
    let base = rows / nbands;
    let extra = rows % nbands;
    // The first `extra` bands take one row more, so the split is exact.
    let start = b * base + b.min(extra);
    let len = base + usize::from(b < extra);
    (start, start + len)
}

/* ---------------- thread-count control ---------------- */

static REQUESTED_THREADS: AtomicUsize = AtomicUsize::new(0);

/// Set the number of threads used by the registration and warp loops.
///
/// `1` disables parallel execution. `0` restores the default (the value of
/// `CRABSTACK_NUM_THREADS`, else the machine's available parallelism).
///
/// Output is unaffected: the reduction order is fixed by [`BANDS`], so any
/// thread count produces bit-identical results. Call before the first
/// registration to size the pool; later calls can still switch execution
/// between serial (`1`) and pooled, but cannot grow a pool that already exists.
pub fn set_num_threads(n: usize) {
    REQUESTED_THREADS.store(n, Ordering::Relaxed);
}

/// The currently configured thread count (`1` means serial).
pub fn num_threads() -> usize {
    match REQUESTED_THREADS.load(Ordering::Relaxed) {
        0 => default_threads(),
        n => n,
    }
}

fn default_threads() -> usize {
    if let Ok(v) = std::env::var("CRABSTACK_NUM_THREADS") {
        if let Ok(n) = v.parse::<usize>() {
            if n > 0 {
                return n;
            }
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/* ---------------- the pool ---------------- */

/// Spins a worker performs before parking on the condvar.
///
/// A registration issues hundreds of parallel regions back to back, and at the
/// coarse pyramid levels a futex sleep/wake round trip costs more than the
/// region's actual work. Spinning first means the common case — the next region
/// arriving within microseconds — never touches the kernel. Roughly 8 µs on
/// contemporary x86 before falling back to a real sleep.
const SPIN_LIMIT: u32 = 500;

type BandFn<'a> = &'a (dyn Fn(usize) + Sync);

struct Job {
    /// Type-erased pointer to the caller's band closure. Valid for exactly as
    /// long as `pending != 0`; the submitter blocks until then.
    func: *const (dyn Fn(usize) + Sync),
    nbands: usize,
    next: AtomicUsize,
    /// Participants (workers + submitter) that have not yet finished.
    pending: AtomicUsize,
    panicked: AtomicUsize,
}

// SAFETY: `func` points at a `Sync` closure owned by the submitter's stack
// frame, which outlives the job (the submitter blocks until `pending == 0`).
unsafe impl Send for Job {}
unsafe impl Sync for Job {}

struct Pool {
    size: usize,
    /// Bumped once per region. Workers poll this without taking any lock.
    epoch: AtomicU64,
    /// The region currently in flight. Only read after observing a new epoch,
    /// and a worker can never miss an epoch — see [`worker`].
    job: AtomicPtr<Job>,
    /// Workers blocked on `work`. Lets the submitter skip lock-and-notify
    /// entirely while everyone is still spinning.
    parked: AtomicUsize,
    /// Set while the submitter is blocked on `done`.
    waiting: AtomicBool,
    shutdown: AtomicBool,
    lock: Mutex<()>,
    work: Condvar,
    done: Condvar,
    /// Held for the duration of a region. There is one job slot, so only one
    /// region may be in flight; see [`for_each_band`] for what happens to a
    /// caller that finds it taken.
    submit: Mutex<()>,
}

static POOL: OnceLock<&'static Pool> = OnceLock::new();

fn pool() -> &'static Pool {
    POOL.get_or_init(|| {
        let size = num_threads().saturating_sub(1);
        let pool: &'static Pool = Box::leak(Box::new(Pool {
            size,
            epoch: AtomicU64::new(0),
            job: AtomicPtr::new(std::ptr::null_mut()),
            parked: AtomicUsize::new(0),
            waiting: AtomicBool::new(false),
            shutdown: AtomicBool::new(false),
            lock: Mutex::new(()),
            work: Condvar::new(),
            done: Condvar::new(),
            submit: Mutex::new(()),
        }));
        for i in 0..size {
            let builder = std::thread::Builder::new().name(format!("crabstack-{i}"));
            // A worker that fails to spawn simply means less parallelism: the
            // submitter counts `size` participants and each one decrements
            // exactly once, so a missing worker would hang. Fail loudly.
            builder
                .spawn(move || worker(pool))
                .expect("crabstack: could not spawn worker thread");
        }
        pool
    })
}

/// Wait until the epoch moves past `seen`. `None` means the pool is shutting
/// down.
fn await_epoch(pool: &Pool, seen: u64) -> Option<u64> {
    for _ in 0..SPIN_LIMIT {
        if pool.shutdown.load(Ordering::Relaxed) {
            return None;
        }
        let e = pool.epoch.load(Ordering::Acquire);
        if e != seen {
            return Some(e);
        }
        std::hint::spin_loop();
    }
    // Spinning did not pay off. Announce the park *before* re-checking under
    // the lock, so a submitter that reads `parked == 0` and skips its notify
    // cannot leave this thread asleep.
    pool.parked.fetch_add(1, Ordering::SeqCst);
    let mut guard = pool.lock.lock().unwrap_or_else(|e| e.into_inner());
    let found = loop {
        if pool.shutdown.load(Ordering::Relaxed) {
            break None;
        }
        let e = pool.epoch.load(Ordering::Acquire);
        if e != seen {
            break Some(e);
        }
        guard = pool.work.wait(guard).unwrap_or_else(|e| e.into_inner());
    };
    drop(guard);
    pool.parked.fetch_sub(1, Ordering::SeqCst);
    found
}

fn worker(pool: &'static Pool) {
    // Starts at 0, the epoch of a pool that has posted nothing, so a worker
    // that is slow to start still picks up the first region.
    //
    // A worker can never *skip* an epoch: the submitter of region N blocks
    // until all `size` workers have decremented its `pending`, and it holds
    // `submit` throughout, so region N+1 cannot exist until every worker has
    // handled N. That is what makes the unsynchronised `job` read below safe —
    // whenever a new epoch is observed, that region is still in flight.
    let mut seen = 0u64;
    while let Some(epoch) = await_epoch(pool, seen) {
        seen = epoch;
        let job = pool.job.load(Ordering::Acquire);
        if job.is_null() {
            continue;
        }
        // SAFETY: the region is in flight (see above), so `job` points at the
        // submitter's live stack frame.
        let job = unsafe { &*job };
        run_bands(job);
        finish(pool, job);
    }
}

fn run_bands(job: &Job) {
    loop {
        let b = job.next.fetch_add(1, Ordering::Relaxed);
        if b >= job.nbands {
            return;
        }
        // SAFETY: `func` stays live until every participant has decremented
        // `pending`, which happens strictly after this call returns.
        let f = unsafe { &*job.func };
        // A panicking band must not strand the submitter waiting on `pending`,
        // so absorb it here and re-raise on the submitter's thread.
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(b))).is_err() {
            job.panicked.store(1, Ordering::Relaxed);
        }
    }
}

fn finish(pool: &Pool, job: &Job) {
    if job.pending.fetch_sub(1, Ordering::AcqRel) == 1 && pool.waiting.load(Ordering::SeqCst) {
        // Take the lock so the submitter cannot miss the notification between
        // its `pending` check and its wait.
        let _g = pool.lock.lock().unwrap_or_else(|e| e.into_inner());
        pool.done.notify_all();
    }
}

/// Run `f(band)` for every band in `0..nbands`, returning once all have
/// completed. Bands may execute in any order and on any thread.
pub(crate) fn for_each_band(nbands: usize, f: BandFn<'_>) {
    let serial = nbands <= 1 || num_threads() <= 1;
    if serial {
        for b in 0..nbands {
            f(b);
        }
        return;
    }
    let pool = pool();
    // Claim the single job slot. A caller that cannot get it — a second
    // application thread registering concurrently, or a band that itself
    // parallelises — runs its own bands inline instead of queueing behind the
    // region in flight. That keeps the pool deadlock-free without making
    // callers coordinate, and the busy region is already using the cores.
    let _slot = match pool.submit.try_lock() {
        Ok(g) => g,
        Err(std::sync::TryLockError::Poisoned(g)) => g.into_inner(),
        Err(std::sync::TryLockError::WouldBlock) => {
            for b in 0..nbands {
                f(b);
            }
            return;
        }
    };
    if pool.size == 0 {
        for b in 0..nbands {
            f(b);
        }
        return;
    }

    // SAFETY: the lifetime is erased only for the duration of the job; the
    // submitter blocks below until `pending` reaches 0, after which no thread
    // holds a reference to `f`.
    let func: *const (dyn Fn(usize) + Sync) = f;
    let func: *const (dyn Fn(usize) + Sync + 'static) = unsafe { std::mem::transmute(func) };
    let job = Job {
        func,
        nbands,
        next: AtomicUsize::new(0),
        pending: AtomicUsize::new(pool.size + 1),
        panicked: AtomicUsize::new(0),
    };

    // Publish the job before the epoch: workers observe the epoch with acquire
    // ordering, so seeing the bump guarantees seeing the pointer.
    pool.job
        .store(&job as *const Job as *mut Job, Ordering::Release);
    pool.epoch.fetch_add(1, Ordering::Release);
    if pool.parked.load(Ordering::SeqCst) > 0 {
        let _g = pool.lock.lock().unwrap_or_else(|e| e.into_inner());
        pool.work.notify_all();
    }

    // The submitter is a participant too, so a busy pool never idles it.
    run_bands(&job);
    if job.pending.fetch_sub(1, Ordering::AcqRel) != 1 {
        wait_for_completion(pool, &job);
    }

    if job.panicked.load(Ordering::Relaxed) != 0 {
        panic!("crabstack: a parallel band panicked");
    }
}

fn wait_for_completion(pool: &Pool, job: &Job) {
    for _ in 0..SPIN_LIMIT {
        if job.pending.load(Ordering::Acquire) == 0 {
            return;
        }
        std::hint::spin_loop();
    }
    // Announce before the final check, mirroring `await_epoch`: a worker that
    // reads `waiting == false` and skips its notify must not be able to do so
    // after this thread has committed to sleeping.
    pool.waiting.store(true, Ordering::SeqCst);
    let mut guard = pool.lock.lock().unwrap_or_else(|e| e.into_inner());
    while job.pending.load(Ordering::Acquire) != 0 {
        guard = pool.done.wait(guard).unwrap_or_else(|e| e.into_inner());
    }
    drop(guard);
    pool.waiting.store(false, Ordering::SeqCst);
}

/// Run `f(band, slot)` for every band, where `slot` is that band's own output
/// cell. Returns the partials **in band order**, so callers can reduce them
/// deterministically.
pub(crate) fn map_bands<T>(nbands: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T>
where
    T: Send + Default,
{
    let slots = Slots::new(nbands);
    for_each_band(nbands, &|b| {
        // SAFETY: each band index is handed out exactly once by `run_bands`, so
        // no two threads touch the same slot.
        unsafe { slots.set(b, f(b)) };
    });
    slots.into_inner()
}

struct Slots<T>(Vec<UnsafeCell<T>>);

// SAFETY: `set` is only ever called once per index, from the single thread that
// owns that band.
unsafe impl<T: Send> Sync for Slots<T> {}

impl<T: Default> Slots<T> {
    fn new(n: usize) -> Self {
        Slots((0..n).map(|_| UnsafeCell::new(T::default())).collect())
    }

    /// SAFETY: caller must guarantee unique access to index `i`.
    unsafe fn set(&self, i: usize, v: T) {
        *self.0[i].get() = v;
    }

    fn into_inner(self) -> Vec<T> {
        self.0.into_iter().map(UnsafeCell::into_inner).collect()
    }
}

/// Split `data` into `nbands` row bands of `stride` elements and run
/// `f(band, first_row, rows)` on each. Used by the warps, where every band
/// writes a disjoint range of output rows.
pub(crate) fn for_each_band_mut(
    data: &mut [f64],
    stride: usize,
    rows: usize,
    nbands: usize,
    f: impl Fn(usize, &mut [f64]) + Sync,
) {
    if nbands <= 1 {
        f(0, data);
        return;
    }
    let chunks = Chunks::new(data, stride, rows, nbands);
    for_each_band(nbands, &|b| {
        let (start, end) = band_range(b, nbands, rows);
        // SAFETY: band ranges are disjoint and each index is dispatched once.
        let slice = unsafe { chunks.slice(start, end) };
        f(start, slice);
    });
}

struct Chunks {
    ptr: *mut f64,
    stride: usize,
}

// SAFETY: `slice` hands out disjoint sub-slices, one per band.
unsafe impl Sync for Chunks {}

impl Chunks {
    fn new(data: &mut [f64], stride: usize, rows: usize, _nbands: usize) -> Self {
        debug_assert!(stride * rows <= data.len());
        Chunks {
            ptr: data.as_mut_ptr(),
            stride,
        }
    }

    /// SAFETY: caller must ensure `[start, end)` does not overlap any other
    /// live slice from this `Chunks`.
    #[allow(clippy::mut_from_ref)]
    unsafe fn slice(&self, start: usize, end: usize) -> &mut [f64] {
        std::slice::from_raw_parts_mut(
            self.ptr.add(start * self.stride),
            (end - start) * self.stride,
        )
    }
}

/// A raw pointer shareable across bands.
///
/// Used by the strided column passes: a band owns a range of *columns* but
/// touches every row, so its elements cannot be expressed as a sub-slice.
/// Bands still write disjoint elements — the caller must guarantee that.
/// The pointer is deliberately private: an edition-2021 closure captures
/// individual fields, so a public field would be captured as a bare `*mut T`
/// and lose the `Sync` this wrapper exists to provide.
#[derive(Clone, Copy)]
pub(crate) struct SyncPtr<T>(*mut T);

// SAFETY: upheld by callers, which hand each band a disjoint element set.
unsafe impl<T> Sync for SyncPtr<T> {}
unsafe impl<T> Send for SyncPtr<T> {}

impl<T> SyncPtr<T> {
    pub(crate) fn new(p: *mut T) -> Self {
        SyncPtr(p)
    }

    /// SAFETY: the caller must only access elements owned by its own band.
    pub(crate) unsafe fn at(self, offset: usize) -> *mut T {
        self.0.add(offset)
    }
}

/* ---------------- SIMD dispatch ---------------- */

/// Set by [`set_simd_enabled`] to force the scalar kernels on a CPU that could
/// run the vectorised ones.
static SIMD_DISABLED: AtomicBool = AtomicBool::new(false);

/// Whether the CPU supports the instruction set the vectorised kernels are
/// built for. Cached: `is_x86_feature_detected!` is cheap but not free, and a
/// single 512×512 registration reaches the dispatch tens of thousands of times.
#[cfg(target_arch = "x86_64")]
#[inline]
fn cpu_has_simd() -> bool {
    static CACHED: AtomicUsize = AtomicUsize::new(usize::MAX);
    match CACHED.load(Ordering::Relaxed) {
        usize::MAX => {
            let v = usize::from(
                std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma"),
            );
            CACHED.store(v, Ordering::Relaxed);
            v != 0
        }
        v => v != 0,
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn cpu_has_simd() -> bool {
    false
}

/// Whether the vectorised kernels should run: the CPU supports them and nothing
/// has switched them off.
///
/// Callers pair this with a `#[target_feature(enable = "avx2,fma")]` wrapper
/// around the kernel *body*. Wrapping only a call is not enough: LLVM is free
/// to leave that call outstanding — or tail-jump to it — in which case the
/// kernel is still compiled with the crate's baseline features. The kernel must
/// be `#[inline(always)]` so it is guaranteed to land inside the wrapper.
#[inline]
pub(crate) fn has_simd() -> bool {
    !SIMD_DISABLED.load(Ordering::Relaxed) && cpu_has_simd()
}

/// Force the scalar kernels (`false`) or restore automatic selection (`true`).
///
/// Exists so both kernel paths can be tested on one machine. Without it only the
/// path this CPU selects ever runs, and the other ships to users having been
/// executed nowhere — see the sweep in `tests/parity.rs`, which puts each path
/// against the C++ oracle in turn. It also makes a good first question when a
/// result looks wrong on one machine and not another.
///
/// The two paths are not guaranteed bit-identical and nothing needs them to be:
/// the differences at stake are ~1e-14 relative, against a registration accuracy
/// nearer 1e-2 px. What the sweep is looking for is a vectorised kernel that is
/// actually *wrong* — a bad lane index, a mishandled tail — which misses the
/// oracle by orders of magnitude, not by ulps.
///
/// Process-wide, and not synchronised with work already running: set it before
/// starting a registration, not during one.
pub fn set_simd_enabled(enabled: bool) {
    SIMD_DISABLED.store(!enabled, Ordering::Relaxed);
}

/// Whether the vectorised kernels will be used for the next call: false on a CPU
/// without AVX2+FMA, and false after `set_simd_enabled(false)`.
pub fn simd_enabled() -> bool {
    has_simd()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn band_ranges_tile_exactly() {
        for rows in [1usize, 7, 32, 33, 100, 511, 512] {
            for nbands in [1usize, 2, 3, 5, 32] {
                let nbands = nbands.min(rows);
                let mut covered = 0;
                let mut prev_end = 0;
                for b in 0..nbands {
                    let (s, e) = band_range(b, nbands, rows);
                    assert_eq!(s, prev_end, "gap at band {b} of {nbands} over {rows}");
                    assert!(e >= s);
                    covered += e - s;
                    prev_end = e;
                }
                assert_eq!(covered, rows);
                assert_eq!(prev_end, rows);
            }
        }
    }

    #[test]
    fn map_bands_returns_partials_in_band_order() {
        let out = map_bands(16, |b| b * b);
        assert_eq!(out, (0..16).map(|b| b * b).collect::<Vec<_>>());
    }

    #[test]
    fn for_each_band_mut_writes_disjoint_rows() {
        let (stride, rows) = (7usize, 40usize);
        let mut data = vec![0.0f64; stride * rows];
        let nbands = bands_for_grid(rows, 1024);
        for_each_band_mut(&mut data, stride, rows, nbands, |first_row, slice| {
            for (i, v) in slice.iter_mut().enumerate() {
                *v = (first_row + i / stride) as f64;
            }
        });
        for r in 0..rows {
            for c in 0..stride {
                assert_eq!(data[r * stride + c], r as f64);
            }
        }
    }
}
