"""Stack registration on the ``crabstack`` backend.

``crabstack`` is a Rust registration backend built for this package — it lives in
``crabstack-rs/`` and is compiled into the wheel as ``caliana._crabstack``. It
exposes the same two native primitives pystackreg's C++ extension did —
``_register(ref, mov, model)`` and ``_transform(mov, matrix)`` — but no
orchestration around them. So the loop that used to come from
``pystackreg.StackReg`` lives here: register every frame against the chosen
reference, compose matrices when the reference is the previous frame, and convert
between TurboReg's short matrix form and the canonical 3x3.

Only the surface this package uses is implemented — the four linear models (see
``registration._STACKREG_TRANSFORMS`` for why bilinear is out), the three
reference modes, and single-image ``transform``. Frames are always on axis 0;
unlike pystackreg there is no axis guessing and so no axis warning to silence.

``_register`` also takes optional per-frame masks, which pystackreg never
exposed: see ``StackReg.register``.

The extension is imported at module level, but this module is itself imported
lazily from ``registration``, so ``import caliana`` still does not pull it in.
"""
from __future__ import annotations

import numpy as np

from . import _crabstack as crabstack


def _as_f64(a) -> np.ndarray:
    """Frames as float64 — the only dtype the native entry points accept."""
    return np.asarray(a, dtype=np.float64)


def _as_mask(a) -> np.ndarray | None:
    """A registration mask as the bool array the native entry point accepts."""
    return None if a is None else np.asarray(a, dtype=bool)


class StackReg:
    """Per-frame registration with a fixed transformation model.

    Mirrors the part of ``pystackreg.StackReg`` this package relied on, so
    matrices keep their meaning: 3x3 homogeneous, acting on ``(x, y, 1)``, mapping
    the moving frame back onto the reference.
    """

    TRANSLATION = crabstack.TRANSLATION
    RIGID_BODY = crabstack.RIGID_BODY
    SCALED_ROTATION = crabstack.SCALED_ROTATION
    AFFINE = crabstack.AFFINE

    _valid_transformations = (TRANSLATION, RIGID_BODY, SCALED_ROTATION, AFFINE)

    def __init__(self, transformation: int):
        if transformation not in self._valid_transformations:
            raise ValueError(f"Invalid transformation {transformation!r}")
        self._transformation = transformation
        self._m = None          # short-form matrix from the last register()

    def register(self, ref, mov, ref_mask=None, mov_mask=None) -> np.ndarray:
        """Estimate the 3x3 transform aligning ``mov`` onto ``ref``.

        The last row and column are dropped before estimation, as TurboReg's own
        wrapper does — the pyramid needs the working size to be one less than the
        image, and keeping the crop here keeps matrices interchangeable with
        pystackreg's.

        ref_mask / mov_mask: optional boolean arrays, frame-shaped, True where a
        pixel may take part in the fit. A pixel counts only where both frames
        keep it, so masking one side is enough to drop it. Cut masks a couple of
        pixels wider than whatever you are excluding — the spline interpolation
        and the pyramid spread a feature about that far past its own edge.
        The real intensities are left in place; a mask only narrows which of them
        are compared (this is what ``registration.leaf_mask`` feeds, from the
        outline drawn inside a leaf box).
        """
        ref, mov = _as_f64(ref), _as_f64(mov)
        if ref.shape != mov.shape:
            raise ValueError(f"shape mismatch: reference {ref.shape} vs moving {mov.shape}")
        ref_mask, mov_mask = _as_mask(ref_mask), _as_mask(mov_mask)
        for name, m in (("ref_mask", ref_mask), ("mov_mask", mov_mask)):
            if m is not None and m.shape != ref.shape:
                raise ValueError(f"{name} shape {m.shape} does not match frames {ref.shape}")
        self._m, _refpts, _movpts = crabstack._register(
            ref[:-1, :-1],
            mov[:-1, :-1],
            self._transformation,
            None if ref_mask is None else ref_mask[:-1, :-1],
            None if mov_mask is None else mov_mask[:-1, :-1],
        )
        return self.get_matrix()

    def transform(self, mov, tmat=None) -> np.ndarray:
        """Warp ``mov`` by ``tmat`` (3x3), or by the last estimate if omitted."""
        if tmat is None:
            if self._m is None:
                raise ValueError("register() must run before transform() without a matrix")
            short = self._m
        else:
            short = self._matrix_long_to_short(np.asarray(tmat, dtype=np.float64))
        return crabstack._transform(_as_f64(mov), short)

    def get_matrix(self) -> np.ndarray:
        return self._matrix_short_to_long(self._m)

    def register_stack(
        self, img, reference: str = "previous", n_frames: int = 1, masks=None
    ) -> np.ndarray:
        """Estimate one 3x3 transform per frame of ``img`` (frames on axis 0).

        reference: ``"previous"`` (each frame onto its predecessor, matrices
            composed so they all land in frame 0's coordinates), ``"first"`` (onto
            the mean of the first ``n_frames``), or ``"mean"`` (onto the mean of
            the whole stack).
        masks: optional ``(T, Y, X)`` boolean stack, one mask per frame (see
            ``register``). Which mask guards the reference side follows what the
            reference is: under ``"previous"`` it is frame ``i-1``'s own mask,
            and where the reference is an average it is the intersection of the
            averaged frames' masks — a pixel some frame masked out has already
            polluted that average.

        Returns an ``(T, 3, 3)`` array; the reference frame's own entry is the
        identity, which is why ``"previous"``/``"first"`` start at frame 1.
        """
        img = _as_f64(img)
        if img.ndim != 3:
            raise ValueError(f"stack must be 3-D (T, Y, X), got shape {img.shape}")
        if masks is not None:
            masks = np.asarray(masks, dtype=bool)
            if masks.shape != img.shape:
                raise ValueError(f"masks shape {masks.shape} does not match stack {img.shape}")

        idx_start = 1
        ref_mask = None
        if reference == "first":
            ref = img[:n_frames].mean(axis=0)
            if masks is not None:
                ref_mask = masks[:n_frames].all(axis=0)
        elif reference == "mean":
            ref = img.mean(axis=0)
            if masks is not None:
                ref_mask = masks.all(axis=0)
            idx_start = 0      # no frame is the reference, so none is exempt
        elif reference == "previous":
            ref = None
        else:
            raise ValueError(
                f"Unknown reference {reference!r} (expected 'previous' | 'first' | 'mean')"
            )

        tmats = np.repeat(np.identity(3)[None, ...], len(img), axis=0)
        for i in range(idx_start, len(img)):
            if reference == "previous":
                ref = img[i - 1]
                if masks is not None:
                    ref_mask = masks[i - 1]
            tmats[i] = self.register(
                ref, img[i], ref_mask, None if masks is None else masks[i]
            )
            if reference == "previous" and i > 0:
                # Frame i is aligned to i-1, which is aligned to i-2, ...
                tmats[i] = tmats[i] @ tmats[i - 1]
        return tmats

    # ------------------------------------------------------ matrix conventions
    # TurboReg stores a transform column-shifted: the translation column comes
    # first, then the linear part. These two put it back in canonical order and
    # take it apart again, keeping pystackreg's convention exactly so stored
    # matrices from either backend stay interchangeable.
    def _matrix_short_to_long(self, m: np.ndarray) -> np.ndarray:
        mat = np.identity(3)
        if self._transformation == self.TRANSLATION:
            mat[0:2, 2] = m[:, 0]
        else:
            mat[0:2, :] = m[:, [1, 2, 0]]
        return mat

    def _matrix_long_to_short(self, mat: np.ndarray) -> np.ndarray:
        if mat.shape != (3, 3):
            raise ValueError(f"transformation matrix must be 3x3, got {mat.shape}")
        if self._transformation == self.TRANSLATION:
            return mat[0:2, 2].reshape((2, 1))
        return mat[0:2, [2, 0, 1]]
