from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable

import numpy as np
import scipy.sparse as sp

from asterra.data.support import SupportSpec
from asterra.utils.checks import check_1d
from asterra.utils.sparse import row_normalize_csr, to_csr

from .overlap import OverlapWeighter
from .align import assert_compatible_grids


@dataclass(frozen=True, slots=True)
class SupportMatrix:
    """Sparse operator mapping values from a source support to a target support.

    The underlying sparse matrix is shaped (n_target, n_source). A projection of
    features is computed as::

        X_target = M @ X_source

    where X_source has shape (n_source, n_features).
    """

    matrix: sp.csr_matrix
    source: SupportSpec
    target: SupportSpec
    normalized: bool = True

    def __post_init__(self) -> None:
        mat = to_csr(self.matrix)
        if mat.shape != (self.target.n_items, self.source.n_items):
            raise ValueError(
                "SupportMatrix matrix shape must be (n_target, n_source). "
                f"Got matrix.shape={mat.shape} but expected ({self.target.n_items}, {self.source.n_items})."
            )
        object.__setattr__(self, "matrix", mat)

    @classmethod
    def from_grid_to_grid(
        cls,
        *,
        source: SupportSpec,
        target: SupportSpec,
        normalize: bool = True,
        min_overlap: float = 0.0,
    ) -> "SupportMatrix":
        """Construct a grid→grid SupportMatrix using overlap-area weights."""

        assert_compatible_grids(source, target)
        weighter = OverlapWeighter(min_overlap=min_overlap)
        mat = weighter.grid_to_grid(source=source, target=target)
        normalized = False
        if normalize:
            mat = row_normalize_csr(mat)
            normalized = True
        return cls(matrix=mat, source=source, target=target, normalized=normalized)

    @classmethod
    def from_samples_to_groups(
        cls,
        *,
        group_ids: Iterable[Hashable],
        labels: Iterable[Hashable] | None = None,
        normalize: bool = True,
    ) -> "SupportMatrix":
        """Construct a samples→groups SupportMatrix from membership identifiers.

        Parameters
        ----------
        group_ids:
            Group identifier for each sample (length n_samples).
        labels:
            Optional explicit group label ordering. If omitted, labels are taken
            in order of first appearance in ``group_ids``.
        normalize:
            If True, the matrix is row-normalized so that group projections
            compute a mean over member samples.
        """

        group_arr = np.asarray(list(group_ids), dtype=object)
        group_arr = check_1d(group_arr, name="group_ids")
        n_samples = int(group_arr.shape[0])

        if labels is None:
            label_to_idx: dict[Hashable, int] = {}
            ordered: list[Hashable] = []
            for g in group_arr:
                if g not in label_to_idx:
                    label_to_idx[g] = len(ordered)
                    ordered.append(g)
            labels_t = tuple(ordered)
        else:
            labels_t = tuple(labels)
            label_to_idx = {g: i for (i, g) in enumerate(labels_t)}
            missing = set(group_arr).difference(label_to_idx)
            if missing:
                raise ValueError(f"labels is missing group ids present in group_ids: {sorted(missing)!r}")

        n_groups = len(labels_t)
        rows = np.fromiter((label_to_idx[g] for g in group_arr), dtype=int, count=n_samples)
        cols = np.arange(n_samples, dtype=int)
        data = np.ones(n_samples, dtype=float)
        mat = sp.coo_matrix((data, (rows, cols)), shape=(n_groups, n_samples)).tocsr()
        normalized_flag = False
        if normalize:
            mat = row_normalize_csr(mat)
            normalized_flag = True

        source = SupportSpec.samples_n(n_samples)
        target = SupportSpec.groups(labels_t)
        return cls(matrix=mat, source=source, target=target, normalized=normalized_flag)

    def project_features(self, X_source: np.ndarray) -> np.ndarray:
        X = np.asarray(X_source)
        if X.ndim != 2:
            raise ValueError(f"X_source must be 2D (n_source, n_features). Got shape={X.shape}.")
        if X.shape[0] != self.source.n_items:
            raise ValueError(
                "X_source first dimension must match source support size. "
                f"Got X_source.shape[0]={X.shape[0]} expected {self.source.n_items}."
            )
        return (self.matrix @ X).astype(X.dtype, copy=False)

    def project_labels(self, y_source: np.ndarray) -> np.ndarray:
        y = np.asarray(y_source)
        y = check_1d(y, name="y_source").astype(float, copy=False)
        if y.shape[0] != self.source.n_items:
            raise ValueError(
                "y_source length must match source support size. "
                f"Got len(y_source)={y.shape[0]} expected {self.source.n_items}."
            )
        return np.asarray(self.matrix @ y)

    def transpose(self) -> "SupportMatrix":
        """Return the transposed mapping (target becomes source)."""

        return SupportMatrix(
            matrix=self.matrix.transpose().tocsr(),
            source=self.target,
            target=self.source,
            normalized=self.normalized,
        )
