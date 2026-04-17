from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def row_sums(mat: sp.spmatrix) -> np.ndarray:
    """Return dense row sums as a 1D float array."""

    sums = np.asarray(mat.sum(axis=1)).ravel()
    return sums.astype(float, copy=False)


def row_normalize_csr(mat: sp.csr_matrix) -> sp.csr_matrix:
    """Row-normalize a CSR matrix (only non-zero rows).

    Rows with sum 0 remain all-zero.
    """

    if not sp.isspmatrix_csr(mat):
        raise TypeError("row_normalize_csr expects a CSR matrix.")
    sums = row_sums(mat)
    nz = sums != 0.0
    if not np.any(nz):
        return mat
    inv = np.zeros_like(sums)
    inv[nz] = 1.0 / sums[nz]
    return sp.diags(inv) @ mat


def to_csr(mat: sp.spmatrix) -> sp.csr_matrix:
    if sp.isspmatrix_csr(mat):
        return mat
    return mat.tocsr()

