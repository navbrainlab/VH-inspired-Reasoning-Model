"""
Place cell attractor / denoiser for Complex Vector-HASH.

Two implementations:
1. PlaceAttractorDenoiser: Pure numpy nearest-neighbor (recommended for storage test)
2. PlaceHopfieldDenoiser: torch-based with hflayers support (from VH_complex_with_Attrac)

Both use joint [real, imag] retrieval to ensure retrieved real and imaginary
parts come from the same stored pattern.
"""

import numpy as np


# ============================================================================
# Lightweight numpy-only denoiser (for storage/capacity test)
# ============================================================================

class PlaceAttractorDenoiser:
    """
    Nearest-neighbor attractor for place cell vectors.

    Works in two modes:
    - Complex: patterns are (Np, Nstored) complex, similarity in [re;im] space
    - Real (H'): patterns are (D, Nstored) real, similarity directly in D-dim space

    The mode is auto-detected from the dtype of stored patterns.
    """

    def __init__(self):
        self.stored = None               # (D, Nstored) — raw stored patterns
        self.stored_normed = None         # (D_sim, Nstored) — normalized for similarity
        self.n_stored = 0
        self._is_complex = False

    def store_patterns(self, patterns):
        """
        Store patterns for nearest-neighbor retrieval.

        Args:
            patterns: (D, Nstored) array, real or complex.
        """
        patterns = np.asarray(patterns)
        self._is_complex = np.iscomplexobj(patterns)
        self.stored = patterns.copy()
        self.n_stored = patterns.shape[1]

        if self._is_complex:
            sim_mat = np.vstack([patterns.real, patterns.imag])
        else:
            sim_mat = patterns

        norms = np.linalg.norm(sim_mat, axis=0, keepdims=True) + 1e-12
        self.stored_normed = sim_mat / norms

    def denoise_batch(self, p_batch):
        """
        Denoise a batch of patterns via nearest-neighbor retrieval.

        Args:
            p_batch: (nruns, D, Npatts) array, real or complex

        Returns:
            (nruns, D, Npatts) array of denoised patterns (same dtype as stored)
        """
        if self.stored is None:
            raise RuntimeError("No patterns stored. Call store_patterns() first.")

        nruns, D, Npatts = p_batch.shape
        out_dtype = self.stored.dtype
        result = np.zeros((nruns, D, Npatts), dtype=out_dtype)

        for ru in range(nruns):
            p = p_batch[ru]  # (D, Npatts)

            if self._is_complex:
                if np.iscomplexobj(p):
                    query = np.vstack([p.real, p.imag])
                else:
                    query = np.vstack([p, np.zeros_like(p)])
            else:
                query = p.real if np.iscomplexobj(p) else p

            qnorms = np.linalg.norm(query, axis=0, keepdims=True) + 1e-12
            query_normed = query / qnorms

            sim = query_normed.T @ self.stored_normed   # (Npatts, Nstored)
            best_idx = sim.argmax(axis=1)
            result[ru] = self.stored[:, best_idx]

        return result

    def clear(self):
        """Clear all stored patterns."""
        self.stored = None
        self.stored_normed = None
        self.n_stored = 0
        self._is_complex = False


# ============================================================================
# Full Hopfield-based denoiser (torch + hflayers, from VH_complex_with_Attrac)
# ============================================================================

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from hflayers import Hopfield
    _HFLAYERS_AVAILABLE = True
except ImportError:
    _HFLAYERS_AVAILABLE = False


def create_hopfield_retrieval(beta=100):
    """
    Create Hopfield layer for pattern retrieval (no projection, no normalization).

    Requires: pip install hflayers
    """
    if not _HFLAYERS_AVAILABLE:
        raise ImportError("hflayers not installed. Install with: pip install hflayers")
    return Hopfield(
        scaling=beta,
        state_pattern_as_static=True,
        stored_pattern_as_static=True,
        pattern_projection_as_static=True,
        normalize_stored_pattern=False,
        normalize_stored_pattern_affine=False,
        normalize_state_pattern=False,
        normalize_state_pattern_affine=False,
        normalize_pattern_projection=False,
        normalize_pattern_projection_affine=False,
        disable_out_projection=True
    )


class PlaceHopfieldDenoiser:
    """
    Hopfield/nearest-neighbor denoiser for complex place cell vectors.

    Uses joint [real, imag] retrieval to ensure returned real and imaginary
    parts come from the same stored pattern.

    Supports two modes:
    - 'nearest': cosine-similarity nearest neighbor (default, recommended)
    - 'hopfield': modern Hopfield network via hflayers (needs large beta)

    Requires torch for both modes. For a pure-numpy version, use
    PlaceAttractorDenoiser instead.
    """

    def __init__(self, pattern_dim, beta=1.0, mode='nearest',
                 device='cuda' if _TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'):
        if not _TORCH_AVAILABLE:
            raise ImportError("torch not installed. Use PlaceAttractorDenoiser for numpy-only.")

        self.pattern_dim = pattern_dim
        self.beta = beta
        self.mode = mode
        self.device = device

        self.stored_patterns_concat = []
        self.stored_patterns_real = []
        self.stored_patterns_imag = []

        self.hopfield = None
        self.is_built = False

    def add_pattern(self, p_true_real, p_true_imag=None):
        """Add a single p_true pattern (real or complex)."""
        p_true_real = np.asarray(p_true_real)

        if np.iscomplexobj(p_true_real):
            real_part = p_true_real.real.copy()
            imag_part = p_true_real.imag.copy()
        else:
            real_part = p_true_real.copy()
            imag_part = np.zeros_like(p_true_real) if p_true_imag is None else np.asarray(p_true_imag).copy()

        self.stored_patterns_real.append(real_part)
        self.stored_patterns_imag.append(imag_part)
        self.stored_patterns_concat.append(np.concatenate([real_part, imag_part]))

    def build(self):
        """Build the denoiser after all patterns have been added."""
        if len(self.stored_patterns_concat) == 0:
            raise ValueError("No patterns stored. Call add_pattern() first.")

        num_patterns = len(self.stored_patterns_concat)

        self.Y_concat = torch.tensor(
            np.stack(self.stored_patterns_concat, axis=0),
            dtype=torch.float64, device=self.device
        )
        self.Y_real = torch.tensor(
            np.stack(self.stored_patterns_real, axis=0),
            dtype=torch.float64, device=self.device
        )
        self.Y_imag = torch.tensor(
            np.stack(self.stored_patterns_imag, axis=0),
            dtype=torch.float64, device=self.device
        )
        self.Y_concat_norm = self.Y_concat / (torch.norm(self.Y_concat, dim=1, keepdim=True) + 1e-8)

        if self.mode == 'hopfield':
            self.hopfield = create_hopfield_retrieval(self.beta).to(self.device)

        self.is_built = True

    def denoise(self, p_real_pred, p_imag_pred, return_complex=True):
        """
        Denoise predicted place cell vectors.

        Args:
            p_real_pred: predicted real part (Np,) or (batch, Np)
            p_imag_pred: predicted imag part (Np,) or (batch, Np)
            return_complex: if True, return complex array; else (real, imag) tuple
        """
        if not self.is_built:
            raise RuntimeError("Denoiser not built. Call build() first.")

        p_real_pred = np.asarray(p_real_pred)
        p_imag_pred = np.asarray(p_imag_pred)

        if self.mode == 'nearest':
            p_real_np, p_imag_np = self._denoise_nearest(p_real_pred, p_imag_pred)
        else:
            p_real_np, p_imag_np = self._denoise_hopfield(p_real_pred, p_imag_pred)

        if return_complex:
            return p_real_np + 1j * p_imag_np
        return p_real_np, p_imag_np

    def _denoise_nearest(self, p_real_pred, p_imag_pred):
        p_real = torch.tensor(p_real_pred, dtype=torch.float64, device=self.device)
        p_imag = torch.tensor(p_imag_pred, dtype=torch.float64, device=self.device)

        single_input = p_real.ndim == 1
        if single_input:
            p_real = p_real.unsqueeze(0)
            p_imag = p_imag.unsqueeze(0)

        p_concat = torch.cat([p_real, p_imag], dim=1)
        p_concat_norm = p_concat / (torch.norm(p_concat, dim=1, keepdim=True) + 1e-8)
        sim = p_concat_norm @ self.Y_concat_norm.T
        best_idx = torch.argmax(sim, dim=1)

        p_real_denoised = self.Y_real[best_idx]
        p_imag_denoised = self.Y_imag[best_idx]

        p_real_np = p_real_denoised.cpu().numpy()
        p_imag_np = p_imag_denoised.cpu().numpy()

        if single_input:
            p_real_np = p_real_np[0]
            p_imag_np = p_imag_np[0]

        return p_real_np, p_imag_np

    def _denoise_hopfield(self, p_real_pred, p_imag_pred):
        if not _HFLAYERS_AVAILABLE:
            raise ImportError("hflayers not installed for hopfield mode.")

        p_real = torch.tensor(p_real_pred, dtype=torch.float64, device=self.device)
        p_imag = torch.tensor(p_imag_pred, dtype=torch.float64, device=self.device)

        single_input = p_real.ndim == 1
        if single_input:
            p_real = p_real.unsqueeze(0)
            p_imag = p_imag.unsqueeze(0)

        batch_size = p_real.shape[0]
        p_concat = torch.cat([p_real, p_imag], dim=1)
        Y_batch = self.Y_concat.unsqueeze(0).expand(batch_size, -1, -1)
        R = p_concat.unsqueeze(1)

        with torch.no_grad():
            output = self.hopfield((Y_batch, R, Y_batch))

        output = output.squeeze(1)
        p_real_np = output[:, :self.pattern_dim].cpu().numpy()
        p_imag_np = output[:, self.pattern_dim:].cpu().numpy()

        if single_input:
            p_real_np = p_real_np[0]
            p_imag_np = p_imag_np[0]

        return p_real_np, p_imag_np

    def get_num_patterns(self):
        return len(self.stored_patterns_concat)

    def clear_patterns(self):
        self.stored_patterns_concat = []
        self.stored_patterns_real = []
        self.stored_patterns_imag = []
        self.is_built = False
        self.hopfield = None
