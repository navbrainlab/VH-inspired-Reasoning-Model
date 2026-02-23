"""
Complex VH storage capacity utilities.

Provides complex versions of the storage/capacity test functions from
src/senstranspose_utils.py, adding:
  - Complex phase support for grid and place cells
  - Fourier harmonic expansion for representing >2 phases
  - Random expansion mode as an alternative to Fourier
  - Place cell attractor denoiser (Hopfield-like nearest neighbor)
  - Phase demodulation in the dynamics loop

The reconstruction pipeline for each stored pattern:
  s → Wps → H'(p) → [place attractor in H' space]
    → collapse H'→p → Wgp → g_noisy → [grid attractor] → g_clean
    → Wpg → p_raw → phase_demod + nonlin → [place attractor in H' space]
    → collapse H'→p → expand p→H' → Wsp → s_recon

Two expansion modes:
  expand_mode='fourier' (default):
    Given p ∈ C^{Np}, compute H'(p) by stacking [Re(p·e^{ikφ}); Im(p·e^{ikφ})]
    for harmonics k = 1, ..., K.  The incremental scheme:
      nph=2: K=1, H' = [Re(p·e^{iφ}); Im(p·e^{iφ})]          dim = 2Np
      nph=3: K=2, H' = [Re; Im; sin(2φ)·|p|]                   dim = 3Np  (+sin2φ)
      nph=4: K=2, H' = [Re; Im; Re(p·e^{i2φ}); Im(p·e^{i2φ})] dim = 4Np  (+cos2φ,sin2φ)
      nph=5: K=3, H' = 4Np block + [sin(3φ)·|p|]               dim = 5Np  (+sin3φ)
      nph=6: K=3, 4Np block + [Re(p·e^{i3φ}); Im(p·e^{i3φ})]  dim = 6Np  (+cos3φ,sin3φ)

  expand_mode='random':
    H' = [Re(p·e^{iφ}); Im(p·e^{iφ}); R₁; R₂; ...]
    where Rⱼ are iid Gaussian random blocks of shape (Np, Ntotal).
    For Num_phi=3: 1 extra block → dim = 3Np
    For Num_phi=4: 2 extra blocks → dim = 4Np  ...etc.
    The random blocks are generated once at codebook creation and cached.
    During dynamics, the cached random portion is looked up by nearest-neighbor
    index from the denoiser rather than re-derived from p.

All weight matrices are real-valued.
Wgp uses pseudoinverse on real pbook (matching VH_complex_with_Attrac).
Wpg remains random projection on real pbook.
"""

import numpy as np
from numpy.random import randn, randint
from itertools import combinations
from tqdm import tqdm

from src.assoc_utils_np import nonlin, train_gcpc
from src.senstranspose_utils import corrupt_p_1
from src_complex.phase_utils import (phase_demodulate, gridCAN_2d_complex,
                                     gridNN_2d_complex, module_wise_NN_2d_complex)
from src_complex.place_denoiser import PlaceAttractorDenoiser


# ============================================================================
# Auto-phi selection  &  Fourier harmonic helpers
# ============================================================================

def auto_select_phis(Num_phi):
    """Select *Num_phi* phases with no harmonic collisions.

    Requirements
    ------------
    * K = ceil(Num_phi / 2)  is the max harmonic order used.
    * For every pair (k, phi_i) != (p, phi_j),  k*phi_i != p*phi_j.
    * max(k * phi_i) < pi   for all k in 1..K.

    Strategy
    --------
    Choose  c_j = K * j + 1   for j = 0 .. Num_phi-1,  then
    phi_j = c_j * pi / D   where  D = K * c_{Num_phi-1} + 1.

    Collision-free proof:  k * c_j = k * (K*j + 1).
      - k * c_j  mod K  = k mod K, so different k yield different remainders.
      - Within the same k, values are strictly increasing in j.
    Therefore all K * Num_phi products are distinct.
    The maximum product  K * c_{Num_phi-1} = D - 1  so  k*phi_j < pi.

    Parameters
    ----------
    Num_phi : int  (>= 1)

    Returns
    -------
    phis : ndarray of shape (Num_phi,), in (0, pi)
    """
    if Num_phi < 1:
        raise ValueError("Num_phi must be >= 1")
    if Num_phi == 1:
        return np.array([0.0])
    K = int(np.ceil(Num_phi / 2))
    c = [K * j + 1 for j in range(Num_phi)]      # collision-free base indices
    denom = K * c[-1] + 1                          # ensures max(k * c_j) < denom
    phis = np.array([cj * np.pi / denom for cj in c])
    return phis


def harmonic_dim(Num_phi):
    """Return the H' dimension multiplier for *Num_phi* phases.

    nph→dim_mult: 1→1(real), 2→2, 3→3, 4→4, 5→5, 6→6, ...
    i.e. dim_mult = max(1, Num_phi).  Total H' dim = dim_mult * Np.
    """
    return max(1, Num_phi)



def fourier_expand_from_base(pbook_base, phis, mode='phi_first'):
    """Build Fourier harmonic H' from real *pbook_base* and phase list.

    Parameters
    ----------
    pbook_base : ndarray, shape (nruns, Np, Npos), real
        Base place codebook (no phase applied).
    phis : ndarray, shape (nph,)
    mode : str, 'phi_first' or 'map_first'

    Returns
    -------
    H : ndarray, shape (nruns, dim_mult * Np, Ntotal), real
        Where Ntotal = Npos * nph, dim_mult = max(1, nph).
    """
    phis = np.asarray(phis, dtype=np.float64)
    nph = len(phis)
    nruns, Np, Npos = pbook_base.shape
    Ntotal = Npos * nph
    dim_mult = harmonic_dim(nph)

    if nph <= 1:
        return pbook_base.copy()

    # Phase factors for each column: shape (Ntotal,) — the phase of each column
    # Build per-column phase array depending on mode
    if mode == 'phi_first':
        # idx = pos * nph + phi_idx → col_phases[idx] = phis[phi_idx]
        col_phases = np.tile(phis, Npos)  # (Ntotal,)
    elif mode == 'map_first':
        # idx = phi_idx * Npos + pos → col_phases[idx] = phis[phi_idx]
        col_phases = np.repeat(phis, Npos)  # (Ntotal,)
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    # Build complex phased pbook: pbook_base * e^{i * col_phase}
    # pbook_base: (nruns, Np, Npos) → tile to (nruns, Np, Ntotal)
    # Each position pos appears nph times with different phases
    if mode == 'phi_first':
        # For each pos, repeat nph times: pbook_base[:, :, pos] appears at
        # indices pos*nph .. pos*nph + nph - 1
        pbook_tiled = np.repeat(pbook_base, nph, axis=2)  # (nruns, Np, Ntotal)
    else:  # map_first
        # For each phi_idx, the full pbook_base[:, :, :] appears at
        # indices phi_idx*Npos .. phi_idx*Npos + Npos - 1
        pbook_tiled = np.tile(pbook_base, (1, 1, nph))  # (nruns, Np, Ntotal)

    # H' blocks: incremental expansion scheme
    # nph=2: [Re(p*e^{iφ}), Im(p*e^{iφ})]                  = 2Np
    # nph=3: [Re, Im, sin(2φ)*p]                             = 3Np
    # nph=4: [Re, Im, Re(p*e^{i2φ}), Im(p*e^{i2φ})]         = 4Np
    # nph=5: [Re, Im, Re(p*e^{i2φ}), Im(p*e^{i2φ}), sin(3φ)*p] = 5Np
    # ...
    # Pattern: each new harmonic k adds sin(kφ) first (odd nph),
    #          then cos(kφ)+sin(kφ) together (even nph).

    blocks = []
    # Harmonic k=1: always [cos(φ)*p, sin(φ)*p] = [Re, Im]
    blocks.append(pbook_tiled * np.cos(col_phases)[None, None, :])  # Re
    blocks.append(pbook_tiled * np.sin(col_phases)[None, None, :])  # Im
    dims_so_far = 2

    K = int(np.ceil(nph / 2))
    for k in range(2, K + 1):
        needed = dim_mult - dims_so_far
        if needed >= 2:
            # Add both cos(kφ)*p and sin(kφ)*p
            blocks.append(pbook_tiled * np.cos(k * col_phases)[None, None, :])
            blocks.append(pbook_tiled * np.sin(k * col_phases)[None, None, :])
            dims_so_far += 2
        elif needed == 1:
            # Add only sin(kφ)*p  (odd nph case)
            blocks.append(pbook_tiled * np.sin(k * col_phases)[None, None, :])
            dims_so_far += 1

    H = np.concatenate(blocks, axis=1)  # (nruns, dim_mult * Np, Ntotal)
    return H


def fourier_collapse_to_complex(H_prime, Np):
    """Collapse H' back to complex place vectors using first 2Np dims.

    Parameters
    ----------
    H_prime : ndarray, shape (nruns, dim_mult*Np, Npatts) or (dim_mult*Np, Npatts)
        Fourier harmonic representation.
    Np : int
        Number of place cells.

    Returns
    -------
    p_complex : ndarray, same leading dims + (Np, Npatts), complex
        First Np rows = Re, next Np rows = Im → complex.
    """
    if H_prime.ndim == 2:
        return H_prime[:Np, :] + 1j * H_prime[Np:2*Np, :]
    else:  # ndim == 3
        return H_prime[:, :Np, :] + 1j * H_prime[:, Np:2*Np, :]


def fourier_expand_from_complex(p_complex, phis, mode='phi_first', Npos=None):
    """Expand complex p vectors back to H' (for dynamics: p → H' → Wsp → s).

    Given recovered p_complex of shape (nruns, Np, Npatts), rebuild the full
    H' representation by applying Fourier harmonics with the known per-column
    phases.

    Parameters
    ----------
    p_complex : ndarray, shape (nruns, Np, Npatts), complex
    phis : ndarray, shape (nph,)
    mode : str
    Npos : int, optional.  If None, inferred as Npatts // nph.

    Returns
    -------
    H : ndarray, shape (nruns, dim_mult*Np, Npatts), real
    """
    phis = np.asarray(phis, dtype=np.float64)
    nph = len(phis)
    nruns, Np, Npatts = p_complex.shape
    dim_mult = harmonic_dim(nph)

    if nph <= 1:
        return p_complex.real

    if Npos is None:
        Npos = Npatts // nph if nph > 0 else Npatts

    # Per-column phases (same logic as fourier_expand_from_base)
    if mode == 'phi_first':
        col_phases = np.tile(phis, max(1, Npos))[:Npatts]
    elif mode == 'map_first':
        col_phases = np.repeat(phis, max(1, Npos))[:Npatts]
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    # Extract base real magnitude: p_base = |p| * sign(Re(p * e^{-iφ}))
    # Actually p_complex = p_base * e^{iφ}, so p_base = Re(p_complex * e^{-iφ})
    # when p_base is real and nonneg after nonlin.  But for generality,
    # we use p_base = p_complex * e^{-iφ_col} which should be real-valued.
    # Then harmonics are p_base * cos(kφ) and p_base * sin(kφ).
    demod = p_complex * np.exp(-1j * col_phases[None, None, :])  # should be ~real
    p_base = demod.real  # (nruns, Np, Npatts)

    blocks = []
    blocks.append(p_base * np.cos(col_phases)[None, None, :])
    blocks.append(p_base * np.sin(col_phases)[None, None, :])
    dims_so_far = 2

    K = int(np.ceil(nph / 2))
    for k in range(2, K + 1):
        needed = dim_mult - dims_so_far
        if needed >= 2:
            blocks.append(p_base * np.cos(k * col_phases)[None, None, :])
            blocks.append(p_base * np.sin(k * col_phases)[None, None, :])
            dims_so_far += 2
        elif needed == 1:
            blocks.append(p_base * np.sin(k * col_phases)[None, None, :])
            dims_so_far += 1

    return np.concatenate(blocks, axis=1)


# ============================================================================
# Random expansion mode
# ============================================================================

def random_expand_from_base(pbook_base, phis, mode='phi_first', rng_seed=42):
    """Build random-augmented H' from real *pbook_base* and phase list.

    H' = [Re(p·e^{iφ}); Im(p·e^{iφ}); R₁; R₂; ...]
    where Rⱼ ∈ R^{Np × Ntotal} are iid Gaussian random blocks.
    Total extra random dims = (Num_phi - 2) × Np.

    Parameters
    ----------
    pbook_base : ndarray, shape (nruns, Np, Npos), real
    phis : ndarray, shape (nph,)
    mode : str, 'phi_first' or 'map_first'
    rng_seed : int, seed for reproducible random blocks

    Returns
    -------
    H : ndarray, shape (nruns, dim_mult * Np, Ntotal), real
    random_blocks : ndarray, shape (nruns, (nph-2)*Np, Ntotal), real
        The cached random portion, needed for dynamics reconstruction.
    """
    phis = np.asarray(phis, dtype=np.float64)
    nph = len(phis)
    nruns, Np, Npos = pbook_base.shape
    Ntotal = Npos * nph
    dim_mult = harmonic_dim(nph)

    if nph <= 2:
        # No extra random dims needed; fall back to standard [Re; Im]
        H = fourier_expand_from_base(pbook_base, phis, mode=mode)
        random_blocks = np.empty((nruns, 0, Ntotal))
        return H, random_blocks

    # Build per-column phase array
    if mode == 'phi_first':
        col_phases = np.tile(phis, Npos)
    elif mode == 'map_first':
        col_phases = np.repeat(phis, Npos)
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    # Tile base to Ntotal columns
    if mode == 'phi_first':
        pbook_tiled = np.repeat(pbook_base, nph, axis=2)
    else:
        pbook_tiled = np.tile(pbook_base, (1, 1, nph))

    # First 2 blocks: [Re; Im] = [cos(φ)·p; sin(φ)·p]
    block_re = pbook_tiled * np.cos(col_phases)[None, None, :]
    block_im = pbook_tiled * np.sin(col_phases)[None, None, :]

    # Extra random blocks: (nph - 2) × Np dims, iid Gaussian
    n_extra = (nph - 2) * Np
    rng = np.random.RandomState(rng_seed)
    random_blocks = rng.randn(nruns, n_extra, Ntotal)

    H = np.concatenate([block_re, block_im, random_blocks], axis=1)
    return H, random_blocks


def random_expand_from_complex(p_complex, phis, random_blocks_cache,
                                denoiser, mode='phi_first', Npos=None):
    """Expand complex p back to H' using random mode (for dynamics loop).

    The first 2Np dims are rebuilt from p_complex (same as fourier mode).
    The extra random dims are looked up from the cached random_blocks
    using the denoiser's nearest-neighbor index.

    Parameters
    ----------
    p_complex : ndarray, shape (nruns, Np, Npatts), complex
    phis : ndarray, shape (nph,)
    random_blocks_cache : ndarray, shape (nruns, (nph-2)*Np, Npatts_stored)
        The cached random portion of the codebook (first Npatts columns).
    denoiser : PlaceAttractorDenoiser
        Used to find nearest-neighbor indices for looking up cached randoms.
    mode : str
    Npos : int or None

    Returns
    -------
    H : ndarray, shape (nruns, dim_mult*Np, Npatts), real
    """
    phis = np.asarray(phis, dtype=np.float64)
    nph = len(phis)
    nruns, Np, Npatts = p_complex.shape
    dim_mult = harmonic_dim(nph)

    if nph <= 2:
        return fourier_expand_from_complex(p_complex, phis, mode=mode, Npos=Npos)

    if Npos is None:
        Npos = Npatts // nph if nph > 0 else Npatts

    # Per-column phases
    if mode == 'phi_first':
        col_phases = np.tile(phis, max(1, Npos))[:Npatts]
    elif mode == 'map_first':
        col_phases = np.repeat(phis, max(1, Npos))[:Npatts]
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    # Demodulate to get p_base
    demod = p_complex * np.exp(-1j * col_phases[None, None, :])
    p_base = demod.real

    # First 2 blocks: [Re; Im]
    block_re = p_base * np.cos(col_phases)[None, None, :]
    block_im = p_base * np.sin(col_phases)[None, None, :]

    # Extra dims: look up cached random blocks by NN index.
    # The denoiser stores full H' patterns in self.stored (dim_mult*Np, Npatts_stored).
    # We compare only the first 2Np dims to find the best match.
    stored = denoiser.stored  # (dim_mult*Np, Npatts_stored)
    stored_re_im = stored[:2*Np, :]  # (2Np, Npatts_stored)

    n_extra = (nph - 2) * Np
    random_out = np.zeros((nruns, n_extra, Npatts))

    # Build the [Re; Im] query from the blocks we already computed
    H_partial = np.concatenate([block_re, block_im], axis=1)  # (nruns, 2Np, Npatts)

    # Vectorized NN lookup: compute all distances at once
    for ru in range(nruns):
        # query: (2Np, Npatts), stored_re_im: (2Np, Npatts_stored)
        # distance matrix: (Npatts, Npatts_stored)
        q = H_partial[ru]  # (2Np, Npatts)
        dists = (np.sum(q**2, axis=0, keepdims=True).T       # (Npatts, 1)
                 - 2 * q.T @ stored_re_im                     # (Npatts, Nstored)
                 + np.sum(stored_re_im**2, axis=0, keepdims=True))  # (1, Nstored)
        nn_idx = np.argmin(dists, axis=1)                     # (Npatts,)
        random_out[ru] = random_blocks_cache[ru][:, nn_idx]   # (n_extra, Npatts)

    H = np.concatenate([block_re, block_im, random_out], axis=1)
    return H


def pseudotrain_Wsp_cca(sbook, H_prime, Npatts):
    """P→S on Fourier harmonic representation H'.

    Parameters
    ----------
    sbook : (Ns, Ntotal)
    H_prime : (nruns, dim_mult*Np, Ntotal)  — already expanded
    Npatts : int

    Returns
    -------
    Wsp_cca : ndarray, shape (nruns, Ns, dim_mult*Np)
    """
    H = H_prime[:, :, :Npatts]                    # (nruns, D, Npatts)
    Hinv = np.linalg.pinv(H)                       # (nruns, Npatts, D)
    return np.einsum('ij, kjl -> kil', sbook[:, :Npatts], Hinv)


def pseudotrain_Wps_cca(H_prime, sbook, Npatts):
    """S→P on Fourier harmonic representation H'.

    Returns a single matrix Wps_cca of shape (nruns, dim_mult*Np, Ns)
    that maps s → H'(p).
    """
    sbookinv = np.linalg.pinv(sbook[:, :Npatts])  # (Npatts, Ns)
    H = H_prime[:, :, :Npatts]                    # (nruns, D, Npatts)
    Wps_cca = np.einsum('ij, kli -> klj', sbookinv, H)
    return Wps_cca  # (nruns, dim_mult*Np, Ns)


def pseudotrain_Wps_split(pbook_complex, sbook, Npatts):
    """Learn split S→P maps: Wps_cos (s→p.real), Wps_sin (s→p.imag).

    Returns
    -------
    Wps_cos : ndarray, shape (nruns, Np, Ns)
    Wps_sin : ndarray, shape (nruns, Np, Ns)
    """
    sbookinv = np.linalg.pinv(sbook[:, :Npatts])  # (Npatts, Ns)
    Wps_cos = np.einsum('ij, kli -> klj', sbookinv, pbook_complex[:, :, :Npatts].real)
    Wps_sin = np.einsum('ij, kli -> klj', sbookinv, pbook_complex[:, :, :Npatts].imag)
    return Wps_cos, Wps_sin


def _phase_demod_nonlin_batch(p_raw, thresh, phis):
    """
    Phase-demodulate, apply nonlinearity, and re-phase for batch of place vectors.

    For real inputs, simply applies nonlin (fast path).
    For complex inputs, processes each pattern individually.

    Args:
        p_raw: (nruns, Np, Npatts) complex or real
        thresh: nonlinearity threshold
        phis: known phases for snapping (can be None)

    Returns:
        (nruns, Np, Npatts) array
    """
    # Fast path: effectively real
    if not np.iscomplexobj(p_raw):
        return nonlin(p_raw, thresh)
    if np.max(np.abs(p_raw.imag)) < 1e-10:
        return nonlin(p_raw.real, thresh)

    nruns, Np, Npatts = p_raw.shape
    p_out = np.zeros_like(p_raw)

    for ru in range(nruns):
        for patt in range(Npatts):
            vec = p_raw[ru, :, patt]
            signed_real, phi_hat = phase_demodulate(vec, phis=phis)
            mag = nonlin(signed_real, thresh)
            p_out[ru, :, patt] = mag * np.exp(1j * phi_hat)

    return p_out


def dynamics_gs_vectorized_patts_complex(sinit, Niter, sbook, H_prime, gbook_phased,
                                          Wgp, Wpg, Wsp, Wps,
                                          lambdas, sparsity, thresh, Npatts, Np,
                                          denoiser, phis=None, mode='phi_first',
                                          Npos=None,
                                          grid_attractor='argmax',
                                          module_gbooks=None, module_sizes=None,
                                          expand_mode='fourier',
                                          random_blocks_cache=None):
    """
    Complex VH dynamics with Fourier harmonic or random H' representation.

    Pipeline:
      s_init → Wps → H'_pred → [place attractor in H' space]
      → collapse H'→p_complex (first 2Np dims)
      → for each iteration:
          Wgp → g_noisy → [grid attractor] → g_clean
          → Wpg → p_raw → phase_demod + nonlin → expand p→H'
          → [place attractor in H' space] → collapse H'→p_complex
      → expand p→H' → Wsp → sign → s_recon

    Args:
        sinit:        (nruns, Ns, >=Npatts) sensory input
        Niter:        number of attractor iterations
        sbook:        (Ns, Ntotal) sensory codebook
        H_prime:      (nruns, dim_mult*Np, Ntotal) harmonic/random codebook
        gbook_phased: (Ng, Ntotal) phased grid codebook (complex)
        Wgp:          (nruns, Ng, Np) place-to-grid weights (real)
        Wpg:          (nruns, Np, Ng) grid-to-place weights (real)
        Wsp:          (nruns, Ns, dim_mult*Np)  H'→S weight
        Wps:          (nruns, dim_mult*Np, Ns)  S→H' weight
        lambdas:      grid periods
        sparsity:     unused
        thresh:       nonlinearity threshold
        Npatts:       number of patterns
        Np:           number of place cells
        denoiser:     PlaceAttractorDenoiser (in H' space)
        phis:         known phases
        mode:         'phi_first' or 'map_first'
        Npos:         spatial positions
        grid_attractor: 'argmax' or 'nn'
        module_gbooks: for 'nn' grid attractor
        module_sizes:  for 'nn' grid attractor
        expand_mode:  'fourier' or 'random'
        random_blocks_cache: ndarray, shape (nruns, (nph-2)*Np, Npatts_stored)
                      Cached random blocks for random mode. Required if
                      expand_mode='random' and nph > 2.

    Returns:
        errpc, errgc, errsens, errsenscup, errsensl1  — each shape (nruns,)
    """
    Ns = sbook.shape[0]
    Ng = gbook_phased.shape[0]
    nruns = sinit.shape[0]
    nph = len(phis) if phis is not None else 1
    dim_mult = harmonic_dim(nph)

    # ------ Step 1: s → H' ------
    s_in = sinit[:, :, :Npatts]
    H_pred = Wps @ s_in                                  # (nruns, dim_mult*Np, Npatts)

    # ------ Step 2: Place attractor in H' space ------
    H_pred = denoiser.denoise_batch(H_pred)

    # ------ Collapse H'→complex p (first 2Np: Re + j*Im) ------
    p = fourier_collapse_to_complex(H_pred, Np)           # (nruns, Np, Npatts) complex

    # ------ Iterative dynamics ------
    gout = None
    for i in range(Niter):
        gin = Wgp @ p                                     # (nruns, Ng, Npatts)
        if grid_attractor == 'nn' and module_gbooks is not None:
            g = gridNN_2d_complex(gin, module_gbooks, module_sizes, phis=phis)
        else:
            g = gridCAN_2d_complex(gin, lambdas, phis=phis)
        p_raw = Wpg @ g                                   # (nruns, Np, Npatts)
        p = _phase_demod_nonlin_batch(p_raw, thresh, phis)

        # Expand p→H' for place attractor, then collapse back
        if expand_mode == 'random' and random_blocks_cache is not None:
            H_cur = random_expand_from_complex(
                p, phis, random_blocks_cache, denoiser, mode=mode, Npos=Npos)
        else:
            H_cur = fourier_expand_from_complex(p, phis, mode=mode, Npos=Npos)
        H_cur = denoiser.denoise_batch(H_cur)
        p = fourier_collapse_to_complex(H_cur, Np)
        gout = g

    pout = p.copy()

    # ------ Final: p → H' → Wsp → s ------
    if expand_mode == 'random' and random_blocks_cache is not None:
        H_final = random_expand_from_complex(
            p, phis, random_blocks_cache, denoiser, mode=mode, Npos=Npos)
    else:
        H_final = fourier_expand_from_complex(p, phis, mode=mode, Npos=Npos)
    sout = np.sign(Wsp @ H_final)                         # (nruns, Ns, Npatts)

    # ------ Error computation ------
    strue = sbook[:, :Npatts]
    gtrue = gbook_phased[:, :Npatts]
    ptrue = fourier_collapse_to_complex(H_prime[:, :, :Npatts], Np)

    p_l2_err = np.linalg.norm(pout - ptrue, axis=(1, 2)) / Np

    if gout is not None:
        g_l2_err = np.linalg.norm(gout - gtrue, axis=(1, 2)) / Ng
    else:
        g_l2_err = np.nan * np.zeros(nruns)

    s_diff = np.abs(sout - strue)
    s_l1_err = s_diff.mean(axis=(1, 2)) / 2
    s_l2_err = np.linalg.norm(sout - strue, axis=(1, 2)) / Ns

    errpc = p_l2_err
    errgc = g_l2_err
    errsens = s_l2_err
    errsenscup = np.nan * np.zeros_like(errsens)
    errsensl1 = s_l1_err

    return errpc, errgc, errsens, errsenscup, errsensl1


def senstrans_gs_vectorized_patts_complex(lambdas, Ng, Np, pflip, Niter, Npos,
                                           gbook, Npatts_lst, nruns, Ns, sbook,
                                           sparsity, phis=None, Num_phi=None,
                                           mode='phi_first',
                                           ps_learn='cca', grid_attractor='argmax',
                                           expand_mode='fourier'):
    """
    Complex version of senstrans_gs_vectorized_patts with Fourier or random H'.

    Args:
        lambdas: grid periods
        Ng: number of grid cells
        Np: number of place cells
        pflip: noise level for sensory corruption
        Niter: number of attractor iterations
        Npos: total spatial positions
        gbook: (Ng, Npos) grid codebook
        Npatts_lst: list of pattern counts to test
        nruns: number of independent runs
        Ns: number of sensory cells
        sbook: (Ns, Ntotal) sensory codebook
        sparsity: unused
        phis: known phases.  If None and Num_phi is given, auto-selected
              (for fourier mode) or required (for random mode).
        Num_phi: int, number of phases.  For fourier mode, auto-selects phis.
                 For random mode, if phis is not given, auto-selects phis too.
        mode: 'phi_first' or 'map_first'
        ps_learn: 'cca' only
        grid_attractor: 'argmax' or 'nn'
        expand_mode: 'fourier' or 'random'
        rng_seed: int, seed for random blocks (only used when expand_mode='random')

    Returns:
        err_pc, err_gc, err_sens, err_senscup, err_sensl1, H_prime_book
        Each error has shape (len(Npatts_lst), nruns).
        H_prime_book: (nruns, dim_mult*Np, Ntotal) — the expanded codebook.
    """
    # Resolve phases
    if Num_phi is not None:
        phis_arr = auto_select_phis(Num_phi)
    elif phis is not None and len(phis) > 0:
        phis_arr = np.asarray(phis, dtype=np.float64)
    else:
        phis_arr = np.array([0.0])
    nph = len(phis_arr)
    Ntotal = Npos * nph
    dim_mult = harmonic_dim(nph)

    err_pc = -1 * np.ones((len(Npatts_lst), nruns))
    err_sens = -1 * np.ones((len(Npatts_lst), nruns))
    err_senscup = -1 * np.ones((len(Npatts_lst), nruns))
    err_gc = -1 * np.ones((len(Npatts_lst), nruns))
    err_sensl1 = -1 * np.ones((len(Npatts_lst), nruns))

    # Generate Wpg: random sparse projection
    Wpg = randn(nruns, Np, Ng)
    c = 0.60
    prune = int((1 - c) * Np * Ng)
    mask = np.ones((Np, Ng))
    mask[randint(low=0, high=Np, size=prune),
         randint(low=0, high=Ng, size=prune)] = 0
    Wpg = np.multiply(mask, Wpg)

    thresh = 0.5
    print(f'thresh={thresh}')

    # Generate real-valued base pbook
    pbook_base = nonlin(np.einsum('ijk,kl->ijl', Wpg, gbook), thresh)  # (nruns, Np, Npos)

    # Learn Wgp from real base pbook/gbook (pseudoinverse)
    pbook_pinv = np.linalg.pinv(pbook_base)                      # (nruns, Npos, Np)
    Wgp = np.einsum('ij, kjl -> kil', gbook, pbook_pinv)         # (nruns, Ng, Np)

    # Build module codebooks for NN grid attractor
    module_sizes = [lam ** 2 for lam in lambdas]
    module_gbooks = None
    if grid_attractor == 'nn':
        module_gbooks = []
        idx = 0
        for ms in module_sizes:
            module_gbooks.append(gbook[idx:idx + ms, :])
            idx += ms

    # Build phased grid codebook (complex, for error computation)
    phase_factors = np.exp(1j * phis_arr)                         # (nph,)
    gbook_4d = gbook[..., None] * phase_factors[None, :]          # (Ng, Npos, nph)
    if mode == 'phi_first':
        gbook_phased = gbook_4d.reshape(Ng, Ntotal)
    elif mode == 'map_first':
        gbook_phased = gbook_4d.transpose(0, 2, 1).reshape(Ng, Ntotal)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'phi_first' or 'map_first'.")

    # Build H' codebook from pbook_base
    random_blocks = None
    if expand_mode == 'random':
        H_prime_book, random_blocks = random_expand_from_base(
            pbook_base, phis_arr, mode=mode, rng_seed=rng_seed)
    else:
        H_prime_book = fourier_expand_from_base(pbook_base, phis_arr, mode=mode)
    # shape: (nruns, dim_mult*Np, Ntotal)

    print(f'nph={nph}, Ntotal={Ntotal}, mode={mode}, ps_learn={ps_learn}, '
          f'dim_mult={dim_mult}, H_prime shape={H_prime_book.shape}, '
          f'expand_mode={expand_mode}, phis={phis_arr}')

    # Place attractor denoiser (works in H' space)
    denoiser = PlaceAttractorDenoiser()

    k = 0
    for Npatts in tqdm(Npatts_lst):
        # Learn P→S (Wsp) on H' representation
        Wsp = pseudotrain_Wsp_cca(sbook, H_prime_book, Npatts)

        # Learn S→P (Wps) on H' representation
        Wps = pseudotrain_Wps_cca(H_prime_book, sbook, Npatts)

        # Build denoiser with first Npatts H' patterns
        denoiser.store_patterns(H_prime_book[0, :, :Npatts])

        # Cache random blocks for this Npatts subset
        rb_cache = None
        if expand_mode == 'random' and random_blocks is not None:
            rb_cache = random_blocks[:, :, :Npatts]

        # Prepare corrupted sensory input
        srep = np.repeat(sbook[None, :], nruns, axis=0)
        sinit = corrupt_p_1(srep, p=pflip)

        # Run dynamics with H' representation
        err_pc[k], err_gc[k], err_sens[k], _, err_sensl1[k] = \
            dynamics_gs_vectorized_patts_complex(
                sinit, Niter, sbook, H_prime_book, gbook_phased,
                Wgp, Wpg, Wsp, Wps,
                lambdas, sparsity, thresh, Npatts, Np,
                denoiser, phis=phis_arr, mode=mode, Npos=Npos,
                grid_attractor=grid_attractor,
                module_gbooks=module_gbooks, module_sizes=module_sizes,
                expand_mode=expand_mode,
                random_blocks_cache=rb_cache
            )

        k += 1

    return err_pc, err_gc, err_sens, err_senscup, err_sensl1, H_prime_book


def capacity_complex(sensory_model, lambdas, Ng, Np_lst, pflip, Niter, Npos,
                     gbook, Npatts_lst, nruns, Ns, sbook, sparsity, phis=None,
                     Num_phi=None, mode='phi_first', ps_learn='cca',
                     grid_attractor='argmax', expand_mode='fourier', rng_seed=42):
    """
    Capacity sweep: run storage test over multiple Np values.

    Args:
        sensory_model: capacity test function
        lambdas, Ng, Np_lst, pflip, etc.: same as real version
        phis: known phases (passed through). Overridden by Num_phi if given.
        Num_phi: int, number of phases (auto-selected if given)
        mode: 'phi_first' or 'map_first'
        ps_learn: 'cca'
        grid_attractor: 'argmax' or 'nn'
        expand_mode: 'fourier' or 'random'
        rng_seed: int, seed for random blocks (only for expand_mode='random')

    Returns:
        err_pc, err_gc, err_sens, err_senscup, err_sensl1, H_prime_all
        First 5 have shape (len(Np_lst), len(Npatts_lst), nruns).
        H_prime_all is a list of len(Np_lst), each element is
        H_prime_book of shape (nruns, dim_mult*Np, Ntotal) for that Np.
    """
    err_gc = -1 * np.ones((len(Np_lst), len(Npatts_lst), nruns))
    err_pc = -1 * np.ones((len(Np_lst), len(Npatts_lst), nruns))
    err_sens = -1 * np.ones((len(Np_lst), len(Npatts_lst), nruns))
    err_senscup = -1 * np.ones((len(Np_lst), len(Npatts_lst), nruns))
    err_sensl1 = -1 * np.ones((len(Np_lst), len(Npatts_lst), nruns))
    H_prime_all = []

    for l, Np in enumerate(Np_lst):
        print(f"l = {l}")
        result = sensory_model(lambdas, Ng, Np, pflip, Niter, Npos,
                         gbook, Npatts_lst, nruns, Ns, sbook, sparsity,
                         phis=phis, Num_phi=Num_phi, mode=mode,
                         ps_learn=ps_learn, grid_attractor=grid_attractor,
                         expand_mode=expand_mode, rng_seed=rng_seed)
        err_pc[l], err_gc[l], err_sens[l], err_senscup[l], err_sensl1[l] = result[:5]
        H_prime_all.append(result[5])

    return err_pc, err_gc, err_sens, err_senscup, err_sensl1, H_prime_all
