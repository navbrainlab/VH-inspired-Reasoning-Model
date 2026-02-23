"""
Phase utilities for Complex Vector-HASH.

Provides:
- phase_demodulate: Extract signed magnitudes and global phase from complex vectors
- module_wise_NN_2d_complex: Complex-valued modular NN attractor (single vector)
- gridCAN_2d_complex: Complex-valued grid attractor (batch processing)
"""

import numpy as np


def phase_demodulate(z_complex, phis=None):
    """
    Phase demodulation: extract signed real magnitudes and estimate global phase.

    For complex vector z = signed_real * e^{i * phi_global}, recover both
    the signed real magnitudes and the global phase.

    Because Wpg has negative entries, some elements of z may have their
    sign absorbed into the phase (angle becomes phi - pi). This function
    corrects for that.

    Args:
        z_complex: 1D complex numpy array
        phis: optional 1D array of known phases for snapping.
              If provided, the estimated phase is snapped to nearest value.

    Returns:
        z_signed: 1D float64 array, signed magnitudes
        phi_hat: float, estimated global phase in [0, pi)
    """
    z_complex = np.asarray(z_complex, dtype=np.complex128)
    z_mag = np.abs(z_complex)
    z_phase = np.angle(z_complex)

    # Special case: purely real vector (all phases are 0 or +/-pi)
    z_phase_mod_pi = np.abs(np.mod(z_phase + 1e-10, np.pi))
    if np.all(z_phase_mod_pi < 1e-5):
        return z_complex.real.astype(np.float64), 0.0

    # Step 1: correct negative phases
    # Elements with angle < 0 have absorbed a negative sign
    z_phase_corr = z_phase.copy()
    mask = z_phase < 0
    z_signed = z_mag.copy()
    z_signed[mask] = -z_signed[mask]
    z_phase_corr[mask] += np.pi

    # Step 2: weighted circular mean for global phase estimation
    weights = z_mag.flatten()
    z_phase_flat = z_phase_corr.flatten()
    phasor_sum = np.sum(weights * np.exp(1j * z_phase_flat))
    phi = np.angle(phasor_sum)
    if phi < 0:
        phi += np.pi

    # Wrap to [0, pi)
    phi = np.mod(phi, np.pi)
    if phi > 0.75 * np.pi:
        phi = 0.0

    # Snap to nearest known phase if provided
    if phis is not None and len(phis) > 0:
        phis_arr = np.asarray(phis, dtype=np.float64)
        phis_mod = np.mod(phis_arr, np.pi)
        nearest_idx = int(np.argmin(np.abs(phis_mod - phi)))
        phi = float(phis_mod[nearest_idx])

    return z_signed.astype(np.float64), phi


def module_wise_NN_2d_complex(gin, module_gbooks, module_sizes, phis=None):
    """
    Complex-valued modular nearest-neighbor attractor for a single grid vector.

    Processes input of shape (1, Ng, 1):
    1. Phase-demodulate to get signed real values and global phase
    2. Apply real-valued module-wise nearest-neighbor
    3. Reconstruct complex output with estimated phase

    Args:
        gin: complex array, shape (1, Ng, 1) or (nruns, Ng, 1)
        module_gbooks: list of module codebooks (real)
        module_sizes: list of module sizes (e.g. [9, 16, 25] for lambdas=[3,4,5])
        phis: optional known phases for snapping

    Returns:
        g_out: complex array, same shape as gin
    """
    from src.assoc_utils_np_2D import module_wise_NN_2d

    gin_signed, phi_hat = phase_demodulate(gin.flatten(), phis=phis)
    gin_signed = gin_signed.reshape(gin.shape)
    g_mag_clean = module_wise_NN_2d(gin_signed, module_gbooks, module_sizes)
    g_out = g_mag_clean.astype(np.complex128) * np.exp(1j * phi_hat)
    return g_out


def gridCAN_2d_complex(gs_complex, lambdas, phis=None):
    """
    Complex-valued grid attractor (batch processing).

    For each pattern in the batch:
    1. Phase-demodulate to get signed real grid vector and global phase
    2. Apply argmax per module (standard grid CAN attractor)
    3. Reconstruct complex output with estimated phase

    For real inputs, falls back to standard gridCAN_2d.

    Args:
        gs_complex: (nruns, Ng, Npatts) array, real or complex
        lambdas: list of grid periods, e.g. [3, 4, 5]
        phis: optional known phases for snapping

    Returns:
        g_clean: (nruns, Ng, Npatts) array (one-hot per module * e^{i*phi})
    """
    if not np.iscomplexobj(gs_complex):
        from src.assoc_utils_np import gridCAN_2d
        return gridCAN_2d(gs_complex, lambdas)

    # Fast path: effectively real (imag negligible)
    if np.max(np.abs(gs_complex.imag)) < 1e-10:
        from src.assoc_utils_np import gridCAN_2d
        return gridCAN_2d(gs_complex.real, lambdas).astype(np.complex128)

    nruns, Ng, Npatts = gs_complex.shape
    ls = [l ** 2 for l in lambdas]
    gout = np.zeros_like(gs_complex)

    for ru in range(nruns):
        for patt in range(Npatts):
            vec = gs_complex[ru, :, patt]
            signed_real, phi_hat = phase_demodulate(vec, phis=phis)

            # Argmax per module on signed real values
            idx = 0
            g_real = np.zeros(Ng)
            for mod_size in ls:
                g_mod = signed_real[idx:idx + mod_size]
                best = np.argmax(g_mod)
                g_real[idx + best] = 1.0
                idx += mod_size

            gout[ru, :, patt] = g_real * np.exp(1j * phi_hat)

    return gout


def gridNN_2d_complex(gs_complex, module_gbooks, module_sizes, phis=None):
    """
    Complex-valued grid attractor using module-wise nearest-neighbor (batch).

    Same pipeline as gridCAN_2d_complex but uses nearest-neighbor matching
    against module codebooks instead of argmax. For one-hot grid codes the two
    are equivalent; NN generalises to non-one-hot codes.

    Args:
        gs_complex: (nruns, Ng, Npatts) array, real or complex
        module_gbooks: list of module codebooks, each (mod_size, Npos)
        module_sizes: list of int, number of cells per module
        phis: optional known phases for snapping

    Returns:
        g_clean: (nruns, Ng, Npatts) array
    """
    if not np.iscomplexobj(gs_complex):
        from src.assoc_utils_np_2D import module_wise_NN_2d
        nruns, Ng, Npatts = gs_complex.shape
        gout = np.zeros_like(gs_complex)
        for patt in range(Npatts):
            vec = gs_complex[:, :, patt:patt+1]  # (nruns, Ng, 1)
            gout[:, :, patt:patt+1] = module_wise_NN_2d(vec, module_gbooks, module_sizes)
        return gout

    nruns, Ng, Npatts = gs_complex.shape
    gout = np.zeros_like(gs_complex)

    for ru in range(nruns):
        for patt in range(Npatts):
            vec = gs_complex[ru, :, patt]
            g_clean = module_wise_NN_2d_complex(
                vec.reshape(1, Ng, 1), module_gbooks, module_sizes, phis=phis
            )
            gout[ru, :, patt] = g_clean.flatten()

    return gout
