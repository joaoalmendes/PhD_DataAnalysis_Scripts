#!/usr/bin/env python3
"""
Expected d_eff and Fraunhofer period for a thin-flake twisted cuprate JJ
using the London expression of Zhao et al. (Science 2023, Suppl. Eq. S17)

Now takes λ_ab and λ_c as independent parameters and reports γ = λ_c/λ_ab.
"""

import numpy as np

# ------------------------------------------------------------------
# Physical constants
Phi0 = 2.067833848e-15   # Wb = T m²

# ------------------------------------------------------------------
# Your device parameters (edit these)
t1_nm   = 77.5      # thickness of first flake (nm)
t2_nm   = 87.5      # thickness of second flake (nm)
w_um    = 10.0      # junction width perpendicular to B (µm)
s_nm    = 1.2       # van-der-Waals gap (nm)

# London penetration depths to explore (nm)
# You can add or remove values freely
lambda_ab_list = [260, 300, 350, 438]
lambda_c_list  = [1e5, 1.37e5, 1.5e5, 1.75e5, 2.0e5, 2.5e5, 2.75e5, 3e5, 3.5e5]   # 100–400 µm

# ------------------------------------------------------------------
def d_eff(t1, t2, w, lab, lc, s=1.2, Nsum=200):
    """
    All lengths in nm.
    Returns d_eff in nm.
    """
    def term(t):
        # first (tanh) contribution
        part1 = lab * np.tanh(t / (2 * lab))
        # finite-width correction series
        beta = (w * np.pi * lab) / (t * lc)
        ssum = 0.0
        for n in range(Nsum):
            odd = 2*n + 1
            ssum += (1.0 / odd**3) * (1.0 - np.exp(-odd * beta)) / beta
        part2 = (4 * t / np.pi**2) * ssum
        return part1 - part2
    return s + term(t1) + term(t2)

# ------------------------------------------------------------------
print("Expected d_eff and ΔB for your geometry")
print(f"t1={t1_nm} nm, t2={t2_nm} nm, w={w_um} µm, s={s_nm} nm\n")
print(f"{'λ_ab (nm)':>10} {'λ_c (nm)':>12} {'γ':>8} {'d_eff (nm)':>12} {'ΔB (mT)':>10}")
print("-" * 56)

w_nm = w_um * 1e3   # convert to nm for the formula

for lab in lambda_ab_list:
    for lc in lambda_c_list:
        gamma = lc / lab
        d = d_eff(t1_nm, t2_nm, w_nm, lab, lc, s=s_nm)
        # ΔB = Φ0 / (w * d)   (everything in SI)
        d_m = d * 1e-9
        w_m = w_um * 1e-6
        dB_mT = Phi0 / (w_m * d_m) * 1e3
        print(f"{lab:10.0f} {lc:12.0f} {gamma:8.1f} {d:12.2f} {dB_mT:10.2f}")
