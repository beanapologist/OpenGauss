"""Numerical validation of the mathematical identities from the Kernel
formal-lean proofs.

Each test class mirrors a section from the corresponding Lean 4 source file
at https://github.com/beanapologist/Kernel/tree/main/formal-lean and verifies
the same properties using Python's standard ``cmath``/``math`` libraries.

Sections covered
────────────────
  1.  CriticalEigenvalue.lean  — μ, C(r), silver ratio δS, orbit trichotomy,
                                  palindrome residual, Pythagorean coherence identity
  2.  TimeCrystal.lean         — time-evolution operator U(H,t) = exp(−i·H·t)
  3.  FineStructure.lean       — fine-structure constant α_FS = 1/137 and its
                                  effect on coherence / Floquet quasi-energy

All assertions use ``pytest.approx`` so that floating-point rounding is
handled consistently.
"""

import cmath
import math

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Helpers — reproduce the definitions from CriticalEigenvalue.lean in Python
# ─────────────────────────────────────────────────────────────────────────────

def mu() -> complex:
    """Critical eigenvalue μ = exp(I · 3π/4)."""
    return cmath.exp(1j * 3 * math.pi / 4)


def coherence(r: float) -> float:
    """Coherence function C(r) = 2r / (1 + r²) for r ≥ 0."""
    return 2 * r / (1 + r * r)


SILVER_RATIO = 1 + math.sqrt(2)   # δS ≈ 2.41421356…
ETA = 1 / math.sqrt(2)            # η = 1/√2


def palindrome_residual(r: float) -> float:
    """Palindrome residual R(r) = (r − 1/r) / δS."""
    return (r - 1 / r) / SILVER_RATIO


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Critical Eigenvalue  (CriticalEigenvalue.lean §1–6)
# ─────────────────────────────────────────────────────────────────────────────

class TestCriticalEigenvalue:
    """μ = exp(I · 3π/4) core properties."""

    def test_mu_cartesian_form(self):
        """μ = (−1 + i) / √2  (Theorem mu_eq_cart)."""
        expected = (-1 + 1j) / math.sqrt(2)
        assert mu() == pytest.approx(expected, abs=1e-12)

    def test_mu_abs_one(self):
        """|μ| = 1  (Theorem mu_abs_one)."""
        assert abs(mu()) == pytest.approx(1.0, abs=1e-12)

    def test_mu_pow_eight(self):
        """μ^8 = 1  (Theorem mu_pow_eight, 8-cycle closure)."""
        assert mu() ** 8 == pytest.approx(1.0 + 0j, abs=1e-10)

    def test_mu_powers_distinct(self):
        """μ^0, …, μ^7 are pairwise distinct  (Theorem mu_powers_distinct)."""
        powers = [mu() ** k for k in range(8)]
        # Round to 10 decimal places to bucket identical values
        rounded = [(round(z.real, 10), round(z.imag, 10)) for z in powers]
        assert len(set(rounded)) == 8, "expected 8 distinct powers of μ"

    def test_mu_real_part(self):
        """Re(μ) = −1/√2 ≈ −0.7071."""
        assert mu().real == pytest.approx(-1 / math.sqrt(2), abs=1e-12)

    def test_mu_imag_part(self):
        """Im(μ) = 1/√2 ≈ 0.7071."""
        assert mu().imag == pytest.approx(1 / math.sqrt(2), abs=1e-12)

    def test_mu_angle(self):
        """arg(μ) = 3π/4  (μ is at 135° on the unit circle)."""
        assert cmath.phase(mu()) == pytest.approx(3 * math.pi / 4, abs=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Coherence Function  (CriticalEigenvalue.lean §5, 8, 13)
# ─────────────────────────────────────────────────────────────────────────────

class TestCoherenceFunction:
    """C(r) = 2r / (1 + r²): bounds, symmetry, and monotonicity."""

    def test_coherence_at_one(self):
        """C(1) = 1  (coherence_eq_one_iff ←)."""
        assert coherence(1.0) == pytest.approx(1.0, abs=1e-12)

    def test_coherence_le_one(self):
        """C(r) ≤ 1 for r > 0  (coherence_le_one)."""
        for r in [0.1, 0.5, 1.0, 2.0, 10.0, 100.0]:
            assert coherence(r) <= 1.0 + 1e-12, f"C({r}) = {coherence(r)} > 1"

    def test_coherence_lt_one_away_from_one(self):
        """C(r) < 1 for r ≠ 1  (coherence_lt_one)."""
        for r in [0.1, 0.5, 1.5, 2.0, 10.0]:
            assert coherence(r) < 1.0 - 1e-12, f"C({r}) = {coherence(r)} not < 1"

    def test_coherence_positive(self):
        """C(r) > 0 for r > 0  (coherence_pos)."""
        for r in [0.001, 0.5, 1.0, 5.0, 1000.0]:
            assert coherence(r) > 0.0, f"C({r}) = {coherence(r)} not > 0"

    def test_coherence_symmetry(self):
        """C(r) = C(1/r) for r > 0  (coherence_symm, even symmetry about r=1)."""
        for r in [0.1, 0.5, 2.0, 5.0, 10.0]:
            assert coherence(r) == pytest.approx(coherence(1 / r), abs=1e-12)

    def test_coherence_strictly_increasing_below_one(self):
        """0 < r < s ≤ 1 → C(r) < C(s)  (coherence_strictMono)."""
        pairs = [(0.1, 0.5), (0.3, 0.7), (0.5, 0.9), (0.8, 1.0)]
        for r, s in pairs:
            assert coherence(r) < coherence(s), f"C({r}) >= C({s})"

    def test_coherence_strictly_decreasing_above_one(self):
        """1 ≤ r < s → C(s) < C(r)  (coherence_strictAnti)."""
        pairs = [(1.0, 1.5), (1.5, 2.0), (2.0, 5.0), (5.0, 10.0)]
        for r, s in pairs:
            assert coherence(s) < coherence(r), f"C({s}) >= C({r})"

    def test_canonical_norm(self):
        """η² + |μ · η|² = 1  (canonical_norm)."""
        # |μ · η|² = |μ|² · η² = 1 · η² = η² so η² + η² = 2η² = 2·(1/√2)² = 1.
        canonical_norm_sum = ETA ** 2 + abs(mu() * ETA) ** 2
        assert canonical_norm_sum == pytest.approx(1.0, abs=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Silver Ratio  (CriticalEigenvalue.lean §7, 20)
# ─────────────────────────────────────────────────────────────────────────────

class TestSilverRatio:
    """δS = 1 + √2 algebraic identities."""

    def test_silver_ratio_value(self):
        """δS = 1 + √2 ≈ 2.41421356."""
        assert SILVER_RATIO == pytest.approx(1 + math.sqrt(2), abs=1e-12)

    def test_silver_ratio_positive(self):
        """δS > 0  (silverRatio_pos)."""
        assert SILVER_RATIO > 0

    def test_silver_ratio_mul_conj(self):
        """δS · (√2 − 1) = 1  (silverRatio_mul_conj)."""
        assert SILVER_RATIO * (math.sqrt(2) - 1) == pytest.approx(1.0, abs=1e-12)

    def test_silver_ratio_squared(self):
        """δS² = 2·δS + 1  (silverRatio_sq)."""
        assert SILVER_RATIO ** 2 == pytest.approx(2 * SILVER_RATIO + 1, abs=1e-12)

    def test_silver_ratio_inv(self):
        """1/δS = √2 − 1  (silverRatio_inv)."""
        assert 1 / SILVER_RATIO == pytest.approx(math.sqrt(2) - 1, abs=1e-12)

    def test_silver_ratio_continued_fraction(self):
        """δS = 2 + 1/δS  (silverRatio_cont_frac, continued-fraction fixed point)."""
        assert SILVER_RATIO == pytest.approx(2 + 1 / SILVER_RATIO, abs=1e-12)

    def test_silver_ratio_min_poly(self):
        """δS² − 2·δS − 1 = 0  (silverRatio_minPoly)."""
        assert SILVER_RATIO ** 2 - 2 * SILVER_RATIO - 1 == pytest.approx(0.0, abs=1e-12)

    def test_coherence_at_silver_is_eta(self):
        """C(δS) = η = 1/√2  (coherence_at_silver_is_eta)."""
        assert coherence(SILVER_RATIO) == pytest.approx(ETA, abs=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Palindrome Residual  (CriticalEigenvalue.lean §9)
# ─────────────────────────────────────────────────────────────────────────────

class TestPalindromeResidual:
    """R(r) = (r − 1/r) / δS properties."""

    def test_residual_zero_at_one(self):
        """R(1) = 0  (palindrome_residual_zero_iff ←)."""
        assert palindrome_residual(1.0) == pytest.approx(0.0, abs=1e-12)

    def test_residual_zero_iff_one(self):
        """R(r) = 0 ↔ r = 1  (palindrome_residual_zero_iff)."""
        for r in [0.1, 0.5, 2.0, 5.0]:
            assert abs(palindrome_residual(r)) > 1e-10, f"R({r}) unexpectedly zero"

    def test_residual_positive_above_one(self):
        """R(r) > 0 for r > 1  (palindrome_residual_pos)."""
        for r in [1.001, 1.5, 2.0, 5.0, 10.0]:
            assert palindrome_residual(r) > 0, f"R({r}) not > 0"

    def test_residual_negative_below_one(self):
        """R(r) < 0 for 0 < r < 1  (palindrome_residual_neg)."""
        for r in [0.001, 0.1, 0.5, 0.9, 0.999]:
            assert palindrome_residual(r) < 0, f"R({r}) not < 0"

    def test_residual_antisymmetry(self):
        """R(1/r) = −R(r)  (palindrome_residual_antisymm)."""
        for r in [0.2, 0.5, 2.0, 5.0]:
            assert palindrome_residual(1 / r) == pytest.approx(-palindrome_residual(r), abs=1e-12)

    def test_palindrome_sum_zero(self):
        """R(r) + R(1/r) = 0  (palindrome_sum_zero)."""
        for r in [0.3, 1.5, 3.0, 7.0]:
            assert palindrome_residual(r) + palindrome_residual(1 / r) == pytest.approx(
                0.0, abs=1e-12
            )


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Lyapunov–Coherence Duality  (CriticalEigenvalue.lean §10, 19)
# ─────────────────────────────────────────────────────────────────────────────

class TestLyapunovCoherenceDuality:
    """C(exp λ) = sech λ  (Theorem 14 / lyapunov_coherence_sech)."""

    def test_coherence_exp_is_sech(self):
        """C(e^λ) = sech λ for various λ."""
        for lam in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
            c_val = coherence(math.exp(lam))
            sech_val = 1 / math.cosh(lam)
            assert c_val == pytest.approx(sech_val, abs=1e-12)

    def test_coherence_is_sech_of_log(self):
        """C(r) = sech(log r)  (coherence_is_sech_of_log, master Lyapunov link)."""
        for r in [0.1, 0.5, 1.0, 2.0, 5.0]:
            c_val = coherence(r)
            sech_val = 1 / math.cosh(math.log(r))
            assert c_val == pytest.approx(sech_val, abs=1e-12)

    def test_lyapunov_bound(self):
        """C(exp λ) ≤ 1  (lyapunov_bound)."""
        for lam in [-3.0, -1.0, 0.0, 1.0, 3.0]:
            assert coherence(math.exp(lam)) <= 1.0 + 1e-12

    def test_coherence_orbit_sech(self):
        """C(r^n) = sech(n · log r)  (coherence_orbit_sech)."""
        r, n = 2.0, 3
        assert coherence(r ** n) == pytest.approx(1 / math.cosh(n * math.log(r)), abs=1e-12)

    def test_coherence_orbit_at_one(self):
        """C(1^n) = 1  (orbit_coherence_at_one)."""
        for n in range(1, 6):
            assert coherence(1.0 ** n) == pytest.approx(1.0, abs=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Pythagorean Coherence Identity  (CriticalEigenvalue.lean §18, 22)
# ─────────────────────────────────────────────────────────────────────────────

class TestPythagoreanCoherence:
    """C(r)² + ((r²−1)/(1+r²))² = 1  (coherence_pythagorean)."""

    def test_pythagorean_identity(self):
        """C(r)² + ((r²−1)/(1+r²))² = 1 for r > 0."""
        for r in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            c = coherence(r)
            complement_term = (r ** 2 - 1) / (1 + r ** 2)
            assert c ** 2 + complement_term ** 2 == pytest.approx(1.0, abs=1e-12)

    def test_coherence_lyapunov_pythag(self):
        """C(e^λ)² + tanh²λ = 1  (coherence_lyapunov_pythag)."""
        for lam in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            c = coherence(math.exp(lam))
            t = math.tanh(lam)
            assert c ** 2 + t ** 2 == pytest.approx(1.0, abs=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — Orbit Magnitude and Trichotomy  (CriticalEigenvalue.lean §12)
# ─────────────────────────────────────────────────────────────────────────────

class TestOrbitTrichotomy:
    """|(r·μ)^n| = r^n  and the three-way trichotomy for r > 0."""

    def test_scaled_orbit_abs(self):
        """|(r·μ)^n| = r^n  (scaled_orbit_abs)."""
        for r in [0.5, 1.0, 2.0]:
            for n in range(1, 9):
                assert abs((r * mu()) ** n) == pytest.approx(r ** n, abs=1e-10)

    def test_unit_orbit_stable(self):
        """r = 1: |(1·μ)^n| = 1 for all n  (trichotomy_unit_orbit)."""
        for n in range(0, 10):
            assert abs(mu() ** n) == pytest.approx(1.0, abs=1e-10)

    def test_expanding_orbit(self):
        """r > 1: magnitudes strictly increasing  (trichotomy_grow)."""
        r = 1.5
        magnitudes = [abs((r * mu()) ** n) for n in range(1, 9)]
        for i in range(len(magnitudes) - 1):
            assert magnitudes[i] < magnitudes[i + 1]

    def test_contracting_orbit(self):
        """0 < r < 1: magnitudes strictly decreasing  (trichotomy_shrink)."""
        r = 0.7
        magnitudes = [abs((r * mu()) ** n) for n in range(1, 9)]
        for i in range(len(magnitudes) - 1):
            assert magnitudes[i] > magnitudes[i + 1]

    def test_mu_pow_abs(self):
        """|μ^n| = 1 for all n  (mu_pow_abs)."""
        for n in range(-8, 9):
            assert abs(mu() ** n) == pytest.approx(1.0, abs=1e-10)

    def test_mu_inv_eq_pow7(self):
        """μ⁷ = μ⁻¹ in the 8-cycle  (mu_inv_eq_pow7)."""
        assert mu() ** 7 == pytest.approx(1 / mu(), abs=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# Section 8 — Z/8Z Rotational Memory  (CriticalEigenvalue.lean §15)
# ─────────────────────────────────────────────────────────────────────────────

class TestZ8ZRotationalMemory:
    """μ^(j+8) = μ^j  and the Z/8Z address reconstruction identity."""

    def test_z8z_period(self):
        """(n + 8) % 8 = n % 8  (z8z_period)."""
        for n in range(-20, 20):
            assert (n + 8) % 8 == n % 8

    def test_mu_z8z_period(self):
        """μ^(j+8) = μ^j  (mu_z8z_period)."""
        for j in range(0, 16):
            assert mu() ** (j + 8) == pytest.approx(mu() ** j, abs=1e-10)

    def test_z8z_reconstruction(self):
        """addr % 8 + 8 * (addr // 8) = addr  (z8z_reconstruction)."""
        for addr in range(0, 50):
            assert addr % 8 + 8 * (addr // 8) == addr

    def test_nullslice_coverage_bijective(self):
        """k ↦ 3k mod 8 is a bijection on ZMod 8  (nullslice_coverage_bijective)."""
        outputs = [(3 * k) % 8 for k in range(8)]
        # Injectivity: all 8 outputs are distinct
        assert len(set(outputs)) == 8
        # Surjectivity: the image covers all of {0,…,7}
        assert set(outputs) == set(range(8))

    def test_nullslice_involution(self):
        """3·(3·k) = k in ZMod 8  (nullslice_involution)."""
        for k in range(8):
            assert (3 * (3 * k)) % 8 == k % 8


# ─────────────────────────────────────────────────────────────────────────────
# Section 9 — Palindrome Arithmetic  (CriticalEigenvalue.lean §14)
# ─────────────────────────────────────────────────────────────────────────────

class TestPalindromeArithmetic:
    """Digit-palindrome arithmetic used in the Kernel torus period structure."""

    def test_palindrome_comp(self):
        """987654321 = 8 × 123456789 + 9  (palindrome_comp)."""
        assert 987654321 == 8 * 123456789 + 9

    def test_precession_period_factor(self):
        """9 × 13717421 = 123456789  (precession_period_factor)."""
        assert 9 * 13717421 == 123456789

    def test_precession_gcd_one(self):
        """gcd(8, 13717421) = 1  (precession_gcd_one, coprime periods)."""
        assert math.gcd(8, 13717421) == 1

    def test_precession_lcm(self):
        """lcm(8, 13717421) = 8 · 13717421  (precession_lcm, torus super-period)."""
        assert math.lcm(8, 13717421) == 8 * 13717421


# ─────────────────────────────────────────────────────────────────────────────
# Section 10 — Time Evolution  (TimeCrystal.lean §1–4)
# ─────────────────────────────────────────────────────────────────────────────

def time_evolution(H: float, t: float) -> complex:
    """U(H, t) = exp(−I · H · t)  — single-mode time evolution operator."""
    return cmath.exp(-1j * H * t)


class TestTimeEvolution:
    """Schrödinger-picture time evolution U(H,t) = exp(−iHt)."""

    def test_identity_at_zero(self):
        """U(H, 0) = 1  (timeEvolution_zero)."""
        for H in [0.0, 1.0, -2.5, 100.0]:
            assert time_evolution(H, 0.0) == pytest.approx(1.0 + 0j, abs=1e-12)

    def test_unitary(self):
        """|U(H, t)| = 1  (timeEvolution_unitary)."""
        for H in [0.5, 1.0, 3.0]:
            for t in [0.0, 0.5, 1.0, 2.0, 10.0]:
                assert abs(time_evolution(H, t)) == pytest.approx(1.0, abs=1e-12)

    def test_floquet_period_doubling(self):
        """Floquet phase φ = π implies period doubling: U^2 = 1  (timeCrystal_period_double)."""
        phi = math.pi
        U = cmath.exp(-1j * phi)
        assert U ** 2 == pytest.approx(1.0 + 0j, abs=1e-12)

    def test_floquet_phase_unitary(self):
        """|e^{−iφ}| = 1  (floquetPhase_unitary)."""
        for phi in [0.0, math.pi / 4, math.pi / 2, math.pi, 2 * math.pi]:
            assert abs(cmath.exp(-1j * phi)) == pytest.approx(1.0, abs=1e-12)

    def test_time_crystal_quasi_energy(self):
        """Quasi-energy ε_quasi = π/T satisfies e^{−i·ε_quasi·T} = −1  (quasiEnergy_half_drive)."""
        T = 2.0
        eps_quasi = math.pi / T
        assert time_evolution(eps_quasi, T) == pytest.approx(-1.0 + 0j, abs=1e-12)

    def test_time_crystal_eight_period(self):
        """8-period structure: μ is the Kernel eigenvalue for a time crystal with
        quasi-energy ε = 3π/(4T) → U(ε, 8T) = 1  (kernelEigenvalue_recipe)."""
        T = 1.0
        eps = 3 * math.pi / (4 * T)
        assert time_evolution(eps, 8 * T) == pytest.approx(1.0 + 0j, abs=1e-10)

    def test_phase_full_cycle(self):
        """D · (2π/D) = 2π  (phase_full_cycle)."""
        for D in [1, 2, 4, 8, 16]:
            assert D * (2 * math.pi / D) == pytest.approx(2 * math.pi, abs=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# Section 11 — Fine Structure Constant  (FineStructure.lean §1–6)
# ─────────────────────────────────────────────────────────────────────────────

ALPHA_FS = 1 / 137  # Sommerfeld rational approximation


class TestFineStructureConstant:
    """α_FS = 1/137 and derived quantities."""

    def test_alpha_fs_positive(self):
        """α_FS > 0  (alpha_fs_pos)."""
        assert ALPHA_FS > 0

    def test_alpha_fs_lt_one(self):
        """α_FS < 1  (alpha_fs_lt_one)."""
        assert ALPHA_FS < 1

    def test_alpha_fs_small(self):
        """α_FS < 0.01  (α ≈ 1/137, small coupling)."""
        assert ALPHA_FS < 0.01

    def test_alpha_fs_value(self):
        """α_FS ≈ 7.299 × 10⁻³."""
        assert ALPHA_FS == pytest.approx(1 / 137, abs=1e-15)

    def test_alpha_fs_squared_positive(self):
        """α_FS² > 0  (alpha_fs_sq_pos)."""
        assert ALPHA_FS ** 2 > 0

    def test_em_coherence_reduction(self):
        """C_EM(r) = (1 − α_FS) · C(r) < C(r) for r > 0  (coherence_em_reduction)."""
        for r in [0.5, 1.0, 2.0]:
            c_em = (1 - ALPHA_FS) * coherence(r)
            assert c_em < coherence(r) + 1e-14

    def test_floquet_quasi_energy_fine_shift(self):
        """ε_F_fine = ε_F · (1 + α_FS²) > ε_F  (floquet_fine_shift_pos)."""
        eps_F = 0.5
        eps_fine = eps_F * (1 + ALPHA_FS ** 2)
        assert eps_fine > eps_F

    def test_rydberg_energy_negative(self):
        """E_n = −1/n² < 0  (rydberg_energy_neg)."""
        for n in [1, 2, 3, 4, 5]:
            E_n = -1 / n ** 2
            assert E_n < 0

    def test_rydberg_energy_seq_increasing(self):
        """E_n < E_{n+1} for n ≥ 1: energy levels rise toward zero  (rydberg_seq_increasing)."""
        for n in range(1, 6):
            assert -1 / n ** 2 < -1 / (n + 1) ** 2

    def test_fine_structure_energy_splitting(self):
        """Δε = α_FS² · ε_base > 0  (fineStructure_energy_split_pos)."""
        eps_base = 1.0
        delta = ALPHA_FS ** 2 * eps_base
        assert delta > 0
