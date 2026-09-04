"""Test the 9-step design calculator with a known photoredox example.

Expected values (hand-calculated):
  Step 1: T=25°C, C₀=0.1M, t_batch=86400s, X=0.72
  Step 2: k_batch = -ln(0.28)/86400 = 1.474e-5 s⁻¹
          IF = 48, k_flow = 7.077e-4 s⁻¹
          τ = 86400/48 = 1800 s = 30 min
  Step 3: Q = 0.5 mL/min, V_R = 30*0.5 = 15 mL
          d = 1.0 mm, L = 4*15e-6/(π*1e-6) = 19.10 m
  Step 4: v = 4*8.33e-9/(π*1e-6) = 0.01061 m/s
          Re = 944*0.01061*1e-3 / 0.92e-3 = 10.88 (laminar)
  Step 5: ΔP = 128*0.92e-3*19.10*8.33e-9/(π*1e-12) = 0.0596 bar
  Step 6: t_mix = (1e-3)²/(1e-9*π²) = 101.3 s
          Da = 7.077e-4 * (1e-3)² / 1e-9 = 0.708
  Step 7: r = 7.077e-4 * 100 = 0.0708 mol/(m³·s)
          Q_gen = 50000 * 0.0708 * 15e-6 = 0.0531 W
          A_wall = π*1e-3*19.10 = 0.06000 m²
          Q_rem = 300 * 0.06000 * 10 = 180.0 W
          Da_th = 0.0531/180.0 = 0.000295
  Step 8: BPR not required (DMF bp=153°C, T=25°C)
  Step 9: STY = 0.1*0.72*60/30 = 0.144 mol/(L·h)
          Productivity = 0.5*0.1*0.72*60 = 2.16 mmol/h
          IF = 86400/1800 = 48×
"""
import math
import sys
sys.path.insert(0, ".")

from flora_translate.schemas import BatchRecord
from flora_translate.design_calculator import DesignCalculator

PI = math.pi

def test_photoredox():
    """Ir(ppy)3 photoredox, DMF, 24h, 72% yield, 0.1M, 25°C."""
    br = BatchRecord(
        reaction_description="Ir(ppy)3 photoredox decarboxylative radical addition",
        photocatalyst="Ir(ppy)3",
        catalyst_loading_mol_pct=1.0,
        solvent="DMF",
        temperature_C=25.0,
        reaction_time_h=24.0,
        concentration_M=0.1,
        yield_pct=72.0,
        wavelength_nm=450.0,
        atmosphere="N2",
    )

    calc = DesignCalculator().run(br)

    print("=" * 72)
    print("  9-STEP DESIGN CALCULATOR — TEST RESULTS")
    print("=" * 72)

    for s in calc.steps:
        icon = {"PASS": "✓", "WARNING": "⚠", "FAIL": "✗",
                "ADJUSTED": "↻", "ESTIMATED": "≈"}[s.status]
        print(f"\n{'─'*72}")
        print(f"  Step {s.step}: {s.name}  [{icon} {s.status}]")
        print(f"  {s.summary}")
        if s.equations:
            print("  Equations:")
            for eq in s.equations:
                print(f"    {eq}")
        if s.warnings:
            for w in s.warnings:
                print(f"  ⚠ {w}")
        if s.adjustments:
            for a in s.adjustments:
                print(f"  ↻ {a}")
        if s.assumptions:
            print(f"  Assumptions: {', '.join(s.assumptions)}")

    print(f"\n{'='*72}")
    print("  CONSISTENCY CHECK")
    print(f"{'='*72}")
    if calc.consistent:
        print("  ✓ All consistency checks passed")
    else:
        for n in calc.consistency_notes:
            print(f"  ✗ {n}")

    # ── Hand-verification ───────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  HAND VERIFICATION")
    print(f"{'='*72}")

    errors = []

    # Step 1
    assert calc.temperature_C == 25.0
    assert calc.concentration_M == 0.1
    assert abs(calc.batch_time_s - 86400) < 1
    assert abs(calc.target_conversion - 0.72) < 0.001

    # Step 2: kinetics
    k_batch_expected = -math.log(1 - 0.72) / 86400
    k_flow_expected = k_batch_expected * 48
    tau_expected = 86400 / 48  # = 1800 s
    tau_min_expected = 30.0

    _check(errors, "k_flow", calc.rate_constant, k_flow_expected, tol=0.01)
    _check(errors, "tau_s", calc.residence_time_s, tau_expected, tol=0.01)
    _check(errors, "tau_min", calc.residence_time_min, tau_min_expected, tol=0.01)

    # Step 3: geometry (Q = 0.5 mL/min default)
    Q_mL = 0.5
    V_mL_expected = tau_min_expected * Q_mL  # 15 mL
    d_mm = 1.0  # photochem → 1.0 mm
    d_m = 1e-3
    V_m3 = V_mL_expected * 1e-6
    L_expected = 4 * V_m3 / (PI * d_m**2)

    _check(errors, "V_R_mL", calc.reactor_volume_mL, V_mL_expected, tol=0.01)
    _check(errors, "L_m", calc.tubing_length_m, L_expected, tol=0.01)
    _check(errors, "d_mm", calc.tubing_ID_mm, d_mm, tol=0.01)

    # Step 4: fluid dynamics
    Q_m3s = Q_mL * 1e-6 / 60
    A = PI * (d_m / 2)**2
    v_expected = Q_m3s / A
    # DMF: ρ = 0.944 g/mL = 944 kg/m³, μ = 0.92 cP = 0.92e-3 Pa·s
    rho = 944
    mu = 0.92e-3
    Re_expected = rho * v_expected * d_m / mu

    _check(errors, "v_m_s", calc.velocity_m_s, v_expected, tol=0.01)
    _check(errors, "Re", calc.reynolds_number, Re_expected, tol=0.02)

    # Step 5: pressure drop
    dP_expected_Pa = 128 * mu * L_expected * Q_m3s / (PI * d_m**4)
    dP_expected_bar = dP_expected_Pa * 1e-5

    _check(errors, "dP_bar", calc.pressure_drop_bar, dP_expected_bar, tol=0.02)

    # Step 6: mass transfer
    D = 1e-9
    t_mix_expected = d_m**2 / (D * PI**2)
    Da_expected = k_flow_expected * d_m**2 / D

    _check(errors, "t_mix_s", calc.mixing_time_s, t_mix_expected, tol=0.01)
    _check(errors, "Da_mass", calc.damkohler_mass, Da_expected, tol=0.01)

    # Step 7: heat transfer
    C0_mol_m3 = 100  # 0.1 mol/L = 100 mol/m³
    r_expected = k_flow_expected * C0_mol_m3
    Q_gen_expected = 50000 * r_expected * V_m3
    A_wall_expected = PI * d_m * L_expected
    Q_rem_expected = 300 * A_wall_expected * 10
    Da_th_expected = Q_gen_expected / Q_rem_expected

    _check(errors, "Q_gen_W", calc.heat_generation_W, Q_gen_expected, tol=0.02)
    _check(errors, "Q_rem_W", calc.heat_removal_W, Q_rem_expected, tol=0.02)
    _check(errors, "Da_th", calc.thermal_damkohler, Da_th_expected, tol=0.05)

    # Step 8: BPR
    assert calc.bpr_required is False, "BPR should not be required (DMF bp=153, T=25)"

    # Step 9: metrics
    STY_expected = 0.1 * 0.72 * 60 / tau_min_expected
    prod_expected = Q_mL * 0.1 * 0.72 * 60

    _check(errors, "STY", calc.space_time_yield_mol_L_h, STY_expected, tol=0.01)
    _check(errors, "productivity", calc.productivity_mmol_h, prod_expected, tol=0.01)

    if errors:
        print("\n  ERRORS:")
        for e in errors:
            print(f"  ✗ {e}")
        print(f"\n  {len(errors)} error(s)")
    else:
        print("\n  ✓ ALL HAND CALCULATIONS MATCH")

    # Print prompt block
    print(f"\n{'='*72}")
    print("  PROMPT BLOCK (injected into LLM)")
    print(f"{'='*72}")
    print(calc.to_prompt_block())

    return len(errors) == 0


def _check(errors, name, actual, expected, tol=0.01):
    if actual is None:
        errors.append(f"{name}: got None, expected {expected}")
        return
    rel_err = abs(actual - expected) / max(abs(expected), 1e-15)
    status = "✓" if rel_err <= tol else "✗"
    print(f"  {status} {name}: calculated={actual:.6g}, expected={expected:.6g}, err={rel_err*100:.2f}%")
    if rel_err > tol:
        errors.append(f"{name}: {actual:.6g} vs {expected:.6g} (err={rel_err*100:.1f}%)")


if __name__ == "__main__":
    ok = test_photoredox()
    sys.exit(0 if ok else 1)
