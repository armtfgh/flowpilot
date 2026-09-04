from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    title: str
    protocol: str
    precedent_level: str
    difficulty: str
    notes: str = ""
    cached_result_path: str | None = None
    batch_record_data: dict | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)


CASES: dict[str, BenchmarkCase] = {
    "snar": BenchmarkCase(
        case_id="snar",
        title="SNAr of 4-fluoronitrobenzene with piperazine",
        protocol="""
Nucleophilic aromatic substitution (SNAr) of 4-fluoronitrobenzene (0.3 M)
with piperazine (1.2 equiv) in N,N-dimethylformamide at 120 °C.
No catalyst required. The reaction was run under air for 6 hours and
gave the N-aryl piperazine product in 81% isolated yield after extraction.
Reaction was performed at 1.0 mmol scale with continuous magnetic stirring.
No special atmosphere control was used.
""".strip(),
        precedent_level="analogy_supported",
        difficulty="medium",
        notes="Thermal case with partial literature support and meaningful geometry tradeoffs.",
        cached_result_path="outputs/snar_council_run.json",
        batch_record_data={
            "reaction_description": "Nucleophilic aromatic substitution (SNAr) of 4-fluoronitrobenzene with piperazine in DMF",
            "reaction_time_h": 6.0,
            "temperature_C": 120.0,
            "solvent": "DMF",
            "concentration_M": 0.3,
            "yield_pct": 81.0,
            "atmosphere": "air",
        },
        tags=("thermal", "snar", "analogy"),
    ),
    "thermal_knoevenagel": BenchmarkCase(
        case_id="thermal_knoevenagel",
        title="Thermal Knoevenagel condensation",
        protocol="""
Thermal Knoevenagel condensation of benzaldehyde (0.5 M) with ethyl
cyanoacetate (0.55 M, 1.1 equiv) using piperidine base-catalyst (0.05 M,
10 mol%) in ethanol at 80 °C. Mildly exothermic aldol-type C-C bond
formation. Batch reaction time: 8 h at reflux, isolated yield: 92 %.
No photocatalyst. Product: ethyl (E)-2-cyano-3-phenylacrylate.
""".strip(),
        precedent_level="near_literature",
        difficulty="medium",
        notes="Thermal benchmark with explicit BPR/material reasoning pressure.",
        cached_result_path="outputs/thermal_council_run.json",
        batch_record_data={
            "reaction_description": "Thermal Knoevenagel condensation of benzaldehyde with ethyl cyanoacetate",
            "reaction_time_h": 8.0,
            "temperature_C": 80.0,
            "solvent": "EtOH",
            "concentration_M": 0.5,
            "yield_pct": 92.0,
        },
        tags=("thermal", "knoevenagel", "published_like"),
    ),
    "mock_photoredox": BenchmarkCase(
        case_id="mock_photoredox",
        title="Mock photoredox dehalogenation",
        protocol="""
Ru(bpy)3Cl2-catalyzed visible-light photoredox dehalogenation of
alpha-bromoacetophenone in MeCN/H2O (9:1, v/v), 0.1 M substrate,
460 nm blue LED irradiation, strict N2 atmosphere, 85 degC,
24 h batch time, 78% isolated yield.
Conditions: 1 mol% Ru(bpy)3Cl2 photocatalyst, 2.0 equiv DIPEA
as sacrificial reductant, reaction quenched with sat. NH4Cl (aq).
""".strip(),
        precedent_level="synthetic_challenging",
        difficulty="high",
        notes="Photoredox case with narrow photonics and atmosphere constraints.",
        cached_result_path="outputs/mock_council_run.json",
        batch_record_data={
            "reaction_description": "Ru(bpy)3Cl2-catalyzed visible-light photoredox dehalogenation of alpha-bromoacetophenone",
            "photocatalyst": "Ru(bpy)3Cl2",
            "catalyst_loading_mol_pct": 1.0,
            "solvent": "MeCN/H2O (9:1)",
            "temperature_C": 85.0,
            "reaction_time_h": 24.0,
            "concentration_M": 0.1,
            "yield_pct": 78.0,
            "light_source": "460 nm blue LED",
            "wavelength_nm": 460.0,
            "atmosphere": "N2",
        },
        tags=("photoredox", "synthetic", "challenging"),
    ),
    "cycloaddition_des": BenchmarkCase(
        case_id="cycloaddition_des",
        title="1,3-dipolar cycloaddition in DES",
        protocol="""
1,3-dipolar cycloaddition of phenylacetylene with ethyl nitroacetate to form
ethyl 5-phenylisoxazole-3-carboxylate in TBAB/ethylene glycol (1:5) deep
eutectic solvent at 120 °C. No catalyst or base. Reaction performed as a
single homogeneous batch for 6 hours with 1 equivalent phenylacetylene and
1 equivalent ethyl nitroacetate. Isolated yield 78%.
""".strip(),
        precedent_level="weak_precedent",
        difficulty="high",
        notes="Weak-precedent thermal cycloaddition close to the recent Streamlit case.",
        batch_record_data={
            "reaction_description": "1,3-dipolar cycloaddition of phenylacetylene with ethyl nitroacetate to form ethyl 5-phenylisoxazole-3-carboxylate",
            "temperature_C": 120.0,
            "reaction_time_h": 6.0,
            "solvent": "TBAB/EG (1:5) DES",
            "yield_pct": 78.0,
        },
        tags=("thermal", "weak_precedent", "cycloaddition"),
    ),
}


def get_case(case_id: str) -> BenchmarkCase:
    if case_id not in CASES:
        raise KeyError(f"Unknown benchmark case: {case_id}")
    return CASES[case_id]


def list_cases() -> list[BenchmarkCase]:
    return list(CASES.values())
