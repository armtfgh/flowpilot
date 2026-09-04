"""FLORA-Translate — Configuration and constants."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RECORDS_DIR = DATA_DIR / "records"
CHROMA_DIR = DATA_DIR / "chroma_db"
PROMPTS_DIR = BASE_DIR / "prompts"
LAB_INVENTORY_PATH = DATA_DIR / "lab_inventory.json"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

TRANSLATION_MODEL  = "claude-sonnet-4-20250514"
SUMMARY_MODEL      = "claude-sonnet-4-20250514"
CHEMISTRY_MODEL      = "claude-opus-4-6"         # Opus for Layer 1 — deeper reasoning
CHEMISTRY_MAX_TOKENS = 8192                      # max output tokens — do not limit
EMBEDDING_MODEL    = "text-embedding-3-small"
EMBEDDING_DIM      = 1536

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

COLLECTION_NAME = "flora_records"
PAIRS_COLLECTION_NAME = "flora_pairs"
TOP_K_RETRIEVAL = 20
TOP_K_ANALOGIES = 3

# Reranking weights
W_SEMANTIC = 0.6
W_FIELD = 0.4

# Field similarity weights
W_PHOTOCATALYST = 0.30
W_SOLVENT = 0.20
W_WAVELENGTH = 0.20
W_TEMPERATURE = 0.15
W_CONCENTRATION = 0.15

# ---------------------------------------------------------------------------
# ENGINE
# ---------------------------------------------------------------------------

MAX_COUNCIL_ROUNDS = 3

# Kinetics thresholds
INTENSIFICATION_LOW_BOUND = 0.3   # factor below median → warning
INTENSIFICATION_HIGH_BOUND = 3.0  # factor above median → warning
DEFAULT_INTENSIFICATION = 50.0    # median for photocatalytic reactions

# Fluidics thresholds
PRESSURE_WARNING_FRACTION = 0.8   # warn at 80% of pump max
RE_TURBULENT = 2300
RE_VERY_LOW = 1

# ---------------------------------------------------------------------------
# Photocatalyst class matching
# ---------------------------------------------------------------------------

PHOTOCATALYST_CLASSES: dict[str, str] = {
    "Ir(ppy)3": "Ir_cyclometalated",
    "fac-Ir(ppy)3": "Ir_cyclometalated",
    "[Ir(ppy)2(dtbbpy)]PF6": "Ir_cyclometalated",
    "[Ir(dF(CF3)ppy)2(dtbbpy)]PF6": "Ir_cyclometalated",
    "Ir(dF(CF3)ppy)2(dtbbpy)PF6": "Ir_cyclometalated",
    "Ru(bpy)3Cl2": "Ru_polypyridyl",
    "[Ru(bpy)3]Cl2": "Ru_polypyridyl",
    "[Ru(bpy)3]2+": "Ru_polypyridyl",
    "Ru(bpy)3(PF6)2": "Ru_polypyridyl",
    "4CzIPN": "organic_donor_acceptor",
    "3DPA2FBN": "organic_donor_acceptor",
    "TPT": "organic_donor_acceptor",
    "Eosin Y": "organic_xanthene",
    "Rose Bengal": "organic_xanthene",
    "Fluorescein": "organic_xanthene",
    "Methylene Blue": "organic_thiazine",
    "Acridinium": "organic_acridinium",
}

# Map classes to metal families for partial matching
METAL_FAMILIES: dict[str, str] = {
    "Ir_cyclometalated": "Ir",
    "Ru_polypyridyl": "Ru",
    "organic_donor_acceptor": "organic",
    "organic_xanthene": "organic",
    "organic_thiazine": "organic",
    "organic_acridinium": "organic",
}

# ---------------------------------------------------------------------------
# Solvent properties
# ---------------------------------------------------------------------------

SOLVENT_VISCOSITY_cP: dict[str, float] = {
    "MeCN": 0.37, "acetonitrile": 0.37, "CH3CN": 0.37,
    "DMF": 0.92, "N,N-dimethylformamide": 0.92,
    "DMSO": 2.00, "dimethyl sulfoxide": 2.00,
    "THF": 0.46, "tetrahydrofuran": 0.46,
    "EtOAc": 0.45, "ethyl acetate": 0.45,
    "DCM": 0.44, "dichloromethane": 0.44, "CH2Cl2": 0.44,
    "MeOH": 0.54, "methanol": 0.54,
    "EtOH": 1.08, "ethanol": 1.08,
    "iPrOH": 2.04, "isopropanol": 2.04,
    "toluene": 0.56,
    "water": 1.00, "H2O": 1.00,
    "DMA": 0.92, "dimethylacetamide": 0.92,
    "NMP": 1.67, "N-methylpyrrolidone": 1.67,
    "dioxane": 1.18, "1,4-dioxane": 1.18,
    "benzene": 0.60,
    "hexane": 0.31,
    "acetone": 0.31,
}

SOLVENT_DENSITY_g_mL: dict[str, float] = {
    "MeCN": 0.786, "acetonitrile": 0.786,
    "DMF": 0.944, "DMSO": 1.100,
    "THF": 0.889, "EtOAc": 0.902,
    "DCM": 1.327, "dichloromethane": 1.327,
    "MeOH": 0.791, "EtOH": 0.789,
    "toluene": 0.867, "water": 1.000,
    "DMA": 0.937, "NMP": 1.028,
    "dioxane": 1.034, "acetone": 0.784,
    "hexane": 0.659, "benzene": 0.879,
}

SOLVENT_BOILING_POINT_C: dict[str, float] = {
    "MeCN": 82, "acetonitrile": 82,
    "DMF": 153, "DMSO": 189,
    "THF": 66, "EtOAc": 77,
    "DCM": 40, "dichloromethane": 40,
    "MeOH": 65, "EtOH": 78,
    "toluene": 111, "water": 100,
    "DMA": 165, "NMP": 202,
    "dioxane": 101, "acetone": 56,
    "hexane": 69, "benzene": 80,
}

# Material incompatibility table: (material, solvent) → concern
INCOMPATIBLE_COMBOS: dict[tuple[str, str], str] = {
    ("FEP", "THF"): "FEP swells in THF above 60°C",
    ("PTFE", "DCM"): "PTFE not recommended for DCM at elevated temperature",
    ("SS", "HCl"): "SS corrodes in HCl",
    ("FEP", "toluene"): "FEP may swell in toluene above 80°C",
}
