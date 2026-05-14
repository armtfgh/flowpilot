"""Deterministic process-intensification mandate helpers."""

from __future__ import annotations

from flora_translate.schemas import BatchRecord, ChemistryPlan, IntensificationMandate


def build_intensification_mandate(
    batch_record: BatchRecord,
    plan: ChemistryPlan | None,
) -> IntensificationMandate:
    """Build a conservative, explicit reason why flow should add value.

    This is code-owned so weak/local upstream models do not have to invent the
    process-intensification target reliably.
    """
    reaction_class = ((plan.reaction_class if plan else "") or "").lower()
    mechanism = ((plan.mechanism_type if plan else "") or "").lower()
    description = (batch_record.reaction_description or "").lower()
    text = " ".join([reaction_class, mechanism, description])

    is_photo = any(token in text for token in ("photo", "photoredox", "photochem", "light", "led"))
    has_hazardous_intermediate = any(
        token in text
        for token in (
            "diazonium",
            "peroxide",
            "organolithium",
            "azide",
            "nitrile oxide",
            "chlorine",
            "phosgene",
            "hazardous intermediate",
        )
    )
    exothermic = any(token in text for token in ("exotherm", "exothermic", "runaway", "strongly exothermic"))
    selectivity_sensitive = bool(plan and (plan.incompatible_pairs or plan.light_sensitive_reagents))
    if any(token in text for token in ("selectiv", "overreaction", "competing", "side reaction")):
        selectivity_sensitive = True

    if has_hazardous_intermediate or exothermic or is_photo:
        target = 6.0
    elif any(token in text for token in ("thermal", "coupling", "substitution", "addition", "condensation")):
        target = 3.0
    else:
        target = 2.5

    if exothermic:
        advantage = "heat_transfer"
    elif has_hazardous_intermediate:
        advantage = "hazardous_intermediate"
    elif selectivity_sensitive:
        advantage = "selectivity"
    elif is_photo:
        advantage = "productivity"
    else:
        advantage = "productivity"

    if selectivity_sensitive or exothermic:
        regime = "enhanced_laminar_mixing"
    elif has_hazardous_intermediate:
        regime = "slug_flow"
    else:
        regime = "laminar_acceptable"

    features: list[str] = []
    if is_photo:
        features.append("photochemical photon delivery")
    if exothermic:
        features.append("heat-removal demand")
    if has_hazardous_intermediate:
        features.append("hazardous-intermediate holdup reduction")
    if selectivity_sensitive:
        features.append("selectivity-sensitive stream/contact logic")
    if not features:
        features.append("productivity and reactor-volume reduction")

    basis = (
        f"Flow must demonstrate {target:.1f}x residence-time reduction or a clear "
        f"{advantage} advantage based on {', '.join(features)}."
    )

    return IntensificationMandate(
        tau_reduction_target=target,
        minimum_flow_advantage=advantage,
        required_mixing_regime=regime,
        flow_justification_basis=basis,
    )


def ensure_intensification_mandate(
    batch_record: BatchRecord,
    plan: ChemistryPlan,
) -> ChemistryPlan:
    mandate = getattr(plan, "intensification_mandate", None)
    if mandate is None or not getattr(mandate, "flow_justification_basis", ""):
        plan.intensification_mandate = build_intensification_mandate(batch_record, plan)
    return plan
