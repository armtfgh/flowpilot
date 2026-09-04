"""Standardized FlowPilot intake before batch-to-flow design.

The intake layer turns a chemist's free-form protocol plus follow-up answers
into a frozen ``DesignInputPackage``. Question selection is deterministic and
uses stable IDs from the fixed question bank so benchmarks and GUI sessions are
reproducible.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import flora_translate.config as cfg
from flora_translate.config import PROMPTS_DIR
from flora_translate.engine.llm_agents import call_model_text
from flora_translate.executable_artifacts import transformation_family
from flora_translate.schemas import DesignInputPackage, IntakeAnswer, IntakeQuestion

logger = logging.getLogger("flora.intake")


QUESTION_BANK: dict[str, IntakeQuestion] = {
    "Q-BATCH-001": IntakeQuestion(
        question_id="Q-BATCH-001",
        section="batch_protocol",
        question=(
            "Please provide or confirm the batch reaction protocol, including "
            "substrate, product, solvent, concentration, scale, reagents, "
            "equivalents, catalyst/additives, gas atmosphere, temperature, "
            "time, irradiation/light source if any, and reported yield."
        ),
        expected_format="Free-text protocol or structured JSON.",
        required=True,
        why_needed="The design pipeline needs protocol facts before translation.",
    ),
    "Q-OBJ-001": IntakeQuestion(
        question_id="Q-OBJ-001",
        section="objective",
        question=(
            "What is the main goal for the first flow design? Examples: "
            "first safe screening design, reproduce batch yield, improve "
            "throughput, maximize conversion, minimize residence time, or "
            "test a specific hypothesis."
        ),
        expected_format="Short text objective.",
        required=True,
        why_needed="The council ranks candidate designs against this objective.",
    ),
    "Q-CHEM-001": IntakeQuestion(
        question_id="Q-CHEM-001",
        section="chemistry_identity",
        question=(
            "Please confirm the intended reaction transformation. State the "
            "transformation family and, where known, the substrate, product, "
            "bond formed, and bond broken."
        ),
        expected_format=(
            "Short text such as 'hydrogenolysis/debenzylation of A to B; C-N "
            "protecting-group bond broken'."
        ),
        required=True,
        why_needed=(
            "A model-only chemistry identity cannot authorize an executable design."
        ),
    ),
    "Q-HIST-001": IntakeQuestion(
        question_id="Q-HIST-001",
        section="historical_data",
        question=(
            "Do you have historical experimental data or previous flow attempts "
            "for this reaction? If yes, provide a table with conditions and "
            "results."
        ),
        expected_format=(
            "Table/CSV/text with entry, T, concentration, pressure, flows, "
            "residence times, reactor volume, tubing ID, product/yield/"
            "conversion/selectivity, and notes; or mark unavailable."
        ),
        required=True,
        why_needed="Measured evidence has highest design authority.",
    ),
    "Q-INV-001": IntakeQuestion(
        question_id="Q-INV-001",
        section="inventory",
        question=(
            "Do you have inventory constraints that the design must use? "
            "Provide available reactor volumes, tubing IDs/materials, pumps, "
            "pressure/BPR options, gas MFC limits, light sources, and "
            "temperature ranges."
        ),
        expected_format="LabInventory JSON preferred; free text accepted; or mark unavailable.",
        required=True,
        why_needed="Hard hardware constraints must be enforced before final design.",
    ),
    "Q-CONSTR-001": IntakeQuestion(
        question_id="Q-CONSTR-001",
        section="operating_limits",
        question=(
            "Are there operating limits that must not be exceeded? Examples: "
            "max pressure, max temperature, max/min flow rate, max residence "
            "time, forbidden solvents, gas safety limits, oxygen limits, or "
            "material compatibility limits."
        ),
        expected_format="Free text or JSON; or mark unavailable.",
        required=True,
        why_needed="Safety and operating limits constrain feasible candidates.",
    ),
    "Q-HYP-001": IntakeQuestion(
        question_id="Q-HYP-001",
        section="hypotheses",
        question=(
            "Do you have chemical or engineering hypotheses the agent should "
            "consider? Examples: oxygen mass-transfer limitation, photon "
            "limitation, slow kinetics, degradation, gas-liquid ratio, "
            "residence-time sensitivity, or light intensity importance."
        ),
        expected_format="Bullet list/free text; or mark unavailable.",
        required=True,
        why_needed="Hypotheses guide experiment design but are not treated as facts.",
    ),
    "Q-PREF-001": IntakeQuestion(
        question_id="Q-PREF-001",
        section="output_preferences",
        question=(
            "What output do you want? Examples: one best design, short/"
            "intermediate/long residence-time screen, inventory-constrained "
            "design only, benchmark comparison, or design with calculations "
            "and uncertainty."
        ),
        expected_format="Short text preference; optional.",
        required=False,
        why_needed="Output preferences tune presentation and candidate selection.",
    ),
}

REQUIRED_READINESS_IDS = {
    "Q-BATCH-001",
    "Q-OBJ-001",
    "Q-CHEM-001",
    "Q-HIST-001",
    "Q-INV-001",
    "Q-CONSTR-001",
    "Q-HYP-001",
}


def get_question(question_id: str) -> IntakeQuestion:
    return QUESTION_BANK[question_id]


def intake_context_block(package: DesignInputPackage | dict | None) -> str:
    """Format the frozen intake package for upstream/council prompts."""

    if not package:
        return "## FlowPilot Intake Context\nNo standardized intake package was provided."
    pkg = _coerce_package(package)
    # The complete inventory remains frozen on the package for deterministic
    # allocation. Prompts receive only capability-defining fields so the same
    # profile is not repeated in Q-INV-001 and the answer log.
    inventory_context = _compact_inventory_for_prompt(pkg.inventory_constraints)
    answer_log = [
        {
            "question_id": answer.question_id,
            "status": answer.status,
            "source": answer.source,
            "answer": (
                "See hard constraints and operating limits above."
                if answer.question_id in {"Q-INV-001", "Q-CONSTR-001"}
                else answer.answer
            ),
        }
        for answer in pkg.answers
    ]
    return (
        "## FlowPilot Intake Context - authority labeled\n"
        "Authority order: measured evidence > hard constraints > protocol facts > "
        "chemist hypotheses > model inference.\n"
        f"Schema: {pkg.schema_version}\n"
        f"Ready for design: {pkg.ready_for_design}\n\n"
        "### Design objective\n"
        f"{pkg.objective or '(not provided)'}\n\n"
        "### Protocol facts\n"
        f"{json.dumps(pkg.extracted_batch_fields or {}, indent=2, default=str)}\n\n"
        "### Frozen chemistry identity\n"
        f"{json.dumps(pkg.chemistry_identity_confirmation or {}, indent=2, default=str)}\n\n"
        "### Measured evidence / historical data\n"
        f"{_format_any(pkg.historical_data)}\n\n"
        "### Hard constraints / inventory\n"
        f"{_format_any(inventory_context)}\n\n"
        "### Safety and operating limits\n"
        f"{_format_any(pkg.operating_limits)}\n\n"
        "### Chemist hypotheses\n"
        f"{json.dumps(pkg.hypotheses, indent=2, default=str)}\n\n"
        "### Intake questions and answers\n"
        f"{json.dumps(answer_log, indent=2, default=str)}"
    )


_PROMPT_EQUIPMENT_FIELDS = (
    "equipment_id", "name", "quantity", "service_status", "type", "material",
    "volume_mL", "ID_mm", "configuration", "component_volumes_mL", "gas",
    "setpoints_bar", "min_pressure_bar", "max_pressure_bar", "min_temperature_C",
    "max_temperature_C", "allowed_temperatures_C", "min_flow_rate_mL_min",
    "max_flow_rate_mL_min", "min_flow_sccm", "max_flow_sccm", "wavelength_nm",
    "power_W", "intensity_mW_cm2", "compatible_reactor",
)


def _compact_inventory_for_prompt(value: Any) -> Any:
    """Keep design limits while removing notes and repeated profile metadata."""

    if not isinstance(value, dict):
        return value
    compact: dict[str, Any] = {}
    for category, items in value.items():
        if isinstance(items, list):
            compact[category] = [
                {
                    key: item[key]
                    for key in _PROMPT_EQUIPMENT_FIELDS
                    if key in item and item[key] not in (None, "", [], {})
                }
                if isinstance(item, dict)
                else item
                for item in items
            ]
        elif items not in (None, "", [], {}):
            compact[category] = items
    return compact


def batch_input_from_package(package: DesignInputPackage | dict) -> str | dict:
    """Return the best batch input payload for the existing parser."""

    pkg = _coerce_package(package)
    fields = dict(pkg.extracted_batch_fields or {})
    if fields:
        fields.setdefault("raw_text", pkg.raw_protocol)
        if pkg.raw_protocol and not fields.get("reaction_description"):
            fields["reaction_description"] = pkg.raw_protocol[:500]
        return fields
    return pkg.raw_protocol


def historical_text_from_package(package: DesignInputPackage | dict | None) -> str:
    """Return measured feedback text for closed-loop extraction."""

    if not package:
        return ""
    pkg = _coerce_package(package)
    return _format_any(pkg.historical_data)


class IntakeAgent:
    """Build and update a standardized DesignInputPackage."""

    def analyze(
        self,
        raw_protocol: str = "",
        *,
        existing_package: DesignInputPackage | dict | None = None,
        answers: list[IntakeAnswer | dict] | None = None,
        use_llm: bool = True,
    ) -> DesignInputPackage:
        pkg = _coerce_package(existing_package) if existing_package else DesignInputPackage()
        raw_protocol = raw_protocol or pkg.raw_protocol or ""
        merged_answers = list(pkg.answers)
        for answer in answers or []:
            merged_answers.append(_coerce_answer(answer))

        extracted = dict(pkg.extracted_batch_fields or {})
        if raw_protocol and not extracted:
            extracted = self._extract_batch_fields(raw_protocol, use_llm=use_llm)

        updated = DesignInputPackage(
            schema_version=pkg.schema_version,
            raw_protocol=raw_protocol,
            extracted_batch_fields=extracted,
            objective=pkg.objective,
            historical_data=pkg.historical_data,
            inventory_constraints=pkg.inventory_constraints,
            hypotheses=list(pkg.hypotheses or []),
            operating_limits=pkg.operating_limits,
            inventory_profile_snapshot=dict(pkg.inventory_profile_snapshot or {}),
            output_preferences=pkg.output_preferences,
            chemistry_identity_confirmation=dict(
                pkg.chemistry_identity_confirmation or {}
            ),
            question_log=list(pkg.question_log or []),
            answers=merged_answers,
        )
        updated = self._apply_answers(updated)
        if not updated.chemistry_identity_confirmation:
            family = transformation_family(updated.raw_protocol)
            multistep = bool(
                re.search(
                    r"(?i)\b(two[- ]stage|multi[- ]step|sequential(?:ly)?|followed by)\b",
                    updated.raw_protocol,
                )
            )
            if family != "unknown" and not multistep:
                updated.chemistry_identity_confirmation = {
                    "transformation_family": family,
                    "confirmed": True,
                    "source": "protocol_fact",
                }
        missing = self._missing_required_ids(updated)
        logged = {q.question_id for q in updated.question_log}
        question_log = list(updated.question_log)
        for qid in missing:
            if qid not in logged:
                question_log.append(QUESTION_BANK[qid])
        updated.question_log = question_log
        updated.missing_question_ids = missing
        updated.ready_for_design = not missing
        return updated

    def pending_questions(self, package: DesignInputPackage | dict) -> list[IntakeQuestion]:
        pkg = _coerce_package(package)
        return [QUESTION_BANK[qid] for qid in pkg.missing_question_ids if qid in QUESTION_BANK]

    def _extract_batch_fields(self, raw_protocol: str, *, use_llm: bool) -> dict[str, Any]:
        if use_llm:
            try:
                return self._llm_extract(raw_protocol)
            except Exception as exc:
                logger.warning("Intake LLM extraction failed; using fallback extraction: %s", exc)
        return _fallback_extract(raw_protocol)

    def _llm_extract(self, raw_protocol: str) -> dict[str, Any]:
        system = (PROMPTS_DIR / "intake_system.txt").read_text()
        user_template = (PROMPTS_DIR / "intake_user.txt").read_text()
        result = call_model_text(
            model=cfg.MODEL_INPUT_PARSER,
            api_name="intake_agent",
            max_tokens=1200,
            system=system,
            user_content=user_template.replace("{raw_protocol}", raw_protocol),
        )
        data = _parse_json(result.text)
        fields = data.get("extracted_batch_fields", data)
        if not isinstance(fields, dict):
            return _fallback_extract(raw_protocol)
        fields.setdefault("raw_text", raw_protocol)
        if raw_protocol and not fields.get("reaction_description"):
            fields["reaction_description"] = raw_protocol[:500]
        return fields

    def _apply_answers(self, package: DesignInputPackage) -> DesignInputPackage:
        answer_map = package.answer_map()

        batch_answer = answer_map.get("Q-BATCH-001")
        if batch_answer and batch_answer.status == "answered" and batch_answer.answer:
            package.raw_protocol = str(batch_answer.answer)
            if not package.extracted_batch_fields:
                package.extracted_batch_fields = _fallback_extract(package.raw_protocol)

        obj_answer = answer_map.get("Q-OBJ-001")
        if obj_answer and obj_answer.status == "answered":
            package.objective = str(obj_answer.answer or "").strip()

        chemistry_answer = answer_map.get("Q-CHEM-001")
        if chemistry_answer and chemistry_answer.status == "answered":
            answer_text = str(chemistry_answer.answer or "").strip()
            family = transformation_family(
                " ".join(part for part in (answer_text, package.raw_protocol) if part)
            )
            package.chemistry_identity_confirmation = {
                "transformation_family": family,
                "chemist_description": answer_text,
                "confirmed": bool(answer_text),
                "source": "chemist_confirmed",
            }

        hist_answer = answer_map.get("Q-HIST-001")
        if hist_answer:
            package.historical_data = None if hist_answer.status == "unavailable" else hist_answer.answer

        inv_answer = answer_map.get("Q-INV-001")
        if inv_answer:
            package.inventory_constraints = (
                None if inv_answer.status == "unavailable" else _parse_json_or_text(inv_answer.answer)
            )

        constr_answer = answer_map.get("Q-CONSTR-001")
        if constr_answer:
            package.operating_limits = (
                None if constr_answer.status == "unavailable" else _parse_json_or_text(constr_answer.answer)
            )

        hyp_answer = answer_map.get("Q-HYP-001")
        if hyp_answer:
            package.hypotheses = [] if hyp_answer.status == "unavailable" else _split_hypotheses(hyp_answer.answer)

        pref_answer = answer_map.get("Q-PREF-001")
        if pref_answer and pref_answer.status == "answered":
            package.output_preferences = str(pref_answer.answer or "").strip()

        return package

    def _missing_required_ids(self, package: DesignInputPackage) -> list[str]:
        answer_map = package.answer_map()
        missing: list[str] = []

        if not _has_protocol(package):
            missing.append("Q-BATCH-001")

        if not package.objective.strip():
            missing.append("Q-OBJ-001")

        identity = package.chemistry_identity_confirmation or {}
        if not identity.get("confirmed"):
            missing.append("Q-CHEM-001")

        for qid, attr in (
            ("Q-HIST-001", "historical_data"),
            ("Q-INV-001", "inventory_constraints"),
            ("Q-CONSTR-001", "operating_limits"),
            ("Q-HYP-001", "hypotheses"),
        ):
            answer = answer_map.get(qid)
            value = getattr(package, attr)
            if answer and answer.status in {"answered", "unavailable"}:
                continue
            if _has_value(value):
                continue
            missing.append(qid)

        return missing


def _coerce_package(package: DesignInputPackage | dict) -> DesignInputPackage:
    if isinstance(package, DesignInputPackage):
        return package
    return DesignInputPackage.model_validate(package)


def _coerce_answer(answer: IntakeAnswer | dict) -> IntakeAnswer:
    if isinstance(answer, IntakeAnswer):
        return answer
    return IntakeAnswer.model_validate(answer)


def _has_protocol(package: DesignInputPackage) -> bool:
    if package.raw_protocol.strip():
        return True
    desc = str((package.extracted_batch_fields or {}).get("reaction_description") or "")
    return bool(desc.strip())


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def _fallback_extract(raw_protocol: str) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "reaction_description": raw_protocol[:500],
        "raw_text": raw_protocol,
    }
    temp = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:°\s*C|deg\s*C|C\b)", raw_protocol, re.I)
    if temp:
        fields["temperature_C"] = float(temp.group(1))
    conc = re.search(r"(\d+(?:\.\d+)?)\s*M\b", raw_protocol, re.I)
    if conc:
        fields["concentration_M"] = float(conc.group(1))
    wave = re.search(r"(\d+(?:\.\d+)?)\s*nm\b", raw_protocol, re.I)
    if wave:
        fields["wavelength_nm"] = float(wave.group(1))
    time_h = re.search(r"(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)\b", raw_protocol, re.I)
    if time_h:
        fields["reaction_time_h"] = float(time_h.group(1))
    yield_pct = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:isolated\s*)?yield", raw_protocol, re.I)
    if yield_pct:
        fields["yield_pct"] = float(yield_pct.group(1))
    photocatalyst = re.search(
        r"\bphotocatalyst\s*:\s*(.+?)(?=,\s*\d+(?:\.\d+)?\s*mol\s*%|[.;\n])",
        raw_protocol,
        re.I,
    )
    if photocatalyst:
        fields["photocatalyst"] = photocatalyst.group(1).strip()
    loading = re.search(
        r"\bphotocatalyst\s*:.{0,250}?,\s*(\d+(?:\.\d+)?)\s*mol\s*%",
        raw_protocol,
        re.I,
    )
    if loading:
        fields["catalyst_loading_mol_pct"] = float(loading.group(1))
    solvent = re.search(
        r"\bsolvent\s*:\s*(.+?)(?=\.\s*(?:Concentration|Step|Stage)\b|[;\n]|$)",
        raw_protocol,
        re.I,
    )
    if solvent:
        fields["solvent"] = solvent.group(1).strip().rstrip(".")
    if re.search(r"\b(?:under|maintain under)\s+argon\b", raw_protocol, re.I):
        fields["atmosphere"] = "argon"
    if wave:
        fields["light_source"] = f"{float(wave.group(1)):g} nm light"
    return fields


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    return json.loads(text)


def _parse_json_or_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return ""
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    return value


def _split_hypotheses(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value)
    pieces = re.split(r"(?:\n|;)+", text)
    return [p.strip(" -\t") for p in pieces if p.strip(" -\t")]


def _format_any(value: Any) -> str:
    if value is None:
        return "(unavailable)"
    if isinstance(value, str):
        return value.strip() or "(unavailable)"
    return json.dumps(value, indent=2, default=str)
