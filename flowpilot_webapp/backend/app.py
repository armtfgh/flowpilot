"""FastAPI adapter for the validated FlowPilot Python pipeline."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import os
import sys
from typing import Any

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .jobs import JOBS


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass


def _frontend_dist() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "flowpilot_webapp" / "frontend" / "dist"
    return ROOT / "flowpilot_webapp" / "frontend" / "dist"


DIST = _frontend_dist()


class IntakePayload(BaseModel):
    raw_protocol: str = ""
    existing_package: dict[str, Any] | None = None
    answers: list[dict[str, Any]] = Field(default_factory=list)
    use_llm: bool = True
    inventory_profile: dict[str, Any] | None = None
    upstream_model_id: str | None = None


class DesignPayload(BaseModel):
    batch_input: str = ""
    intake_package: dict[str, Any]
    inventory_profile: dict[str, Any] | None = None
    inventory_path: str = "flora_translate/data/lab_inventory.json"
    runtime_options: dict[str, Any] | None = None
    upstream_model_id: str | None = None
    downstream_model_id: str | None = None


class InventorySavePayload(BaseModel):
    profile: dict[str, Any]


class RefinementPayload(BaseModel):
    job_id: str | None = None
    current_result: dict[str, Any] | None = None
    experiments: list[dict[str, Any]] = Field(default_factory=list)
    target_yield_pct: float = 80.0
    target_conversion_pct: float = 90.0
    target_selectivity_pct: float = 85.0


app = FastAPI(title="FlowPilot API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    from flora_translate import config as cfg
    from flora_translate.inventory_profiles import list_inventory_profiles
    from flora_translate.model_catalog import (
        available_default_route_ids,
        model_route_statuses,
        model_routes,
    )

    corpus_records = 0
    try:
        import chromadb

        client = chromadb.PersistentClient(path="flora_translate/data/chroma_db")
        corpus_records = client.get_or_create_collection("flora_records").count()
    except Exception:
        pass
    routes = model_routes()
    route_defaults = available_default_route_ids(model_route_statuses())
    upstream = routes[route_defaults["upstream"]]
    downstream = routes[route_defaults["downstream"]]
    return {
        "status": "ok",
        "app": "FlowPilot",
        "version": "2.0.0",
        "models": {
            "intake": cfg.MODEL_INPUT_PARSER,
            "chemistry": upstream.model,
            "translation": downstream.model,
            "council_provider": downstream.provider,
            "council_model": downstream.model,
        },
        "inventory_profiles": len(list_inventory_profiles()),
        "corpus_records": corpus_records,
    }


@app.get("/api/models")
def available_models() -> dict[str, Any]:
    from flora_translate.model_catalog import (
        available_default_route_ids,
        model_route_statuses,
        model_routes,
    )

    routes = model_routes()
    statuses = model_route_statuses()
    return {
        "models": [
            {**route.public(), **statuses[route_id]}
            for route_id, route in routes.items()
        ],
        "defaults": available_default_route_ids(statuses),
    }


def _apply_inventory(package: dict[str, Any], profile_payload: dict[str, Any] | None) -> dict[str, Any]:
    if not profile_payload:
        return package
    from flora_translate.intake_agent import IntakeAgent
    from flora_translate.inventory_profiles import inventory_profile_from_payload
    from flora_translate.schemas import IntakeAnswer

    profile = inventory_profile_from_payload(profile_payload)
    answers = [
        IntakeAnswer(
            question_id="Q-INV-001",
            answer=profile.lab_inventory.model_dump(),
            status="answered",
            source="inventory_profile",
        ),
        IntakeAnswer(
            question_id="Q-CONSTR-001",
            answer=profile.design_operating_limits(),
            status="answered",
            source="inventory_profile",
        ),
    ]
    updated = IntakeAgent().analyze(
        package.get("raw_protocol") or "",
        existing_package=package,
        answers=answers,
        use_llm=False,
    )
    updated.inventory_profile_snapshot = profile.model_dump()
    return updated.model_dump()


@app.post("/api/intake/analyze")
def analyze_intake(payload: IntakePayload) -> dict[str, Any]:
    from flora_translate.intake_agent import IntakeAgent
    from flora_translate.pipeline_runtime import (
        PipelineRuntimeOptions,
        runtime_model_routing,
    )

    try:
        runtime = PipelineRuntimeOptions()
        if payload.use_llm and payload.upstream_model_id:
            from flora_translate.model_catalog import (
                model_route_availability,
                route_for_id,
            )

            route = route_for_id(payload.upstream_model_id, role="upstream")
            status = model_route_availability(route)
            if not status["available"]:
                raise ValueError(
                    f"Upstream model {route.label} is unavailable: {status['reason']}"
                )
            runtime = PipelineRuntimeOptions(
                upstream_model=route.model,
                upstream_provider=route.provider,
                model_endpoints={route.model: route.base_url} if route.base_url else {},
            )
        agent = IntakeAgent()
        with runtime_model_routing(runtime):
            package = agent.analyze(
                payload.raw_protocol,
                existing_package=payload.existing_package,
                answers=payload.answers,
                use_llm=payload.use_llm,
            ).model_dump()
        package = _apply_inventory(package, payload.inventory_profile)
        pending = [
            question.model_dump()
            for question in agent.pending_questions(package)
        ]
        return {"package": package, "pending_questions": pending}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/inventory/profiles")
def inventory_profiles() -> dict[str, Any]:
    from flora_translate.inventory_profiles import list_inventory_profiles

    return {"profiles": list_inventory_profiles()}


@app.get("/api/inventory/profiles/{profile_id}")
def inventory_profile(profile_id: str) -> dict[str, Any]:
    from flora_translate.inventory_profiles import load_inventory_profile

    try:
        return load_inventory_profile(profile_id).model_dump()
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/inventory/import")
def import_inventory(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    from flora_translate.inventory_profiles import inventory_profile_from_payload

    try:
        return inventory_profile_from_payload(payload).model_dump()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/inventory/extract")
async def extract_inventory(
    name: str = Form("Imported laboratory inventory"),
    laboratory: str = Form(""),
    source_text: str = Form(""),
    use_llm: bool = Form(True),
    files: list[UploadFile] = File(default=[]),
) -> dict[str, Any]:
    from flora_translate.inventory_profiles import (
        extract_inventory_profile,
        extract_uploaded_documents,
    )

    uploaded: list[tuple[str, bytes, str]] = []
    for item in files:
        uploaded.append((item.filename or "upload", await item.read(), item.content_type or ""))
    try:
        document_text, provenance, errors = extract_uploaded_documents(uploaded)
        combined = "\n\n".join(value for value in (source_text, document_text) if value.strip())
        profile = extract_inventory_profile(
            combined,
            name=name,
            laboratory=laboratory,
            use_llm=use_llm,
        )
        profile.provenance = provenance
        response = profile.model_dump()
        response["document_errors"] = errors
        response["source_preview"] = combined[:5000]
        return response
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/inventory/save")
def save_inventory(payload: InventorySavePayload) -> dict[str, Any]:
    from flora_translate.inventory_profiles import (
        inventory_profile_from_payload,
        save_inventory_profile,
    )

    try:
        profile = inventory_profile_from_payload(payload.profile)
        saved, path = save_inventory_profile(profile)
        return {"profile": saved.model_dump(), "path": str(path)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/design/jobs", status_code=202)
def create_design_job(payload: DesignPayload) -> dict[str, Any]:
    try:
        package = _apply_inventory(deepcopy(payload.intake_package), payload.inventory_profile)
        if not package.get("ready_for_design"):
            missing = ", ".join(package.get("missing_question_ids") or [])
            raise ValueError(f"Intake is incomplete: {missing}")
        request = payload.model_dump()
        request["intake_package"] = package
        if payload.upstream_model_id or payload.downstream_model_id:
            from flora_translate.model_catalog import (
                model_route_availability,
                route_for_id,
                runtime_model_options,
            )

            selected = {
                "upstream": route_for_id(payload.upstream_model_id, role="upstream"),
                "downstream": route_for_id(payload.downstream_model_id, role="downstream"),
            }
            for role, route in selected.items():
                status = model_route_availability(route)
                if not status["available"]:
                    raise ValueError(
                        f"{role.capitalize()} model {route.label} is unavailable: "
                        f"{status['reason']}"
                    )

            runtime_options = dict(payload.runtime_options or {})
            runtime_options.update(
                runtime_model_options(
                    payload.upstream_model_id,
                    payload.downstream_model_id,
                )
            )
            request["runtime_options"] = runtime_options
        job = JOBS.submit(request)
        return job.public(include_result=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/design/jobs")
def list_design_jobs() -> dict[str, Any]:
    return {"jobs": JOBS.list()}


@app.get("/api/design/jobs/{job_id}")
def get_design_job(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Design job not found")
    return job.public(include_result=True)


def _artifact_for_job(job_id: str, kind: str) -> Path:
    job = JOBS.get(job_id)
    if not job or not job.result:
        raise HTTPException(status_code=404, detail="Completed design job not found")
    keys = {
        "process-png": "png_path",
        "process-svg": "svg_path",
        "diagnostic-png": "diagnostic_png_path",
        "diagnostic-svg": "diagnostic_svg_path",
    }
    key = keys.get(kind)
    if not key:
        raise HTTPException(status_code=404, detail="Unknown artifact")
    path = Path(str(job.result.get(key) or ""))
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact is unavailable")
    return path


@app.get("/api/design/jobs/{job_id}/artifacts/{kind}")
def design_artifact(job_id: str, kind: str) -> FileResponse:
    path = _artifact_for_job(job_id, kind)
    media_type = "image/svg+xml" if path.suffix.lower() == ".svg" else "image/png"
    return FileResponse(path, media_type=media_type, filename=path.name)


@app.get("/api/design/jobs/{job_id}/download")
def download_design(job_id: str) -> JSONResponse:
    job = JOBS.get(job_id)
    if not job or not job.result:
        raise HTTPException(status_code=404, detail="Completed design job not found")
    return JSONResponse(job.result, headers={"Content-Disposition": f'attachment; filename="flowpilot_{job_id}.json"'})


@app.post("/api/refinement")
def refine_design(payload: RefinementPayload) -> dict[str, Any]:
    from flora_translate.experiment_loop import (
        ExperimentResult,
        refine_from_experiment,
        refine_from_experimental_campaign,
    )
    from flora_translate.gui_autosave import autosave_gui_result

    current = payload.current_result
    if payload.job_id:
        job = JOBS.get(payload.job_id)
        current = job.result if job else None
    if not current:
        raise HTTPException(status_code=400, detail="A completed design result is required")
    if not payload.experiments:
        raise HTTPException(status_code=400, detail="At least one experimental result is required")
    try:
        experiments = [ExperimentResult.model_validate(item) for item in payload.experiments]
        if len(experiments) == 1:
            closed = refine_from_experiment(
                current,
                experiments[0],
                target_yield_pct=payload.target_yield_pct,
                target_conversion_pct=payload.target_conversion_pct,
                target_selectivity_pct=payload.target_selectivity_pct,
            )
        else:
            closed = refine_from_experimental_campaign(
                current,
                experiments,
                target_yield_pct=payload.target_yield_pct,
                target_conversion_pct=payload.target_conversion_pct,
                target_selectivity_pct=payload.target_selectivity_pct,
            )
        result = closed.model_dump()
        save_dir = autosave_gui_result(
            result["refined_result"],
            source="webapp_refinement",
            user_input=json.dumps(payload.experiments, default=str),
        )
        result["autosave_dir"] = str(save_dir)
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/runs")
def saved_runs(limit: int = 50) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    root = Path("outputs/gui_runs")
    if root.is_dir():
        for directory in sorted((item for item in root.iterdir() if item.is_dir()), reverse=True):
            summary_path = directory / "summary.json"
            if not summary_path.is_file():
                continue
            try:
                summary = json.loads(summary_path.read_text())
            except Exception:
                continue
            summary["run_id"] = directory.name
            summary["path"] = str(directory)
            summary["has_process_png"] = (directory / "process.png").is_file()
            rows.append(summary)
            if len(rows) >= max(1, min(limit, 200)):
                break
    return {"runs": rows}


def _saved_run_directory(run_id: str) -> Path:
    if not run_id or Path(run_id).name != run_id:
        raise HTTPException(status_code=400, detail="Invalid saved run identifier")
    directory = Path("outputs/gui_runs") / run_id
    if not directory.is_dir():
        raise HTTPException(status_code=404, detail="Saved run not found")
    return directory


@app.get("/api/runs/{run_id}")
def saved_run(run_id: str) -> dict[str, Any]:
    directory = _saved_run_directory(run_id)
    result_path = directory / "result.json"
    if not result_path.is_file():
        raise HTTPException(status_code=404, detail="Saved result JSON is unavailable")
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Saved result is invalid: {exc}") from exc
    result["autosave_dir"] = str(directory)
    return result


@app.get("/api/runs/{run_id}/artifacts/{kind}")
def saved_run_artifact(run_id: str, kind: str) -> FileResponse:
    directory = _saved_run_directory(run_id)
    names = {
        "process-png": "process.png",
        "process-svg": "process.svg",
        "diagnostic-png": "diagnostic_process.png",
        "diagnostic-svg": "diagnostic_process.svg",
    }
    filename = names.get(kind)
    if not filename:
        raise HTTPException(status_code=404, detail="Unknown artifact")
    path = directory / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Saved artifact is unavailable")
    media_type = "image/svg+xml" if path.suffix.lower() == ".svg" else "image/png"
    return FileResponse(path, media_type=media_type, filename=path.name)


@app.get("/api/runs/{run_id}/download")
def download_saved_run(run_id: str) -> FileResponse:
    path = _saved_run_directory(run_id) / "result.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Saved result JSON is unavailable")
    return FileResponse(path, media_type="application/json", filename=f"flowpilot_{run_id}.json")


if (DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")


@app.get("/{full_path:path}")
def frontend(full_path: str) -> FileResponse:
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API route not found")
    candidate = (DIST / full_path).resolve()
    if full_path and candidate.is_file() and DIST.resolve() in candidate.parents:
        return FileResponse(candidate)
    index = DIST / "index.html"
    if index.is_file():
        return FileResponse(index)
    raise HTTPException(status_code=503, detail="Frontend has not been built. Run npm run build.")
