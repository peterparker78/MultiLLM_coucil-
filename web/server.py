"""Web UI for the council (FastAPI).

    PYTHONPATH=. python3 -m web.server                 # live (needs API keys)
    PYTHONPATH=. COUNCIL_DEMO=1 python3 -m web.server  # offline demo, no keys

A run is executed in a background thread so the browser can poll live stage
progress. Jobs are kept in memory (single-user local tool).
"""

from __future__ import annotations

import os
import shutil
import threading
import uuid
import webbrowser
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import json
import re
import tempfile

from council.clientfactory import make_client
from council.config import CouncilConfig
from council.council import run_council
from council.fake import DemoClient
from council import report
from council.serialize import run_to_dict

from advisory.advisory import run_advisory, to_dict as adv_to_dict
from advisory.lenses import AdvisoryConfig, lens_for
from council.brief import combine_briefs, text_from_bytes
from . import models as model_catalogue
from advisory import report as adv_report

_RUNS_DIR = Path(__file__).parent.parent / "council_runs"
_ADV_RUNS_DIR = Path(__file__).parent.parent / "advisory_runs"

app = FastAPI(title="LLM Council")
_HERE = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")

@app.middleware("http")
async def _no_store(request, call_next):
    """Pages and static assets change with every commit; never let the browser
    show a stale copy after the server is restarted."""
    resp = await call_next(request)
    if request.url.path == "/" or request.url.path in ("/advisory",) or request.url.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


# job_id -> {status, stage, label, result, error}
_JOBS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()


def _demo_mode() -> bool:
    return os.getenv("COUNCIL_DEMO") == "1"


def _client() -> Any:
    return DemoClient() if _demo_mode() else make_client()


def _keys_present() -> dict[str, bool]:
    return {
        "openai (OPENAI_API_KEY)": bool(os.getenv("OPENAI_API_KEY")),
        "anthropic (ANTHROPIC_API_KEY)": bool(os.getenv("ANTHROPIC_API_KEY")),
        "moonshot (MOONSHOT_API_KEY)": bool(os.getenv("MOONSHOT_API_KEY")),
        "your endpoint (BYO_API_URL)": bool(os.getenv("BYO_API_URL")),
        "claude subscription (Claude Code CLI)": bool(shutil.which("claude")),
        "chatgpt subscription (Codex CLI)": bool(shutil.which("codex")),
        "openrouter (OPENROUTER_API_KEY)": bool(os.getenv("OPENROUTER_API_KEY")),
    }


class BriefIn(BaseModel):
    name: str = "document"
    text: str = ""


class RunRequest(BaseModel):
    prompt: str
    seats: Optional[list[str]] = None
    chairman: Optional[str] = None
    modes: list[str] = ["baseline", "preserving"]
    verify: bool = False  # fact-check syntheses with the web-capable verifier
    compose: bool = False  # hidden-profile pass: emergent cross-model composites
    briefs: list[BriefIn] = []  # briefing documents shown to every seat


class ReviewRequest(BaseModel):
    overall: str = ""
    ideas: dict[str, dict] = {}  # idea_id -> {verdict, comment}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_HERE / "index.html")


@app.get("/api/config")
def config() -> dict[str, Any]:
    cfg = CouncilConfig()
    return {
        "seats": cfg.seats,
        "chairman": cfg.chairman,
        "demo": _demo_mode(),
        "keys": _keys_present(),
    }


@app.get("/api/models")
def models() -> dict[str, Any]:
    """Provider/model lists for the pickers; each source degrades independently."""
    return model_catalogue.catalogue()


@app.post("/api/brief")
async def upload_briefs(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """Convert one or more briefing files to text. Files that cannot be read are
    reported in `errors` rather than failing the whole upload."""
    cap = max(CouncilConfig().brief_max_chars, AdvisoryConfig().brief_max_chars)
    briefs, errors = [], []
    for f in files:
        name = f.filename or "document"
        try:
            text = text_from_bytes(name, await f.read(), max_chars=cap)
            briefs.append({"name": name, "chars": len(text), "text": text})
        except Exception as e:
            errors.append(f"{name}: {e}")
    return {"briefs": briefs, "errors": errors}


_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def _runs_dir_for(result: dict) -> Path:
    return _ADV_RUNS_DIR if result.get("kind") == "advisory" else _RUNS_DIR


def _report_for(result: dict):
    return adv_report if result.get("kind") == "advisory" else report


def _save_run(job_id: str, result: dict) -> None:
    # id-based filename so runs are addressable for reopen; prompt is inside.
    d = _runs_dir_for(result)
    d.mkdir(exist_ok=True)
    with open(d / f"{job_id}.json", "w") as f:
        json.dump(result, f, indent=2)


def _worker(job_id: str, req: RunRequest) -> None:
    def progress(key: str, label: str) -> None:
        with _LOCK:
            _JOBS[job_id]["stage"] = key
            _JOBS[job_id]["label"] = label

    try:
        cfg = CouncilConfig()
        if req.seats:
            cfg.seats = [s.strip() for s in req.seats if s.strip()]
        if req.chairman:
            cfg.chairman = req.chairman
        cfg.verify = req.verify
        cfg.compose = req.compose
        brief_text = combine_briefs([(b.name, b.text) for b in req.briefs], max_chars=cfg.brief_max_chars)
        run = run_council(
            _client(), req.prompt, cfg, modes=tuple(req.modes), progress=progress,
            brief_text=brief_text,
        )
        result = run_to_dict(run)
        _save_run(job_id, result)
        with _LOCK:
            _JOBS[job_id].update(status="done", result=result)
    except Exception as e:  # surface the error to the UI instead of dying silently
        with _LOCK:
            _JOBS[job_id].update(status="error", error=f"{type(e).__name__}: {e}")


@app.post("/api/run")
def start_run(req: RunRequest) -> dict[str, str]:
    if not req.prompt.strip():
        raise HTTPException(400, "prompt is required")
    job_id = uuid.uuid4().hex[:12]
    with _LOCK:
        _JOBS[job_id] = {"status": "running", "stage": "queued", "label": "Queued"}
    threading.Thread(target=_worker, args=(job_id, req), daemon=True).start()
    return {"id": job_id}


@app.get("/api/run/{job_id}")
def run_status(job_id: str) -> dict[str, Any]:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job")
        return dict(job)


@app.get("/api/runs")
def list_runs() -> dict[str, Any]:
    """List saved runs (newest first) with a small summary for the history view."""
    runs = []
    if _RUNS_DIR.exists():
        for p in sorted(_RUNS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            gaps = {
                m: round(r["survival"]["shared_rate"] - r["survival"]["unique_rate"], 2)
                for m, r in (d.get("results") or {}).items()
            }
            review = d.get("review") or {}
            runs.append({
                "id": p.stem,
                "prompt": d.get("prompt", ""),
                "mtime": int(p.stat().st_mtime),
                "seats": [m.get("model") for m in d.get("members", [])],
                "cost_usd": (d.get("cost") or {}).get("total_cost_usd"),
                "gaps": gaps,
                "reviewed": bool(review.get("ideas") or review.get("overall")),
            })
    return {"runs": runs}


@app.get("/api/runs/{run_id}")
def open_run(run_id: str) -> dict[str, Any]:
    """Load a saved run and re-register it in memory so review/export work on it."""
    if not _ID_RE.match(run_id):
        raise HTTPException(400, "bad run id")
    p = _RUNS_DIR / f"{run_id}.json"
    if not p.exists():
        raise HTTPException(404, "unknown run")
    result = json.loads(p.read_text())
    with _LOCK:
        _JOBS[run_id] = {"status": "done", "result": result}
    return {"id": run_id, "result": result}


@app.post("/api/run/{job_id}/review")
def save_review(job_id: str, req: ReviewRequest) -> dict[str, bool]:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job")
        if job.get("status") != "done" or not job.get("result"):
            raise HTTPException(409, "run not finished")
        job["result"]["review"] = {"overall": req.overall, "ideas": req.ideas}
        result = job["result"]
    _save_run(job_id, result)  # re-persist so the review survives and exports include it
    return {"ok": True}


def _completed_result(job_id: str) -> dict:
    with _LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    if job.get("status") != "done" or not job.get("result"):
        raise HTTPException(409, "run not finished")
    return job["result"]


def _filename(result: dict, ext: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in result.get("prompt", ""))[:40]
    return f"{safe or 'council'}.{ext}"


@app.get("/api/run/{job_id}/export.md")
def export_md(job_id: str) -> PlainTextResponse:
    result = _completed_result(job_id)
    return PlainTextResponse(
        _report_for(result).to_markdown(result),
        headers={"Content-Disposition": f'attachment; filename="{_filename(result, "md")}"'},
    )


@app.get("/api/run/{job_id}/export.docx")
def export_docx(job_id: str) -> Response:
    result = _completed_result(job_id)
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        _report_for(result).to_docx(result, tmp_path)
        data = open(tmp_path, "rb").read()
    finally:
        os.unlink(tmp_path)  # don't accumulate export copies in /tmp
    return Response(
        content=data,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{_filename(result, "docx")}"'},
    )


# --- Advisory board (separate page, shared backend) -------------------------

class LensIn(BaseModel):
    name: str
    model: str
    system: str = ""  # editable persona; blank → resolved from defaults/generic


class AdvisoryRunRequest(BaseModel):
    question: str
    lenses: Optional[list[LensIn]] = None
    brief_text: str = ""  # legacy single briefing; `briefs` is the multi-document form
    briefs: list[BriefIn] = []
    verify: bool = False
    ground: bool = False  # search PubMed + Europe PMC and ground the board
    compose: bool = False  # hidden-profile pass: emergent cross-lens composites


@app.get("/advisory")
def advisory_page() -> FileResponse:
    return FileResponse(_HERE / "advisory.html")


@app.get("/api/advisory/config")
def advisory_config() -> dict[str, Any]:
    cfg = AdvisoryConfig()
    return {
        "lenses": [{"name": l.name, "model": l.model, "system": l.system} for l in cfg.lenses],
        "verifier": cfg.verifier,
        "demo": _demo_mode(),
        "keys": _keys_present(),
    }


def _advisory_worker(job_id: str, req: AdvisoryRunRequest) -> None:
    def progress(key: str, label: str) -> None:
        with _LOCK:
            _JOBS[job_id]["stage"] = key
            _JOBS[job_id]["label"] = label

    try:
        cfg = AdvisoryConfig()
        if req.lenses:
            cfg.lenses = [lens_for(l.name, l.model, l.system or None)
                          for l in req.lenses if l.model.strip()]
        cfg.verify = req.verify
        cfg.ground = req.ground
        cfg.compose = req.compose
        docs = [(b.name, b.text) for b in req.briefs]
        if req.brief_text.strip():
            docs.insert(0, ("briefing", req.brief_text))
        brief_text = combine_briefs(docs, max_chars=cfg.brief_max_chars)
        res = run_advisory(_client(), req.question, cfg, brief_text=brief_text, progress=progress)
        result = adv_to_dict(res)
        _save_run(job_id, result)
        with _LOCK:
            _JOBS[job_id].update(status="done", result=result)
    except Exception as e:
        with _LOCK:
            _JOBS[job_id].update(status="error", error=f"{type(e).__name__}: {e}")


@app.post("/api/advisory/run")
def start_advisory(req: AdvisoryRunRequest) -> dict[str, str]:
    if not req.question.strip():
        raise HTTPException(400, "question is required")
    job_id = uuid.uuid4().hex[:12]
    with _LOCK:
        _JOBS[job_id] = {"status": "running", "stage": "queued", "label": "Queued"}
    threading.Thread(target=_advisory_worker, args=(job_id, req), daemon=True).start()
    return {"id": job_id}


@app.get("/api/advisory/runs")
def advisory_runs() -> dict[str, Any]:
    runs = []
    if _ADV_RUNS_DIR.exists():
        for p in sorted(_ADV_RUNS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            review = d.get("review") or {}
            runs.append({
                "id": p.stem,
                "question": d.get("question", ""),
                "mtime": int(p.stat().st_mtime),
                "lenses": [e.get("label") for e in d.get("experts", [])],
                "cost_usd": (d.get("cost") or {}).get("total_cost_usd"),
                "reviewed": bool(review.get("ideas") or review.get("overall")),
            })
    return {"runs": runs}


@app.get("/api/advisory/runs/{run_id}")
def advisory_open(run_id: str) -> dict[str, Any]:
    if not _ID_RE.match(run_id):
        raise HTTPException(400, "bad run id")
    p = _ADV_RUNS_DIR / f"{run_id}.json"
    if not p.exists():
        raise HTTPException(404, "unknown run")
    result = json.loads(p.read_text())
    with _LOCK:
        _JOBS[run_id] = {"status": "done", "result": result}
    return {"id": run_id, "result": result}


def main() -> None:
    host = os.getenv("COUNCIL_HOST", "127.0.0.1")
    port = int(os.getenv("COUNCIL_PORT", "8000"))
    mode = "DEMO (no keys)" if _demo_mode() else "LIVE"
    url = f"http://{host}:{port}"
    print(f"LLM Council [{mode}] -> {url}")
    # Auto-open the browser shortly after the server starts (disable with COUNCIL_NO_BROWSER=1).
    if not os.getenv("COUNCIL_NO_BROWSER"):
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
