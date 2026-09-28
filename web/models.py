"""Model catalogue behind the UI's provider/model pickers.

Sources, each fetched independently and cached (an hour for catalogues, 30s
for local daemons whose model lists change with a pull/load):
  - openrouter    OpenRouter's public rate card (no key needed)
  - ollama        models the local Ollama daemon has pulled (incl. cloud tags)
  - ollama_cloud  Ollama's cloud library (search page + per-model tag pages)
  - anthropic     Anthropic's Models API (direct; needs ANTHROPIC_API_KEY)
  - openai        OpenAI's models endpoint (direct; needs OPENAI_API_KEY)
  - lmstudio      models loaded in a local LM Studio server
  - moonshot      Moonshot AI (Kimi) direct; needs MOONSHOT_API_KEY
  - byo           your own OpenAI-compatible endpoint (BYO_API_URL / BYO_API_KEY)
  - claudecode    Claude via the Claude Code CLI (claude.ai subscription login)
  - codex         ChatGPT models via the Codex CLI (ChatGPT subscription login)

A source that fails yields [] plus an entry in `errors`; the UI shows what it
got and the free-text override still works.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

_TTL = 3600.0
_cache: dict[str, tuple[float, Any]] = {}
_lock = threading.Lock()


def _cached(key: str, fn: Callable[[], Any], ttl: float = _TTL) -> Any:
    now = time.time()
    with _lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
    val = fn()
    with _lock:
        _cache[key] = (now, val)
    return val


class Unavailable(RuntimeError):
    """A source that cannot be queried for a stated reason (shown verbatim in the UI)."""


def _get(url: str, timeout: float = 8.0, headers: dict[str, str] | None = None) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "llm-council/0.1", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="ignore")


# --- parsers (pure; unit-tested offline) --------------------------------------

def parse_openrouter(data: dict) -> list[dict[str, str]]:
    """Rate card -> [{id, label}] sorted by id; ':batch' variants dropped."""
    out = []
    for m in data.get("data", []):
        mid = str(m.get("id") or "")
        if not mid or mid.endswith(":batch"):
            continue
        p = m.get("pricing") or {}
        try:
            pin = float(p.get("prompt") or 0) * 1e6
            pout = float(p.get("completion") or 0) * 1e6
            label = f"{mid} · ${pin:.2f}/${pout:.2f} per M"
        except (TypeError, ValueError):
            label = mid
        out.append({"id": mid, "label": label})
    out.sort(key=lambda x: x["id"])
    return out


def parse_ollama_tags(data: dict) -> list[dict[str, str]]:
    names = sorted({str(m.get("name") or "") for m in data.get("models", []) if m.get("name")})
    return [{"id": n, "label": n + (" (cloud)" if n.endswith("cloud") else " (local)")} for n in names]


_LIB_RE = re.compile(r'href="/library/([a-z0-9._-]+)"')


def parse_cloud_library(html: str) -> list[str]:
    return sorted(set(_LIB_RE.findall(html)))


def parse_cloud_tags(name: str, html: str) -> list[str]:
    return sorted(set(re.findall(re.escape(name) + r":[a-z0-9._-]*cloud[a-z0-9._-]*", html)))


def parse_anthropic_models(data: dict) -> list[dict[str, str]]:
    out = []
    for m in data.get("data", []):
        mid = str(m.get("id") or "")
        if mid:
            name = str(m.get("display_name") or "")
            out.append({"id": mid, "label": f"{mid} · {name}" if name and name != mid else mid})
    out.sort(key=lambda x: x["id"])
    return out


_OPENAI_SKIP = ("audio", "realtime", "image", "tts", "transcribe", "embedding", "moderation",
                "whisper", "dall-e", "davinci", "babbage", "instruct", "search")


def parse_openai_models(data: dict) -> list[dict[str, str]]:
    """Chat-capable models only: gpt-* and o-series, minus audio/image/embedding variants."""
    out = []
    for m in data.get("data", []):
        mid = str(m.get("id") or "")
        if not mid or any(k in mid for k in _OPENAI_SKIP):
            continue
        if not (mid.startswith("gpt") or re.match(r"^o\d", mid) or mid.startswith("chatgpt")):
            continue
        out.append({"id": mid, "label": mid})
    out.sort(key=lambda x: x["id"])
    return out


def derive_direct_from_openrouter(openrouter: list[dict[str, str]], vendor: str) -> list[dict[str, str]]:
    """Without a vendor key we can't list its models, but OpenRouter's catalogue
    names them. OpenAI slugs are the vendor's own ids; Anthropic's direct ids
    replace the version dot with a dash (anthropic/claude-opus-4.8 -> claude-opus-4-8)."""
    out = []
    for m in openrouter:
        mid = m["id"]
        if not mid.startswith(vendor + "/"):
            continue
        slug = mid.split("/", 1)[1]
        if vendor == "anthropic":
            slug = slug.replace(".", "-")
        out.append({"id": slug, "label": slug})
    return out


def parse_openai_compatible_models(data: dict, suffix: str = "") -> list[dict[str, str]]:
    ids = sorted({str(m.get("id") or "") for m in data.get("data", []) if m.get("id")})
    return [{"id": i, "label": i + suffix} for i in ids]


# --- fetchers -----------------------------------------------------------------

def openrouter_models() -> list[dict[str, str]]:
    return _cached("openrouter", lambda: parse_openrouter(json.loads(_get("https://openrouter.ai/api/v1/models"))))


def ollama_local_models() -> list[dict[str, str]]:
    base = (os.getenv("OLLAMA_API_URL") or "http://localhost:11434").rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    # Short TTL: `ollama pull` changes this list.
    return _cached("ollama", lambda: parse_ollama_tags(json.loads(_get(f"{base}/api/tags", timeout=3))), ttl=30)


def ollama_cloud_models() -> list[dict[str, str]]:
    def fetch() -> list[dict[str, str]]:
        names = parse_cloud_library(_get("https://ollama.com/search?c=cloud"))

        def tags_for(name: str) -> list[str]:
            try:
                return parse_cloud_tags(name, _get(f"https://ollama.com/library/{name}/tags"))
            except Exception:
                return [f"{name}:cloud"]  # best guess when the tag page is unreachable

        with ThreadPoolExecutor(max_workers=8) as ex:
            tags = sorted({t for lst in ex.map(tags_for, names) for t in lst})
        return [{"id": t, "label": t} for t in tags]

    return _cached("ollama_cloud", fetch)


def anthropic_models() -> list[dict[str, str]]:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise Unavailable("set ANTHROPIC_API_KEY")
    hdr = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    return _cached("anthropic", lambda: parse_anthropic_models(
        json.loads(_get("https://api.anthropic.com/v1/models?limit=1000", headers=hdr))))


def openai_models() -> list[dict[str, str]]:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise Unavailable("set OPENAI_API_KEY")
    hdr = {"Authorization": f"Bearer {key}"}
    return _cached("openai", lambda: parse_openai_models(
        json.loads(_get("https://api.openai.com/v1/models", headers=hdr))))


def lmstudio_models() -> list[dict[str, str]]:
    base = (os.getenv("LMSTUDIO_API_URL") or "http://localhost:1234").rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    key = os.getenv("LMSTUDIO_API_KEY")
    hdr = {"Authorization": f"Bearer {key}"} if key else {}
    return _cached("lmstudio", lambda: parse_openai_compatible_models(
        json.loads(_get(f"{base}/models", timeout=3, headers=hdr)), " (LM Studio)"), ttl=30)


def moonshot_models() -> list[dict[str, str]]:
    key = os.getenv("MOONSHOT_API_KEY")
    if not key:
        raise Unavailable("set MOONSHOT_API_KEY")
    base = (os.getenv("MOONSHOT_API_URL") or "https://api.moonshot.ai/v1").rstrip("/")
    hdr = {"Authorization": f"Bearer {key}"}
    return _cached("moonshot", lambda: parse_openai_compatible_models(
        json.loads(_get(f"{base}/models", headers=hdr))))


def byo_models() -> list[dict[str, str]]:
    base = (os.getenv("BYO_API_URL") or "").rstrip("/")
    if not base:
        raise Unavailable("set BYO_API_URL")
    key = os.getenv("BYO_API_KEY")
    hdr = {"Authorization": f"Bearer {key}"} if key else {}
    return _cached("byo", lambda: parse_openai_compatible_models(
        json.loads(_get(f"{base}/models", headers=hdr))), ttl=30)


_CLAUDE_CODE_MODELS = ["default", "opus", "sonnet", "haiku",
                       "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5"]


def claudecode_models() -> list[dict[str, str]]:
    if not shutil.which("claude"):
        raise Unavailable("install Claude Code and run `claude` once to log in")
    return [{"id": m, "label": m + (" (Claude Code's configured model)" if m == "default" else "")}
            for m in _CLAUDE_CODE_MODELS]


def _codex_default_model() -> str:
    try:
        for line in open(os.path.expanduser("~/.codex/config.toml"), encoding="utf-8"):
            m = re.match(r'\s*model\s*=\s*"([^"]+)"', line)
            if m:
                return m.group(1)
    except OSError:
        pass
    return ""


def parse_codex_cache(data: dict) -> list[dict[str, str]]:
    """Codex's per-account model cache (~/.codex/models_cache.json): the models
    the ChatGPT login is entitled to. Only 'list'-visible ones, in Codex's order."""
    rows = [m for m in data.get("models", []) if m.get("slug") and m.get("visibility", "list") == "list"]
    rows.sort(key=lambda m: m.get("priority", 999))
    return [{"id": m["slug"], "label": f"{m['slug']} · {m.get('display_name') or ''}".rstrip(" ·")} for m in rows]


def codex_models() -> list[dict[str, str]]:
    if not shutil.which("codex"):
        raise Unavailable("install the Codex CLI and run `codex login`")
    d = _codex_default_model()
    out = [{"id": "default", "label": f"default ({d})" if d else "default (Codex's configured model)"}]
    try:
        out += parse_codex_cache(json.load(open(os.path.expanduser("~/.codex/models_cache.json"), encoding="utf-8")))
    except (OSError, ValueError):
        pass  # no cache yet: "default" still works; a ChatGPT login rejects arbitrary ids
    return out


def catalogue() -> dict[str, Any]:
    """All sources, fetched concurrently; per-source failure is reported, not raised."""
    sources = (("openrouter", openrouter_models), ("ollama", ollama_local_models),
               ("ollama_cloud", ollama_cloud_models), ("anthropic", anthropic_models),
               ("openai", openai_models), ("lmstudio", lmstudio_models),
               ("moonshot", moonshot_models), ("claudecode", claudecode_models),
               ("codex", codex_models), ("byo", byo_models))
    # notes: a source that can't be queried for a stated reason (shown verbatim);
    # errors: a source that failed unexpectedly (shown as "unavailable").
    out: dict[str, Any] = {"errors": {}, "notes": {}}

    def one(item: tuple[str, Callable[[], Any]]) -> None:
        key, fn = item
        try:
            out[key] = fn()
        except Unavailable as e:
            out[key] = []
            out["notes"][key] = str(e)
        except Exception as e:
            out[key] = []
            out["errors"][key] = f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=len(sources)) as ex:
        list(ex.map(one, sources))
    # No vendor key yet: still offer a dropdown, derived from OpenRouter's catalogue,
    # and keep the "set KEY" note so the UI can say the seat won't run until it's set.
    for vendor in ("anthropic", "openai"):
        if not out.get(vendor) and str(out["notes"].get(vendor, "")).startswith("set "):
            out[vendor] = derive_direct_from_openrouter(out.get("openrouter") or [], vendor)
    return out
