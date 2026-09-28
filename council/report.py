"""Export a council run to Markdown or DOCX. Shared by the web server and CLI.

DOCX follows the house style: Garamond throughout, headings in the deep-blue
palette colour, dissent/minority figures flagged in alert red.
"""

from __future__ import annotations

import re
from typing import Any

_TEX = {
    "\\Delta": "Δ", "\\delta": "δ", "\\ge": "≥", "\\geq": "≥", "\\le": "≤", "\\leq": "≤",
    "\\rightarrow": "→", "\\to": "→", "\\leftarrow": "←", "\\times": "×", "\\approx": "≈",
    "\\pm": "±", "\\neq": "≠", "\\alpha": "α", "\\beta": "β", "\\mu": "µ", "\\sigma": "σ",
    "\\sim": "~", "\\cdot": "·", "\\%": "%",
}


def _strip_math(s: str) -> str:
    """Convert inline LaTeX ($...$, \\(...\\)) to plain Unicode so the DOCX
    doesn't show raw $\\Delta$ etc. Single $ (currency) is left alone."""
    def tex(t: str) -> str:
        for k, v in _TEX.items():
            t = t.replace(k, v)
        return t.replace("{", "").replace("}", "")
    s = re.sub(r"\$([^$\n]+?)\$", lambda m: tex(m.group(1)), s)
    return re.sub(r"\\\(([^\n]+?)\\\)", lambda m: tex(m.group(1)), s)

# House palette (from the PPTX template spec)
NAVY = "015685"
BLUE = "016AA3"
SLATE = "2C3E50"
GREEN = "38906C"
RED = "E74C3C"
MUTED = "7F8C8D"


_VERDICT = {"adopt": "Adopt", "explore": "Explore", "reject": "Reject"}


def _pct(x: float) -> str:
    return f"{round(100 * x)}%"


_COMPOSITE_NOTE = ("_Emergent composites combine fragments across models; by design "
                   "they are not traceable to any single model and are not fact-checked. "
                   "Evaluate independently._")


def composites_markdown(composites: list) -> list[str]:
    out: list[str] = []
    if not composites:
        return out
    out.append("## Emergent composites (cross-model)\n")
    out.append(_COMPOSITE_NOTE + "\n")
    for c in composites:
        who = ", ".join(c.get("source_models", []))
        badge = c.get("verdict") or c.get("value", "")
        out.append(f"- **[{badge}]** {c.get('summary','')}  \n  _combines: {who}_"
                   + (f"  \n  _judge: {c.get('judge_note','')}_" if c.get("judge_note") else ""))
    out.append("")
    return out


def composites_docx(doc: Any, composites: list) -> None:
    if not composites:
        return
    _docx_heading(doc, "Emergent composites (cross-model)")
    note = doc.add_paragraph()
    nr = note.add_run(_COMPOSITE_NOTE.strip("_"))
    nr.italic = True
    nr.font.name = "Garamond"
    for c in composites:
        p = doc.add_paragraph(style="List Bullet")
        who = ", ".join(c.get("source_models", []))
        badge = c.get("verdict") or c.get("value", "")
        note = f" — judge: {c.get('judge_note','')}" if c.get("judge_note") else ""
        _inline_runs(p, f"**[{badge}]** {c.get('summary','')} (combines: {who}){note}")


def _rated_ideas(result: dict[str, Any]) -> list[tuple[str, str, dict]]:
    """Return [(summary, verdict_label, annotation)] for ideas the reviewer rated."""
    review = result.get("review") or {}
    ideas = review.get("ideas") or {}
    summ = {c["id"]: c.get("summary", "") for c in result.get("ledger", [])}
    out = []
    for iid, a in ideas.items():
        if a.get("verdict") or a.get("comment"):
            out.append((summ.get(iid, iid), _VERDICT.get(a.get("verdict", ""), "—"), a))
    return out


def to_markdown(result: dict[str, Any]) -> str:
    out: list[str] = []
    out.append(f"# Council report\n")
    out.append(f"**Prompt:** {result.get('prompt','')}\n")

    out.append("## Members\n")
    for m in result.get("members", []):
        st = m.get("status", "ok")
        detail = f"FAILED — {m.get('error')}" if st == "failed" else st
        out.append(f"- **{m.get('label')}** ({m.get('model')}) — {detail}")
    out.append("")

    ledger = result.get("ledger", [])
    uniq = sum(1 for c in ledger if c.get("is_unique"))
    out.append("## Idea-survival (groupthink metric)\n")
    out.append(f"Ledger: {len(ledger)} distinct ideas ({uniq} minority / {len(ledger)-uniq} shared)\n")
    out.append("| Mode | Minority kept | Consensus kept | Gap |")
    out.append("|---|---|---|---|")
    for mode, res in result.get("results", {}).items():
        s = res["survival"]
        gap = s["shared_rate"] - s["unique_rate"]
        out.append(
            f"| {mode} | {_pct(s['unique_rate'])} ({s['unique_survived']}/{s['unique_total']}) "
            f"| {_pct(s['shared_rate'])} ({s['shared_survived']}/{s['shared_total']}) | {_pct(gap)} |"
        )
    out.append("")

    cost = result.get("cost")
    if cost:
        out.append("## Cost\n")
        partial = " (partial — some models unpriced)" if cost.get("cost_partial") else ""
        out.append(f"Total: **${cost.get('total_cost_usd', 0):.4f}**{partial} · "
                   f"{cost.get('total_tokens', 0):,} tokens\n")
        out.append("| Model | Calls | Prompt tok | Completion tok | Cost |")
        out.append("|---|---|---|---|---|")
        for r in cost.get("per_model", []):
            c = f"${r['cost_usd']:.4f}" if r.get("cost_usd") is not None else "n/a"
            out.append(f"| {r['model']} | {r['calls']} | {r['prompt_tokens']:,} "
                       f"| {r['completion_tokens']:,} | {c} |")
        out.append("")

    review = result.get("review") or {}
    rated = _rated_ideas(result)
    if review.get("overall") or rated:
        out.append("## Reviewer assessment\n")
        if review.get("overall"):
            out.append(review["overall"].strip() + "\n")
        if rated:
            out.append("| Idea | Verdict | Comment |")
            out.append("|---|---|---|")
            for summary, verdict, a in rated:
                out.append(f"| {summary} | {verdict} | {a.get('comment', '')} |")
        out.append("")

    for mode, res in result.get("results", {}).items():
        out.append(f"## Synthesis — {mode}\n")
        out.append(res.get("synthesis", "").strip() or "_(empty)_")
        unsup = res.get("unsupported_claims") or []
        if unsup:
            out.append("\n**⚠ Claims not traceable to any member:** " + "; ".join(unsup))
        verification = res.get("verification") or []
        if verification:
            out.append("\n**Fact-check:**\n")
            out.append("| Claim | Verdict | Note |")
            out.append("|---|---|---|")
            for v in verification:
                out.append(f"| {v.get('claim','')} | {v.get('verdict','')} | {v.get('note','')} |")
        out.append("")

    out.extend(composites_markdown(result.get("composites") or []))

    best = result.get("best_raw")
    if best:
        out.append(
            f"## Top-ranked member answer — {best.get('label')} ({best.get('model')})\n"
        )
        out.append("_Peer-voted as strongest; not fact-checked. Treat specific "
                   "claims (identifiers, numbers, dates) as unverified._\n")
        out.append((best.get("text") or "").strip())
        out.append("")

    return "\n".join(out)


def _docx_heading(doc: Any, text: str) -> None:
    from docx.shared import Pt, RGBColor

    h = doc.add_paragraph()
    r = h.add_run(text)
    r.bold = True
    r.font.name = "Garamond"
    r.font.size = Pt(14)
    r.font.color.rgb = RGBColor.from_string(NAVY)


def _inline_runs(paragraph: Any, text: str) -> None:
    """Add text to a docx paragraph, honouring **bold** spans."""
    for i, chunk in enumerate(text.split("**")):
        if not chunk:
            continue
        run = paragraph.add_run(chunk)
        run.font.name = "Garamond"
        if i % 2 == 1:  # odd chunks are inside ** **
            run.bold = True


def _add_markdown(doc: Any, md: str) -> None:
    """Render a light subset of Markdown (headings, bullets, bold, tables, and
    inline LaTeX -> Unicode) into a doc."""
    from docx.shared import Pt, RGBColor

    def is_sep(l: str) -> bool:
        return bool(re.match(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$", l))

    def cells(r: str) -> list[str]:
        return [c.strip() for c in r.strip().strip("|").split("|")]

    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = _strip_math(lines[i].rstrip())
        s = line.strip()
        if not s or (set(s) <= {"-", "*", "_"} and len(s) >= 3):
            i += 1
            continue
        # Markdown table: a |...| row followed by a |---| separator
        if s.startswith("|") and i + 1 < len(lines) and is_sep(lines[i + 1]):
            header = cells(line)
            i += 2
            rows = []
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                rows.append(cells(_strip_math(lines[i])))
                i += 1
            table = doc.add_table(rows=1, cols=len(header))
            table.style = "Light Grid Accent 1"
            for j, h in enumerate(header):
                run = table.rows[0].cells[j].paragraphs[0].add_run(h.replace("**", ""))
                run.bold = True
                run.font.name = "Garamond"
            for r in rows:
                rc = table.add_row().cells
                for j in range(len(header)):
                    _inline_runs(rc[j].paragraphs[0], r[j] if j < len(r) else "")
            continue
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            head = doc.add_paragraph()
            run = head.add_run(line.lstrip("# ").strip())
            run.bold = True
            run.font.name = "Garamond"
            run.font.size = Pt(15 - min(level, 3))
            run.font.color.rgb = RGBColor.from_string(NAVY)
            i += 1
            continue
        if line.lstrip().startswith(("- ", "* ", "+ ")):
            p = doc.add_paragraph(style="List Bullet")
            _inline_runs(p, line.lstrip()[2:])
            i += 1
            continue
        p = doc.add_paragraph()
        _inline_runs(p, line)
        i += 1


def to_docx(result: dict[str, Any], path: str) -> None:
    from docx import Document
    from docx.shared import Pt, RGBColor

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Garamond"
    style.font.size = Pt(11)

    title = doc.add_paragraph()
    r = title.add_run("Council Report")
    r.bold = True
    r.font.name = "Garamond"
    r.font.size = Pt(20)
    r.font.color.rgb = RGBColor.from_string(NAVY)

    sub = doc.add_paragraph()
    sr = sub.add_run(result.get("prompt", ""))
    sr.italic = True
    sr.font.name = "Garamond"
    sr.font.color.rgb = RGBColor.from_string(SLATE)

    # Members
    h = doc.add_paragraph()
    hr = h.add_run("Members")
    hr.bold = True
    hr.font.name = "Garamond"
    hr.font.size = Pt(14)
    hr.font.color.rgb = RGBColor.from_string(NAVY)
    for m in result.get("members", []):
        st = m.get("status", "ok")
        detail = f"FAILED — {m.get('error')}" if st == "failed" else st
        p = doc.add_paragraph(style="List Bullet")
        _inline_runs(p, f"**{m.get('label')}** ({m.get('model')}) — {detail}")

    # Survival table
    ledger = result.get("ledger", [])
    uniq = sum(1 for c in ledger if c.get("is_unique"))
    h = doc.add_paragraph()
    hr = h.add_run("Idea-survival (groupthink metric)")
    hr.bold = True
    hr.font.name = "Garamond"
    hr.font.size = Pt(14)
    hr.font.color.rgb = RGBColor.from_string(NAVY)
    doc.add_paragraph(
        f"Ledger: {len(ledger)} distinct ideas ({uniq} minority / {len(ledger)-uniq} shared)"
    )
    results = result.get("results", {})
    if results:
        table = doc.add_table(rows=1, cols=4)
        table.style = "Light Grid Accent 1"
        for j, head in enumerate(["Mode", "Minority kept", "Consensus kept", "Gap"]):
            table.rows[0].cells[j].paragraphs[0].add_run(head).bold = True
        for mode, res in results.items():
            s = res["survival"]
            gap = s["shared_rate"] - s["unique_rate"]
            cells = table.add_row().cells
            cells[0].text = mode
            cells[1].text = f"{_pct(s['unique_rate'])} ({s['unique_survived']}/{s['unique_total']})"
            cells[2].text = f"{_pct(s['shared_rate'])} ({s['shared_survived']}/{s['shared_total']})"
            cells[3].text = _pct(gap)

    # Cost
    cost = result.get("cost")
    if cost:
        partial = " (partial)" if cost.get("cost_partial") else ""
        _docx_heading(doc, "Cost")
        doc.add_paragraph(
            f"Total: ${cost.get('total_cost_usd', 0):.4f}{partial} · "
            f"{cost.get('total_tokens', 0):,} tokens"
        )
        ct = doc.add_table(rows=1, cols=4)
        ct.style = "Light Grid Accent 1"
        for j, head in enumerate(["Model", "Calls", "Tokens", "Cost"]):
            ct.rows[0].cells[j].paragraphs[0].add_run(head).bold = True
        for r in cost.get("per_model", []):
            cells = ct.add_row().cells
            cells[0].text = r["model"]
            cells[1].text = str(r["calls"])
            cells[2].text = f"{r['prompt_tokens'] + r['completion_tokens']:,}"
            cells[3].text = f"${r['cost_usd']:.4f}" if r.get("cost_usd") is not None else "n/a"

    # Reviewer assessment (human adjudication)
    review = result.get("review") or {}
    rated = _rated_ideas(result)
    if review.get("overall") or rated:
        _docx_heading(doc, "Reviewer assessment")
        if review.get("overall"):
            _add_markdown(doc, review["overall"].strip())
        if rated:
            rt = doc.add_table(rows=1, cols=3)
            rt.style = "Light Grid Accent 1"
            for j, head in enumerate(["Idea", "Verdict", "Comment"]):
                rt.rows[0].cells[j].paragraphs[0].add_run(head).bold = True
            for summary, verdict, a in rated:
                cells = rt.add_row().cells
                cells[0].text = summary
                cells[1].text = verdict
                cells[2].text = a.get("comment", "")

    # Syntheses
    from docx.shared import Pt as _Pt, RGBColor as _RGB

    for mode, res in results.items():
        _docx_heading(doc, f"Synthesis — {mode}")
        _add_markdown(doc, res.get("synthesis", "").strip() or "_(empty)_")
        unsup = res.get("unsupported_claims") or []
        if unsup:
            p = doc.add_paragraph()
            r = p.add_run("⚠ Claims not traceable to any member: " + "; ".join(unsup))
            r.font.name = "Garamond"
            r.font.color.rgb = _RGB.from_string(RED)
        verification = res.get("verification") or []
        if verification:
            fp = doc.add_paragraph()
            fr = fp.add_run("Fact-check")
            fr.bold = True
            fr.font.name = "Garamond"
            vt = doc.add_table(rows=1, cols=3)
            vt.style = "Light Grid Accent 1"
            for j, head in enumerate(["Claim", "Verdict", "Note"]):
                vt.rows[0].cells[j].paragraphs[0].add_run(head).bold = True
            for v in verification:
                cells = vt.add_row().cells
                cells[0].text = v.get("claim", "")
                cells[1].text = v.get("verdict", "")
                cells[2].text = v.get("note", "")

    composites_docx(doc, result.get("composites") or [])

    best = result.get("best_raw")
    if best:
        _docx_heading(doc, f"Top-ranked member answer — {best.get('label')} ({best.get('model')})")
        note = doc.add_paragraph()
        nr = note.add_run("Peer-voted as strongest; not fact-checked. Treat specific "
                          "claims (identifiers, numbers, dates) as unverified.")
        nr.italic = True
        nr.font.name = "Garamond"
        nr.font.color.rgb = _RGB.from_string(MUTED)
        _add_markdown(doc, (best.get("text") or "").strip())

    doc.save(path)
