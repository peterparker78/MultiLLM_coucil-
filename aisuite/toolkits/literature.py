"""Literature search toolkit — cited, scholarly evidence for grounding pipelines.

Free REST sources, no API keys, no MCP: NCBI E-utilities (PubMed) and Europe PMC.
Every function is defensive (timeouts, returns [] on failure) so a pipeline that
grounds in literature degrades gracefully if a source is unreachable.

Each result is normalized to:
    {source, id, title, authors, year, journal, abstract, doi, url}
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Optional

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_UA = {"User-Agent": "aisuite-council/0.1"}


def _get(url: str, timeout: float = 12.0) -> bytes:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def pubmed_search(query: str, limit: int = 8, email: Optional[str] = None) -> list[dict]:
    """Search PubMed via NCBI E-utilities (esearch -> efetch). `email` is a
    courtesy parameter NCBI requests for rate-limit purposes."""
    try:
        sp = {"db": "pubmed", "term": query, "retmax": str(limit),
              "retmode": "json", "tool": "aisuite-council"}
        if email:
            sp["email"] = email
        ids = json.loads(_get(f"{_EUTILS}/esearch.fcgi?{urllib.parse.urlencode(sp)}")) \
            .get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        fp = {"db": "pubmed", "id": ",".join(ids), "retmode": "xml", "tool": "aisuite-council"}
        if email:
            fp["email"] = email
        return _parse_pubmed_xml(_get(f"{_EUTILS}/efetch.fcgi?{urllib.parse.urlencode(fp)}"))
    except Exception:
        return []


def _parse_pubmed_xml(xml_bytes: bytes) -> list[dict]:
    out: list[dict] = []
    try:
        root = ET.fromstring(xml_bytes)
    except Exception:
        return out
    for art in root.findall(".//PubmedArticle"):
        mc = art.find(".//MedlineCitation")
        if mc is None:
            continue
        pmid = mc.findtext("PMID") or ""
        article = mc.find("Article")
        if article is None:
            continue
        title_el = article.find("ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""
        abs_parts = []
        for at in article.findall(".//Abstract/AbstractText"):
            label = at.get("Label")
            txt = "".join(at.itertext()).strip()
            abs_parts.append(f"{label}: {txt}" if label else txt)
        authors = []
        for a in article.findall(".//AuthorList/Author"):
            ln, init = a.findtext("LastName"), a.findtext("Initials")
            if ln:
                authors.append(f"{ln} {init}" if init else ln)
        authors_str = ", ".join(authors[:6]) + (" et al." if len(authors) > 6 else "")
        year = (article.findtext(".//Journal/JournalIssue/PubDate/Year")
                or (article.findtext(".//Journal/JournalIssue/PubDate/MedlineDate") or "")[:4])
        doi = ""
        for aid in art.findall(".//ArticleIdList/ArticleId"):
            if aid.get("IdType") == "doi":
                doi = (aid.text or "").lower()
        out.append({
            "source": "PubMed", "id": pmid, "title": title,
            "authors": authors_str, "year": str(year)[:4],
            "journal": article.findtext(".//Journal/Title") or "",
            "abstract": " ".join(abs_parts).strip(), "doi": doi,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
        })
    return out


def europepmc_search(query: str, limit: int = 8) -> list[dict]:
    """Search Europe PMC (REST, core result type includes abstracts)."""
    try:
        params = {"query": query, "format": "json", "pageSize": str(limit), "resultType": "core"}
        data = json.loads(_get(f"{_EPMC}?{urllib.parse.urlencode(params)}"))
        out = []
        for r in data.get("resultList", {}).get("result", []):
            doi = (r.get("doi") or "").lower()
            pmid = r.get("pmid") or ""
            url = (f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid
                   else f"https://doi.org/{doi}" if doi
                   else f"https://europepmc.org/article/{r.get('source', 'MED')}/{r.get('id', '')}")
            out.append({
                "source": "EuropePMC", "id": pmid or r.get("id", ""),
                "title": (r.get("title") or "").strip(), "authors": r.get("authorString", ""),
                "year": str(r.get("pubYear", "")), "journal": r.get("journalTitle", ""),
                "abstract": (r.get("abstractText") or "").strip(), "doi": doi, "url": url,
            })
        return out
    except Exception:
        return []


def gather_evidence(query: str, limit: int = 8, email: Optional[str] = None) -> list[dict]:
    """Query PubMed + Europe PMC, dedupe (by DOI, then PMID/id, then title), and
    return up to `limit` cited results. PubMed first (clinical authority)."""
    pool = pubmed_search(query, limit=limit, email=email) + europepmc_search(query, limit=limit)
    seen, merged = set(), []
    for r in pool:
        key = (r.get("doi") or r.get("id") or r.get("title", "")[:80]).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(r)
        if len(merged) >= limit:
            break
    return merged
