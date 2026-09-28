"""Offline tests for the literature toolkit and advisory grounding.
HTTP is mocked — no network. Run: PYTHONPATH=. python3 tests/test_literature.py
"""

from aisuite.toolkits import literature

_PUBMED_XML = b"""<?xml version="1.0"?>
<PubmedArticleSet>
 <PubmedArticle>
  <MedlineCitation>
   <PMID>12345678</PMID>
   <Article>
    <ArticleTitle>Blinatumomab versus chemotherapy in ALL</ArticleTitle>
    <Abstract><AbstractText Label="RESULTS">Median OS 7.7 vs 4.0 months.</AbstractText></Abstract>
    <AuthorList><Author><LastName>Kantarjian</LastName><Initials>H</Initials></Author></AuthorList>
    <Journal><Title>NEJM</Title><JournalIssue><PubDate><Year>2017</Year></PubDate></JournalIssue></Journal>
   </Article>
  </MedlineCitation>
  <PubmedData><ArticleIdList><ArticleId IdType="doi">10.1056/abc</ArticleId></ArticleIdList></PubmedData>
 </PubmedArticle>
</PubmedArticleSet>"""


def test_pubmed_xml_parsing():
    rows = literature._parse_pubmed_xml(_PUBMED_XML)
    assert len(rows) == 1
    r = rows[0]
    assert r["id"] == "12345678" and r["year"] == "2017" and r["doi"] == "10.1056/abc"
    assert "Median OS" in r["abstract"] and "Kantarjian" in r["authors"]
    assert r["url"].endswith("/12345678/")


def test_gather_evidence_dedupes(monkeypatch):
    pm = {"source": "PubMed", "id": "1", "title": "X", "doi": "10.1/x"}
    dup = {"source": "EuropePMC", "id": "1b", "title": "X-dup", "doi": "10.1/x"}  # same DOI
    other = {"source": "EuropePMC", "id": "2", "title": "Y", "doi": ""}
    monkeypatch.setattr(literature, "pubmed_search", lambda *a, **k: [dict(pm)])
    monkeypatch.setattr(literature, "europepmc_search", lambda *a, **k: [dict(dup), dict(other)])
    out = literature.gather_evidence("q", limit=8)
    assert len(out) == 2  # the duplicate DOI collapsed


def test_advisory_grounding_injects_evidence(monkeypatch):
    from advisory.advisory import run_advisory, to_dict
    from advisory.lenses import AdvisoryConfig
    from council.fake import DemoClient

    canned = [{"source": "PubMed", "id": "999", "title": "Relevant trial", "authors": "Smith J",
               "year": "2020", "journal": "Lancet", "abstract": "Key finding.", "doi": "", "url": "u"}]
    monkeypatch.setattr(literature, "gather_evidence", lambda *a, **k: canned)

    cfg = AdvisoryConfig()
    cfg.lenses = cfg.lenses[:2]
    cfg.ground = True
    res = run_advisory(DemoClient(), "Endpoint for trial X?", cfg)
    assert res.evidence == canned and res.evidence_query  # query extracted, evidence attached
    assert to_dict(res)["evidence"][0]["id"] == "999"


def _run(monkeypatch_stub=None):
    import types
    class MP:
        def __init__(self): self._undo = []
        def setattr(self, obj, name, val):
            self._undo.append((obj, name, getattr(obj, name)))
            setattr(obj, name, val)
        def undo(self):
            for o, n, v in reversed(self._undo): setattr(o, n, v)
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, fn in tests:
        mp = MP()
        try:
            fn(mp) if fn.__code__.co_argcount else fn()
        finally:
            mp.undo()
        print(f"  ok  {name}")
    print(f"\n{len(tests)} literature tests passed.")


if __name__ == "__main__":
    _run()
