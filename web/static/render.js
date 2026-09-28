// Shared Markdown renderer for both the council and advisory pages.
// Handles headings, bold/italic/code, links, lists, blockquotes, hr, fenced
// code, GitHub-style tables, and converts inline LaTeX math ($...$) to plain
// Unicode so model output doesn't show raw $\Delta$ etc.

window.esc = (s) => (s == null ? "" : String(s)).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

const _TEX = {
  "\\Delta": "Δ", "\\delta": "δ", "\\ge": "≥", "\\geq": "≥", "\\le": "≤", "\\leq": "≤",
  "\\rightarrow": "→", "\\to": "→", "\\leftarrow": "←", "\\times": "×", "\\approx": "≈",
  "\\pm": "±", "\\neq": "≠", "\\alpha": "α", "\\beta": "β", "\\mu": "µ", "\\sigma": "σ",
  "\\sim": "~", "\\cdot": "·", "\\%": "%", "\\,": " ", "\\ ": " ",
};
function _tex(t) { for (const k in _TEX) t = t.split(k).join(_TEX[k]); return t.replace(/[{}]/g, ""); }
// $...$ and \( \) -> plain; single $ (currency) is left alone.
function _stripMath(s) {
  return s.replace(/\$([^$\n]+?)\$/g, (_, i) => _tex(i)).replace(/\\\(([^\n]+?)\\\)/g, (_, i) => _tex(i));
}

window.renderMarkdown = function (src) {
  if (!src) return "<p><em>(empty)</em></p>";
  src = _stripMath(src.replace(/\r/g, ""));
  const e = window.esc;
  const inline = (s) =>
    e(s)
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  const lines = src.split("\n");
  let html = "", i = 0, list = null;
  const close = () => { if (list) { html += `</${list}>`; list = null; } };
  const special = (l) => /^\s*(#{1,6}\s|[-*+]\s|\d+[.)]\s|>|```|---|\*\*\*|___|\|)/.test(l);
  const cells = (r) => r.trim().replace(/^\||\|$/g, "").split("|").map((c) => inline(c.trim()));
  const isSep = (l) => /^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$/.test(l);
  while (i < lines.length) {
    const ln = lines[i];
    if (/^\s*```/.test(ln)) { close(); i++; let c = ""; while (i < lines.length && !/^\s*```/.test(lines[i])) { c += lines[i] + "\n"; i++; } i++; html += `<pre class="code">${e(c)}</pre>`; continue; }
    if (/^\s*$/.test(ln)) { close(); i++; continue; }
    // table: a |...| row followed by a |---|--- separator
    if (/^\s*\|.*\|/.test(ln) && i + 1 < lines.length && isSep(lines[i + 1])) {
      close();
      const header = cells(ln); i += 2;
      const rows = [];
      while (i < lines.length && /\|/.test(lines[i]) && lines[i].trim()) { rows.push(cells(lines[i])); i++; }
      html += `<table class="md-table"><thead><tr>${header.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
      continue;
    }
    if (/^\s*(---|\*\*\*|___)\s*$/.test(ln)) { close(); html += "<hr>"; i++; continue; }
    let m = ln.match(/^(#{1,6})\s+(.*)$/);
    if (m) { close(); const l = m[1].length; html += `<h${l}>${inline(m[2])}</h${l}>`; i++; continue; }
    m = ln.match(/^\s*[-*+]\s+(.*)$/);
    if (m) { if (list !== "ul") { close(); html += "<ul>"; list = "ul"; } html += `<li>${inline(m[1])}</li>`; i++; continue; }
    m = ln.match(/^\s*\d+[.)]\s+(.*)$/);
    if (m) { if (list !== "ol") { close(); html += "<ol>"; list = "ol"; } html += `<li>${inline(m[1])}</li>`; i++; continue; }
    m = ln.match(/^\s*>\s?(.*)$/);
    if (m) { close(); html += `<blockquote>${inline(m[1])}</blockquote>`; i++; continue; }
    close(); let para = ln; i++;
    while (i < lines.length && !/^\s*$/.test(lines[i]) && !special(lines[i])) { para += " " + lines[i]; i++; }
    html += `<p>${inline(para)}</p>`;
  }
  close();
  return html;
};
