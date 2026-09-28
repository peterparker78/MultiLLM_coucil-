// Shared provider/model picker for the council and advisory pages.
// A seat spec is "provider:model[:online]"; the picker splits it into a
// provider group (OpenRouter / Ollama cloud / Ollama local / custom) plus a
// model id typed or chosen from a datalist fed by /api/models.
// Depends on esc() from render.js.

const PROVIDERS = [
  ["openrouter", "OpenRouter"],
  ["ollama_cloud", "Ollama cloud"],
  ["ollama", "Ollama local"],
  ["anthropic", "Anthropic (direct)"],
  ["openai", "OpenAI (direct)"],
  ["lmstudio", "LM Studio"],
  ["moonshot", "Moonshot (Kimi)"],
  ["claudecode", "Claude (subscription)"],
  ["codex", "ChatGPT (subscription)"],
  ["byo", "Your endpoint (BYO)"],
  ["custom", "Custom"],
];
const LISTED = ["openrouter", "ollama_cloud", "ollama", "anthropic", "openai", "lmstudio", "moonshot", "claudecode", "codex", "byo"];
const PRETTY = { openrouter: "OpenRouter", ollama_cloud: "Ollama cloud", ollama: "Ollama local", anthropic: "Anthropic", openai: "OpenAI", lmstudio: "LM Studio", moonshot: "Moonshot", claudecode: "Claude subscription", codex: "ChatGPT subscription", byo: "Your endpoint" };
let catalogue = { errors: {}, notes: {} };
// Groups whose list is exact (what the account/daemon actually has) get a strict
// <select>; the others keep a searchable text box because their lists are catalogues.
const STRICT = ["codex", "lmstudio", "ollama"];
document.head.insertAdjacentHTML("beforeend", `<style>
  select.picker-model { background: var(--panel2); color: var(--txt); border: 1px solid var(--line); border-radius: 8px; padding: 9px 10px; font-size: 13px; font-family: var(--mono); flex: 1; min-width: 0; }
</style>`);

function providerOf(spec) {
  const s = (spec || "").trim();
  const online = /:online$/.test(s);
  const base = s.replace(/:online$/, "");
  const i = base.indexOf(":");
  const pfx = i < 0 ? "" : base.slice(0, i);
  const model = i < 0 ? base : base.slice(i + 1);
  if (pfx === "openrouter") return { provider: "openrouter", model, online };
  if (pfx === "ollama") return { provider: /cloud$/.test(model) ? "ollama_cloud" : "ollama", model, online: false };
  if (["anthropic", "openai", "lmstudio", "moonshot", "claudecode", "codex", "byo"].includes(pfx)) return { provider: pfx, model, online: false };
  return { provider: "custom", model: base, online: false };
}

function specOf(row) {
  const m = (row.model || "").trim();
  if (!m) return "";
  if (row.provider === "custom") return m;
  const pfx = row.provider === "ollama_cloud" ? "ollama" : row.provider;
  const base = pfx + ":" + m;
  return row.provider === "openrouter" && row.online ? base + ":online" : base;
}

function pickerHtml(cls, row) {
  const opts = PROVIDERS.map(([v, l]) => `<option value="${v}" ${row.provider === v ? "selected" : ""}>${l}</option>`).join("");
  let modelCtl;
  if (STRICT.includes(row.provider)) {
    const list = catalogue[row.provider] || [];
    const cur = (row.model || "").trim();
    const known = list.some((m) => m.id === cur);
    const why = (catalogue.notes || {})[row.provider] || (catalogue.errors || {})[row.provider] || "list not loaded";
    const head = !list.length
      ? `<option value="">(none available — ${esc(why)})</option>`
      : (cur && known ? "" : `<option value="" ${cur ? "" : "selected"}>choose…</option>`);
    const stray = cur && !known ? `<option value="${esc(cur)}" selected>${esc(cur)} (not available)</option>` : "";
    const options = list.map((m) => `<option value="${esc(m.id)}" ${m.id === cur ? "selected" : ""}>${esc(m.label)}</option>`).join("");
    modelCtl = `<select class="${cls}-model picker-model">${head}${stray}${options}</select>`;
  } else {
    const list = row.provider === "custom" ? "" : `list="dl-${row.provider}"`;
    const ph = row.provider === "custom" ? "provider:model" : "type to search…";
    modelCtl = `<input type="text" class="${cls}-model picker-model" ${list} value="${esc(row.model || "")}" placeholder="${ph}" autocomplete="off" />`;
  }
  return `<select class="${cls}-provider picker-provider">${opts}</select>` + modelCtl;
}

// "" if the row can run; otherwise why not (strict groups only — a model that
// is not on the account's/daemon's list would be rejected at run time).
function pickerProblem(row, what) {
  if (!STRICT.includes(row.provider)) return "";
  const list = catalogue[row.provider] || [];
  const m = (row.model || "").trim();
  if (!m) return `${what}: choose a ${PRETTY[row.provider]} model.`;
  if (!list.some((x) => x.id === m)) {
    const why = (catalogue.notes || {})[row.provider] || "it is not on the list of models you can use";
    return `${what}: "${m}" is not available on ${PRETTY[row.provider]} (${why}).`;
  }
  return "";
}

// Wire a rendered picker: `row` is the state object mutated in place;
// `onProviderChange` re-renders the host (the datalist must follow the provider).
function bindPicker(el, cls, row, onProviderChange) {
  el.querySelector(`.${cls}-provider`).onchange = (e) => { row.provider = e.target.value; row.model = ""; onProviderChange(); };
  el.querySelector(`.${cls}-model`).oninput = (e) => (row.model = e.target.value);
}

function renderDatalists() {
  for (const key of LISTED) {
    let dl = document.getElementById("dl-" + key);
    if (!dl) { dl = document.createElement("datalist"); dl.id = "dl-" + key; document.body.appendChild(dl); }
    dl.innerHTML = (catalogue[key] || []).map((m) => `<option value="${esc(m.id)}">${esc(m.label)}</option>`).join("");
  }
}

async function loadModels(statusEl) {
  if (statusEl) statusEl.textContent = "Loading model lists…";
  try {
    catalogue = await (await fetch("/api/models")).json();
    renderDatalists();
    const errs = catalogue.errors || {}, notes = catalogue.notes || {};
    const part = (k) => {
      const n = (catalogue[k] || []).length;
      if (errs[k] !== undefined) return `${PRETTY[k]}: unavailable`;
      if (notes[k] !== undefined) return /^set [A-Z_]+$/.test(notes[k]) ? `${PRETTY[k]} ${n} (needs ${notes[k].slice(4)} to run)` : `${PRETTY[k]}: ${notes[k]}`;
      return `${PRETTY[k]} ${n}`;
    };
    if (statusEl) statusEl.textContent = "Models · " + LISTED.map(part).join(" · ");
  } catch (e) {
    if (statusEl) statusEl.textContent = "Model lists unavailable — type provider:model under Custom.";
  }
}

// Briefing documents: upload several, keep {name, chars, text} per document.
function briefRowsHtml(briefs) {
  return briefs.map((b, i) => `<div class="brief-row"><span class="model-id" style="margin:0;flex:1">📄 ${esc(b.name)} · ${b.chars.toLocaleString()} chars</span><a class="lens-rm brief-rm" data-i="${i}">×</a></div>`).join("") +
    (briefs.length > 1 ? `<div class="note">${briefs.length} documents are combined under one shared budget and shown to every seat.</div>` : "");
}

async function uploadBriefs(files) {
  const fd = new FormData();
  [...files].forEach((f) => fd.append("files", f));
  const res = await fetch("/api/brief", { method: "POST", body: fd });
  if (!res.ok) throw new Error(`upload failed (${res.status})`);
  return res.json();
}
