const form = document.getElementById("decode-form");
const statusEl = document.getElementById("status");
const resultsCard = document.getElementById("results-card");
const tableBody = document.querySelector("#results-table tbody");
const pathsEl = document.getElementById("paths");
// Species selection removed (human only)
const summaryEl = document.getElementById("results-summary");
const sequenceField = document.getElementById("sequence_text");
const demoButton = document.getElementById("load-demo");
const demoFastaButton = document.getElementById("load-demo-fasta");
const fastaInput = document.getElementById("fasta_file");
const fastaLoadedBadge = document.getElementById("fasta-loaded-badge");
const lenHeader = document.getElementById("len-header");
const modeRadios = document.querySelectorAll("input[name='mode']");
// Min length and method selection removed
const downloadBtn = document.getElementById("download-results");
const downloadMenu = document.getElementById("download-menu");

const DEMO_SEQUENCE =
  "ATA CCC ATG GCC AAC CTC CTA CTC CTC ATT GTA CCC ATT CTA ATC GCA ATG GCA TTC CTA ATG CTT ACC GAA CGA AAA ATT CTA GGC TAT ATA CAA CTA CGC AAA GGC CCC AAC GTT GTA GGC CCC TAC GGG CTA CTA CAA CCC TTC GCT GAC GCC ATA AAA CTC TTC ACC AAA GAG CCC CTA AAA CCC GCC ACA TCT ACC ATC ACC CTC TAC ATC ACC GCC CCG ACC TTA GCT CTC ACC ATC GCT CTT CTA CTA TGA ACC CCC CTC CCC ATA CCC AAC CCC CTG GTC AAC CTC AAC CTA GGC CTC CTA TTT ATT CTA GCC ACC TCT AGC CTA GCC GTT TAC TCA ATC CTC TGA TCA GGG TGA GCA TCA AAC TCA AAC TAC GCC CTG ATC GGC GCA CTG CGA GCA GTA GCC CAA ACA ATC TCA TAT GAA GTC ACC CTA GCC ATC ATT CTA CTA TCA ACA TTA CTA ATA AGT GGC TCC TTT AAC CTC TCC ACC CTT ATC ACA ACA CAA GAA CAC CTC TGA TTA CTC CTG CCA TCA TGA CCC TTG GCC ATA ATA TGA TTT ATC TCC ACA CTA GCA GAG ACC AAC CGA ACC CCC TTC GAC CTT GCC GAA GGG GAG TCC GAA CTA GTC TCA GGC TTC AAC ATC GAA TAC GCC GCA GGC CCC TTC GCC CTA TTC TTC ATA GCC GAA TAC ACA AAC ATT ATT ATA ATA AAC ACC CTC ACC ACT ACA ATC TTC CTA GGA ACA ACA TAT GAC GCA CTC TCC CCT GAA CTC TAC ACA ACA TAT TTT GTC ACC AAG ACC CTA CTT CTA ACC TCC CTG TTC TTA TGA ATT CGA ACA GCA TAC CCC CGA TTC CGC TAC GAC CAA CTC ATA CAC CTC CTA TGA AAA AAC TTC CTA CCA CTC ACC CTA GCA TTA CTT ATA TGA TAT GTC TCC ATA CCC ATT ACA ATC TCC AGC ATT CCC CCT CAA ACC";

if (demoButton && sequenceField) {
  demoButton.addEventListener("click", () => {
    sequenceField.value = DEMO_SEQUENCE;
    sequenceField.focus();
    setStatus("Demo data loaded. Adjust parameters or run decode.", "info");
  });
}

let DEMO_FASTA_FILE = null;
if (demoFastaButton) {
  demoFastaButton.addEventListener("click", async () => {
    try {
      const res = await fetch("/static/Rickettsia_prowazekii_str_Madrid_E.fa");
      if (!res.ok) throw new Error("Failed to fetch demo FASTA");
      const blob = await res.blob();
      DEMO_FASTA_FILE = new File([blob], "Rickettsia_prowazekii_str_Madrid_E.fa", { type: "text/plain" });
      setStatus("Demo FASTA loaded. It will be attached on submit.", "info");
      if (fastaLoadedBadge) fastaLoadedBadge.hidden = false;
    } catch (e) {
      console.error(e);
      setStatus("Could not load demo FASTA.", "error");
    }
  });
}

if (fastaInput) {
  fastaInput.addEventListener("change", () => {
    if (fastaLoadedBadge) fastaLoadedBadge.hidden = !(fastaInput.files && fastaInput.files.length > 0);
  });
}

let CONFIG = null;
let CURRENT_MODE = "nt1";
let LAST_RESULT = null;

// Species selection removed

function setUnitsHeader(mode) {
  if (!lenHeader) return;
  if (mode === "nt1") {
    lenHeader.textContent = "Nucleotides";
  } else if (mode === "nt2") {
    lenHeader.textContent = "Dinucleotides";
  } else if (mode === "nt3") {
    lenHeader.textContent = "Trinucleotides";
  } else {
    lenHeader.textContent = "Nucleotides";
  }
}

function updateModeUI(_mode) {}

// Initialize UI state based on current radio selection as early as possible
(() => {
  const initial = Array.from(modeRadios).find((r) => r.checked)?.value || CURRENT_MODE || "nt1";
  updateModeUI(initial);
})();

async function fetchDefaults() {
  try {
    const res = await fetch("/api/config");
    if (!res.ok) return;
    const data = await res.json();
    CONFIG = data;
    const defaults = data?.defaults ?? {};

    CURRENT_MODE = defaults.mode || "nt1";
    // Set initial radio state
    for (const r of modeRadios) {
      r.checked = r.value === CURRENT_MODE;
    }

    setUnitsHeader(CURRENT_MODE);
    updateModeUI(CURRENT_MODE);

    // No initial status message
  } catch (err) {
    console.warn("Failed to load config", err);
  }
}

function setStatus(message, tone = "info") {
  statusEl.textContent = message;
  statusEl.dataset.tone = tone;
}

function renderTable(records = []) {
  tableBody.innerHTML = "";
  records.forEach((row) => {
    const tr = document.createElement("tr");
    const lengthVal = row.len_nt ?? row.length ?? 0;
    const callText = row.winner || row.call || "";
    const callClass = (callText.toLowerCase().includes("mito")) ? "mito" : (callText.toLowerCase().includes("nuclear") ? "nuclear" : "ambiguous");
    tr.innerHTML = `
      <td>${row.id}</td>
      <td><span class="call-badge ${callClass}">${callText}</span></td>
      <td>${(row.nuclear_frac * 100).toFixed(1)}%</td>
      <td>${(row.mito_frac * 100).toFixed(1)}%</td>
      <td>${lengthVal}</td>
    `;
    tableBody.appendChild(tr);
  });
}

function renderSummary(records = []) {
  if (!summaryEl) return;
  if (!records.length) {
    summaryEl.hidden = true;
    summaryEl.textContent = "";
    return;
  }

  const first = records[0];
  const winner = (first.winner || "").toLowerCase();
  let message = "";
  if (winner.includes("mito")) {
    message = "This sequence is predicted to be mitochondrial. See below for detailed information.";
  } else if (winner.includes("nuclear")) {
    message = "This sequence is predicted to be nuclear. See below for detailed information.";
  } else {
    message = "Classification inconclusive.";
  }
  summaryEl.innerHTML = message;
  summaryEl.hidden = false;
}

function renderPaths(paths = []) {
  if (!paths.length) {
    pathsEl.innerHTML = "";
    return;
  }
  const fr = document.createDocumentFragment();

  paths.forEach((p) => {
    const id = p.id || "sequence";
    const rawStates = Array.isArray(p.states) ? p.states : [];
    // Map 0 -> N, 1 -> M; leave others as-is
    const nmStates = rawStates.map((s) => (s === "1" ? "M" : s === "0" ? "N" : s));

    // Build run-length encoded segments for compact view
    const runs = [];
    for (let i = 0; i < nmStates.length; i++) {
      const sym = nmStates[i];
      if (i === 0 || sym !== nmStates[i - 1]) {
        runs.push({ sym, len: 1 });
      } else {
        runs[runs.length - 1].len += 1;
      }
    }
    const total = nmStates.length || 1;

    const block = document.createElement("div");
    block.className = "path-block";

    const header = document.createElement("div");
    header.className = "path-header";
    header.innerHTML = `
      <strong>${id}</strong>
      <span class="path-legend">
        <span class="legend-item n">N</span> Nuclear
        <span class="legend-item m">M</span> Mito
      </span>
    `;
    block.appendChild(header);

    const bar = document.createElement("div");
    bar.className = "path-bar";
    runs.forEach(({ sym, len }) => {
      const seg = document.createElement("div");
      seg.className = `segment ${sym === "M" ? "m" : "n"}`;
      seg.style.flex = String(len);
      seg.title = `${sym} × ${len}`;
      bar.appendChild(seg);
    });
    block.appendChild(bar);

    const runsText = document.createElement("div");
    runsText.className = "path-runs";
    runsText.textContent = runs.map((r) => `${r.sym}×${r.len}`).join(" • ");
    block.appendChild(runsText);

    fr.appendChild(block);
  });

  pathsEl.innerHTML = "";
  pathsEl.appendChild(fr);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const submitBtn = form.querySelector("button[type='submit']");
  submitBtn.disabled = true;

  const fd = new FormData(form);
  fd.set("emit_path", form.emit_path.checked ? "true" : "false");
  // Ensure selected mode is sent
  const checkedMode = Array.from(modeRadios).find((r) => r.checked)?.value || CURRENT_MODE || "nt1";
  fd.set("mode", checkedMode);
  CURRENT_MODE = checkedMode;
  setUnitsHeader(CURRENT_MODE);
  updateModeUI(CURRENT_MODE);

  // Attach demo FASTA if present and user didn’t pick a file
  if (DEMO_FASTA_FILE && fastaInput && (!fastaInput.files || fastaInput.files.length === 0)) {
    fd.set("fasta_file", DEMO_FASTA_FILE);
  }

  setStatus("Submitting job…", "pending");
  resultsCard.hidden = true;
  if (downloadBtn) downloadBtn.hidden = true;

  try {
    const res = await fetch("/api/run", { method: "POST", body: fd });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data?.error || "Job submission failed");
    }

    if (!data.records?.length) {
      setStatus("Finished without matches. No ORF passed the filters.", "warning");
      resultsCard.hidden = true;
      return;
    }

    LAST_RESULT = data;
    renderTable(data.records);
    renderPaths(data.paths || []);
    renderSummary(data.records);
    resultsCard.hidden = false;
    // Suppress raw decoder stdout; show concise status only
    setStatus("Results ready.", "success");
    if (downloadBtn) downloadBtn.hidden = false;
  } catch (error) {
    console.error(error);
    setStatus(error.message || "Submission failed. Check your inputs.", "error");
  } finally {
    submitBtn.disabled = false;
  }
});

form.addEventListener("reset", () => {
  tableBody.innerHTML = "";
  pathsEl.innerHTML = "";
  resultsCard.hidden = true;
  if (downloadBtn) downloadBtn.hidden = true;
  if (downloadMenu) downloadMenu.hidden = true;
  LAST_RESULT = null;
  DEMO_FASTA_FILE = null;
  if (fastaLoadedBadge) fastaLoadedBadge.hidden = true;
  if (summaryEl) {
    summaryEl.hidden = true;
    summaryEl.textContent = "";
  }
  setStatus("Cleared input.", "info");
  const mode = Array.from(modeRadios).find((r) => r.checked)?.value || CURRENT_MODE || "nt1";
  updateModeUI(mode);
});

fetchDefaults();

// Respond to mode changes by refreshing species/options header
for (const r of modeRadios) {
  r.addEventListener("change", () => {
    if (!CONFIG) return;
    CURRENT_MODE = r.value;
    setUnitsHeader(CURRENT_MODE);
    updateModeUI(CURRENT_MODE);
  });
}

// Build TSV/CSV strings
function buildTSV(data) {
  const rows = [];
  rows.push(["#id","loglik","nuclear_frac","mito_frac","call","seq_len_nt","n_tokens"].join("\t"));
  const recs = data?.records || [];
  for (const r of recs) {
    const loglik = (typeof r.loglik === "number") ? r.loglik : (typeof r.logprob === "number" ? r.logprob : 0);
    rows.push([
      r.id ?? "",
      (loglik ?? 0).toFixed(3),
      (r.nuclear_frac ?? 0).toString(),
      (r.mito_frac ?? 0).toString(),
      (r.winner ?? r.call ?? ""),
      (r.len_nt ?? r.length ?? 0).toString(),
      (r.tokens ?? 0).toString(),
    ].join("\t"));
  }
  const paths = data?.paths || [];
  for (const p of paths) {
    const states = (p.states || []).join(" ");
    rows.push(`#${p.id}_PATH\t${states}`);
  }
  return rows.join("\n") + "\n";
}

function csvEscape(val) {
  const s = String(val ?? "");
  if (/[",\n]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
  return s;
}

function buildCSV(data) {
  const rows = [];
  rows.push(["id","loglik","nuclear_frac","mito_frac","call","seq_len_nt","n_tokens"].join(","));
  const recs = data?.records || [];
  for (const r of recs) {
    const loglik = (typeof r.loglik === "number") ? r.loglik : (typeof r.logprob === "number" ? r.logprob : 0);
    rows.push([
      csvEscape(r.id ?? ""),
      csvEscape((loglik ?? 0).toFixed(3)),
      csvEscape(r.nuclear_frac ?? 0),
      csvEscape(r.mito_frac ?? 0),
      csvEscape(r.winner ?? r.call ?? ""),
      csvEscape(r.len_nt ?? r.length ?? 0),
      csvEscape(r.tokens ?? 0),
    ].join(","));
  }
  // PATH rows don’t translate cleanly to CSV header; append as comments at end
  const paths = data?.paths || [];
  for (const p of paths) {
    const states = (p.states || []).join(" ");
    rows.push(`#${p.id}_PATH\t${states}`);
  }
  return rows.join("\n") + "\n";
}

function triggerDownload(content, filename, mime) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

if (downloadBtn) {
  // Toggle menu
  downloadBtn.addEventListener("click", () => {
    if (!LAST_RESULT) return;
    if (downloadMenu) {
      downloadMenu.hidden = !downloadMenu.hidden;
    }
  });
}

if (downloadMenu) {
  downloadMenu.addEventListener("click", (e) => {
    const t = e.target;
    if (!(t instanceof HTMLElement)) return;
    const fmt = t.getAttribute("data-format");
    if (!fmt || !LAST_RESULT) return;
    if (fmt === "tsv") {
      const tsv = buildTSV(LAST_RESULT);
      triggerDownload(tsv, "mitoSpotter_results.tsv", "text/tab-separated-values;charset=utf-8");
    } else if (fmt === "csv") {
      const csv = buildCSV(LAST_RESULT);
      triggerDownload(csv, "mitoSpotter_results.csv", "text/csv;charset=utf-8");
    }
    downloadMenu.hidden = true;
  });
  // Hide menu when clicking outside
  document.addEventListener("click", (e) => {
    if (!downloadMenu || downloadMenu.hidden) return;
    const within = downloadMenu.contains(e.target) || downloadBtn.contains(e.target);
    if (!within) downloadMenu.hidden = true;
  });
}
