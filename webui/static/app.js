const form = document.getElementById("decode-form");
const statusEl = document.getElementById("status");
const resultsCard = document.getElementById("results-card");
const tableBody = document.querySelector("#results-table tbody");
const pathsEl = document.getElementById("paths");
const speciesSelect = document.getElementById("species");
const summaryEl = document.getElementById("results-summary");
const sequenceField = document.getElementById("sequence_text");
const demoButton = document.getElementById("load-demo");
const lenHeader = document.getElementById("len-header");
const modeRadios = document.querySelectorAll("input[name='mode']");
const minOrfGroup = document.getElementById("min_orf_group");

const DEMO_SEQUENCE =
  "ATA CCC ATG GCC AAC CTC CTA CTC CTC ATT GTA CCC ATT CTA ATC GCA ATG GCA TTC CTA ATG CTT ACC GAA CGA AAA ATT CTA GGC TAT ATA CAA CTA CGC AAA GGC CCC AAC GTT GTA GGC CCC TAC GGG CTA CTA CAA CCC TTC GCT GAC GCC ATA AAA CTC TTC ACC AAA GAG CCC CTA AAA CCC GCC ACA TCT ACC ATC ACC CTC TAC ATC ACC GCC CCG ACC TTA GCT CTC ACC ATC GCT CTT CTA CTA TGA ACC CCC CTC CCC ATA CCC AAC CCC CTG GTC AAC CTC AAC CTA GGC CTC CTA TTT ATT CTA GCC ACC TCT AGC CTA GCC GTT TAC TCA ATC CTC TGA TCA GGG TGA GCA TCA AAC TCA AAC TAC GCC CTG ATC GGC GCA CTG CGA GCA GTA GCC CAA ACA ATC TCA TAT GAA GTC ACC CTA GCC ATC ATT CTA CTA TCA ACA TTA CTA ATA AGT GGC TCC TTT AAC CTC TCC ACC CTT ATC ACA ACA CAA GAA CAC CTC TGA TTA CTC CTG CCA TCA TGA CCC TTG GCC ATA ATA TGA TTT ATC TCC ACA CTA GCA GAG ACC AAC CGA ACC CCC TTC GAC CTT GCC GAA GGG GAG TCC GAA CTA GTC TCA GGC TTC AAC ATC GAA TAC GCC GCA GGC CCC TTC GCC CTA TTC TTC ATA GCC GAA TAC ACA AAC ATT ATT ATA ATA AAC ACC CTC ACC ACT ACA ATC TTC CTA GGA ACA ACA TAT GAC GCA CTC TCC CCT GAA CTC TAC ACA ACA TAT TTT GTC ACC AAG ACC CTA CTT CTA ACC TCC CTG TTC TTA TGA ATT CGA ACA GCA TAC CCC CGA TTC CGC TAC GAC CAA CTC ATA CAC CTC CTA TGA AAA AAC TTC CTA CCA CTC ACC CTA GCA TTA CTT ATA TGA TAT GTC TCC ATA CCC ATT ACA ATC TCC AGC ATT CCC CCT CAA ACC";

if (demoButton && sequenceField) {
  demoButton.addEventListener("click", () => {
    sequenceField.value = DEMO_SEQUENCE;
    sequenceField.focus();
    setStatus("Demo sequence loaded. Adjust parameters or run decode.", "info");
  });
}

let CONFIG = null;
let CURRENT_MODE = "codon";

function populateSpecies(mode, defaults) {
  const opts = CONFIG?.modes?.[mode]?.species_options ?? [];
  if (!speciesSelect) return;
  speciesSelect.innerHTML = opts.map((opt) => `<option value="${opt.value}">${opt.label}</option>`).join("");
  const preferred = defaults?.species_id;
  if (preferred && opts.some((o) => o.value === preferred)) {
    speciesSelect.value = preferred;
  }
}

function setUnitsHeader(mode) {
  if (!lenHeader) return;
  if (mode === "nt1") {
    lenHeader.textContent = "Nucleotides";
  } else if (mode === "nt2") {
    lenHeader.textContent = "Dinucleotides";
  } else if (mode === "nt3") {
    lenHeader.textContent = "Trinucleotides";
  } else if (mode === "aa") {
    lenHeader.textContent = "Amino acids";
  } else {
    lenHeader.textContent = "Codons";
  }
}

function updateModeUI(mode) {
  if (minOrfGroup) {
    // AA mode also uses an ORF/nt threshold, similar to codon mode
    minOrfGroup.hidden = !(mode === "codon" || mode === "aa");
  }
}

// Initialize UI state based on current radio selection as early as possible
(() => {
  const initial = Array.from(modeRadios).find((r) => r.checked)?.value || CURRENT_MODE || "codon";
  updateModeUI(initial);
})();

async function fetchDefaults() {
  try {
    const res = await fetch("/api/config");
    if (!res.ok) return;
    const data = await res.json();
    CONFIG = data;
    const defaults = data?.defaults ?? {};

    CURRENT_MODE = defaults.mode || "codon";
    // Set initial radio state
    for (const r of modeRadios) {
      r.checked = r.value === CURRENT_MODE;
    }

    populateSpecies(CURRENT_MODE, defaults);
    setUnitsHeader(CURRENT_MODE);
    updateModeUI(CURRENT_MODE);

    if (defaults.min_orf_nt) {
      document.getElementById("min_orf_nt").value = defaults.min_orf_nt;
    }
    if (defaults.code) {
      document.getElementById("code").value = defaults.code;
    }
    if (defaults.model_json) {
      const opts = CONFIG?.modes?.[CURRENT_MODE]?.species_options ?? [];
      const speciesLabel = opts.find((opt) => opt.value === defaults.species_id)?.label ?? defaults.species_id;
      setStatus(`Ready to decode with ${speciesLabel}.`, "info");
    }
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
    const lengthVal = row.length ?? row.len_codons ?? row.len_nt ?? 0;
    tr.innerHTML = `
      <td>${row.id}</td>
      <td>${row.logprob.toFixed(3)}</td>
      <td class="${row.winner}">${row.winner}</td>
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
  pathsEl.innerHTML = paths
    .map(
      (p) => `
        <div>
          <strong>${p.id}</strong> [${p.context}]<br />
          <span>${p.states.join(" ")}</span>
        </div>
      `
    )
    .join("");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const submitBtn = form.querySelector("button[type='submit']");
  submitBtn.disabled = true;

  const fd = new FormData(form);
  fd.set("emit_path", form.emit_path.checked ? "true" : "false");
  // Ensure selected mode is sent
  const checkedMode = Array.from(modeRadios).find((r) => r.checked)?.value || CURRENT_MODE || "codon";
  fd.set("mode", checkedMode);
  CURRENT_MODE = checkedMode;
  setUnitsHeader(CURRENT_MODE);
  updateModeUI(CURRENT_MODE);

  setStatus("Submitting job…", "pending");
  resultsCard.hidden = true;

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

    renderTable(data.records);
    renderPaths(data.paths || []);
    renderSummary(data.records);
    resultsCard.hidden = false;
    const speciesLabel = speciesSelect?.selectedOptions?.[0]?.textContent?.trim() || speciesSelect?.value || "Job complete";
    const message = data.stdout || `Job complete (${speciesLabel})`;
    setStatus(message, "success");
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
  if (summaryEl) {
    summaryEl.hidden = true;
    summaryEl.textContent = "";
  }
  setStatus("Cleared input.", "info");
  const mode = Array.from(modeRadios).find((r) => r.checked)?.value || CURRENT_MODE || "codon";
  updateModeUI(mode);
});

fetchDefaults();

// Respond to mode changes by refreshing species/options header
for (const r of modeRadios) {
  r.addEventListener("change", () => {
    if (!CONFIG) return;
    CURRENT_MODE = r.value;
    populateSpecies(CURRENT_MODE, CONFIG.defaults);
    setUnitsHeader(CURRENT_MODE);
    updateModeUI(CURRENT_MODE);
  });
}
