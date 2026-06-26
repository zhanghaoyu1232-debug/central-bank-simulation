const FRONTEND_VERSION = "scenario-front-panel-v18";
const API_PORT = "8010";

const elements = {
  statusBox: document.getElementById("statusBox"),
  statusText: document.getElementById("statusText"),
  runButton: document.getElementById("runButton"),
  decreaseBanksButton: document.getElementById("decreaseBanksButton"),
  increaseBanksButton: document.getElementById("increaseBanksButton"),
  stepsInput: document.getElementById("stepsInput"),
  banksInput: document.getElementById("banksInput"),
  seedInput: document.getElementById("seedInput"),
  marketModeInput: document.getElementById("marketModeInput"),
  rolloverInput: document.getElementById("rolloverInput"),
  policySupportInput: document.getElementById("policySupportInput"),
  dangerZoneInput: document.getElementById("dangerZoneInput"),
  riskValue: document.getElementById("riskValue"),
  activeValue: document.getElementById("activeValue"),
  suspendedValue: document.getElementById("suspendedValue"),
  warningValue: document.getElementById("warningValue"),
  exposureValue: document.getElementById("exposureValue"),
  warningCallout: document.getElementById("warningCallout"),
  stepCount: document.getElementById("stepCount"),
  summaryList: document.getElementById("summaryList"),
  tableCount: document.getElementById("tableCount"),
  bankRows: document.getElementById("bankRows"),
  riskCanvas: document.getElementById("riskCanvas"),
  riskCallout: document.getElementById("riskCallout"),
};

const chartContext = elements.riskCanvas.getContext("2d");

function setStatus(message, state = "ready") {
  elements.statusText.textContent = message;
  elements.statusBox.classList.remove("running", "error");

  if (state === "running" || state === "error") {
    elements.statusBox.classList.add(state);
  }
}

function number(value, digits = 2) {
  const numericValue = Number(value);

  if (!Number.isFinite(numericValue)) {
    return "--";
  }

  return numericValue.toLocaleString(undefined, {
    maximumFractionDigits: digits,
  });
}

function percent(value) {
  const numericValue = Number(value);

  if (!Number.isFinite(numericValue)) {
    return "--";
  }

  return `${(numericValue * 100).toFixed(2)}%`;
}

function balanceSheetAmount(value) {
  const numericValue = Number(value);

  if (!Number.isFinite(numericValue)) {
    return "--";
  }

  return numericValue.toLocaleString(undefined, {
    maximumFractionDigits: 0,
  });
}

function exposureAmount(value) {
  const numericValue = Number(value);

  if (!Number.isFinite(numericValue)) {
    return "--";
  }

  if (numericValue <= 0) {
    return "—";
  }

  return balanceSheetAmount(numericValue);
}

function classForZone(zone) {
  const normalized = String(zone || "").toLowerCase();

  if (normalized === "normal" || normalized === "active") {
    return "good";
  }

  if (normalized === "warning") {
    return "warn";
  }

  if (normalized === "suspended" || normalized === "defaulted" || normalized === "inactive") {
    return "bad";
  }

  return "";
}

function pill(text, kind) {
  return `<span class="pill ${kind}">${text}</span>`;
}

function normalizeSimulationData(data) {
  const summary = { ...data.summary };
  summary.apiVersion = summary.apiVersion || data.apiVersion;
  const enrichedBanks = enrichBanksWithDangerSummary(data.banks || [], data.dangerSummary || []);
  const normalizedBanks = enrichedBanks;
  const requestedNumBanks = Number.isFinite(Number(summary.requestedNumBanks))
    ? Number(summary.requestedNumBanks)
    : Number(elements.banksInput.value || summary.numBanks || 30);
  const nonCentralBanks = Number.isFinite(Number(summary.nonCentralBanks))
    ? Number(summary.nonCentralBanks)
    : Math.max(1, normalizedBanks.filter((bank) => !isCentralBank(bank)).length || Number(summary.numBanks || 1) - 1);
  const normalizedSuspendedBanks = normalizedBanks.filter(
    (bank) => !isCentralBank(bank) && bank.dangerZone === "suspended"
  ).length;
  const normalizedWarningBanks = normalizedBanks.filter(
    (bank) => !isCentralBank(bank) && bank.dangerZone === "warning"
  ).length;
  const normalizedActiveBanks = normalizedBanks.filter(
    (bank) => !isCentralBank(bank) && bank.active
  ).length;
  const rawModelRisk = Number.isFinite(Number(summary.rawModelRisk))
    ? Number(summary.rawModelRisk)
    : Number(summary.systemicRisk || 0);
  const suspendedShare = Number.isFinite(Number(summary.suspendedShare))
    ? Number(summary.suspendedShare)
    : normalizedSuspendedBanks / nonCentralBanks;
  const warningShare = Number.isFinite(Number(summary.warningShare))
    ? Number(summary.warningShare)
    : normalizedWarningBanks / nonCentralBanks;
  const lowLcrShare = Number.isFinite(Number(summary.lowLcrShare)) ? Number(summary.lowLcrShare) : 0;
  const lowCarShare = Number.isFinite(Number(summary.lowCarShare)) ? Number(summary.lowCarShare) : 0;

  summary.requestedNumBanks = requestedNumBanks;
  summary.nonCentralBanks = nonCentralBanks;
  summary.activeBanks = Number.isFinite(Number(summary.activeBanks)) ? Number(summary.activeBanks) : normalizedActiveBanks;
  summary.suspendedBanks = Number.isFinite(Number(summary.suspendedBanks)) ? Number(summary.suspendedBanks) : normalizedSuspendedBanks;
  summary.warningBanks = Number.isFinite(Number(summary.warningBanks)) ? Number(summary.warningBanks) : normalizedWarningBanks;
  summary.rawModelRisk = rawModelRisk;
  summary.suspendedShare = suspendedShare;
  summary.warningShare = warningShare;
  summary.lowLcrShare = lowLcrShare;
  summary.lowCarShare = lowCarShare;
  summary.systemicRisk = Number.isFinite(Number(summary.systemicRisk))
    ? Math.min(1, Math.max(0, Number(summary.systemicRisk)))
    : Math.min(1, Math.max(0, rawModelRisk));
  summary.reconciledBanks = Number.isFinite(Number(summary.reconciledBanks))
    ? Number(summary.reconciledBanks)
    : 0;
  summary.dangerSummaryRows = Array.isArray(data.dangerSummary) ? data.dangerSummary.length : 0;

  const lcrStats = lcrStatsFromBanks(normalizedBanks);
  if (!Number.isFinite(Number(summary.avgLcr))) {
    summary.avgLcr = lcrStats.avgLcr;
  }
  if (!Number.isFinite(Number(summary.minLcr))) {
    summary.minLcr = lcrStats.minLcr;
  }

  const history = (data.history || []).map((point) => {
    const pointRawRisk = Number.isFinite(Number(point.rawModelRisk))
      ? Number(point.rawModelRisk)
      : Number(point.risk || 0);

    return {
      ...point,
      rawModelRisk: pointRawRisk,
      risk: Number.isFinite(Number(point.risk))
        ? Math.min(1, Math.max(0, Number(point.risk)))
        : Math.min(1, Math.max(0, pointRawRisk)),
    };
  });

  summary.peakWarningBanks = Number.isFinite(Number(summary.peakWarningBanks))
    ? Number(summary.peakWarningBanks)
    : Math.max(0, ...history.map((point) => Number(point.warningBanks || 0)), summary.warningBanks || 0);

  return {
    ...data,
    summary,
    history,
    banks: normalizedBanks,
  };
}

function enrichBanksWithDangerSummary(banks, dangerSummary) {
  const dangerByBank = new Map(
    dangerSummary.map((row) => [Number(row.bank_idx), row])
  );

  return banks.map((bank) => {
    const danger = dangerByBank.get(Number(bank.id));
    if (!danger) {
      return bank;
    }

    return {
      ...bank,
      suspendSteps: Number.isFinite(Number(bank.suspendSteps))
        ? bank.suspendSteps
        : Number(danger.suspend_steps || 0),
      dangerZone: bank.dangerZone || danger.zone || "normal",
      zoneReason: bank.zoneReason,
    };
  });
}

function isCentralBank(bank) {
  return bank.id === 0 || bank.type === "central" || bank.name === "CentralBank";
}

function lcrStatsFromBanks(banks) {
  const values = banks
    .filter((bank) => !isCentralBank(bank))
    .map((bank) => Number(bank.liquidityCoverageRatio))
    .filter((value) => Number.isFinite(value));

  if (!values.length) {
    return { avgLcr: null, minLcr: null };
  }

  return {
    avgLcr: values.reduce((sum, value) => sum + value, 0) / values.length,
    minLcr: Math.min(...values),
  };
}

function renderMetrics(summary) {
  elements.riskValue.textContent = percent(summary.systemicRisk);
  elements.activeValue.textContent = `${summary.activeBanks}/${summary.nonCentralBanks}`;
  elements.suspendedValue.textContent = number(summary.suspendedBanks, 0);
  elements.warningValue.textContent =
    summary.peakWarningBanks > summary.warningBanks
      ? `${number(summary.warningBanks, 0)} (peak ${number(summary.peakWarningBanks, 0)})`
      : number(summary.warningBanks, 0);
  elements.exposureValue.textContent = number(summary.totalExposure, 0);
  elements.stepCount.textContent = `${summary.completedSteps} step(s) completed`;
  elements.riskCallout.textContent =
    `${summary.modelLabel || "Selected model"} finished at ${percent(summary.systemicRisk)} ` +
    `after ${summary.completedSteps} step(s). Requested ${summary.requestedNumBanks} banks; ` +
    `server simulated ${summary.numBanks} banks. Raw model risk was ${percent(summary.rawModelRisk)}.`;
}

function renderWarningCallout(summary, banks) {
  const warningBanks = banks.filter((bank) => !isCentralBank(bank) && bank.dangerZone === "warning");

  if (!warningBanks.length) {
    if ((summary.peakWarningBanks || 0) > 0) {
      elements.warningCallout.className = "warning-callout muted";
      elements.warningCallout.innerHTML = `
        <div class="warning-callout-head">
          <strong>No banks currently in warning</strong>
          <span>Peak during run: ${number(summary.peakWarningBanks, 0)}</span>
        </div>
        <p>Warning states can be transient as banks recover or move into suspension.</p>
      `;
      return;
    }

    elements.warningCallout.className = "warning-callout hidden";
    elements.warningCallout.innerHTML = "";
    return;
  }

  elements.warningCallout.className = "warning-callout";
  elements.warningCallout.innerHTML = `
    <div class="warning-callout-head">
      <strong>${warningBanks.length} bank(s) in warning zone</strong>
      <span>Peak during run: ${number(summary.peakWarningBanks, 0)}</span>
    </div>
    <ul>
      ${warningBanks
        .map(
          (bank) => `
            <li>
              <span>${bank.name}</span>
              <span>${bank.zoneReason || "near policy threshold"}</span>
              <span>Assets ${balanceSheetAmount(bank.totalAssets)} / Liabilities ${balanceSheetAmount(bank.currentLiabilities)}</span>
            </li>
          `
        )
        .join("")}
    </ul>
  `;
}

function renderSummary(summary) {
  const rows = [
    ["Frontend version", FRONTEND_VERSION],
    ["API version", summary.apiVersion || "old server / not returned"],
    ["Model", summary.modelLabel || summary.modelKey || "--"],
    ["Rollover", summary.rolloverEnabled ? "Enabled" : "Disabled"],
    ["Policy support", summary.policySupportEnabled ? "Enabled" : "Disabled"],
    ["Danger-zone", summary.dangerZoneEnabled ? "Enabled" : "Disabled"],
    ["Project spread", percent(summary.projectSpread)],
    ["Project return default", percent(summary.projectReturnDefault)],
    ["Project return clip", `${percent(summary.projectReturnClipLow)} ~ ${percent(summary.projectReturnClipHigh)}`],
    ["Project maturity", summary.projectMaturitySteps || "--"],
    ["Interbank contract maturity", number(summary.interbankContractMaturity, 0)],
    ["Interbank role LCR cutoff", percent(summary.interbankRoleLcrCutoff)],
    ["Interbank intention LCR target", percent(summary.interbankIntentionLcrTarget)],
    ["Opportunity margin", percent(summary.opportunityMargin)],
    ["LCR repair target", percent(summary.lowLcrRepairTarget)],
    ["Liquidity rebuild target", percent(summary.liquidityRebuildTarget)],
    ["Market", summary.marketEnvironment],
    ["Requested banks", summary.requestedNumBanks],
    ["Total banks", summary.numBanks],
    ["Non-central banks", summary.nonCentralBanks],
    ["Base rate", percent(summary.baseRate)],
    ["Raw model risk", percent(summary.rawModelRisk)],
    ["Suspended share", percent(summary.suspendedShare)],
    ["Warning share", percent(summary.warningShare)],
    ["Average LCR", percent(summary.avgLcr)],
    ["Minimum LCR", percent(summary.minLcr)],
    ["Low LCR share", percent(summary.lowLcrShare)],
    ["Low CAR share", percent(summary.lowCarShare)],
    ["Warning banks", number(summary.warningBanks, 0)],
    ["Peak warning banks", number(summary.peakWarningBanks, 0)],
    ["Reconciled banks", number(summary.reconciledBanks, 0)],
    ["Danger summary rows", number(summary.dangerSummaryRows, 0)],
    ["Exposure edges", number(summary.edgeCount, 0)],
    ["Peak exposure edges", number(summary.peakExposureEdges, 0)],
    ["Max exposure", number(summary.maxExposure, 0)],
    ["Peak total exposure", number(summary.peakTotalExposure, 0)],
    ["Network stable step", summary.networkStableStep ?? "Not stable"],
    ["All default step", summary.allDefaultStep ?? "No"],
    ["Total restructures", number(summary.totalRestructures, 0)],
    ["Seed", summary.seed],
  ];

  elements.summaryList.innerHTML = rows
    .map(([label, value]) => `
      <div class="summary-row">
        <span>${label}</span>
        <strong>${value}</strong>
      </div>
    `)
    .join("");
}

function drawChart(history) {
  const canvas = elements.riskCanvas;
  const ctx = chartContext;
  const width = canvas.width;
  const height = canvas.height;
  const padding = 38;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "#d9e2ef";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding, padding);
  ctx.lineTo(padding, height - padding);
  ctx.lineTo(width - padding, height - padding);
  ctx.stroke();

  ctx.fillStyle = "#64748b";
  ctx.font = "12px Arial";
  ctx.fillText("Risk", 10, padding);
  ctx.fillText("Step", width - padding - 8, height - 12);

  if (!history || history.length === 0) {
    ctx.fillText("No simulation data yet.", padding + 12, padding + 34);
    return;
  }

  const values = history.map((point) => Number(point.risk) || 0);
  const max = Math.max(0.01, ...values);
  const min = Math.min(0, ...values);
  const range = Math.max(max - min, 0.01);
  const plotWidth = width - padding * 2;
  const plotHeight = height - padding * 2;

  ctx.strokeStyle = "#2563eb";
  ctx.lineWidth = 3;
  ctx.beginPath();

  history.forEach((point, index) => {
    const x = padding + (history.length === 1 ? 0 : (index / (history.length - 1)) * plotWidth);
    const y = height - padding - ((Number(point.risk) - min) / range) * plotHeight;

    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });

  ctx.stroke();

  const last = history[history.length - 1];
  const lastX = padding + plotWidth;
  const lastY = height - padding - ((Number(last.risk) - min) / range) * plotHeight;

  ctx.fillStyle = "#2563eb";
  ctx.beginPath();
  ctx.arc(lastX, lastY, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillText(percent(last.risk), Math.max(padding, lastX - 75), Math.max(18, lastY - 12));
}

function renderBanks(banks) {
  elements.tableCount.textContent = `${banks.length} bank(s)`;

  if (!banks.length) {
    elements.bankRows.innerHTML = `<tr><td colspan="13" class="empty">No bank data returned.</td></tr>`;
    return;
  }

  elements.bankRows.innerHTML = banks
    .map((bank) => {
      const status = bank.active ? "active" : "inactive";
      const zone = bank.dangerZone || "normal";
      const zoneReason = bank.canExitSuspended
        ? "can exit now"
        : bank.zoneReason || "--";
      const rowClass = zone === "warning" ? "row-warning" : zone === "suspended" ? "row-suspended" : "";

      return `
        <tr class="${rowClass}">
          <td>${bank.name}</td>
          <td>${bank.type}</td>
          <td>${pill(status, classForZone(status))}</td>
          <td>${pill(zone, classForZone(zone))}</td>
          <td>${zoneReason}</td>
          <td>${Number.isFinite(Number(bank.suspendSteps)) ? number(bank.suspendSteps, 0) : "--"}</td>
          <td>${balanceSheetAmount(bank.totalAssets)}</td>
          <td>${balanceSheetAmount(bank.liquidAssets)}</td>
          <td>${balanceSheetAmount(bank.currentLiabilities)}</td>
          <td>${percent(bank.liquidityCoverageRatio)}</td>
          <td>${percent(bank.capitalAdequacyRatio)}</td>
          <td>${exposureAmount(bank.interbankAssets)}</td>
          <td>${exposureAmount(bank.interbankLiabilities)}</td>
        </tr>
      `;
    })
    .join("");
}

function buildRequest() {
  const steps = clampInput(elements.stepsInput, 1, 500, 100);
  const banks = clampInput(elements.banksInput, 5, 100, 30);
  const marketMode = elements.marketModeInput.value;
  const dangerZoneEnabled = marketMode === "decentralized" && elements.dangerZoneInput.checked;
  const body = {
    steps,
    numBanks: banks,
    seed: elements.seedInput.value || "",
    marketMode,
    rolloverEnabled: elements.rolloverInput.checked,
    policySupportEnabled: elements.policySupportInput.checked,
    dangerZoneEnabled,
    cacheBust: String(Date.now()),
  };
  const params = new URLSearchParams(body);
  const apiBase =
    window.location.port === API_PORT
      ? ""
      : `http://127.0.0.1:${API_PORT}`;

  return {
    url: `${apiBase}/api/run-simulation-v2`,
    getUrl: `${apiBase}/api/run-simulation-v2?${params.toString()}`,
    body,
  };
}

function clampInput(input, min, max, fallback) {
  const value = Number(input.value);
  const safeValue = Number.isFinite(value) ? value : fallback;
  const clampedValue = Math.max(min, Math.min(max, Math.round(safeValue)));
  input.value = clampedValue;
  return String(clampedValue);
}

function changeBankCount(delta) {
  const currentValue = Number(elements.banksInput.value) || 30;
  elements.banksInput.value = currentValue + delta;
  clampInput(elements.banksInput, 5, 100, 30);
}

function sanitizeBankInput() {
  elements.banksInput.value = elements.banksInput.value.replace(/\D/g, "");
}

function updateFeatureControls() {
  const centralized = elements.marketModeInput.value === "centralized";
  elements.dangerZoneInput.disabled = centralized;
  if (centralized) {
    elements.dangerZoneInput.checked = false;
  } else if (!elements.dangerZoneInput.dataset.touched) {
    elements.dangerZoneInput.checked = true;
  }
}

async function fetchSimulationData(request) {
  const postResponse = await fetch(request.url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Cache-Control": "no-store",
    },
    body: JSON.stringify(request.body),
    cache: "no-store",
  });
  const postText = await postResponse.text();

  if (!looksLikeHtml(postText)) {
    return parseSimulationResponse(postResponse, postText, `POST ${request.url}`);
  }

  setStatus("POST returned HTML; retrying with GET...", "running");
  const getResponse = await fetch(request.getUrl, {
    method: "GET",
    headers: {
      "Cache-Control": "no-store",
    },
    cache: "no-store",
  });
  const getText = await getResponse.text();
  return parseSimulationResponse(getResponse, getText, `GET ${request.getUrl}`);
}

function looksLikeHtml(text) {
  const trimmed = text.trim().toLowerCase();
  return trimmed.startsWith("<!doctype") || trimmed.startsWith("<html");
}

function parseSimulationResponse(response, responseText, label) {
  const contentType = response.headers.get("Content-Type") || "";

  if (looksLikeHtml(responseText)) {
    throw new Error(
      `${label} returned HTML instead of JSON. ` +
      `Open http://127.0.0.1:${API_PORT}/api/health and verify apiVersion/supportsPost.`
    );
  }

  let data;
  try {
    data = JSON.parse(responseText);
  } catch (parseError) {
    throw new Error(
      `${label} returned non-JSON content (${contentType}): ` +
      responseText.slice(0, 160)
    );
  }

  if (!response.ok || data.error) {
    const detail = data.traceback ? ` ${data.traceback}` : "";
    throw new Error(`${data.error || "Simulation failed."}${detail}`);
  }

  return data;
}

async function runSimulation() {
  const bankCount = clampInput(elements.banksInput, 5, 100, 30);
  const stepCount = clampInput(elements.stepsInput, 1, 500, 100);
  const scenarioText = [
    elements.marketModeInput.value === "centralized" ? "centralized" : "decentralized",
    elements.rolloverInput.checked ? "with rollover" : "without rollover",
    elements.policySupportInput.checked ? "with policy support" : "without policy support",
    elements.dangerZoneInput.checked && !elements.dangerZoneInput.disabled ? "with danger-zone" : "without danger-zone",
  ].join(", ");
  setStatus("Running...", "running");
  elements.riskCallout.textContent =
    `Running ${scenarioText} for ${bankCount} banks and ${stepCount} step(s)...`;
  elements.runButton.disabled = true;

  try {
    const request = buildRequest();
    const data = await fetchSimulationData(request);
    if (!data.apiVersion && !data.summary?.apiVersion) {
      throw new Error("The simulation API is still returning the old format. Restart the Python server and check /api/health.");
    }

    const normalizedData = normalizeSimulationData(data);
    renderMetrics(normalizedData.summary);
    renderSummary(normalizedData.summary);
    renderWarningCallout(normalizedData.summary, normalizedData.banks);
    drawChart(normalizedData.history);
    renderBanks(normalizedData.banks);
    setStatus("Completed");
  } catch (error) {
    setStatus(`${error.message}. Start Python server first: python server.py`, "error");
  } finally {
    elements.runButton.disabled = false;
  }
}

drawChart([]);
updateFeatureControls();
elements.runButton.addEventListener("click", runSimulation);
elements.decreaseBanksButton.addEventListener("click", () => changeBankCount(-1));
elements.increaseBanksButton.addEventListener("click", () => changeBankCount(1));
elements.banksInput.addEventListener("input", sanitizeBankInput);
elements.banksInput.addEventListener("blur", () => clampInput(elements.banksInput, 5, 100, 30));
elements.marketModeInput.addEventListener("change", updateFeatureControls);
elements.dangerZoneInput.addEventListener("change", () => {
  elements.dangerZoneInput.dataset.touched = "true";
});
