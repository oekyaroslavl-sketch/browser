"use strict";

// Dual-transport bridge:
// - Tauri 2 native window (`window.__TAURI__`): IPC `invoke` + `event.listen("solve-progress")`.
// - HTTP/SSE preview server (`plo_desktop_preview` on `0.0.0.0`): relative `api/cmd` + `api/events`.

const hasTauri = typeof window !== "undefined" && (window.__TAURI__ || window.__TAURI_INTERNALS__);

async function invokeCmd(cmd, args = {}) {
  if (hasTauri && window.__TAURI__ && window.__TAURI__.core) {
    return await window.__TAURI__.core.invoke(cmd, { args });
  }
  const resp = await fetch("api/cmd", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cmd, args }),
  });
  const payload = await resp.json();
  if (!payload.ok) {
    throw new Error(payload.error || "Command failed");
  }
  return payload.data;
}

function subscribeSolveProgress(onEvent) {
  if (hasTauri && window.__TAURI__ && window.__TAURI__.event) {
    window.__TAURI__.event.listen("solve-progress", (ev) => onEvent(ev.payload));
    return;
  }
  if (typeof EventSource !== "undefined") {
    const es = new EventSource("api/events");
    es.addEventListener("solve-progress", (e) => {
      try {
        onEvent(JSON.parse(e.data));
      } catch (_) {}
    });
  }
}

const el = (id) => document.getElementById(id);

const RANKS_DESC = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"];
const SUITS_ORDER = ["s", "h", "d", "c"];
const SUIT_GLYPH = { s: "♠", h: "♥", d: "♦", c: "♣" };

function parseCardTokens(raw) {
  const cleaned = String(raw || "").replace(/[\s,-]/g, "");
  const out = [];
  const seen = new Set();
  for (let i = 0; i + 1 < cleaned.length; i += 2) {
    const r = cleaned[i].toUpperCase();
    const s = cleaned[i + 1].toLowerCase();
    const key = `${r}${s}`;
    if ("23456789TJQKA".includes(r) && "shdc".includes(s) && !seen.has(key)) {
      seen.add(key);
      out.push({ rank: r, suit: s, str: key });
    }
  }
  return out;
}

function renderCardBadges(cards, size = "lg", minSlots = 0) {
  const cls = size === "xs" ? "pcard pcard-xs" : size === "sm" ? "pcard pcard-sm" : "pcard";
  const html = cards.map(
    (c) =>
      `<span class="${cls} suit-${c.suit}"><span class="rk">${c.rank}</span><span class="st">${SUIT_GLYPH[c.suit] || c.suit}</span></span>`
  );
  for (let i = cards.length; i < minSlots; i++) {
    html.push(`<span class="card-slot-empty"></span>`);
  }
  return html.join("");
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// GTO+ Action Palette Classifier (Fold = Blue, Check/Call = Green, 1st Bet = Orange, Larger Bet/Raise = Red, All-In = Purple)
function gtoActionTheme(actionLabel, actionIdx = 0, totalActions = 2) {
  const low = String(actionLabel || "").toLowerCase();
  if (low.startsWith("fold")) {
    return { bg: "bg-fold", tile: "tile-fold", hex: "#4a78b5" };
  }
  if (low.startsWith("check") || low.startsWith("call")) {
    return { bg: "bg-check", tile: "tile-check", hex: "#3b9e66" };
  }
  if (low.startsWith("allin") || low.includes("all-in")) {
    return { bg: "bg-allin", tile: "tile-allin", hex: "#8b3fb8" };
  }
  if (low.startsWith("raise") || (actionIdx >= 2 && totalActions >= 3)) {
    return { bg: "bg-bet2", tile: "tile-bet2", hex: "#c93b3b" };
  }
  return { bg: "bg-bet1", tile: "tile-bet1", hex: "#d97a2b" };
}

function renderGtoStackedBarHtml(items) {
  // items: [{ action, probability }]
  return items
    .map((it, idx) => {
      const pct = Math.max(0, Math.min(100, (it.probability || 0) * 100));
      if (pct < 0.05) return "";
      const th = gtoActionTheme(it.action, idx, items.length);
      const text = pct >= 12 ? `${escapeHtml(it.action)} ${pct.toFixed(0)}%` : pct >= 6 ? `${pct.toFixed(0)}%` : "";
      return `<div class="gto-seg ${th.bg}" style="width:${pct.toFixed(2)}%" title="${escapeHtml(it.action)}: ${pct.toFixed(1)}%">${text}</div>`;
    })
    .join("");
}

function currentHeroSeat() {
  const checked = document.querySelector('input[name="hero-seat"]:checked');
  return checked ? checked.value : "IP";
}

function setHeroSeat(val) {
  const r = document.querySelector(`input[name="hero-seat"][value="${val}"]`);
  if (r) r.checked = true;
}

function parseBudgetTextToMib(rawStr) {
  const s = String(rawStr || "").trim().toLowerCase();
  if (!s) return 8192;
  let mult = 1;
  let numStr = s;
  if (s.endsWith("gib") || s.endsWith("gb") || s.endsWith("g")) {
    mult = 1024;
    numStr = s.replace(/(gib|gb|g)$/, "").trim();
  } else if (s.endsWith("mib") || s.endsWith("mb") || s.endsWith("m")) {
    mult = 1;
    numStr = s.replace(/(mib|mb|m)$/, "").trim();
  }
  const val = parseFloat(numStr);
  if (!Number.isFinite(val) || val <= 0) return 8192;
  return Math.min(1048576, Math.max(16, Math.round(val * mult)));
}

function formatMibHuman(mib) {
  const m = Number(mib) || 0;
  if (m >= 1024) {
    const g = m / 1024;
    return Math.abs(g - Math.round(g)) < 0.05 ? `${Math.round(g)} GiB` : `${g.toFixed(1)} GiB`;
  }
  return m >= 10 ? `${Math.round(m)} MiB` : `${m.toFixed(1)} MiB`;
}

function syncBudgetInputs(mib) {
  const human = mib >= 1024 && mib % 1024 === 0 ? `${mib / 1024} GiB` : `${mib} MiB`;
  el("inp-budget").value = human;
  if (el("inp-settings-budget")) {
    el("inp-settings-budget").value = String(mib);
  }
  el("st-ram").textContent = `Est. — / Budget ${formatMibHuman(mib)}`;
}

let hasSolvedNode = false;
let lastRenderedNode = null;
let userManuallyPickedSpot = false;
let activeSizeTab = "river";
let activeSolveScope = "All"; // "All" | "FlopOnly" | "TurnOnly" | "RiverOnly"
let activeRightView = "classes"; // "classes" | "combos" | "blockers" | "eqgraph"
let activeLockMode = "uniform"; // "uniform" | "classes"
let savedSpotsCache = [];
let drillModeActive = false;
let drillHandsPlayed = 0;
let drillTotalEvLoss = 0.0;
let botModeActive = false;
let liveConvPoints = [];
let lastAutoBoardStreetCount = 5;
let packLoadedState = {
  loaded: false,
  default_spot_oop: "RFI_BTN",
  default_spot_ip: "BB_DEF_VS_1_RAISER",
  last_spot_id: null,
};

function setRightView(view) {
  const valid = ["classes", "combos", "blockers", "eqgraph"];
  activeRightView = valid.includes(view) ? view : "classes";
  if (el("tab-view-classes")) el("tab-view-classes").classList.toggle("active", activeRightView === "classes");
  if (el("tab-view-combos")) el("tab-view-combos").classList.toggle("active", activeRightView === "combos");
  if (el("tab-view-blockers")) el("tab-view-blockers").classList.toggle("active", activeRightView === "blockers");
  if (el("tab-view-eqgraph")) el("tab-view-eqgraph").classList.toggle("active", activeRightView === "eqgraph");

  if (el("villain-classes-container")) {
    el("villain-classes-container").classList.toggle("hidden", activeRightView !== "classes");
  }
  if (el("detail-table-container")) {
    el("detail-table-container").classList.toggle("hidden", activeRightView !== "combos");
  }
  if (el("villain-blockers-container")) {
    el("villain-blockers-container").classList.toggle("hidden", activeRightView !== "blockers");
  }
  if (el("villain-eqgraph-container")) {
    el("villain-eqgraph-container").classList.toggle("hidden", activeRightView !== "eqgraph");
  }
}

function setSolveScope(scope) {
  const valid = ["All", "FlopOnly", "TurnOnly", "RiverOnly", "FlopTurn"];
  activeSolveScope = valid.includes(scope) ? scope : "All";
  for (const btn of document.querySelectorAll(".scope-btn")) {
    btn.classList.toggle("active", btn.dataset.scope === activeSolveScope);
  }
  if (activeSolveScope === "FlopOnly") setActiveSizeTab("flop");
  else if (activeSolveScope === "TurnOnly") setActiveSizeTab("turn");
  else if (activeSolveScope === "RiverOnly") setActiveSizeTab("river");
  updateSprAndLocalCards();
}

function setActiveSizeTab(streetKey) {
  if (!["flop", "turn", "river"].includes(streetKey)) return;
  activeSizeTab = streetKey;
  for (const s of ["flop", "turn", "river"]) {
    const tab = el(`tab-sz-${s}`);
    const panel = el(`panel-sz-${s}`);
    if (tab) tab.classList.toggle("active", s === streetKey);
    if (panel) panel.classList.toggle("hidden", s !== streetKey);
  }
}

function summarizeStreetBets(oopStr, ipStr) {
  const clean = (s) => (s || "").replace(/\s+/g, "");
  const o = clean(oopStr);
  const i = clean(ipStr);
  if (o === i) return o || "—";
  return `${o || "—"} / ${i || "—"}`;
}

function updateStreetSizingOverview(boardLen) {
  const fStr = summarizeStreetBets(el("inp-flop-oop-bets").value, el("inp-flop-ip-bets").value);
  const tStr = summarizeStreetBets(el("inp-turn-oop-bets").value, el("inp-turn-ip-bets").value);
  const rStr = summarizeStreetBets(el("inp-river-oop-bets").value, el("inp-river-ip-bets").value);

  // Determine which streets actually have active betting under (boardLen, activeSolveScope)
  let flopActive = boardLen === 3;
  let turnActive = boardLen <= 4;
  let riverActive = true;
  if (activeSolveScope === "FlopOnly") {
    flopActive = true;
    turnActive = false;
    riverActive = false;
  } else if (activeSolveScope === "TurnOnly") {
    flopActive = false;
    turnActive = true;
    riverActive = false;
  } else if (activeSolveScope === "RiverOnly") {
    flopActive = false;
    turnActive = false;
    riverActive = true;
  } else if (activeSolveScope === "FlopTurn") {
    flopActive = true;
    turnActive = true;
    riverActive = false;
  }

  const ov = el("lbl-streets-overview");
  if (ov) {
    ov.innerHTML = `
      <span class="${flopActive ? "" : "locked-street"}"><b>F:</b> ${escapeHtml(flopActive ? fStr : "check")}</span>
      <span class="${turnActive ? "" : "locked-street"}"><b>T:</b> ${escapeHtml(turnActive ? tStr : "check")}</span>
      <span class="${riverActive ? "" : "locked-street"}"><b>R:</b> ${escapeHtml(riverActive ? rStr : "check")}</span>
    `;
  }
  if (el("tab-sz-flop")) el("tab-sz-flop").classList.toggle("inactive-street", !flopActive);
  if (el("tab-sz-turn")) el("tab-sz-turn").classList.toggle("inactive-street", !turnActive);
  if (el("tab-sz-river")) el("tab-sz-river").classList.toggle("inactive-street", !riverActive);

  // Update scope hint & mark buttons that require more board cards
  for (const btn of document.querySelectorAll(".scope-btn")) {
    const sc = btn.dataset.scope;
    const needsMore =
      (sc === "TurnOnly" && boardLen < 4) || (sc === "RiverOnly" && boardLen < 5);
    btn.classList.toggle("needs-cards", needsMore);
  }
  const hintEl = el("lbl-scope-hint");
  if (hintEl) {
    if (activeSolveScope === "FlopOnly") {
      hintEl.textContent = boardLen > 3 ? "Uses 3 flop cards · T/R check" : "Flop bets · T/R check";
    } else if (activeSolveScope === "TurnOnly") {
      hintEl.textContent = boardLen < 4 ? "Pick ≥4 board cards!" : "Turn bets · R checks";
    } else if (activeSolveScope === "RiverOnly") {
      hintEl.textContent = boardLen < 5 ? "Pick 5 board cards!" : "River bets only";
    } else {
      hintEl.textContent =
        boardLen === 3 ? "Flop + Turn + River" : boardLen === 4 ? "Turn + River" : "River";
    }
  }
}

function updateSprAndLocalCards() {
  const pot = parseFloat(el("inp-pot").value) || 0;
  const stack = parseFloat(el("inp-stack").value) || 0;
  el("lbl-spr").textContent = pot > 0 ? (stack / pot).toFixed(2) : "—";

  const bCards = parseCardTokens(el("inp-board").value);
  const stLabel =
    bCards.length === 3 ? "FLOP" : bCards.length === 4 ? "TURN" : bCards.length === 5 ? "RIVER" : `${bCards.length} CARDS`;
  el("lbl-board-street").textContent = stLabel;
  el("left-board-slots").innerHTML = renderCardBadges(bCards, "sm", 5);

  if (bCards.length !== lastAutoBoardStreetCount && (bCards.length === 3 || bCards.length === 4 || bCards.length === 5)) {
    lastAutoBoardStreetCount = bCards.length;
    setActiveSizeTab(bCards.length === 3 ? "flop" : bCards.length === 4 ? "turn" : "river");
  }
  updateStreetSizingOverview(bCards.length);

  const hCards = parseCardTokens(el("inp-hero").value);
  el("hero-cards-mini").innerHTML = renderCardBadges(hCards, "sm", 4);
  el("hero-center-cards").innerHTML = renderCardBadges(hCards, "xs", 0);
  el("lbl-hero-combo").textContent = el("inp-hero").value.trim() || "—";

  const seat = currentHeroSeat();
  el("seat-desc-oop").textContent = seat === "OOP" ? "Hero (4c)" : "Villain";
  el("seat-desc-ip").textContent = seat === "IP" ? "Hero (4c)" : "Villain";

  if (!hasSolvedNode) {
    el("board-cards-render").innerHTML = renderCardBadges(bCards, "lg", 0);
    el("lbl-node-street").textContent = stLabel;
    el("lbl-node-pot").textContent = String(Math.round(pot || 100));
  }
}

function hasEnabledOption(sel, val) {
  return [...sel.options].some((o) => o.value === val && !o.disabled);
}

function smartDefaultForSeat(seat) {
  return seat === "IP"
    ? packLoadedState.default_spot_ip || "BB_DEF_VS_1_RAISER"
    : packLoadedState.default_spot_oop || "RFI_BTN";
}

function collectSpotConfig() {
  const clampedBudget = parseBudgetTextToMib(el("inp-budget").value);
  const riverOopBets = el("inp-river-oop-bets").value.trim() || "50%, 100%";
  const riverOopRaises = el("inp-river-oop-raises").value.trim() || "100%";
  return {
    board: el("inp-board").value.trim(),
    hero_hand: el("inp-hero").value.trim(),
    hero_seat: currentHeroSeat(),
    use_pack: el("chk-use-pack").checked,
    villain_spot_id: el("sel-villain-spot").value || "",
    villain_override_range: el("inp-villain-override").value.trim(),
    starting_pot: parseInt(el("inp-pot").value, 10) || 100,
    effective_stack: parseInt(el("inp-stack").value, 10) || 100,
    bet_sizes: riverOopBets,
    raise_sizes: riverOopRaises,
    flop_oop_bets: el("inp-flop-oop-bets").value.trim(),
    flop_oop_raises: el("inp-flop-oop-raises").value.trim(),
    flop_ip_bets: el("inp-flop-ip-bets").value.trim(),
    flop_ip_raises: el("inp-flop-ip-raises").value.trim(),
    turn_oop_bets: el("inp-turn-oop-bets").value.trim(),
    turn_oop_raises: el("inp-turn-oop-raises").value.trim(),
    turn_oop_donks: el("inp-turn-oop-donks").value.trim(),
    turn_ip_bets: el("inp-turn-ip-bets").value.trim(),
    turn_ip_raises: el("inp-turn-ip-raises").value.trim(),
    river_oop_bets: riverOopBets,
    river_oop_raises: riverOopRaises,
    river_oop_donks: el("inp-river-oop-donks").value.trim(),
    river_ip_bets: el("inp-river-ip-bets").value.trim(),
    river_ip_raises: el("inp-river-ip-raises").value.trim(),
    solve_streets: activeSolveScope,
    abstraction_mode: el("sel-abstraction").value,
    max_iterations: parseInt(el("inp-iters").value, 10) || 1000,
    target_dev_pct: parseFloat(el("inp-target-dev") ? el("inp-target-dev").value : "0") || 0,
    rake_pct: parseFloat(el("inp-rake-pct") ? el("inp-rake-pct").value : "0") || 0,
    rake_cap: parseFloat(el("inp-rake-cap") ? el("inp-rake-cap").value : "0") || 0,
    memory_budget_mib: clampedBudget,
    lab_two_range_mode: false,
    lab_oop_range: "",
    lab_ip_range: "",
  };
}

function applySpotConfigToForm(cfg) {
  if (cfg.board !== undefined) el("inp-board").value = cfg.board;
  if (cfg.hero_hand !== undefined) el("inp-hero").value = cfg.hero_hand;
  if (cfg.hero_seat) setHeroSeat(cfg.hero_seat);
  if (cfg.use_pack !== undefined) el("chk-use-pack").checked = Boolean(cfg.use_pack);
  if (cfg.villain_spot_id) {
    el("sel-villain-spot").value = cfg.villain_spot_id;
    userManuallyPickedSpot = true;
  }
  if (cfg.villain_override_range !== undefined) el("inp-villain-override").value = cfg.villain_override_range;
  if (cfg.starting_pot !== undefined) el("inp-pot").value = cfg.starting_pot;
  if (cfg.effective_stack !== undefined) el("inp-stack").value = cfg.effective_stack;

  const fbBets = cfg.bet_sizes || "50%, 100%";
  const fbRaises = cfg.raise_sizes || "100%";
  el("inp-flop-oop-bets").value = cfg.flop_oop_bets || fbBets;
  el("inp-flop-oop-raises").value = cfg.flop_oop_raises || fbRaises;
  el("inp-flop-ip-bets").value = cfg.flop_ip_bets || fbBets;
  el("inp-flop-ip-raises").value = cfg.flop_ip_raises || fbRaises;
  el("inp-turn-oop-bets").value = cfg.turn_oop_bets || fbBets;
  el("inp-turn-oop-raises").value = cfg.turn_oop_raises || fbRaises;
  el("inp-turn-oop-donks").value = cfg.turn_oop_donks || "";
  el("inp-turn-ip-bets").value = cfg.turn_ip_bets || fbBets;
  el("inp-turn-ip-raises").value = cfg.turn_ip_raises || fbRaises;
  el("inp-river-oop-bets").value = cfg.river_oop_bets || fbBets;
  el("inp-river-oop-raises").value = cfg.river_oop_raises || fbRaises;
  el("inp-river-oop-donks").value = cfg.river_oop_donks || "";
  el("inp-river-ip-bets").value = cfg.river_ip_bets || fbBets;
  el("inp-river-ip-raises").value = cfg.river_ip_raises || fbRaises;
  if (cfg.solve_streets) setSolveScope(cfg.solve_streets);

  if (cfg.abstraction_mode) el("sel-abstraction").value = cfg.abstraction_mode;
  if (cfg.max_iterations !== undefined) el("inp-iters").value = cfg.max_iterations;
  if (cfg.target_dev_pct !== undefined && el("inp-target-dev")) el("inp-target-dev").value = cfg.target_dev_pct;
  if (cfg.rake_pct !== undefined && el("inp-rake-pct")) el("inp-rake-pct").value = cfg.rake_pct;
  if (cfg.rake_cap !== undefined && el("inp-rake-cap")) el("inp-rake-cap").value = cfg.rake_cap;
  if (cfg.memory_budget_mib !== undefined) syncBudgetInputs(Number(cfg.memory_budget_mib));
  hasSolvedNode = false;
  syncPackToggleUi();
  updateSprAndLocalCards();
  refreshVillainPreview();
}

function showError(msg) {
  const box = el("alert-box");
  if (!msg) {
    box.classList.add("hidden");
    box.textContent = "";
  } else {
    box.textContent = msg;
    box.classList.remove("hidden");
  }
}

function showWarnings(list) {
  const box = el("warnings-box");
  if (!list || list.length === 0) {
    box.classList.add("hidden");
    box.innerHTML = "";
  } else {
    box.innerHTML = list.map((w) => `<div>⚠ ${escapeHtml(w)}</div>`).join("");
    box.classList.remove("hidden");
  }
}

function syncPackToggleUi() {
  const usePack = el("chk-use-pack").checked;
  el("pack-spot-box").classList.toggle("hidden", !usePack);
  el("override-range-box").classList.toggle("hidden", usePack);
}

function populateSpotsState(data) {
  if (data.footer) {
    el("mandatory-footer").textContent = data.footer;
  }
  if (data.memory_budget_mib) {
    syncBudgetInputs(Number(data.memory_budget_mib));
  }
  if (el("lbl-sys-ram")) {
    el("lbl-sys-ram").textContent = data.system_ram_mib
      ? `Host RAM: ${formatMibHuman(data.system_ram_mib)}`
      : "";
  }
  if (data.native_threads && el("lbl-threads-pill")) {
    el("lbl-threads-pill").textContent = `⚡ ${data.native_threads}T`;
  }
  if (Array.isArray(data.saved_spots)) {
    savedSpotsCache = data.saved_spots;
    renderSavedSpotsLibrary();
  }
  const pill = el("pack-status-pill");
  const sel = el("sel-villain-spot");
  const srcSel = el("sel-bind-source");
  const prevVal = sel.value;

  sel.innerHTML = "";
  srcSel.innerHTML = "";

  packLoadedState = {
    loaded: Boolean(data.loaded),
    default_spot_oop: data.default_spot_oop || "RFI_BTN",
    default_spot_ip: data.default_spot_ip || "BB_DEF_VS_1_RAISER",
    last_spot_id: data.last_spot_id || null,
  };

  if (data.loaded && Array.isArray(data.spots) && data.spots.length > 0) {
    const folderShort = (data.pack_folder || "").split(/[\\/]/).pop() || data.pack_folder;
    const availCount = data.spots.filter((s) => s.available !== false).length;
    pill.textContent = `${folderShort} (${availCount} spots)`;
    pill.className = "gto-pack-pill loaded";
    if (data.pack_folder) {
      el("inp-pack-folder").value = data.pack_folder;
    }

    const groupMap = new Map();
    for (const s of data.spots) {
      const gName = s.group || (s.is_canonical ? "Other" : "Other / unmapped files");
      if (!groupMap.has(gName)) groupMap.set(gName, []);
      groupMap.get(gName).push(s);
    }

    for (const [gName, items] of groupMap.entries()) {
      const optgroup = document.createElement("optgroup");
      optgroup.label = gName;
      for (const s of items) {
        const opt = document.createElement("option");
        const id = s.id || s.spot_id;
        opt.value = id;
        const avail = s.available !== false;
        const fullCombos = s.combo_count_full !== undefined ? s.combo_count_full : s.raw_combos;
        if (avail) {
          opt.textContent = `${s.label} — ${fullCombos.toLocaleString()} (${s.frequency_pct.toFixed(1)}%)`;
        } else {
          opt.disabled = true;
          opt.textContent = `${s.label} — not in pack`;
        }
        optgroup.appendChild(opt);
      }
      sel.appendChild(optgroup);
    }

    const seatDefault = smartDefaultForSeat(currentHeroSeat());
    if (prevVal && hasEnabledOption(sel, prevVal)) {
      sel.value = prevVal;
    } else if (data.last_spot_id && hasEnabledOption(sel, data.last_spot_id)) {
      sel.value = data.last_spot_id;
      userManuallyPickedSpot = true;
    } else if (hasEnabledOption(sel, seatDefault)) {
      sel.value = seatDefault;
    } else {
      const firstEnabled = [...sel.options].find((o) => !o.disabled);
      if (firstEnabled) sel.value = firstEnabled.value;
    }

    for (const src of data.available_sources || []) {
      const o = document.createElement("option");
      o.value = src;
      o.textContent = src;
      srcSel.appendChild(o);
    }
  } else {
    pill.textContent = "no pack";
    pill.className = "gto-pack-pill";
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "— Load folder in Pack & Settings —";
    sel.appendChild(opt);
  }
}

async function refreshVillainPreview() {
  updateSprAndLocalCards();
  const cfg = collectSpotConfig();
  if (cfg.use_pack && !cfg.villain_spot_id) {
    el("villain-counts-text").textContent = "no pack loaded";
    return;
  }
  try {
    const p = await invokeCmd("preview_villain", {
      use_pack: cfg.use_pack,
      spot_id: cfg.villain_spot_id,
      villain_override_range: cfg.villain_override_range,
      board: cfg.board,
      hero_hand: cfg.hero_hand,
    });
    showError("");
    el("villain-counts-text").textContent =
      `Villain ${p.live_combos.toLocaleString()} combos (${p.blocked_combos.toLocaleString()} blocked)`;

    if (!hasSolvedNode && Array.isArray(p.sample_top_20)) {
      renderPreviewDetailTable(p);
    }
  } catch (err) {
    el("villain-counts-text").textContent = err.message;
  }
}

function renderPreviewDetailTable(p) {
  setRightView("combos");
  el("lbl-detail-title").textContent = "3. Villain Live Range";
  el("lbl-detail-mode").textContent = `${p.live_combos.toLocaleString()} live`;
  if (el("lbl-vfilter-count")) {
    el("lbl-vfilter-count").textContent = `${p.live_combos.toLocaleString()} live`;
  }
  el("villain-agg-wrap").classList.add("hidden");

  const rows = p.sample_top_20 || [];
  if (!rows.length) {
    el("villain-detail-table").classList.add("hidden");
    el("detail-empty").classList.remove("hidden");
    el("villain-detail-more").classList.add("hidden");
    return;
  }
  el("detail-empty").classList.add("hidden");
  el("villain-detail-thead").innerHTML = `<tr><th>PLO-4 Combo</th><th>Cards</th><th class="num">Weight</th></tr>`;
  el("villain-detail-tbody").innerHTML = rows
    .map((r) => {
      const mini = renderCardBadges(parseCardTokens(r.hand), "xs", 0);
      return `<tr><td>${escapeHtml(r.hand)}</td><td><div class="inline-mini-cards">${mini}</div></td><td class="num">${(r.weight * 100).toFixed(1)}%</td></tr>`;
    })
    .join("");
  el("villain-detail-table").classList.remove("hidden");

  const rem = Math.max(0, (p.live_combos || 0) - rows.length);
  if (rem > 0) {
    el("villain-detail-more").textContent = `+ ${rem.toLocaleString()} more live combos`;
    el("villain-detail-more").classList.remove("hidden");
  } else {
    el("villain-detail-more").classList.add("hidden");
  }
}

function renderConvergenceChart(points) {
  const wrap = el("conv-chart-wrap");
  const svg = el("svg-conv-chart");
  if (!wrap || !svg) return;
  if (!Array.isArray(points) || points.length === 0) {
    wrap.classList.add("hidden");
    return;
  }
  wrap.classList.remove("hidden");
  const W = 260;
  const H = 64;
  const padX = 6;
  const padY = 6;
  const maxIter = Math.max(1, ...points.map((p) => Number(p.iteration || 0)));
  const devs = points.map((p) => Math.max(0.001, Number(p.dev_pct_pot || 0)));
  const logDevs = devs.map((d) => Math.log10(d));
  const minLog = Math.min(...logDevs);
  const maxLog = Math.max(minLog + 0.15, ...logDevs);

  const coords = points.map((p, idx) => {
    const x =
      points.length === 1
        ? W / 2
        : padX + ((Number(p.iteration || idx + 1) / maxIter) * (W - 2 * padX));
    const ly = Math.log10(Math.max(0.001, Number(p.dev_pct_pot || 0)));
    const y = H - padY - ((ly - minLog) / (maxLog - minLog)) * (H - 2 * padY);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });

  const last = points[points.length - 1];
  if (el("lbl-conv-last")) {
    el("lbl-conv-last").textContent = `${Number(last.dev_pct_pot || 0).toFixed(2)}% pot @ ${last.iteration}`;
  }

  const areaPts = `${padX},${H - padY} ${coords.join(" ")} ${coords[coords.length - 1].split(",")[0]},${H - padY}`;
  svg.innerHTML = `
    <polygon points="${areaPts}" fill="rgba(56, 189, 248, 0.14)" />
    <polyline points="${coords.join(" ")}" fill="none" stroke="#38bdf8" stroke-width="1.8" stroke-linejoin="round" />
  `;
}

function renderHeroNutOutMatrix(nutInfo) {
  const box = el("hero-nut-outs-box");
  const body = el("hero-nut-outs-body");
  if (!box || !body) return;
  if (!nutInfo || !nutInfo.suit_structure) {
    box.classList.add("hidden");
    return;
  }
  box.classList.remove("hidden");
  if (el("lbl-hero-suit-struct")) {
    el("lbl-hero-suit-struct").textContent = `${nutInfo.suit_structure} · ${nutInfo.current_category || ""}`;
  }
  if (nutInfo.street === "River") {
    const tag = nutInfo.is_river_nuts_vs_villain
      ? `<span class="grade-pill grade-optimal">✓ Effective 1st Nuts vs Villain Range</span>`
      : `<span class="saved-spot-sub">River complete — Villain range contains stronger 2+3 hands on this board.</span>`;
    body.innerHTML = `<div>${tag}</div>`;
    return;
  }

  const renderTier = (lbl, cls, arr) => {
    if (!Array.isArray(arr) || arr.length === 0) {
      return `<div class="nut-tier-row"><span class="nut-tier-lbl ${cls}">${lbl} (0):</span><span class="saved-spot-sub">None</span></div>`;
    }
    const chips = arr
      .map((it) => {
        const s = String(it.card || "").slice(-1).toLowerCase();
        const r = String(it.card || "").slice(0, -1);
        return `<span class="nut-card-chip suit-${s}" title="${escapeHtml(it.category || "")}">${escapeHtml(r)}${SUIT_GLYPH[s] || s}</span>`;
      })
      .join("");
    return `<div class="nut-tier-row"><span class="nut-tier-lbl ${cls}">${lbl} (${arr.length}):</span><div class="nut-cards-wrap">${chips}</div></div>`;
  };

  body.innerHTML = [
    renderTier("1st Nut Outs", "tier-1", nutInfo.nut_outs),
    renderTier("2nd Nut Outs", "tier-2", nutInfo.second_nut_outs),
    renderTier("Dominated Outs", "tier-dom", nutInfo.non_nut_outs),
  ].join("");
}

function renderProgress(ev) {
  if (!ev) return;
  el("st-iters").textContent = `${ev.iterations} / ${ev.max_iterations}`;
  el("st-speed").textContent = `${ev.iters_per_sec.toFixed(1)} it/s`;
  el("st-expl").textContent = ev.exploitability_sci;
  el("st-ev-hero").textContent = `${ev.ev_hero.toFixed(2)} (${ev.hero_ev_pct_pot.toFixed(1)}%)`;
  el("kpi-hero-ev").textContent = `${ev.ev_hero.toFixed(2)} (${ev.hero_ev_pct_pot.toFixed(1)}%)`;
  el("kpi-villain-ev").textContent = `${ev.ev_villain.toFixed(2)}`;

  if (ev.iterations > 0) {
    const devPct =
      ev.exploitability_pct_pot !== undefined
        ? Number(ev.exploitability_pct_pot)
        : Number(ev.exploitability || 0);
    if (
      liveConvPoints.length === 0 ||
      liveConvPoints[liveConvPoints.length - 1].iteration < ev.iterations
    ) {
      liveConvPoints.push({
        iteration: ev.iterations,
        dev_pct_pot: devPct,
        hero_ev: ev.ev_hero,
      });
      renderConvergenceChart(liveConvPoints);
    }
  }

  const pct = ev.max_iterations > 0 ? Math.min(100, (ev.iterations / ev.max_iterations) * 100) : 0;
  el("solve-progress-fill").style.width = `${pct.toFixed(1)}%`;

  const running = ev.status === "running";
  el("btn-solve").disabled = running;
  el("btn-stop").disabled = !running;
}

function renderTreeNavigator(historyArr) {
  const wrap = el("history-breadcrumbs");
  wrap.innerHTML = "";

  const rootBtn = document.createElement("span");
  rootBtn.className = "tree-node-box" + (historyArr.length === 0 ? " active" : "");
  rootBtn.textContent = "Root";
  rootBtn.title = "Jump to Root";
  rootBtn.addEventListener("click", async () => {
    try {
      const n = await invokeCmd("rewind_to", { depth: 0 });
      renderNode(n);
    } catch (e) {
      showError(e.message);
    }
  });
  wrap.appendChild(rootBtn);

  historyArr.forEach((stepLabel, idx) => {
    const arr = document.createElement("span");
    arr.className = "tree-arrow";
    arr.textContent = "▶";
    wrap.appendChild(arr);

    const box = document.createElement("span");
    const depth = idx + 1;
    box.className = "tree-node-box" + (depth === historyArr.length ? " active" : "");
    box.textContent = stepLabel;
    box.title = `Rewind tree to step ${depth}`;
    box.addEventListener("click", async () => {
      try {
        const n = await invokeCmd("rewind_to", { depth });
        renderNode(n);
      } catch (e) {
        showError(e.message);
      }
    });
    wrap.appendChild(box);
  });
}

function updateDrillScoreUi() {
  if (!el("lbl-drill-score")) return;
  const avg = drillHandsPlayed > 0 ? drillTotalEvLoss / drillHandsPlayed : 0.0;
  el("lbl-drill-score").textContent = `Decisions: ${drillHandsPlayed} · Avg ΔEV Loss: ${avg.toFixed(2)}`;
}

function setLockMode(mode) {
  activeLockMode = mode === "classes" ? "classes" : "uniform";
  if (el("tab-lock-uniform")) el("tab-lock-uniform").classList.toggle("active", activeLockMode === "uniform");
  if (el("tab-lock-classes")) el("tab-lock-classes").classList.toggle("active", activeLockMode === "classes");
  if (el("node-lock-sliders")) el("node-lock-sliders").classList.toggle("hidden", activeLockMode !== "uniform");
  if (el("node-lock-classes-wrap")) el("node-lock-classes-wrap").classList.toggle("hidden", activeLockMode !== "classes");
}

function populateLockDrawer(node) {
  const list = el("node-lock-sliders");
  if (!list || !node || !Array.isArray(node.actions) || !node.actions.length) return;
  setLockMode(activeLockMode);
  const stratSource = node.hero_acts_here ? node.hero_strategy : node.villain_aggregate_strategy;
  const freqMap = new Map();
  if (Array.isArray(stratSource)) {
    for (const s of stratSource) freqMap.set(s.action_index, (s.probability || 0) * 100);
  }
  list.innerHTML = node.actions
    .map((a, idx) => {
      const initPct = freqMap.has(a.index) ? freqMap.get(a.index) : 100 / node.actions.length;
      return `
        <div class="lock-slider-row" data-act-idx="${idx}">
          <span>${escapeHtml(a.label)}</span>
          <input type="range" min="0" max="100" step="1" value="${initPct.toFixed(0)}" class="lock-rng" />
          <input type="number" min="0" max="100" step="0.5" value="${initPct.toFixed(1)}" class="lock-pct-inp" />
        </div>
      `;
    })
    .join("");
  for (const row of list.querySelectorAll(".lock-slider-row")) {
    const rng = row.querySelector(".lock-rng");
    const inp = row.querySelector(".lock-pct-inp");
    rng.addEventListener("input", () => {
      inp.value = Number(rng.value).toFixed(1);
    });
    inp.addEventListener("input", () => {
      rng.value = String(Math.max(0, Math.min(100, parseFloat(inp.value) || 0)));
    });
  }

  // #6 Populate Per-Hand-Class Lock Rows
  const clsWrap = el("node-lock-classes-list");
  if (clsWrap) {
    const vClasses = Array.isArray(node.villain_hand_classes) ? node.villain_hand_classes : [];
    const fallbackClasses = [
      { id: "fh_plus", label: "Quads / Full House+", action_probs: [] },
      { id: "flush", label: "Flush", action_probs: [] },
      { id: "straight", label: "Straight", action_probs: [] },
      { id: "set_trips", label: "Set / Trips", action_probs: [] },
      { id: "two_pair", label: "Two Pair", action_probs: [] },
      { id: "combo_draw", label: "Combo Draw", action_probs: [] },
      { id: "nut_fd", label: "Nut Flush Draw", action_probs: [] },
      { id: "wrap", label: "Wrap (9+ Outs)", action_probs: [] },
      { id: "nut_blocker", label: "Bare Nut Blocker", action_probs: [] },
      { id: "air", label: "Air / Backdoors", action_probs: [] },
    ];
    const rowsToRender = vClasses.length > 0 ? vClasses : fallbackClasses;
    clsWrap.innerHTML = rowsToRender
      .map((c) => {
        const inputsHtml = node.actions
          .map((a, aIdx) => {
            const defP =
              Array.isArray(c.action_probs) && c.action_probs[aIdx] !== undefined
                ? (Number(c.action_probs[aIdx]) * 100).toFixed(0)
                : (100 / node.actions.length).toFixed(0);
            return `<label>${escapeHtml(a.label)} <input type="number" min="0" max="100" step="5" value="${defP}" class="cls-act-inp" data-act-idx="${aIdx}" /></label>`;
          })
          .join("");
        return `
          <div class="lock-class-row" data-class-id="${escapeHtml(c.id)}">
            <label class="lock-class-name"><input type="checkbox" class="cls-lock-chk" /> ${escapeHtml(c.label)}</label>
            <div class="lock-class-inputs">${inputsHtml}</div>
          </div>
        `;
      })
      .join("");
  }
}

function renderSavedSpotsLibrary() {
  const box = el("saved-spots-list");
  if (!box) return;
  if (!Array.isArray(savedSpotsCache) || savedSpotsCache.length === 0) {
    box.innerHTML = `<div class="gto-empty-box">No saved spots yet — click <strong>★ Bookmark Spot</strong> above to save your first spot!</div>`;
    return;
  }
  box.innerHTML = savedSpotsCache
    .map((s) => {
      const evTxt =
        typeof s.summary_hero_ev === "number" ? ` · Hero EV ${s.summary_hero_ev.toFixed(2)}` : "";
      const cfg = s.config || {};
      return `
        <div class="saved-spot-item" data-bm-id="${escapeHtml(s.id)}">
          <div class="saved-spot-meta">
            <span class="saved-spot-title">[${escapeHtml(s.folder || "General")}] ${escapeHtml(s.name)}</span>
            <span class="saved-spot-sub">${escapeHtml(cfg.board || "")} · Hero ${escapeHtml(cfg.hero_hand || "")} (${escapeHtml(cfg.hero_seat || "IP")}) · Pot ${cfg.starting_pot || 100}${evTxt}</span>
          </div>
          <div class="lock-hdr-btns">
            <button type="button" class="mini-card-btn btn-bm-load" data-id="${escapeHtml(s.id)}">Load</button>
            <button type="button" class="mini-card-btn btn-bm-del" data-id="${escapeHtml(s.id)}">✕</button>
          </div>
        </div>
      `;
    })
    .join("");

  for (const b of box.querySelectorAll(".btn-bm-load")) {
    b.addEventListener("click", () => {
      const found = savedSpotsCache.find((x) => x.id === b.dataset.id);
      if (found && found.config) {
        applySpotConfigToForm(found.config);
        el("library-modal").classList.add("hidden");
      }
    });
  }
  for (const b of box.querySelectorAll(".btn-bm-del")) {
    b.addEventListener("click", async () => {
      try {
        savedSpotsCache = await invokeCmd("delete_saved_spot", { id: b.dataset.id });
        renderSavedSpotsLibrary();
      } catch (e) {
        showError(e.message);
      }
    });
  }
}

function renderStreetMinimap(node) {
  const wrap = el("street-minimap-nodes");
  if (!wrap) return;
  const mm = Array.isArray(node.tree_minimap) ? node.tree_minimap : [];
  if (!mm.length) {
    wrap.innerHTML = `<span class="saved-spot-sub">No street nodes</span>`;
    return;
  }
  wrap.innerHTML = "";
  for (const item of mm) {
    const btn = document.createElement("button");
    btn.type = "button";
    const pathArr = Array.isArray(item.action_path) ? item.action_path : [];
    const indent = pathArr.length > 0 ? `${"↳".repeat(Math.min(3, pathArr.length))} ` : "";
    const roleTag = item.node_type === "oop" ? "OOP" : item.node_type === "ip" ? "IP" : item.node_type.toUpperCase();
    btn.className =
      "mm-node-btn" +
      (item.is_current ? " current" : item.is_on_path ? " on-path" : "");
    btn.textContent = `${indent}${item.edge_label} → ${roleTag} (${item.pot})`;
    btn.title = `Jump directly to ${item.edge_label} (${roleTag}, Pot ${item.pot})`;
    btn.addEventListener("click", async () => {
      try {
        showError("");
        const n = await invokeCmd("jump_street_path", { action_path: pathArr });
        renderNode(n);
      } catch (e) {
        showError(e.message);
      }
    });
    wrap.appendChild(btn);
  }
}

function heatColorForEv(ev, minEv, maxEv) {
  if (!Number.isFinite(ev)) return "";
  const span = Math.max(0.5, maxEv - minEv);
  const t = Math.max(0, Math.min(1, (ev - minEv) / span));
  // Red (0.0) -> Slate/Amber (0.5) -> Emerald Green (1.0)
  const r = Math.round(t < 0.5 ? 195 : 195 - (t - 0.5) * 2 * 155);
  const g = Math.round(t < 0.5 ? 55 + t * 2 * 95 : 150 + (t - 0.5) * 2 * 30);
  const b = Math.round(68);
  return `rgb(${r}, ${g}, ${b})`;
}

function renderNode(node) {
  if (!node) return;
  lastRenderedNode = node;
  hasSolvedNode = true;
  showWarnings(node.warnings || []);

  el("btn-reset-cursor").disabled = false;
  el("btn-step-more").disabled = false;
  if (el("btn-bot-toggle")) el("btn-bot-toggle").disabled = false;
  if (el("btn-copy-report")) el("btn-copy-report").disabled = false;
  if (Array.isArray(node.convergence_history) && node.convergence_history.length > 0) {
    liveConvPoints = node.convergence_history;
    renderConvergenceChart(liveConvPoints);
  }
  renderHeroNutOutMatrix(node.hero_nut_outs || null);

  botModeActive = Boolean(node.bot_active);
  if (el("btn-bot-toggle")) {
    el("btn-bot-toggle").classList.toggle("active-mode", botModeActive);
  }
  if (el("bot-banner")) {
    el("bot-banner").classList.toggle("hidden", !botModeActive);
  }
  if (el("bot-showdown-box")) {
    const sd = node.bot_showdown;
    if (sd && sd.bot_hand) {
      const botCards = renderCardBadges(parseCardTokens(sd.bot_hand), "xs", 0);
      const sdNote = sd.went_to_showdown
        ? `${escapeHtml(sd.winner)} — Hero (${escapeHtml(sd.hero_category)}) vs Bot (${escapeHtml(sd.bot_category)})`
        : `${escapeHtml(sd.winner)} (Fold before Showdown — Bot held ${escapeHtml(sd.bot_hand)})`;
      el("bot-showdown-box").innerHTML = `<div><strong>🏁 Hand Complete:</strong> ${sdNote} · Bot Hand: <span class="inline-mini-cards">${botCards}</span> <code>${escapeHtml(
        sd.bot_hand
      )}</code></div>`;
      el("bot-showdown-box").classList.remove("hidden");
    } else {
      el("bot-showdown-box").classList.add("hidden");
    }
  }
  if (el("lbl-threads-pill") && node.native_threads) {
    el("lbl-threads-pill").textContent = `⚡ ${node.native_threads}T`;
  }
  if (el("btn-resolve-subgame")) {
    el("btn-resolve-subgame").classList.toggle("hidden", !node.can_subgame_resolve);
  }
  if (el("btn-exit-subgame")) {
    el("btn-exit-subgame").classList.toggle("hidden", !node.in_subgame);
  }
  if (el("lbl-subgame-badge")) {
    el("lbl-subgame-badge").classList.toggle("hidden", !node.in_subgame);
  }
  renderStreetMinimap(node);

  const nt = String(node.node_type || "").toLowerCase();
  const isPlayerNode = nt === "oop" || nt === "ip";
  if (el("btn-open-lock")) {
    el("btn-open-lock").disabled = !isPlayerNode;
    el("btn-open-lock").textContent = node.is_locked ? "🔒 Locked (Edit)" : "🔒 Lock Node";
    el("btn-open-lock").classList.toggle("active-mode", Boolean(node.is_locked));
  }
  if (el("lbl-locked-badge")) {
    const cnt = Number(node.locked_node_count || 0);
    el("lbl-locked-badge").textContent = `🔒 ${cnt} Locked`;
    el("lbl-locked-badge").classList.toggle("hidden", cnt <= 0);
  }
  if (!isPlayerNode && el("node-lock-panel")) {
    el("node-lock-panel").classList.add("hidden");
  } else if (el("node-lock-panel") && !el("node-lock-panel").classList.contains("hidden")) {
    populateLockDrawer(node);
  }

  // Felt table board + pot + seat highlights
  const boardParsed = parseCardTokens(node.board);
  el("board-cards-render").innerHTML = renderCardBadges(boardParsed, "lg", 0);
  el("lbl-node-street").textContent = String(node.street || "").toUpperCase();
  el("lbl-node-pot").textContent = String(node.pot);
  el("lbl-node-committed").textContent = `[${node.committed[0]}, ${node.committed[1]}]`;
  el("lbl-commit-oop").textContent = String(node.committed[0]);
  el("lbl-commit-ip").textContent = String(node.committed[1]);

  el("seat-card-oop").classList.toggle("acting-seat", nt === "oop");
  el("seat-card-ip").classList.toggle("acting-seat", nt === "ip");
  el("seat-desc-oop").textContent = node.hero_seat === "OOP" ? `Hero (${node.hero_hand || "4c"})` : "Villain";
  el("seat-desc-ip").textContent = node.hero_seat === "IP" ? `Hero (${node.hero_hand || "4c"})` : "Villain";

  const badge = el("node-badge");
  const lockSuffix = node.is_locked ? " · 🔒 LOCKED" : "";
  if (nt === "oop") {
    badge.textContent = (node.hero_acts_here ? "OOP TO ACT · HERO" : "OOP TO ACT · VILLAIN") + lockSuffix;
    badge.className = "turn-badge turn-oop";
  } else if (nt === "ip") {
    badge.textContent = (node.hero_acts_here ? "IP TO ACT · HERO" : "IP TO ACT · VILLAIN") + lockSuffix;
    badge.className = "turn-badge turn-ip";
  } else {
    badge.textContent = nt.toUpperCase();
    badge.className = "turn-badge";
  }

  // EVs & telemetry
  const hideHeroFreqsForDrill = (drillModeActive || botModeActive) && Boolean(node.hero_acts_here);
  el("kpi-hero-ev").textContent = `${node.root_ev_hero.toFixed(2)} (${node.root_ev_hero_pct_pot.toFixed(1)}%)`;
  el("kpi-node-ev").textContent = hideHeroFreqsForDrill ? "Hidden (Drill)" : `${node.node_ev_hero.toFixed(2)}`;
  el("kpi-villain-ev").textContent = `${node.root_ev_villain.toFixed(2)}`;
  el("st-expl").textContent = node.exploitability_sci;
  el("st-ev-hero").textContent = `${node.root_ev_hero.toFixed(2)} (${node.root_ev_hero_pct_pot.toFixed(1)}%)`;
  el("st-iters").textContent = String(node.iterations);

  // Hero PLO-4 Hand Class pill
  if (el("lbl-hero-class")) {
    if (node.hero_hand_class) {
      el("lbl-hero-class").textContent = node.hero_hand_class;
      el("lbl-hero-class").classList.remove("hidden");
    } else {
      el("lbl-hero-class").classList.add("hidden");
    }
  }

  // Horizontal GTO+ tree navigator
  renderTreeNavigator(node.history || []);

  // Active frequencies & EVs at this node
  const activeFreqMap = new Map();
  const heroStratByIdx = new Map();
  if (Array.isArray(node.hero_strategy)) {
    for (const s of node.hero_strategy) {
      heroStratByIdx.set(s.action_index, s);
    }
  }
  const stratSource = node.hero_acts_here ? node.hero_strategy : node.villain_aggregate_strategy;
  if (Array.isArray(stratSource)) {
    for (const s of stratSource) {
      activeFreqMap.set(s.action_index, s.probability);
    }
  }

  // GTO+ Color-Coded Action Tiles
  const actBox = el("tree-actions-box");
  const actBtns = el("tree-action-buttons");
  if (Array.isArray(node.actions) && node.actions.length > 0) {
    el("lbl-step-prompt").textContent = hideHeroFreqsForDrill
      ? "🎯 Drill Mode — Pick Hero's best GTO action:"
      : node.hero_acts_here
      ? "Hero Decision — Click an action tile to step down tree:"
      : "Villain Decision — Click a Villain branch to inspect Hero's response:";
    actBtns.innerHTML = "";
    node.actions.forEach((a, idx) => {
      const prob = activeFreqMap.has(a.index) ? activeFreqMap.get(a.index) : 0;
      const pct = Math.max(0, Math.min(100, prob * 100));
      const th = gtoActionTheme(a.label, idx, node.actions.length);

      const tile = document.createElement("button");
      tile.type = "button";
      tile.className = `gto-act-tile ${th.tile}`;
      tile.innerHTML = hideHeroFreqsForDrill
        ? `
        <div class="tile-top">
          <span class="tile-name">${escapeHtml(a.label)}</span>
          <span class="tile-pct">? %</span>
        </div>
        <div class="tile-bar-bg">
          <div class="tile-bar-fill ${th.bg}" style="width:0%"></div>
        </div>
      `
        : `
        <div class="tile-top">
          <span class="tile-name">${escapeHtml(a.label)}</span>
          <span class="tile-pct">${pct.toFixed(1)}%</span>
        </div>
        <div class="tile-bar-bg">
          <div class="tile-bar-fill ${th.bg}" style="width:${pct.toFixed(1)}%"></div>
        </div>
      `;
      tile.addEventListener("click", async () => {
        try {
          showError("");
          if (botModeActive && node.hero_acts_here) {
            const res = await invokeCmd("bot_play_hero_action", { action_index: a.index });
            const picked = res.hero_action_grade || {};
            const loss = Number(picked.ev_loss || 0);
            const freqPct = Number(picked.probability || 0) * 100;
            const evts = Array.isArray(res.bot_events) && res.bot_events.length
              ? ` · Next: ${res.bot_events.join(" → ")}`
              : "";
            if (el("bot-events-log")) {
              el("bot-events-log").innerHTML = `You played <b>${escapeHtml(a.label)}</b> (GTO <b>${freqPct.toFixed(
                1
              )}%</b>, ΔEV loss <b>-${loss.toFixed(2)}</b>)${escapeHtml(evts)}`;
            }
            if (res.node) renderNode(res.node);
            return;
          }
          if (drillModeActive && node.hero_acts_here && heroStratByIdx.has(a.index)) {
            const picked = heroStratByIdx.get(a.index);
            const loss = Number(picked.ev_loss || 0);
            const freqPct = Number(picked.probability || 0) * 100;
            drillHandsPlayed += 1;
            drillTotalEvLoss += loss;
            updateDrillScoreUi();
            let bestAct = a.label;
            let bestFreq = freqPct;
            for (const cand of node.hero_strategy || []) {
              if ((cand.probability || 0) * 100 > bestFreq) {
                bestFreq = (cand.probability || 0) * 100;
                bestAct = cand.action;
              }
            }
            const gradeCls =
              loss <= 0.25 || freqPct >= 20
                ? "drill-grade-good"
                : loss <= 1.5
                ? "drill-grade-warn"
                : "drill-grade-bad";
            const verdict =
              loss <= 0.25 || freqPct >= 20
                ? "✓ GTO Approved!"
                : loss <= 1.5
                ? "⚠ Minor Inaccuracy"
                : "✗ Blunder";
            if (el("drill-feedback")) {
              el("drill-feedback").innerHTML = `<span class="${gradeCls}">${verdict}</span> — You chose <b>${escapeHtml(
                a.label
              )}</b> (GTO freq <b>${freqPct.toFixed(1)}%</b>, EV <b>${Number(picked.ev || 0).toFixed(
                2
              )}</b>, ΔEV loss <b>-${loss.toFixed(2)}</b> chips · Top GTO: <b>${escapeHtml(
                bestAct
              )} ${bestFreq.toFixed(0)}%</b>).`;
            }
          }
          const next = await invokeCmd("play_action", { action_index: a.index });
          renderNode(next);
        } catch (e) {
          showError(e.message);
        }
      });
      actBtns.appendChild(tile);
    });
    actBox.classList.remove("hidden");
  } else {
    actBox.classList.add("hidden");
  }

  // GTO+ 4x13 Suit-Rank Chance Runout Matrix + EV Heatmap (#3)
  const chBox = el("tree-chance-box");
  const chGrid = el("tree-chance-buttons");
  if (nt === "chance" && Array.isArray(node.legal_chance_cards)) {
    const legalSet = new Set(node.legal_chance_cards);
    const runoutMap = new Map();
    let minEv = Infinity;
    let maxEv = -Infinity;
    for (const ro of node.chance_runout_stats || []) {
      runoutMap.set(ro.card, ro);
      if (ro.hero_ev !== null && ro.hero_ev !== undefined && Number.isFinite(ro.hero_ev)) {
        if (ro.hero_ev < minEv) minEv = ro.hero_ev;
        if (ro.hero_ev > maxEv) maxEv = ro.hero_ev;
      }
    }
    const useHeatmap = !el("chk-runout-heatmap") || el("chk-runout-heatmap").checked;
    chGrid.innerHTML = "";
    for (const s of SUITS_ORDER) {
      for (const r of RANKS_DESC) {
        const code = `${r}${s}`;
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = `runout-card-btn suit-${s}`;
        btn.textContent = `${r}${SUIT_GLYPH[s]}`;
        const ro = runoutMap.get(code);
        if (!legalSet.has(code)) {
          btn.disabled = true;
        } else {
          if (useHeatmap && ro) {
            if (ro.hero_blocked) {
              btn.style.opacity = "0.42";
            } else if (ro.hero_ev !== null && ro.hero_ev !== undefined) {
              btn.style.background = heatColorForEv(ro.hero_ev, minEv, maxEv);
              btn.style.borderColor = "rgba(255,255,255,0.45)";
            }
          }
          btn.addEventListener("mouseenter", () => {
            if (!el("runout-hover-info") || !ro) return;
            if (ro.hero_blocked) {
              el("runout-hover-info").textContent = `${code}: Blocked by Hero's 4-card hand (Villain EV ${Number(
                ro.villain_ev || 0
              ).toFixed(2)})`;
              return;
            }
            const actsTxt = (ro.next_actions || [])
              .map((na) => `${na.action} ${(na.probability * 100).toFixed(0)}%`)
              .join(" · ");
            const evStr =
              ro.hero_ev !== null && ro.hero_ev !== undefined
                ? `Hero EV: ${ro.hero_ev.toFixed(2)} (${(ro.hero_ev_pct_pot || 0).toFixed(1)}% pot)`
                : "";
            el("runout-hover-info").textContent = `${code} → ${evStr}${actsTxt ? "  |  Next: " + actsTxt : ""}`;
          });
          btn.addEventListener("click", async () => {
            try {
              showError("");
              const next = await invokeCmd("play_chance", { card: code });
              renderNode(next);
            } catch (e) {
              showError(e.message);
            }
          });
        }
        chGrid.appendChild(btn);
      }
    }
    chBox.classList.remove("hidden");
  } else {
    chBox.classList.add("hidden");
  }

  // Hero Strategy Panel (GTO+ Stacked Bar + Table)
  el("lbl-hero-combo").textContent = node.hero_hand || `${node.hero_seat}`;
  if (node.hero_hand) {
    el("hero-center-cards").innerHTML = renderCardBadges(parseCardTokens(node.hero_hand), "xs", 0);
  }
  const hStack = el("hero-stacked-bar");
  if (hideHeroFreqsForDrill) {
    hStack.classList.add("hidden");
    el("hero-strategy-table").classList.add("hidden");
    el("hero-strategy-empty").textContent =
      "🎯 Drill Mode active — select Hero's action above to reveal GTO frequencies & EV loss grading.";
    el("hero-strategy-empty").classList.remove("hidden");
  } else if (node.hero_acts_here && Array.isArray(node.hero_strategy) && node.hero_strategy.length > 0) {
    el("hero-strategy-empty").classList.add("hidden");
    hStack.innerHTML = renderGtoStackedBarHtml(node.hero_strategy);
    hStack.classList.remove("hidden");

    el("hero-strategy-tbody").innerHTML = node.hero_strategy
      .map((r, idx) => {
        const pct = Math.max(0, Math.min(100, r.probability * 100));
        const th = gtoActionTheme(r.action, idx, node.hero_strategy.length);
        const evTxt = r.ev !== undefined ? ` <span class="val-dim">(${Number(r.ev).toFixed(2)})</span>` : "";
        return `
          <tr>
            <td><span class="act-swatch ${th.bg}"></span><strong>${escapeHtml(r.action)}</strong>${evTxt}</td>
            <td style="width:55%">
              <div class="row-bar-track"><div class="row-bar-fill ${th.bg}" style="width:${pct.toFixed(1)}%"></div></div>
            </td>
            <td class="num"><strong>${pct.toFixed(1)}%</strong></td>
          </tr>
        `;
      })
      .join("");
    el("hero-strategy-table").classList.remove("hidden");
  } else {
    hStack.classList.add("hidden");
    el("hero-strategy-table").classList.add("hidden");
    el("hero-strategy-empty").textContent =
      nt === "chance"
        ? "Chance node — hover or click any live Turn/River card in the 4×13 EV heatmap above."
        : nt === "terminal"
        ? `Terminal node reached — Hero EV at this leaf: ${node.node_ev_hero.toFixed(2)} chips.`
        : "Villain acts at this node — click one of Villain's colored action tiles above to inspect Hero's response.";
    el("hero-strategy-empty").classList.remove("hidden");
  }

  renderRightStrategyDetail(node);
}

async function applyVillainFilterQuery(q) {
  if (el("inp-villain-filter")) el("inp-villain-filter").value = q;
  if (el("btn-clear-vfilter")) el("btn-clear-vfilter").classList.toggle("hidden", !q);
  for (const ch of document.querySelectorAll(".vchip")) {
    ch.classList.toggle("active", ch.dataset.q === q && Boolean(q));
  }
  if (!hasSolvedNode) return;
  try {
    const n = await invokeCmd("set_villain_filter", { filter: q });
    renderNode(n);
  } catch (e) {
    showError(e.message);
  }
}

function renderRightStrategyDetail(node) {
  setRightView(activeRightView);

  // Sync filter input + live filtered combo badge (#5)
  const fq = String(node.villain_filter || "");
  if (el("inp-villain-filter") && document.activeElement !== el("inp-villain-filter")) {
    el("inp-villain-filter").value = fq;
  }
  if (el("btn-clear-vfilter")) {
    el("btn-clear-vfilter").classList.toggle("hidden", !fq);
  }
  for (const ch of document.querySelectorAll(".vchip")) {
    ch.classList.toggle("active", Boolean(fq) && ch.dataset.q === fq);
  }
  if (el("lbl-vfilter-count")) {
    const fc = Number(node.filtered_villain_combos || 0);
    const totR = Number(node.villain_reach || 0);
    const filR = Number(node.filtered_villain_reach || 0);
    const share = totR > 1e-9 ? ((filR / totR) * 100).toFixed(0) : "0";
    el("lbl-vfilter-count").textContent = fq ? `${fc} (${share}%)` : `${fc} live`;
  }

  const aggWrap = el("villain-agg-wrap");
  if (!node.hero_acts_here && Array.isArray(node.villain_aggregate_strategy) && node.villain_aggregate_strategy.length > 0) {
    el("villain-stacked-bar").innerHTML = renderGtoStackedBarHtml(node.villain_aggregate_strategy);
    el("villain-agg-bars").innerHTML = node.villain_aggregate_strategy
      .map((r, idx) => {
        const pct = Math.max(0, Math.min(100, r.probability * 100));
        const th = gtoActionTheme(r.action, idx, node.villain_aggregate_strategy.length);
        return `<span class="vleg-item"><span class="act-swatch ${th.bg}"></span>${escapeHtml(r.action)}: <strong>${pct.toFixed(1)}%</strong></span>`;
      })
      .join("");
    aggWrap.classList.remove("hidden");
  } else {
    aggWrap.classList.add("hidden");
  }

  const actions = Array.isArray(node.actions) ? node.actions : [];

  // #1 Render 15-Category PLO-4 Made Hand & Draw Breakdown (`villain_hand_classes`)
  const classesList = el("villain-classes-list");
  const classesEmpty = el("classes-empty");
  const vClasses = Array.isArray(node.villain_hand_classes) ? node.villain_hand_classes : [];
  if (classesList && classesEmpty) {
    if (!vClasses.length) {
      classesList.classList.add("hidden");
      classesEmpty.textContent = fq
        ? `No live Villain hands match filter "${fq}".`
        : "No active Villain reach at this node.";
      classesEmpty.classList.remove("hidden");
    } else {
      classesEmpty.classList.add("hidden");
      classesList.innerHTML = vClasses
        .map((c) => {
          const share = Number(c.share_pct || 0);
          const probs = Array.isArray(c.action_probs) ? c.action_probs : [];
          let barInner = "";
          if (probs.length > 0) {
            barInner = probs
              .map((p, idx) => {
                const pct = Math.max(0, Math.min(100, p * 100));
                if (pct < 0.2) return "";
                const lbl = actions[idx] ? actions[idx].label : "";
                const th = gtoActionTheme(lbl, idx, actions.length);
                return `<div class="${th.bg}" style="width:${pct.toFixed(1)}%;height:100%" title="${escapeHtml(lbl)}: ${pct.toFixed(1)}%"></div>`;
              })
              .join("");
          } else {
            barInner = `<div class="bg-check" style="width:${Math.min(100, share).toFixed(1)}%;height:100%"></div>`;
          }
          const isActFilter = fq.toLowerCase() === String(c.class_id || "").toLowerCase();
          return `
            <div class="plo-class-card${isActFilter ? " active" : ""}" data-class-id="${escapeHtml(c.class_id)}" title="Click to filter Villain range to ${escapeHtml(c.label)}">
              <div class="plo-class-top">
                <span class="plo-class-name">${escapeHtml(c.label)}</span>
                <span class="plo-class-meta">${share.toFixed(1)}% · ${c.combo_count} c</span>
              </div>
              <div class="plo-class-share-track">${barInner}</div>
              <div class="plo-class-sub">
                <span>Sample: ${escapeHtml(c.sample_hand || "—")}</span>
                <span>EV: ${Number(c.ev || 0).toFixed(2)}</span>
              </div>
            </div>
          `;
        })
        .join("");
      for (const card of classesList.querySelectorAll(".plo-class-card")) {
        card.addEventListener("click", () => {
          const cid = card.dataset.classId || "";
          const nextQ = fq.toLowerCase() === cid.toLowerCase() ? "" : cid;
          applyVillainFilterQuery(nextQ);
        });
      }
      classesList.classList.remove("hidden");
    }
  }

  const mode = node.villain_detail_mode || "buckets";
  const rows = node.villain_detail_rows || node.villain_bucket_mix || [];
  const more = Number(node.villain_detail_more || 0);

  el("lbl-detail-title").textContent = "3. Villain Breakdown";
  el("lbl-detail-mode").textContent = node.abstraction || mode;

  if (!rows.length) {
    el("villain-detail-table").classList.add("hidden");
    el("detail-empty").textContent = fq
      ? `No live Villain combos match filter "${fq}".`
      : "No active Villain reach at this node.";
    el("detail-empty").classList.remove("hidden");
    el("villain-detail-more").classList.add("hidden");
    return;
  }

  el("detail-empty").classList.add("hidden");

  if (mode === "combos_reach" || !rows[0].action_probs || rows[0].action_probs.length === 0) {
    el("villain-detail-thead").innerHTML = `<tr><th>Hand</th><th>Cards</th><th class="num">Reach</th><th class="num">EV</th></tr>`;
    el("villain-detail-tbody").innerHTML = rows
      .slice(0, 30)
      .map((r) => {
        const w = r.weight !== undefined ? r.weight : r.reach_weight || 0;
        const ev = r.ev !== undefined ? r.ev : 0;
        const hStr = r.id || r.sample_hand || "";
        const cTag = r.hand_class ? `<span class="combo-class-tag">${escapeHtml(r.hand_class)}</span>` : "";
        const mini = renderCardBadges(parseCardTokens(hStr), "xs", 0);
        return `<tr><td>${escapeHtml(hStr)}${cTag}</td><td><div class="inline-mini-cards">${mini}</div></td><td class="num">${w.toFixed(2)}</td><td class="num">${ev.toFixed(2)}</td></tr>`;
      })
      .join("");
  } else {
    const isBucket = mode === "buckets";
    const actHdrs = actions.map((a) => `<th class="num">${escapeHtml(a.label)}</th>`).join("");
    el("villain-detail-thead").innerHTML = isBucket
      ? `<tr><th>Bkt</th><th>Sample</th><th>Mix</th><th class="num">Wt</th>${actHdrs}</tr>`
      : `<tr><th>Combo</th><th>Mix</th><th class="num">Wt</th>${actHdrs}</tr>`;

    el("villain-detail-tbody").innerHTML = rows
      .slice(0, 30)
      .map((r) => {
        const w = r.weight !== undefined ? r.weight : r.reach_weight || 0;
        const probs = r.action_probs || [];
        const cTag = r.hand_class ? `<span class="combo-class-tag">${escapeHtml(r.hand_class)}</span>` : "";
        const miniBar = probs
          .map((p, idx) => {
            const pct = Math.max(0, Math.min(100, p * 100));
            if (pct < 0.2) return "";
            const lbl = actions[idx] ? actions[idx].label : "";
            const th = gtoActionTheme(lbl, idx, actions.length);
            return `<div class="${th.bg}" style="width:${pct.toFixed(1)}%;height:100%"></div>`;
          })
          .join("");
        const cols = probs.map((p) => `<td class="num">${(p * 100).toFixed(1)}%</td>`).join("");
        if (isBucket) {
          const bid = r.id || `#${r.bucket_id}`;
          return `<tr><td>${escapeHtml(bid)}</td><td>${escapeHtml(r.sample_hand)}</td><td><div class="mini-stack">${miniBar}</div></td><td class="num">${w.toFixed(1)}</td>${cols}</tr>`;
        }
        return `<tr><td>${escapeHtml(r.id || r.sample_hand)}${cTag}</td><td><div class="mini-stack">${miniBar}</div></td><td class="num">${w.toFixed(2)}</td>${cols}</tr>`;
      })
      .join("");
  }

  el("villain-detail-table").classList.remove("hidden");
  if (more > 0) {
    el("villain-detail-more").textContent = `+ ${more.toLocaleString()} more`;
    el("villain-detail-more").classList.remove("hidden");
  } else {
    el("villain-detail-more").classList.add("hidden");
  }

  // #4a Render 52-Card Blocker Impact Matrix
  renderBlockerImpactMatrix(node);

  // #4b Render 5-Quintile Villain Equity / EV Distribution
  renderEquityQuintilesGraph(node, actions);
}

function renderBlockerImpactMatrix(node) {
  const grid = el("blocker-52-matrix");
  if (!grid) return;
  const arr = Array.isArray(node.blocker_impact) ? node.blocker_impact : [];
  if (!arr.length) {
    grid.innerHTML = `<div class="gto-empty-box">Solve a spot to inspect 52-card blocker impact.</div>`;
    return;
  }
  const metric = el("sel-blocker-metric") ? el("sel-blocker-metric").value : "value";
  const byCard = new Map();
  let maxVal = 0.01;
  for (const item of arr) {
    byCard.set(item.card, item);
    if (!item.on_board) {
      const v =
        metric === "air"
          ? Number(item.blocked_air_pct || 0)
          : metric === "total"
          ? Number(item.blocked_total_pct || 0)
          : Number(item.blocked_value_pct || 0);
      if (v > maxVal) maxVal = v;
    }
  }

  grid.innerHTML = "";
  for (const s of SUITS_ORDER) {
    for (const r of RANKS_DESC) {
      const code = `${r}${s}`;
      const info = byCard.get(code);
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `runout-card suit-${s}`;
      if (!info || info.on_board) {
        btn.disabled = true;
        btn.classList.add("dead-card");
        btn.textContent = `${r}${SUIT_GLYPH[s]}`;
      } else {
        const val =
          metric === "air"
            ? Number(info.blocked_air_pct || 0)
            : metric === "total"
            ? Number(info.blocked_total_pct || 0)
            : Number(info.blocked_value_pct || 0);
        const ratio = Math.max(0, Math.min(1, val / maxVal));
        // Amber-crimson intensity for strong value/air removal
        const red = Math.round(35 + ratio * 165);
        const grn = Math.round(45 + ratio * 45);
        const blu = Math.round(65 - ratio * 25);
        btn.style.background = `rgb(${red}, ${grn}, ${blu})`;
        if (info.in_hero) {
          btn.style.outline = "2px solid #fde047";
        }
        const star = info.in_hero ? `<span class="blocker-hero-star">★</span>` : "";
        btn.innerHTML = `<span>${star}${r}${SUIT_GLYPH[s]}</span><span style="display:block;font-size:8.5px;opacity:0.95">${val.toFixed(0)}%</span>`;
        const summary = `${code}${info.in_hero ? " [IN HERO HAND ★]" : ""}: Blocks ${Number(info.blocked_value_pct || 0).toFixed(1)}% of Villain Value, ${Number(info.blocked_total_pct || 0).toFixed(1)}% of Total Range, ${Number(info.blocked_air_pct || 0).toFixed(1)}% of Air`;
        btn.title = summary + " (Click to filter Villain combos containing this card)";
        btn.addEventListener("mouseenter", () => {
          if (el("blocker-hover-info")) el("blocker-hover-info").textContent = summary;
        });
        btn.addEventListener("click", () => {
          if (!info.in_hero) applyVillainFilterQuery(code);
        });
      }
      grid.appendChild(btn);
    }
  }
}

function renderEquityQuintilesGraph(node, actions) {
  const list = el("villain-eqgraph-list");
  const empty = el("eqgraph-empty");
  if (!list || !empty) return;
  const bins = Array.isArray(node.equity_buckets) ? node.equity_buckets : [];
  if (!bins.length) {
    list.classList.add("hidden");
    empty.classList.remove("hidden");
    return;
  }
  empty.classList.add("hidden");
  list.classList.remove("hidden");
  list.innerHTML = bins
    .map((b) => {
      const share = Math.max(0, Math.min(100, Number(b.share_pct || 0)));
      const probs = Array.isArray(b.action_probs) ? b.action_probs : [];
      const barSegs =
        probs.length > 0
          ? probs
              .map((p, idx) => {
                const pct = Math.max(0, Math.min(100, p * 100));
                if (pct < 0.2) return "";
                const lbl = actions[idx] ? actions[idx].label : "";
                const th = gtoActionTheme(lbl, idx, actions.length);
                return `<div class="${th.bg}" style="width:${pct.toFixed(1)}%;height:100%" title="${escapeHtml(lbl)}: ${pct.toFixed(1)}%"></div>`;
              })
              .join("")
          : `<div class="bg-check" style="width:${Math.max(4, Math.min(100, Number(b.ev_pct_pot || 0))).toFixed(1)}%;height:100%"></div>`;
      return `
        <div class="plo-class-card">
          <div class="plo-class-top">
            <span class="plo-class-name">${escapeHtml(b.label)} <small style="color:#8b9bb0">(${b.combo_count})</small></span>
            <span class="plo-class-stats"><strong>${share.toFixed(1)}%</strong> · EV ${Number(b.avg_ev || 0).toFixed(1)} (${Number(b.ev_pct_pot || 0).toFixed(0)}% pot)</span>
          </div>
          <div class="plo-class-bar-wrap">
            <div class="plo-class-reach-bar" style="width:100%">${barSegs}</div>
          </div>
        </div>
      `;
    })
    .join("");
}

// =========================================================
// GRAPHICAL 52-CARD PICKER MODAL (Board 3–5 or Hero 4)
// =========================================================
let pickerTarget = "board"; // "board" | "hero"
let pickerCards = [];       // ["Ks", "Qs", ...]
let pickerDeadSet = new Set();

function openCardPicker(target) {
  pickerTarget = target;
  const boardToks = parseCardTokens(el("inp-board").value).map((c) => c.str);
  const heroToks = parseCardTokens(el("inp-hero").value).map((c) => c.str);

  if (target === "board") {
    pickerCards = [...boardToks];
    pickerDeadSet = new Set(heroToks);
    el("picker-modal-title").textContent = "Select Postflop Board (3 Flop, 4 Turn, or 5 River Cards)";
  } else {
    pickerCards = [...heroToks];
    pickerDeadSet = new Set(boardToks);
    el("picker-modal-title").textContent = "Select Hero PLO-4 Hand (Exact 4 Hole Cards)";
  }
  renderCardPickerModal();
  el("card-picker-modal").classList.remove("hidden");
}

function renderCardPickerModal() {
  const maxLen = pickerTarget === "board" ? 5 : 4;
  const parsed = pickerCards.map((s) => ({ rank: s[0], suit: s[1], str: s }));
  el("picker-selected-preview").innerHTML = renderCardBadges(parsed, "sm", maxLen);

  const grid = el("picker-52-grid");
  grid.innerHTML = "";
  for (const s of SUITS_ORDER) {
    for (const r of RANKS_DESC) {
      const code = `${r}${s}`;
      const btn = document.createElement("button");
      btn.type = "button";
      const isSel = pickerCards.includes(code);
      const isDead = pickerDeadSet.has(code);
      btn.className = `deck-card-btn suit-${s}` + (isSel ? " selected" : "");
      btn.textContent = `${r}${SUIT_GLYPH[s]}`;
      if (isDead) {
        btn.disabled = true;
        btn.title = "Used in the other slot";
      } else {
        btn.addEventListener("click", () => {
          const idx = pickerCards.indexOf(code);
          if (idx >= 0) {
            pickerCards.splice(idx, 1);
          } else if (pickerCards.length < maxLen) {
            pickerCards.push(code);
          }
          renderCardPickerModal();
        });
      }
      grid.appendChild(btn);
    }
  }
}

async function applyCardPicker() {
  if (pickerTarget === "board") {
    if (pickerCards.length < 3 || pickerCards.length > 5) {
      showError("Postflop board requires 3 (Flop), 4 (Turn), or 5 (River) cards.");
      return;
    }
    el("inp-board").value = pickerCards.join("");
  } else {
    if (pickerCards.length !== 4) {
      showError("PLO-4 Hero hand requires exact 4 hole cards.");
      return;
    }
    el("inp-hero").value = pickerCards.join("");
  }
  el("card-picker-modal").classList.add("hidden");
  hasSolvedNode = false;
  showError("");
  await refreshVillainPreview();
}

// =========================================================
// SOLVER ACTIONS
// =========================================================
async function onEstimateMemory() {
  showError("");
  try {
    const cfg = collectSpotConfig();
    const est = await invokeCmd("estimate_spot", cfg);
    showWarnings(est.warnings || []);
    const summary =
      est.summary_text ||
      `Est. ${formatMibHuman(est.total_estimated_mib)} / Budget ${formatMibHuman(est.budget_mib)}`;
    el("st-ram").textContent = summary;
  } catch (e) {
    showError(e.message);
  }
}

async function onComputeEquity() {
  showError("");
  try {
    const cfg = collectSpotConfig();
    const eq = await invokeCmd("compute_equity", cfg);
    el("kpi-equity").textContent =
      `${(eq.equity_hero * 100).toFixed(1)}% (W ${(eq.win_hero * 100).toFixed(1)} / T ${(eq.tie * 100).toFixed(1)})`;
  } catch (e) {
    showError(e.message);
  }
}

let pollTimer = null;

async function onStartSolve() {
  showError("");
  liveConvPoints = [];
  const cfg = collectSpotConfig();
  el("btn-solve").disabled = true;
  el("btn-stop").disabled = false;
  el("solve-progress-fill").style.width = "3%";

  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const st = await invokeCmd("latest_progress", {});
      if (st.progress) renderProgress(st.progress);
      if (!st.solving) {
        clearInterval(pollTimer);
        pollTimer = null;
        try {
          const n = await invokeCmd("query_node", {});
          renderNode(n);
        } catch (_) {}
      }
    } catch (_) {}
  }, 250);

  try {
    const res = await invokeCmd("solve_spot", cfg);
    if (res && res.progress) renderProgress(res.progress);
    if (res && res.node) {
      setRightView("classes");
      renderNode(res.node);
    }
  } catch (e) {
    showError(e.message);
  } finally {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
    el("btn-solve").disabled = false;
    el("btn-stop").disabled = true;
  }
}

async function onStopSolve() {
  try {
    await invokeCmd("stop_solve", {});
  } catch (e) {
    showError(e.message);
  }
}

async function initApp() {
  syncPackToggleUi();
  updateSprAndLocalCards();
  subscribeSolveProgress((ev) => renderProgress(ev));

  // Footer "?" popover
  el("btn-footer-help").addEventListener("click", (e) => {
    e.stopPropagation();
    el("footer-popover").classList.toggle("hidden");
  });
  document.addEventListener("click", (e) => {
    const pop = el("footer-popover");
    if (!pop.classList.contains("hidden") && !pop.contains(e.target) && e.target !== el("btn-footer-help")) {
      pop.classList.add("hidden");
    }
  });

  // 52-card graphical picker wiring
  el("btn-open-board-picker").addEventListener("click", () => openCardPicker("board"));
  el("btn-open-hero-picker").addEventListener("click", () => openCardPicker("hero"));
  el("btn-close-picker").addEventListener("click", () => el("card-picker-modal").classList.add("hidden"));
  el("btn-picker-clear").addEventListener("click", () => {
    pickerCards = [];
    renderCardPickerModal();
  });
  el("btn-picker-apply").addEventListener("click", applyCardPicker);

  el("chk-use-pack").addEventListener("change", () => {
    syncPackToggleUi();
    refreshVillainPreview();
  });

  el("sel-villain-spot").addEventListener("change", async () => {
    userManuallyPickedSpot = true;
    const chosen = el("sel-villain-spot").value;
    if (chosen) {
      try {
        await invokeCmd("select_spot", { spot_id: chosen });
      } catch (_) {}
    }
    await refreshVillainPreview();
  });

  for (const r of document.querySelectorAll('input[name="hero-seat"]')) {
    r.addEventListener("change", async () => {
      const sel = el("sel-villain-spot");
      if (packLoadedState.loaded && !userManuallyPickedSpot) {
        try {
          const defRes = await invokeCmd("default_spot_for_seat", {
            hero_seat: currentHeroSeat(),
          });
          const candidate = (defRes && defRes.spot_id) || smartDefaultForSeat(currentHeroSeat());
          if (candidate && hasEnabledOption(sel, candidate)) {
            sel.value = candidate;
          }
        } catch (_) {
          const fallback = smartDefaultForSeat(currentHeroSeat());
          if (hasEnabledOption(sel, fallback)) sel.value = fallback;
        }
      }
      await refreshVillainPreview();
    });
  }

  el("btn-refresh-preview").addEventListener("click", refreshVillainPreview);
  el("inp-board").addEventListener("input", updateSprAndLocalCards);
  el("inp-board").addEventListener("change", () => {
    hasSolvedNode = false;
    refreshVillainPreview();
  });
  el("inp-hero").addEventListener("input", updateSprAndLocalCards);
  el("inp-hero").addEventListener("change", () => {
    hasSolvedNode = false;
    refreshVillainPreview();
  });
  el("inp-pot").addEventListener("input", updateSprAndLocalCards);
  el("inp-stack").addEventListener("input", updateSprAndLocalCards);
  el("inp-villain-override").addEventListener("change", refreshVillainPreview);

  // Streets to Solve Scope Selector ("All" / "FlopOnly" / "TurnOnly" / "RiverOnly")
  for (const btn of document.querySelectorAll(".scope-btn")) {
    btn.addEventListener("click", () => {
      hasSolvedNode = false;
      setSolveScope(btn.dataset.scope);
    });
  }

  // GTO+ Flop / Turn / River Street Bet Sizing Tabs, Presets & Sync
  for (const s of ["flop", "turn", "river"]) {
    el(`tab-sz-${s}`).addEventListener("click", () => setActiveSizeTab(s));
  }
  const allStreetInputIds = [
    "inp-flop-oop-bets",
    "inp-flop-oop-raises",
    "inp-flop-ip-bets",
    "inp-flop-ip-raises",
    "inp-turn-oop-bets",
    "inp-turn-oop-raises",
    "inp-turn-oop-donks",
    "inp-turn-ip-bets",
    "inp-turn-ip-raises",
    "inp-river-oop-bets",
    "inp-river-oop-raises",
    "inp-river-oop-donks",
    "inp-river-ip-bets",
    "inp-river-ip-raises",
  ];
  for (const id of allStreetInputIds) {
    el(id).addEventListener("input", () => {
      updateStreetSizingOverview(parseCardTokens(el("inp-board").value).length);
    });
  }

  for (const btn of document.querySelectorAll(".sz-preset-btn")) {
    btn.addEventListener("click", () => {
      const p = btn.dataset.preset;
      const s = activeSizeTab;
      if (p === "pot") {
        el(`inp-${s}-oop-bets`).value = "100%";
        el(`inp-${s}-oop-raises`).value = "100%";
        el(`inp-${s}-ip-bets`).value = "100%";
        el(`inp-${s}-ip-raises`).value = "100%";
      } else if (p === "half-pot") {
        el(`inp-${s}-oop-bets`).value = "50%, 100%";
        el(`inp-${s}-oop-raises`).value = "100%";
        el(`inp-${s}-ip-bets`).value = "50%, 100%";
        el(`inp-${s}-ip-raises`).value = "100%";
      } else if (p === "two-size") {
        el(`inp-${s}-oop-bets`).value = "33%, 75%";
        el(`inp-${s}-oop-raises`).value = "100%";
        el(`inp-${s}-ip-bets`).value = "33%, 75%";
        el(`inp-${s}-ip-raises`).value = "100%";
      }
      updateStreetSizingOverview(parseCardTokens(el("inp-board").value).length);
    });
  }

  el("btn-copy-flop-all").addEventListener("click", () => {
    const src = activeSizeTab;
    const oopB = el(`inp-${src}-oop-bets`).value;
    const oopR = el(`inp-${src}-oop-raises`).value;
    const ipB = el(`inp-${src}-ip-bets`).value;
    const ipR = el(`inp-${src}-ip-raises`).value;
    for (const dst of ["flop", "turn", "river"]) {
      el(`inp-${dst}-oop-bets`).value = oopB;
      el(`inp-${dst}-oop-raises`).value = oopR;
      el(`inp-${dst}-ip-bets`).value = ipB;
      el(`inp-${dst}-ip-raises`).value = ipR;
    }
    updateStreetSizingOverview(parseCardTokens(el("inp-board").value).length);
  });

  el("btn-estimate").addEventListener("click", onEstimateMemory);
  el("btn-equity").addEventListener("click", onComputeEquity);
  el("btn-solve").addEventListener("click", onStartSolve);
  el("btn-stop").addEventListener("click", onStopSolve);

  el("btn-reset-cursor").addEventListener("click", async () => {
    try {
      const n = await invokeCmd("reset_cursor", {});
      renderNode(n);
    } catch (e) {
      showError(e.message);
    }
  });

  el("btn-step-more").addEventListener("click", async () => {
    try {
      el("btn-step-more").disabled = true;
      const res = await invokeCmd("step_iterations", { iterations: 250 });
      if (res.progress) renderProgress(res.progress);
      if (res.node) renderNode(res.node);
    } catch (e) {
      showError(e.message);
    } finally {
      el("btn-step-more").disabled = false;
    }
  });

  // #1, #4 & #5: Right pane Classes / Combos / Blockers / Eq% tabs + PPT/Monker Villain Range Filter
  if (el("tab-view-classes")) el("tab-view-classes").addEventListener("click", () => setRightView("classes"));
  if (el("tab-view-combos")) el("tab-view-combos").addEventListener("click", () => setRightView("combos"));
  if (el("tab-view-blockers")) el("tab-view-blockers").addEventListener("click", () => setRightView("blockers"));
  if (el("tab-view-eqgraph")) el("tab-view-eqgraph").addEventListener("click", () => setRightView("eqgraph"));
  if (el("sel-blocker-metric")) {
    el("sel-blocker-metric").addEventListener("change", () => {
      if (lastRenderedNode) renderBlockerImpactMatrix(lastRenderedNode);
    });
  }

  let vFilterTimer = null;
  if (el("inp-villain-filter")) {
    el("inp-villain-filter").addEventListener("input", () => {
      clearTimeout(vFilterTimer);
      const q = el("inp-villain-filter").value.trim();
      vFilterTimer = setTimeout(() => applyVillainFilterQuery(q), 140);
    });
  }
  if (el("btn-clear-vfilter")) {
    el("btn-clear-vfilter").addEventListener("click", () => applyVillainFilterQuery(""));
  }
  for (const ch of document.querySelectorAll(".vchip")) {
    ch.addEventListener("click", () => {
      const q = ch.dataset.q || "";
      const cur = el("inp-villain-filter") ? el("inp-villain-filter").value.trim() : "";
      applyVillainFilterQuery(cur === q ? "" : q);
    });
  }

  // #2 & #6: Node Locking Drawer (Uniform & Per-Hand-Class)
  if (el("tab-lock-uniform")) el("tab-lock-uniform").addEventListener("click", () => setLockMode("uniform"));
  if (el("tab-lock-classes")) el("tab-lock-classes").addEventListener("click", () => setLockMode("classes"));
  if (el("btn-open-lock")) {
    el("btn-open-lock").addEventListener("click", () => {
      const panel = el("node-lock-panel");
      if (!panel) return;
      const opening = panel.classList.contains("hidden");
      panel.classList.toggle("hidden", !opening);
      if (opening && lastRenderedNode) populateLockDrawer(lastRenderedNode);
    });
  }
  if (el("btn-close-lock")) {
    el("btn-close-lock").addEventListener("click", () => el("node-lock-panel").classList.add("hidden"));
  }
  if (el("btn-apply-lock")) {
    el("btn-apply-lock").addEventListener("click", async () => {
      try {
        showError("");
        const iters = parseInt(el("inp-lock-iters") ? el("inp-lock-iters").value : "250", 10) || 0;
        el("btn-apply-lock").disabled = true;
        let n;
        if (activeLockMode === "classes") {
          const classLocks = {};
          for (const row of document.querySelectorAll("#node-lock-classes-list .lock-class-row")) {
            const chk = row.querySelector(".cls-lock-chk");
            if (chk && chk.checked) {
              const cid = row.dataset.classId;
              const pVec = [...row.querySelectorAll(".cls-act-inp")].map((inp) =>
                Math.max(0, parseFloat(inp.value) || 0)
              );
              classLocks[cid] = pVec;
            }
          }
          if (!Object.keys(classLocks).length) {
            throw new Error("Check at least one PLO Hand Class to lock (or switch to Uniform tab).");
          }
          n = await invokeCmd("lock_node_by_classes", {
            class_locks: classLocks,
            resolve_iterations: iters,
          });
        } else {
          const probs = [...document.querySelectorAll("#node-lock-sliders .lock-pct-inp")].map(
            (inp) => Math.max(0, parseFloat(inp.value) || 0)
          );
          n = await invokeCmd("lock_node", { action_probs: probs, resolve_iterations: iters });
        }
        el("node-lock-panel").classList.add("hidden");
        renderNode(n);
      } catch (e) {
        showError(e.message);
      } finally {
        el("btn-apply-lock").disabled = false;
      }
    });
  }

  // Phase-2 #1: Exact Turn/River Subgame Resolving & Exit Subgame
  if (el("btn-resolve-subgame")) {
    el("btn-resolve-subgame").addEventListener("click", async () => {
      try {
        showError("");
        el("btn-resolve-subgame").disabled = true;
        const maxIters = Math.max(200, parseInt(el("inp-iters").value || "500", 10) || 500);
        const targetDev = Math.max(0.1, parseFloat(el("inp-target-dev").value || "0.2") || 0.2);
        const n = await invokeCmd("resolve_subgame", {
          max_iterations: maxIters,
          target_dev_pct: targetDev,
        });
        renderNode(n);
      } catch (e) {
        showError(e.message);
      } finally {
        el("btn-resolve-subgame").disabled = false;
      }
    });
  }
  if (el("btn-exit-subgame")) {
    el("btn-exit-subgame").addEventListener("click", async () => {
      try {
        showError("");
        const n = await invokeCmd("exit_subgame", {});
        renderNode(n);
      } catch (e) {
        showError(e.message);
      }
    });
  }

  // Phase-2 #7: Street Tree Minimap toggle & Global Keyboard Shortcuts
  if (el("btn-toggle-minimap")) {
    el("btn-toggle-minimap").addEventListener("click", () => {
      const bar = el("street-minimap-bar");
      if (bar) {
        bar.classList.toggle("hidden");
        el("btn-toggle-minimap").classList.toggle("active-mode", !bar.classList.contains("hidden"));
      }
    });
  }
  if (el("btn-copy-report")) {
    el("btn-copy-report").addEventListener("click", async () => {
      try {
        const rep = await invokeCmd("export_node_report", {});
        if (navigator.clipboard && rep.markdown) {
          await navigator.clipboard.writeText(rep.markdown);
        }
        showWarnings(["Node Markdown report copied to clipboard!"]);
      } catch (e) {
        showError(e.message);
      }
    });
  }

  // Phase-2 #2: Saved Spots Library & CSV/MD Export Modal
  if (el("btn-open-library")) {
    el("btn-open-library").addEventListener("click", async () => {
      try {
        savedSpotsCache = await invokeCmd("list_saved_spots", {});
        renderSavedSpotsLibrary();
        el("library-modal").classList.remove("hidden");
      } catch (e) {
        showError(e.message);
      }
    });
  }
  if (el("btn-close-library")) {
    el("btn-close-library").addEventListener("click", () => el("library-modal").classList.add("hidden"));
  }
  if (el("btn-save-bookmark")) {
    el("btn-save-bookmark").addEventListener("click", async () => {
      try {
        const cfg = collectSpotConfig();
        const folder = el("inp-bm-folder") ? el("inp-bm-folder").value.trim() : "General";
        const name = el("inp-bm-name") ? el("inp-bm-name").value.trim() : "";
        const res = await invokeCmd("save_spot_bookmark", { name, folder, config: cfg });
        savedSpotsCache = res.saved_spots || [];
        if (el("inp-bm-name")) el("inp-bm-name").value = "";
        renderSavedSpotsLibrary();
      } catch (e) {
        showError(e.message);
      }
    });
  }
  if (el("btn-export-csv")) {
    el("btn-export-csv").addEventListener("click", async () => {
      try {
        const rep = await invokeCmd("export_node_report", {});
        const blob = new Blob([rep.csv || ""], { type: "text/csv;charset=utf-8" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `plo4_node_${(lastRenderedNode && lastRenderedNode.board) || "report"}.csv`;
        a.click();
        URL.revokeObjectURL(url);
        if (el("lbl-export-status")) el("lbl-export-status").textContent = "CSV downloaded";
      } catch (e) {
        showError(e.message);
      }
    });
  }
  if (el("btn-export-md")) {
    el("btn-export-md").addEventListener("click", async () => {
      try {
        const rep = await invokeCmd("export_node_report", {});
        if (navigator.clipboard && rep.markdown) {
          await navigator.clipboard.writeText(rep.markdown);
        }
        if (el("lbl-export-status")) el("lbl-export-status").textContent = "Markdown copied!";
      } catch (e) {
        showError(e.message);
      }
    });
  }

  // Phase-3 #5: Binary Solved Tree Snapshots (.plotree)
  if (el("btn-save-snapshot")) {
    el("btn-save-snapshot").addEventListener("click", async () => {
      try {
        showError("");
        const p = (el("inp-snapshot-path") ? el("inp-snapshot-path").value : "").trim();
        if (!p) throw new Error("Provide a .plotree path first.");
        const res = await invokeCmd("save_tree_snapshot", { path: p });
        const st = el("lbl-snapshot-status");
        if (st) {
          const kb = (Number(res.bytes_written || 0) / 1024).toFixed(1);
          st.textContent = `✓ Saved ${res.path} (${kb} KiB, ${res.iterations} iters)`;
          st.classList.remove("hidden");
        }
      } catch (e) {
        showError(e.message);
      }
    });
  }
  if (el("btn-load-snapshot")) {
    el("btn-load-snapshot").addEventListener("click", async () => {
      try {
        showError("");
        const p = (el("inp-snapshot-path") ? el("inp-snapshot-path").value : "").trim();
        if (!p) throw new Error("Provide a .plotree path first.");
        const res = await invokeCmd("load_tree_snapshot", { path: p });
        if (res.config) applySpotConfigToForm(res.config);
        if (res.node) renderNode(res.node);
        const st = el("lbl-snapshot-status");
        if (st) {
          st.textContent = `✓ Loaded ${res.path} (${res.iterations} iters)`;
          st.classList.remove("hidden");
        }
        setTimeout(() => el("library-modal").classList.add("hidden"), 450);
      } catch (e) {
        showError(e.message);
      }
    });
  }

  // Phase-3 #2: Play vs Solver Bot
  const dealBotHand = async () => {
    try {
      showError("");
      const res = await invokeCmd("start_bot_hand", {});
      const evts = Array.isArray(res.bot_events) && res.bot_events.length
        ? res.bot_events.join(" → ")
        : "Your turn to act!";
      if (el("bot-events-log")) {
        el("bot-events-log").textContent = `🃏 New Hand Dealt · ${evts}`;
      }
      if (res.node) renderNode(res.node);
    } catch (e) {
      showError(e.message);
    }
  };
  if (el("btn-bot-toggle")) {
    el("btn-bot-toggle").addEventListener("click", async () => {
      if (botModeActive) {
        try {
          const n = await invokeCmd("stop_bot_hand", {});
          renderNode(n);
        } catch (e) {
          showError(e.message);
        }
      } else {
        await dealBotHand();
      }
    });
  }
  if (el("btn-bot-deal")) {
    el("btn-bot-deal").addEventListener("click", dealBotHand);
  }
  if (el("btn-bot-exit")) {
    el("btn-bot-exit").addEventListener("click", async () => {
      try {
        const n = await invokeCmd("stop_bot_hand", {});
        renderNode(n);
      } catch (e) {
        showError(e.message);
      }
    });
  }

  // Phase-3 #3: Auto-Simplify Strategy / Pure-Size Optimizer Modal
  if (el("btn-open-simplify")) {
    el("btn-open-simplify").addEventListener("click", () => el("simplify-modal").classList.remove("hidden"));
  }
  if (el("btn-close-simplify")) {
    el("btn-close-simplify").addEventListener("click", () => el("simplify-modal").classList.add("hidden"));
  }
  if (el("btn-run-simplify")) {
    el("btn-run-simplify").addEventListener("click", async () => {
      try {
        showError("");
        el("btn-run-simplify").disabled = true;
        if (el("simplify-status")) {
          el("simplify-status").textContent = "Solving Multi-Size baseline + single-size candidates…";
        }
        const cfg = collectSpotConfig();
        const rawSizes = el("inp-simplify-sizes") ? el("inp-simplify-sizes").value : "";
        const candidate_sizes = rawSizes
          .split("|")
          .map((s) => s.trim())
          .filter(Boolean);
        const iterations = parseInt(el("inp-simplify-iters") ? el("inp-simplify-iters").value : "120", 10) || 120;
        const rep = await invokeCmd("compare_bet_simplifications", {
          config: cfg,
          candidate_sizes,
          iterations,
        });
        const rows = Array.isArray(rep.rows) ? rep.rows : [];
        if (el("simplify-status")) {
          el("simplify-status").textContent = `Recommended Pure Size on ${rep.street}: ${rep.recommended_size || "—"} (Baseline Hero EV ${Number(rep.baseline_ev || 0).toFixed(2)})`;
        }
        const tbody = el("simplify-results-tbody");
        tbody.innerHTML = rows
          .map((r) => {
            const isBest = !r.is_baseline && r.sizes === rep.recommended_size;
            const stratTxt = (r.root_strategy || [])
              .filter((s) => s.probability >= 0.03)
              .map((s) => `${escapeHtml(s.action)} ${(s.probability * 100).toFixed(0)}%`)
              .join(" · ");
            return `
              <tr class="${isBest ? "best-simplify-row" : ""}">
                <td><strong>${escapeHtml(r.label)}</strong>${isBest ? ` <span class="grade-pill grade-optimal">★ Best Pure</span>` : ""}</td>
                <td class="num">${Number(r.hero_ev || 0).toFixed(2)}</td>
                <td class="num">${Number(r.ev_diff_pct_pot || 0) >= 0 ? "+" : ""}${Number(r.ev_diff_pct_pot || 0).toFixed(2)}%</td>
                <td class="num"><strong>${Number(r.ev_retention_pct || 0).toFixed(1)}%</strong></td>
                <td>${stratTxt || "—"}</td>
                <td class="num"><button type="button" class="mini-card-btn btn-use-sz" data-sz="${escapeHtml(r.sizes)}" data-st="${escapeHtml(rep.street)}">Apply</button></td>
              </tr>
            `;
          })
          .join("");
        el("simplify-results-table").classList.remove("hidden");
        for (const b of tbody.querySelectorAll(".btn-use-sz")) {
          b.addEventListener("click", () => {
            const st = String(b.dataset.st || "River").toLowerCase();
            const sz = b.dataset.sz || "75%";
            if (el(`inp-${st}-oop-bets`)) el(`inp-${st}-oop-bets`).value = sz;
            if (el(`inp-${st}-ip-bets`)) el(`inp-${st}-ip-bets`).value = sz;
            updateStreetSizingOverview(parseCardTokens(el("inp-board").value).length);
            el("simplify-modal").classList.add("hidden");
          });
        }
      } catch (e) {
        showError(e.message);
      } finally {
        el("btn-run-simplify").disabled = false;
      }
    });
  }

  // Phase-2 #3: Hand History Parser Modal
  if (el("btn-open-hh")) {
    el("btn-open-hh").addEventListener("click", () => el("hh-modal").classList.remove("hidden"));
  }
  if (el("btn-close-hh")) {
    el("btn-close-hh").addEventListener("click", () => el("hh-modal").classList.add("hidden"));
  }
  if (el("btn-load-sample-hh")) {
    el("btn-load-sample-hh").addEventListener("click", () => {
      el("txt-hh-input").value = [
        "PokerStars Hand #249001234: Omaha Pot Limit ($1/$2 USD) - 2026/11/22 18:00:00 ET",
        "Table 'PLO4-Arena' 6-max Seat #1 is the button",
        "Seat 1: Hero ($200 in chips)",
        "Seat 2: SmallBlind ($200 in chips)",
        "Seat 3: BigBlind ($200 in chips)",
        "SmallBlind: posts small blind $1",
        "BigBlind: posts big blind $2",
        "*** HOLE CARDS ***",
        "Dealt to Hero [As Ks Qh Jh]",
        "Hero: raises $5 to $7",
        "SmallBlind: folds",
        "BigBlind: calls $5",
        "*** FLOP *** [Ts 9s 4d]",
        "BigBlind: checks",
        "Hero: bets $10",
        "BigBlind: calls $10",
        "*** TURN *** [Ts 9s 4d] [2c]",
        "BigBlind: checks",
      ].join("\n");
    });
  }
  if (el("btn-parse-apply-hh")) {
    el("btn-parse-apply-hh").addEventListener("click", async () => {
      try {
        showError("");
        const raw = el("txt-hh-input").value;
        const parsed = await invokeCmd("parse_hand_history", { text: raw });
        const targetStreet = el("sel-hh-street") ? el("sel-hh-street").value : "final";
        const boardToUse = targetStreet === "flop" ? parsed.flop_board : parsed.final_board;

        el("inp-hero").value = parsed.hero_hand;
        el("inp-board").value = boardToUse;
        setHeroSeat(parsed.hero_seat || "IP");
        el("inp-pot").value = String(parsed.starting_pot);
        el("inp-stack").value = String(parsed.effective_stack);

        if (packLoadedState.loaded && parsed.suggested_spot_id) {
          const sel = el("sel-villain-spot");
          if (hasEnabledOption(sel, parsed.suggested_spot_id)) {
            sel.value = parsed.suggested_spot_id;
            el("chk-use-pack").checked = true;
            syncPackToggleUi();
          }
        }
        updateSprAndLocalCards();
        await refreshVillainPreview();
        const summary = el("hh-parsed-summary");
        if (summary) {
          summary.textContent = `Loaded ${parsed.room}: Hero ${parsed.hero_hand} (${parsed.hero_seat} / ${parsed.hero_table_pos}) vs ${parsed.villain_table_pos} [${parsed.suggested_spot_id}] · Board ${boardToUse} · Pot ${parsed.starting_pot}, Stack ${parsed.effective_stack}`;
          summary.classList.remove("hidden");
        }
        setTimeout(() => el("hh-modal").classList.add("hidden"), 750);
      } catch (e) {
        showError(e.message);
      }
    });
  }

  // Phase-3 #1: Full Hand EV Audit (Flop -> Turn -> River)
  if (el("btn-audit-hh")) {
    el("btn-audit-hh").addEventListener("click", async () => {
      try {
        showError("");
        el("btn-audit-hh").disabled = true;
        const raw = el("txt-hh-input").value;
        const iters = parseInt(el("inp-hh-audit-iters") ? el("inp-hh-audit-iters").value : "100", 10) || 100;
        const vOv = el("chk-use-pack").checked ? "" : el("inp-villain-override").value.trim();
        const summary = el("hh-parsed-summary");
        if (summary) {
          summary.textContent = `Auditing all streets (${iters} DCFR iters/street)…`;
          summary.classList.remove("hidden");
        }
        const rep = await invokeCmd("audit_hand_history", {
          text: raw,
          villain_override_range: vOv,
          iterations_per_street: iters,
        });
        if (summary) {
          summary.textContent = `✓ Audited ${rep.decision_count} Hero decision(s) for ${rep.hero_hand} (${rep.hero_seat}) vs ${rep.villain_spot_id} · Total EV Loss: -${Number(
            rep.total_ev_loss_chips || 0
          ).toFixed(2)} chips (-${Number(rep.total_ev_loss_bb || 0).toFixed(2)} bb)`;
          summary.classList.remove("hidden");
        }
        const decisions = Array.isArray(rep.decisions) ? rep.decisions : [];
        const tbody = el("hh-audit-tbody");
        tbody.innerHTML = decisions
          .map((d, idx) => {
            const gCls =
              d.grade === "Optimal"
                ? "grade-optimal"
                : d.grade === "Inaccuracy"
                ? "grade-inacc"
                : "grade-blunder";
            return `
              <tr>
                <td><strong>${escapeHtml(d.street)}</strong></td>
                <td><code>${escapeHtml(d.board)}</code></td>
                <td>${escapeHtml(d.mapped_tree_action)} <span class="val-dim">(${escapeHtml(d.raw_hh_action)})</span></td>
                <td class="num">${(Number(d.gto_frequency || 0) * 100).toFixed(1)}%</td>
                <td><strong>${escapeHtml(d.best_gto_action)}</strong> (${(Number(d.best_gto_freq || 0) * 100).toFixed(0)}%)</td>
                <td class="num">-${Number(d.ev_loss_chips || 0).toFixed(2)} (${Number(d.ev_loss_bb || 0).toFixed(2)} bb)</td>
                <td><span class="grade-pill ${gCls}">${escapeHtml(d.grade)}</span></td>
                <td class="num"><button type="button" class="mini-card-btn btn-open-audit-spot" data-idx="${idx}">Open</button></td>
              </tr>
            `;
          })
          .join("");
        el("hh-audit-wrap").classList.remove("hidden");
        for (const b of tbody.querySelectorAll(".btn-open-audit-spot")) {
          b.addEventListener("click", async () => {
            const d = decisions[Number(b.dataset.idx || 0)];
            if (d && d.spot_config) {
              applySpotConfigToForm(d.spot_config);
              el("hh-modal").classList.add("hidden");
              await onStartSolve();
            }
          });
        }
      } catch (e) {
        showError(e.message);
      } finally {
        el("btn-audit-hh").disabled = false;
      }
    });
  }

  // Phase-2 #5: Multi-Flop Subset C-Bet Report Modal
  if (el("btn-open-subset")) {
    el("btn-open-subset").addEventListener("click", () => el("subset-modal").classList.remove("hidden"));
  }
  if (el("btn-close-subset")) {
    el("btn-close-subset").addEventListener("click", () => el("subset-modal").classList.add("hidden"));
  }
  if (el("btn-run-subset")) {
    el("btn-run-subset").addEventListener("click", async () => {
      try {
        showError("");
        const cfg = collectSpotConfig();
        const flops = String(el("inp-subset-flops").value || "")
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean);
        const iters = parseInt(el("inp-subset-iters").value || "80", 10) || 80;
        el("btn-run-subset").disabled = true;
        el("subset-status").textContent = `Solving ${flops.length || 8} flops (${iters} iters each)…`;
        const res = await invokeCmd("run_flop_subset_report", {
          config: cfg,
          flops,
          iterations_per_flop: iters,
        });
        const rows = res.rows || [];
        const skipped = res.skipped || [];
        el("subset-status").textContent =
          `Solved ${rows.length} flops` + (skipped.length ? ` (skipped: ${skipped.join("; ")})` : "");
        const tbody = el("subset-results-tbody");
        tbody.innerHTML = rows
          .map((r) => {
            const strat = (r.root_strategy || [])
              .map((s) => `${escapeHtml(s.action)}: ${(Number(s.probability || 0) * 100).toFixed(0)}%`)
              .join(" · ");
            return `
              <tr>
                <td><strong class="mono-code">${escapeHtml(r.flop)}</strong></td>
                <td>${escapeHtml(r.texture)}</td>
                <td>${escapeHtml(r.hero_class || "—")}</td>
                <td class="num">${Number(r.hero_ev || 0).toFixed(2)}</td>
                <td class="num"><strong>${Number(r.hero_ev_pct_pot || 0).toFixed(1)}%</strong></td>
                <td>${strat}</td>
                <td><button type="button" class="mini-card-btn btn-load-subflop" data-flop="${escapeHtml(r.flop)}">Open</button></td>
              </tr>
            `;
          })
          .join("");
        el("subset-results-table").classList.toggle("hidden", rows.length === 0);
        for (const b of tbody.querySelectorAll(".btn-load-subflop")) {
          b.addEventListener("click", async () => {
            el("inp-board").value = b.dataset.flop;
            updateSprAndLocalCards();
            await refreshVillainPreview();
            el("subset-modal").classList.add("hidden");
            onStartSolve();
          });
        }
      } catch (e) {
        showError(e.message);
        el("subset-status").textContent = `Error: ${e.message}`;
      } finally {
        el("btn-run-subset").disabled = false;
      }
    });
  }

  // Phase-2 #7: Global Keyboard Shortcuts (`1-5` actions, `Backspace/ArrowLeft` step back, `R` root, `F` filter, `M` tree map)
  window.addEventListener("keydown", async (ev) => {
    const tag = (ev.target && ev.target.tagName ? ev.target.tagName : "").toLowerCase();
    if (tag === "input" || tag === "textarea" || tag === "select" || ev.metaKey || ev.ctrlKey || ev.altKey) {
      if (ev.key === "Escape" && el("inp-villain-filter") && ev.target === el("inp-villain-filter")) {
        el("inp-villain-filter").blur();
      }
      return;
    }
    if (ev.key === "f" || ev.key === "F") {
      ev.preventDefault();
      if (el("inp-villain-filter")) el("inp-villain-filter").focus();
      return;
    }
    if (ev.key === "m" || ev.key === "M") {
      ev.preventDefault();
      if (el("btn-toggle-minimap")) el("btn-toggle-minimap").click();
      return;
    }
    if (!hasSolvedNode || !lastRenderedNode) return;

    if (ev.key >= "1" && ev.key <= "5") {
      const idx = parseInt(ev.key, 10) - 1;
      const tiles = document.querySelectorAll("#tree-action-buttons .gto-act-tile");
      if (idx < tiles.length) {
        ev.preventDefault();
        tiles[idx].click();
      }
    } else if (ev.key === "Backspace" || ev.key === "ArrowLeft") {
      const hist = lastRenderedNode.history || [];
      if (hist.length > 0) {
        ev.preventDefault();
        try {
          const n = await invokeCmd("rewind_to", { depth: hist.length - 1 });
          renderNode(n);
        } catch (e) {
          showError(e.message);
        }
      }
    } else if (ev.key === "r" || ev.key === "R") {
      ev.preventDefault();
      try {
        const n = await invokeCmd("reset_cursor", {});
        renderNode(n);
      } catch (e) {
        showError(e.message);
      }
    }
  });
  if (el("btn-unlock-this")) {
    el("btn-unlock-this").addEventListener("click", async () => {
      try {
        showError("");
        const iters = parseInt(el("inp-lock-iters") ? el("inp-lock-iters").value : "250", 10) || 0;
        const n = await invokeCmd("unlock_node", { all: false, resolve_iterations: iters });
        el("node-lock-panel").classList.add("hidden");
        renderNode(n);
      } catch (e) {
        showError(e.message);
      }
    });
  }
  if (el("btn-unlock-all")) {
    el("btn-unlock-all").addEventListener("click", async () => {
      try {
        showError("");
        const iters = parseInt(el("inp-lock-iters") ? el("inp-lock-iters").value : "250", 10) || 0;
        const n = await invokeCmd("unlock_node", { all: true, resolve_iterations: iters });
        el("node-lock-panel").classList.add("hidden");
        renderNode(n);
      } catch (e) {
        showError(e.message);
      }
    });
  }

  // #3: Runout Heatmap checkbox toggle
  if (el("chk-runout-heatmap")) {
    el("chk-runout-heatmap").addEventListener("change", () => {
      if (lastRenderedNode) renderNode(lastRenderedNode);
    });
  }

  // #6: Drill Mode toggle & reset
  if (el("btn-drill-toggle")) {
    el("btn-drill-toggle").addEventListener("click", () => {
      drillModeActive = !drillModeActive;
      el("btn-drill-toggle").classList.toggle("active-mode", drillModeActive);
      if (el("drill-banner")) el("drill-banner").classList.toggle("hidden", !drillModeActive);
      if (lastRenderedNode) renderNode(lastRenderedNode);
    });
  }
  if (el("btn-drill-reset")) {
    el("btn-drill-reset").addEventListener("click", () => {
      drillHandsPlayed = 0;
      drillTotalEvLoss = 0.0;
      updateDrillScoreUi();
      if (el("drill-feedback")) {
        el("drill-feedback").textContent =
          "Frequencies hidden when Hero acts — pick the action you would play in-game!";
      }
    });
  }

  // Settings Modal
  el("btn-open-settings").addEventListener("click", () => {
    el("settings-modal").classList.remove("hidden");
  });
  el("btn-close-settings").addEventListener("click", () => {
    el("settings-modal").classList.add("hidden");
  });
  el("btn-browse-folder").addEventListener("click", async () => {
    if (hasTauri && window.__TAURI__ && window.__TAURI__.dialog) {
      const picked = await window.__TAURI__.dialog.open({ directory: true, multiple: false });
      if (typeof picked === "string" && picked.length > 0) {
        el("inp-pack-folder").value = picked;
      }
    } else {
      el("inp-pack-folder").focus();
    }
  });
  el("btn-load-folder").addEventListener("click", async () => {
    const p = el("inp-pack-folder").value.trim();
    if (!p) return;
    try {
      const data = await invokeCmd("set_pack_folder", { path: p });
      userManuallyPickedSpot = false;
      populateSpotsState(data);
      el("chk-use-pack").checked = true;
      syncPackToggleUi();
      await refreshVillainPreview();
      el("settings-modal").classList.add("hidden");
    } catch (e) {
      showError(e.message);
      el("settings-modal").classList.add("hidden");
    }
  });

  async function persistBudgetRaw(rawStr) {
    try {
      const res = await invokeCmd("set_memory_budget", { budget: rawStr });
      if (res && res.memory_budget_mib) {
        syncBudgetInputs(Number(res.memory_budget_mib));
      }
      showError("");
    } catch (e) {
      showError(e.message);
    }
  }

  el("inp-budget").addEventListener("change", () => persistBudgetRaw(el("inp-budget").value));
  el("btn-save-budget").addEventListener("click", async () => {
    await persistBudgetRaw(el("inp-settings-budget").value);
    el("settings-modal").classList.add("hidden");
  });

  el("btn-apply-bind").addEventListener("click", async () => {
    const sid = el("inp-bind-spot-id").value.trim();
    const src = el("sel-bind-source").value;
    if (!sid) return;
    try {
      const data = await invokeCmd("bind_spot", { spot_id: sid, source_key: src });
      populateSpotsState(data);
      await refreshVillainPreview();
    } catch (e) {
      showError(e.message);
    }
  });

  // Save / Load Config JSON
  el("btn-save-cfg").addEventListener("click", async () => {
    try {
      const cfg = collectSpotConfig();
      const exported = await invokeCmd("export_config", cfg);
      const blob = new Blob([JSON.stringify(exported, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `plo4_hero_vs_range_${cfg.board || "spot"}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      showError(e.message);
    }
  });
  el("btn-load-cfg").addEventListener("click", () => el("file-load-cfg").click());
  el("file-load-cfg").addEventListener("change", async (ev) => {
    const file = ev.target.files && ev.target.files[0];
    if (!file) return;
    try {
      const txt = await file.text();
      const loadedCfg = await invokeCmd("import_config", { json_text: txt });
      applySpotConfigToForm(loadedCfg);
    } catch (e) {
      showError(e.message);
    } finally {
      ev.target.value = "";
    }
  });

  // Initial load
  try {
    const stateData = await invokeCmd("list_spots", {});
    populateSpotsState(stateData);
    if (!stateData.loaded) {
      el("chk-use-pack").checked = false;
      syncPackToggleUi();
    }
    await refreshVillainPreview();
  } catch (e) {
    showError(e.message);
  }
}

window.addEventListener("DOMContentLoaded", initApp);
