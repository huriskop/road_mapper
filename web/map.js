/* Map page: pins, heatmap, filters, incident log and details. All incidents are loaded at once from
   /api/incidents; filtering and sorting happen here in the browser. */
(() => {
  "use strict";

  const ENUMS = {
    defect_type: ["pothole", "bump", "crack", "unclassified"],
    severity: ["false_positive", "no_repair_needed", "low", "medium", "high", "critical"],
    repair_status: ["awaiting_repair", "in_progress", "completed"],
  };
  const LABELS = {
    pothole: "Pothole", bump: "Speed bump", crack: "Crack", unclassified: "Unclassified",
    false_positive: "False Positive", no_repair_needed: "No Repair Needed", low: "Low",
    medium: "Medium", high: "High", critical: "Critical",
    awaiting_repair: "Awaiting Repair", in_progress: "In Progress", completed: "Completed",
  };
  const SEV_WEIGHT = { false_positive: 0, no_repair_needed: 0.15, low: 0.35, medium: 0.6, high: 0.85, critical: 1 };
  const STATUS_URGENCY = { completed: 0, in_progress: 1, awaiting_repair: 2 };

  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const colorFor = (field, value) => cssVar(field === "severity" ? `--sev-${value}` : `--st-${value}`);

  const state = {
    filters: { start: "", end: "", defect_type: new Set(), severity: new Set(), repair_status: new Set() },
    sort: { field: "created_at", order: "desc" },
    page: 1,
    pageSize: 20,
    rev: 0,
    all: [],             // every incident
    mapItems: new Map(), // id -> incident, after filters
    rows: [],            // current page of the log table
    selectedId: null,
    current: null,       // incident shown in drawer
    editing: false,
    colorBy: "severity",
    pendingOps: 0,
    firstLoad: true,
  };

  // ------------------------------------------------------------------ utils
  const $ = (sel, root = document) => root.querySelector(sel);
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmtDate = (iso) => (iso ? new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "medium" }) : "—");
  const fmtCoord = (lat, lng) => `${lat.toFixed(6)}, ${lng.toFixed(6)}`;
  const label = (v) => LABELS[v] || v;
  const badge = (field, value) => {
    const cls = field === "severity" ? `sev sev-${value}` : field === "repair_status" ? "st" : "";
    const style = field === "defect_type" ? "" : ` style="background:${colorFor(field, value)}"`;
    return `<span class="badge ${cls}"${style}>${esc(label(value))}</span>`;
  };

  let toastTimer;
  function toast(msg, isError = false) {
    const el = $("#toast");
    el.textContent = msg;
    el.className = `toast show${isError ? " error" : ""}`;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (el.className = "toast"), 3200);
  }

  async function api(path, opts = {}) {
    const res = await fetch("/api" + path, { headers: { "Content-Type": "application/json" }, ...opts });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || res.statusText);
    return data;
  }

  function matches(i) {
    const f = state.filters, t = Date.parse(i.created_at);
    if (f.start && t < new Date(f.start).getTime()) return false;
    if (f.end && t > new Date(f.end).getTime()) return false;
    return ["defect_type", "severity", "repair_status"].every((k) => !f[k].size || f[k].has(i[k]));
  }

  // -------------------------------------------------------------------- map
  const map = L.map("map", { zoomControl: true, maxZoom: 20 }).setView([37.4563, 126.7052], 12);   // Incheon
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 20, maxNativeZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  }).addTo(map);

  const markerLayer = L.markerClusterGroup({
    disableClusteringAtZoom: 18,
    maxClusterRadius: 50,
    showCoverageOnHover: false,
    iconCreateFunction(cluster) {
      const items = cluster.getAllChildMarkers().map((m) => m.options.incident);
      let worst;
      if (state.colorBy === "severity") {
        worst = items.reduce((a, b) => (SEV_WEIGHT[b.severity] > SEV_WEIGHT[a.severity] ? b : a)).severity;
      } else {
        worst = items.reduce((a, b) => (STATUS_URGENCY[b.repair_status] > STATUS_URGENCY[a.repair_status] ? b : a)).repair_status;
      }
      const n = cluster.getChildCount();
      const size = n < 10 ? 34 : n < 50 ? 42 : 50;
      return L.divIcon({
        html: `<div class="cluster" style="width:${size}px;height:${size}px;background:${colorFor(state.colorBy, worst)}">${n}</div>`,
        className: "", iconSize: [size, size],
      });
    },
  });
  const heatLayer = L.heatLayer([], { radius: 28, blur: 18, maxZoom: 17, minOpacity: 0.25 });
  const markers = new Map(); // id -> L.Marker
  markerLayer.addTo(map);

  function pinIcon(item) {
    const color = colorFor(state.colorBy, item[state.colorBy]);
    const sel = item.id === state.selectedId ? " selected" : "";
    const txt = item.report_count > 1 ? item.report_count : "";
    return L.divIcon({
      html: `<div class="pin${sel}" style="background:${color}">${txt}</div>`,
      className: "", iconSize: [22, 22], iconAnchor: [11, 11],
    });
  }

  function renderMap() {
    markerLayer.clearLayers();
    markers.clear();
    const list = [];
    const heat = [];
    for (const item of state.mapItems.values()) {
      const m = L.marker([item.lat, item.lng], { icon: pinIcon(item), incident: item, title: `${label(item.defect_type)} · ${label(item.severity)}` });
      m.on("click", () => selectIncident(item.id, { pan: false }));
      markers.set(item.id, m);
      list.push(m);
      // Heat weight: severity scaled by report frequency (log-damped); resolved defects fade out.
      const resolved = item.repair_status === "completed" ? 0.25 : 1;
      const w = SEV_WEIGHT[item.severity] * resolved * (1 + Math.log2(item.report_count || 1) * 0.5);
      if (w > 0) heat.push([item.lat, item.lng, Math.min(w, 2)]);
    }
    markerLayer.addLayers(list);
    heatLayer.setLatLngs(heat);
    heatLayer.setOptions({ max: 1.5 });
  }

  function refreshMarker(id) {
    const m = markers.get(id);
    const item = state.mapItems.get(id);
    if (!m || !item) return;
    m.options.incident = item;
    m.setIcon(pinIcon(item));
    markerLayer.refreshClusters(m);
  }

  function renderLegend() {
    $("#legend").innerHTML = ENUMS[state.colorBy]
      .map((v) => `<span><i class="swatch" style="background:${colorFor(state.colorBy, v)}"></i>${esc(label(v))}</span>`)
      .join("");
  }

  function loadMap() {
    state.mapItems = new Map(state.all.filter(matches).map((i) => [i.id, i]));
    renderMap();
    if (state.firstLoad && state.mapItems.size) {
      map.fitBounds(L.latLngBounds([...state.mapItems.values()].map((i) => [i.lat, i.lng])).pad(0.1), { maxZoom: 17 });
      state.firstLoad = false;
    }
  }

  // ------------------------------------------------------------------ table
  function sortKey(i) {
    const f = state.sort.field;
    return ENUMS[f] ? ENUMS[f].indexOf(i[f]) : f === "created_at" ? Date.parse(i.created_at) : i[f];
  }

  function loadTable() {
    const dir = state.sort.order === "asc" ? 1 : -1;
    const list = [...state.mapItems.values()].sort((a, b) => dir * (sortKey(a) - sortKey(b)));
    const pages = Math.max(1, Math.ceil(list.length / state.pageSize));
    state.page = Math.min(Math.max(state.page, 1), pages);
    state.rows = list.slice((state.page - 1) * state.pageSize, state.page * state.pageSize);
    $("#log-total").textContent = `(${list.length})`;
    $("#pg-info").textContent = `Page ${state.page} of ${pages}`;
    $("#pg-prev").disabled = state.page <= 1;
    $("#pg-next").disabled = state.page >= pages;
    renderTable();
  }

  const thumb = (i) => photos(i).at(-1)?.image;
  const photos = (i) => i.reports.filter((r) => r.image);

  function renderTable() {
    const tbody = $("#log-table tbody");
    tbody.innerHTML = state.rows
      .map((r) => `
        <tr data-id="${esc(r.id)}" class="${r.id === state.selectedId ? "selected" : ""}">
          <td>${thumb(r) ? `<img class="thumb" loading="lazy" src="${esc(thumb(r))}" alt="">` : ""}</td>
          <td class="coords">${fmtCoord(r.lat, r.lng)}</td>
          <td>${esc(fmtDate(r.created_at))}${r.report_count > 1 ? `<div class="muted">latest ${esc(fmtDate(r.last_report_at))}</div>` : ""}</td>
          <td>${badge("defect_type", r.defect_type)}</td>
          <td>${badge("severity", r.severity)}</td>
          <td>${badge("repair_status", r.repair_status)}</td>
          <td class="num">${r.report_count}</td>
        </tr>`)
      .join("");
    $("#log-empty").hidden = state.rows.length > 0;
    document.querySelectorAll("#log-table th.sortable").forEach((th) => {
      th.classList.toggle("sort-asc", th.dataset.sort === state.sort.field && state.sort.order === "asc");
      th.classList.toggle("sort-desc", th.dataset.sort === state.sort.field && state.sort.order === "desc");
    });
  }

  function loadStats() {
    const n = (f) => state.all.filter(f).length;
    const reports = state.all.reduce((sum, i) => sum + i.report_count, 0);
    $("#stats").innerHTML =
      `<span><b>${state.all.length}</b> incidents</span><span><b>${n((i) => i.repair_status !== "completed")}</b> open</span>` +
      `<span><b>${n((i) => i.severity === "high" || i.severity === "critical")}</b> high/critical</span><span><b>${reports}</b> reports</span>`;
  }

  function showAll() { loadMap(); loadTable(); loadStats(); }

  // ----------------------------------------------------------------- drawer
  function selectIncident(id, { pan }) {
    const prev = state.selectedId;
    state.selectedId = id;
    state.editing = false;
    if (prev) refreshMarker(prev);
    refreshMarker(id);
    document.querySelectorAll("#log-table tbody tr").forEach((tr) => tr.classList.toggle("selected", tr.dataset.id === id));

    state.current = state.all.find((i) => i.id === id) || null;
    if (!state.current) return toast("That incident is no longer on the map", true);
    if (pan) map.flyTo([state.current.lat, state.current.lng], Math.max(map.getZoom(), 18), { duration: 0.8 });
    renderDrawer();
  }

  function closeDrawer() {
    const prev = state.selectedId;
    state.selectedId = null;
    state.current = null;
    state.editing = false;
    $("#drawer").classList.remove("open");
    $("#drawer").setAttribute("aria-hidden", "true");
    if (prev) refreshMarker(prev);
    document.querySelectorAll("#log-table tbody tr.selected").forEach((tr) => tr.classList.remove("selected"));
  }

  function editable(field, value) {
    if (!state.editing) return badge(field, value);
    // Defect Type dropdown offers Pothole/Bump/Crack; "Unclassified" only while still unclassified.
    const opts = field === "defect_type" && value !== "unclassified" ? ENUMS.defect_type.filter((v) => v !== "unclassified") : ENUMS[field];
    return `<select data-field="${field}">${opts
      .map((v) => `<option value="${v}"${v === value ? " selected" : ""}>${esc(label(v))}</option>`)
      .join("")}</select>`;
  }

  function renderDrawer() {
    const inc = state.current;
    if (!inc) return;
    const images = photos(inc);

    // Strict vertical order: Gallery → Timestamp → Defect Type → Severity → Repair Status → Actions
    $("#drawer-body").innerHTML = `
      <div class="section" id="sec-gallery">
        <div class="section-label">Media gallery (${images.length} photo${images.length === 1 ? "" : "s"})</div>
        ${images.length ? `
        <div class="carousel" id="carousel">
          <img class="slide" id="slide" alt="Road surface photo">
          ${images.length > 1 ? `<button class="nav prev" type="button" aria-label="Previous photo">‹</button><button class="nav next" type="button" aria-label="Next photo">›</button>` : ""}
          <div class="caption"><span id="cap-device"></span><span id="cap-index"></span></div>
        </div>
        <div class="camera-found" id="cap-camera"></div>
        ${images.length > 1 ? `<div class="thumbs" id="thumbs">${images.map((im, i) => `<img src="${esc(im.image)}" data-i="${i}" alt="">`).join("")}</div>` : ""}`
        : `<div class="muted">No photos.</div>`}
      </div>

      <div class="section" id="sec-time">
        <div class="section-label">Timestamp</div>
        <dl class="kv">
          <dt>Detected</dt><dd>${esc(fmtDate(inc.created_at))}</dd>
          ${inc.report_count > 1 ? `<dt>Latest report</dt><dd>${esc(fmtDate(inc.last_report_at))}</dd>` : ""}
          <dt>Reports</dt><dd>${inc.report_count}</dd>
          <dt>Location</dt><dd class="coords">${fmtCoord(inc.lat, inc.lng)}${inc.reports.some((r) => r.location !== "gps") ? " (not from GPS)" : ""}</dd>
          ${inc.peak_g != null ? `<dt>Peak vibration</dt><dd>${inc.peak_g.toFixed(2)} g</dd>` : ""}
        </dl>
      </div>

      <div class="section" id="sec-type"><div class="section-label">Defect type</div>${editable("defect_type", inc.defect_type)}
        ${inc.type_source ? `<div class="type-source">${esc(TYPE_SOURCE[inc.type_source] || inc.type_source)}${cameraSummary(images)}</div>` : ""}</div>
      <div class="section" id="sec-sev"><div class="section-label">Severity level</div>${editable("severity", inc.severity)}</div>
      <div class="section" id="sec-status"><div class="section-label">Repair status</div>${editable("repair_status", inc.repair_status)}</div>

      <div class="section" id="sec-actions">
        <div class="actions">
          <button class="btn" id="act-edit" type="button">${state.editing ? "Cancel" : "Edit"}</button>
          <button class="btn primary" id="act-save" type="button" ${state.editing ? "" : "disabled"}>Save Changes</button>
          <button class="btn danger" id="act-delete" type="button">Delete Incident</button>
        </div>
      </div>`;

    if (images.length) setupCarousel(images);
    $("#act-edit").onclick = () => { state.editing = !state.editing; renderDrawer(); };
    $("#act-save").onclick = saveChanges;
    $("#act-delete").onclick = deleteIncident;
    $("#drawer").classList.add("open");
    $("#drawer").setAttribute("aria-hidden", "false");
  }

  const TYPE_SOURCE = {
    camera: "Classified by the camera",
    sensor: "Classified from vibration (no camera confirmation)",
    operator: "Set by hand",
  };
  const CAMERA_TYPE = { pothole: "pothole", speed_bump: "bump" };
  const REASON = { down: "Jolt down", up: "Jolt up", upload: "Uploaded photo" };

  // What the photo detector found in one photo, e.g. "Pothole 89% · 3 cracks"
  function cameraText(im) {
    const v = im.camera;
    if (!v) return "Camera: not checked";
    const parts = [];
    if (v.potholes) parts.push(`Pothole${v.potholes > 1 ? ` ×${v.potholes}` : ""}${v.pothole_confidence != null ? ` ${Math.round(v.pothole_confidence * 100)}%` : ""}`);
    if (v.speed_bumps) parts.push(`Speed bump${v.speed_bump_confidence != null ? ` ${Math.round(v.speed_bump_confidence * 100)}%` : ""}`);
    if (v.cracks) parts.push(`${v.cracks} crack mark${v.cracks > 1 ? "s" : ""}`);
    return "Camera: " + (parts.join(" · ") || "nothing found");
  }

  // e.g. " · 3 of 4 photos show a pothole"
  function cameraSummary(images) {
    const verdicts = images.map((im) => CAMERA_TYPE[im.camera?.verdict]).filter(Boolean);
    if (!verdicts.length) return "";
    const counts = {};
    verdicts.forEach((t) => (counts[t] = (counts[t] || 0) + 1));
    const [top, n] = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
    return ` · ${n} of ${images.length} photo${images.length === 1 ? "" : "s"} show${images.length === 1 ? "s" : ""} a ${label(top).toLowerCase()}`;
  }

  function setupCarousel(images) {
    let idx = 0;
    const show = (i) => {
      idx = (i + images.length) % images.length;
      const im = images[idx];
      $("#slide").src = im.image;
      $("#cap-camera").textContent = cameraText(im);
      $("#cap-device").textContent = `${REASON[im.reason] || im.reason} · ${fmtDate(im.time)}`;
      $("#cap-index").textContent = `${idx + 1}/${images.length}`;
      document.querySelectorAll("#thumbs img").forEach((t) => t.classList.toggle("active", +t.dataset.i === idx));
    };
    show(0);
    const car = $("#carousel");
    car.querySelector(".prev")?.addEventListener("click", () => show(idx - 1));
    car.querySelector(".next")?.addEventListener("click", () => show(idx + 1));
    $("#thumbs")?.addEventListener("click", (e) => { if (e.target.dataset.i) show(+e.target.dataset.i); });
    // swipe support
    let x0 = null;
    car.addEventListener("pointerdown", (e) => (x0 = e.clientX));
    car.addEventListener("pointerup", (e) => {
      if (x0 === null || images.length < 2) return;
      const dx = e.clientX - x0;
      if (Math.abs(dx) > 40) show(idx + (dx < 0 ? 1 : -1));
      x0 = null;
    });
  }

  // ------------------------------------------------------------------ edits
  async function write(id, opts, done) {
    state.pendingOps++;
    try {
      await api(`/incidents/${encodeURIComponent(id)}`, opts);
      toast(done);
    } catch (e) {
      toast(`Failed: ${e.message}`, true);
    } finally {
      state.pendingOps--;
    }
    await refresh();
  }

  async function saveChanges() {
    const inc = state.current;
    const changes = {};
    document.querySelectorAll("#drawer-body select[data-field]").forEach((s) => {
      if (s.value !== inc[s.dataset.field]) changes[s.dataset.field] = s.value;
    });
    state.editing = false;
    if (Object.keys(changes).length) await write(inc.id, { method: "PATCH", body: JSON.stringify(changes) }, "Changes saved");
    renderDrawer();
  }

  async function deleteIncident() {
    const inc = state.current;
    if (!inc || !confirm(`Delete this incident and its ${inc.report_count} report(s) from the map? This cannot be undone.`)) return;
    closeDrawer();
    await write(inc.id, { method: "DELETE" }, "Incident deleted");
  }

  // ------------------------------------------------------------ live updates
  async function refresh() {
    const live = $("#live-indicator");
    try {
      const data = await api(`/incidents?rev=${state.rev}`);
      live.classList.add("on");
      live.querySelector(".label").textContent = "live";
      if (!data.items || state.pendingOps) return;
      state.rev = data.rev;
      state.all = data.items;
      showAll();
      if (state.selectedId && !state.editing) {
        state.current = state.all.find((i) => i.id === state.selectedId) || null;
        if (state.current) renderDrawer(); else { closeDrawer(); toast("The open incident was deleted"); }
      }
    } catch {
      live.classList.remove("on");
      live.querySelector(".label").textContent = "reconnecting…";
    }
  }

  // ------------------------------------------------------------ wiring / UI
  function buildFilterGroups() {
    for (const key of ["defect_type", "severity", "repair_status"]) {
      const fs = $(`#f-${key}`);
      fs.insertAdjacentHTML("beforeend", ENUMS[key]
        .map((v) => `<label class="check"><input type="checkbox" value="${v}"> ${esc(label(v))}</label>`).join(""));
      fs.addEventListener("change", (e) => {
        const set = state.filters[key];
        e.target.checked ? set.add(e.target.value) : set.delete(e.target.value);
        onFiltersChanged();
      });
    }
  }

  let filterTimer;
  function onFiltersChanged() {
    state.page = 1;
    clearTimeout(filterTimer);
    filterTimer = setTimeout(showAll, 200);
  }

  function wire() {
    buildFilterGroups();
    $("#f-start").addEventListener("change", (e) => { state.filters.start = e.target.value; onFiltersChanged(); });
    $("#f-end").addEventListener("change", (e) => { state.filters.end = e.target.value; onFiltersChanged(); });
    $("#f-reset").addEventListener("click", () => {
      state.filters = { start: "", end: "", defect_type: new Set(), severity: new Set(), repair_status: new Set() };
      $("#f-start").value = $("#f-end").value = "";
      document.querySelectorAll(".multi input").forEach((i) => (i.checked = false));
      onFiltersChanged();
    });

    $("#layer-markers").addEventListener("change", (e) => (e.target.checked ? map.addLayer(markerLayer) : map.removeLayer(markerLayer)));
    $("#layer-heat").addEventListener("change", (e) => (e.target.checked ? map.addLayer(heatLayer) : map.removeLayer(heatLayer)));
    $("#color-by").addEventListener("change", (e) => { state.colorBy = e.target.value; renderLegend(); renderMap(); });

    document.querySelectorAll("#log-table th.sortable").forEach((th) =>
      th.addEventListener("click", () => {
        const f = th.dataset.sort;
        state.sort = state.sort.field === f
          ? { field: f, order: state.sort.order === "desc" ? "asc" : "desc" }
          : { field: f, order: "desc" };
        state.page = 1;
        loadTable();
      }));
    $("#log-table tbody").addEventListener("click", (e) => {
      const tr = e.target.closest("tr[data-id]");
      if (tr) selectIncident(tr.dataset.id, { pan: true });
    });
    $("#pg-prev").addEventListener("click", () => { state.page--; loadTable(); });
    $("#pg-next").addEventListener("click", () => { state.page++; loadTable(); });
    $("#page-size").addEventListener("change", (e) => { state.pageSize = +e.target.value; state.page = 1; loadTable(); });

    // Hide / show the incident log (remembered per browser).
    const setLogHidden = (hidden) => {
      $(".layout").classList.toggle("log-hidden", hidden);
      $("#log-show").hidden = !hidden;
      try { localStorage.setItem("rdm.logHidden", hidden ? "1" : "0"); } catch { /* storage unavailable */ }
      setTimeout(() => map.invalidateSize(), 0);
    };
    $("#log-hide").addEventListener("click", () => setLogHidden(true));
    $("#log-show").addEventListener("click", () => setLogHidden(false));
    let startHidden = false;
    try { startHidden = localStorage.getItem("rdm.logHidden") === "1"; } catch { /* ignore */ }
    if (startHidden) setLogHidden(true);

    $("#drawer-close").addEventListener("click", closeDrawer);
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && state.selectedId) closeDrawer(); });
  }

  // Open an incident from a link such as /#incident-<id> (used by the photo test page).
  function openFromHash() {
    const m = /^#incident-([0-9a-f-]{36})$/i.exec(location.hash);
    if (m) selectIncident(m[1], { pan: true });
  }

  wire();
  renderLegend();
  refresh().then(openFromHash);
  window.addEventListener("hashchange", openFromHash);
  setInterval(refresh, 2000);
})();
