/* Road Defect Monitor – map, marker drawer and dashboard log. */
(() => {
  "use strict";

  const API = "/api/v1";

  const ENUMS = {
    defect_type: ["pothole", "bump", "crack", "unclassified"],
    severity: ["false_positive", "no_repair_needed", "low", "medium", "high", "critical"],
    repair_status: ["awaiting_repair", "in_progress", "completed"],
  };
  const LABELS = {
    pothole: "Pothole", bump: "Bump", crack: "Crack", unclassified: "Unclassified",
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
    pages: 1,
    mapItems: new Map(), // id -> lightweight incident
    rows: [],            // current page of the log table
    selectedId: null,
    current: null,       // full incident shown in drawer
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
    const res = await fetch(API + path, { headers: { "Content-Type": "application/json" }, ...opts });
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch { /* ignore */ }
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return res.status === 204 ? null : res.json();
  }

  function filterQuery() {
    const q = new URLSearchParams();
    const f = state.filters;
    if (f.start) q.set("start", new Date(f.start).toISOString());
    if (f.end) q.set("end", new Date(f.end).toISOString());
    for (const key of ["defect_type", "severity", "repair_status"]) for (const v of f[key]) q.append(key, v);
    return q;
  }

  // -------------------------------------------------------------------- map
  // Default service area (Incheon); replaced by /api/v1/config on startup.
  let serviceBounds = L.latLngBounds([37.30, 126.10], [37.82, 126.80]);
  const map = L.map("map", { zoomControl: true, maxBoundsViscosity: 1.0, maxZoom: 20 })
    .fitBounds(serviceBounds);

  function mapNotice(html) {
    const el = document.createElement("div");
    el.className = "map-notice";
    el.innerHTML = html;
    $(".map-wrap").appendChild(el);
  }

  function loadGoogleMaps(key) {
    return new Promise((resolve, reject) => {
      if (window.google?.maps) return resolve();
      const timer = setTimeout(() => reject(new Error("timed out loading Google Maps")), 15000);
      window.__rdmGoogleReady = () => { clearTimeout(timer); resolve(); };
      window.gm_authFailure = () => { // invalid / unauthorised key
        mapNotice("Google Maps rejected the API key – check <code>RDM_GOOGLE_MAPS_API_KEY</code> and its referrer restrictions.");
      };
      const s = document.createElement("script");
      s.src = `https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(key)}&loading=async&language=en&region=KR&callback=__rdmGoogleReady`;
      s.async = true;
      s.onerror = () => { clearTimeout(timer); reject(new Error("could not load Google Maps")); };
      document.head.appendChild(s);
    });
  }

  function useFallbackTiles(reason) {
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 20, maxNativeZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    }).addTo(map);
    mapNotice(`${esc(reason)} Showing OpenStreetMap instead.`);
  }

  async function setupBaseMap() {
    let cfg = { google_maps_api_key: "" };
    try {
      cfg = await api("/config");
      const a = cfg.service_area;
      serviceBounds = L.latLngBounds([a.south, a.west], [a.north, a.east]);
    } catch { /* keep defaults */ }

    // Lock the viewport to the service area.
    map.setMaxBounds(serviceBounds.pad(0.02));
    map.setMinZoom(map.getBoundsZoom(serviceBounds, false));
    if (state.firstLoad) map.fitBounds(serviceBounds);

    if (!cfg.google_maps_api_key) {
      useFallbackTiles("No Google Maps API key configured (set RDM_GOOGLE_MAPS_API_KEY).");
      return;
    }
    try {
      await loadGoogleMaps(cfg.google_maps_api_key);
    } catch (e) {
      useFallbackTiles(`Google Maps unavailable: ${e.message}.`);
      return;
    }
    const gOpts = { maxZoom: 20 };
    const roadmap = L.gridLayer.googleMutant({ ...gOpts, type: "roadmap" }).addTo(map);
    const satellite = L.gridLayer.googleMutant({ ...gOpts, type: "satellite" });
    const hybrid = L.gridLayer.googleMutant({ ...gOpts, type: "hybrid" });
    L.control.layers({ "Google Roadmap": roadmap, "Google Satellite": satellite, "Google Hybrid": hybrid },
      null, { position: "topright" }).addTo(map);
  }

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

  async function loadMap() {
    const data = await api(`/incidents/map?${filterQuery()}`);
    state.mapItems = new Map(data.items.map((i) => [i.id, i]));
    renderMap();
    if (state.firstLoad && data.items.length) {
      const inArea = data.items.filter((i) => serviceBounds.contains([i.lat, i.lng]));
      if (inArea.length) map.fitBounds(L.latLngBounds(inArea.map((i) => [i.lat, i.lng])).pad(0.1), { maxZoom: 17 });
      state.firstLoad = false;
    }
  }

  // ------------------------------------------------------------------ table
  async function loadTable() {
    const q = filterQuery();
    q.set("sort", state.sort.field);
    q.set("order", state.sort.order);
    q.set("page", state.page);
    q.set("page_size", state.pageSize);
    const data = await api(`/incidents?${q}`);
    if (data.items.length === 0 && state.page > 1 && data.total > 0) {
      state.page = data.pages;
      return loadTable();
    }
    state.rows = data.items;
    state.pages = data.pages;
    $("#log-total").textContent = `(${data.total})`;
    $("#pg-info").textContent = `Page ${data.page} of ${data.pages}`;
    $("#pg-prev").disabled = data.page <= 1;
    $("#pg-next").disabled = data.page >= data.pages;
    renderTable();
  }

  function renderTable() {
    const tbody = $("#log-table tbody");
    tbody.innerHTML = state.rows
      .map((r) => `
        <tr data-id="${esc(r.id)}" class="${r.id === state.selectedId ? "selected" : ""}">
          <td>${r.thumbnail_url ? `<img class="thumb" loading="lazy" src="${esc(r.thumbnail_url)}" alt="">` : ""}</td>
          <td class="coords">${fmtCoord(r.coordinates.lat, r.coordinates.lng)}</td>
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

  async function loadStats() {
    const s = await api("/stats");
    const open = (s.by_repair_status.awaiting_repair || 0) + (s.by_repair_status.in_progress || 0);
    const crit = (s.by_severity.critical || 0) + (s.by_severity.high || 0);
    $("#stats").innerHTML =
      `<span><b>${s.incidents}</b> incidents</span><span><b>${open}</b> open</span>` +
      `<span><b>${crit}</b> high/critical</span><span><b>${s.packets}</b> reports</span><span><b>${s.devices}</b> devices</span>`;
  }

  async function refreshAll() {
    try {
      await Promise.all([loadMap(), loadTable(), loadStats()]);
    } catch (e) {
      toast(`Failed to load data: ${e.message}`, true);
    }
  }

  // ----------------------------------------------------------------- drawer
  async function selectIncident(id, { pan }) {
    const prev = state.selectedId;
    state.selectedId = id;
    state.editing = false;
    if (prev) refreshMarker(prev);
    refreshMarker(id);
    document.querySelectorAll("#log-table tbody tr").forEach((tr) => tr.classList.toggle("selected", tr.dataset.id === id));

    if (pan) {
      const item = state.mapItems.get(id) || state.rows.find((r) => r.id === id);
      if (item) {
        const ll = item.lat !== undefined ? [item.lat, item.lng] : [item.coordinates.lat, item.coordinates.lng];
        map.flyTo(ll, Math.max(map.getZoom(), 18), { duration: 0.8 });
      }
    }
    try {
      state.current = await api(`/incidents/${encodeURIComponent(id)}`);
      if (state.selectedId === id) renderDrawer();
    } catch (e) {
      toast(`Could not load incident: ${e.message}`, true);
    }
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
    const images = inc.images || [];
    const devices = new Set(images.map((i) => i.device_id)).size;

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
        ${images.length > 1 ? `<div class="thumbs" id="thumbs">${images.map((im, i) => `<img src="${esc(im.url)}" data-i="${i}" alt="">`).join("")}</div>` : ""}`
        : `<div class="muted">No photos.</div>`}
      </div>

      <div class="section" id="sec-time">
        <div class="section-label">Timestamp</div>
        <dl class="kv">
          <dt>Detected</dt><dd>${esc(fmtDate(inc.created_at))}</dd>
          ${inc.packet_count > 1 ? `<dt>Latest report</dt><dd>${esc(fmtDate(inc.last_report_at))}</dd>` : ""}
          <dt>Reports</dt><dd>${inc.report_count} device${inc.report_count === 1 ? "" : "s"} · ${inc.packet_count} packet${inc.packet_count === 1 ? "" : "s"}</dd>
          <dt>Location</dt><dd class="coords">${fmtCoord(inc.coordinates.lat, inc.coordinates.lng)}</dd>
          ${inc.peak_intensity != null ? `<dt>Peak vibration</dt><dd>${inc.peak_intensity.toFixed(2)} g</dd>` : ""}
        </dl>
      </div>

      <div class="section" id="sec-type"><div class="section-label">Defect type</div>${editable("defect_type", inc.defect_type)}</div>
      <div class="section" id="sec-sev"><div class="section-label">Severity level</div>${editable("severity", inc.severity)}</div>
      <div class="section" id="sec-status"><div class="section-label">Repair status</div>${editable("repair_status", inc.repair_status)}</div>

      <div class="section" id="sec-actions">
        <div class="actions">
          <button class="btn" id="act-edit" type="button">${state.editing ? "Cancel" : "Edit"}</button>
          <button class="btn primary" id="act-save" type="button" ${state.editing ? "" : "disabled"}>Save Changes</button>
          <button class="btn danger" id="act-delete" type="button">Delete Incident</button>
        </div>
      </div>`;

    if (images.length) setupCarousel(images, devices);
    $("#act-edit").onclick = () => { state.editing = !state.editing; renderDrawer(); };
    $("#act-save").onclick = saveChanges;
    $("#act-delete").onclick = deleteIncident;
    $("#drawer").classList.add("open");
    $("#drawer").setAttribute("aria-hidden", "false");
  }

  function setupCarousel(images) {
    let idx = 0;
    const show = (i) => {
      idx = (i + images.length) % images.length;
      const im = images[idx];
      $("#slide").src = im.url;
      $("#cap-device").textContent = `${im.device_id} · ${fmtDate(im.timestamp)}`;
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

  // ------------------------------------------------- optimistic write paths
  function applyLocal(id, changes) {
    if (state.current?.id === id) Object.assign(state.current, changes);
    const mi = state.mapItems.get(id);
    if (mi) Object.assign(mi, changes);
    const row = state.rows.find((r) => r.id === id);
    if (row) Object.assign(row, changes);
    refreshMarker(id);
    renderTable();
    renderMap(); // heat weights depend on severity/status
  }

  async function saveChanges() {
    const inc = state.current;
    const changes = {};
    document.querySelectorAll("#drawer-body select[data-field]").forEach((s) => {
      if (s.value !== inc[s.dataset.field]) changes[s.dataset.field] = s.value;
    });
    state.editing = false;
    if (!Object.keys(changes).length) { renderDrawer(); return; }

    const before = Object.fromEntries(Object.keys(changes).map((k) => [k, inc[k]]));
    applyLocal(inc.id, changes);
    renderDrawer();
    state.pendingOps++;
    try {
      const updated = await api(`/incidents/${encodeURIComponent(inc.id)}`, { method: "PATCH", body: JSON.stringify(changes) });
      if (state.current?.id === updated.id) { state.current = updated; renderDrawer(); }
      toast("Changes saved");
    } catch (e) {
      applyLocal(inc.id, before);
      if (state.current?.id === inc.id) renderDrawer();
      toast(`Save failed: ${e.message}`, true);
    } finally {
      state.pendingOps--;
      loadStats().catch(() => {});
    }
  }

  async function deleteIncident() {
    const inc = state.current;
    if (!inc || !confirm(`Delete this incident and its ${inc.packet_count} report(s)? This cannot be undone.`)) return;
    const snapshotMap = state.mapItems.get(inc.id);
    const snapshotRows = state.rows.slice();
    state.mapItems.delete(inc.id);
    state.rows = state.rows.filter((r) => r.id !== inc.id);
    closeDrawer();
    renderMap();
    renderTable();
    state.pendingOps++;
    try {
      await api(`/incidents/${encodeURIComponent(inc.id)}`, { method: "DELETE" });
      toast("Incident deleted");
      loadTable().catch(() => {});
      loadStats().catch(() => {});
    } catch (e) {
      if (snapshotMap) state.mapItems.set(inc.id, snapshotMap);
      state.rows = snapshotRows;
      renderMap();
      renderTable();
      toast(`Delete failed: ${e.message}`, true);
    } finally {
      state.pendingOps--;
    }
  }

  // ------------------------------------------------------------ live events
  let liveTimer = null;
  const touched = new Set();
  function scheduleLiveRefresh(evt) {
    touched.add(evt.id);
    if (evt.type === "incident.deleted" && state.selectedId === evt.id && !state.editing) {
      closeDrawer();
      toast("The open incident was deleted");
    }
    clearTimeout(liveTimer);
    liveTimer = setTimeout(async function run() {
      if (state.pendingOps > 0) { liveTimer = setTimeout(run, 500); return; }
      const ids = new Set(touched);
      touched.clear();
      await refreshAll();
      if (state.selectedId && ids.has(state.selectedId) && !state.editing) {
        try {
          state.current = await api(`/incidents/${encodeURIComponent(state.selectedId)}`);
          renderDrawer();
        } catch { /* deleted elsewhere */ }
      }
    }, 600);
  }

  function connectEvents() {
    const live = $("#live-indicator");
    const es = new EventSource(`${API}/events`);
    es.onopen = () => { live.classList.add("on"); live.querySelector(".label").textContent = "live"; };
    es.onerror = () => { live.classList.remove("on"); live.querySelector(".label").textContent = "reconnecting…"; };
    es.onmessage = (m) => { try { scheduleLiveRefresh(JSON.parse(m.data)); } catch { /* ignore */ } };
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
    filterTimer = setTimeout(refreshAll, 200);
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
        loadTable().catch((e) => toast(e.message, true));
      }));
    $("#log-table tbody").addEventListener("click", (e) => {
      const tr = e.target.closest("tr[data-id]");
      if (tr) selectIncident(tr.dataset.id, { pan: true });
    });
    $("#pg-prev").addEventListener("click", () => { state.page--; loadTable().catch((e) => toast(e.message, true)); });
    $("#pg-next").addEventListener("click", () => { state.page++; loadTable().catch((e) => toast(e.message, true)); });
    $("#page-size").addEventListener("change", (e) => { state.pageSize = +e.target.value; state.page = 1; loadTable().catch((er) => toast(er.message, true)); });

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

  wire();
  renderLegend();
  setupBaseMap();
  refreshAll();
  connectEvents();
})();
