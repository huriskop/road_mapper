// Shared by index.html and capture.html: live connection, camera view, MPU-6050 sphere, GPS panel.
"use strict";

const Bench = (() => {
  const $ = (id) => document.getElementById(id);
  const handlers = { state: [], capture: [] };
  const D2R = Math.PI / 180;

  function on(type, fn) { handlers[type].push(fn); }

  async function post(path, body) {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || res.statusText);
    return data;
  }

  function setBadge(el, dev, realName) {
    if (!el) return;
    let cls = "ok", text = realName;
    if (!dev.ok) { cls = "bad"; text = "error"; }
    el.className = "badge " + cls;
    el.textContent = text;
  }

  function setText(id, text) { const el = $(id); if (el && el.textContent !== text) el.textContent = text; }
  const fmt = (v, d = 2, unit = "") => (v === null || v === undefined ? "—" : Number(v).toFixed(d) + unit);

  // ------------------------------------------------------------------ canvas helper
  function fitCanvas(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w: canvas.clientWidth, h: canvas.clientHeight };
  }
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  // ------------------------------------------------------------------ camera
  let camStarted = false;
  function startCamera() {
    const img = $("camImg");
    if (!img || camStarted) return;
    camStarted = true;
    const load = () => { img.src = "/api/camera.mjpg?" + Date.now(); };
    img.onerror = () => setTimeout(load, 2000);
    load();
  }
  function flashCamera() {
    const f = document.querySelector(".camview .flash");
    if (!f) return;
    f.classList.add("on");
    requestAnimationFrame(() => requestAnimationFrame(() => f.classList.remove("on")));
  }

  // ------------------------------------------------------------------ sphere
  // Body frame: X forward, Y left, Z up. Rotation body -> world is Z(yaw) * Y(pitch) * X(roll).
  function rotation(roll, pitch, yaw) {
    const [cr, sr] = [Math.cos(roll * D2R), Math.sin(roll * D2R)];
    const [cp, sp] = [Math.cos(pitch * D2R), Math.sin(pitch * D2R)];
    const [cy, sy] = [Math.cos(yaw * D2R), Math.sin(yaw * D2R)];
    return [
      [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
      [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
      [-sp, cp * sr, cp * cr],
    ];
  }
  const mul = (m, p) => [
    m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2],
    m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2],
    m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2],
  ];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = (a) => { const l = Math.hypot(...a); return a.map((v) => v / l); };

  // Viewer behind the vehicle, a little to the right and above.
  const VIEW = (() => {
    const az = 215 * D2R, el = 22 * D2R;
    const d = [Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el)];
    const r = norm(cross([0, 0, 1], d));
    return { d, r, u: cross(d, r) };
  })();
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

  function circle(fn, n = 72) { const pts = []; for (let i = 0; i <= n; i++) pts.push(fn((i / n) * 2 * Math.PI)); return pts; }

  function drawSphere(canvas, imu) {
    const { ctx, w, h } = fitCanvas(canvas);
    ctx.clearRect(0, 0, w, h);
    const R = Math.min(w, h) * 0.32, cx = w / 2, cy = h / 2 + 4;
    const proj = (p) => [cx + dot(p, VIEW.r) * R, cy - dot(p, VIEW.u) * R, dot(p, VIEW.d)];
    const M = rotation(imu.roll, imu.pitch, imu.yaw);

    // silhouette
    const g = ctx.createRadialGradient(cx - R * 0.35, cy - R * 0.4, R * 0.1, cx, cy, R);
    g.addColorStop(0, "rgba(90,169,255,0.22)");
    g.addColorStop(1, "rgba(90,169,255,0.03)");
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.arc(cx, cy, R, 0, 2 * Math.PI); ctx.fill();
    ctx.strokeStyle = css("--line"); ctx.lineWidth = 1; ctx.stroke();

    // polyline split into back (faint) and front (bright) halves
    function line(points, color, width, world = false) {
      const pr = points.map((p) => proj(world ? p : mul(M, p)));
      for (const front of [false, true]) {
        ctx.beginPath();
        for (let i = 1; i < pr.length; i++) {
          if ((pr[i - 1][2] + pr[i][2] > 0) !== front) continue;
          ctx.moveTo(pr[i - 1][0], pr[i - 1][1]);
          ctx.lineTo(pr[i][0], pr[i][1]);
        }
        ctx.strokeStyle = color;
        ctx.globalAlpha = front ? 1 : 0.22;
        ctx.lineWidth = width;
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }

    // level reference: the world horizon (dashed, does not move)
    ctx.setLineDash([4, 4]);
    line(circle((a) => [1.12 * Math.cos(a), 1.12 * Math.sin(a), 0]), css("--muted"), 1, true);
    ctx.setLineDash([]);

    const grid = "rgba(200,215,235,0.55)";
    for (const lat of [-60, -30, 30, 60]) {
      const z = Math.sin(lat * D2R), rr = Math.cos(lat * D2R);
      line(circle((a) => [rr * Math.cos(a), rr * Math.sin(a), z]), grid, 1);
    }
    for (let lon = 0; lon < 180; lon += 30) {
      const c = Math.cos(lon * D2R), s = Math.sin(lon * D2R);
      line(circle((a) => [Math.cos(a) * c, Math.cos(a) * s, Math.sin(a)]), grid, 1);
    }
    line(circle((a) => [Math.cos(a), Math.sin(a), 0]), css("--accent"), 2.2);   // sensor's own equator

    // axes
    const axes = [[[1.45, 0, 0], css("--x"), "X fwd"], [[0, 1.45, 0], css("--y"), "Y left"], [[0, 0, 1.45], css("--z"), "Z up"]];
    const c0 = proj([0, 0, 0]);
    axes.map(([v, color, label]) => [proj(mul(M, v)), color, label])
      .sort((a, b) => a[0][2] - b[0][2])
      .forEach(([p, color, label]) => {
        ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2.5;
        ctx.globalAlpha = p[2] < 0 ? 0.45 : 1;
        ctx.beginPath(); ctx.moveTo(c0[0], c0[1]); ctx.lineTo(p[0], p[1]); ctx.stroke();
        ctx.beginPath(); ctx.arc(p[0], p[1], 4, 0, 2 * Math.PI); ctx.fill();
        ctx.font = "600 11px system-ui";
        const tw = ctx.measureText(label).width;      // keep the label inside the canvas
        ctx.fillText(label, Math.min(Math.max(p[0] + 6, 2), w - tw - 2), Math.min(Math.max(p[1] - 6, 12), h - 4));
        ctx.globalAlpha = 1;
      });
  }

  // ------------------------------------------------------------------ vertical acceleration chart
  const dynPts = [];          // [t, g], last 5 s at 100 Hz
  function drawDyn(canvas, threshold) {
    const { ctx, w, h } = fitCanvas(canvas);
    ctx.clearRect(0, 0, w, h);
    const tEnd = dynPts.length ? dynPts[dynPts.length - 1][0] : 0, span = 5;
    const peak = Math.max(0, ...dynPts.map((p) => Math.abs(p[1])));
    const range = Math.max(threshold * 1.6, 0.6, peak * 1.1);
    const y = (g) => h / 2 - (g / range) * (h / 2 - 4);
    const x = (t) => w - ((tEnd - t) / span) * w;
    ctx.strokeStyle = css("--line"); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, y(0)); ctx.lineTo(w, y(0)); ctx.stroke();
    ctx.setLineDash([5, 4]);
    for (const [g, c] of [[threshold, css("--up")], [-threshold, css("--down")]]) {
      ctx.strokeStyle = c; ctx.beginPath(); ctx.moveTo(0, y(g)); ctx.lineTo(w, y(g)); ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.strokeStyle = css("--text"); ctx.lineWidth = 1.3;
    ctx.beginPath();
    dynPts.forEach(([t, g], i) => (i ? ctx.lineTo(x(t), y(g)) : ctx.moveTo(x(t), y(g))));
    ctx.stroke();
    ctx.fillStyle = css("--muted"); ctx.font = "10px system-ui";
    ctx.fillText(`+${threshold.toFixed(2)} g  up`, 4, y(threshold) - 3);
    ctx.fillText(`−${threshold.toFixed(2)} g  down`, 4, y(-threshold) + 11);
  }

  // ------------------------------------------------------------------ GPS trail
  const trail = [];
  function drawTrail(canvas, gps) {
    const { ctx, w, h } = fitCanvas(canvas);
    ctx.clearRect(0, 0, w, h);
    if (!trail.length) {
      ctx.fillStyle = css("--muted"); ctx.font = "12px system-ui"; ctx.textAlign = "center";
      ctx.fillText("waiting for a fix…", w / 2, h / 2); ctx.textAlign = "start";
      return;
    }
    const last = trail[trail.length - 1];
    const kx = 111320 * Math.cos(last.lat * D2R), ky = 111320;
    const pts = trail.map((p) => [(p.lng - last.lng) * kx, (p.lat - last.lat) * ky]);
    const ext = Math.max(20, ...pts.map(([a, b]) => Math.max(Math.abs(a), Math.abs(b))));
    const s = (Math.min(w, h) / 2 - 14) / ext;
    const X = (m) => w / 2 + m * s, Y = (m) => h / 2 - m * s;
    ctx.strokeStyle = css("--accent"); ctx.lineWidth = 2; ctx.globalAlpha = 0.8;
    ctx.beginPath();
    pts.forEach(([a, b], i) => (i ? ctx.lineTo(X(a), Y(b)) : ctx.moveTo(X(a), Y(b))));
    ctx.stroke(); ctx.globalAlpha = 1;
    // current position with heading
    ctx.fillStyle = css("--ok");
    ctx.beginPath(); ctx.arc(w / 2, h / 2, 5, 0, 2 * Math.PI); ctx.fill();
    if (gps.course !== null && gps.course !== undefined) {
      const a = gps.course * D2R;
      ctx.strokeStyle = css("--ok"); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(w / 2, h / 2); ctx.lineTo(w / 2 + Math.sin(a) * 18, h / 2 - Math.cos(a) * 18); ctx.stroke();
    }
    // north + scale bar
    ctx.fillStyle = css("--muted"); ctx.font = "11px system-ui";
    ctx.fillText("N ↑", w - 28, 14);
    const nice = [5, 10, 20, 50, 100, 200, 500, 1000].find((m) => m * s > 40) || 1000;
    ctx.strokeStyle = css("--muted"); ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(8, h - 10); ctx.lineTo(8 + nice * s, h - 10); ctx.stroke();
    ctx.fillText(nice + " m", 12 + nice * s, h - 6);
  }

  // ------------------------------------------------------------------ state -> panels
  function render(s) {
    const cam = s.camera, imu = s.imu, gps = s.gps;

    startCamera();
    setBadge($("camBadge"), cam, "Pi camera");
    setText("camInfo", `${cam.size[0]}×${cam.size[1]} · ${cam.fps.toFixed(1)} fps`);
    setText("camError", cam.error || "");

    setBadge($("imuBadge"), imu, "MPU-6050");
    setText("imuError", imu.error || "");
    setText("roll", fmt(imu.roll, 1, "°"));
    setText("pitch", fmt(imu.pitch, 1, "°"));
    setText("yaw", fmt(imu.yaw, 1, "°"));
    setText("accel", imu.acc.map((v) => v.toFixed(2)).join("  ") + " g");
    setText("gyro", imu.gyro.map((v) => v.toFixed(1)).join("  ") + " °/s");
    setText("temp", fmt(imu.temp, 1, " °C"));
    setText("dynNow", (imu.dyn >= 0 ? "+" : "") + imu.dyn.toFixed(2) + " g");
    const calib = $("calib"); if (calib) calib.hidden = !imu.calibrating;
    for (const p of imu.series || []) dynPts.push(p);
    while (dynPts.length && dynPts[0][0] < dynPts[dynPts.length - 1][0] - 5) dynPts.shift();
    if ($("sphere")) drawSphere($("sphere"), imu);
    if ($("dynChart")) drawDyn($("dynChart"), s.trigger.threshold_g);

    const gb = $("gpsBadge");
    if (gb) {
      const [cls, text] = gps.fix ? ["ok", "fix"] : ["bad", "no fix"];
      gb.className = "badge " + cls;
      gb.textContent = text;
    }
    setText("gpsStatus", gps.status);
    setText("gpsError", gps.error || "");
    setText("lat", fmt(gps.lat, 6));
    setText("lng", fmt(gps.lng, 6));
    setText("alt", fmt(gps.alt, 1, " m"));
    setText("sats", gps.sats ?? "—");
    setText("hdop", fmt(gps.hdop, 1));
    setText("speed", fmt(gps.speed_kmh, 1, " km/h"));
    setText("course", fmt(gps.course, 0, "°"));
    setText("utc", gps.utc || "—");
    setText("nmea", `${gps.baud ? gps.baud + " baud · " : ""}${gps.sentences} sentences · last: ${gps.last_line || "—"}`);
    if (gps.fix && gps.lat !== null) {
      const l = trail[trail.length - 1];
      if (!l || l.lat !== gps.lat || l.lng !== gps.lng) { trail.push({ lat: gps.lat, lng: gps.lng }); if (trail.length > 600) trail.shift(); }
    }
    if ($("trail")) drawTrail($("trail"), gps);
  }

  function setLink(ok) {
    const el = $("link");
    if (!el) return;
    el.className = "badge " + (ok ? "ok" : "bad");
    el.textContent = ok ? "live" : "disconnected";
  }

  function start() {
    on("state", render);
    const es = new EventSource("/api/stream");
    es.onopen = () => setLink(true);
    es.onerror = () => setLink(false);
    es.onmessage = (e) => { const s = JSON.parse(e.data); handlers.state.forEach((f) => f(s)); };
    es.addEventListener("capture", (e) => {
      const item = JSON.parse(e.data);
      flashCamera();
      handlers.capture.forEach((f) => f(item));
    });
    $("zeroBtn")?.addEventListener("click", () => post("/api/imu/zero").catch((err) => alert(err.message)));
  }

  return { start, on, post, fmt, $ };
})();
