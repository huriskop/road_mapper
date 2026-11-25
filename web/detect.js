// "Detect" chooser in the page header: which defects the device reports (pothole, speed bump, crack).
// The choice is kept by run.py (until it restarts) and shared by every open page.
"use strict";

(() => {
  const KINDS = [["pothole", "Potholes"], ["speed_bump", "Speed bumps"], ["crack", "Cracks"]];
  const header = document.querySelector("header");
  if (!header) return;

  const box = document.createElement("div");
  box.className = "detect-choice";
  box.title = "What the device looks for and puts on the map. Unticked types are ignored.";
  box.innerHTML = "<span>Detect</span>" + KINDS.map(([k, name]) =>
    `<label><input type="checkbox" value="${k}" checked> ${name}</label>`).join("");
  header.insertBefore(box, header.querySelector(".spacer, .live"));
  const boxes = [...box.querySelectorAll("input")];

  function show(detect) {
    boxes.forEach((b) => { b.checked = detect.includes(b.value); });
  }

  async function load() {
    try {
      const s = await (await fetch("/api/state")).json();
      if (s.detector && s.detector.detect && !box.contains(document.activeElement)) show(s.detector.detect);
    } catch (e) { /* offline: keep what is shown */ }
  }

  boxes.forEach((b) => b.addEventListener("change", async () => {
    const detect = boxes.filter((x) => x.checked).map((x) => x.value);
    if (!detect.length) { b.checked = true; return; }   // at least one type
    box.classList.add("busy");
    try {
      const res = await fetch("/api/detect", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ detect }),
      });
      const info = await res.json();
      if (res.ok) show(info.detect); else throw new Error(info.error);
    } catch (e) {
      alert("Could not change the detection: " + e.message);
      load();
    } finally {
      box.classList.remove("busy");
    }
  }));

  load();
  setInterval(load, 5000);   // follow changes made on another page or device
})();
