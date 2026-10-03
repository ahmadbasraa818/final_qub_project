"""A drag-and-drop interface in the browser, served locally with the standard library.

`blurfft gui` starts it on http://127.0.0.1:8765 and opens the page. Images are
sent to this machine's own server and analysed there; nothing leaves it.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from .detector import BlurDetector
from .imageio import load_gray
from .precision import parse_precision
from .render import map_overlay, png_base64, spectrum_picture, thumbnail

__all__ = ["serve", "make_server", "PAGE_PRECISIONS"]

MAX_UPLOAD = 30 * 1024 * 1024
PAGE_PRECISIONS = ("float64", "float32", "float16", "bfloat16", "floatx", "e8m6", "e8m4", "e8m2")


def _analyse(data: bytes, precision: str) -> dict:
    gray = load_gray(data)
    detector = BlurDetector(precision)
    report = detector.analyse(gray)
    blur_map = detector.map(gray)
    finite = blur_map.probability[np.isfinite(blur_map.probability)]
    metrics = {k: v for k, v in report.metrics.items() if k not in ("radial_frequency", "radial_power")}
    return {
        "report": {**report.to_dict(), "metrics": metrics, "summary": report.summary()},
        "map_share": float(np.mean(finite >= 0.5)) if finite.size else 0.0,
        "images": {
            "input": png_base64(thumbnail(gray)),
            "map": png_base64(map_overlay(gray, blur_map)),
            "spectrum": png_base64(spectrum_picture(gray, precision)),
        },
    }


def _compare(data: bytes) -> dict:
    gray = load_gray(data)
    rows = []
    for name in PAGE_PRECISIONS:
        p = parse_precision(name)
        r = BlurDetector(p).analyse(gray)
        rows.append({"precision": p.name, "bits": p.bits, "native": p.native, "blurry": r.blurry,
                     "probability": r.probability, "sigma": r.sigma, "fft_ms": r.fft_ms})
    return {"rows": rows}


class _Handler(BaseHTTPRequestHandler):
    server_version = "blurfft"

    def log_message(self, format, *args):  # quiet by default
        pass

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json")

    def do_GET(self):  # noqa: N802 (http.server naming)
        path = urlparse(self.path).path
        if path == "/":
            self._send(HTTPStatus.OK, PAGE.encode(), "text/html; charset=utf-8")
        elif path == "/api/formats":
            self._json(HTTPStatus.OK, {"formats": [{"name": parse_precision(p).name, "label": p, "bits": parse_precision(p).bits,
                                                     "native": parse_precision(p).native} for p in PAGE_PRECISIONS]})
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        url = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return self._json(HTTPStatus.BAD_REQUEST, {"error": "send the image as the request body"})
        if length > MAX_UPLOAD:
            return self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "images up to 30 MB, please"})
        data = self.rfile.read(length)
        query = parse_qs(url.query)
        try:
            if url.path == "/api/analyse":
                return self._json(HTTPStatus.OK, _analyse(data, query.get("precision", ["float64"])[0]))
            if url.path == "/api/compare":
                return self._json(HTTPStatus.OK, _compare(data))
            return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except (ValueError, OSError) as error:
            return self._json(HTTPStatus.BAD_REQUEST, {"error": f"could not read that image: {error}"})


def make_server(port: int = 8765) -> ThreadingHTTPServer:
    """A server bound to this machine only (127.0.0.1); port 0 picks a free one."""
    return ThreadingHTTPServer(("127.0.0.1", port), _Handler)


def serve(port: int = 8765, open_browser: bool = True) -> None:
    server = make_server(port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"blurfft is running at {url} (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:,">
<title>blurfft</title>
<style>
  :root { --paper: #f4f4f1; --paper-2: #e9e9e4; --ink: #121314; --ink-2: #555855; --rule: rgb(18 19 20 / 18%); --signal: #e8401f; }
  @media (prefers-color-scheme: dark) {
    :root { --paper: #141516; --paper-2: #1e1f20; --ink: #ecece7; --ink-2: #a5a8a2; --rule: rgb(236 236 231 / 18%); --signal: #f25a35; }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--paper); color: var(--ink); font: 16px/1.5 "Helvetica Neue", Arial, sans-serif; }
  main { max-width: 1200px; margin: 0 auto; padding: 32px 24px 64px; }
  h1 { font-size: clamp(2.2rem, 5vw, 3.6rem); line-height: 0.95; margin: 0 0 8px; letter-spacing: -0.02em; text-transform: uppercase; }
  .lede { color: var(--ink-2); max-width: 62ch; margin: 0 0 24px; }
  .controls { display: flex; flex-wrap: wrap; gap: 12px 24px; align-items: end; border-top: 2px solid var(--ink); padding-top: 16px; }
  label { font-weight: 600; font-size: 0.875rem; display: grid; gap: 6px; }
  select, button { font: inherit; color: var(--ink); background: transparent; border: 1px solid var(--ink); min-height: 44px; padding: 0 14px; border-radius: 0; }
  button.primary { background: var(--ink); color: var(--paper); font-weight: 600; cursor: pointer; }
  button:disabled { opacity: 0.4; cursor: default; }
  .drop { margin-top: 20px; min-height: 160px; display: grid; place-items: center; text-align: center; border: 1px dashed var(--ink-2); padding: 24px; cursor: pointer; }
  .drop.over { background: var(--signal); color: var(--ink); border-color: var(--ink); }
  .drop strong { font-size: 1.25rem; display: block; }
  .verdict { margin-top: 28px; display: grid; gap: 6px; }
  .verdict .big { font-size: clamp(2rem, 4vw, 3rem); font-weight: 800; line-height: 1; letter-spacing: -0.02em; }
  .bar { height: 10px; background: var(--paper-2); position: relative; max-width: 480px; }
  .bar span { position: absolute; inset: 0 auto 0 0; background: var(--signal); }
  .facts { display: flex; flex-wrap: wrap; gap: 8px 28px; margin: 12px 0 0; padding: 0; }
  .facts div { display: grid; }
  .facts dt { font-size: 0.8125rem; color: var(--ink-2); }
  .facts dd { margin: 0; font-weight: 600; font-variant-numeric: tabular-nums; }
  .plates { display: grid; gap: 20px; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); margin-top: 28px; }
  figure { margin: 0; }
  figure img { width: 100%; height: auto; display: block; outline: 1px solid var(--rule); outline-offset: -1px; image-rendering: auto; }
  figcaption { font-size: 0.875rem; color: var(--ink-2); margin-top: 6px; }
  table { border-collapse: collapse; width: 100%; margin-top: 16px; font-variant-numeric: tabular-nums; }
  th, td { text-align: left; padding: 8px 12px 8px 0; border-bottom: 1px solid var(--rule); }
  th { font-size: 0.8125rem; color: var(--ink-2); font-weight: 600; }
  .error { color: var(--signal); font-weight: 600; margin-top: 16px; }
  .muted { color: var(--ink-2); }
</style>
</head>
<body>
<main>
  <h1>blurfft</h1>
  <p class="lede">Drop a photo to check it for blur. Its spectrum shows how much fine detail survives; the FFT behind it can run in any precision, from 64-bit floating point down to formats a few bits wide.</p>
  <div class="controls">
    <label>Precision of the FFT
      <select id="precision"></select>
    </label>
    <button id="compare" disabled>Compare every precision</button>
  </div>
  <div class="drop" id="drop" tabindex="0" role="button" aria-label="Choose an image or drop one here">
    <div><strong>Drop an image here</strong><span class="muted">or click to choose one. It is analysed on this computer.</span></div>
    <input id="file" type="file" accept="image/*" hidden>
  </div>
  <p class="error" id="error" role="alert" hidden></p>
  <section id="result" hidden aria-live="polite">
    <div class="verdict">
      <div class="big" id="verdict"></div>
      <div class="bar" aria-hidden="true"><span id="bar"></span></div>
      <dl class="facts" id="facts"></dl>
    </div>
    <div class="plates">
      <figure><img id="map" alt="The image with blurred areas tinted vermilion"><figcaption>Blur map: tinted where tiles are more likely blurred than sharp</figcaption></figure>
      <figure><img id="spectrum" alt="The image's frequency spectrum"><figcaption>Frequency spectrum: the centre is coarse detail, the edges fine detail</figcaption></figure>
    </div>
    <div id="comparison"></div>
  </section>
</main>
<script>
const $ = (id) => document.getElementById(id);
let current = null;
fetch("/api/formats").then((r) => r.json()).then(({ formats }) => {
  for (const f of formats) {
    const o = document.createElement("option");
    o.value = f.label;
    o.textContent = `${f.label} (${f.bits} bits${f.native ? "" : ", emulated"})`;
    $("precision").append(o);
  }
});
function fail(message) { $("error").textContent = message; $("error").hidden = false; }
async function analyse() {
  if (!current) return;
  $("error").hidden = true;
  const precision = $("precision").value;
  const response = await fetch(`/api/analyse?precision=${encodeURIComponent(precision)}`, { method: "POST", body: current });
  const data = await response.json();
  if (!response.ok) return fail(data.error || "Something went wrong.");
  const r = data.report;
  $("verdict").textContent = r.blurry ? `Blurred (${r.severity})` : "Sharp";
  // The bar shows the stronger evidence: the whole image, or the share of blurred tiles.
  $("bar").style.width = `${(Math.max(r.probability, r.tile_share) * 100).toFixed(1)}%`;
  const facts = [
    ["Whole image", `${(r.probability * 100).toFixed(1)}% likely blurred`],
    ["Blur radius", r.blurry ? `${r.sigma.toFixed(2)} px` : "none"],
    ["Kind", r.blur_type ? (r.blur_type === "motion" ? `motion, ${r.motion_angle.toFixed(0)}°` : "no single direction") : "n/a"],
    ["Blurred tiles", `${(r.tile_share * 100).toFixed(0)}%`],
    ["Size", `${r.width} × ${r.height}`],
    ["FFT", `${r.fft_ms.toFixed(1)} ms in ${r.precision}`],
  ];
  $("facts").innerHTML = facts.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");
  $("map").src = `data:image/png;base64,${data.images.map}`;
  $("spectrum").src = `data:image/png;base64,${data.images.spectrum}`;
  $("result").hidden = false;
  $("compare").disabled = false;
}
async function compare() {
  if (!current) return;
  $("compare").disabled = true;
  $("comparison").innerHTML = '<p class="muted">Measuring at every precision…</p>';
  const response = await fetch("/api/compare", { method: "POST", body: current });
  const data = await response.json();
  $("compare").disabled = false;
  if (!response.ok) return fail(data.error || "Something went wrong.");
  const rows = data.rows.map((row) => `<tr><td>${row.precision}${row.native ? "" : " (emulated)"}</td><td>${row.bits}</td><td>${row.blurry ? "blurred" : "sharp"}</td><td>${(row.probability * 100).toFixed(1)}%</td><td>${row.fft_ms.toFixed(1)} ms</td></tr>`).join("");
  $("comparison").innerHTML = `<table><thead><tr><th>Precision</th><th>Bits</th><th>Verdict</th><th>Whole image</th><th>FFT</th></tr></thead><tbody>${rows}</tbody></table>`;
}
function take(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) return fail("That file is not an image.");
  current = file;
  $("comparison").innerHTML = "";
  analyse();
}
const drop = $("drop");
drop.addEventListener("click", () => $("file").click());
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file").click(); } });
$("file").addEventListener("change", (e) => take(e.target.files[0]));
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); take(e.dataTransfer.files[0]); });
$("precision").addEventListener("change", analyse);
$("compare").addEventListener("click", compare);
</script>
</body>
</html>
"""
