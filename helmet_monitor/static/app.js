"use strict";

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- tabs ----

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    document.querySelectorAll(".view").forEach((v) =>
      v.classList.toggle("active", v.id === `view-${tab.dataset.view}`));
    if (tab.dataset.view === "workers") {
      loadWorkers();
      if (photoSource === "camera" && !capturedBlob) startCaptureCamera();
    } else {
      stopCaptureCamera();
    }
    if (tab.dataset.view === "violations") loadViolations();
  });
});

function el(tag, props = {}, children = []) {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
}

function formatTime(iso) {
  return new Date(iso).toLocaleString();
}

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join(", ") : body.detail;
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return response.status === 204 ? null : response.json();
}

// -------------------------------------------------------------- health ----

async function checkHealth() {
  try {
    const health = await api("/api/health");
    const problems = [];
    if (!health.detector_loaded) problems.push("Helmet detector not loaded: live monitoring is unavailable.");
    if (!health.face_model_loaded) problems.push("Face model not loaded: workers can't be registered or identified.");
    $("health").hidden = problems.length === 0;
    $("health").textContent = problems.join(" ");
  } catch {
    $("health").hidden = false;
    $("health").textContent = "Server unreachable.";
  }
}

// ------------------------------------------------------------- monitor ----

const video = $("video");
const overlay = $("overlay");
const ctx = overlay.getContext("2d");
const capture = document.createElement("canvas");

let stream = null;
let socket = null;
let sendTimer = null;
let awaitingResult = false;
let lastSent = 0;

async function startMonitoring() {
  const cameraId = $("camera-id").value.trim() || "cam-1";
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
  } catch (error) {
    alert(`Could not open the camera: ${error.message}`);
    return;
  }
  video.srcObject = stream;
  $("stage-empty").hidden = true;
  $("start").hidden = true;
  $("stop").hidden = false;

  const scheme = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${scheme}://${location.host}/ws/monitor/${encodeURIComponent(cameraId)}`);
  socket.binaryType = "arraybuffer";
  socket.onmessage = (message) => handleMessage(JSON.parse(message.data));
  socket.onclose = () => stopMonitoring();
  socket.onopen = () => {
    const interval = 1000 / Number($("fps").value);
    sendTimer = setInterval(sendFrame, interval);
  };
}

function stopMonitoring() {
  clearInterval(sendTimer);
  sendTimer = null;
  if (socket && socket.readyState <= WebSocket.OPEN) socket.close();
  socket = null;
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = null;
  video.srcObject = null;
  awaitingResult = false;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  $("stage-empty").hidden = false;
  $("start").hidden = false;
  $("stop").hidden = true;
}

function sendFrame() {
  if (!socket || socket.readyState !== WebSocket.OPEN || !video.videoWidth) return;
  // Backpressure: wait for the previous result (or 2 s) before sending more.
  if (awaitingResult && performance.now() - lastSent < 2000) return;
  capture.width = video.videoWidth;
  capture.height = video.videoHeight;
  capture.getContext("2d").drawImage(video, 0, 0);
  capture.toBlob((blob) => {
    if (blob && socket && socket.readyState === WebSocket.OPEN) {
      socket.send(blob);
      awaitingResult = true;
      lastSent = performance.now();
    }
  }, "image/jpeg", 0.8);
}

function handleMessage(message) {
  if (message.type === "error") {
    $("health").hidden = false;
    $("health").textContent = message.message;
    return;
  }
  awaitingResult = false;
  drawResult(message);
  $("stat-people").textContent = message.total_people;
  $("stat-helmet").textContent = message.with_helmet;
  $("stat-nohelmet").textContent = message.without_helmet;
  $("stat-latency").textContent = Math.round(message.inference_ms);
  message.new_violations.forEach(addEvent);
}

function drawResult(result) {
  // Match the overlay to the displayed video area (object-fit: contain).
  const rect = overlay.getBoundingClientRect();
  overlay.width = rect.width;
  overlay.height = rect.height;
  const scale = Math.min(rect.width / result.frame.width, rect.height / result.frame.height);
  const offsetX = (rect.width - result.frame.width * scale) / 2;
  const offsetY = (rect.height - result.frame.height * scale) / 2;
  const map = ([x1, y1, x2, y2]) =>
    [offsetX + x1 * scale, offsetY + y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale];

  ctx.clearRect(0, 0, overlay.width, overlay.height);
  ctx.lineWidth = 3;
  ctx.font = "600 14px system-ui, sans-serif";
  for (const person of result.people) {
    const color = person.wearing_helmet ? "#2e9e5b" : "#d64545";
    const [x, y, w, h] = map(person.head_box);
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, w, h);

    const parts = [person.wearing_helmet ? "Hard hat" : "NO HARD HAT"];
    if (person.name) parts.push(person.name);
    if (person.timer) {
      parts.push(person.timer.recorded ? "RECORDED" : `${Math.ceil(person.timer.remaining)}s`);
    }
    const label = parts.join(" · ");
    const textWidth = ctx.measureText(label).width + 10;
    ctx.fillStyle = color;
    ctx.fillRect(x, Math.max(0, y - 22), textWidth, 22);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, x + 5, Math.max(16, y - 6));

    if (person.face_box) {
      ctx.setLineDash([5, 4]);
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = "#ffffff";
      ctx.strokeRect(...map(person.face_box));
      ctx.setLineDash([]);
      ctx.lineWidth = 3;
    }
  }
}

function addEvent(violation) {
  const list = $("events");
  list.querySelector(".muted")?.remove();
  const details = el("div", {}, [
    el("div", { className: "who", textContent: violation.name }),
    el("div", { className: "muted", textContent: `${violation.camera_id} · ${formatTime(violation.created_at)}` }),
  ]);
  const children = violation.snapshot ? [el("img", { src: violation.snapshot, alt: "" }), details] : [details];
  list.prepend(el("li", {}, children));
  while (list.children.length > 50) list.lastChild.remove();
}

$("start").addEventListener("click", startMonitoring);
$("stop").addEventListener("click", stopMonitoring);

// ------------------------------------------------------------- workers ----

async function loadWorkers() {
  try {
    const workers = await api("/api/workers");
    $("worker-count").textContent = `(${workers.length})`;
    $("workers").replaceChildren(...workers.map((worker) => el("tr", {}, [
      el("td", { textContent: worker.id }),
      el("td", { textContent: worker.name }),
      el("td", { textContent: worker.email || "–" }),
      el("td", { textContent: formatTime(worker.created_at) }),
      el("td", {}, [el("button", {
        className: "danger",
        textContent: "Remove",
        onclick: () => removeWorker(worker),
      })]),
    ])));
  } catch (error) {
    showWorkerMessage(error.message, false);
  }
}

async function removeWorker(worker) {
  if (!confirm(`Remove ${worker.name}? Their face will no longer be recognised.`)) return;
  try {
    await api(`/api/workers/${worker.id}`, { method: "DELETE" });
    loadWorkers();
  } catch (error) {
    showWorkerMessage(error.message, false);
  }
}

function showWorkerMessage(text, ok) {
  $("worker-message").textContent = text;
  $("worker-message").className = `message ${ok ? "ok" : "bad"}`;
}

// Photo source: file upload or a capture from the live camera.
let photoSource = "upload";
let captureStream = null;
let capturedBlob = null;

function showPreview(blobOrFile) {
  $("photo-preview").hidden = !blobOrFile;
  if (blobOrFile) $("photo-preview").src = URL.createObjectURL(blobOrFile);
}

function stopCaptureCamera() {
  if (captureStream) captureStream.getTracks().forEach((track) => track.stop());
  captureStream = null;
  $("capture-video").srcObject = null;
}

async function startCaptureCamera() {
  capturedBlob = null;
  showPreview(null);
  $("capture-video").hidden = false;
  $("capture-take").hidden = false;
  $("capture-retake").hidden = true;
  try {
    captureStream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
    $("capture-video").srcObject = captureStream;
  } catch (error) {
    showWorkerMessage(`Could not open the camera: ${error.message}`, false);
  }
}

function setPhotoSource(source) {
  photoSource = source;
  document.querySelectorAll(".seg").forEach((b) => b.classList.toggle("active", b.dataset.source === source));
  const camera = source === "camera";
  $("photo-file").hidden = camera;
  $("photo-file").required = !camera;
  $("capture").hidden = !camera;
  showPreview(camera ? null : $("photo-file").files[0]);
  if (camera) startCaptureCamera();
  else stopCaptureCamera();
}

document.querySelectorAll(".seg").forEach((button) =>
  button.addEventListener("click", () => setPhotoSource(button.dataset.source)));

$("capture-take").addEventListener("click", () => {
  const video = $("capture-video");
  if (!video.videoWidth) return;
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  canvas.toBlob((blob) => {
    capturedBlob = blob;
    stopCaptureCamera();
    showPreview(blob);
    video.hidden = true;
    $("capture-take").hidden = true;
    $("capture-retake").hidden = false;
  }, "image/jpeg", 0.92);
});

$("capture-retake").addEventListener("click", startCaptureCamera);

$("photo-file").addEventListener("change", (event) => showPreview(event.target.files[0]));

$("worker-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.target;
  const data = new FormData(form);
  if (photoSource === "camera") {
    if (!capturedBlob) {
      showWorkerMessage("Capture a photo first.", false);
      return;
    }
    data.set("photo", capturedBlob, "camera.jpg");
  }
  const button = form.querySelector("button[type=submit]");
  button.disabled = true;
  showWorkerMessage("Registering…", true);
  try {
    const worker = await api("/api/workers", { method: "POST", body: data });
    const note = worker.faces_in_photo > 1 ? " (several faces found; used the largest)" : "";
    showWorkerMessage(`Registered ${worker.name}${note}.`, true);
    form.reset();
    if (photoSource === "camera") startCaptureCamera();
    else showPreview(null);
    loadWorkers();
  } catch (error) {
    showWorkerMessage(error.message, false);
  } finally {
    button.disabled = false;
  }
});

// ---------------------------------------------------------- violations ----

async function loadViolations() {
  try {
    const [records, summary] = await Promise.all([api("/api/violations?limit=100"), api("/api/violations/summary")]);
    $("sum-total").textContent = summary.total;
    $("sum-today").textContent = summary.today;
    $("sum-top").textContent = summary.by_worker[0]
      ? `${summary.by_worker[0].worker_name} (${summary.by_worker[0].count})` : "–";
    $("violations").replaceChildren(...records.map((v) => el("tr", {}, [
      el("td", {}, v.snapshot ? [el("a", { href: v.snapshot, target: "_blank" }, [el("img", { src: v.snapshot, alt: "Snapshot" })])] : ["–"]),
      el("td", { textContent: v.worker_name }),
      el("td", { textContent: v.similarity == null ? "–" : `${Math.round(v.similarity * 100)}%` }),
      el("td", { textContent: v.camera_id }),
      el("td", { textContent: formatTime(v.created_at) }),
    ])));
    if (!records.length) {
      $("violations").replaceChildren(el("tr", {}, [el("td", { colSpan: 5, className: "muted", textContent: "No violations recorded." })]));
    }
  } catch (error) {
    $("violations").replaceChildren(el("tr", {}, [el("td", { colSpan: 5, textContent: error.message })]));
  }
}

$("refresh-violations").addEventListener("click", loadViolations);

checkHealth();
