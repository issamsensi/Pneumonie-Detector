const form = document.getElementById("analysis-form");
const imageInput = document.getElementById("image-input");
const dropZone = document.getElementById("drop-zone");
const previewImage = document.getElementById("preview-image");
const fileMeta = document.getElementById("file-meta");
const analyzeButton = document.getElementById("analyze-button");
const spinner = document.getElementById("spinner");
const resultsSection = document.getElementById("results-section");
const verdictCard = document.getElementById("verdict-card");
const verdictLabel = document.getElementById("verdict-label");
const verdictValue = document.getElementById("verdict-value");
const verdictDetail = document.getElementById("verdict-detail");
const originalResult = document.getElementById("original-result");
const overlayResult = document.getElementById("overlay-result");
const confidenceValue = document.getElementById("confidence-value");
const regionsValue = document.getElementById("regions-value");
const filenameValue = document.getElementById("filename-value");
const reportText = document.getElementById("report-text");
const patientId = document.getElementById("patient-id");
const downloadOverlayButton = document.getElementById("download-overlay");
const downloadReportButton = document.getElementById("download-report");

let currentResult = null;
let selectedFileName = "";

const allowedTypes = new Set(["image/jpeg", "image/png"]);
const maxFileSizeBytes = 10 * 1024 * 1024;

function formatLabel(verdict) {
  return verdict === "pneumonia" ? "PNEUMONIA DETECTED" : "NORMAL";
}

function setBusy(isBusy) {
  spinner.classList.toggle("hidden", !isBusy);
  analyzeButton.disabled = isBusy || !imageInput.files.length;
  downloadReportButton.disabled = isBusy || !currentResult;
  downloadOverlayButton.disabled = isBusy || !currentResult;
}

function previewFile(file) {
  if (!allowedTypes.has(file.type)) {
    fileMeta.textContent = "Unsupported file type. Use JPG, JPEG, or PNG.";
    analyzeButton.disabled = true;
    return;
  }

  if (file.size > maxFileSizeBytes) {
    fileMeta.textContent = "File exceeds the 10MB limit.";
    analyzeButton.disabled = true;
    return;
  }

  selectedFileName = file.name;
  fileMeta.textContent = `${file.name} • ${(file.size / (1024 * 1024)).toFixed(2)} MB`;
  analyzeButton.disabled = false;

  const reader = new FileReader();
  reader.onload = () => {
    previewImage.src = reader.result;
    previewImage.classList.remove("hidden");
    dropZone.classList.add("has-preview");
  };
  reader.readAsDataURL(file);
}

function downloadDataUrl(dataUrl, filename) {
  const anchor = document.createElement("a");
  anchor.href = dataUrl;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
}

function updateResults(result) {
  currentResult = result;
  const verdictText = formatLabel(result.verdict);
  const confidence = Math.round(result.confidence * 100);

  verdictCard.dataset.state = result.verdict;
  verdictLabel.textContent = verdictText;
  verdictValue.textContent = verdictText;
  verdictDetail.textContent = result.verdict === "pneumonia"
    ? "The model identified suspicious regions that warrant clinical confirmation."
    : "The model did not identify suspicious regions in this study.";

  originalResult.src = result.original_b64;
  overlayResult.src = result.overlay_b64;
  confidenceValue.textContent = `${confidence}%`;
  regionsValue.textContent = `${result.regions}`;
  filenameValue.textContent = selectedFileName;
  reportText.value = result.report_text;

  resultsSection.classList.remove("hidden");
  downloadOverlayButton.disabled = false;
  downloadReportButton.disabled = false;
}

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (!file) {
    analyzeButton.disabled = true;
    fileMeta.textContent = "No file selected";
    return;
  }
  previewFile(file);
});

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropZone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropZone.classList.remove("dragover");
  });
});

dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (!file) {
    return;
  }
  const dataTransfer = new DataTransfer();
  dataTransfer.items.add(file);
  imageInput.files = dataTransfer.files;
  previewFile(file);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = imageInput.files[0];
  if (!file) {
    fileMeta.textContent = "Please choose an image first.";
    return;
  }

  const formData = new FormData();
  formData.append("image", file);

  setBusy(true);

  try {
    const response = await fetch("/analyze", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Analysis failed.");
    }

    updateResults(payload);
  } catch (error) {
    verdictCard.dataset.state = "normal";
    verdictLabel.textContent = "Analysis failed";
    verdictValue.textContent = "Unable to analyze image";
    verdictDetail.textContent = error.message;
    resultsSection.classList.remove("hidden");
  } finally {
    setBusy(false);
  }
});

downloadOverlayButton.addEventListener("click", () => {
  if (!currentResult) {
    return;
  }
  const filename = `pneumonia_overlay_${Date.now()}.png`;
  downloadDataUrl(currentResult.overlay_b64, filename);
});

downloadReportButton.addEventListener("click", () => {
  if (!currentResult) {
    return;
  }
  const patientValue = (patientId.value || "anonymous").trim() || "anonymous";
  const params = new URLSearchParams({
    verdict: currentResult.verdict,
    confidence: currentResult.confidence,
    regions: currentResult.regions,
    date: currentResult.analysis_date,
  });
  window.open(`/download/report/${encodeURIComponent(patientValue)}?${params.toString()}`, "_blank", "noopener,noreferrer");
});

analyzeButton.disabled = true;
downloadOverlayButton.disabled = true;
downloadReportButton.disabled = true;
