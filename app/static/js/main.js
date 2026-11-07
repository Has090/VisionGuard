document.addEventListener("DOMContentLoaded", () => {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const previewContainer = document.getElementById("preview-container");
  const previewImage = document.getElementById("preview-image");
  const submitBtn = document.getElementById("submit-btn");
  const resetBtn = document.getElementById("reset-btn");
  const spinnerContainer = document.getElementById("spinner-container");
  const resultCard = document.getElementById("result-card");
  const toast = document.getElementById("toast");

  // History modal elements
  const openHistoryBtn = document.getElementById("open-history-btn");
  const closeHistoryBtn = document.getElementById("close-history-btn");
  const historyModal = document.getElementById("history-modal");
  const historyTbody = document.getElementById("history-tbody");

  // Metric displays
  const badge = document.getElementById("badge");
  const latencyDisplay = document.getElementById("latency-display");
  const confidencePercent = document.getElementById("confidence-percent");
  const confidenceBarFill = document.getElementById("confidence-bar-fill");
  const modelUsedVal = document.getElementById("model-used-val");
  const rawScore = document.getElementById("raw-score");
  const thresholdVal = document.getElementById("threshold-val");
  const feedbackMsg = document.getElementById("feedback-msg");

  let currentFile = null;
  let lastPrediction = null;
  let lastPredictionId = null;

  function showToast(message, isError = true) {
    toast.textContent = message;
    toast.style.background = isError ? "#ef4444" : "#10b981";
    toast.style.display = "block";
    setTimeout(() => {
      toast.style.display = "none";
    }, 4000);
  }

  // Click dropzone to open native file browser
  dropzone.addEventListener("click", () => fileInput.click());

  // Drag and drop handlers
  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFile(files[0]);
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
  });

  function handleFile(file) {
    const allowed = ["image/jpeg", "image/png", "image/webp"];
    if (!allowed.includes(file.type)) {
      showToast("Invalid file type. Please upload a JPG, PNG, or WEBP image.");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      showToast("File size exceeds 10 MB limit.");
      return;
    }

    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImage.src = e.target.result;
      previewContainer.style.display = "block";
      dropzone.style.display = "none";
      submitBtn.disabled = false;
      resultCard.style.display = "none";
    };
    reader.readAsDataURL(file);
  }

  // Reset UI
  resetBtn.addEventListener("click", () => {
    currentFile = null;
    fileInput.value = "";
    previewContainer.style.display = "none";
    dropzone.style.display = "block";
    resultCard.style.display = "none";
    submitBtn.disabled = true;
    feedbackMsg.textContent = "";
  });

  // Call /predict
  submitBtn.addEventListener("click", async () => {
    if (!currentFile) return;

    spinnerContainer.style.display = "block";
    resultCard.style.display = "none";
    submitBtn.disabled = true;

    const formData = new FormData();
    formData.append("file", currentFile);

    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      spinnerContainer.style.display = "none";
      submitBtn.disabled = false;

      if (!response.ok || data.status === "error") {
        showToast(data.message || "Failed to process image.");
        return;
      }

      lastPrediction = data.prediction;
      lastPredictionId = data.prediction_id || null;
      const isAI = data.prediction === "AiArtData";

      badge.textContent = isAI ? "AI Generated Art" : "Authentic Real Art";
      badge.className = isAI ? "badge badge-ai" : "badge badge-real";

      latencyDisplay.textContent = `Latency: ${data.latency_ms} ms`;
      confidencePercent.textContent = `${data.confidence}%`;
      confidenceBarFill.style.width = `${data.confidence}%`;
      confidenceBarFill.style.background = isAI ? "#ef4444" : "#10b981";

      modelUsedVal.textContent = data.model_name || "VisionGuard_MobileNetV2_Champion";
      rawScore.textContent = data.raw_score;
      thresholdVal.textContent = data.threshold_applied;

      resultCard.style.display = "block";
      resultCard.scrollIntoView({ behavior: "smooth" });
    } catch (err) {
      spinnerContainer.style.display = "none";
      submitBtn.disabled = false;
      showToast("Network error communicating with the inference server.");
    }
  });

  // Call /feedback
  window.submitFeedback = async (verdict) => {
    if (!lastPrediction) return;

    try {
      const response = await fetch("/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prediction_id: lastPredictionId,
          prediction: lastPrediction,
          user_verdict: verdict,
          notes: `User rated verdict via UI as ${verdict}`
        })
      });

      if (response.ok) {
        feedbackMsg.textContent = `Thank you! Feedback recorded (${verdict}).`;
        feedbackMsg.style.color = "#10b981";
      } else {
        feedbackMsg.textContent = "Error saving feedback.";
        feedbackMsg.style.color = "#ef4444";
      }
    } catch (err) {
      feedbackMsg.textContent = "Network error while saving feedback.";
    }
  };

  // Open History Modal -> Query /history
  openHistoryBtn.addEventListener("click", async () => {
    historyModal.style.display = "flex";
    historyTbody.innerHTML = '<tr><td colspan="8" style="text-align: center; color: var(--text-muted);">Fetching from SQLite database...</td></tr>';

    try {
      const res = await fetch("/history");
      const json = await res.json();

      if (!res.ok || json.status === "error") {
        historyTbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: #ef4444;">${json.message || "Failed to load database history."}</td></tr>`;
        return;
      }

      if (!json.history || json.history.length === 0) {
        historyTbody.innerHTML = '<tr><td colspan="8" style="text-align: center; color: var(--text-muted);">No predictions recorded in the database yet.</td></tr>';
        return;
      }

      historyTbody.innerHTML = json.history.map((row) => {
        const isAI = row.prediction === "AiArtData";
        const pillClass = isAI ? "pill-ai" : "pill-real";
        const feedbackText = row.user_verdict 
          ? (row.user_verdict === "correct" ? "👍 Correct" : "👎 Incorrect") 
          : '<span style="color: var(--text-muted);">Unrated</span>';

        return `
          <tr>
            <td>#${row.id}</td>
            <td style="white-space: nowrap;">${row.timestamp.replace("T", " ").substring(0, 19)}</td>
            <td><code style="color: var(--accent-cyan); font-size: 0.75rem;">${row.model_name || "MobileNetV2"}</code></td>
            <td style="max-width: 130px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${row.filename}">${row.filename}</td>
            <td><span class="${pillClass}">${row.prediction}</span></td>
            <td><strong>${row.confidence}%</strong></td>
            <td>${row.latency_ms} ms</td>
            <td>${feedbackText}</td>
          </tr>
        `;
      }).join("");
    } catch (e) {
      historyTbody.innerHTML = '<tr><td colspan="8" style="text-align: center; color: #ef4444;">Could not connect to /history endpoint.</td></tr>';
    }
  });

  closeHistoryBtn.addEventListener("click", () => {
    historyModal.style.display = "none";
  });

  window.addEventListener("click", (e) => {
    if (e.target === historyModal) {
      historyModal.style.display = "none";
    }
  });
});