(() => {
  "use strict";

  const bySelector = (selector, root = document) => root.querySelector(selector);
  const all = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  function updateInputPanels() {
    const jobChoice = bySelector('input[name="job_mode"]:checked');
    const resumeChoice = bySelector('input[name="resume_mode"]:checked');
    const jobMode = jobChoice ? jobChoice.value : "url";
    const resumeMode = resumeChoice ? resumeChoice.value : "master";

    all("[data-job-panel]").forEach((panel) => {
      panel.hidden = panel.dataset.jobPanel !== jobMode;
    });
    all("[data-resume-panel]").forEach((panel) => {
      panel.hidden = panel.dataset.resumePanel !== resumeMode;
    });
    const identity = bySelector("[data-manual-identity]");
    if (identity) {
      identity.hidden = jobMode === "url";
    }
  }

  function showClientError(message) {
    const panel = bySelector("#form-error");
    if (!panel) return;
    panel.textContent = message;
    panel.hidden = !message;
    if (message) panel.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  function validateDashboardForm(form) {
    const jobMode = bySelector('input[name="job_mode"]:checked', form)?.value;
    const resumeMode = bySelector('input[name="resume_mode"]:checked', form)?.value;
    if (!jobMode) return "Choose one job-description input mode.";
    if (!resumeMode) return "Choose a résumé source.";
    if (resumeMode === "upload") {
      const resume = bySelector("#resume-upload", form);
      if (!resume?.files?.length || !resume.files[0].name.toLowerCase().endsWith(".docx")) {
        return "Choose a valid .docx résumé upload.";
      }
    }
    if (jobMode === "url") {
      const url = bySelector("#job-url", form)?.value.trim() || "";
      if (!url.startsWith("https://")) return "Enter an HTTPS LinkedIn job URL.";
      try {
        const parsed = new URL(url);
        if (!["linkedin.com", "www.linkedin.com"].includes(parsed.hostname.toLowerCase())) {
          return "Only LinkedIn job URLs are supported.";
        }
      } catch (_) {
        return "Enter a valid LinkedIn job URL.";
      }
    } else {
      if (!(bySelector("#company", form)?.value.trim())) return "Enter the company name.";
      if (!(bySelector("#role", form)?.value.trim())) return "Enter the role title.";
      if (jobMode === "pasted" && !(bySelector("#pasted-description", form)?.value.trim())) {
        return "Paste the complete job description.";
      }
      if (jobMode === "file" && !(bySelector("#job-file", form)?.files?.length)) {
        return "Choose a UTF-8 .txt job-description file.";
      }
    }
    return "";
  }

  function setupDashboard() {
    const form = bySelector("#new-run-form");
    if (!form) return;
    all('input[name="job_mode"], input[name="resume_mode"]', form).forEach((input) => {
      input.addEventListener("change", updateInputPanels);
    });
    updateInputPanels();

    const pasteButton = bySelector("#paste-button");
    if (pasteButton) {
      pasteButton.addEventListener("click", async () => {
        const status = bySelector("#clipboard-status");
        try {
          if (!navigator.clipboard?.readText) throw new Error("Clipboard API unavailable");
          const text = await navigator.clipboard.readText();
          if (!text.trim()) throw new Error("Clipboard is empty");
          bySelector("#pasted-description").value = text;
          status.textContent = "Clipboard text pasted locally.";
        } catch (_) {
          status.textContent = "Clipboard access was unavailable. Use Ctrl+V to paste manually.";
        }
      });
    }

    form.addEventListener("submit", (event) => {
      const error = validateDashboardForm(form);
      showClientError(error);
      if (error) {
        event.preventDefault();
        return;
      }
      const button = bySelector("#start-button");
      if (button) {
        button.disabled = true;
        button.firstElementChild.textContent = "Starting safely…";
      }
    });
  }

  function setWorkflowStage(stageIndex, status) {
    all("#workflow-stepper [data-stage-index]").forEach((item) => {
      const index = Number(item.dataset.stageIndex);
      const isActive = index === stageIndex && status !== "COMPLETE";
      item.classList.toggle("done", index < stageIndex || status === "COMPLETE");
      item.classList.toggle("active", isActive);
      if (isActive) {
        item.setAttribute("aria-current", "step");
      } else {
        item.removeAttribute("aria-current");
      }
    });
  }

  function renderActivity(events) {
    const list = bySelector("#activity-list");
    if (!list) return;
    list.replaceChildren();
    events.forEach((event) => {
      const item = document.createElement("li");
      const time = document.createElement("time");
      time.textContent = event.time;
      const node = document.createElement("span");
      node.className = "activity-node";
      const detail = document.createElement("div");
      const message = document.createElement("strong");
      message.textContent = event.message;
      const stage = document.createElement("small");
      stage.textContent = event.stage.replaceAll("_", " ");
      detail.append(message, stage);
      item.append(time, node, detail);
      list.append(item);
    });
  }

  function setupFallback() {
    const form = bySelector("[data-fallback-form]");
    const show = bySelector("[data-show-fallback]");
    const hide = bySelector("[data-hide-fallback]");
    if (!form || !show) return;
    show.addEventListener("click", () => {
      form.hidden = false;
      bySelector("textarea", form)?.focus();
    });
    hide?.addEventListener("click", () => {
      form.hidden = true;
    });
  }

  function setupSubmissionLocks() {
    all("form").forEach((form) => {
      if (form.id === "new-run-form") return;
      form.addEventListener("submit", () => {
        all('button[type="submit"]', form).forEach((button) => {
          button.disabled = true;
        });
      });
    });
  }

  function setupRunPolling() {
    const page = bySelector("#run-page");
    if (!page) return;
    const runId = page.dataset.runId;
    let revision = Number(page.dataset.revision || "0");
    let status = page.dataset.status;
    let stopped = ["COMPLETE", "FAILED", "CANCELLED"].includes(status);

    const poll = async () => {
      if (stopped) return;
      try {
        const response = await fetch(`/api/runs/${encodeURIComponent(runId)}`, {
          credentials: "same-origin",
          headers: { Accept: "application/json" },
        });
        if (!response.ok) throw new Error("Status request failed");
        const data = await response.json();
        bySelector("#run-message").textContent = data.message;
        const current = bySelector("#activity-current");
        if (current) current.textContent = data.message;
        setWorkflowStage(data.stage_index, data.status);
        renderActivity(data.events);

        const mustReload =
          data.status !== status &&
          (data.status === "AWAITING_APPROVAL" ||
            ["COMPLETE", "FAILED", "CANCELLED"].includes(data.status));
        const newGate = data.approval_kind && data.revision !== revision;
        revision = data.revision;
        status = data.status;
        if (mustReload || newGate) {
          window.location.reload();
          return;
        }
      } catch (_) {
        const current = bySelector("#activity-current");
        if (current) {
          current.textContent = "The local status connection paused. Retrying…";
        }
      }
      window.setTimeout(poll, 900);
    };
    window.setTimeout(poll, 700);
  }

  document.addEventListener("DOMContentLoaded", () => {
    setupDashboard();
    setupFallback();
    setupSubmissionLocks();
    setupRunPolling();
  });
})();
