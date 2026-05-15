const rootEl = document.getElementById("control-rail-root");
const themeButtonEl = document.getElementById("theme-toggle-button");
const themeIconEl = document.getElementById("theme-toggle-icon");
const workspaceAppSelectEl = document.getElementById("workspace-app-select");
const modelSelectEl = document.getElementById("model-select");
const planningButtonEl = document.getElementById("planning-toggle-button");
const reflectionButtonEl = document.getElementById("reflection-toggle-button");
const assistantButtonEl = document.getElementById("assistant-toggle-button");
const assistantIconEl = document.getElementById("assistant-toggle-icon");
const assistantPulseEl = document.getElementById("assistant-toggle-pulse");

let eventCounter = 0;
let workspaceAppOptionsMarkup = "";
let modelOptionsMarkup = "";

function sendMessageToStreamlitClient(type, data) {
  const outData = Object.assign(
    {
      isStreamlitMessage: true,
      type: type,
    },
    data
  );
  window.parent.postMessage(outData, "*");
}

function init() {
  sendMessageToStreamlitClient("streamlit:componentReady", { apiVersion: 1 });
}

function setFrameHeight(height) {
  sendMessageToStreamlitClient("streamlit:setFrameHeight", { height });
}

function sendDataToPython(data) {
  sendMessageToStreamlitClient("streamlit:setComponentValue", {
    value: data,
    dataType: "json",
  });
}

function clearButtonInteractionState(buttonEl) {
  if (!buttonEl) return;

  window.requestAnimationFrame(() => {
    if (typeof buttonEl.blur === "function") {
      buttonEl.blur();
    }
  });
}

function applyTheme(theme) {
  const normalizedTheme = theme === "light" ? "light" : "dark";
  rootEl.classList.remove("theme-dark", "theme-light");
  rootEl.classList.add(`theme-${normalizedTheme}`);
}

function applyAssistantState(isOpen) {
  const assistantOpen = Boolean(isOpen);

  if (assistantButtonEl) {
    assistantButtonEl.classList.toggle("active", assistantOpen);
    assistantButtonEl.setAttribute("aria-pressed", assistantOpen ? "true" : "false");
    assistantButtonEl.title = assistantOpen ? "Assistant open" : "Assistant closed";
  }

  if (assistantPulseEl) {
    assistantPulseEl.style.opacity = assistantOpen ? "1" : "1";
  }
}

function applyModeToggleState(buttonEl, isEnabled, activeTitle, inactiveTitle) {
  if (!buttonEl) return;

  const enabled = Boolean(isEnabled);
  buttonEl.classList.toggle("active", enabled);
  buttonEl.setAttribute("aria-pressed", enabled ? "true" : "false");
  buttonEl.title = enabled ? activeTitle : inactiveTitle;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function applyWorkspaceApps(apps, activeAppId) {
  if (!workspaceAppSelectEl) {
    return;
  }

  const normalizedApps = Array.isArray(apps) ? apps : [];
  const markup = normalizedApps
    .map((app) => {
      const appId = typeof app.app_id === "string" ? app.app_id : "";
      const appLabel = typeof app.app_label === "string" ? app.app_label : appId;
      if (!appId) {
        return "";
      }

      return `<option value="${escapeHtml(appId)}">${escapeHtml(appLabel)}</option>`;
    })
    .join("");

  if (markup !== workspaceAppOptionsMarkup) {
    workspaceAppSelectEl.innerHTML = markup;
    workspaceAppOptionsMarkup = markup;
  }

  const hasOptions = normalizedApps.length > 0;
  workspaceAppSelectEl.disabled = !hasOptions;
  if (!hasOptions) {
    return;
  }

  const selectedAppId = typeof activeAppId === "string" && activeAppId
    ? activeAppId
    : workspaceAppSelectEl.options[0]?.value || "";

  if (workspaceAppSelectEl.value !== selectedAppId) {
    workspaceAppSelectEl.value = selectedAppId;
  }
}

function applyModelOptions(options, activeValue) {
  if (!modelSelectEl) {
    return;
  }

  const normalizedOptions = Array.isArray(options) ? options : [];
  const hasOptions = normalizedOptions.length > 0;
  const markup = hasOptions
    ? normalizedOptions
      .map((option) => {
        const value = typeof option.value === "string" ? option.value : "";
        const label = typeof option.label === "string" ? option.label : value;
        if (!value) {
          return "";
        }

        return `<option value="${escapeHtml(value)}">${escapeHtml(label)}</option>`;
      })
      .join("")
    : '<option value="">No models available</option>';

  if (markup !== modelOptionsMarkup) {
    modelSelectEl.innerHTML = markup;
    modelOptionsMarkup = markup;
  }

  modelSelectEl.disabled = !hasOptions;
  if (!hasOptions) {
    modelSelectEl.value = "";
    return;
  }

  const selectedValue = typeof activeValue === "string" && activeValue
    ? activeValue
    : modelSelectEl.options[0]?.value || "";

  if (modelSelectEl.value !== selectedValue) {
    modelSelectEl.value = selectedValue;
  }
}

function onDataFromPython(event) {
  if (!event.data || event.data.type !== "streamlit:render") return;

  const args = event.data.args || {};
  applyTheme(args.theme);
  applyAssistantState(args.assistant_open);
  applyModeToggleState(planningButtonEl, args.planning_enabled, "Planning enabled", "Planning disabled");
  applyModeToggleState(reflectionButtonEl, args.reflection_enabled, "Reflection enabled", "Reflection disabled");
  applyWorkspaceApps(args.available_workspace_apps, args.active_workspace_app_id);
  applyModelOptions(args.available_model_options, args.active_model_option_value);
  if (!args.assistant_open) {
    clearButtonInteractionState(assistantButtonEl);
  }
  setFrameHeight(104);
}

if (themeButtonEl) {
  themeButtonEl.addEventListener("click", () => {
    eventCounter += 1;
    sendDataToPython({ type: "toggle_theme", event_id: eventCounter });
    clearButtonInteractionState(themeButtonEl);
  });
}

if (assistantButtonEl) {
  assistantButtonEl.addEventListener("click", () => {
    eventCounter += 1;
    sendDataToPython({ type: "toggle_assistant", event_id: eventCounter });
    clearButtonInteractionState(assistantButtonEl);
  });
}

if (planningButtonEl) {
  planningButtonEl.addEventListener("click", () => {
    eventCounter += 1;
    sendDataToPython({ type: "toggle_planning", event_id: eventCounter });
    clearButtonInteractionState(planningButtonEl);
  });
}

if (reflectionButtonEl) {
  reflectionButtonEl.addEventListener("click", () => {
    eventCounter += 1;
    sendDataToPython({ type: "toggle_reflection", event_id: eventCounter });
    clearButtonInteractionState(reflectionButtonEl);
  });
}

if (workspaceAppSelectEl) {
  workspaceAppSelectEl.addEventListener("change", () => {
    const selectedAppId = workspaceAppSelectEl.value || "";
    if (!selectedAppId) {
      return;
    }

    eventCounter += 1;
    sendDataToPython({
      type: "set_workspace_app",
      event_id: eventCounter,
      app_id: selectedAppId,
    });
    clearButtonInteractionState(workspaceAppSelectEl);
  });
}

if (modelSelectEl) {
  modelSelectEl.addEventListener("change", () => {
    const selectedOption = modelSelectEl.selectedOptions[0];
    const selectedValue = modelSelectEl.value || "";
    if (!selectedOption || !selectedValue) {
      return;
    }

    const [providerName = "", modelName = ""] = selectedValue.split("::");
    if (!providerName || !modelName) {
      return;
    }

    eventCounter += 1;
    sendDataToPython({
      type: "set_model_selection",
      event_id: eventCounter,
      value: selectedValue,
      provider_name: providerName,
      model_name: modelName,
    });
    clearButtonInteractionState(modelSelectEl);
  });
}

window.addEventListener("message", onDataFromPython);

window.addEventListener("load", () => {
  init();
  setTimeout(() => {
    setFrameHeight(104);
  }, 0);
});
