let shellTabsState = {
  tabs: [],
  activeTab: "",
  themeName: "light",
};

let eventCounter = 0;

function getShellTabsConfig() {
  const config = window.SHELL_TABS_CONFIG || {};
  return {
    tabs: Array.isArray(config.tabs) ? config.tabs : [],
    activeTab: typeof config.activeTab === "string" ? config.activeTab : "",
    themeName: typeof config.themeName === "string" ? config.themeName : "light",
  };
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function buildTabHtml(tabName, activeTab) {
  const isActive = tabName === activeTab;
  const activeClass = isActive ? " is-active" : "";
  const indicatorHtml = isActive ? '<span class="shell-tab-indicator"></span>' : "";
  const safeLabel = escapeHtml(tabName);

  return `
    <button
      type="button"
      class="shell-tab${activeClass}"
      data-shell-tab-value="${safeLabel}"
      aria-pressed="${isActive ? "true" : "false"}"
    >
      ${indicatorHtml}
      <span class="shell-tab-label">${safeLabel}</span>
    </button>
  `;
}

function sendMessageToStreamlitClient(type, data) {
  const outData = Object.assign({
    isStreamlitMessage: true,
    type,
  }, data);

  window.parent.postMessage(outData, "*");
}

function sendDataToPython(data) {
  sendMessageToStreamlitClient("streamlit:setComponentValue", {
    value: data,
    dataType: "json",
  });
}

function setFrameHeight(height) {
  sendMessageToStreamlitClient("streamlit:setFrameHeight", { height });
}

function applyTheme(themeName) {
  const root = document.querySelector("[data-shell-tabs-root]");
  if (!root) {
    return;
  }

  root.setAttribute("data-theme-name", themeName || "light");
}

function renderTabs() {
  const track = document.querySelector("[data-shell-tabs-track]");
  if (!track) {
    return;
  }

  track.innerHTML = shellTabsState.tabs
    .map((tabName) => buildTabHtml(tabName, shellTabsState.activeTab))
    .join("");

  applyTheme(shellTabsState.themeName);

  track.querySelectorAll("[data-shell-tab-value]").forEach((button) => {
    button.addEventListener("click", () => {
      const selectedTab = button.getAttribute("data-shell-tab-value") || "";
      if (!selectedTab || selectedTab === shellTabsState.activeTab) {
        return;
      }

      eventCounter += 1;
      sendDataToPython({
        activeTab: selectedTab,
        eventCounter,
      });
    });
  });

  setFrameHeight(document.documentElement.scrollHeight || document.body.scrollHeight || 0);
}

function onDataFromPython(event) {
  if (!event || !event.data || event.data.type !== "streamlit:render") {
    return;
  }

  const args = event.data.args || {};
  const fallbackConfig = getShellTabsConfig();

  shellTabsState = {
    tabs: Array.isArray(args.tabs) ? args.tabs : fallbackConfig.tabs,
    activeTab: typeof args.activeTab === "string" ? args.activeTab : fallbackConfig.activeTab,
    themeName: typeof args.themeName === "string" ? args.themeName : fallbackConfig.themeName,
  };

  renderTabs();
}

function initializeShellTabs() {
  shellTabsState = getShellTabsConfig();

  window.addEventListener("message", onDataFromPython);

  sendMessageToStreamlitClient("streamlit:componentReady", {
    apiVersion: 1,
  });

  renderTabs();
}

window.addEventListener("DOMContentLoaded", initializeShellTabs);
