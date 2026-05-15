const rootEl = document.getElementById("chat-root");

const floatingShellEl = document.getElementById("floating-shell");
const floatingLauncherShellEl = document.getElementById("floating-launcher-shell");
const floatingLauncherEl = document.getElementById("floating-launcher");
const floatingPopupEl = document.getElementById("floating-popup");
const floatingCloseEl = document.getElementById("floating-close");
const floatingMessagesEl = document.getElementById("messages");
const floatingFormEl = document.getElementById("chat-form");
const floatingInputEl = document.getElementById("message-input");
const floatingSendButtonEl = document.getElementById("send-button");

const fullChatShellEl = document.getElementById("full-chat-shell");
const fullChatHeaderEl = document.getElementById("full-chat-header");
const fullChatTitleEl = document.getElementById("full-chat-title");
const fullChatSubtitleEl = document.getElementById("full-chat-subtitle");
const fullMessagesEl = document.getElementById("full-messages");
const fullFormEl = document.getElementById("full-chat-form");
const fullInputEl = document.getElementById("full-message-input");
const fullSendButtonEl = document.getElementById("full-send-button");

let currentMode = "full";
let floatingOpen = false;
let floatingHasLocalInteraction = false;
let launcherMode = "internal";
let eventCounter = 0;
let isRunning = false;
let configuredPlaceholder = "Ask the assistant something...";
let configuredSendLabel = "Send";
let configuredStopLabel = "Stop";

function createEventId() {
  eventCounter += 1;
  const randomPart = Math.random().toString(36).slice(2, 10);
  return `${Date.now()}-${eventCounter}-${randomPart}`;
}

function getFloatingFrameHeight() {
  if (!floatingOpen) {
    return 96;
  }

  const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 800;
  return Math.min(760, Math.max(620, viewportHeight - 24));
}

function updateFloatingFrameHeight() {
  const nextHeight = getFloatingFrameHeight();
  setFrameHeight(nextHeight);
}

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


function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}


function applyInlineMarkdown(escapedText) {
  return escapedText
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>");
}

function isTableSeparatorLine(line) {
  const trimmed = String(line || "").trim();
  if (!trimmed.includes("|")) return false;

  const normalized = trimmed.replace(/^\|/, "").replace(/\|$/, "");
  const cells = normalized.split("|").map((cell) => cell.trim());
  if (!cells.length) return false;

  return cells.every((cell) => /^:?-{3,}:?$/.test(cell));
}

function isPotentialTableLine(line) {
  const trimmed = String(line || "").trim();
  return trimmed.includes("|");
}

function splitTableCells(line) {
  const trimmed = String(line || "").trim().replace(/^\|/, "").replace(/\|$/, "");
  return trimmed.split("|").map((cell) => applyInlineMarkdown(escapeHtml(cell.trim())));
}

function buildMarkdownTableHtml(tableLines) {
  if (!Array.isArray(tableLines) || tableLines.length < 2) return "";

  const headerCells = splitTableCells(tableLines[0]);
  const bodyLines = tableLines.slice(2);

  const theadHtml = `<thead><tr>${headerCells.map((cell) => `<th>${cell}</th>`).join("")}</tr></thead>`;
  const tbodyHtml = bodyLines.length
    ? `<tbody>${bodyLines
        .map((line) => {
          const cells = splitTableCells(line);
          return `<tr>${cells.map((cell) => `<td>${cell}</td>`).join("")}</tr>`;
        })
        .join("")}</tbody>`
    : "";

  return `<div class="message-table-wrapper"><table>${theadHtml}${tbodyHtml}</table></div>`;
}

function buildHeadingHtml(level, text) {
  const normalizedLevel = Math.min(4, Math.max(1, Number(level) || 1));
  const safeText = applyInlineMarkdown(escapeHtml(text || ""));
  return `<div class="message-heading heading-${normalizedLevel}">${safeText}</div>`;
}


function renderAssistantMarkdown(content) {
  const normalizedContent = String(content ?? "").replace(/\r\n/g, "\n");
  const lines = normalizedContent.split("\n");
  const blocks = [];
  let paragraphLines = [];
  let listItems = [];
  let listType = null;
  let inCodeBlock = false;
  let codeLines = [];
  let tableLines = [];

  function flushParagraph() {
    if (!paragraphLines.length) return;
    const paragraphHtml = applyInlineMarkdown(
      paragraphLines.map((line) => escapeHtml(line)).join("<br>")
    );
    blocks.push(`<p>${paragraphHtml}</p>`);
    paragraphLines = [];
  }

  function flushList() {
    if (!listItems.length || !listType) return;
    const itemsHtml = listItems
      .map((item) => `<li>${applyInlineMarkdown(escapeHtml(item))}</li>`)
      .join("");
    blocks.push(`<${listType}>${itemsHtml}</${listType}>`);
    listItems = [];
    listType = null;
  }

  function flushCodeBlock() {
    if (!codeLines.length) {
      blocks.push("<pre><code></code></pre>");
      return;
    }
    blocks.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
    codeLines = [];
  }

  function flushTable() {
    if (!tableLines.length) return;
    const tableHtml = buildMarkdownTableHtml(tableLines);
    if (tableHtml) {
      blocks.push(tableHtml);
    } else {
      paragraphLines.push(...tableLines);
      flushParagraph();
    }
    tableLines = [];
  }

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const trimmed = line.trim();

    if (trimmed.startsWith("```")) {
      flushParagraph();
      flushList();
      flushTable();

      if (inCodeBlock) {
        flushCodeBlock();
        inCodeBlock = false;
      } else {
        inCodeBlock = true;
        codeLines = [];
      }
      continue;
    }

    if (inCodeBlock) {
      codeLines.push(line);
      continue;
    }

    if (!trimmed) {
      flushParagraph();
      flushList();
      flushTable();
      continue;
    }

    const headingMatch = trimmed.match(/^(#{1,4})\s+(.*)$/);
    if (headingMatch) {
      flushParagraph();
      flushList();
      flushTable();
      const level = headingMatch[1].length;
      const headingText = headingMatch[2];
      blocks.push(buildHeadingHtml(level, headingText));
      continue;
    }

    const nextLine = index + 1 < lines.length ? lines[index + 1] : "";
    if (!tableLines.length && isPotentialTableLine(line) && isTableSeparatorLine(nextLine)) {
      flushParagraph();
      flushList();
      tableLines.push(line, nextLine);
      index += 1;
      continue;
    }

    if (tableLines.length) {
      if (isPotentialTableLine(line)) {
        tableLines.push(line);
        continue;
      }
      flushTable();
    }

    const orderedMatch = trimmed.match(/^\d+\.\s+(.*)$/);
    if (orderedMatch) {
      flushParagraph();
      if (listType && listType !== "ol") {
        flushList();
      }
      listType = "ol";
      listItems.push(orderedMatch[1]);
      continue;
    }

    const unorderedMatch = trimmed.match(/^[-*]\s+(.*)$/);
    if (unorderedMatch) {
      flushParagraph();
      if (listType && listType !== "ul") {
        flushList();
      }
      listType = "ul";
      listItems.push(unorderedMatch[1]);
      continue;
    }

    flushList();
    paragraphLines.push(line);
  }

  if (inCodeBlock) {
    flushCodeBlock();
  }

  flushParagraph();
  flushList();
  flushTable();

  return blocks.join("");
}

function coerceFiniteNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function formatChartNumber(value) {
  const numeric = coerceFiniteNumber(value);
  if (numeric === null) return "";
  if (Math.abs(numeric) >= 1000) {
    return numeric.toLocaleString(undefined, { maximumFractionDigits: 0 });
  }
  if (Math.abs(numeric) >= 100) {
    return numeric.toLocaleString(undefined, { maximumFractionDigits: 1 });
  }
  return numeric.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function getChartDimensions() {
  if (currentMode === "floating") {
    return {
      width: 300,
      height: 210,
      marginTop: 24,
      marginRight: 16,
      marginBottom: 44,
      marginLeft: 42,
    };
  }

  return {
    width: 520,
    height: 300,
    marginTop: 28,
    marginRight: 18,
    marginBottom: 52,
    marginLeft: 52,
  };
}

function getChartInnerDimensions(dimensions) {
  const innerWidth = Math.max(40, dimensions.width - dimensions.marginLeft - dimensions.marginRight);
  const innerHeight = Math.max(40, dimensions.height - dimensions.marginTop - dimensions.marginBottom);
  return { innerWidth, innerHeight };
}

function escapeAttribute(value) {
  return escapeHtml(value).replace(/\n/g, " ");
}

function getChartAxisLabels(block) {
  return {
    xLabel: String(block?.x_label || block?.xField || block?.x_field || ""),
    yLabel: String(block?.y_label || block?.yField || block?.y_field || ""),
  };
}

function getChartSeriesData(block) {
  return Array.isArray(block?.data) ? block.data : [];
}

function getDatumXValue(datum, block) {
  if (!datum || typeof datum !== "object") return null;
  const explicitField = block?.x_field || block?.xField;
  if (explicitField && datum[explicitField] !== undefined) return datum[explicitField];
  if (datum.x !== undefined) return datum.x;
  if (datum.label !== undefined) return datum.label;
  if (datum.category !== undefined) return datum.category;
  return null;
}

function getDatumYValue(datum, block) {
  if (!datum || typeof datum !== "object") return null;
  const explicitField = block?.y_field || block?.yField;
  if (explicitField && datum[explicitField] !== undefined) return datum[explicitField];
  if (datum.y !== undefined) return datum.y;
  if (datum.value !== undefined) return datum.value;
  if (datum.count !== undefined) return datum.count;
  return null;
}

function getHistogramBinLabel(datum) {
  if (!datum || typeof datum !== "object") return "";
  if (datum.label !== undefined) return String(datum.label);
  const x0 = datum.x0 ?? datum.bin_start ?? datum.start;
  const x1 = datum.x1 ?? datum.bin_end ?? datum.end;
  const startValue = coerceFiniteNumber(x0);
  const endValue = coerceFiniteNumber(x1);
  if (startValue === null && endValue === null) return "";
  if (startValue !== null && endValue !== null) {
    return `${formatChartNumber(startValue)}–${formatChartNumber(endValue)}`;
  }
  return formatChartNumber(startValue ?? endValue);
}


function buildChartGridAndAxesSvg(dimensions, maxY, tickCount = 4, minY = 0) {
  const { innerWidth, innerHeight } = getChartInnerDimensions(dimensions);
  const left = dimensions.marginLeft;
  const top = dimensions.marginTop;
  const bottomY = top + innerHeight;
  const rightX = left + innerWidth;
  const safeMinY = Number.isFinite(minY) ? minY : 0;
  const safeMaxY = Number.isFinite(maxY) ? maxY : 1;
  const yRange = safeMaxY - safeMinY || 1;
  const ticks = [];

  for (let index = 0; index <= tickCount; index += 1) {
    const value = safeMinY + (yRange / tickCount) * index;
    const y = bottomY - ((value - safeMinY) / yRange) * innerHeight;
    ticks.push(
      `<line x1="${left}" y1="${y}" x2="${rightX}" y2="${y}" class="chart-grid-line"></line>` +
        `<text x="${left - 8}" y="${y + 4}" text-anchor="end" class="chart-axis-tick">${escapeHtml(formatChartNumber(value))}</text>`
    );
  }

  const zeroValue = Math.min(Math.max(0, safeMinY), safeMaxY);
  const zeroY = bottomY - ((zeroValue - safeMinY) / yRange) * innerHeight;

  return (
    ticks.join("") +
    `<line x1="${left}" y1="${zeroY}" x2="${rightX}" y2="${zeroY}" class="chart-axis-line"></line>` +
    `<line x1="${left}" y1="${top}" x2="${left}" y2="${bottomY}" class="chart-axis-line"></line>`
  );
}

function buildScatterAxesSvg(dimensions, minX, maxX, minY, maxY, tickCount = 4) {
  const { innerWidth, innerHeight } = getChartInnerDimensions(dimensions);
  const left = dimensions.marginLeft;
  const top = dimensions.marginTop;
  const bottomY = top + innerHeight;
  const rightX = left + innerWidth;

  const safeMinX = Number.isFinite(minX) ? minX : 0;
  const safeMaxX = Number.isFinite(maxX) ? maxX : 1;
  const safeMinY = Number.isFinite(minY) ? minY : 0;
  const safeMaxY = Number.isFinite(maxY) ? maxY : 1;
  const xRange = safeMaxX - safeMinX || 1;
  const yRange = safeMaxY - safeMinY || 1;

  const yTicks = [];
  for (let index = 0; index <= tickCount; index += 1) {
    const value = safeMinY + (yRange / tickCount) * index;
    const y = bottomY - ((value - safeMinY) / yRange) * innerHeight;
    yTicks.push(
      `<line x1="${left}" y1="${y}" x2="${rightX}" y2="${y}" class="chart-grid-line"></line>` +
        `<text x="${left - 8}" y="${y + 4}" text-anchor="end" class="chart-axis-tick">${escapeHtml(formatChartNumber(value))}</text>`
    );
  }

  const xTicks = [];
  for (let index = 0; index <= tickCount; index += 1) {
    const value = safeMinX + (xRange / tickCount) * index;
    const x = left + ((value - safeMinX) / xRange) * innerWidth;
    xTicks.push(
      `<text x="${x}" y="${bottomY + 18}" text-anchor="middle" class="chart-axis-tick">${escapeHtml(formatChartNumber(value))}</text>`
    );
  }

  return (
    yTicks.join("") +
    xTicks.join("") +
    `<line x1="${left}" y1="${bottomY}" x2="${rightX}" y2="${bottomY}" class="chart-axis-line"></line>` +
    `<line x1="${left}" y1="${top}" x2="${left}" y2="${bottomY}" class="chart-axis-line"></line>`
  );
}

function buildBarChartSvg(block) {
  const dimensions = getChartDimensions();
  const { innerWidth, innerHeight } = getChartInnerDimensions(dimensions);
  const data = getChartSeriesData(block)
    .map((datum) => {
      const xValue = getDatumXValue(datum, block);
      const yValue = coerceFiniteNumber(getDatumYValue(datum, block));
      if (xValue === null || yValue === null) return null;
      return { label: String(xValue), value: yValue };
    })
    .filter(Boolean);

  if (!data.length) return "";

  const minY = Math.min(...data.map((item) => item.value), 0);
  const maxY = Math.max(...data.map((item) => item.value), 0);
  const yRange = maxY - minY || 1;
  const step = innerWidth / data.length;
  const barWidth = Math.max(12, step * 0.64);
  const bottomY = dimensions.marginTop + innerHeight;
  const left = dimensions.marginLeft;
  const zeroY = bottomY - ((0 - minY) / yRange) * innerHeight;
  const bars = data
    .map((item, index) => {
      const scaledHeight = (Math.abs(item.value) / yRange) * innerHeight;
      const x = left + index * step + (step - barWidth) / 2;
      const y = item.value >= 0 ? zeroY - scaledHeight : zeroY;
      const labelX = left + index * step + step / 2;
      return (
        `<g>` +
        `<rect x="${x}" y="${y}" width="${barWidth}" height="${scaledHeight}" rx="6" class="chart-bar"></rect>` +
        `<title>${escapeHtml(`${item.label}: ${formatChartNumber(item.value)}`)}</title>` +
        `</g>` +
        `<text x="${labelX}" y="${bottomY + 18}" text-anchor="middle" class="chart-axis-tick">${escapeHtml(item.label)}</text>`
      );
    })
    .join("");

  return buildChartSvgFrame(block, dimensions, maxY, bars, "", minY);
}

// Helper functions for line chart axis label formatting and stepping
function getLineLabelStep(pointCount) {
  if (pointCount <= 5) return 1;
  return Math.ceil(pointCount / 5);
}

function formatLineAxisLabel(label, totalPoints) {
  const raw = String(label ?? "");
  const date = new Date(raw);
  if (Number.isNaN(date.getTime())) {
    return raw;
  }

  const year = date.getUTCFullYear();
  const month = String(date.getUTCMonth() + 1).padStart(2, "0");
  const day = String(date.getUTCDate()).padStart(2, "0");

  if (totalPoints > 24) {
    return String(year);
  }
  if (totalPoints > 8) {
    return `${year}-${month}`;
  }
  return `${year}-${month}-${day}`;
}

function shouldShowLineAxisLabel(index, totalPoints, labelStep) {
  return index === totalPoints - 1 || index % labelStep === 0;
}

function buildLineChartSvg(block) {
  const dimensions = getChartDimensions();
  const { innerWidth, innerHeight } = getChartInnerDimensions(dimensions);
  const data = getChartSeriesData(block)
    .map((datum, index) => {
      const xValue = getDatumXValue(datum, block);
      const yValue = coerceFiniteNumber(getDatumYValue(datum, block));
      if (xValue === null || yValue === null) return null;
      return { label: String(xValue), value: yValue, index };
    })
    .filter(Boolean);

  if (data.length < 2) return "";

  const minY = Math.min(...data.map((item) => item.value), 0);
  const maxY = Math.max(...data.map((item) => item.value), 0);
  const yRange = maxY - minY || 1;
  const bottomY = dimensions.marginTop + innerHeight;
  const left = dimensions.marginLeft;
  const step = data.length > 1 ? innerWidth / (data.length - 1) : innerWidth;
  const points = data.map((item, index) => {
    const x = left + index * step;
    const y = bottomY - ((item.value - minY) / yRange) * innerHeight;
    return { ...item, x, y };
  });
  const labelStep = getLineLabelStep(points.length);

  const path = points.map((point, index) => `${index === 0 ? "M" : "L"}${point.x} ${point.y}`).join(" ");
  const pointHtml = points
    .map((point, index) => {
      if (!shouldShowLineAxisLabel(index, points.length, labelStep)) {
        return "";
      }
      const formattedLabel = formatLineAxisLabel(point.label, points.length);
      return `<text x="${point.x}" y="${bottomY + 18}" text-anchor="middle" class="chart-axis-tick">${escapeHtml(formattedLabel)}</text>`;
    })
    .join("");

  const content = `<path d="${path}" fill="none" class="chart-line"></path>${pointHtml}`;
  return buildChartSvgFrame(block, dimensions, maxY, content, "", minY);
}

function buildScatterChartSvg(block) {
  const dimensions = getChartDimensions();
  const { innerWidth, innerHeight } = getChartInnerDimensions(dimensions);
  const data = getChartSeriesData(block)
    .map((datum) => {
      const xValue = coerceFiniteNumber(getDatumXValue(datum, block));
      const yValue = coerceFiniteNumber(getDatumYValue(datum, block));
      if (xValue === null || yValue === null) return null;
      const label = datum.label !== undefined ? String(datum.label) : `${formatChartNumber(xValue)}, ${formatChartNumber(yValue)}`;
      return { xValue, yValue, label };
    })
    .filter(Boolean);

  if (!data.length) return "";

  const minX = Math.min(...data.map((item) => item.xValue));
  const maxX = Math.max(...data.map((item) => item.xValue));
  const minY = Math.min(...data.map((item) => item.yValue));
  const maxY = Math.max(...data.map((item) => item.yValue));
  const safeMinX = Number.isFinite(minX) ? minX : 0;
  const safeMaxX = Number.isFinite(maxX) ? maxX : 1;
  const safeMinY = Number.isFinite(minY) ? minY : 0;
  const safeMaxY = Number.isFinite(maxY) ? maxY : 1;
  const xRange = safeMaxX - safeMinX || 1;
  const yRange = safeMaxY - safeMinY || 1;
  const left = dimensions.marginLeft;
  const top = dimensions.marginTop;
  const bottomY = top + innerHeight;

  const points = data
    .map((item) => {
      const cx = left + ((item.xValue - safeMinX) / xRange) * innerWidth;
      const cy = bottomY - ((item.yValue - safeMinY) / yRange) * innerHeight;
      return `<g><circle cx="${cx}" cy="${cy}" r="4.5" class="chart-scatter-point"></circle><title>${escapeHtml(`${item.label}`)}</title></g>`;
    })
    .join("");

  const axes = buildScatterAxesSvg(dimensions, safeMinX, safeMaxX, safeMinY, safeMaxY);
  return buildChartSvgFrame(block, dimensions, safeMaxY, points, axes);
}

function getHistogramLabelStep(binCount) {
  if (binCount <= 8) return 1;
  if (binCount <= 16) return 2;
  if (binCount <= 24) return 3;
  return 4;
}

function buildHistogramChartSvg(block) {
  const dimensions = getChartDimensions();
  const { innerWidth, innerHeight } = getChartInnerDimensions(dimensions);
  const data = getChartSeriesData(block)
    .map((datum) => {
      const count = coerceFiniteNumber(datum?.count ?? datum?.y ?? datum?.value);
      const label = getHistogramBinLabel(datum);
      if (count === null || !label) return null;
      return { label, value: count };
    })
    .filter(Boolean);

  if (!data.length) return "";

  const maxY = Math.max(...data.map((item) => item.value), 0);
  const step = innerWidth / data.length;
  const barWidth = Math.max(10, step * 0.9);
  const bottomY = dimensions.marginTop + innerHeight;
  const left = dimensions.marginLeft;
  const labelStep = getHistogramLabelStep(data.length);
  const bars = data
    .map((item, index) => {
      const scaledHeight = maxY > 0 ? (item.value / maxY) * innerHeight : 0;
      const x = left + index * step + (step - barWidth) / 2;
      const y = bottomY - scaledHeight;
      const labelX = left + index * step + step / 2;
      const shouldShowLabel = index % labelStep === 0;
      const labelHtml = shouldShowLabel
        ? `<text x="${labelX}" y="${bottomY + 18}" text-anchor="middle" class="chart-axis-tick">${escapeHtml(item.label)}</text>`
        : "";
      return (
        `<g>` +
        `<rect x="${x}" y="${y}" width="${barWidth}" height="${scaledHeight}" rx="2" class="chart-histogram-bar"></rect>` +
        `<title>${escapeHtml(`${item.label}: ${formatChartNumber(item.value)}`)}</title>` +
        `</g>` +
        labelHtml
      );
    })
    .join("");

  return buildChartSvgFrame(block, dimensions, maxY, bars);
}

function buildChartSvgFrame(block, dimensions, maxY, contentSvg, axesSvg = "", minY = 0) {
  const { xLabel, yLabel } = getChartAxisLabels(block);
  const computedAxesSvg = axesSvg || buildChartGridAndAxesSvg(dimensions, maxY, 4, minY);
  const viewBox = `0 0 ${dimensions.width} ${dimensions.height}`;
  const xLabelHtml = xLabel
    ? `<text x="${dimensions.marginLeft + (dimensions.width - dimensions.marginLeft - dimensions.marginRight) / 2}" y="${dimensions.height - 10}" text-anchor="middle" class="chart-axis-label">${escapeHtml(xLabel)}</text>`
    : "";
  return `<svg class="chat-chart-svg chat-chart-${escapeAttribute(block?.chart_type || block?.chartType || "bar")}" viewBox="${viewBox}" role="img" aria-label="${escapeAttribute(block?.title || "Chart")}">${computedAxesSvg}${contentSvg}${xLabelHtml}</svg>`;
}

function buildChartSvg(block) {
  const chartType = String(block?.chart_type || block?.chartType || "").trim().toLowerCase();
  if (chartType === "bar") return buildBarChartSvg(block);
  if (chartType === "line") return buildLineChartSvg(block);
  if (chartType === "scatter") return buildScatterChartSvg(block);
  if (chartType === "histogram") return buildHistogramChartSvg(block);
  return "";
}

function buildChartBlockHtml(block) {
  const svgHtml = buildChartSvg(block);
  if (!svgHtml) return "";

  const title = String(block?.title || "Chart");
  const subtitle = String(block?.subtitle || "");
  const summary = String(block?.summary || "");

  const titleHtml = `<div class="message-chart-title">${escapeHtml(title)}</div>`;
  const subtitleHtml = subtitle
    ? `<div class="message-chart-subtitle">${escapeHtml(subtitle)}</div>`
    : "";
  const summaryHtml = summary
    ? `<div class="message-chart-summary">${escapeHtml(summary)}</div>`
    : "";

  return `<div class="message-block message-block-chart">${titleHtml}${subtitleHtml}<div class="message-chart-body">${svgHtml}</div>${summaryHtml}</div>`;
}

function buildMessageBlocksHtml(message) {
  const blocks = Array.isArray(message?.blocks) ? message.blocks : [];
  if (!blocks.length) return "";

  return blocks
    .map((block) => {
      const blockType = String(block?.type || "").trim().toLowerCase();
      if (blockType === "chart") {
        return buildChartBlockHtml(block);
      }
      return "";
    })
    .filter(Boolean)
    .join("");
}

function buildMessageHtml(message) {
  const role = message && message.role === "user" ? "user" : "assistant";
  const content = message?.content ?? "";
  const toolStatusText =
    role === "user" && typeof message?.meta?.run_status_text === "string"
      ? message.meta.run_status_text.trim()
      : role === "user" && typeof message?.meta?.tool_status_text === "string"
      ? message.meta.tool_status_text.trim()
      : "";
  const toolStatusState =
    role === "user" && typeof message?.meta?.run_status_state === "string"
      ? message.meta.run_status_state.trim().toLowerCase()
      : role === "user" && typeof message?.meta?.tool_status_state === "string"
      ? message.meta.tool_status_state.trim().toLowerCase()
      : "";
  const safeContent = role === "assistant" ? renderAssistantMarkdown(content) : escapeHtml(content);
  const contentHtml = safeContent
    ? `<div class="message-content">${safeContent}</div>`
    : "";
  const blocksHtml = role === "assistant" ? buildMessageBlocksHtml(message) : "";
  const statusLines =
    role === "user" && Array.isArray(message?.meta?.run_status_lines)
      ? message.meta.run_status_lines
          .map((line) => String(line || "").trim())
          .filter(Boolean)
      : toolStatusText
      ? [toolStatusText]
      : [];
  const statusHtml = statusLines.length
    ? `<div class="message-status-stack">${statusLines
        .map((line) => `<div class="message-status ${escapeAttribute(toolStatusState || "info")}">${escapeHtml(line)}</div>`)
        .join("")}</div>`
    : "";

  return `<div class="message-row ${role}">
    <div class="message-stack ${role}">
      <div class="message-bubble ${role}">
        <div class="message-role">${role}</div>
        ${contentHtml}
        ${blocksHtml}
      </div>
      ${statusHtml}
    </div>
  </div>`;
}

function getActiveMessagesEl() {
  return currentMode === "floating" ? floatingMessagesEl : fullMessagesEl;
}

function getActiveInputEl() {
  return currentMode === "floating" ? floatingInputEl : fullInputEl;
}

function getActiveSendButtonEl() {
  return currentMode === "floating" ? floatingSendButtonEl : fullSendButtonEl;
}

function scrollMessagesToBottom() {
  const activeMessagesEl = getActiveMessagesEl();
  if (!activeMessagesEl) return;

  requestAnimationFrame(() => {
    activeMessagesEl.scrollTop = activeMessagesEl.scrollHeight;
  });

  setTimeout(() => {
    activeMessagesEl.scrollTop = activeMessagesEl.scrollHeight;
  }, 50);
}

function renderMessages(messages) {
  const html = messages.map(buildMessageHtml).join("");

  if (floatingMessagesEl) {
    floatingMessagesEl.innerHTML = html;
  }
  if (fullMessagesEl) {
    fullMessagesEl.innerHTML = html;
  }

  scrollMessagesToBottom();
}

function applyTheme(theme) {
  const normalizedTheme = theme === "light" ? "light" : "dark";
  rootEl.classList.remove("theme-dark", "theme-light");
  rootEl.classList.add(`theme-${normalizedTheme}`);
}

function applyPlaceholder(placeholder) {
  configuredPlaceholder = placeholder || "Ask the assistant something...";
  updateInputControls();
}

function applySendLabel(sendLabel) {
  configuredSendLabel = sendLabel || "Send";
  updateInputControls();
}

function applyRunningState(nextIsRunning, nextStopLabel) {
  isRunning = Boolean(nextIsRunning);
  configuredStopLabel = nextStopLabel || "Stop";
  updateInputControls();
}

function resizeMessageInput(inputEl) {
  if (!inputEl || inputEl.getClientRects().length === 0) return;

  inputEl.classList.remove("is-scrollable");
  inputEl.style.height = "auto";

  const computedStyle = window.getComputedStyle(inputEl);
  const maxHeight = Number.parseFloat(computedStyle.maxHeight);
  const scrollHeight = inputEl.scrollHeight;
  if (!scrollHeight) return;

  const nextHeight =
    Number.isFinite(maxHeight) && maxHeight > 0
      ? Math.min(scrollHeight, maxHeight)
      : scrollHeight;

  inputEl.style.height = `${nextHeight}px`;
  inputEl.classList.toggle("is-scrollable", scrollHeight > nextHeight + 1);
}

function resizeMessageInputs() {
  [floatingInputEl, fullInputEl].forEach((inputEl) => {
    resizeMessageInput(inputEl);
  });
}

function updateInputControls() {
  const placeholderValue = isRunning ? "Agent is running..." : configuredPlaceholder;
  const buttonValue = isRunning ? configuredStopLabel : configuredSendLabel;

  [floatingInputEl, fullInputEl].forEach((inputEl) => {
    if (!inputEl) return;
    inputEl.placeholder = placeholderValue;
    inputEl.readOnly = isRunning;
    inputEl.classList.toggle("is-running", isRunning);
    inputEl.setAttribute("aria-readonly", isRunning ? "true" : "false");
    resizeMessageInput(inputEl);
  });

  [floatingSendButtonEl, fullSendButtonEl].forEach((buttonEl) => {
    if (!buttonEl) return;
    buttonEl.textContent = buttonValue;
    buttonEl.classList.toggle("is-stop", isRunning);
    buttonEl.setAttribute("aria-label", buttonValue);
  });
}

function applyHeader(mode, title, subtitle) {
  const shouldShowHeader = mode === "pane";

  if (fullChatHeaderEl) {
    fullChatHeaderEl.classList.toggle("visible", shouldShowHeader);
    fullChatHeaderEl.setAttribute("aria-hidden", shouldShowHeader ? "false" : "true");
  }

  if (fullChatTitleEl) {
    fullChatTitleEl.textContent = shouldShowHeader ? title || "Assistant" : "";
  }

  if (fullChatSubtitleEl) {
    fullChatSubtitleEl.textContent = shouldShowHeader
      ? subtitle || "Context-aware, tool-enabled chat"
      : "";
  }
}

function applyMode(mode) {
  currentMode = mode === "floating" ? "floating" : "full";
  const visualMode = mode === "pane" ? "pane" : currentMode;
  rootEl.classList.remove("mode-full", "mode-floating", "mode-pane");
  rootEl.classList.add(`mode-${visualMode}`);

  if (floatingShellEl) {
    floatingShellEl.setAttribute("aria-hidden", currentMode === "floating" ? "false" : "true");
  }
  if (fullChatShellEl) {
    fullChatShellEl.setAttribute("aria-hidden", currentMode === "full" ? "false" : "true");
  }
}

function applyLauncherMode(mode) {
  launcherMode = mode === "external" ? "external" : "internal";
  rootEl.classList.remove("launcher-internal", "launcher-external");
  rootEl.classList.add(`launcher-${launcherMode}`);

  if (floatingLauncherShellEl) {
    floatingLauncherShellEl.setAttribute(
      "aria-hidden",
      launcherMode === "external" ? "true" : "false"
    );
  }
}

function setFloatingOpen(isOpen, source = "python") {
  floatingOpen = Boolean(isOpen);

  if (source === "local") {
    floatingHasLocalInteraction = true;
  }

  if (!floatingPopupEl || !floatingLauncherEl) return;

  floatingPopupEl.classList.toggle("open", floatingOpen);
  floatingPopupEl.setAttribute("aria-hidden", floatingOpen ? "false" : "true");
  floatingLauncherEl.setAttribute("aria-expanded", floatingOpen ? "true" : "false");

  if (!floatingOpen) {
    clearButtonInteractionState(floatingLauncherEl);
    clearButtonInteractionState(floatingCloseEl);
  }
}

function onDataFromPython(event) {
  if (!event.data || event.data.type !== "streamlit:render") return;

  const args = event.data.args || {};
  const messages = Array.isArray(args.messages) ? args.messages : [];
  const mode = args.mode || "full";
  const nextLauncherMode = args.launcher_mode || "internal";
  const nextHeight = Number(args.height) || 620;

  applyTheme(args.theme);
  applyMode(mode);
  applyHeader(mode, args.header_title, args.header_subtitle);
  applyLauncherMode(nextLauncherMode);
  applyPlaceholder(args.placeholder);
  applySendLabel(args.send_label);
  applyRunningState(Boolean(args.is_running), args.stop_label);
  renderMessages(messages);

  if (currentMode === "full") {
    rootEl.style.height = `${nextHeight}px`;
    setFrameHeight(nextHeight);
  } else {
    const pythonOpen = Boolean(args.is_open);

    rootEl.style.height = `${getFloatingFrameHeight()}px`;

    if (launcherMode === "external") {
      setFloatingOpen(pythonOpen, "python");
      floatingHasLocalInteraction = false;
    } else if (!floatingHasLocalInteraction) {
      setFloatingOpen(pythonOpen, "python");
    } else {
      setFloatingOpen(floatingOpen, "python");
    }

    updateFloatingFrameHeight();
  }
}

function submitMessage(inputEl) {
  if (isRunning) {
    sendDataToPython({ type: "stop_active_run", event_id: createEventId() });
    clearButtonInteractionState(getActiveSendButtonEl());
    return;
  }

  const text = inputEl.value.trim();
  if (!text) return;

  sendDataToPython({ type: "submit_message", event_id: createEventId(), value: text });
  inputEl.value = "";
  resizeMessageInput(inputEl);
  if (currentMode === "floating") {
    clearButtonInteractionState(floatingSendButtonEl);
  } else {
    clearButtonInteractionState(fullSendButtonEl);
  }
  inputEl.focus();
}

function handleFormSubmit(event, inputEl) {
  event.preventDefault();
  submitMessage(inputEl);
}

function handleInputKeydown(event, inputEl) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    submitMessage(inputEl);
  }
}

function handleInputChange(inputEl) {
  resizeMessageInput(inputEl);
}

if (floatingFormEl) {
  floatingFormEl.addEventListener("submit", (event) => {
    handleFormSubmit(event, floatingInputEl);
  });
}

if (fullFormEl) {
  fullFormEl.addEventListener("submit", (event) => {
    handleFormSubmit(event, fullInputEl);
  });
}

if (floatingInputEl) {
  floatingInputEl.addEventListener("keydown", (event) => {
    handleInputKeydown(event, floatingInputEl);
  });
  floatingInputEl.addEventListener("input", () => {
    handleInputChange(floatingInputEl);
  });
}

if (fullInputEl) {
  fullInputEl.addEventListener("keydown", (event) => {
    handleInputKeydown(event, fullInputEl);
  });
  fullInputEl.addEventListener("input", () => {
    handleInputChange(fullInputEl);
  });
}

if (floatingLauncherEl) {
  floatingLauncherEl.addEventListener("click", () => {
    if (currentMode !== "floating" || launcherMode === "external") return;
    setFloatingOpen(!floatingOpen, "local");
    clearButtonInteractionState(floatingLauncherEl);
    rootEl.style.height = `${getFloatingFrameHeight()}px`;
    updateFloatingFrameHeight();
  });
}

if (floatingCloseEl) {
  floatingCloseEl.addEventListener("click", () => {
    if (currentMode !== "floating") return;
    if (launcherMode === "external") {
      setFloatingOpen(false, "local");
      clearButtonInteractionState(floatingCloseEl);
      rootEl.style.height = `${getFloatingFrameHeight()}px`;
      updateFloatingFrameHeight();
      sendDataToPython({ type: "toggle_assistant_close", event_id: createEventId() });
      return;
    }
    setFloatingOpen(false, "local");
    clearButtonInteractionState(floatingCloseEl);
    rootEl.style.height = `${getFloatingFrameHeight()}px`;
    updateFloatingFrameHeight();
  });
}

window.addEventListener("message", onDataFromPython);

window.addEventListener("load", () => {
  init();
  resizeMessageInputs();
  setTimeout(() => {
    setFrameHeight(document.documentElement.clientHeight || 96);
  }, 0);
});
