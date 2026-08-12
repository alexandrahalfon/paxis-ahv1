/**
 * chartUtils.js — Shared chart rendering for Paxis
 *
 * Features:
 *   - Smart chart type selection based on data shape
 *   - Data labels rendered directly on bars
 *   - Forest plot for hazard ratios + CI
 *   - Horizontal bars for toxicity / long-label data
 *   - Grouped survival endpoint charts (OS/PFS/DFS in one)
 *   - Author-shorthand labels ("Al-Sarraf 1998" vs truncated titles)
 *   - Consistent color palette with study-level assignment
 *
 * Usage:
 *   const mgr = new ChartManager();
 *   mgr.render('canvasId', chartConfig);
 *   mgr.renderForestPlot('canvasId', forestData);
 *   mgr.renderGroupedSurvival('canvasId', survivalData);
 *   mgr.destroyAll();
 */

// ─── Color Palette ───────────────────────────────────────────────
const PALETTE = {
    studies: [
        { bg: 'rgba(37, 99, 235, 0.78)',  border: 'rgba(37, 99, 235, 1)',  light: 'rgba(37, 99, 235, 0.12)' },
        { bg: 'rgba(5, 150, 105, 0.78)',   border: 'rgba(5, 150, 105, 1)',  light: 'rgba(5, 150, 105, 0.12)' },
        { bg: 'rgba(217, 119, 6, 0.78)',   border: 'rgba(217, 119, 6, 1)',  light: 'rgba(217, 119, 6, 0.12)' },
        { bg: 'rgba(220, 38, 38, 0.78)',   border: 'rgba(220, 38, 38, 1)',  light: 'rgba(220, 38, 38, 0.12)' },
    ],
    endpoints: {
        'OS':  { bg: 'rgba(37, 99, 235, 0.75)', border: 'rgba(37, 99, 235, 1)' },
        'PFS': { bg: 'rgba(5, 150, 105, 0.75)', border: 'rgba(5, 150, 105, 1)' },
        'DFS': { bg: 'rgba(217, 119, 6, 0.75)', border: 'rgba(217, 119, 6, 1)' },
        'LC':  { bg: 'rgba(124, 58, 237, 0.75)', border: 'rgba(124, 58, 237, 1)' },
    },
    forestLine: 'rgba(51, 65, 85, 0.7)',
    forestDiamond: 'rgba(37, 99, 235, 0.9)',
    gridLine: 'rgba(226, 232, 240, 0.7)',
    noEffect: 'rgba(220, 38, 38, 0.45)',
};

const FONT = {
    family: "'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif",
    sizeXs: 10, sizeSm: 11, sizeMd: 12, sizeLg: 13,
};

// ─── Label Helpers ───────────────────────────────────────────────
function shortenLabel(label, maxLen) {
    maxLen = maxLen || 28;
    if (!label) return '—';
    if (/^[A-Z][a-z]+ et al/i.test(label) && label.length < 40) return label;
    if (label.length <= maxLen) return label;
    return label.slice(0, maxLen - 1).trim() + '…';
}

function authorShorthand(study) {
    if (study.author && study.year) {
        var surname = study.author.replace(/ et al\.?$/i, '').trim().split(' ').pop();
        return surname + ' ' + study.year;
    }
    return shortenLabel(study.title || study.doc_id || 'Study', 24);
}

// ─── Data Label Plugin ───────────────────────────────────────────
function makeDataLabelPlugin(unit, horizontal) {
    return {
        id: 'exueedDataLabels',
        afterDatasetsDraw: function(chart) {
            var ctx = chart.ctx;
            ctx.save();
            ctx.font = '600 ' + FONT.sizeSm + 'px ' + FONT.family;
            ctx.textBaseline = 'middle';

            chart.data.datasets.forEach(function(dataset, dsIndex) {
                var meta = chart.getDatasetMeta(dsIndex);
                if (meta.hidden) return;

                meta.data.forEach(function(bar, index) {
                    var value = dataset.data[index];
                    if (value === 0 || value == null) return;

                    var displayVal = Number.isInteger(value) ? value : value.toFixed(1);
                    var text = displayVal + (unit || '');

                    if (horizontal) {
                        ctx.textAlign = 'left';
                        ctx.fillStyle = 'rgba(51, 65, 85, 0.85)';
                        ctx.fillText(text, bar.x + 6, bar.y);
                    } else {
                        ctx.textAlign = 'center';
                        ctx.fillStyle = 'rgba(51, 65, 85, 0.85)';
                        ctx.fillText(text, bar.x, bar.y - 8);
                    }
                });
            });
            ctx.restore();
        }
    };
}

// ─── Chart Manager ───────────────────────────────────────────────
function ChartManager() {
    this.instances = [];
}

ChartManager.prototype.destroyAll = function() {
    this.instances.forEach(function(c) { try { c.destroy(); } catch(_){} });
    this.instances = [];
};

/**
 * Should this chart be horizontal?
 */
ChartManager.prototype._shouldBeHorizontal = function(cfg) {
    var labels = cfg.labels || [];
    var avgLen = labels.reduce(function(s, l) { return s + (l || '').length; }, 0) / (labels.length || 1);
    if (/toxicit|adverse|side effect|grade/i.test(cfg.title || '')) return true;
    if (avgLen > 22 && labels.length > 2) return true;
    return false;
};

/**
 * Build scales configuration
 */
ChartManager.prototype._buildScales = function(chartType, indexAxis, unit, isHorizontal) {
    if (chartType === 'pie' || chartType === 'doughnut') return undefined;
    var valueAxis = isHorizontal ? 'x' : 'y';
    var categoryAxis = isHorizontal ? 'y' : 'x';
    var scales = {};
    scales[valueAxis] = {
        beginAtZero: true,
        title: {
            display: !!unit,
            text: unit || '',
            font: { family: FONT.family, size: FONT.sizeSm },
            color: 'rgba(100, 116, 139, 0.9)',
        },
        grid: { color: PALETTE.gridLine, drawBorder: false },
        ticks: {
            font: { family: FONT.family, size: FONT.sizeXs },
            color: 'rgba(100, 116, 139, 0.8)',
            padding: 4,
        },
        border: { display: false },
    };
    scales[categoryAxis] = {
        grid: { display: false },
        ticks: {
            font: { family: FONT.family, size: FONT.sizeSm },
            color: 'rgba(51, 65, 85, 0.85)',
            padding: 6,
        },
        border: { display: false },
    };
    return scales;
};

/**
 * Main render — takes backend ChartArtifact shape
 * @param {string} canvasId
 * @param {Object} cfg  { type, title, labels, datasets, unit, source }
 * @param {Object} opts { horizontal, showDataLabels }
 */
ChartManager.prototype.render = function(canvasId, cfg, opts) {
    opts = opts || {};
    var canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === 'undefined') {
        console.warn('[ChartUtils] canvas or Chart.js missing:', canvasId);
        return null;
    }

    var forceHorizontal = opts.horizontal != null ? opts.horizontal : this._shouldBeHorizontal(cfg);
    var chartType = forceHorizontal ? 'bar' : (cfg.type || 'bar');
    var indexAxis = forceHorizontal ? 'y' : 'x';
    var ctx = canvas.getContext('2d');
    var showLabels = opts.showDataLabels != null ? opts.showDataLabels : true;
    var unit = cfg.unit || '';

    var datasets = (cfg.datasets || []).map(function(ds, i) {
        var color = PALETTE.studies[i % PALETTE.studies.length];
        return {
            label: ds.label || ('Dataset ' + (i + 1)),
            data: ds.data || [],
            backgroundColor: ds.backgroundColor || color.bg,
            borderColor: ds.borderColor || color.border,
            borderWidth: 1,
            borderRadius: forceHorizontal ? 3 : 4,
            maxBarThickness: forceHorizontal ? 22 : 56,
        };
    });

    var labels = (cfg.labels || []).map(function(l) { return shortenLabel(l); });

    var config = {
        type: chartType,
        data: { labels: labels, datasets: datasets },
        options: {
            indexAxis: indexAxis,
            responsive: true,
            maintainAspectRatio: false,
            layout: {
                padding: {
                    top: showLabels ? 24 : 8,
                    right: forceHorizontal && showLabels ? 52 : 12,
                    bottom: 4,
                    left: 4,
                }
            },
            plugins: {
                legend: {
                    display: datasets.length > 1,
                    position: 'top',
                    labels: {
                        font: { family: FONT.family, size: FONT.sizeSm },
                        boxWidth: 12, padding: 12,
                        usePointStyle: true, pointStyle: 'rectRounded',
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.92)',
                    titleFont: { family: FONT.family, size: FONT.sizeMd, weight: '600' },
                    bodyFont: { family: FONT.family, size: FONT.sizeSm },
                    padding: 10, cornerRadius: 6,
                    callbacks: {
                        label: function(context) {
                            var lbl = context.dataset.label || '';
                            if (lbl) lbl += ': ';
                            var val = forceHorizontal ? context.parsed.x : context.parsed.y;
                            lbl += val;
                            if (unit) lbl += unit;
                            return lbl;
                        }
                    }
                },
            },
            scales: this._buildScales(chartType, indexAxis, unit, forceHorizontal),
            animation: { duration: 600, easing: 'easeOutQuart' }
        },
        plugins: showLabels ? [makeDataLabelPlugin(unit, forceHorizontal)] : [],
    };

    try {
        var chart = new Chart(ctx, config);
        this.instances.push(chart);
        return chart;
    } catch (err) {
        console.error('[ChartUtils] render error:', err);
        return null;
    }
};

/**
 * Forest Plot for hazard ratios
 * @param {string} canvasId
 * @param {Object} data { title, studies: [{ label, hr, ciLow, ciHigh, weight? }] }
 */
ChartManager.prototype.renderForestPlot = function(canvasId, data) {
    var canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === 'undefined') return null;
    var ctx = canvas.getContext('2d');

    var studies = data.studies || [];
    var labels = studies.map(function(s) { return shortenLabel(s.label, 36); });
    var hrs = studies.map(function(s) { return s.hr; });
    var ciLows = studies.map(function(s) { return s.ciLow; });
    var ciHighs = studies.map(function(s) { return s.ciHigh; });

    var config = {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'HR',
                data: hrs.map(function(hr, i) { return { x: hr, y: i }; }),
                backgroundColor: PALETTE.forestDiamond,
                borderColor: PALETTE.forestDiamond,
                pointRadius: studies.map(function(s) { return Math.max(5, Math.min(s.weight || 6, 12)); }),
                pointStyle: 'rectRot',
                showLine: false,
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            layout: { padding: { top: 8, bottom: 8, left: 8, right: 80 } },
            scales: {
                x: {
                    title: {
                        display: true, text: 'Hazard Ratio',
                        font: { family: FONT.family, size: FONT.sizeMd, weight: '600' },
                    },
                    grid: { color: PALETTE.gridLine },
                    ticks: { font: { family: FONT.family, size: FONT.sizeXs } },
                },
                y: {
                    type: 'linear', reverse: true,
                    min: -0.5, max: studies.length - 0.5,
                    ticks: {
                        stepSize: 1,
                        callback: function(val) { return labels[val] || ''; },
                        font: { family: FONT.family, size: FONT.sizeSm },
                    },
                    grid: { display: false },
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.92)',
                    callbacks: {
                        title: function(items) { return labels[items[0] && items[0].parsed ? items[0].parsed.y : 0] || ''; },
                        label: function(item) {
                            var i = item.parsed.y;
                            return 'HR ' + hrs[i] + ' (' + ciLows[i] + '–' + ciHighs[i] + ')';
                        }
                    }
                }
            },
        },
        plugins: [
            // CI horizontal lines with serifs
            {
                id: 'forestCI',
                afterDatasetsDraw: function(chart) {
                    var chartCtx = chart.ctx;
                    var x = chart.scales.x, y = chart.scales.y;
                    chartCtx.save();
                    chartCtx.strokeStyle = PALETTE.forestLine;
                    chartCtx.lineWidth = 1.5;
                    studies.forEach(function(s, i) {
                        var yPos = y.getPixelForValue(i);
                        var xLow = x.getPixelForValue(s.ciLow);
                        var xHigh = x.getPixelForValue(s.ciHigh);
                        chartCtx.beginPath(); chartCtx.moveTo(xLow, yPos); chartCtx.lineTo(xHigh, yPos); chartCtx.stroke();
                        chartCtx.beginPath(); chartCtx.moveTo(xLow, yPos - 4); chartCtx.lineTo(xLow, yPos + 4); chartCtx.stroke();
                        chartCtx.beginPath(); chartCtx.moveTo(xHigh, yPos - 4); chartCtx.lineTo(xHigh, yPos + 4); chartCtx.stroke();
                    });
                    chartCtx.restore();
                }
            },
            // Dashed vertical line at HR=1.0
            {
                id: 'forestNoEffect',
                beforeDatasetsDraw: function(chart) {
                    var chartCtx = chart.ctx;
                    var x = chart.scales.x, area = chart.chartArea;
                    var xPx = x.getPixelForValue(1.0);
                    if (xPx < area.left || xPx > area.right) return;
                    chartCtx.save();
                    chartCtx.strokeStyle = PALETTE.noEffect;
                    chartCtx.lineWidth = 1;
                    chartCtx.setLineDash([4, 3]);
                    chartCtx.beginPath(); chartCtx.moveTo(xPx, area.top); chartCtx.lineTo(xPx, area.bottom); chartCtx.stroke();
                    chartCtx.restore();
                }
            },
            // HR value labels right of CI
            {
                id: 'forestLabels',
                afterDatasetsDraw: function(chart) {
                    var chartCtx = chart.ctx;
                    var x = chart.scales.x, y = chart.scales.y;
                    chartCtx.save();
                    chartCtx.font = FONT.sizeSm + 'px ' + FONT.family;
                    chartCtx.fillStyle = 'rgba(51, 65, 85, 0.85)';
                    chartCtx.textAlign = 'left';
                    chartCtx.textBaseline = 'middle';
                    studies.forEach(function(s, i) {
                        var xPx = x.getPixelForValue(s.ciHigh) + 8;
                        var yPx = y.getPixelForValue(i);
                        chartCtx.fillText(s.hr + ' (' + s.ciLow + '–' + s.ciHigh + ')', xPx, yPx);
                    });
                    chartCtx.restore();
                }
            }
        ],
    };

    try {
        var chart = new Chart(ctx, config);
        this.instances.push(chart);
        return chart;
    } catch (err) {
        console.error('[ChartUtils] forest plot error:', err);
        return null;
    }
};

/**
 * Grouped survival bar chart — OS/PFS/DFS as grouped bars per study
 * @param {string} canvasId
 * @param {Object} data { title, studyLabels:[], endpoints:{ OS:[...], PFS:[...] } }
 */
ChartManager.prototype.renderGroupedSurvival = function(canvasId, data) {
    var endpoints = data.endpoints || {};
    var keys = Object.keys(endpoints).filter(function(k) {
        var vals = endpoints[k];
        return vals && vals.some(function(v) { return v !== null && v !== 0; });
    });
    if (keys.length === 0) return null;

    var datasets = keys.map(function(key, i) {
        var c = PALETTE.endpoints[key] || PALETTE.studies[i % PALETTE.studies.length];
        return {
            label: key,
            data: endpoints[key].map(function(v) { return v || 0; }),
            backgroundColor: c.bg,
            borderColor: c.border,
        };
    });

    var labels = (data.studyLabels || []).map(function(l) { return shortenLabel(l); });

    return this.render(canvasId, {
        type: 'bar',
        title: data.title || 'Survival Endpoints',
        labels: labels,
        datasets: datasets,
        unit: '%',
    }, { showDataLabels: true, horizontal: false });
};

// ── Export globally ───────────────────────────────────────────────
window.ChartManager = ChartManager;
window.ChartUtils = { PALETTE: PALETTE, FONT: FONT, shortenLabel: shortenLabel, authorShorthand: authorShorthand };
