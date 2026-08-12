/**
 * LitAnalyticsPanel
 *
 * Reusable slide-in drawer that searches PubMed or ClinicalTrials.gov and
 * renders a chart. Designed to be dropped into any tool page.
 *
 * Public API:
 *   LitAnalyticsPanel.showButton(contextStr, anchorEl)
 *     — Inserts a "Show Literature Analytics" button after anchorEl.
 *       Replaces any existing button. Clicking calls open(contextStr).
 *
 *   LitAnalyticsPanel.open(contextStr)
 *     — Opens the drawer. Pre-fills NL textarea with contextStr.
 *       Auto-runs the search if contextStr is non-empty.
 *
 *   LitAnalyticsPanel.close()
 *     — Closes the drawer.
 *
 * Dependencies: Chart.js (CDN), chartUtils.js (ChartManager)
 */

const LitAnalyticsPanel = (function () {

    // ── State ────────────────────────────────────────────────────────────────
    var _ready = false;
    var _source = 'pubmed';       // 'pubmed' | 'clinicaltrials'
    var _mode   = 'nl';           // 'nl' | 'structured'
    var _chartInst = null;
    var _BASE = null;

    function _getBase() {
        if (_BASE) return _BASE;
        _BASE = (typeof API_BASE_URL !== 'undefined'
            ? API_BASE_URL
            : 'http://localhost:8000/api/rag').replace('/rag', '');
        return _BASE;
    }

    // ── CSS ─────────────────────────────────────────────────────────────────
    var _CSS = `
        #litAnalyticsDrawer {
            position: fixed;
            right: 0;
            top: 0;
            height: 100vh;
            width: 420px;
            max-width: 100vw;
            background: #fff;
            box-shadow: -4px 0 24px rgba(0,0,0,0.12);
            z-index: 20000;
            display: flex;
            flex-direction: column;
            transform: translateX(100%);
            transition: transform 0.22s ease;
            overflow: hidden;
        }
        #litAnalyticsDrawer.lap-open { transform: translateX(0); }
        @media (max-width: 767px) {
            #litAnalyticsDrawer { width: 100%; }
        }
        #litAnalyticsPanelOverlay {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.22);
            z-index: 19999;
        }
        #litAnalyticsPanelOverlay.lap-open { display: block; }

        .lap-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.85rem 1rem;
            border-bottom: 1px solid #e2e8f0;
            flex-shrink: 0;
        }
        .lap-header-title {
            font-weight: 700;
            font-size: 0.95rem;
            color: #0f172a;
        }
        .lap-close {
            background: none;
            border: none;
            font-size: 1.3rem;
            color: #94a3b8;
            cursor: pointer;
            padding: 0 0.25rem;
            line-height: 1;
        }
        .lap-close:hover { color: #334155; }

        .lap-body {
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.6rem;
        }

        /* Toggles */
        .lap-toggle-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 0.4rem;
        }
        .lap-pill-group {
            display: flex;
            border-radius: 6px;
            overflow: hidden;
            border: 1px solid #e2e8f0;
        }
        .lap-pill {
            padding: 0.32rem 0.75rem;
            background: #fff;
            border: none;
            font-size: 0.78rem;
            font-weight: 600;
            color: #64748b;
            cursor: pointer;
            transition: background 0.12s, color 0.12s;
            white-space: nowrap;
        }
        .lap-pill:hover { background: #f1f5f9; }
        .lap-pill.lap-active { background: #0f172a; color: #fff; }
        .lap-pill.lap-active-blue { background: #2563eb; color: #fff; }

        /* NL input */
        .lap-nl-row { display: flex; gap: 0.5rem; align-items: flex-end; }
        .lap-nl-row textarea {
            flex: 1;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 0.55rem 0.7rem;
            font-family: inherit;
            font-size: 0.85rem;
            resize: none;
            min-height: 52px;
            max-height: 110px;
            line-height: 1.5;
            color: #334155;
            background: #fff;
        }
        .lap-nl-row textarea:focus { outline: none; border-color: #2563eb; box-shadow: 0 0 0 2px rgba(37,99,235,0.1); }

        /* Structured input */
        .lap-struct-grid {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }
        .lap-struct-grid input {
            width: 100%;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 0.5rem 0.7rem;
            font-family: inherit;
            font-size: 0.85rem;
            color: #334155;
            background: #fff;
        }
        .lap-struct-grid input:focus { outline: none; border-color: #2563eb; }
        .lap-field-label {
            font-size: 0.73rem;
            font-weight: 600;
            color: #64748b;
            margin-bottom: 0.15rem;
            display: block;
        }

        /* Run button */
        .lap-run-btn {
            background: #2563eb;
            color: #fff;
            border: none;
            border-radius: 6px;
            padding: 0.55rem 1rem;
            font-weight: 700;
            font-size: 0.83rem;
            cursor: pointer;
            white-space: nowrap;
            transition: background 0.12s;
        }
        .lap-run-btn:hover { background: #1d4ed8; }
        .lap-run-btn:disabled { background: #94a3b8; cursor: not-allowed; }
        .lap-run-btn-full {
            width: 100%;
            margin-top: 0.25rem;
        }

        /* Results area */
        .lap-parsed-info {
            font-size: 0.73rem;
            color: #94a3b8;
            font-style: italic;
        }
        .lap-chart-title {
            font-size: 0.85rem;
            font-weight: 600;
            color: #94a3b8;
            line-height: 1.4;
        }
        .lap-chart-wrap {
            position: relative;
            height: 260px;
            display: none;
        }
        .lap-meta {
            font-size: 0.71rem;
            color: #94a3b8;
            margin-top: 0.2rem;
        }
        .lap-sources {
            font-size: 0.73rem;
            color: #64748b;
            line-height: 1.6;
            margin-top: 0.4rem;
        }
        .lap-sources a { color: #2563eb; text-decoration: none; }
        .lap-sources a:hover { text-decoration: underline; }
        .lap-caveats {
            font-size: 0.71rem;
            color: #94a3b8;
            font-style: italic;
            margin-top: 0.25rem;
        }

        /* Trigger button */
        .lap-trigger-btn {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.45rem 0.9rem;
            background: #fff;
            border: 1px solid #2563eb;
            border-radius: 6px;
            color: #2563eb;
            font-size: 0.82rem;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.12s, color 0.12s;
            margin-top: 0.75rem;
        }
        .lap-trigger-btn:hover { background: #eff6ff; }
        .lap-trigger-wrap {
            margin-top: 0.5rem;
        }
    `;

    // ── HTML ─────────────────────────────────────────────────────────────────
    function _drawerHTML() {
        return `
        <div id="litAnalyticsDrawer" role="dialog" aria-label="Literature Analytics">
            <div class="lap-header">
                <span class="lap-header-title">Literature Analytics</span>
                <button class="lap-close" id="lapCloseBtn" aria-label="Close">&times;</button>
            </div>
            <div class="lap-body">
                <!-- Source + mode toggles -->
                <div class="lap-toggle-row">
                    <div class="lap-pill-group">
                        <button class="lap-pill lap-active" id="lapSrcPubmed">PubMed</button>
                        <button class="lap-pill" id="lapSrcTrials">ClinicalTrials.gov</button>
                    </div>
                    <div class="lap-pill-group">
                        <button class="lap-pill lap-active-blue" id="lapModeNl">Natural language</button>
                        <button class="lap-pill" id="lapModeStruct">Structured</button>
                    </div>
                </div>

                <!-- NL input -->
                <div id="lapNlWrap" class="lap-nl-row">
                    <textarea id="lapNlInput" rows="2"
                        placeholder="e.g. OS rates for NSCLC pembrolizumab grouped by treatment arm"></textarea>
                    <button class="lap-run-btn" id="lapRunNl">Search</button>
                </div>

                <!-- Structured input -->
                <div id="lapStructWrap" class="lap-struct-grid" style="display:none;">
                    <div>
                        <span class="lap-field-label">Search query</span>
                        <input type="text" id="lapStructQuery" placeholder="e.g. NSCLC pembrolizumab OS">
                    </div>
                    <div>
                        <span class="lap-field-label">Metric to extract</span>
                        <input type="text" id="lapStructMetric" placeholder="e.g. overall survival rate">
                    </div>
                    <div>
                        <span class="lap-field-label">Group by (optional)</span>
                        <input type="text" id="lapStructGroupBy" placeholder="e.g. treatment arm, year">
                    </div>
                    <button class="lap-run-btn lap-run-btn-full" id="lapRunStruct">Search &amp; Chart</button>
                </div>

                <!-- Results -->
                <div class="lap-parsed-info" id="lapParsedInfo" style="display:none;"></div>
                <div class="lap-chart-title" id="lapChartTitle">Fill in the field above and click Search.</div>
                <div class="lap-chart-wrap" id="lapChartWrap"><canvas id="lapChart"></canvas></div>
                <div class="lap-meta" id="lapMeta"></div>
                <div class="lap-sources" id="lapSources"></div>
                <div class="lap-caveats" id="lapCaveats"></div>
            </div>
        </div>
        <div id="litAnalyticsPanelOverlay"></div>`;
    }

    // ── Init (lazy, called once) ──────────────────────────────────────────────
    function _init() {
        if (_ready) return;
        _ready = true;

        // Inject CSS
        if (!document.getElementById('lap-styles')) {
            var style = document.createElement('style');
            style.id = 'lap-styles';
            style.textContent = _CSS;
            document.head.appendChild(style);
        }

        // Inject HTML
        if (!document.getElementById('litAnalyticsDrawer')) {
            document.body.insertAdjacentHTML('beforeend', _drawerHTML());
        }

        // Source toggle
        document.getElementById('lapSrcPubmed').addEventListener('click', function () { _setSource('pubmed'); });
        document.getElementById('lapSrcTrials').addEventListener('click', function () { _setSource('clinicaltrials'); });

        // Mode toggle
        document.getElementById('lapModeNl').addEventListener('click', function () { _setMode('nl'); });
        document.getElementById('lapModeStruct').addEventListener('click', function () { _setMode('structured'); });

        // Run buttons
        document.getElementById('lapRunNl').addEventListener('click', function () {
            var val = (document.getElementById('lapNlInput').value || '').trim();
            if (!val) { _setTitle('Please describe what you want to chart.', '#dc2626'); return; }
            _runSearch(this, { nl_input: val });
        });

        document.getElementById('lapRunStruct').addEventListener('click', function () {
            var q = (document.getElementById('lapStructQuery').value || '').trim();
            var m = (document.getElementById('lapStructMetric').value || '').trim();
            var g = (document.getElementById('lapStructGroupBy').value || '').trim();
            if (!q || !m) { _setTitle('Enter a search query and a metric.', '#dc2626'); return; }
            _runSearch(this, { search_query: q, metric: m, group_by: g || null });
        });

        // Close
        document.getElementById('lapCloseBtn').addEventListener('click', close);
        document.getElementById('litAnalyticsPanelOverlay').addEventListener('click', close);
        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape' && document.getElementById('litAnalyticsDrawer').classList.contains('lap-open')) {
                close();
            }
        });
    }

    // ── Internals ─────────────────────────────────────────────────────────────
    function _setSource(src) {
        _source = src;
        document.getElementById('lapSrcPubmed').classList.toggle('lap-active', src === 'pubmed');
        document.getElementById('lapSrcTrials').classList.toggle('lap-active', src === 'clinicaltrials');
        if (_mode === 'nl') {
            document.getElementById('lapNlInput').placeholder = src === 'clinicaltrials'
                ? 'e.g. enrollment by phase for immunotherapy trials in lung cancer'
                : 'e.g. OS rates for NSCLC pembrolizumab grouped by treatment arm';
        }
    }

    function _setMode(mode) {
        _mode = mode;
        document.getElementById('lapModeNl').classList.toggle('lap-active-blue', mode === 'nl');
        document.getElementById('lapModeStruct').classList.toggle('lap-active-blue', mode === 'structured');
        document.getElementById('lapNlWrap').style.display = mode === 'nl' ? '' : 'none';
        document.getElementById('lapStructWrap').style.display = mode === 'structured' ? '' : 'none';
    }

    function _setTitle(text, color) {
        var el = document.getElementById('lapChartTitle');
        el.textContent = text;
        el.style.color = color || '#94a3b8';
    }

    function _esc(str) {
        if (!str) return '';
        var d = document.createElement('div');
        d.textContent = str;
        return d.innerHTML;
    }

    async function _runSearch(btn, payload) {
        var parsedEl  = document.getElementById('lapParsedInfo');
        var chartWrap = document.getElementById('lapChartWrap');
        var metaEl    = document.getElementById('lapMeta');
        var srcEl     = document.getElementById('lapSources');
        var cavEl     = document.getElementById('lapCaveats');

        btn.disabled = true;
        var origText = btn.textContent;
        btn.textContent = 'Searching…';
        var srcLabel = _source === 'clinicaltrials' ? 'ClinicalTrials.gov' : 'PubMed';
        _setTitle('Searching ' + srcLabel + '…', '#94a3b8');
        parsedEl.style.display = 'none';
        chartWrap.style.display = 'none';
        metaEl.textContent = '';
        srcEl.innerHTML = '';
        cavEl.textContent = '';

        try {
            var body = Object.assign({ source: _source }, payload);
            var resp = await fetch(_getBase() + '/analytics/online-aggregate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            if (!resp.ok) {
                var err = await resp.json().catch(function () { return {}; });
                throw new Error(err.detail || ('HTTP ' + resp.status));
            }
            var r = await resp.json();
            console.log('[LitAnalyticsPanel] result:', r);

            if (r.parsed_query || r.parsed_metric) {
                parsedEl.textContent = 'Searched: "' + (r.parsed_query || '') + '" · Metric: ' + (r.parsed_metric || '');
                parsedEl.style.display = '';
            }

            if (!r.labels || r.labels.length === 0) {
                _setTitle('No quantitative data found. Try rephrasing or a broader description.', '#92400e');
                return;
            }

            _setTitle(r.title || r.parsed_metric || 'Results', '#334155');
            chartWrap.style.display = '';

            // Render chart via ChartManager if available, else skip
            if (typeof ChartManager !== 'undefined') {
                if (_chartInst) { _chartInst.destroy(); _chartInst = null; }
                _chartInst = new ChartManager().render('lapChart', {
                    type: 'bar',
                    title: r.title || '',
                    labels: r.labels,
                    unit: r.unit || '',
                    datasets: [{ label: r.parsed_metric || 'Value', data: r.values }]
                }, { showDataLabels: true });
            }

            var nLabel = _source === 'clinicaltrials' ? 'trial' : 'article';
            metaEl.textContent = 'Based on ' + (r.n_studies || r.labels.length) + ' ' +
                srcLabel + ' ' + nLabel + ((r.n_studies || r.labels.length) !== 1 ? 's' : '') +
                (r.unit ? ' · Unit: ' + r.unit : '');

            if (r.sources && r.sources.length > 0) {
                var html = '<strong>Sources:</strong> ';
                html += r.sources.slice(0, 8).map(function (s) {
                    var label = _esc((s.title || '').substring(0, 65));
                    if (s.pmid) {
                        label += s.year ? ' (' + s.year + ')' : '';
                        return '<a href="https://pubmed.ncbi.nlm.nih.gov/' + _esc(s.pmid) + '/" target="_blank" rel="noopener">' + label + '</a>';
                    }
                    if (s.nct_id) {
                        label += s.phase ? ' [' + _esc(s.phase) + ']' : '';
                        return '<a href="https://clinicaltrials.gov/study/' + _esc(s.nct_id) + '" target="_blank" rel="noopener">' + label + '</a>';
                    }
                    return label;
                }).join('; ');
                srcEl.innerHTML = html;
            }

            if (r.caveats) { cavEl.textContent = r.caveats; }

        } catch (e) {
            console.error('[LitAnalyticsPanel] error:', e);
            _setTitle('Error: ' + e.message, '#dc2626');
        } finally {
            btn.disabled = false;
            btn.textContent = origText;
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    function showButton(contextStr, anchorEl) {
        // Remove any previous trigger
        var existing = document.getElementById('lap-trigger');
        if (existing) existing.remove();

        var wrap = document.createElement('div');
        wrap.className = 'lap-trigger-wrap';

        var btn = document.createElement('button');
        btn.id = 'lap-trigger';
        btn.className = 'lap-trigger-btn';
        btn.innerHTML =
            '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>' +
            'Show Literature Analytics';
        btn.addEventListener('click', function () {
            open(contextStr || '');
        });

        wrap.appendChild(btn);

        if (anchorEl && anchorEl.parentNode) {
            anchorEl.insertAdjacentElement('afterend', wrap);
        } else if (anchorEl) {
            anchorEl.appendChild(wrap);
        } else {
            document.body.appendChild(wrap);
        }
    }

    function open(contextStr) {
        _init();
        var drawer  = document.getElementById('litAnalyticsDrawer');
        var overlay = document.getElementById('litAnalyticsPanelOverlay');
        drawer.classList.add('lap-open');
        overlay.classList.add('lap-open');

        // Pre-fill NL input
        if (contextStr) {
            document.getElementById('lapNlInput').value = contextStr;
        }

        // Auto-run if context provided and mode is NL
        if (contextStr && _mode === 'nl') {
            var runBtn = document.getElementById('lapRunNl');
            if (runBtn) runBtn.click();
        }
    }

    function close() {
        var drawer  = document.getElementById('litAnalyticsDrawer');
        var overlay = document.getElementById('litAnalyticsPanelOverlay');
        if (drawer)  drawer.classList.remove('lap-open');
        if (overlay) overlay.classList.remove('lap-open');
    }

    return { showButton: showButton, open: open, close: close };

})();
