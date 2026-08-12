/**
 * Patient Signal — passive patient-profile capture.
 *
 * Drop-in module for any query mode (chat, patient matching, trial search,
 * etc.). After a query response comes back, call:
 *
 *   PatientSignal.checkAndPrompt(queryText);
 *
 * It no-ops entirely if the physician isn't logged in — passive capture
 * never forces a login on a previously login-free page. If logged in, it
 * asks the backend (/api/patients/signal) whether the query looked
 * patient-specific, then either:
 *   - does nothing (not patient-specific enough)
 *   - silently attaches to the active patient + shows a small dismissable
 *     toast (confident match)
 *   - shows a non-blocking confirm banner asking which patient this
 *     belongs to, or whether to create a new one (ambiguous / no active
 *     patient — per Aysha's answer, always ask in this case)
 *
 * Also renders a small "Active patient" indicator (top-right, fixed) so a
 * physician can select which patient is "active" while working across
 * pages — this is what lets silent_attach happen at all. Include this
 * script anywhere CONFIG (js/config.js) is already loaded.
 */

(function () {
    const ACTIVE_PATIENT_KEY = 'exueed_active_patient_id';
    const ACTIVE_PATIENT_NAME_KEY = 'exueed_active_patient_name';
    const PENDING_KEY = 'exueed_pending_signals';
    const PENDING_MAX = 8;

    function getToken() {
        return localStorage.getItem('exueed_token');
    }

    function authHeaders() {
        const token = getToken();
        return token ? { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' }
                     : { 'Content-Type': 'application/json' };
    }

    function apiBase() {
        return (typeof CONFIG !== 'undefined' && CONFIG.API_BASE) ? CONFIG.API_BASE : '/api';
    }

    async function apiFetch(path, options = {}) {
        const res = await fetch(`${apiBase()}${path}`, {
            ...options,
            headers: { ...authHeaders(), ...(options.headers || {}) },
        });
        if (!res.ok) {
            let detail = res.statusText;
            try { const body = await res.json(); detail = body.detail || detail; } catch (e) {}
            throw new Error(detail);
        }
        return res.json();
    }

    function getActivePatientId() {
        return localStorage.getItem(ACTIVE_PATIENT_KEY) || null;
    }

    function setActivePatient(id, name) {
        if (id) {
            localStorage.setItem(ACTIVE_PATIENT_KEY, id);
            localStorage.setItem(ACTIVE_PATIENT_NAME_KEY, name || '');
        } else {
            localStorage.removeItem(ACTIVE_PATIENT_KEY);
            localStorage.removeItem(ACTIVE_PATIENT_NAME_KEY);
        }
        renderIndicator();
    }

    // ------------------------------------------------------------------
    // Pending signals — for visitors who aren't logged in yet. Their raw
    // query text is queued here (not the extracted fields — extraction
    // costs an LLM call and must stay behind auth, see
    // PatientSignalService.is_patient_specific). Replayed through the real
    // /signal endpoint once they authenticate, in flushPendingSignals().
    // ------------------------------------------------------------------

    function getPendingSignals() {
        try {
            return JSON.parse(localStorage.getItem(PENDING_KEY) || '[]');
        } catch (e) {
            return [];
        }
    }

    function addPendingSignal(text) {
        const pending = getPendingSignals();
        if (pending.includes(text)) return;
        pending.push(text);
        while (pending.length > PENDING_MAX) pending.shift();
        localStorage.setItem(PENDING_KEY, JSON.stringify(pending));
    }

    function clearPendingSignals() {
        localStorage.removeItem(PENDING_KEY);
    }

    // ------------------------------------------------------------------
    // Toast / banner UI (self-contained styles, injected once)
    // ------------------------------------------------------------------

    function ensureStyles() {
        if (document.getElementById('patientSignalStyles')) return;
        const style = document.createElement('style');
        style.id = 'patientSignalStyles';
        style.textContent = `
            #psIndicator {
                position: fixed; top: 70px; right: 16px; z-index: 9998;
                background: #fff; border: 1px solid #e5e7eb; border-radius: 999px;
                padding: 0.35rem 0.75rem; font-size: 0.8rem; font-family: inherit;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08); cursor: pointer;
                display: flex; align-items: center; gap: 0.4rem; color: #374151;
            }
            #psIndicator .ps-dot { width: 8px; height: 8px; border-radius: 50%; background: #d1d5db; }
            #psIndicator.ps-active .ps-dot { background: #059669; }
            #psToast {
                position: fixed; bottom: 20px; right: 16px; z-index: 9999;
                background: #111827; color: #fff; padding: 0.6rem 1rem; border-radius: 8px;
                font-size: 0.85rem; max-width: 320px; box-shadow: 0 4px 14px rgba(0,0,0,0.2);
                opacity: 0; transform: translateY(8px); transition: all 0.2s ease;
            }
            #psToast.ps-show { opacity: 1; transform: translateY(0); }
            #psBanner {
                position: fixed; bottom: 20px; right: 16px; z-index: 9999;
                background: #fff; border: 1px solid #e5e7eb; border-radius: 12px;
                padding: 1rem; max-width: 340px; box-shadow: 0 8px 24px rgba(0,0,0,0.15);
                font-size: 0.85rem; color: #111827;
            }
            #psBanner h4 { margin: 0 0 0.5rem; font-size: 0.9rem; }
            #psBanner .ps-row { margin-bottom: 0.5rem; }
            #psBanner button { font-size: 0.8rem; padding: 0.3rem 0.6rem; margin-right: 0.35rem; margin-bottom: 0.35rem;
                border-radius: 6px; border: 1px solid #e5e7eb; background: #f9fafb; cursor: pointer; }
            #psBanner button.ps-primary { background: #e11d48; color: #fff; border-color: #e11d48; }
            #psBanner input { width: 100%; padding: 0.35rem 0.5rem; margin-bottom: 0.35rem;
                border: 1px solid #e5e7eb; border-radius: 6px; font-size: 0.8rem; box-sizing: border-box; }

            /* Patient picker panel */
            #psPatientPanel {
                position: fixed; top: 110px; right: 16px; z-index: 9999;
                background: #fff; border: 1px solid #e5e7eb; border-radius: 12px;
                width: 300px; max-width: calc(100vw - 32px);
                box-shadow: 0 12px 32px rgba(0,0,0,0.16);
                font-size: 0.85rem; color: #111827;
                display: flex; flex-direction: column;
                max-height: 70vh;
            }
            .pp-header {
                padding: 0.75rem 0.9rem; border-bottom: 1px solid #f1f5f9;
                display: flex; align-items: center; justify-content: space-between;
                flex-shrink: 0;
            }
            .pp-header h4 { margin: 0; font-size: 0.9rem; }
            .pp-close { background: none; border: none; font-size: 1.1rem; color: #9ca3af; cursor: pointer; line-height: 1; padding: 0; }
            .pp-close:hover { color: #374151; }
            .pp-list { overflow-y: auto; flex: 1; padding: 0.4rem; }
            .pp-row {
                display: flex; flex-direction: column; gap: 0.1rem;
                padding: 0.55rem 0.6rem; border-radius: 8px; cursor: pointer;
                border: 1px solid transparent;
            }
            .pp-row:hover { background: #f9fafb; }
            .pp-row.pp-active { background: #ecfdf5; border-color: #a7f3d0; }
            .pp-row-name { font-weight: 600; color: #111827; display: flex; align-items: center; gap: 0.35rem; }
            .pp-row-name .pp-check { color: #059669; font-weight: 700; }
            .pp-row-sub { font-size: 0.75rem; color: #6b7280; }
            .pp-empty { padding: 1.25rem 0.9rem; color: #6b7280; text-align: center; }
            .pp-footer {
                padding: 0.6rem 0.9rem; border-top: 1px solid #f1f5f9; flex-shrink: 0;
                display: flex; flex-direction: column; gap: 0.4rem;
            }
            .pp-clear-btn {
                font-size: 0.78rem; background: none; border: none; color: #6b7280;
                cursor: pointer; text-align: left; padding: 0.1rem 0;
            }
            .pp-clear-btn:hover { color: #374151; text-decoration: underline; }
            .pp-manage-link {
                font-size: 0.78rem; color: #2563eb; text-decoration: none;
            }
            .pp-manage-link:hover { text-decoration: underline; }
        `;
        document.head.appendChild(style);
    }

    function showToast(msg) {
        ensureStyles();
        let el = document.getElementById('psToast');
        if (!el) {
            el = document.createElement('div');
            el.id = 'psToast';
            document.body.appendChild(el);
        }
        el.textContent = msg;
        requestAnimationFrame(() => el.classList.add('ps-show'));
        clearTimeout(el._hideTimer);
        el._hideTimer = setTimeout(() => el.classList.remove('ps-show'), 4000);
    }

    function removeBanner() {
        const el = document.getElementById('psBanner');
        if (el) el.remove();
    }

    async function renderAskBanner(signal, opts = {}) {
        ensureStyles();
        removeBanner();

        const onResolved = opts.onResolved || (() => {});
        const el = document.createElement('div');
        el.id = 'psBanner';

        const candidatesHtml = (signal.candidates || []).map(c =>
            `<button data-patient-id="${c.patient_id}" class="ps-attach-btn">${c.name}</button>`
        ).join('');

        el.innerHTML = `
            <h4>${opts.heading || 'Add this to a patient profile?'}</h4>
            <div class="ps-row" style="color:#6b7280;">${opts.subheading || 'This looks like patient-specific information.'}</div>
            ${candidatesHtml ? `<div class="ps-row"><strong>Attach to:</strong><br>${candidatesHtml}</div>` : ''}
            <div class="ps-row"><strong>Or create a new patient:</strong></div>
            <input type="text" id="psNewFirst" placeholder="First name">
            <input type="text" id="psNewLast" placeholder="Last name">
            <button class="ps-primary" id="psCreateBtn">Create &amp; attach</button>
            <button id="psIgnoreBtn">Ignore</button>
        `;
        document.body.appendChild(el);

        el.querySelectorAll('.ps-attach-btn').forEach(btn => {
            btn.onclick = async () => {
                try {
                    await apiFetch('/patients/signal/apply', {
                        method: 'POST',
                        body: JSON.stringify({
                            action: 'attach',
                            patient_id: btn.dataset.patientId,
                            extracted: signal.extracted,
                            raw_text: signal.raw_text,
                        }),
                    });
                    showToast(`Added to ${btn.textContent}'s profile.`);
                    removeBanner();
                    onResolved();
                } catch (e) {
                    showToast(`Couldn't save: ${e.message}`);
                }
            };
        });

        el.querySelector('#psCreateBtn').onclick = async () => {
            const first = document.getElementById('psNewFirst').value.trim();
            const last = document.getElementById('psNewLast').value.trim();
            if (!first || !last) {
                showToast('Enter a first and last name first.');
                return;
            }
            try {
                const result = await apiFetch('/patients/signal/apply', {
                    method: 'POST',
                    body: JSON.stringify({
                        action: 'create_new',
                        new_patient_first_name: first,
                        new_patient_last_name: last,
                        extracted: signal.extracted,
                        raw_text: signal.raw_text,
                    }),
                });
                const name = result.patient_name || `${first} ${last}`;
                setActivePatient(result.patient_id, name);
                // A patient with this name already existed — reused it
                // instead of creating a duplicate record.
                showToast(result.reused_existing
                    ? `${name} already existed — attached this instead of creating a duplicate.`
                    : `Created ${name} and attached this.`);
                removeBanner();
                onResolved();
            } catch (e) {
                showToast(`Couldn't create patient: ${e.message}`);
            }
        };

        el.querySelector('#psIgnoreBtn').onclick = () => {
            removeBanner();
            onResolved();
        };
    }

    function renderSignupNudge(queryText) {
        ensureStyles();
        removeBanner();

        const el = document.createElement('div');
        el.id = 'psBanner';
        el.innerHTML = `
            <h4>Save this to a patient profile?</h4>
            <div class="ps-row" style="color:#6b7280;">
                This looks like patient-specific information. Create a free account and
                it'll be waiting for you to save — nothing you asked gets lost.
            </div>
            <a class="ps-primary" href="login.html" style="display:inline-block; text-decoration:none; padding:0.3rem 0.6rem; border-radius:6px; font-size:0.8rem; margin-right:0.35rem;">Create account</a>
            <button id="psNudgeDismiss">Not now</button>
        `;
        document.body.appendChild(el);
        el.querySelector('#psNudgeDismiss').onclick = () => removeBanner();
    }

    // ------------------------------------------------------------------
    // Active-patient indicator (top-right pill, switches active patient)
    // ------------------------------------------------------------------

    async function renderIndicator() {
        if (!getToken()) return;
        ensureStyles();

        let el = document.getElementById('psIndicator');
        if (!el) {
            el = document.createElement('div');
            el.id = 'psIndicator';
            document.body.appendChild(el);
            el.onclick = openPatientPicker;
        }
        const name = localStorage.getItem(ACTIVE_PATIENT_NAME_KEY);
        el.classList.toggle('ps-active', !!name);
        el.innerHTML = `<span class="ps-dot"></span>${name ? `Active: ${name}` : 'No active patient'}`;
    }

    function closePatientPanel() {
        const el = document.getElementById('psPatientPanel');
        if (el) el.remove();
        document.removeEventListener('click', _patientPanelOutsideClick, true);
    }

    function _patientPanelOutsideClick(e) {
        const panel = document.getElementById('psPatientPanel');
        const indicator = document.getElementById('psIndicator');
        if (!panel) return;
        if (panel.contains(e.target) || (indicator && indicator.contains(e.target))) return;
        closePatientPanel();
    }

    function _diagnosisSummary(p) {
        const d = p.latest_diagnosis;
        if (!d) return 'No diagnosis on file yet';
        const parts = [d.cancer_site, d.stage ? `Stage ${d.stage}` : null].filter(Boolean);
        return parts.length ? parts.join(' · ') : 'No diagnosis on file yet';
    }

    async function openPatientPicker() {
        ensureStyles();
        closePatientPanel();

        const el = document.createElement('div');
        el.id = 'psPatientPanel';
        el.innerHTML = `
            <div class="pp-header">
                <h4>Active patient</h4>
                <button class="pp-close" id="ppCloseBtn">&times;</button>
            </div>
            <div class="pp-list" id="ppList">
                <div class="pp-empty">Loading…</div>
            </div>
            <div class="pp-footer">
                <button class="pp-clear-btn" id="ppClearBtn">Clear active patient</button>
                <a class="pp-manage-link" href="my-saves.html#patients">View / manage all patients &rarr;</a>
            </div>
        `;
        document.body.appendChild(el);
        el.querySelector('#ppCloseBtn').onclick = closePatientPanel;
        el.querySelector('#ppClearBtn').onclick = () => {
            setActivePatient(null, null);
            closePatientPanel();
        };
        // Defer binding the outside-click listener so the click that opened
        // the panel doesn't immediately close it.
        setTimeout(() => document.addEventListener('click', _patientPanelOutsideClick, true), 0);

        const listEl = el.querySelector('#ppList');
        try {
            const data = await apiFetch('/patients?limit=50&include_diagnosis=true');
            const patients = data.patients || [];
            const activeId = getActivePatientId();

            if (!patients.length) {
                listEl.innerHTML = `<div class="pp-empty">No patients yet. Start a patient-specific chat, or add one from "My Collections".</div>`;
                return;
            }

            listEl.innerHTML = patients.map(p => {
                const isActive = p.id === activeId;
                const name = `${p.first_name} ${p.last_name}`;
                return `
                    <div class="pp-row ${isActive ? 'pp-active' : ''}" data-patient-id="${p.id}" data-patient-name="${name.replace(/"/g, '&quot;')}">
                        <div class="pp-row-name">${isActive ? '<span class="pp-check">&#10003;</span>' : ''}${_esc(name)}</div>
                        <div class="pp-row-sub">${_esc(_diagnosisSummary(p))}</div>
                    </div>
                `;
            }).join('');

            listEl.querySelectorAll('.pp-row').forEach(row => {
                row.onclick = () => {
                    setActivePatient(row.dataset.patientId, row.dataset.patientName);
                    closePatientPanel();
                };
            });
        } catch (e) {
            listEl.innerHTML = `<div class="pp-empty">Couldn't load patients: ${_esc(e.message)}</div>`;
        }
    }

    function _esc(str) {
        if (!str) return '';
        const d = document.createElement('div');
        d.textContent = str;
        return d.innerHTML;
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    async function checkAndPrompt(queryText) {
        if (!queryText || !queryText.trim()) return;

        // Not logged in: cheap no-auth pre-check only (no LLM extraction —
        // see /patients/signal/preview). Queue the raw text and nudge
        // toward account creation rather than silently dropping it.
        if (!getToken()) {
            try {
                const preview = await apiFetch('/patients/signal/preview', {
                    method: 'POST',
                    body: JSON.stringify({ query_text: queryText }),
                });
                if (preview.is_patient_specific) {
                    addPendingSignal(queryText);
                    renderSignupNudge(queryText);
                }
            } catch (e) {
                console.warn('[PatientSignal] preview check failed:', e.message);
            }
            return;
        }

        try {
            const signal = await apiFetch('/patients/signal', {
                method: 'POST',
                body: JSON.stringify({
                    query_text: queryText,
                    active_patient_id: getActivePatientId(),
                }),
            });

            if (signal.action === 'ignore') return;

            if (signal.action === 'silent_attach') {
                await apiFetch('/patients/signal/apply', {
                    method: 'POST',
                    body: JSON.stringify({
                        action: 'attach',
                        patient_id: signal.target_patient_id,
                        extracted: signal.extracted,
                        raw_text: signal.raw_text,
                    }),
                });
                showToast(`Added to ${signal.target_patient_name}'s profile.`);
                return;
            }

            if (signal.action === 'ask') {
                await renderAskBanner(signal);
            }
        } catch (e) {
            // Non-fatal — passive capture should never interrupt the main flow.
            console.warn('[PatientSignal] check failed:', e.message);
        }
    }

    // ------------------------------------------------------------------
    // Carry-forward: replay pre-account queued text once the visitor
    // actually authenticates, so "sign up to save this" is true in
    // practice and not just copy. Runs on every page load; if there's
    // nothing pending or the visitor still isn't logged in, it's a no-op.
    // ------------------------------------------------------------------

    async function flushPendingSignals() {
        if (!getToken()) return;
        const pending = getPendingSignals();
        if (!pending.length) return;

        const combinedText = pending.join('. ');
        try {
            const signal = await apiFetch('/patients/signal', {
                method: 'POST',
                body: JSON.stringify({
                    query_text: combinedText,
                    active_patient_id: getActivePatientId(),
                }),
            });

            if (signal.action === 'ignore') {
                clearPendingSignals();
                return;
            }

            if (signal.action === 'silent_attach') {
                await apiFetch('/patients/signal/apply', {
                    method: 'POST',
                    body: JSON.stringify({
                        action: 'attach',
                        patient_id: signal.target_patient_id,
                        extracted: signal.extracted,
                        raw_text: signal.raw_text,
                    }),
                });
                showToast(`Added your earlier questions to ${signal.target_patient_name}'s profile.`);
                clearPendingSignals();
                return;
            }

            if (signal.action === 'ask') {
                await renderAskBanner(signal, {
                    heading: 'Save what you asked before signing up?',
                    subheading: `You asked ${pending.length > 1 ? `${pending.length} patient-specific questions` : 'a patient-specific question'} before creating your account — add it to a patient profile now, or ignore it.`,
                    onResolved: clearPendingSignals,
                });
            }
        } catch (e) {
            console.warn('[PatientSignal] flush failed:', e.message);
        }
    }

    window.PatientSignal = {
        checkAndPrompt,
        setActivePatient,
        getActivePatientId,
        renderIndicator,
        flushPendingSignals,
    };

    document.addEventListener('DOMContentLoaded', () => {
        renderIndicator();
        flushPendingSignals();
    });
})();
