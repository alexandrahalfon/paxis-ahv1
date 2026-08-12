const api = new PaxisAPI();

const trialForm = document.getElementById('trialSearchForm');
const resultsContainer = document.getElementById('results');
const resultsList = document.getElementById('resultsList');
const extractedProfileCard = document.getElementById('extractedProfileCard');

// Check for follow-up context from main query page
try {
    const ctx = JSON.parse(sessionStorage.getItem('followupContext') || 'null');
    if (ctx && ctx.query && (Date.now() - ctx.timestamp < 300000)) {
        const trialQueryEl = document.getElementById('trialQuery');
        if (trialQueryEl) {
            trialQueryEl.value = ctx.query;
            console.log('[TrialSearch] Pre-filled from follow-up context');
        }
        sessionStorage.removeItem('followupContext');
    }
} catch (e) {
    console.error('[TrialSearch] Error reading follow-up context:', e);
}

function getMultiSelectValues(select) {
    if (!select) return [];
    return Array.from(select.selectedOptions || []).map(opt => opt.value);
}

function showLoading(container, message) {
    container.innerHTML = `
        <div class="loading">
            <div class="spinner"></div>
            <span>${message || 'Searching...'}</span>
        </div>
    `;
}

function showError(container, message) {
    container.innerHTML = `
        <div class="alert alert-error">
            ${escapeHtml(message || 'An error occurred.')}
        </div>
    `;
}

function escapeHtml(s) {
    if (!s) return '';
    const div = document.createElement('div');
    div.textContent = s;
    return div.innerHTML;
}

function renderExtractedProfile(profile) {
    if (!profile || Object.keys(profile).length === 0) {
        extractedProfileCard.style.display = 'none';
        return;
    }
    extractedProfileCard.style.display = 'block';
    const lines = [];
    if (profile.age != null) lines.push(`<div><strong>Age:</strong> ${profile.age}</div>`);
    if (profile.gender) lines.push(`<div><strong>Gender:</strong> ${profile.gender}</div>`);
    if (profile.cancer_type) lines.push(`<div><strong>Cancer type:</strong> ${profile.cancer_type}</div>`);
    if (profile.anatomical_site) lines.push(`<div><strong>Site:</strong> ${profile.anatomical_site}</div>`);
    if (profile.cancer_stage) lines.push(`<div><strong>Stage:</strong> ${profile.cancer_stage}</div>`);
    if (profile.histology) lines.push(`<div><strong>Histology:</strong> ${profile.histology}</div>`);
    if (profile.molecular_markers && profile.molecular_markers.length) {
        lines.push(`<div><strong>Markers:</strong> ${profile.molecular_markers.join(', ')}</div>`);
    }
    if (profile.performance_status != null) lines.push(`<div><strong>ECOG:</strong> ${profile.performance_status}</div>`);
    if (profile.smoking_status) lines.push(`<div><strong>Smoking:</strong> ${profile.smoking_status}</div>`);
    if (profile.prior_treatments && profile.prior_treatments.length) {
        const txList = Array.isArray(profile.prior_treatments) ? profile.prior_treatments : [profile.prior_treatments];
        lines.push(`<div><strong>Prior treatments:</strong> ${txList.join(', ')}</div>`);
    }
    extractedProfileCard.innerHTML = `
        <div class="card-body">
            <h3 style="margin: 0 0 0.5rem 0; font-size: 1rem; color: var(--gray-700);">Parsed profile</h3>
            <div style="display: grid; gap: 0.5rem;">${lines.join('')}</div>
        </div>
    `;
}

function renderResults(trials) {
    resultsList.innerHTML = '';
    if (!trials || trials.length === 0) {
        resultsList.innerHTML = `
            <div class="alert alert-info">
                No matching trials found. Try adding more detail or adjusting filters.
            </div>
        `;
        return;
    }

    const summary = document.createElement('div');
    summary.className = 'alert alert-success';
    summary.innerHTML = `<strong>Found ${trials.length} trials</strong>`;
    resultsList.appendChild(summary);

    trials.forEach((trial) => {
        const card = document.createElement('div');
        card.className = 'card';
        card.style.marginBottom = '1rem';

        const scorePct = Math.round((trial.match_score || 0) * 100);
        const phase = (trial.phase && trial.phase.length) ? trial.phase.join(', ') : 'N/A';
        const conditions = (trial.conditions || []).join(', ') || 'N/A';
        const interventions = (trial.interventions || []).join(', ') || 'N/A';
        const locations = (trial.locations || []).slice(0, 2).join(' • ');
        const summaryText = trial.summary ? escapeHtml(trial.summary) : 'Summary not available.';

        card.innerHTML = `
            <div class="card-body">
                <div style="display: flex; justify-content: space-between; align-items: start; gap: 1rem;">
                    <div style="flex: 1;">
                        <h3 style="margin: 0 0 0.5rem 0; font-size: 1.1rem; color: var(--gray-900);">
                            ${escapeHtml(trial.title || 'Untitled trial')}
                        </h3>
                        <div style="font-size: 0.875rem; color: var(--gray-600);">
                            NCT: ${escapeHtml(trial.nct_id || 'N/A')} • Status: ${escapeHtml(trial.status || 'Unknown')} • Phase: ${escapeHtml(phase)}
                        </div>
                    </div>
                    ${scorePct > 0 ? `
                        <div style="text-align: right; min-width: 80px;">
                            <div style="font-size: 0.75rem; color: var(--gray-500);">Match</div>
                            <div style="font-size: 1.1rem; font-weight: 600; color: var(--secondary);">${scorePct}%</div>
                        </div>
                    ` : ''}
                </div>
                <div style="margin-top: 0.75rem; color: var(--gray-700); font-size: 0.9rem;">
                    <div><strong>Conditions:</strong> ${escapeHtml(conditions)}</div>
                    <div><strong>Interventions:</strong> ${escapeHtml(interventions)}</div>
                    ${locations ? `<div><strong>Locations:</strong> ${escapeHtml(locations)}</div>` : ''}
                </div>
                <div style="margin-top: 0.75rem; background: var(--gray-50); padding: 0.75rem; border-radius: 0.375rem; font-size: 0.9rem; color: var(--gray-700);">
                    ${summaryText}
                </div>
                ${trial.url ? `
                    <div style="margin-top: 0.75rem; text-align: right;">
                        <a class="btn btn-outline btn-sm" href="${trial.url}" target="_blank" rel="noopener">View on ClinicalTrials.gov</a>
                    </div>
                ` : ''}
            </div>
        `;

        resultsList.appendChild(card);
    });
}

trialForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const query = document.getElementById('trialQuery').value.trim();
    const profile = {};

    const cancerType = document.getElementById('cancer_type').value.trim();
    if (cancerType) profile.cancer_type = cancerType;
    const anatomicalSite = document.getElementById('anatomical_site').value.trim();
    if (anatomicalSite) profile.anatomical_site = anatomicalSite;
    const stage = document.getElementById('cancer_stage').value.trim();
    if (stage) profile.cancer_stage = stage;
    const histology = document.getElementById('histology').value.trim();
    if (histology) profile.histology = histology;
    const markers = document.getElementById('molecular_markers').value.trim();
    if (markers) profile.molecular_markers = markers.split(',').map(m => m.trim()).filter(m => m);
    const gender = document.getElementById('gender').value;
    if (gender) profile.gender = gender;
    const age = document.getElementById('age').value;
    if (age) profile.age = parseInt(age, 10);
    const ecog = document.getElementById('performance_status').value;
    if (ecog) profile.performance_status = ecog;
    const smoking = document.getElementById('smoking_status').value;
    if (smoking) profile.smoking_status = smoking;
    const priorTx = document.getElementById('prior_treatments').value.trim();
    if (priorTx) profile.prior_treatments = priorTx.split(',').map(t => t.trim()).filter(t => t);

    const recruitingStatus = getMultiSelectValues(document.getElementById('recruiting_status'));
    const phase = getMultiSelectValues(document.getElementById('phase'));
    const location = document.getElementById('location').value.trim();

    if (!query && Object.keys(profile).length === 0) {
        alert('Please enter a case description or provide structured details.');
        return;
    }

    resultsContainer.style.display = 'block';
    extractedProfileCard.style.display = 'none';
    showLoading(resultsList, 'Searching ClinicalTrials.gov...');

    try {
        const payload = {
            query: query || null,
            patient_profile: Object.keys(profile).length ? profile : null,
            recruiting_status: recruitingStatus.length ? recruitingStatus : null,
            phase: phase.length ? phase : null,
            location: location || null,
            max_results: 25
        };
        const result = await api.searchClinicalTrials(payload);
        renderExtractedProfile(result.extracted_profile);
        renderResults(result.trials);

        // Passive patient-profile capture (patient-centric pivot).
        // No-ops if not logged in; never blocks or delays results.
        if (typeof PatientSignal !== 'undefined') {
            PatientSignal.checkAndPrompt(query || Object.values(profile).filter(Boolean).join(' '));
        }
    } catch (error) {
        showError(resultsList, error.message || 'Failed to search trials. Make sure the API server is running.');
    }
});
