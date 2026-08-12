/**
 * Study Details Service and UI Components
 * UPDATED: Integrated with PostgreSQL and dynamic renderer
 */

// ============================================
// Study Details API Service
// ============================================

class StudyDetailsService {
    constructor(baseUrl = CONFIG.API_BASE) {
        this.baseUrl = baseUrl;
        this.cache = new Map();
    }

    /**
     * Fetch study details by doc_id, PMID, DOI, or title
     * @param {Object} params - { doc_id, pmid, doi, title, trialData }
     * @returns {Promise<Object>} Study details response
     */
    async getStudyDetails({ doc_id, pmid, doi, title, trialData }) {
        // Create cache key - must be unique per study
        // Use a combination of available identifiers to ensure uniqueness
        const cacheKey = [doc_id, pmid, doi, title].filter(Boolean).join('|') || null;
        
        console.log('[StudyDetailsService] getStudyDetails called with:', { doc_id, pmid, doi, title, cacheKey });
        
        // Check cache first - but only if we have a meaningful cache key
        if (cacheKey && cacheKey.length > 5 && this.cache.has(cacheKey)) {
            console.log('[StudyDetailsService] Returning cached result for:', cacheKey);
            return this.cache.get(cacheKey);
        }

        try {
            // Build headers with auth token if available
            const headers = {
                'Content-Type': 'application/json',
            };
            
            // Add auth token if user is logged in
            const token = localStorage.getItem('exueed_token');
            if (token) {
                headers['Authorization'] = `Bearer ${token}`;
            }
            
            const response = await fetch(`${this.baseUrl}/study-details`, {
                method: 'POST',
                headers: headers,
                body: JSON.stringify({ doc_id, pmid, doi, title })
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            // Cache the result
            if (cacheKey) {
                this.cache.set(cacheKey, data);
            }
            
            return data;
        } catch (error) {
            console.error('Study details fetch error:', error);
            throw error;
        }
    }

    clearCache() {
        this.cache.clear();
    }
}

// Global instance
const studyDetailsService = new StudyDetailsService();

// ============================================
// Split View Manager
// ============================================

class SplitViewManager {
    constructor() {
        this.isOpen = false;
        this.currentStudy = null;
        this.panelElement = null;
        this.initialized = false;
    }

    /**
     * Initialize the split view layout
     */
    init() {
        if (this.initialized) {
            // Even if initialized, ensure panel element reference is valid
            if (!this.panelElement) {
                this.panelElement = document.getElementById('study-details-panel');
            }
            return;
        }
        
        // Create the study details panel container
        this.createPanelContainer();
        this.initialized = true;
    }

    /**
     * Create the panel container element
     */
    createPanelContainer() {
        // Check if panel already exists
        const existingPanel = document.getElementById('study-details-panel');
        if (existingPanel) {
            this.panelElement = existingPanel;
            console.log('[StudyDetails] Found existing panel element');
            return;
        }

        // Create panel element
        this.panelElement = document.createElement('div');
        this.panelElement.id = 'study-details-panel';
        this.panelElement.className = 'study-details-panel';
        this.panelElement.innerHTML = '<div class="panel-placeholder">Select a study to view details</div>';
        
        // Insert after the main content area
        const mainSection = document.querySelector('.section');
        if (mainSection) {
            mainSection.parentNode.insertBefore(this.panelElement, mainSection.nextSibling);
        } else {
            document.body.appendChild(this.panelElement);
        }
        console.log('[StudyDetails] Created new panel element');
    }

    /**
     * Open the split view with study details
     * @param {Object} studyRef - { doc_id, pmid, doi, title, trialData }
     */
    async openStudyDetails(studyRef) {
        console.log('[StudyDetails] openStudyDetails called with:', studyRef);
        console.log('[StudyDetails] Fullscreen active:', document.body.classList.contains('chat-fullscreen-active'));
        console.log('[StudyDetails] Already open:', this.isOpen);
        
        this.init();
        
        // Always re-fetch panel element to ensure we have the correct reference
        this.panelElement = document.getElementById('study-details-panel');
        
        if (!this.panelElement) {
            console.error('[StudyDetails] Panel element not found, creating new one');
            this.createPanelContainer();
            this.panelElement = document.getElementById('study-details-panel');
        }
        
        // Add split class to body - this enables the panel visibility via CSS
        document.body.classList.add('split-view-active');
        this.isOpen = true;
        this.currentStudy = studyRef;
        
        // Force panel visibility in fullscreen mode by ensuring display is set
        if (document.body.classList.contains('chat-fullscreen-active') && this.panelElement) {
            // Force a reflow to ensure CSS changes are applied
            this.panelElement.style.display = 'none';
            // Trigger reflow
            void this.panelElement.offsetHeight;
            this.panelElement.style.display = 'flex';
        }

        // Show loading state - this should always update the panel
        this.showLoading();
        console.log('[StudyDetails] Loading state shown, panel innerHTML length:', this.panelElement?.innerHTML?.length);

        try {
            console.log('[StudyDetails] Fetching from API...');
            const details = await studyDetailsService.getStudyDetails(studyRef);
            console.log('[StudyDetails] API returned:', details);
            this.renderStudyDetails(details);
            console.log('[StudyDetails] Rendered study details from API');
        } catch (error) {
            console.log('[StudyDetails] API error:', error.message);
            console.log('[StudyDetails] Has trialData?', !!studyRef.trialData);
            // If database lookup fails but we have trial data, show that instead
            if (studyRef.trialData) {
                console.log('[StudyDetails] Rendering fallback with trial data:', studyRef.trialData);
                this.renderTrialDataFallback(studyRef.trialData);
                console.log('[StudyDetails] Fallback rendered');
            } else {
                console.log('[StudyDetails] No trial data, showing error');
                this.showError(error.message);
            }
        }
    }

    /**
     * Close the split view
     */
    close() {
        document.body.classList.remove('split-view-active');
        this.isOpen = false;
        this.currentStudy = null;
        
        if (this.panelElement) {
            this.panelElement.innerHTML = '<div class="panel-placeholder">Select a study to view details</div>';
            // Reset display style to let CSS control visibility
            this.panelElement.style.display = '';
        }
    }

    /**
     * Show loading state in panel
     */
    showLoading() {
        // Re-fetch panel element to ensure we have the correct reference
        if (!this.panelElement) {
            this.panelElement = document.getElementById('study-details-panel');
        }
        
        if (!this.panelElement) {
            console.error('[StudyDetails] showLoading: Panel element not found');
            return;
        }
        
        this.panelElement.innerHTML = `
            <div class="panel-header">
                <h3>Loading Study Details...</h3>
                <button class="close-btn" onclick="splitViewManager.close()" title="Close panel">&times;</button>
            </div>
            <div class="panel-content">
                <div class="loading">
                    <div class="spinner"></div>
                    <span>Fetching study information...</span>
                </div>
            </div>
        `;
        
        // Scroll panel to top when loading new study
        this.panelElement.scrollTop = 0;
    }

    /**
     * Show error state in panel
     * @param {string} message - Error message
     */
    showError(message) {
        // Re-fetch panel element to ensure we have the correct reference
        if (!this.panelElement) {
            this.panelElement = document.getElementById('study-details-panel');
        }
        
        if (!this.panelElement) {
            console.error('[StudyDetails] showError: Panel element not found');
            return;
        }
        
        // Check if it's a "not found" error
        const isNotFound = message.toLowerCase().includes('not found') || message.toLowerCase().includes('study not found');
        
        const errorContent = isNotFound 
            ? `<div class="alert alert-warning">
                   <strong>Study profile not available</strong><br>
                   This study hasn't been processed for detailed display yet. 
                   The study information will be available once it's been added to the study profiles database.
               </div>`
            : `<div class="alert alert-error">
                   <strong>Failed to load study details:</strong> ${escapeHtml(message)}
               </div>`;
        
        this.panelElement.innerHTML = `
            <div class="panel-header">
                <h3>${isNotFound ? 'Study Profile Unavailable' : 'Error'}</h3>
                <button class="close-btn" onclick="splitViewManager.close()" title="Close panel">&times;</button>
            </div>
            <div class="panel-content">
                ${errorContent}
                <button class="btn btn-outline" onclick="splitViewManager.close()">Close Panel</button>
            </div>
        `;
    }

    /**
     * Render trial data as fallback when database lookup fails
     * @param {Object} trialData - Trial data from matching results
     */
    renderTrialDataFallback(trialData) {
        console.log('[StudyDetails] renderTrialDataFallback called with:', trialData);
        
        // Re-fetch panel element to ensure we have the correct reference
        if (!this.panelElement) {
            this.panelElement = document.getElementById('study-details-panel');
        }
        
        if (!this.panelElement) {
            console.error('[StudyDetails] renderTrialDataFallback: Panel element not found!');
            return;
        }

        const title = trialData.title || 'Study Details';
        const metaInfo = [];
        
        if (trialData.doi) metaInfo.push(`<span class="meta-item">DOI: ${escapeHtml(trialData.doi)}</span>`);
        if (trialData.author) metaInfo.push(`<span class="meta-item">${escapeHtml(trialData.author)}</span>`);
        if (trialData.year) metaInfo.push(`<span class="meta-item">${trialData.year}</span>`);

        // Build sections from available trial data
        let sectionsHtml = '';

        // Summary section
        if (trialData.relevant_excerpt) {
            sectionsHtml += `
                <div class="detail-section">
                    <div class="section-header">
                        <h4 class="section-title"><span class="expand-icon">▼</span> Summary</h4>
                    </div>
                    <div class="section-content expanded">
                        <div class="field-row">
                            <div class="field-value">${escapeHtml(trialData.relevant_excerpt)}</div>
                        </div>
                    </div>
                </div>
            `;
        }

        // Treatment section
        if (trialData.treatment) {
            sectionsHtml += `
                <div class="detail-section">
                    <div class="section-header">
                        <h4 class="section-title"><span class="expand-icon">▼</span> Treatment</h4>
                    </div>
                    <div class="section-content expanded">
                        <div class="field-row">
                            <div class="field-value">${escapeHtml(trialData.treatment)}</div>
                        </div>
                    </div>
                </div>
            `;
        }

        // Population section
        if (trialData.population_details) {
            sectionsHtml += `
                <div class="detail-section">
                    <div class="section-header">
                        <h4 class="section-title"><span class="expand-icon">▼</span> Population</h4>
                    </div>
                    <div class="section-content expanded">
                        <div class="field-row">
                            <div class="field-value">${escapeHtml(trialData.population_details)}</div>
                        </div>
                    </div>
                </div>
            `;
        }

        // Eligibility section
        if (trialData.inclusion_criteria || trialData.exclusion_criteria) {
            let eligibilityContent = '';
            if (trialData.inclusion_criteria) {
                eligibilityContent += `
                    <div class="field-row">
                        <div class="field-label">Inclusion Criteria</div>
                        <div class="field-value">${escapeHtml(trialData.inclusion_criteria)}</div>
                    </div>
                `;
            }
            if (trialData.exclusion_criteria) {
                eligibilityContent += `
                    <div class="field-row">
                        <div class="field-label">Exclusion Criteria</div>
                        <div class="field-value">${escapeHtml(trialData.exclusion_criteria)}</div>
                    </div>
                `;
            }
            sectionsHtml += `
                <div class="detail-section">
                    <div class="section-header">
                        <h4 class="section-title"><span class="expand-icon">▼</span> Eligibility</h4>
                    </div>
                    <div class="section-content expanded">
                        ${eligibilityContent}
                    </div>
                </div>
            `;
        }

        // Patient Fit section
        if (trialData.eligibility_notes && trialData.eligibility_notes.length > 0) {
            sectionsHtml += `
                <div class="detail-section">
                    <div class="section-header">
                        <h4 class="section-title"><span class="expand-icon">▼</span> Patient Fit Assessment</h4>
                    </div>
                    <div class="section-content expanded">
                        <div class="field-row">
                            <div class="field-value">${escapeHtml(trialData.eligibility_notes.join(' '))}</div>
                        </div>
                    </div>
                </div>
            `;
        }

        // Match info section
        if (trialData.match_score !== undefined || (trialData.match_reasons && trialData.match_reasons.length > 0)) {
            let matchContent = '';
            if (trialData.match_score !== undefined) {
                const scorePercent = Math.round(trialData.match_score * 100);
                const scoreColor = scorePercent >= 70 ? '#10b981' : scorePercent >= 40 ? '#f59e0b' : '#6b7280';
                matchContent += `
                    <div class="field-row">
                        <div class="field-label">Match Score</div>
                        <div class="field-value" style="color: ${scoreColor}; font-weight: 600;">${scorePercent}%</div>
                    </div>
                `;
            }
            if (trialData.match_reasons && trialData.match_reasons.length > 0) {
                matchContent += `
                    <div class="field-row">
                        <div class="field-label">Match Reasons</div>
                        <div class="field-value">${trialData.match_reasons.map(r => escapeHtml(r)).join(', ')}</div>
                    </div>
                `;
            }
            sectionsHtml += `
                <div class="detail-section">
                    <div class="section-header">
                        <h4 class="section-title"><span class="expand-icon">▼</span> Match Information</h4>
                    </div>
                    <div class="section-content expanded">
                        ${matchContent}
                    </div>
                </div>
            `;
        }

        // If no sections, show a message
        if (!sectionsHtml) {
            sectionsHtml = `
                <div class="alert alert-info">
                    Limited information available for this study. 
                    The full study profile will be available once it's been processed.
                </div>
            `;
        }

        this.panelElement.innerHTML = `
            <div class="panel-header">
                <div class="panel-title-area">
                    <h3 class="full-title">${escapeHtml(title)}</h3>
                    <div class="panel-meta-row">
                        <div class="panel-meta">
                            ${metaInfo.join('')}
                        </div>
                    </div>
                </div>
                <button class="close-btn" onclick="splitViewManager.close()" title="Close panel">&times;</button>
            </div>
            <div class="panel-content">
                <div class="alert alert-info" style="margin-bottom: 1rem;">
                    <strong>Note:</strong> Showing available trial match data. Full study profile not yet in database.
                </div>
                ${sectionsHtml}
            </div>
        `;
    }

    /**
     * Render study details in the panel
     * UPDATED: Uses enhanced renderer from studyDetailsRenderer.js
     * @param {Object} data - Study details response
     */
    renderStudyDetails(data) {
        // Re-fetch panel element to ensure we have the correct reference
        if (!this.panelElement) {
            this.panelElement = document.getElementById('study-details-panel');
        }
        
        if (!this.panelElement) {
            console.error('[StudyDetails] renderStudyDetails: Panel element not found');
            return;
        }

        // Check if enhanced renderer is available (from studyDetailsRenderer.js)
        if (typeof enhancedRenderStudyDetails === 'function') {
            enhancedRenderStudyDetails(this, data);
            return;
        }

        // Fallback to basic rendering if enhanced renderer not loaded
        const title = data.title || data.study_name || 'Study Details';
        const metaInfo = [];
        
        if (data.doc_id) metaInfo.push(`<span class="meta-item">ID: ${escapeHtml(data.doc_id.substring(0, 20))}...</span>`);
        if (data.pmid) metaInfo.push(`<span class="meta-item">PMID: ${escapeHtml(data.pmid)}</span>`);
        if (data.doi) metaInfo.push(`<span class="meta-item">DOI: ${escapeHtml(data.doi)}</span>`);

        this.panelElement.innerHTML = `
            <div class="panel-header">
                <div class="panel-title-area">
                    <h3>${escapeHtml(truncateText(title, 80))}</h3>
                    <div class="panel-meta">
                        ${metaInfo.join('')}
                    </div>
                </div>
                <button class="close-btn" onclick="splitViewManager.close()" title="Close panel">&times;</button>
            </div>
            <div class="panel-content">
                <div class="alert alert-warning">
                    Enhanced renderer (studyDetailsRenderer.js) not loaded. Using basic display.
                </div>
                <pre>${JSON.stringify(data, null, 2)}</pre>
            </div>
        `;
    }
}

// Global instance
const splitViewManager = new SplitViewManager();

// ============================================
// Helper Functions
// ============================================

/**
 * Escape HTML to prevent XSS
 * @param {string} text - Text to escape
 * @returns {string} Escaped text
 */
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Truncate text to specified length
 * @param {string} text - Text to truncate
 * @param {number} maxLength - Maximum length
 * @returns {string} Truncated text
 */
function truncateText(text, maxLength) {
    if (!text || text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

/**
 * Open study details from a source citation
 * Called when user clicks "Study Details" button
 * @param {Object} source - Source object with doc_id, pmid, doi, title
 * @param {Object} trialData - Optional trial data to display if database lookup fails
 */
function openStudyDetailsFromSource(source, trialData = null) {
    console.log('[StudyDetails] openStudyDetailsFromSource called with:', { source, trialData });
    
    const studyRef = {
        doc_id: source.doc_id,
        pmid: source.pmid,
        doi: source.doi,
        title: source.title,  // Pass title for fallback search
        trialData: trialData  // Pass trial data for fallback display
    };
    
    console.log('[StudyDetails] Created studyRef:', studyRef);
    splitViewManager.openStudyDetails(studyRef);
}