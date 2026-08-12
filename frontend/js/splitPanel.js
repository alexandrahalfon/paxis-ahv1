/**
 * SplitPanel - Manages the right-side panel for module execution results.
 * 
 * Features:
 * - 60/40 split on desktop (>1024px)
 * - Full-width modal overlay on mobile (<1024px)
 * - Independent scrolling from chat
 * - Loading states and error handling
 * 
 * Requirements: 4.1, 4.2, 4.3, 4.5, 4.6, 4.7, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
 */
const SplitPanel = {
    isOpen: false,
    currentModule: null,
    _lastQuery: null,
    chartInstances: [],

    /**
     * Initialize the split panel structure in the DOM
     * Creates panel elements if they don't exist
     * Requirements: 4.1, 4.4
     */
    init() {
        if (document.getElementById('splitPanel')) return;

        const panelHtml = `
            <div id="splitPanel" class="split-panel" aria-hidden="true">
                <div class="split-panel-header">
                    <h3 id="splitPanelTitle" class="split-panel-title"></h3>
                    <button id="splitPanelClose" class="split-panel-close" 
                            aria-label="Close panel" type="button">
                        <span aria-hidden="true">&times;</span>
                    </button>
                </div>
                <div id="splitPanelContent" class="split-panel-content">
                    <!-- Module results render here -->
                </div>
                <div class="split-panel-footer">
                    <a id="splitPanelExploreMore" href="#" class="btn btn-outline btn-sm">
                        Explore More
                    </a>
                </div>
            </div>
            <div id="splitPanelOverlay" class="split-panel-overlay" aria-hidden="true"></div>
        `;

        document.body.insertAdjacentHTML('beforeend', panelHtml);
        this._bindEvents();
    },

    /**
     * Bind event listeners for close button, overlay, escape key, and resize
     * @private
     */
    _bindEvents() {
        const closeBtn = document.getElementById('splitPanelClose');
        const overlay = document.getElementById('splitPanelOverlay');

        if (closeBtn) {
            function handleCloseClick() {
                SplitPanel.close();
            }
            closeBtn.addEventListener('click', handleCloseClick);
        }

        if (overlay) {
            function handleOverlayClick() {
                SplitPanel.close();
            }
            overlay.addEventListener('click', handleOverlayClick);
        }

        function handleEscapeKey(e) {
            if (e.key === 'Escape' && SplitPanel.isOpen) {
                SplitPanel.close();
            }
        }
        document.addEventListener('keydown', handleEscapeKey);

        function handleWindowResize() {
            SplitPanel._handleResize();
        }
        window.addEventListener('resize', handleWindowResize);
    },

    /**
     * Open the split panel with a specific module
     * @param {string} module - Module name (treatment-comparison, patient-matching, study-comparison)
     * @param {string} query - The query to execute
     */
    async open(module, query) {
        this.init();

        const panel = document.getElementById('splitPanel');
        const overlay = document.getElementById('splitPanelOverlay');
        const chatContainer = document.querySelector('.chat-container');
        const title = document.getElementById('splitPanelTitle');
        const exploreLink = document.getElementById('splitPanelExploreMore');

        if (!panel) return;

        this.currentModule = module;
        this.isOpen = true;
        this._lastQuery = query;

        const moduleConfig = this._getModuleConfig(module);
        if (title) title.textContent = moduleConfig.title;
        
        // Set up Explore More link with context transfer
        if (exploreLink) {
            const self = this;
            exploreLink.href = moduleConfig.fullPageUrl;
            
            // Remove old listener and add new one
            const newLink = exploreLink.cloneNode(true);
            exploreLink.parentNode.replaceChild(newLink, exploreLink);
            
            newLink.addEventListener('click', function handleExploreMore(e) {
                // Store context for the target page
                const contextKey = module === 'patient-matching' ? 'patientMatchContext' :
                                   module === 'study-comparison' ? 'studyComparisonContext' :
                                   'treatmentComparisonContext';
                
                const contextData = {
                    query: query,
                    timestamp: Date.now()
                };
                
                // Add module-specific data
                if (module === 'patient-matching' && self._matchCriteriaWeights) {
                    contextData.criteria = self._matchCriteriaWeights;
                }
                if (module === 'study-comparison' && self._selectedStudies) {
                    contextData.studies = self._selectedStudies;
                }
                
                sessionStorage.setItem(contextKey, JSON.stringify(contextData));
                sessionStorage.setItem('followupContext', JSON.stringify({
                    query: query,
                    timestamp: Date.now()
                }));
                
                console.log('[SplitPanel] Stored context for', module, ':', contextData);
            });
        }

        this._showLoading();

        panel.classList.add('open');
        panel.setAttribute('aria-hidden', 'false');

        if (this._isMobile()) {
            if (overlay) {
                overlay.classList.add('visible');
                overlay.setAttribute('aria-hidden', 'false');
            }
            document.body.classList.add('panel-open-mobile');
        } else {
            if (chatContainer) {
                chatContainer.classList.add('split-active');
            }
        }

        // Enable split panel layout in fullscreen mode
        document.body.classList.add('split-panel-active');

        try {
            await this._executeModule(module, query);
        } catch (error) {
            console.error('[SplitPanel] Module execution error:', error);
            this._showError(error.message);
        }
    },

    close() {
        const panel = document.getElementById('splitPanel');
        const overlay = document.getElementById('splitPanelOverlay');
        const chatContainer = document.querySelector('.chat-container');

        if (panel) {
            panel.classList.remove('open');
            panel.setAttribute('aria-hidden', 'true');
        }

        if (overlay) {
            overlay.classList.remove('visible');
            overlay.setAttribute('aria-hidden', 'true');
        }

        if (chatContainer) {
            chatContainer.classList.remove('split-active');
        }

        document.body.classList.remove('panel-open-mobile');
        document.body.classList.remove('split-panel-active');
        this.isOpen = false;
        this.currentModule = null;
    },

    _getModuleConfig(module) {
        const configs = {
            'treatment-comparison': {
                title: 'Treatment Evaluation',
                fullPageUrl: 'treatment-comparison.html'
            },
            'patient-matching': {
                title: 'Patient Matching',
                fullPageUrl: 'patient-matching.html'
            },
            'study-comparison': {
                title: 'Review Studies',
                fullPageUrl: 'study-comparison.html'
            }
        };
        return configs[module] || { title: 'Results', fullPageUrl: '#' };
    },

    _isMobile() {
        return window.innerWidth < 1024;
    },

    _handleResize() {
        if (!this.isOpen) return;

        const overlay = document.getElementById('splitPanelOverlay');
        const chatContainer = document.querySelector('.chat-container');

        if (this._isMobile()) {
            if (overlay) {
                overlay.classList.add('visible');
                overlay.setAttribute('aria-hidden', 'false');
            }
            document.body.classList.add('panel-open-mobile');
            if (chatContainer) {
                chatContainer.classList.remove('split-active');
            }
        } else {
            if (overlay) {
                overlay.classList.remove('visible');
                overlay.setAttribute('aria-hidden', 'true');
            }
            document.body.classList.remove('panel-open-mobile');
            if (chatContainer) {
                chatContainer.classList.add('split-active');
            }
        }
    },

    _showLoading() {
        const content = document.getElementById('splitPanelContent');
        if (content) {
            content.innerHTML = `
                <div class="split-panel-loading">
                    <div class="spinner"></div>
                    <span>Loading results...</span>
                </div>
            `;
        }
    },

    _showError(message) {
        const content = document.getElementById('splitPanelContent');
        if (content) {
            content.innerHTML = `
                <div class="split-panel-error">
                    <p>Failed to load results: ${escapeHtml(message)}</p>
                    <button type="button" class="btn btn-outline btn-sm" 
                            onclick="SplitPanel._retry()">
                        Retry
                    </button>
                </div>
            `;
        }
    },

    _retry() {
        if (this.currentModule && this._lastQuery) {
            this.open(this.currentModule, this._lastQuery);
        }
    },

    async _executeModule(module, query) {
        const content = document.getElementById('splitPanelContent');

        switch (module) {
            case 'treatment-comparison':
                await this._executeTreatmentComparison(query);
                break;

            case 'patient-matching':
                await this._executePatientMatching(query);
                break;

            case 'study-comparison':
                await this._executeStudyComparison(query);
                break;

            default:
                throw new Error('Unknown module: ' + module);
        }
    },

    /**
     * Execute Treatment Comparison - uses TreatmentEvaluation workflow
     */
    async _executeTreatmentComparison(query) {
        const content = document.getElementById('splitPanelContent');
        const api = new PaxisAPI();
        
        try {
            // Get clinical context from InChatModules if available
            let effectiveQuery = query;
            if (typeof InChatModules !== 'undefined') {
                const clinicalContext = InChatModules.getClinicalContext();
                if (clinicalContext && clinicalContext.cancerType) {
                    const queryLower = query.toLowerCase();
                    const cancerLower = clinicalContext.cancerType.toLowerCase();
                    // Only add cancer type if not already in query
                    if (!queryLower.includes(cancerLower)) {
                        effectiveQuery = `${query} for ${clinicalContext.cancerType}`;
                        console.log('[SplitPanel] Added clinical context to query:', effectiveQuery);
                    }
                }
            }
            
            console.log('[SplitPanel] Executing treatment comparison with query:', effectiveQuery);
            const result = await api.visualComparison(effectiveQuery, 15);
            console.log('[SplitPanel] Treatment comparison result:', result);
            
            let html = '';
            
            // Summary section
            const summaryText = result.summary || result.short_answer || '';
            if (summaryText) {
                html += '<div class="panel-section">' +
                    '<h4>Summary</h4>' +
                    '<div class="panel-summary">' + markdownToHtml(summaryText) + '</div>' +
                '</div>';
            }
            
            // Detailed analysis
            const analysis = result.detailed_analysis || result.justification || '';
            if (analysis) {
                html += '<div class="panel-section">' +
                    '<h4>Detailed Analysis</h4>' +
                    '<div class="panel-analysis">' + markdownToHtml(analysis) + '</div>' +
                '</div>';
            }
            
            // Treatment arms breakdown
            if (result.treatment_arms && result.treatment_arms.length > 0) {
                const totalStudies = result.treatment_arms.reduce((sum, arm) => {
                    const sources = arm.retrieval_results || [];
                    const docIds = new Set();
                    sources.forEach(r => { if (r.doc_id) docIds.add(r.doc_id); });
                    return sum + docIds.size;
                }, 0);
                
                html += '<div class="panel-section">' +
                    '<h4>Supporting Evidence (' + totalStudies + ' studies)</h4>' +
                    '<div class="treatment-arms-list">';
                
                result.treatment_arms.forEach(arm => {
                    const sources = arm.retrieval_results || [];
                    const docIds = new Set();
                    sources.forEach(r => { if (r.doc_id) docIds.add(r.doc_id); });
                    const studyCount = docIds.size;
                    
                    html += '<div class="treatment-arm-item">' +
                        '<span class="arm-label">' + escapeHtml(arm.arm_label) + '</span>' +
                        '<span class="arm-count">' + studyCount + ' studies</span>' +
                    '</div>';
                });
                
                html += '</div></div>';
            }
            
            // Sources
            const sources = result.retrieval_results || [];
            if (sources.length > 0) {
                html += '<div class="panel-section">' +
                    '<h4>Sources (' + sources.length + ')</h4>' +
                    '<div class="panel-sources">' + this._renderSources(sources.slice(0, 10)) + '</div>' +
                '</div>';
            }
            
            if (!html) {
                html = '<div class="panel-section"><p class="panel-empty">No results available for this query.</p></div>';
            }
            
            if (content) {
                content.innerHTML = html;
                this._initResultInteractions(content);
            }
            
        } catch (error) {
            console.error('[SplitPanel] Treatment comparison error:', error);
            if (content) {
                content.innerHTML = '<div class="alert alert-danger">Error: ' + escapeHtml(error.message) + '</div>';
            }
        }
    },

    // Store match criteria weights
    _matchCriteriaWeights: {
        cancer_type: 1.5,
        stage: 1.25,
        biomarkers: 1.0,
        histology: 1.0,
        age: 0.75,
        performance_status: 0.75
    },

    /**
     * Execute Patient Matching - shows criteria setup first
     */
    async _executePatientMatching(query) {
        const content = document.getElementById('splitPanelContent');
        const self = this;
        
        // Show match criteria setup UI
        let html = '<div class="panel-setup">' +
            '<p class="setup-intro">Adjust how important each criterion is for matching:</p>' +
            '<div class="criteria-sliders">';
        
        const criteriaLabels = {
            cancer_type: 'Cancer Type',
            stage: 'Stage',
            biomarkers: 'Biomarkers',
            histology: 'Histology',
            age: 'Age',
            performance_status: 'Performance Status (ECOG)'
        };
        
        Object.keys(this._matchCriteriaWeights).forEach(key => {
            const label = criteriaLabels[key] || key;
            const value = this._matchCriteriaWeights[key];
            html += '<div class="criteria-slider-row">' +
                '<div class="criteria-slider-label">' + label + '</div>' +
                '<input type="range" class="criteria-slider-input" data-criteria="' + key + '" ' +
                    'min="0" max="2" step="0.25" value="' + value + '">' +
                '<span class="criteria-slider-value">' + value.toFixed(2) + 'x</span>' +
            '</div>';
        });
        
        html += '</div>' +
            '<button type="button" class="btn btn-accent" id="runPatientMatchBtn" style="width: 100%; margin-top: 1rem;">Find Matching Studies</button>' +
            '<p class="setup-hint">Or <a href="patient-matching.html" class="link-accent">go to full page</a> for more options</p>' +
        '</div>';
        
        if (content) {
            content.innerHTML = html;
            
            // Initialize slider interactions
            content.querySelectorAll('.criteria-slider-input').forEach(slider => {
                slider.addEventListener('input', function() {
                    const val = parseFloat(this.value);
                    const key = this.dataset.criteria;
                    self._matchCriteriaWeights[key] = val;
                    this.nextElementSibling.textContent = val.toFixed(2) + 'x';
                });
            });
            
            // Initialize run button
            const runBtn = document.getElementById('runPatientMatchBtn');
            if (runBtn) {
                runBtn.addEventListener('click', function() {
                    self._runPatientMatchingWithCriteria(query);
                });
            }
        }
    },

    /**
     * Run patient matching with the configured criteria weights
     */
    async _runPatientMatchingWithCriteria(query) {
        const content = document.getElementById('splitPanelContent');
        const api = new PaxisAPI();
        
        // Show loading
        if (content) {
            content.innerHTML = '<div class="split-panel-loading">' +
                '<div class="spinner"></div>' +
                '<span>Finding matching studies...</span>' +
            '</div>';
        }
        
        try {
            // Enrich query with clinical context from InChatModules
            let effectiveQuery = query;
            if (typeof InChatModules !== 'undefined') {
                const clinicalContext = InChatModules.getClinicalContext();
                if (clinicalContext) {
                    const queryLower = query.toLowerCase();
                    let contextParts = [];
                    
                    // Add cancer type if not in query
                    if (clinicalContext.cancerType && !queryLower.includes(clinicalContext.cancerType.toLowerCase())) {
                        contextParts.push(clinicalContext.cancerType);
                    }
                    // Add stage if not in query
                    if (clinicalContext.cancerStage && !queryLower.includes('stage')) {
                        contextParts.push('stage ' + clinicalContext.cancerStage);
                    }
                    // Add histology if not in query
                    if (clinicalContext.histology && !queryLower.includes(clinicalContext.histology.toLowerCase())) {
                        contextParts.push(clinicalContext.histology);
                    }
                    
                    if (contextParts.length > 0) {
                        effectiveQuery = query + ' ' + contextParts.join(' ');
                        console.log('[SplitPanel] Enriched patient matching query with clinical context:', effectiveQuery);
                    }
                }
            }
            
            console.log('[SplitPanel] Executing patient matching with query:', effectiveQuery);
            console.log('[SplitPanel] Match criteria weights:', this._matchCriteriaWeights);
            
            let result;
            let usedFallback = false;
            
            // Check for cached profile
            let cachedProfile = null;
            if (api.contextManager) {
                const context = api.contextManager.getContext();
                for (let i = context.length - 1; i >= 0; i--) {
                    if (context[i].action_type === 'patient_match' && context[i].extracted_profile) {
                        cachedProfile = context[i].extracted_profile;
                        console.log('[SplitPanel] Using cached profile from conversation context');
                        break;
                    }
                }
            }
            
            try {
                // Pass criteria weights to the API if supported
                result = await api.matchPatientUnstructured(effectiveQuery, 15, this._matchCriteriaWeights, cachedProfile);
                console.log('[SplitPanel] Patient matching result:', result);
            } catch (matchError) {
                const errorMsg = matchError.message || '';
                if (errorMsg.includes('Could not extract patient characteristics') || 
                    errorMsg.includes('patient characteristics') ||
                    errorMsg.includes('cancer type, stage')) {
                    console.log('[SplitPanel] Patient extraction failed, falling back to study search');
                    usedFallback = true;
                    
                    const searchResult = await api.searchStudiesWithQuery(effectiveQuery, 15);
                    console.log('[SplitPanel] Fallback study search result:', searchResult);
                    
                    const studies = searchResult.studies || [];
                    result = {
                        patient_summary: null,
                        extracted_profile: null,
                        matches: studies.map(s => ({
                            doc_id: s.doc_id,
                            title: s.title || s.study_name || 'Unknown Study',
                            author: s.author || '',
                            year: s.year || '',
                            doi: s.doi || '',
                            pmid: s.pmid || '',
                            match_score: s.match_score || s.score || 0.5,
                            match_reasons: s.cancer_type ? [s.cancer_type] : []
                        }))
                    };
                } else {
                    throw matchError;
                }
            }
            
            let html = '';
            
            // Back to criteria button
            html += '<div class="panel-back-row">' +
                '<button type="button" class="btn btn-outline btn-sm" id="backToCriteriaBtn">Adjust Criteria</button>' +
            '</div>';
            
            if (usedFallback) {
                html += '<div class="alert alert-info" style="margin-bottom: 1rem;">' +
                    'Showing relevant studies based on your query. For more precise patient-to-trial matching, ' +
                    'include specific clinical details like cancer type, stage, or biomarkers.' +
                '</div>';
            }
            
            if (result.patient_summary) {
                html += '<div class="panel-section">' +
                    '<h4>Patient Summary</h4>' +
                    '<div class="panel-summary">' + escapeHtml(result.patient_summary) + '</div>' +
                '</div>';
            }
            
            if (result.extracted_profile) {
                html += '<div class="panel-section">' +
                    '<details class="extracted-profile-details">' +
                    '<summary>View Extracted Profile</summary>' +
                    '<div class="extracted-profile-content">' + 
                        this._renderExtractedProfile(result.extracted_profile) + 
                    '</div>' +
                    '</details>' +
                '</div>';
            }
            
            const matches = result.matches || [];
            if (matches.length > 0) {
                html += '<div class="panel-section">' +
                    '<h4>' + (usedFallback ? 'Related Studies' : 'Matching Studies') + ' (' + matches.length + ' found)</h4>' +
                    '<div class="panel-matches">' + this._renderPatientMatches(matches) + '</div>' +
                '</div>';
            } else {
                html += '<div class="alert alert-info">No matching studies found. Try adding more details about the patient.</div>';
            }
            
            if (content) {
                content.innerHTML = html;
                this._initResultInteractions(content);
                this._initAddToReviewButtons(content);
                
                // Back button handler
                const backBtn = document.getElementById('backToCriteriaBtn');
                if (backBtn) {
                    const self = this;
                    backBtn.addEventListener('click', function() {
                        self._executePatientMatching(query);
                    });
                }
            }
            
        } catch (error) {
            console.error('[SplitPanel] Patient matching error:', error);
            if (content) {
                content.innerHTML = '<div class="alert alert-danger">Error: ' + escapeHtml(error.message) + '</div>';
            }
        }
    },

    // Store selected studies for comparison
    _selectedStudies: [],
    _searchedStudies: [],

    /**
     * Execute Study Comparison - shows study selection first
     */
    async _executeStudyComparison(query) {
        const content = document.getElementById('splitPanelContent');
        const api = new PaxisAPI();
        const self = this;
        
        // Load any existing studies from comparison tray
        try {
            const tray = typeof getComparisonTray === 'function' ? getComparisonTray() : [];
            this._selectedStudies = tray.slice(0, 4);
        } catch (e) {
            this._selectedStudies = [];
        }
        
        // Show loading while searching
        if (content) {
            content.innerHTML = '<div class="split-panel-loading">' +
                '<div class="spinner"></div>' +
                '<span>Finding relevant studies...</span>' +
            '</div>';
        }
        
        try {
            console.log('[SplitPanel] Searching studies for comparison with query:', query);
            
            const searchResult = await api.searchStudiesWithQuery(query, 10);
            console.log('[SplitPanel] Study search result:', searchResult);
            
            this._searchedStudies = searchResult.studies || [];
            
            // Show study selection UI
            this._showStudySelectionUI(query, content);
            
        } catch (error) {
            console.error('[SplitPanel] Study search error:', error);
            if (content) {
                content.innerHTML = '<div class="alert alert-danger">Error: ' + escapeHtml(error.message) + '</div>';
            }
        }
    },

    /**
     * Show the study selection UI
     */
    _showStudySelectionUI(query, content) {
        const self = this;
        
        let html = '<div class="panel-setup">' +
            '<p class="setup-intro">Select 2-4 studies to compare:</p>';
        
        // Selected studies tray
        html += '<div class="study-selection-tray" id="studySelectionTray">' +
            '<div class="tray-header">' +
                '<span class="tray-label">Selected Studies</span>' +
                '<span class="tray-count" id="trayCountDisplay">' + this._selectedStudies.length + '/4</span>' +
            '</div>' +
            '<div class="tray-items" id="trayItems">';
        
        if (this._selectedStudies.length === 0) {
            html += '<div class="tray-empty">Click studies below to add them</div>';
        } else {
            this._selectedStudies.forEach((study, idx) => {
                const shortTitle = (study.title || 'Study').substring(0, 40) + ((study.title || '').length > 40 ? '...' : '');
                html += '<div class="tray-item" data-idx="' + idx + '">' +
                    '<span class="tray-item-title">' + escapeHtml(shortTitle) + '</span>' +
                    '<button type="button" class="tray-item-remove" data-idx="' + idx + '">&times;</button>' +
                '</div>';
            });
        }
        
        html += '</div></div>';
        
        // Compare button
        html += '<button type="button" class="btn btn-accent" id="runComparisonBtn" style="width: 100%; margin-top: 1rem;" ' +
            (this._selectedStudies.length < 2 ? 'disabled' : '') + '>Compare Selected Studies</button>';
        
        // Search results
        html += '<div class="panel-section" style="margin-top: 1.5rem;">' +
            '<h4>Available Studies (' + this._searchedStudies.length + ' found)</h4>' +
            '<div class="study-search-list">';
        
        if (this._searchedStudies.length === 0) {
            html += '<div class="no-data-message">No studies found. Try a different query.</div>';
        } else {
            this._searchedStudies.forEach((study, idx) => {
                const isSelected = this._selectedStudies.some(s => s.doc_id === study.doc_id);
                const title = study.title || study.study_name || 'Unknown Study';
                const author = study.author || '';
                const year = study.year || '';
                const cancerType = study.cancer_type || '';
                
                let metaParts = [];
                if (author) metaParts.push(escapeHtml(author));
                if (year) metaParts.push(year);
                if (cancerType) metaParts.push(escapeHtml(cancerType));
                
                html += '<div class="study-search-item ' + (isSelected ? 'selected' : '') + '" data-doc-id="' + escapeHtml(study.doc_id || '') + '">' +
                    '<div class="study-search-info">' +
                        '<div class="study-search-title">' + escapeHtml(title) + '</div>' +
                        '<div class="study-search-meta">' + metaParts.join(' | ') + '</div>' +
                    '</div>' +
                    '<button type="button" class="btn btn-sm study-toggle-btn ' + (isSelected ? 'btn-primary' : 'btn-outline') + '" ' +
                        'data-doc-id="' + escapeHtml(study.doc_id || '') + '" ' +
                        'data-title="' + escapeHtml(title) + '" ' +
                        'data-doi="' + escapeHtml(study.doi || '') + '" ' +
                        'data-year="' + escapeHtml(String(year || '')) + '">' +
                        (isSelected ? 'Remove' : 'Add') +
                    '</button>' +
                '</div>';
            });
        }
        
        html += '</div></div>';
        
        html += '<p class="setup-hint">Or <a href="study-comparison.html" class="link-accent">go to full page</a> for more options</p>' +
        '</div>';
        
        if (content) {
            content.innerHTML = html;
            
            // Initialize toggle buttons
            content.querySelectorAll('.study-toggle-btn').forEach(btn => {
                btn.addEventListener('click', function() {
                    const docId = this.dataset.docId;
                    const title = this.dataset.title;
                    const doi = this.dataset.doi;
                    const year = this.dataset.year;
                    
                    const idx = self._selectedStudies.findIndex(s => s.doc_id === docId);
                    if (idx >= 0) {
                        // Remove
                        self._selectedStudies.splice(idx, 1);
                    } else {
                        // Add (max 4)
                        if (self._selectedStudies.length >= 4) {
                            // Show modal to manage queue instead of alert
                            if (typeof InChatModules !== 'undefined' && typeof InChatModules.showReviewQueueModal === 'function') {
                                InChatModules.showReviewQueueModal(docId, title, doi, year);
                            } else {
                                alert('Maximum 4 studies can be compared. Remove one first.');
                            }
                            return;
                        }
                        self._selectedStudies.push({ doc_id: docId, title: title, doi: doi, year: year });
                    }
                    
                    // Save to tray
                    if (typeof saveComparisonTray === 'function') {
                        saveComparisonTray(self._selectedStudies);
                    }
                    
                    // Refresh UI
                    self._showStudySelectionUI(query, content);
                });
            });
            
            // Initialize remove buttons in tray
            content.querySelectorAll('.tray-item-remove').forEach(btn => {
                btn.addEventListener('click', function(e) {
                    e.stopPropagation();
                    const idx = parseInt(this.dataset.idx);
                    self._selectedStudies.splice(idx, 1);
                    
                    if (typeof saveComparisonTray === 'function') {
                        saveComparisonTray(self._selectedStudies);
                    }
                    
                    self._showStudySelectionUI(query, content);
                });
            });
            
            // Initialize compare button
            const compareBtn = document.getElementById('runComparisonBtn');
            if (compareBtn) {
                compareBtn.addEventListener('click', function() {
                    self._runStudyComparison(query);
                });
            }
        }
    },

    /**
     * Run the actual study comparison
     */
    async _runStudyComparison(query) {
        const content = document.getElementById('splitPanelContent');
        const api = new PaxisAPI();
        const self = this;
        
        if (this._selectedStudies.length < 2) {
            alert('Please select at least 2 studies to compare.');
            return;
        }
        
        // Show loading
        if (content) {
            content.innerHTML = '<div class="split-panel-loading">' +
                '<div class="spinner"></div>' +
                '<span>Generating comparison...</span>' +
            '</div>';
        }
        
        try {
            const studyIds = this._selectedStudies.map(s => s.doc_id).filter(id => id);
            console.log('[SplitPanel] Comparing studies:', studyIds);
            
            const comparisonResult = await api.compareStudies(studyIds);
            console.log('[SplitPanel] Comparison result:', comparisonResult);
            
            let html = '';
            
            // Back button
            html += '<div class="panel-back-row">' +
                '<button type="button" class="btn btn-outline btn-sm" id="backToSelectionBtn">Edit Selection</button>' +
            '</div>';
            
            // Studies being compared
            html += '<div class="panel-section">' +
                '<h4>Comparing ' + this._selectedStudies.length + ' Studies</h4>' +
                '<div class="compared-studies-list">';
            
            this._selectedStudies.forEach((study, idx) => {
                html += '<div class="compared-study-item">' +
                    '<span class="study-num">' + (idx + 1) + '.</span>' +
                    '<span class="study-title">' + escapeHtml(study.title || 'Unknown') + '</span>' +
                '</div>';
            });
            
            html += '</div></div>';
            
            // Narrative summary
            if (comparisonResult.narrative) {
                html += '<div class="panel-section">' +
                    '<h4>Comparison Summary</h4>' +
                    '<div class="comparison-narrative">' + markdownToHtml(comparisonResult.narrative) + '</div>' +
                '</div>';
            }
            
            // Categories with charts
            if (comparisonResult.categories && comparisonResult.categories.length > 0) {
                html += '<div class="panel-section">' +
                    '<h4>Side-by-Side Comparison</h4>' +
                    this._renderComparisonCategories(comparisonResult.categories, comparisonResult.studies) +
                '</div>';
            }
            
            if (content) {
                content.innerHTML = html;
                this._initResultInteractions(content);
                
                // Back button handler
                const backBtn = document.getElementById('backToSelectionBtn');
                if (backBtn) {
                    backBtn.addEventListener('click', function() {
                        self._showStudySelectionUI(query, content);
                    });
                }
                
                // Render charts
                if (comparisonResult.categories && comparisonResult.categories.length > 0) {
                    setTimeout(() => {
                        this._renderComparisonCharts(comparisonResult.categories);
                    }, 100);
                }
            }
            
        } catch (error) {
            console.error('[SplitPanel] Comparison error:', error);
            if (content) {
                content.innerHTML = '<div class="alert alert-danger">Error: ' + escapeHtml(error.message) + '</div>' +
                    '<button type="button" class="btn btn-outline btn-sm" onclick="SplitPanel._showStudySelectionUI(\'' + escapeHtml(query) + '\', document.getElementById(\'splitPanelContent\'))">Back to Selection</button>';
            }
        }
    },

    _renderExtractedProfile(profile) {
        if (!profile) return '';
        
        const fields = [];
        if (profile.age != null) fields.push('<div><strong>Age:</strong> ' + profile.age + '</div>');
        if (profile.gender) fields.push('<div><strong>Gender:</strong> ' + escapeHtml(profile.gender) + '</div>');
        if (profile.cancer_type) fields.push('<div><strong>Cancer Type:</strong> ' + escapeHtml(profile.cancer_type) + '</div>');
        if (profile.cancer_stage) fields.push('<div><strong>Stage:</strong> ' + escapeHtml(profile.cancer_stage) + '</div>');
        if (profile.histology) fields.push('<div><strong>Histology:</strong> ' + escapeHtml(profile.histology) + '</div>');
        if (profile.molecular_markers && profile.molecular_markers.length) {
            fields.push('<div><strong>Biomarkers:</strong> ' + escapeHtml(profile.molecular_markers.join(', ')) + '</div>');
        }
        if (profile.performance_status != null) {
            fields.push('<div><strong>ECOG:</strong> ' + profile.performance_status + '</div>');
        }
        
        return '<div class="profile-grid">' + fields.join('') + '</div>';
    },

    _renderPatientMatches(matches) {
        if (!matches || matches.length === 0) return '';
        
        const self = this;
        return matches.slice(0, 10).map((match, index) => {
            const scorePct = Math.round((match.match_score || match.score || 0) * 100);
            const title = match.title || 'Unknown Study';
            const author = match.author || '';
            const year = match.year || '';
            const doi = match.doi || '';
            const pmid = match.pmid || '';
            const docId = match.doc_id || '';
            
            let matchReasons = '';
            if (match.match_reasons && match.match_reasons.length > 0) {
                matchReasons = '<div class="match-reasons">' + 
                    match.match_reasons.slice(0, 3).map(r => '<span class="match-tag">' + escapeHtml(r) + '</span>').join('') +
                '</div>';
            }
            
            let studyBtn = '';
            if (docId || pmid || doi) {
                const escapedDocId = (docId || '').replace(/'/g, "\\'").replace(/"/g, '&quot;');
                const escapedPmid = (pmid || '').replace(/'/g, "\\'").replace(/"/g, '&quot;');
                const escapedDoi = (doi || '').replace(/'/g, "\\'").replace(/"/g, '&quot;');
                const escapedTitle = (title || '').replace(/'/g, "\\'").replace(/"/g, '&quot;');
                studyBtn = '<button class="study-details-btn" ' +
                    'onclick="openStudyDetailsFromSource({doc_id:\'' + escapedDocId + '\',pmid:\'' + escapedPmid + '\',doi:\'' + escapedDoi + '\',title:\'' + escapedTitle + '\'})" ' +
                    'title="View detailed study information">' +
                    'Study Details</button>';
            }
            
            let reviewBtn = '';
            if (docId && typeof isStudyInComparisonTray === 'function') {
                const inTray = isStudyInComparisonTray(docId);
                const reviewLabel = inTray ? 'In Review Queue' : 'Add to Review';
                const reviewClass = inTray ? 'btn-primary' : 'btn-outline';
                reviewBtn = '<button class="btn ' + reviewClass + ' btn-sm add-to-review-btn" ' +
                    'data-doc-id="' + escapeHtml(docId) + '" ' +
                    'data-title="' + escapeHtml(title) + '" ' +
                    'data-doi="' + escapeHtml(doi || '') + '" ' +
                    'data-year="' + escapeHtml(String(year || '')) + '" ' +
                    'style="font-size: 0.75rem;">' + reviewLabel + '</button>';
            }
            
            let doiLink = '';
            if (doi) {
                doiLink = '<a href="https://doi.org/' + doi + '" target="_blank" rel="noopener" ' +
                    'class="btn btn-outline btn-sm" style="font-size: 0.75rem;">DOI</a>';
            }
            
            return '<div class="panel-match-item">' +
                '<div class="match-header">' +
                    '<div class="match-title">' + (index + 1) + '. ' + escapeHtml(title) + '</div>' +
                    '<div class="match-score">' + scorePct + '%</div>' +
                '</div>' +
                '<div class="match-meta">' +
                    (author ? escapeHtml(author) : '') +
                    (year ? ' (' + year + ')' : '') +
                '</div>' +
                matchReasons +
                '<div class="match-actions">' + studyBtn + reviewBtn + doiLink + '</div>' +
            '</div>';
        }).join('');
    },

    _renderStudyComparisonList(studies) {
        if (!studies || studies.length === 0) return '';
        
        return studies.slice(0, 10).map((study, index) => {
            const title = study.title || study.study_name || 'Unknown Study';
            const author = study.author || '';
            const year = study.year || '';
            const doi = study.doi || '';
            const docId = study.doc_id || '';
            const cancerType = study.cancer_type || '';
            const phase = study.phase || study.normalized_phase || '';
            const patients = study.num_patients || study.patient_count || study.number_of_patients || '';
            
            let metaParts = [];
            if (author) metaParts.push(escapeHtml(author));
            if (year) metaParts.push(year);
            if (cancerType) metaParts.push(escapeHtml(cancerType));
            if (phase) metaParts.push('Phase ' + escapeHtml(phase));
            if (patients) metaParts.push('n=' + patients);
            
            let studyBtn = '';
            if (docId || doi) {
                const escapedDocId = (docId || '').replace(/'/g, "\\'");
                const escapedDoi = (doi || '').replace(/'/g, "\\'");
                const escapedTitle = (title || '').replace(/'/g, "\\'");
                studyBtn = '<button class="study-details-btn" ' +
                    'onclick="openStudyDetailsFromSource({doc_id:\'' + escapedDocId + '\',doi:\'' + escapedDoi + '\',title:\'' + escapedTitle + '\'})">' +
                    'View Details</button>';
            }
            
            let reviewBtn = '';
            if (docId && typeof isStudyInComparisonTray === 'function') {
                const inTray = isStudyInComparisonTray(docId);
                const reviewLabel = inTray ? 'In Review Queue' : 'Add to Review';
                const reviewClass = inTray ? 'btn-primary' : 'btn-outline';
                reviewBtn = '<button class="btn ' + reviewClass + ' btn-sm add-to-review-btn" ' +
                    'data-doc-id="' + escapeHtml(docId) + '" ' +
                    'data-title="' + escapeHtml(title) + '" ' +
                    'data-doi="' + escapeHtml(doi || '') + '" ' +
                    'data-year="' + escapeHtml(String(year || '')) + '" ' +
                    'style="font-size: 0.75rem;">' + reviewLabel + '</button>';
            }
            
            return '<div class="panel-study-item">' +
                '<div class="study-title">' + (index + 1) + '. ' + escapeHtml(title) + '</div>' +
                '<div class="study-meta">' + metaParts.join(' | ') + '</div>' +
                '<div class="study-actions">' + studyBtn + reviewBtn + '</div>' +
            '</div>';
        }).join('');
    },

    _renderComparisonCategories(categories, studies) {
        if (!categories || categories.length === 0) return '';
        
        let html = '<div class="comparison-categories">';
        let chartIndex = 0;
        
        categories.forEach(cat => {
            html += '<div class="comparison-category">';
            html += '<div class="category-name">' + escapeHtml(cat.title || cat.category || '') + '</div>';
            
            if (!cat.data_available || !cat.charts || cat.charts.length === 0) {
                html += '<div class="no-data-message">' + escapeHtml(cat.summary || 'No data available for this category.') + '</div>';
            } else {
                html += '<div class="category-charts-grid" id="category-charts-' + cat.category + '">';
                
                cat.charts.forEach(artifact => {
                    if (artifact.artifact_type !== 'chart' || !artifact.chart) return;
                    
                    const chartId = 'splitpanel-chart-' + chartIndex;
                    html += '<div class="chart-card">';
                    html += '<div class="chart-card-title">' + escapeHtml(artifact.chart.title || '') + '</div>';
                    html += '<div class="chart-container" style="height: 200px; position: relative;">';
                    html += '<canvas id="' + chartId + '"></canvas>';
                    html += '</div>';
                    html += '</div>';
                    
                    chartIndex++;
                });
                
                html += '</div>';
            }
            
            if (cat.summary && cat.data_available) {
                html += '<div class="category-summary">' + escapeHtml(cat.summary) + '</div>';
            }
            
            html += '</div>';
        });
        
        html += '</div>';
        return html;
    },

    _renderComparisonCharts(categories) {
        if (!categories || categories.length === 0) return;
        
        let chartIndex = 0;
        const self = this;
        
        categories.forEach(cat => {
            if (!cat.data_available || !cat.charts || cat.charts.length === 0) return;
            
            cat.charts.forEach(artifact => {
                if (artifact.artifact_type !== 'chart' || !artifact.chart) return;
                
                const chartId = 'splitpanel-chart-' + chartIndex;
                const canvas = document.getElementById(chartId);
                
                if (canvas && typeof Chart !== 'undefined') {
                    try {
                        const ctx = canvas.getContext('2d');
                        const chartConfig = artifact.chart;
                        
                        const defaultColors = [
                            'rgba(59, 130, 246, 0.8)',
                            'rgba(16, 185, 129, 0.8)',
                            'rgba(245, 158, 11, 0.8)',
                            'rgba(239, 68, 68, 0.8)',
                        ];
                        const defaultBorderColors = [
                            'rgba(59, 130, 246, 1)',
                            'rgba(16, 185, 129, 1)',
                            'rgba(245, 158, 11, 1)',
                            'rgba(239, 68, 68, 1)',
                        ];
                        
                        const config = {
                            type: chartConfig.type || 'bar',
                            data: {
                                labels: chartConfig.labels || [],
                                datasets: (chartConfig.datasets || []).map((ds, i) => ({
                                    label: ds.label || 'Dataset ' + (i + 1),
                                    data: ds.data || [],
                                    backgroundColor: ds.backgroundColor || defaultColors,
                                    borderColor: ds.borderColor || defaultBorderColors,
                                    borderWidth: 1,
                                }))
                            },
                            options: {
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {
                                    legend: {
                                        display: (chartConfig.datasets || []).length > 1,
                                        position: 'top',
                                    }
                                },
                                scales: chartConfig.type !== 'pie' && chartConfig.type !== 'doughnut' ? {
                                    y: {
                                        beginAtZero: true,
                                        title: {
                                            display: !!chartConfig.unit,
                                            text: chartConfig.unit || ''
                                        }
                                    }
                                } : undefined
                            }
                        };
                        
                        const chart = new Chart(ctx, config);
                        self.chartInstances.push(chart);
                    } catch (error) {
                        console.error('[SplitPanel] Chart render error:', error);
                    }
                }
                
                chartIndex++;
            });
        });
    },

    _renderSources(sources) {
        return sources.map((s, i) => {
            const title = s.title || 'Unknown';
            const year = s.year || '';
            const docId = s.doc_id || '';
            const doi = s.doi || '';
            
            let studyBtn = '';
            if (docId || doi) {
                const escapedDocId = (docId || '').replace(/'/g, "\\'");
                const escapedDoi = (doi || '').replace(/'/g, "\\'");
                const escapedTitle = (title || '').replace(/'/g, "\\'");
                studyBtn = '<button class="study-details-btn btn-sm" ' +
                    'onclick="openStudyDetailsFromSource({doc_id:\'' + escapedDocId + '\',doi:\'' + escapedDoi + '\',title:\'' + escapedTitle + '\'})">' +
                    'Details</button>';
            }
            
            return '<div class="panel-source-item">' +
                '<span class="source-num">' + (i + 1) + '.</span>' +
                '<span class="source-title">' + escapeHtml(title) + '</span>' +
                (year ? '<span class="source-year">(' + year + ')</span>' : '') +
                studyBtn +
            '</div>';
        }).join('');
    },

    _initResultInteractions(container) {
        if (!container) return;

        container.querySelectorAll('.study-details-btn').forEach(btn => {
            function handleStudyClick() {
                const docId = btn.dataset.docId;
                if (docId && typeof openStudyDetailsFromSource === 'function') {
                    openStudyDetailsFromSource({ doc_id: docId });
                }
            }
            btn.addEventListener('click', handleStudyClick);
        });
    },

    _initAddToReviewButtons(container) {
        if (!container) return;
        
        container.querySelectorAll('.add-to-review-btn').forEach(btn => {
            btn.addEventListener('click', function handleAddToReview() {
                const docId = btn.dataset.docId;
                const title = btn.dataset.title;
                const doi = btn.dataset.doi;
                const year = btn.dataset.year;
                
                if (!docId) return;
                
                let tray = typeof getComparisonTray === 'function' ? getComparisonTray() : [];
                const index = tray.findIndex(s => s.doc_id === docId);
                
                if (index >= 0) {
                    tray.splice(index, 1);
                    if (typeof saveComparisonTray === 'function') {
                        saveComparisonTray(tray);
                    }
                    btn.textContent = 'Add to Review';
                    btn.classList.remove('btn-primary');
                    btn.classList.add('btn-outline');
                } else {
                    if (tray.length >= 4) {
                        // Show modal to manage queue instead of alert
                        if (typeof InChatModules !== 'undefined' && typeof InChatModules.showReviewQueueModal === 'function') {
                            InChatModules.showReviewQueueModal(docId, title, doi, year);
                        } else {
                            alert('Maximum 4 studies can be added to the review queue. Please remove one first.');
                        }
                        return;
                    }
                    
                    tray.push({
                        doc_id: docId,
                        title: title,
                        doi: doi,
                        year: year
                    });
                    if (typeof saveComparisonTray === 'function') {
                        saveComparisonTray(tray);
                    }
                    btn.textContent = 'In Review Queue';
                    btn.classList.remove('btn-outline');
                    btn.classList.add('btn-primary');
                }
                
                if (typeof updateComparisonBadge === 'function') {
                    updateComparisonBadge();
                }
            });
        });
    }
};

// Export for use
window.SplitPanel = SplitPanel;
