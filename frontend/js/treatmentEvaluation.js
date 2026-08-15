/**
 * Treatment Evaluation Flow
 * Guided workflow for "Evaluate Treatment Options" feature.
 * Two paths: compare specific treatments or explore all relevant options.
 */

const TreatmentEvaluation = {
    queryContext: null,
    currentMessageId: null,
    clinicalContext: null,
    
    /**
     * Initialize with query context from the current session
     * @param {string} query - The current query
     * @param {Object} result - The query result
     * @param {Object} clinicalContext - Clinical context (cancerType, stage, etc.)
     */
    setContext(query, result, clinicalContext = null) {
        this.queryContext = {
            query: query,
            result: result,
            timestamp: Date.now()
        };
        
        // Store clinical context if provided
        if (clinicalContext) {
            this.clinicalContext = clinicalContext;
        }
        
        // Persist to sessionStorage
        sessionStorage.setItem('treatmentEvalContext', JSON.stringify(this.queryContext));
        if (this.clinicalContext) {
            sessionStorage.setItem('treatmentEvalClinicalContext', JSON.stringify(this.clinicalContext));
        }
        
        console.log('[TreatmentEvaluation] Context set:', {
            query: query,
            clinicalContext: this.clinicalContext
        });
    },
    
    /**
     * Get stored context
     */
    getContext() {
        if (this.queryContext) return this.queryContext;
        const stored = sessionStorage.getItem('treatmentEvalContext');
        if (stored) {
            this.queryContext = JSON.parse(stored);
            return this.queryContext;
        }
        return null;
    },
    
    /**
     * Get stored clinical context
     */
    getClinicalContext() {
        if (this.clinicalContext) return this.clinicalContext;
        const stored = sessionStorage.getItem('treatmentEvalClinicalContext');
        if (stored) {
            this.clinicalContext = JSON.parse(stored);
            return this.clinicalContext;
        }
        return null;
    },
    
    /**
     * Display the evaluation prompt with two option buttons
     */
    showEvaluationPrompt(messageId) {
        this.currentMessageId = messageId;
        const context = this.getContext();
        
        const html = `
            <div class="treatment-eval-prompt">
                <p>Would you like to:</p>
                <div class="treatment-eval-options">
                    <button type="button" class="treatment-eval-btn" data-action="compare-specific">
                        Compare specific treatment options
                    </button>
                    <button type="button" class="treatment-eval-btn" data-action="explore-all">
                        Explore all relevant treatment options
                    </button>
                </div>
            </div>
        `;
        
        this.updateMessage(messageId, 'Treatment Evaluation', html);
        this.initOptionButtons(messageId);
    },
    
    /**
     * Initialize click handlers for option buttons
     */
    initOptionButtons(messageId) {
        const container = document.getElementById(messageId);
        if (!container) return;
        
        const self = this;
        container.querySelectorAll('.treatment-eval-btn').forEach(btn => {
            btn.addEventListener('click', async function handleOptionClick() {
                const action = btn.dataset.action;
                container.querySelectorAll('.treatment-eval-btn').forEach(b => b.disabled = true);
                btn.classList.add('loading');
                
                try {
                    if (action === 'compare-specific') {
                        self.showTreatmentInputForm(messageId);
                    } else if (action === 'explore-all') {
                        await self.discoverAndCompare(messageId);
                    }
                } catch (error) {
                    console.error('[TreatmentEvaluation] Error:', error);
                    self.showError(messageId, error.message);
                }
            });
        });
    },
    
    /**
     * Show treatment input form for manual entry
     */
    showTreatmentInputForm(messageId) {
        const html = `
            <div class="treatment-input-form">
                <p>Enter the treatments you want to compare:</p>
                <div class="treatment-inputs" id="treatment-inputs-${messageId}">
                    <div class="treatment-input-row">
                        <label>Treatment A:</label>
                        <input type="text" class="treatment-input" placeholder="e.g., pembrolizumab">
                    </div>
                    <div class="treatment-input-row">
                        <label>Treatment B:</label>
                        <input type="text" class="treatment-input" placeholder="e.g., chemotherapy">
                    </div>
                    <div class="treatment-input-row">
                        <label>Treatment C:</label>
                        <input type="text" class="treatment-input" placeholder="(optional)">
                    </div>
                </div>
                <button type="button" class="add-treatment-btn" id="add-treatment-${messageId}">
                    + Add another treatment
                </button>
                <div class="validation-message" id="validation-${messageId}" style="display: none;"></div>
                <div class="treatment-form-actions">
                    <button type="button" class="btn btn-accent compare-btn" id="compare-btn-${messageId}">
                        Compare Treatments
                    </button>
                </div>
            </div>
        `;
        
        this.updateMessage(messageId, 'Treatment Evaluation', html);
        this.initFormHandlers(messageId);
    },
    
    /**
     * Initialize form handlers
     */
    initFormHandlers(messageId) {
        const self = this;
        let treatmentCount = 3;
        
        // Add treatment button
        const addBtn = document.getElementById(`add-treatment-${messageId}`);
        if (addBtn) {
            addBtn.addEventListener('click', function handleAddTreatment() {
                treatmentCount++;
                const inputsContainer = document.getElementById(`treatment-inputs-${messageId}`);
                if (inputsContainer) {
                    const newRow = document.createElement('div');
                    newRow.className = 'treatment-input-row';
                    newRow.innerHTML = `
                        <label>Treatment ${String.fromCharCode(64 + treatmentCount)}:</label>
                        <input type="text" class="treatment-input" placeholder="(optional)">
                    `;
                    inputsContainer.appendChild(newRow);
                }
            });
        }
        
        // Compare button
        const compareBtn = document.getElementById(`compare-btn-${messageId}`);
        if (compareBtn) {
            compareBtn.addEventListener('click', async function handleCompare() {
                const inputs = document.querySelectorAll(`#treatment-inputs-${messageId} .treatment-input`);
                const treatments = [];
                inputs.forEach(input => {
                    const val = input.value.trim();
                    if (val) treatments.push(val);
                });
                
                // Validate minimum 2 treatments
                if (treatments.length < 2) {
                    const validationMsg = document.getElementById(`validation-${messageId}`);
                    if (validationMsg) {
                        validationMsg.textContent = 'Please enter at least 2 treatments to compare.';
                        validationMsg.style.display = 'block';
                    }
                    return;
                }
                
                compareBtn.disabled = true;
                compareBtn.textContent = 'Comparing...';
                
                try {
                    await self.executeComparison(messageId, treatments);
                } catch (error) {
                    console.error('[TreatmentEvaluation] Comparison error:', error);
                    self.showError(messageId, error.message);
                }
            });
        }
    },
    
    /**
     * Discover treatments from context and run comparison
     */
    async discoverAndCompare(messageId) {
        const context = this.getContext();
        const clinicalContext = this.getClinicalContext();
        
        // Check for valid context with sufficient query length
        if (!context || !context.query || context.query.length < 10) {
            // Try to get context from sessionStorage
            let fallbackQuery = null;
            try {
                const followup = sessionStorage.getItem('followupContext');
                if (followup) {
                    const parsed = JSON.parse(followup);
                    if (parsed.query && parsed.query.length >= 10) {
                        fallbackQuery = parsed.query;
                    }
                }
            } catch (e) {
                console.warn('[TreatmentEvaluation] Failed to parse followup context:', e);
            }
            
            if (!fallbackQuery) {
                this.showError(messageId, 'No clinical context available. Please provide patient details first.');
                return;
            }
            
            // Use fallback query
            this.setContext(fallbackQuery, null);
        }
        
        const effectiveContext = this.getContext();
        
        // Show loading state
        this.updateMessage(messageId, 'Treatment Evaluation', `
            <div class="loading">
                <div class="spinner"></div>
                <span>Discovering relevant treatment options from the literature...</span>
            </div>
        `);
        
        try {
            const api = new PaxisAPI();
            
            // Build a discovery query that includes clinical context (cancer type)
            let discoveryQuery = '';
            const cancerType = clinicalContext?.cancerType;
            const queryText = effectiveContext.query;
            
            if (cancerType) {
                // Include cancer type in the query for proper context
                // Check if the query already mentions the cancer type
                const queryLower = queryText.toLowerCase();
                const cancerLower = cancerType.toLowerCase();
                if (queryLower.includes(cancerLower) || queryLower.includes('nsclc') || queryLower.includes('sclc')) {
                    discoveryQuery = `What are the treatment options for ${queryText}`;
                } else {
                    discoveryQuery = `What are the treatment options for ${cancerType} regarding ${queryText}`;
                }
                console.log('[TreatmentEvaluation] Using clinical context, cancer type:', cancerType);
            } else {
                discoveryQuery = `What are the treatment options for ${queryText}`;
            }
            
            console.log('[TreatmentEvaluation] Discovery query:', discoveryQuery);
            
            // Use the visual comparison endpoint - it will extract treatments automatically
            const result = await api.visualComparison(discoveryQuery, 15);
            
            console.log('[TreatmentEvaluation] API result:', {
                hasArms: result.treatment_arms?.length || 0,
                hasSummary: !!result.summary,
                hasAnalysis: !!result.detailed_analysis
            });
            
            // Render the results
            this.renderComparisonResults(messageId, result);
        } catch (error) {
            console.error('[TreatmentEvaluation] Discovery error:', error);
            this.showError(messageId, error.message);
        }
    },
    
    /**
     * Execute comparison with specified treatments
     */
    async executeComparison(messageId, treatments) {
        const context = this.getContext();
        const clinicalContext = this.getClinicalContext();
        
        // Show loading state
        this.updateMessage(messageId, 'Treatment Evaluation', `
            <div class="loading">
                <div class="spinner"></div>
                <span>Comparing treatments...</span>
            </div>
        `);
        
        try {
            const api = new PaxisAPI();
            
            // Build comparison query from treatments and context
            const treatmentList = treatments.join(' vs ');
            let comparisonQuery = `Compare ${treatmentList}`;
            
            // Add clinical context (cancer type) if available
            const cancerType = clinicalContext?.cancerType;
            if (cancerType) {
                comparisonQuery += ` for ${cancerType}`;
                console.log('[TreatmentEvaluation] Using clinical context in comparison, cancer type:', cancerType);
            }
            
            // Add query context if available and doesn't duplicate cancer type
            if (context && context.query) {
                const queryLower = context.query.toLowerCase();
                const alreadyHasCancer = cancerType && queryLower.includes(cancerType.toLowerCase());
                if (!alreadyHasCancer) {
                    comparisonQuery += ` regarding ${context.query}`;
                }
            }
            
            console.log('[TreatmentEvaluation] Comparison query:', comparisonQuery);
            
            const result = await api.visualComparison(comparisonQuery, 15);
            
            // Render the results
            this.renderComparisonResults(messageId, result);
        } catch (error) {
            console.error('[TreatmentEvaluation] Comparison error:', error);
            this.showError(messageId, error.message);
        }
    },
    
    /**
     * Render comparison results with treatment arms breakdown
     */
    renderComparisonResults(messageId, result) {
        const context = this.getContext();
        const clinicalContext = this.getClinicalContext();
        const queryText = context ? context.query : '';
        const cancerType = clinicalContext?.cancerType;
        
        // Check if we have any meaningful results
        const hasArms = result.treatment_arms && result.treatment_arms.length > 0;
        const totalStudies = hasArms ? result.treatment_arms.reduce((sum, arm) => {
            return sum + this.countUniqueStudies(arm.retrieval_results || []);
        }, 0) : 0;
        const hasSummary = result.summary || result.short_answer || result.detailed_analysis;
        
        // If no studies and no meaningful content, show helpful message
        if (totalStudies === 0 && !hasSummary) {
            this.updateMessage(messageId, 'Treatment Evaluation', `
                <div class="alert alert-info">
                    <p>No relevant studies found for this query.</p>
                    <p style="margin-top: 0.5rem; font-size: 0.9rem;">Try:</p>
                    <ul style="margin: 0.5rem 0; padding-left: 1.5rem; font-size: 0.9rem;">
                        <li>Adding more specific details about the cancer type or stage</li>
                        <li>Using the "Compare specific treatments" option with treatment names</li>
                        <li>Searching directly on the treatment-comparison page</li>
                    </ul>
                </div>
                <div class="continue-research-section" style="margin-top: 1rem;">
                    <button type="button" class="btn btn-outline btn-sm new-conversation-btn">Start a New Conversation</button>
                </div>
            `);
            this.initContinueButtons(messageId);
            return;
        }
        
        let html = '';
        
        // Summary section with query and clinical context
        const summaryText = result.summary || result.short_answer || '';
        html += `<div class="module-section">
            <h4>Summary</h4>
            ${cancerType ? `<div class="comparison-context"><strong>Cancer Type:</strong> ${escapeHtml(cancerType)}</div>` : ''}
            ${queryText ? `<div class="comparison-query"><strong>Query:</strong> ${escapeHtml(queryText)}</div>` : ''}
            <div class="module-summary">${summaryText ? markdownToHtml(summaryText) : 'No summary available.'}</div>
        </div>`;
        
        // Detailed analysis section - SHOW PROMINENTLY (not collapsed)
        const analysis = result.detailed_analysis || result.justification || '';
        if (analysis) {
            html += `<div class="module-section">
                <h4>Comparative Analysis</h4>
                <div class="module-analysis comparison-analysis">${markdownToHtml(analysis)}</div>
            </div>`;
        }
        
        // Charts section
        if (result.charts && result.charts.length > 0) {
            html += `<div class="module-section">
                <h4>Visual Comparison</h4>
                <div class="module-charts" id="charts-${messageId}"></div>
            </div>`;
        }
        
        // Treatment arms breakdown with study counts (collapsible - secondary info)
        if (result.treatment_arms && result.treatment_arms.length > 0) {
            const totalStudies = result.treatment_arms.reduce((sum, arm) => {
                const armStudies = this.countUniqueStudies(arm.retrieval_results || []);
                return sum + armStudies;
            }, 0);
            
            html += `<div class="module-section">
                <details class="arm-details" open>
                    <summary>Supporting Evidence (${totalStudies} studies across ${result.treatment_arms.length} arms)</summary>
                    <div class="treatment-arms-grid" style="margin-top: 1rem;">
                        ${result.treatment_arms.map(arm => {
                            const studyCount = this.countUniqueStudies(arm.retrieval_results || []);
                            return `<div class="treatment-arm-card">
                                <div class="arm-label">${escapeHtml(arm.arm_label)}</div>
                                <div class="arm-count">${studyCount} ${studyCount === 1 ? 'study' : 'studies'}</div>
                            </div>`;
                        }).join('')}
                    </div>
                    <div class="arm-details-content" style="margin-top: 1rem;">
                        ${this.renderArmDetails(result.treatment_arms)}
                    </div>
                </details>
            </div>`;
        }
        
        // Continue your research section
        html += `<div class="continue-research-section">
            <p class="continue-research-label">Continue your research</p>
            <div class="continue-research-buttons">
                <button type="button" class="btn btn-outline btn-sm continue-btn" data-action="studies">
                    Review Studies
                </button>
                <button type="button" class="btn btn-outline btn-sm continue-btn" data-action="analytics">
                    Advanced Analytics
                </button>
            </div>
            <div class="new-conversation-divider">
                <span>or</span>
            </div>
            <button type="button" class="btn btn-outline btn-sm new-conversation-btn">Start a New Conversation</button>
        </div>`;
        
        this.updateMessage(messageId, 'Treatment Comparison', html);
        
        // Initialize toggle buttons for collapsible study cards
        this.initStudyToggleButtons(messageId);
        
        // Initialize continue research buttons
        this.initContinueButtons(messageId);
        
        // Initialize add-to-review button handlers
        if (typeof InChatModules !== 'undefined' && InChatModules.initAddToReviewButtons) {
            InChatModules.initAddToReviewButtons(messageId);
        }
        
        // Render charts after DOM update
        if (result.charts && result.charts.length > 0) {
            setTimeout(() => {
                if (typeof InChatModules !== 'undefined' && InChatModules.renderCharts) {
                    InChatModules.renderCharts(`charts-${messageId}`, result.charts);
                }
            }, 100);
        }
    },
    
    /**
     * Initialize toggle buttons for collapsible study cards
     */
    initStudyToggleButtons(messageId) {
        const container = document.getElementById(messageId);
        if (!container) return;
        
        container.querySelectorAll('.study-breakdown-toggle').forEach(btn => {
            btn.addEventListener('click', function handleStudyToggle() {
                const targetId = btn.dataset.target;
                const body = document.getElementById(targetId);
                if (!body) return;
                const expanded = btn.getAttribute('aria-expanded') === 'true';
                btn.setAttribute('aria-expanded', String(!expanded));
                body.style.display = expanded ? 'none' : 'block';
                const arrow = btn.querySelector('.study-breakdown-arrow');
                if (arrow) arrow.style.transform = expanded ? '' : 'rotate(180deg)';
            });
        });
    },
    
    /**
     * Initialize continue research button handlers
     */
    initContinueButtons(messageId) {
        const container = document.getElementById(messageId);
        if (!container) return;
        
        const context = this.getContext();
        const query = context ? context.query : '';
        
        container.querySelectorAll('.continue-btn').forEach(btn => {
            btn.addEventListener('click', function handleContinueClick() {
                const action = btn.dataset.action;
                
                // Store context for the target page
                if (query) {
                    sessionStorage.setItem('followupContext', JSON.stringify({
                        query: query,
                        timestamp: Date.now()
                    }));
                }
                
                switch (action) {
                    case 'query':
                        // Focus the chat input on the same page
                        const chatInput = document.getElementById('chatInput');
                        if (chatInput) {
                            chatInput.focus();
                            chatInput.scrollIntoView({ behavior: 'smooth' });
                        }
                        break;
                    case 'studies':
                        // Execute study comparison in chat
                        if (typeof InChatModules !== 'undefined' && query && query.length >= 10) {
                            InChatModules.executeStudyComparison(query);
                        } else if (typeof InChatModules !== 'undefined' && InChatModules.showStudyReviewForm) {
                            InChatModules.showStudyReviewForm(query);
                        } else {
                            window.location.href = 'study-comparison.html';
                        }
                        break;
                    case 'analytics':
                        window.location.href = 'analytics.html';
                        break;
                }
            });
        });
        
        // Initialize new conversation button
        const newConvBtn = container.querySelector('.new-conversation-btn');
        if (newConvBtn) {
            newConvBtn.addEventListener('click', function handleNewConversation() {
                if (typeof InChatModules !== 'undefined' && InChatModules.showExitConversationModal) {
                    InChatModules.showExitConversationModal();
                } else {
                    // Fallback: just reload the page
                    window.location.reload();
                }
            });
        }
    },
    
    /**
     * Count unique studies from retrieval results
     */
    countUniqueStudies(retrievalResults) {
        const docIds = new Set();
        (retrievalResults || []).forEach(r => {
            if (r.doc_id) docIds.add(r.doc_id);
        });
        return docIds.size;
    },
    
    /**
     * Render detailed per-arm study list with action buttons and rich profile data
     * Matches the rendering style of treatment-comparison.html
     */
    renderArmDetails(treatmentArms) {
        return treatmentArms.map((arm, armIdx) => {
            const sources = arm.retrieval_results || [];
            const profiles = arm.study_profiles || [];
            
            // Build profile lookup by doc_id
            const profileMap = new Map();
            profiles.forEach(p => {
                const did = p.doc_id || (p.study_details && p.study_details.doc_id && p.study_details.doc_id.value) || '';
                if (did) profileMap.set(did, p);
            });
            
            // Group chunks by study
            const studyMap = new Map();
            sources.forEach(chunk => {
                const key = chunk.doc_id || chunk.title || 'unknown';
                if (!studyMap.has(key)) {
                    studyMap.set(key, {
                        doc_id: chunk.doc_id || '',
                        title: chunk.title || 'Unknown Study',
                        author: chunk.author || '',
                        citation: chunk.citation || '',
                        doi: chunk.doi || '',
                        pmid: chunk.pmid || '',
                        year: chunk.year || '',
                        category: chunk.category || '',
                        chunks: []
                    });
                }
                studyMap.get(key).chunks.push({
                    section: chunk.section || chunk.chunk_type || 'General',
                    content: chunk.content || '',
                    score: chunk.score || 0
                });
            });
            
            // Merge profiles + raw chunks
            const studies = [];
            const usedDocIds = new Set();
            
            profileMap.forEach((profile, docId) => {
                usedDocIds.add(docId);
                const rawStudy = studyMap.get(docId);
                const bestScore = rawStudy ? Math.max(...rawStudy.chunks.map(c => c.score || 0)) : 0;
                studies.push({ profile, rawStudy, docId, score: bestScore });
            });
            
            studyMap.forEach((rawStudy, key) => {
                if (!usedDocIds.has(key)) {
                    studies.push({ profile: null, rawStudy, docId: key, score: Math.max(...rawStudy.chunks.map(c => c.score || 0)) });
                }
            });
            
            studies.sort((a, b) => b.score - a.score);
            
            // Render study cards for this arm
            const studyCards = studies.map((entry, idx) => {
                const studyId = `arm-${armIdx}-study-${idx}`;
                const profile = entry.profile;
                const raw = entry.rawStudy;
                
                const title = profile ? (profile.title || this.getProfileField(profile, 'study_details', 'study_name') || 'Unknown Study') : (raw ? raw.title : 'Unknown Study');
                const year = profile ? (this.getProfileField(profile, 'study_details', 'publish_date') || '') : (raw ? raw.year : '');
                const yearStr = year ? ` (${year})` : '';
                const citation = raw ? raw.citation : '';
                const docId = entry.docId;
                const doi = raw ? raw.doi : '';
                const pmid = raw ? raw.pmid : '';
                const nPatients = profile ? this.getProfileField(profile, 'study_details', 'number_of_patients') : null;
                const studyType = profile ? this.getProfileField(profile, 'study_details', 'study_type') : null;
                const studyPhase = profile ? this.getProfileField(profile, 'study_details', 'study_phase') : null;
                
                let metaParts = [];
                if (citation) metaParts.push(escapeHtml(citation));
                if (nPatients) metaParts.push(`${escapeHtml(String(nPatients))} patients`);
                if (studyType) metaParts.push(escapeHtml(String(studyType)));
                if (studyPhase) metaParts.push(escapeHtml(String(studyPhase)));
                if (!profile && raw) metaParts.push(`${raw.chunks.length} evidence section${raw.chunks.length !== 1 ? 's' : ''}`);
                const metaStr = metaParts.join(' | ');
                
                // Study Details button
                let detailsBtn = '';
                if (docId || pmid || doi) {
                    const eid = (docId || '').replace(/'/g, "\\'").replace(/"/g, '&quot;');
                    const epmid = (pmid || '').replace(/'/g, "\\'").replace(/"/g, '&quot;');
                    const edoi = (doi || '').replace(/'/g, "\\'").replace(/"/g, '&quot;');
                    const etitle = (title || '').replace(/'/g, "\\'").replace(/"/g, '&quot;');
                    detailsBtn = `<button class="study-details-btn" onclick="openStudyDetailsFromSource({doc_id:'${eid}',pmid:'${epmid}',doi:'${edoi}',title:'${etitle}'})" title="View detailed study information">Study Details</button>`;
                }
                
                // Build body content
                let bodyContent = '';
                if (profile && profile.abstract) {
                    bodyContent = this.renderAbstract(profile.abstract, profile);
                } else if (profile) {
                    bodyContent = this.renderStructuredProfile(profile);
                } else if (raw) {
                    bodyContent = this.renderRawChunks(raw);
                }
                
                // Add to Review button
                let reviewBtn = '';
                if (docId && typeof isStudyInComparisonTray === 'function') {
                    const inTray = isStudyInComparisonTray(docId);
                    const reviewLabel = inTray ? 'In Review Queue' : 'Add to Study Review';
                    const reviewClass = inTray ? 'btn-primary' : 'btn-outline';
                    reviewBtn = `<button class="btn ${reviewClass} btn-sm add-to-review-btn" data-doc-id="${escapeHtml(docId)}" data-title="${escapeHtml(title)}" data-doi="${escapeHtml(doi || '')}" data-year="${escapeHtml(String(year || ''))}" style="font-size: 0.75rem;">${reviewLabel}</button>`;
                }
                
                return `
                    <div class="card" style="margin-bottom: 0.5rem; border-left: 3px solid var(--accent-color);">
                        <div class="card-body" style="padding: 0.75rem 1rem;">
                            <button type="button" class="study-breakdown-toggle" aria-expanded="false" data-target="${studyId}">
                                <div style="flex: 1; text-align: left;">
                                    <span class="study-breakdown-title">${escapeHtml(title)}${yearStr}</span>
                                    <div class="study-breakdown-meta">${metaStr}</div>
                                </div>
                                <span class="study-breakdown-arrow">&#9660;</span>
                            </button>
                            <div id="${studyId}" class="study-breakdown-body" style="display: none;">
                                <div style="margin-top: 0.75rem; display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap;">
                                    ${detailsBtn}
                                    ${reviewBtn}
                                    ${doi ? `<a href="https://doi.org/${doi}" target="_blank" rel="noopener" class="btn btn-outline btn-sm" style="font-size: 0.75rem;">DOI</a>` : ''}
                                </div>
                                <div class="study-profile-sections" style="margin-top: 0.75rem;">
                                    ${bodyContent}
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
            
            return `
                <div class="treatment-arm-section" style="margin-bottom: 1.5rem;">
                    <div class="treatment-arm-header" style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.75rem; padding-bottom: 0.5rem; border-bottom: 2px solid var(--accent-color);">
                        <h5 style="margin: 0; font-size: 1rem;">${escapeHtml(arm.arm_label)}</h5>
                        <span style="color: var(--gray-500); font-size: 0.85rem;">${studies.length} ${studies.length === 1 ? 'study' : 'studies'}</span>
                    </div>
                    ${studyCards || '<p style="color: var(--gray-500);">No studies found for this treatment arm.</p>'}
                </div>
            `;
        }).join('');
    },
    
    /**
     * Helper: get a scalar value from a profile section
     */
    getProfileField(profile, section, field) {
        const sec = profile[section];
        if (!sec) return null;
        const f = sec[field];
        if (!f) return null;
        if (typeof f === 'object' && f.value !== undefined) return f.value;
        return f;
    },
    
    /**
     * Render abstract with key study metadata
     */
    renderAbstract(abstractText, profile) {
        let html = '';
        
        // Key metadata pills
        const pills = [];
        const studyType = this.getProfileField(profile, 'study_details', 'study_type');
        const studyPhase = this.getProfileField(profile, 'study_details', 'study_phase');
        const nPatients = this.getProfileField(profile, 'study_details', 'number_of_patients');
        const cancerLoc = this.getProfileField(profile, 'diagnosis', 'cancer_location');
        const cancerType = this.getProfileField(profile, 'diagnosis', 'cancer_type');
        
        if (studyType) pills.push(escapeHtml(String(studyType)));
        if (studyPhase) pills.push(escapeHtml(String(studyPhase)));
        if (nPatients) pills.push(escapeHtml(String(nPatients)) + ' patients');
        if (cancerLoc) pills.push(escapeHtml(String(cancerLoc)));
        if (cancerType && cancerType !== cancerLoc) pills.push(escapeHtml(String(cancerType)));
        
        if (pills.length > 0) {
            html += '<div class="study-meta-pills" style="display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 0.75rem;">';
            pills.forEach(pill => {
                html += `<span style="background: var(--gray-100); color: var(--gray-700); font-size: 0.75rem; padding: 0.2rem 0.6rem; border-radius: 9999px;">${pill}</span>`;
            });
            html += '</div>';
        }
        
        // Abstract text (truncated for in-chat display)
        const truncatedAbstract = abstractText.length > 400 ? abstractText.substring(0, 400) + '...' : abstractText;
        html += `<div class="study-abstract" style="font-size: 0.85rem; line-height: 1.5; color: var(--gray-700);">${escapeHtml(truncatedAbstract)}</div>`;
        
        // Key outcomes summary if available
        const outcomes = profile.outcomes || {};
        const outcomeLines = [];
        for (const [key, val] of Object.entries(outcomes)) {
            if (!val || val.value === null || val.value === undefined) continue;
            const label = val.label || key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            outcomeLines.push(`<strong>${escapeHtml(label)}:</strong> ${escapeHtml(String(val.value))}`);
        }
        if (outcomeLines.length > 0 && outcomeLines.length <= 5) {
            html += `<div class="profile-section" style="margin-top: 0.75rem;">
                <div class="profile-section-label" style="font-weight: 600; font-size: 0.8rem; color: var(--gray-600); margin-bottom: 0.25rem;">Key Outcomes</div>
                ${outcomeLines.map(line => '<div class="profile-item" style="font-size: 0.85rem;">' + line + '</div>').join('')}
            </div>`;
        }
        
        return html;
    },
    
    /**
     * Render structured profile sections (treatment-focused)
     */
    renderStructuredProfile(profile) {
        let html = '';
        
        // Treatment Arms
        const arms = (profile.treatment && profile.treatment.study_arms) || [];
        if (arms.length > 0) {
            html += this.renderProfileSection('Study Arms', arms.slice(0, 3).map(arm => {
                const lines = [];
                if (arm.arm_name) lines.push(`<strong>${escapeHtml(arm.arm_name)}</strong>`);
                if (arm.number_of_patients) lines.push(`N=${escapeHtml(String(arm.number_of_patients))}`);
                if (arm.description) lines.push(escapeHtml(arm.description.substring(0, 150)));
                return { lines };
            }));
        }
        
        // Outcomes (brief)
        const outcomes = profile.outcomes || {};
        const outcomeItems = [];
        for (const [key, val] of Object.entries(outcomes)) {
            if (!val || val.value === null || val.value === undefined) continue;
            const label = val.label || key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            outcomeItems.push({
                lines: [`<strong>${escapeHtml(label)}:</strong> ${escapeHtml(String(val.value))}`]
            });
        }
        if (outcomeItems.length > 0) {
            html += this.renderProfileSection('Outcomes', outcomeItems.slice(0, 4));
        }
        
        if (!html) {
            html = '<div class="study-chunk-text" style="color: var(--gray-500); font-size: 0.85rem;">No structured treatment data available for this study.</div>';
        }
        
        return html;
    },
    
    /**
     * Render a single profile section with items
     */
    renderProfileSection(title, items) {
        let itemsHtml = items.map(item => {
            const content = item.lines.join('<br>');
            return `<div class="profile-item" style="font-size: 0.85rem; margin-bottom: 0.25rem;">${content}</div>`;
        }).join('');
        
        return `
            <div class="profile-section" style="margin-bottom: 0.75rem;">
                <div class="profile-section-label" style="font-weight: 600; font-size: 0.8rem; color: var(--gray-600); margin-bottom: 0.25rem;">${escapeHtml(title)}</div>
                ${itemsHtml}
            </div>
        `;
    },
    
    /**
     * Fallback: render raw Qdrant chunks grouped by section
     */
    renderRawChunks(study) {
        const sectionMap = new Map();
        study.chunks.forEach(c => {
            const sec = c.section || 'General';
            if (!sectionMap.has(sec)) sectionMap.set(sec, []);
            sectionMap.get(sec).push(c);
        });
        
        let html = '';
        let sectionCount = 0;
        sectionMap.forEach((chunks, section) => {
            if (sectionCount >= 2) return; // Limit sections for in-chat display
            sectionCount++;
            const sectionLabel = section.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            html += `<div class="study-chunk-section" style="margin-bottom: 0.5rem;"><div class="study-chunk-section-label" style="font-weight: 600; font-size: 0.8rem; color: var(--gray-600); margin-bottom: 0.25rem;">${escapeHtml(sectionLabel)}</div>`;
            chunks.slice(0, 1).forEach(c => {
                const text = c.content.length > 300 ? c.content.substring(0, 300) + '...' : c.content;
                html += `<div class="study-chunk-text" style="font-size: 0.85rem; color: var(--gray-700);">${escapeHtml(text)}</div>`;
            });
            html += '</div>';
        });
        return html;
    },
    
    /**
     * Get unique studies from retrieval results
     */
    getUniqueStudies(retrievalResults) {
        const studyMap = new Map();
        (retrievalResults || []).forEach(r => {
            if (r.doc_id && !studyMap.has(r.doc_id)) {
                studyMap.set(r.doc_id, {
                    doc_id: r.doc_id,
                    title: r.title,
                    author: r.author,
                    year: r.year,
                    doi: r.doi
                });
            }
        });
        return Array.from(studyMap.values());
    },
    
    /**
     * Update message content
     */
    updateMessage(messageId, title, contentHtml) {
        const messageDiv = document.getElementById(messageId);
        if (!messageDiv) return;
        
        messageDiv.innerHTML = `
            <div class="message-avatar" style="background: var(--gray-200); color: var(--gray-700);"><img src="assets/paxis-mark.png" alt="Paxis" style="width:22px;height:20px;object-fit:contain;"></div>
            <div class="message-content">
                <div style="font-weight: 600; margin-bottom: 0.5rem; color: var(--gray-900);">${escapeHtml(title)}</div>
                <div class="module-result-content">${contentHtml}</div>
            </div>
        `;
        
        const chatMessages = document.getElementById('chatMessages');
        if (chatMessages) {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    },
    
    /**
     * Show error message
     */
    showError(messageId, errorMessage) {
        this.updateMessage(messageId, 'Treatment Evaluation', `
            <div class="alert alert-danger">Error: ${escapeHtml(errorMessage)}</div>
        `);
    }
};
