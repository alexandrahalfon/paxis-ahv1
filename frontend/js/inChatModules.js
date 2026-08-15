/**
 * In-Chat Module Execution
 * Allows running specialized modules (Treatment Comparison, Patient Matching, etc.)
 * directly within the home page chat interface.
 */

const InChatModules = {
    // Store the last query context for module execution
    lastQueryContext: null,
    
    // Store clinical context extracted from conversation
    clinicalContext: null,
    
    // Chart instances for cleanup
    chartInstances: [],
    
    /**
     * Set the context from the last query for module execution
     * Also extracts and accumulates clinical context (cancer type, stage, etc.)
     * @param {string} query - The current query
     * @param {Object} result - The query result from the API
     * @param {Array} conversationHistory - Optional conversation history array
     */
    setQueryContext(query, result, conversationHistory = null) {
        this.lastQueryContext = {
            query: query,
            result: result,
            timestamp: Date.now()
        };
        
        // Extract clinical context from result metadata
        this._updateClinicalContext(result, conversationHistory);
        
        console.log('[InChatModules] Context set, clinical:', this.clinicalContext);
    },
    
    /**
     * Extract and accumulate clinical context from query results
     * @private
     */
    _updateClinicalContext(result, conversationHistory) {
        // Try to extract from result.metadata.query_structure
        if (result && result.metadata && result.metadata.query_structure) {
            const qs = result.metadata.query_structure;
            const cancer = qs.cancer || {};
            const patient = qs.patient || {};
            const treatment = qs.treatment || {};
            
            // Only update if we have meaningful clinical data
            if (cancer.site || cancer.histology || patient.age || treatment.modality) {
                this.clinicalContext = {
                    cancerType: cancer.site || this.clinicalContext?.cancerType || null,
                    cancerStage: cancer.stage || this.clinicalContext?.cancerStage || null,
                    histology: cancer.histology || this.clinicalContext?.histology || null,
                    treatment: treatment.modality || this.clinicalContext?.treatment || null,
                    patientAge: patient.age || this.clinicalContext?.patientAge || null,
                    timestamp: Date.now()
                };
                console.log('[InChatModules] Updated clinical context from query_structure:', this.clinicalContext);
                return;
            }
        }
        
        // Try to extract from extracted_profile (Trial Match mode)
        // Note: patient_profile from analyze-intent uses 'stage' not 'cancer_stage'
        if (result && result.extracted_profile) {
            const profile = result.extracted_profile;
            if (profile.cancer_type || profile.histology || profile.age) {
                this.clinicalContext = {
                    cancerType: profile.cancer_type || this.clinicalContext?.cancerType || null,
                    cancerStage: profile.cancer_stage || profile.stage || this.clinicalContext?.cancerStage || null,
                    histology: profile.histology || this.clinicalContext?.histology || null,
                    treatment: profile.treatment || profile.prior_treatment || this.clinicalContext?.treatment || null,
                    patientAge: profile.age || this.clinicalContext?.patientAge || null,
                    timestamp: Date.now()
                };
                console.log('[InChatModules] Updated clinical context from extracted_profile:', this.clinicalContext);
                return;
            }
        }
        
        // Try to extract from accumulated_context
        if (result && result.accumulated_context) {
            const ac = result.accumulated_context;
            const cancer = ac.cancer || {};
            if (cancer.site) {
                this.clinicalContext = {
                    cancerType: cancer.site || this.clinicalContext?.cancerType || null,
                    cancerStage: cancer.stage || this.clinicalContext?.cancerStage || null,
                    histology: cancer.histology || this.clinicalContext?.histology || null,
                    treatment: (ac.treatment || {}).modality || this.clinicalContext?.treatment || null,
                    patientAge: (ac.patient || {}).age || this.clinicalContext?.patientAge || null,
                    timestamp: Date.now()
                };
                console.log('[InChatModules] Updated clinical context from accumulated_context:', this.clinicalContext);
                return;
            }
        }
        
        // Fallback: try to extract cancer type from conversation history
        if (conversationHistory && conversationHistory.length > 0 && !this.clinicalContext?.cancerType) {
            const cancerType = this._extractCancerTypeFromHistory(conversationHistory);
            if (cancerType) {
                this.clinicalContext = this.clinicalContext || {};
                this.clinicalContext.cancerType = cancerType;
                this.clinicalContext.timestamp = Date.now();
                console.log('[InChatModules] Updated clinical context from conversation history:', this.clinicalContext);
            }
        }
    },
    
    /**
     * Extract cancer type from conversation history using keyword matching
     * @private
     */
    _extractCancerTypeFromHistory(history) {
        const cancerPatterns = [
            { pattern: /\b(nsclc|non[- ]?small[- ]?cell[- ]?lung[- ]?cancer)\b/i, type: 'NSCLC' },
            { pattern: /\b(sclc|small[- ]?cell[- ]?lung[- ]?cancer)\b/i, type: 'SCLC' },
            { pattern: /\blung[- ]?cancer\b/i, type: 'lung cancer' },
            { pattern: /\bbreast[- ]?cancer\b/i, type: 'breast cancer' },
            { pattern: /\bprostate[- ]?cancer\b/i, type: 'prostate cancer' },
            { pattern: /\bcolorectal[- ]?cancer\b/i, type: 'colorectal cancer' },
            { pattern: /\bpancreatic[- ]?cancer\b/i, type: 'pancreatic cancer' },
            { pattern: /\bovarian[- ]?cancer\b/i, type: 'ovarian cancer' },
            { pattern: /\bmelanoma\b/i, type: 'melanoma' },
            { pattern: /\bleukemia\b/i, type: 'leukemia' },
            { pattern: /\blymphoma\b/i, type: 'lymphoma' },
            { pattern: /\bglioblastoma\b/i, type: 'glioblastoma' },
            { pattern: /\bhead[- ]?and[- ]?neck[- ]?cancer\b/i, type: 'head and neck cancer' },
            { pattern: /\besophageal[- ]?cancer\b/i, type: 'esophageal cancer' },
            { pattern: /\bgastric[- ]?cancer\b/i, type: 'gastric cancer' },
            { pattern: /\bhcc|hepatocellular[- ]?carcinoma\b/i, type: 'HCC' },
            { pattern: /\brcc|renal[- ]?cell[- ]?carcinoma\b/i, type: 'RCC' },
            { pattern: /\bbladder[- ]?cancer\b/i, type: 'bladder cancer' }
        ];
        
        // Search through history from oldest to newest to find the first cancer mention
        for (const entry of history) {
            const content = entry.content || '';
            for (const { pattern, type } of cancerPatterns) {
                if (pattern.test(content)) {
                    console.log('[InChatModules] Extracted cancer type from history:', type);
                    return type;
                }
            }
        }
        return null;
    },
    
    /**
     * Get the current clinical context
     * @returns {Object|null} Clinical context with cancerType, stage, etc.
     */
    getClinicalContext() {
        return this.clinicalContext;
    },
    
    /**
     * Clear clinical context (e.g., when starting new conversation)
     */
    clearClinicalContext() {
        this.clinicalContext = null;
        this.lastQueryContext = null;
        
        // Clear all conversation-related session storage
        sessionStorage.removeItem('followupContext');
        sessionStorage.removeItem('treatmentEvalContext');
        sessionStorage.removeItem('treatmentEvalClinicalContext');
        
        // Clear TreatmentEvaluation context if it exists
        if (typeof TreatmentEvaluation !== 'undefined') {
            TreatmentEvaluation.queryContext = null;
            TreatmentEvaluation.clinicalContext = null;
            TreatmentEvaluation.currentMessageId = null;
        }
        
        console.log('[InChatModules] Clinical context cleared');
    },
    
    /**
     * Generate the module action buttons HTML
     * @param {string} originalQuery - The original query
     * @param {boolean} standalone - If true, includes the label; if false, just returns buttons
     */
    generateModuleButtons(originalQuery, standalone = true) {
        const encodedQuery = encodeURIComponent(originalQuery || '');
        const buttonsHtml = `
            <button type="button" class="module-action-btn" data-module="treatment-comparison" data-query="${encodedQuery}">
                Evaluate Treatment Options
            </button>
            <button type="button" class="module-action-btn" data-module="patient-matching" data-query="${encodedQuery}">
                Match Patient to Trials
            </button>
            <button type="button" class="module-action-btn" data-module="study-comparison" data-query="${encodedQuery}">
                Review and Compare Relevant Studies
            </button>
            <button type="button" class="module-action-btn" data-module="analytics" data-query="${encodedQuery}">
                Explore Advanced Analytics
            </button>
        `;
        
        if (standalone) {
            return `
                <div class="module-actions-container">
                    <div class="module-actions-label">Would you like to:</div>
                    <div class="module-actions-buttons">${buttonsHtml}</div>
                </div>
            `;
        }
        return buttonsHtml;
    },

    /**
     * Initialize click handlers for module buttons
     */
    initModuleButtons(containerElement) {
        if (!containerElement) return;
        
        const self = this;
        containerElement.querySelectorAll('.module-action-btn').forEach(btn => {
            btn.addEventListener('click', async function handleModuleClick(e) {
                const module = btn.dataset.module;
                const query = decodeURIComponent(btn.dataset.query || '');
                const originalText = btn.textContent;
                
                // Disable all module buttons while processing
                containerElement.querySelectorAll('.module-action-btn').forEach(b => b.disabled = true);
                btn.classList.add('loading');
                btn.textContent = 'Loading...';
                
                try {
                    await self.executeModule(module, query);
                } catch (error) {
                    console.error('[InChatModules] Error executing ' + module + ':', error);
                    self.addModuleError(module, error.message);
                } finally {
                    containerElement.querySelectorAll('.module-action-btn').forEach(b => b.disabled = false);
                    btn.classList.remove('loading');
                    btn.textContent = originalText;
                }
            });
        });
    },
    
    /**
     * Execute a specific module
     */
    async executeModule(module, query) {
        console.log('[InChatModules] Executing module:', module, 'with query:', query);
        
        switch (module) {
            case 'treatment-comparison':
                await this.executeTreatmentComparison(query);
                break;
            case 'patient-matching':
                await this.executePatientMatching(query);
                break;
            case 'study-comparison':
                await this.executeStudyComparison(query);
                break;
            case 'analytics':
                await this.executeAnalytics(query);
                break;
            default:
                throw new Error('Unknown module: ' + module);
        }
    },

    /**
     * Add a loading message to chat
     */
    addLoadingMessage(title) {
        const messageId = 'module-' + Date.now();
        const messageDiv = document.createElement('div');
        messageDiv.id = messageId;
        messageDiv.className = 'message ai module-result';
        messageDiv.innerHTML = 
            '<div class="message-avatar" style="background: var(--gray-200); color: var(--gray-700);"><img src="assets/paxis-mark.png" alt="Paxis" style="width:22px;height:20px;object-fit:contain;"></div>' +
            '<div class="message-content">' +
                '<div style="font-weight: 600; margin-bottom: 0.5rem; color: var(--gray-900);">' + escapeHtml(title) + '</div>' +
                '<div class="loading"><div class="spinner"></div><span>Processing...</span></div>' +
            '</div>';
        
        const chatMessages = document.getElementById('chatMessages');
        if (chatMessages) {
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        return messageId;
    },
    
    /**
     * Update a message with content
     */
    updateMessage(messageId, title, contentHtml) {
        const messageDiv = document.getElementById(messageId);
        if (!messageDiv) return;
        
        messageDiv.innerHTML = 
            '<div class="message-avatar" style="background: var(--gray-200); color: var(--gray-700);"><img src="assets/paxis-mark.png" alt="Paxis" style="width:22px;height:20px;object-fit:contain;"></div>' +
            '<div class="message-content">' +
                '<div style="font-weight: 600; margin-bottom: 0.5rem; color: var(--gray-900);">' + escapeHtml(title) + '</div>' +
                '<div class="module-result-content">' + contentHtml + '</div>' +
            '</div>';
        
        const chatMessages = document.getElementById('chatMessages');
        if (chatMessages) {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    },
    
    /**
     * Add error message
     */
    addModuleError(module, errorMessage) {
        const titles = {
            'treatment-comparison': 'Treatment Comparison',
            'patient-matching': 'Match Patient to Trials',
            'study-comparison': 'Review and Compare Studies',
            'analytics': 'Advanced Analytics'
        };
        
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message ai module-result';
        messageDiv.innerHTML = 
            '<div class="message-avatar" style="background: var(--gray-200); color: var(--gray-700);"><img src="assets/paxis-mark.png" alt="Paxis" style="width:22px;height:20px;object-fit:contain;"></div>' +
            '<div class="message-content">' +
                '<div style="font-weight: 600; margin-bottom: 0.5rem; color: var(--gray-900);">' + (titles[module] || module) + '</div>' +
                '<div class="alert alert-danger">Failed to execute: ' + escapeHtml(errorMessage) + '</div>' +
            '</div>';
        
        const chatMessages = document.getElementById('chatMessages');
        if (chatMessages) {
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    },

    /**
     * Check if query explicitly requests a comparison
     * @private
     * @param {string} query - Query string to check
     * @returns {boolean} True if query contains comparison keywords
     */
    _isComparisonQuery(query) {
        if (!query) return false;
        const lowerQuery = query.toLowerCase();
        return lowerQuery.includes('vs') || 
               lowerQuery.includes('versus') || 
               lowerQuery.includes('compare') || 
               lowerQuery.includes('comparison') ||
               lowerQuery.includes('compared to') ||
               lowerQuery.includes('difference between');
    },

    /**
     * Execute Treatment Comparison module
     * Now uses the TreatmentEvaluation guided workflow
     */
    async executeTreatmentComparison(query) {
        const messageId = this.addLoadingMessage('Treatment Evaluation');
        
        // Store context for the treatment evaluation flow, including clinical context
        if (typeof TreatmentEvaluation !== 'undefined') {
            TreatmentEvaluation.setContext(query, this.lastQueryContext, this.clinicalContext);
            TreatmentEvaluation.showEvaluationPrompt(messageId);
            return;
        }
        
        // Fallback to direct comparison if TreatmentEvaluation not loaded
        try {
            const api = new PaxisAPI();
            
            // Skip visual comparison for simple questions without explicit comparison intent
            if (!this._isComparisonQuery(query)) {
                console.log('[InChatModules] Skipping visual comparison - no comparison keywords in query');
                const searchResult = await api.searchStudiesWithQuery(query, 15);
                const studies = searchResult.studies || [];
                
                let html = '';
                if (studies.length > 0) {
                    html += '<div class="module-section">' +
                        '<h4>Relevant Studies (' + studies.length + ' found)</h4>' +
                        '<div class="module-sources">' + formatSources(studies.slice(0, 10)) + '</div>' +
                    '</div>';
                } else {
                    html = '<div class="alert alert-info">No relevant studies found for this query.</div>';
                }
                
                this.updateMessage(messageId, 'Treatment Evaluation', html);
                return;
            }
            
            const result = await api.visualComparison(query, 15);
            
            let html = '';
            
            // Summary section
            const summaryText = result.summary || result.short_answer || '';
            if (summaryText) {
                html += '<div class="module-section">' +
                    '<h4>Summary</h4>' +
                    '<div class="module-summary">' + markdownToHtml(summaryText) + '</div>' +
                '</div>';
            }
            
            // Charts section
            if (result.charts && result.charts.length > 0) {
                html += '<div class="module-section">' +
                    '<h4>Visual Comparison</h4>' +
                    '<div class="module-charts" id="charts-' + messageId + '"></div>' +
                '</div>';
            }
            
            // Detailed analysis
            const analysis = result.detailed_analysis || result.justification || '';
            if (analysis) {
                html += '<div class="module-section">' +
                    '<h4>Detailed Analysis</h4>' +
                    '<div class="module-analysis">' + markdownToHtml(analysis) + '</div>' +
                '</div>';
            }
            
            // Sources
            const sources = result.retrieval_results || [];
            if (sources.length > 0) {
                html += '<div class="module-section">' +
                    '<h4>Supporting Evidence (' + sources.length + ' sources)</h4>' +
                    '<div class="module-sources">' + formatSources(sources.slice(0, 10)) + '</div>' +
                '</div>';
            }
            
            this.updateMessage(messageId, 'Treatment Comparison', html);
            
            // Render charts after DOM update
            if (result.charts && result.charts.length > 0) {
                setTimeout(() => {
                    this.renderCharts('charts-' + messageId, result.charts);
                }, 100);
            }
            
        } catch (error) {
            console.error('[InChatModules] Treatment comparison error:', error);
            this.updateMessage(messageId, 'Treatment Comparison', 
                '<div class="alert alert-danger">Error: ' + escapeHtml(error.message) + '</div>');
        }
    },

    /**
     * Execute Patient Matching module
     * Falls back to general study search if patient profile extraction fails
     */
    async executePatientMatching(query) {
        const messageId = this.addLoadingMessage('Finding Similar Studies');
        
        try {
            const api = new PaxisAPI();
            
            // Enrich query with clinical context
            let effectiveQuery = query;
            const clinicalContext = this.getClinicalContext();
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
                    console.log('[InChatModules] Enriched patient matching query with clinical context:', effectiveQuery);
                }
            }
            
            console.log('[InChatModules] Calling matchPatientUnstructured with query:', effectiveQuery);
            
            let result;
            let usedFallback = false;
            
            try {
                result = await api.matchPatientUnstructured(effectiveQuery, 15);
                console.log('[InChatModules] Patient matching result:', result);
            } catch (matchError) {
                // Check if this is the "could not extract patient characteristics" error
                const errorMsg = matchError.message || '';
                if (errorMsg.includes('Could not extract patient characteristics') || 
                    errorMsg.includes('patient characteristics') ||
                    errorMsg.includes('cancer type, stage')) {
                    console.log('[InChatModules] Patient extraction failed, falling back to study search');
                    usedFallback = true;
                    
                    // Fall back to general study search
                    const searchResult = await api.searchStudiesWithQuery(effectiveQuery, 15);
                    console.log('[InChatModules] Fallback study search result:', searchResult);
                    
                    // Convert search result to match format
                    const studies = searchResult.studies || [];
                    result = {
                        patient_summary: null,
                        extracted_profile: null,
                        matches: studies.map(s => ({
                            doc_id: s.doc_id,
                            title: s.title || s.study_name || 'Unknown Study',
                            author: s.author || s.first_author || '',
                            year: s.year || s.publication_year || '',
                            doi: s.doi || '',
                            pmid: s.pmid || '',
                            match_score: s.score || s.relevance_score || 0.5,
                            match_reasons: s.cancer_type ? [s.cancer_type] : []
                        }))
                    };
                } else {
                    // Re-throw other errors
                    throw matchError;
                }
            }
            
            let html = '';
            
            // Show fallback notice if we used the fallback
            if (usedFallback) {
                html += '<div class="alert alert-info" style="margin-bottom: 1rem;">' +
                    'Showing relevant studies based on your query. For more precise patient-to-trial matching, ' +
                    'include specific clinical details like cancer type, stage, or biomarkers.' +
                '</div>';
            }
            
            // Patient summary
            if (result.patient_summary) {
                html += '<div class="module-section">' +
                    '<h4>Patient Summary</h4>' +
                    '<div class="module-summary">' + escapeHtml(result.patient_summary) + '</div>' +
                '</div>';
            }
            
            // Extracted profile (collapsible)
            if (result.extracted_profile) {
                html += '<div class="module-section">' +
                    '<details class="extracted-profile-details">' +
                    '<summary>View Extracted Profile</summary>' +
                    '<div class="extracted-profile-content">' + 
                        this.renderExtractedProfile(result.extracted_profile) + 
                    '</div>' +
                    '</details>' +
                '</div>';
            }
            
            // Matching studies
            const matches = result.matches || [];
            if (matches.length > 0) {
                html += '<div class="module-section">' +
                    '<h4>' + (usedFallback ? 'Related Studies' : 'Matching Studies') + ' (' + matches.length + ' found)</h4>' +
                    '<div class="module-matches">' + this.renderPatientMatches(matches) + '</div>' +
                '</div>';
            } else {
                html += '<div class="alert alert-info">No matching studies found. Try adding more details about the patient (cancer type, stage, biomarkers).</div>';
            }
            
            // Continue your research section
            html += '<div class="continue-research-section">' +
                '<p class="continue-research-label">Continue your research</p>' +
                '<div class="continue-research-buttons">' +
                    '<button type="button" class="btn btn-outline btn-sm continue-btn" data-action="treatment-eval">Evaluate Treatment Options</button>' +
                    '<button type="button" class="btn btn-outline btn-sm continue-btn" data-action="studies">Review Studies</button>' +
                    '<button type="button" class="btn btn-outline btn-sm continue-btn" data-action="analytics">Explore Advanced Analytics</button>' +
                '</div>' +
                '<div class="new-conversation-divider">' +
                    '<span>or</span>' +
                '</div>' +
                '<button type="button" class="btn btn-outline btn-sm new-conversation-btn">Start a New Conversation</button>' +
            '</div>';
            
            this.updateMessage(messageId, usedFallback ? 'Related Studies' : 'Similar Studies', html);
            
            // Initialize continue research buttons
            this.initContinueButtons(messageId, query);
            
            // Initialize add-to-review button handlers
            this.initAddToReviewButtons(messageId);
            
        } catch (error) {
            console.error('[InChatModules] Patient matching error:', error);
            this.updateMessage(messageId, 'Similar Studies', 
                '<div class="alert alert-danger">Error: ' + escapeHtml(error.message) + '</div>');
        }
    },
    
    /**
     * Render extracted patient profile
     */
    renderExtractedProfile(profile) {
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

    /**
     * Render patient matching results with action buttons
     */
    renderPatientMatches(matches) {
        if (!matches || matches.length === 0) return '';
        
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
            
            // Study Details button
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
            
            // Add to Review button
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
            
            // DOI link
            let doiLink = '';
            if (doi) {
                doiLink = '<a href="https://doi.org/' + doi + '" target="_blank" rel="noopener" ' +
                    'class="btn btn-outline btn-sm" style="font-size: 0.75rem;">DOI</a>';
            }
            
            return '<div class="match-item">' +
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

    /**
     * Execute Study Comparison module - Search and compare relevant studies
     */
    async executeStudyComparison(query) {
        const messageId = this.addLoadingMessage('Review and Compare Studies');
        
        try {
            const api = new PaxisAPI();
            console.log('[InChatModules] Searching studies for comparison with query:', query);
            
            // Search for relevant studies
            const searchResult = await api.searchStudiesWithQuery(query, 10);
            console.log('[InChatModules] Study search result:', searchResult);
            
            const studies = searchResult.studies || [];
            
            if (studies.length === 0) {
                this.updateMessage(messageId, 'Review and Compare Studies',
                    '<div class="alert alert-info">No relevant studies found for this query. Try adding more details about the cancer type, treatment, or outcomes you are interested in.</div>');
                return;
            }
            
            let html = '';
            let comparisonResult = null;
            
            // Query interpretation
            if (searchResult.classification) {
                const cls = searchResult.classification;
                let interpretationParts = [];
                if (cls.cancer_type) interpretationParts.push('Cancer: ' + cls.cancer_type);
                if (cls.stage) interpretationParts.push('Stage: ' + cls.stage);
                if (cls.treatment_modality) interpretationParts.push('Treatment: ' + cls.treatment_modality);
                
                if (interpretationParts.length > 0) {
                    html += '<div class="module-section">' +
                        '<div class="query-interpretation">' +
                            '<strong>Search criteria:</strong> ' + escapeHtml(interpretationParts.join(' | ')) +
                        '</div>' +
                    '</div>';
                }
            }
            
            // Studies found
            html += '<div class="module-section">' +
                '<h4>Relevant Studies (' + studies.length + ' found)</h4>' +
                '<div class="study-comparison-list">' + this.renderStudyComparisonList(studies) + '</div>' +
            '</div>';
            
            // If we have enough studies, show a comparison summary
            if (studies.length >= 2) {
                const studyIds = studies.slice(0, 4).map(s => s.doc_id).filter(id => id);
                
                if (studyIds.length >= 2) {
                    try {
                        comparisonResult = await api.compareStudies(studyIds);
                        console.log('[InChatModules] Comparison result:', comparisonResult);
                        
                        if (comparisonResult.narrative) {
                            html += '<div class="module-section">' +
                                '<h4>Comparison Summary</h4>' +
                                '<div class="comparison-narrative">' + markdownToHtml(comparisonResult.narrative) + '</div>' +
                            '</div>';
                        }
                        
                        // Render comparison categories
                        if (comparisonResult.categories && comparisonResult.categories.length > 0) {
                            html += '<div class="module-section">' +
                                '<h4>Side-by-Side Comparison</h4>' +
                                this.renderComparisonCategories(comparisonResult.categories, comparisonResult.studies) +
                            '</div>';
                        }
                    } catch (compErr) {
                        console.error('[InChatModules] Comparison API error:', compErr);
                        // Continue without comparison - just show the study list
                    }
                }
            }
            
            // Link to full comparison page
            html += '<div class="module-section" style="text-align: center; margin-top: 1rem;">' +
                '<a href="study-comparison.html" class="btn btn-outline btn-sm">Open Full Study Comparison</a>' +
            '</div>';
            
            this.updateMessage(messageId, 'Review and Compare Studies', html);
            
            // Render charts after DOM update
            if (comparisonResult && comparisonResult.categories && comparisonResult.categories.length > 0) {
                setTimeout(() => {
                    this.renderComparisonCharts(comparisonResult.categories);
                }, 100);
            }
            
        } catch (error) {
            console.error('[InChatModules] Study comparison error:', error);
            this.updateMessage(messageId, 'Review and Compare Studies',
                '<div class="alert alert-danger">Error: ' + escapeHtml(error.message) + '</div>');
        }
    },
    
    /**
     * Render study list for comparison
     */
    renderStudyComparisonList(studies) {
        if (!studies || studies.length === 0) return '';
        
        return studies.slice(0, 10).map((study, index) => {
            const title = study.title || study.study_name || 'Unknown Study';
            const author = study.author || study.first_author || '';
            const year = study.year || study.publication_year || '';
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
            
            return '<div class="study-comparison-item">' +
                '<div class="study-comparison-header">' +
                    '<div class="study-comparison-title">' + (index + 1) + '. ' + escapeHtml(title) + '</div>' +
                '</div>' +
                '<div class="study-comparison-meta">' + metaParts.join(' | ') + '</div>' +
                '<div class="study-comparison-actions">' + studyBtn + '</div>' +
            '</div>';
        }).join('');
    },
    
    /**
     * Render comparison categories with charts
     */
    renderComparisonCategories(categories, studies) {
        if (!categories || categories.length === 0) return '';
        
        const self = this;
        let html = '<div class="comparison-categories">';
        let chartIndex = 0;
        
        categories.forEach(cat => {
            html += '<div class="comparison-category">';
            html += '<div class="category-name">' + escapeHtml(cat.title || cat.category || '') + '</div>';
            
            if (!cat.data_available || !cat.charts || cat.charts.length === 0) {
                // No data available - show message
                html += '<div class="no-data-message">' + escapeHtml(cat.summary || 'No data available for this category.') + '</div>';
            } else {
                // Render charts container
                html += '<div class="category-charts-grid" id="category-charts-' + cat.category + '">';
                
                cat.charts.forEach(artifact => {
                    if (artifact.artifact_type !== 'chart' || !artifact.chart) return;
                    
                    const chartId = 'inchat-chart-' + chartIndex;
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
            
            // Add summary if available
            if (cat.summary && cat.data_available) {
                html += '<div class="category-summary">' + escapeHtml(cat.summary) + '</div>';
            }
            
            html += '</div>';
        });
        
        html += '</div>';
        return html;
    },
    
    /**
     * Render charts for comparison categories after DOM is updated
     */
    renderComparisonCharts(categories) {
        if (!categories || categories.length === 0) return;
        
        let chartIndex = 0;
        const self = this;
        
        categories.forEach(cat => {
            if (!cat.data_available || !cat.charts || cat.charts.length === 0) return;
            
            cat.charts.forEach(artifact => {
                if (artifact.artifact_type !== 'chart' || !artifact.chart) return;
                
                const chartId = 'inchat-chart-' + chartIndex;
                const canvas = document.getElementById(chartId);
                
                if (canvas && typeof Chart !== 'undefined') {
                    try {
                        const ctx = canvas.getContext('2d');
                        const chartConfig = artifact.chart;
                        
                        // Default colors
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
                                    },
                                    tooltip: {
                                        callbacks: {
                                            label: function(context) {
                                                let label = context.dataset.label || '';
                                                if (label) label += ': ';
                                                label += context.parsed.y || context.parsed;
                                                if (chartConfig.unit) label += chartConfig.unit;
                                                return label;
                                            }
                                        }
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
                        console.error('[InChatModules] Chart render error:', error);
                    }
                }
                
                chartIndex++;
            });
        });
    },

    /**
     * Execute Trial Finder module (ClinicalTrials.gov search)
     */
    async executeTrialFinder(query) {
        const messageId = this.addLoadingMessage('Clinical Trial Search');
        
        try {
            const api = new PaxisAPI();
            console.log('[InChatModules] Searching clinical trials with query:', query);
            
            const payload = {
                query: query,
                patient_profile: null,
                recruiting_status: ['Recruiting', 'Not yet recruiting'],
                phase: null,
                location: null,
                max_results: 15
            };
            
            const result = await api.searchClinicalTrials(payload);
            console.log('[InChatModules] Trial search result:', result);
            
            let html = '';
            
            // Extracted profile
            if (result.extracted_profile) {
                html += '<div class="module-section">' +
                    '<details class="extracted-profile-details">' +
                    '<summary>View Parsed Profile</summary>' +
                    '<div class="extracted-profile-content">' + 
                        this.renderExtractedProfile(result.extracted_profile) + 
                    '</div>' +
                    '</details>' +
                '</div>';
            }
            
            // Trial results
            const trials = result.trials || [];
            if (trials.length > 0) {
                html += '<div class="module-section">' +
                    '<h4>Clinical Trials Found (' + trials.length + ')</h4>' +
                    '<div class="module-trials">' + this.renderTrialResults(trials) + '</div>' +
                '</div>';
            } else {
                html += '<div class="alert alert-info">No matching clinical trials found. Try adding more patient details (cancer type, stage, biomarkers) or adjusting the search criteria.</div>';
            }
            
            // Link to full trial search page
            html += '<div class="module-section" style="text-align: center; margin-top: 1rem;">' +
                '<a href="trial-search.html" class="btn btn-outline btn-sm">Open Full Trial Finder</a>' +
            '</div>';
            
            this.updateMessage(messageId, 'Clinical Trial Search', html);
            
        } catch (error) {
            console.error('[InChatModules] Trial finder error:', error);
            this.updateMessage(messageId, 'Clinical Trial Search', 
                '<div class="alert alert-danger">Error: ' + escapeHtml(error.message) + '</div>');
        }
    },
    
    /**
     * Render clinical trial results
     */
    renderTrialResults(trials) {
        if (!trials || trials.length === 0) return '';
        
        return trials.slice(0, 10).map((trial, index) => {
            const scorePct = Math.round((trial.match_score || 0) * 100);
            const title = trial.title || 'Untitled Trial';
            const nctId = trial.nct_id || 'N/A';
            const status = trial.status || 'Unknown';
            const phase = (trial.phase && trial.phase.length) ? trial.phase.join(', ') : 'N/A';
            const conditions = (trial.conditions || []).slice(0, 2).join(', ') || 'N/A';
            const interventions = (trial.interventions || []).slice(0, 2).join(', ') || 'N/A';
            const url = trial.url || '';
            
            let viewBtn = '';
            if (url) {
                viewBtn = '<a class="btn btn-outline btn-sm" href="' + escapeHtml(url) + '" target="_blank" rel="noopener">View on ClinicalTrials.gov</a>';
            }
            
            return '<div class="trial-item">' +
                '<div class="trial-header">' +
                    '<div class="trial-title">' + (index + 1) + '. ' + escapeHtml(title) + '</div>' +
                    (scorePct > 0 ? '<div class="trial-score">' + scorePct + '% match</div>' : '') +
                '</div>' +
                '<div class="trial-meta">' +
                    'NCT: ' + escapeHtml(nctId) + ' | Status: ' + escapeHtml(status) + ' | Phase: ' + escapeHtml(phase) +
                '</div>' +
                '<div class="trial-details">' +
                    '<div><strong>Conditions:</strong> ' + escapeHtml(conditions) + '</div>' +
                    '<div><strong>Interventions:</strong> ' + escapeHtml(interventions) + '</div>' +
                '</div>' +
                (trial.summary ? '<div class="trial-summary">' + escapeHtml(trial.summary.substring(0, 300)) + (trial.summary.length > 300 ? '...' : '') + '</div>' : '') +
                '<div class="trial-actions">' + viewBtn + '</div>' +
            '</div>';
        }).join('');
    },

    /**
     * Execute Analytics module - Full dashboard with multiple charts
     */
    async executeAnalytics(query) {
        const messageId = this.addLoadingMessage('Knowledge Base Analytics');
        
        try {
            const api = new PaxisAPI();
            const base = api.baseUrl.replace('/rag', '');
            
            // Extract cancer type from query for filtering
            let cancerType = null;
            const cancerTypes = ['lung', 'breast', 'prostate', 'colorectal', 'pancreatic', 'ovarian', 'brain', 'head', 'neck', 'esophageal', 'gastric', 'liver', 'cervical', 'bladder', 'kidney', 'melanoma', 'lymphoma', 'leukemia', 'nsclc', 'sclc'];
            const queryLower = query.toLowerCase();
            for (const ct of cancerTypes) {
                if (queryLower.includes(ct)) {
                    cancerType = ct;
                    break;
                }
            }
            
            console.log('[InChatModules] Analytics for cancer type:', cancerType || 'all');
            
            // Fetch overview stats
            const overviewResp = await fetch(base + '/analytics/overview');
            if (!overviewResp.ok) throw new Error('Failed to fetch analytics overview');
            const overview = await overviewResp.json();
            
            let html = '';
            const chartContainers = [];
            
            // Overview stats
            html += '<div class="module-section">' +
                '<h4>Knowledge Base Overview' + (cancerType ? ' (filtered: ' + cancerType + ')' : '') + '</h4>' +
                '<div class="analytics-stats-grid">' +
                    '<div class="stat-item"><div class="stat-value">' + (overview.total_studies || '-') + '</div><div class="stat-label">Studies</div></div>' +
                    '<div class="stat-item"><div class="stat-value">' + (overview.cancer_types || '-') + '</div><div class="stat-label">Cancer Types</div></div>' +
                    '<div class="stat-item"><div class="stat-value">' + (overview.avg_os_percent != null ? overview.avg_os_percent + '%' : '-') + '</div><div class="stat-label">Avg OS Rate</div></div>' +
                    '<div class="stat-item"><div class="stat-value">' + (overview.avg_followup_months != null ? overview.avg_followup_months : '-') + '</div><div class="stat-label">Avg Follow-up (mo)</div></div>' +
                    '<div class="stat-item"><div class="stat-value">' + (overview.avg_dose_gy != null ? overview.avg_dose_gy : '-') + '</div><div class="stat-label">Avg Dose (Gy)</div></div>' +
                    '<div class="stat-item"><div class="stat-value">' + (overview.unique_techniques || '-') + '</div><div class="stat-label">RT Techniques</div></div>' +
                '</div>' +
            '</div>';
            
            // Chart 1: OS Rate by Cancer Type or Study Phase
            const chartId1 = 'chart1-' + messageId;
            html += '<div class="module-section">' +
                '<h4>Average OS Rate by ' + (cancerType ? 'Study Phase' : 'Cancer Type') + '</h4>' +
                '<div class="analytics-chart-container" id="' + chartId1 + '"></div>' +
            '</div>';
            chartContainers.push({
                id: chartId1,
                endpoint: '/analytics/aggregate',
                body: {
                    metric: 'os_rate_percent',
                    group_by: cancerType ? 'normalized_phase' : 'cancer_type',
                    agg: 'avg',
                    filters: cancerType ? { cancer_type: cancerType } : null
                }
            });
            
            // Chart 2: Dose Distribution
            const chartId2 = 'chart2-' + messageId;
            html += '<div class="module-section">' +
                '<h4>Radiation Dose Distribution (Gy)</h4>' +
                '<div class="analytics-chart-container" id="' + chartId2 + '"></div>' +
            '</div>';
            chartContainers.push({
                id: chartId2,
                endpoint: '/analytics/dose-distribution',
                body: { cancer_type: cancerType, bin_width: 5 }
            });
            
            // Chart 3: Technique Frequency
            const chartId3 = 'chart3-' + messageId;
            html += '<div class="module-section">' +
                '<h4>RT Technique Frequency</h4>' +
                '<div class="analytics-chart-container" id="' + chartId3 + '"></div>' +
            '</div>';
            chartContainers.push({
                id: chartId3,
                endpoint: '/analytics/technique-frequency',
                body: { cancer_type: cancerType }
            });
            
            // Chart 4: Outcomes by Stage (if cancer type specified)
            if (cancerType) {
                const chartId4 = 'chart4-' + messageId;
                html += '<div class="module-section">' +
                    '<h4>Outcomes by Stage</h4>' +
                    '<div class="analytics-chart-container" id="' + chartId4 + '"></div>' +
                '</div>';
                chartContainers.push({
                    id: chartId4,
                    endpoint: '/analytics/outcomes-by-stage',
                    body: { metric: 'os_rate_percent', cancer_type: cancerType }
                });
            }
            
            // Link to full analytics page
            html += '<div class="module-section" style="text-align: center; margin-top: 1rem;">' +
                '<a href="analytics.html" class="btn btn-outline btn-sm">Open Full Analytics Dashboard</a>' +
            '</div>';
            
            this.updateMessage(messageId, 'Knowledge Base Analytics', html);
            
            // Render all charts
            const self = this;
            setTimeout(async function() {
                for (const chart of chartContainers) {
                    try {
                        const resp = await fetch(base + chart.endpoint, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(chart.body)
                        });
                        
                        if (resp.ok) {
                            const data = await resp.json();
                            const container = document.getElementById(chart.id);
                            if (container && data.labels && data.labels.length > 0) {
                                container.innerHTML = '<canvas id="canvas-' + chart.id + '"></canvas>';
                                self.renderSingleChart('canvas-' + chart.id, {
                                    type: 'bar',
                                    labels: data.labels.slice(0, 12),
                                    datasets: [{ label: data.unit || 'Value', data: data.values.slice(0, 12) }],
                                    unit: data.unit
                                });
                            } else if (container) {
                                container.innerHTML = '<div class="alert alert-info" style="font-size: 0.85rem;">No data available for this chart.</div>';
                            }
                        }
                    } catch (e) {
                        console.error('[InChatModules] Chart fetch error for ' + chart.id + ':', e);
                        const container = document.getElementById(chart.id);
                        if (container) {
                            container.innerHTML = '<div class="alert alert-warning" style="font-size: 0.85rem;">Could not load chart data.</div>';
                        }
                    }
                }
            }, 100);
            
        } catch (error) {
            console.error('[InChatModules] Analytics error:', error);
            this.updateMessage(messageId, 'Knowledge Base Analytics', 
                '<div class="alert alert-danger">Error: ' + escapeHtml(error.message) + '</div>');
        }
    },

    /**
     * Render charts in a container
     */
    renderCharts(containerId, charts) {
        const container = document.getElementById(containerId);
        if (!container || !charts || charts.length === 0) return;
        
        charts.forEach((artifact, i) => {
            if (artifact.artifact_type !== 'chart' || !artifact.chart) return;
            
            const chart = artifact.chart;
            const chartDiv = document.createElement('div');
            chartDiv.className = 'chart-item';
            chartDiv.innerHTML = 
                '<div class="chart-title">' + escapeHtml(chart.title) + '</div>' +
                '<div class="chart-canvas-wrap"><canvas id="chart-' + containerId + '-' + i + '"></canvas></div>';
            container.appendChild(chartDiv);
            
            setTimeout(() => {
                this.renderSingleChart('chart-' + containerId + '-' + i, chart);
            }, 50);
        });
    },
    
    /**
     * Render a single chart
     */
    renderSingleChart(canvasId, chartConfig) {
        const canvas = document.getElementById(canvasId);
        if (!canvas || typeof Chart === 'undefined') {
            console.error('[InChatModules] Canvas not found or Chart.js not loaded');
            return;
        }
        
        try {
            const ctx = canvas.getContext('2d');
            const config = {
                type: chartConfig.type || 'bar',
                data: {
                    labels: chartConfig.labels || [],
                    datasets: (chartConfig.datasets || []).map((ds, i) => ({
                        label: ds.label || 'Dataset ' + (i + 1),
                        data: ds.data || [],
                        backgroundColor: ds.backgroundColor || [
                            'rgba(59, 130, 246, 0.8)',
                            'rgba(16, 185, 129, 0.8)',
                            'rgba(245, 158, 11, 0.8)',
                            'rgba(239, 68, 68, 0.8)'
                        ],
                        borderWidth: 1
                    }))
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: { legend: { display: chartConfig.datasets && chartConfig.datasets.length > 1 } }
                }
            };
            
            const chartInstance = new Chart(ctx, config);
            this.chartInstances.push(chartInstance);
        } catch (error) {
            console.error('[InChatModules] Chart render error:', error);
        }
    },
    
    /**
     * Initialize continue research button handlers
     */
    initContinueButtons(messageId, query) {
        const container = document.getElementById(messageId);
        if (!container) return;
        
        const self = this;
        
        // Try to get query from multiple sources if not provided
        let effectiveQuery = query;
        if (!effectiveQuery || effectiveQuery.length < 10) {
            // Try to get from lastQueryContext
            if (self.lastQueryContext && self.lastQueryContext.query) {
                effectiveQuery = self.lastQueryContext.query;
            }
            // Try to get from sessionStorage
            if (!effectiveQuery || effectiveQuery.length < 10) {
                try {
                    const stored = sessionStorage.getItem('treatmentEvalContext');
                    if (stored) {
                        const parsed = JSON.parse(stored);
                        if (parsed.query && parsed.query.length >= 10) {
                            effectiveQuery = parsed.query;
                        }
                    }
                } catch (e) {
                    console.warn('[InChatModules] Failed to parse stored context:', e);
                }
            }
            // Try followupContext
            if (!effectiveQuery || effectiveQuery.length < 10) {
                try {
                    const followup = sessionStorage.getItem('followupContext');
                    if (followup) {
                        const parsed = JSON.parse(followup);
                        if (parsed.query && parsed.query.length >= 10) {
                            effectiveQuery = parsed.query;
                        }
                    }
                } catch (e) {
                    console.warn('[InChatModules] Failed to parse followup context:', e);
                }
            }
        }
        
        container.querySelectorAll('.continue-btn').forEach(btn => {
            btn.addEventListener('click', function handleContinueClick() {
                const action = btn.dataset.action;
                
                // Store context for the target page
                if (effectiveQuery && effectiveQuery.length >= 10) {
                    sessionStorage.setItem('followupContext', JSON.stringify({
                        query: effectiveQuery,
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
                    case 'treatment-eval':
                        // Execute treatment comparison in chat
                        if (effectiveQuery && effectiveQuery.length >= 10) {
                            self.executeTreatmentComparison(effectiveQuery);
                        } else {
                            self.showTreatmentContextPrompt();
                        }
                        break;
                    case 'studies':
                        // Execute study comparison in chat
                        if (effectiveQuery && effectiveQuery.length >= 10) {
                            self.executeStudyComparison(effectiveQuery);
                        } else {
                            self.showStudyReviewForm(effectiveQuery);
                        }
                        break;
                    case 'patient-match':
                        // Execute patient matching in chat
                        if (effectiveQuery && effectiveQuery.length >= 10) {
                            self.executePatientMatching(effectiveQuery);
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
                self.showExitConversationModal();
            });
        }
    },
    
    /**
     * Show prompt for treatment context when no query context is available
     */
    showTreatmentContextPrompt() {
        const messageId = this.addLoadingMessage('Treatment Evaluation');
        const self = this;
        
        const html = `
            <div class="treatment-context-prompt">
                <p>Please describe the clinical context for treatment evaluation:</p>
                <div class="form-group">
                    <textarea id="treatment-context-input-${messageId}" 
                              class="form-input" 
                              rows="3" 
                              placeholder="e.g., Stage IIIA NSCLC with EGFR mutation, or metastatic breast cancer HER2+"></textarea>
                </div>
                <div class="validation-message" id="context-validation-${messageId}" style="display: none; color: var(--danger); font-size: 0.875rem; margin-bottom: 0.5rem;"></div>
                <button type="button" class="btn btn-accent" id="submit-context-${messageId}">
                    Continue to Treatment Evaluation
                </button>
            </div>
        `;
        
        this.updateMessage(messageId, 'Treatment Evaluation', html);
        
        // Initialize submit handler
        const submitBtn = document.getElementById('submit-context-' + messageId);
        const inputField = document.getElementById('treatment-context-input-' + messageId);
        const validationMsg = document.getElementById('context-validation-' + messageId);
        
        if (submitBtn && inputField) {
            submitBtn.addEventListener('click', function handleSubmitContext() {
                const contextText = inputField.value.trim();
                
                if (contextText.length < 10) {
                    if (validationMsg) {
                        validationMsg.textContent = 'Please provide more details (at least 10 characters).';
                        validationMsg.style.display = 'block';
                    }
                    return;
                }
                
                // Set context and show evaluation prompt
                if (typeof TreatmentEvaluation !== 'undefined') {
                    TreatmentEvaluation.setContext(contextText, null);
                    TreatmentEvaluation.showEvaluationPrompt(messageId);
                }
            });
            
            // Allow Enter key to submit
            inputField.addEventListener('keydown', function handleKeyDown(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    submitBtn.click();
                }
            });
        }
    },
    
    /**
     * Show exit conversation modal with report options
     */
    showExitConversationModal() {
        // Remove existing modal if any
        const existing = document.getElementById('exitConversationModal');
        if (existing) existing.remove();

        const overlay = document.createElement('div');
        overlay.id = 'exitConversationModal';
        overlay.className = 'exit-conversation-overlay';
        overlay.innerHTML = `
            <div class="exit-conversation-modal">
                <div class="exit-modal-header">
                    <p>Before starting a new conversation, would you like to save your research?</p>
                </div>
                <div class="exit-modal-buttons">
                    <button type="button" class="btn btn-accent btn-sm" id="saveToMySavesBtn">
                        Save to My Collections
                    </button>
                    <button type="button" class="btn btn-outline btn-sm" id="saveTranscriptBtn">
                        Save Conversation as PDF
                    </button>
                    <button type="button" class="btn btn-outline btn-sm" id="generateReportBtn">
                        Generate Report
                    </button>
                    <button type="button" class="btn btn-outline btn-sm" id="continueNewConvBtn">
                        Skip and Start New Conversation
                    </button>
                </div>
            </div>
        `;

        document.body.appendChild(overlay);

        const self = this;
        
        // Save to My Collections button
        document.getElementById('saveToMySavesBtn').addEventListener('click', async function handleSaveToMySaves() {
            overlay.querySelector('.exit-modal-buttons').innerHTML = '<div class="loading"><div class="spinner"></div><span>Saving to My Collections...</span></div>';
            try {
                await self.saveConversationToMySaves();
                overlay.querySelector('.exit-modal-buttons').innerHTML = '<div class="alert alert-success" style="margin: 0;">Saved successfully!</div>';
                setTimeout(function() {
                    overlay.remove();
                    self.startNewConversation();
                }, 1500);
            } catch (error) {
                console.error('[InChatModules] Save to My Collections error:', error);
                overlay.querySelector('.exit-modal-buttons').innerHTML = '<div class="alert alert-danger" style="margin: 0;">Save failed: ' + escapeHtml(error.message) + '</div>';
                setTimeout(function() {
                    overlay.remove();
                }, 2000);
            }
        });
        
        // Generate Report button
        document.getElementById('generateReportBtn').addEventListener('click', async function handleGenerateReport() {
            overlay.querySelector('.exit-modal-buttons').innerHTML = '<div class="loading"><div class="spinner"></div><span>Generating report...</span></div>';
            try {
                await self.generateSynthesizedReport();
            } catch (error) {
                console.error('[InChatModules] Report generation error:', error);
            }
            overlay.remove();
            self.startNewConversation();
        });

        // Save Transcript button
        document.getElementById('saveTranscriptBtn').addEventListener('click', function handleSaveTranscript() {
            overlay.querySelector('.exit-modal-buttons').innerHTML = '<div class="loading"><div class="spinner"></div><span>Saving transcript...</span></div>';
            try {
                self.saveConversationAsPdf();
            } catch (error) {
                console.error('[InChatModules] PDF save error:', error);
            }
            overlay.remove();
            self.startNewConversation();
        });

        // Continue to New Conversation button
        document.getElementById('continueNewConvBtn').addEventListener('click', function handleContinueNew() {
            overlay.remove();
            self.startNewConversation();
        });

        // Close on overlay click
        overlay.addEventListener('click', function handleOverlayClick(e) {
            if (e.target === overlay) {
                overlay.remove();
            }
        });
    },
    
    /**
     * Save conversation to My Collections
     */
    async saveConversationToMySaves() {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) {
            throw new Error('No conversation to save');
        }

        // Collect conversation content
        const messages = chatMessages.querySelectorAll('.message');
        let questionText = '';
        let answerText = '';
        const sources = [];
        
        messages.forEach(function collectMessage(msg) {
            const isUser = msg.classList.contains('user');
            const content = msg.querySelector('.message-content');
            if (content) {
                if (isUser) {
                    if (!questionText) {
                        questionText = content.textContent.trim();
                    }
                } else {
                    answerText += content.textContent.trim() + '\n\n';
                }
            }
        });

        if (!questionText) {
            throw new Error('No query found in conversation');
        }

        // Build save data
        const saveData = {
            short_answer: answerText.substring(0, 500),
            justification: answerText,
            retrieval_results: sources,
            metadata: {
                saved_from: 'conversation_exit',
                timestamp: new Date().toISOString()
            }
        };

        const api = new PaxisAPI();
        const result = await api.saveCase(questionText, null, saveData);
        console.log('[InChatModules] Saved to My Collections:', result);
        return result;
    },

    /**
     * Generate synthesized report from conversation
     */
    async generateSynthesizedReport() {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return;

        // Collect conversation content for the report
        const messages = chatMessages.querySelectorAll('.message');
        let questionText = '';
        let answerText = '';
        const sources = [];
        
        messages.forEach(msg => {
            const isUser = msg.classList.contains('user');
            const content = msg.querySelector('.message-content');
            if (content) {
                if (isUser) {
                    // Use the first user message as the main question
                    if (!questionText) {
                        questionText = content.textContent.trim();
                    }
                } else {
                    // Concatenate all assistant responses
                    answerText += content.textContent.trim() + '\n\n';
                }
            }
        });

        if (!questionText && !answerText) {
            console.log('[InChatModules] No conversation content to generate report from');
            return;
        }

        // Build payload for the query report endpoint
        const payload = {
            question: questionText || 'Research Session',
            short_answer: answerText.substring(0, 500),
            justification: answerText,
            retrieval_results: sources,
            format: 'standard'
        };

        try {
            const api = new PaxisAPI();
            await api.downloadReport('query', payload, 'case_research_report.pdf');
            console.log('[InChatModules] Report generated successfully');
        } catch (error) {
            console.error('[InChatModules] Report generation failed:', error);
            // Fallback: save conversation as PDF
            this.saveConversationAsPdf();
        }
    },

    /**
     * Save conversation as PDF transcript
     */
    saveConversationAsPdf() {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return;

        // Collect conversation content
        const messages = chatMessages.querySelectorAll('.message');
        let transcriptContent = 'Paxis Research Conversation Transcript\n';
        transcriptContent += '=' .repeat(50) + '\n';
        transcriptContent += 'Date: ' + new Date().toLocaleString() + '\n\n';

        messages.forEach((msg, index) => {
            const isUser = msg.classList.contains('user');
            const content = msg.querySelector('.message-content');
            if (content) {
                const speaker = isUser ? 'You' : 'Paxis';
                transcriptContent += `[${speaker}]\n`;
                transcriptContent += content.textContent.trim() + '\n\n';
                transcriptContent += '-'.repeat(40) + '\n\n';
            }
        });

        this.downloadTextAsPdf(transcriptContent, 'conversation_transcript.pdf');
    },

    /**
     * Download text content as PDF
     */
    downloadTextAsPdf(content, filename) {
        // Create a simple text-based PDF using browser print
        const printWindow = window.open('', '_blank');
        if (!printWindow) {
            // Fallback: download as text file
            const blob = new Blob([content], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename.replace('.pdf', '.txt');
            a.click();
            URL.revokeObjectURL(url);
            return;
        }

        printWindow.document.write(`
            <!DOCTYPE html>
            <html>
            <head>
                <title>${filename}</title>
                <style>
                    body { font-family: 'Nunito', Arial, sans-serif; padding: 40px; line-height: 1.6; }
                    h1 { color: #1e40af; border-bottom: 2px solid #1e40af; padding-bottom: 10px; }
                    pre { white-space: pre-wrap; word-wrap: break-word; }
                </style>
            </head>
            <body>
                <pre>${content.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</pre>
            </body>
            </html>
        `);
        printWindow.document.close();
        
        // Trigger print dialog (user can save as PDF)
        setTimeout(() => {
            printWindow.print();
        }, 250);
    },

    /**
     * Start a new conversation by clearing chat and resetting state
     */
    startNewConversation() {
        // Clear chat messages
        const chatMessages = document.getElementById('chatMessages');
        if (chatMessages) {
            chatMessages.innerHTML = '';
        }

        // Reset conversation history (if exists in global scope)
        if (typeof conversationHistory !== 'undefined') {
            conversationHistory = [];
        }

        // Reset last query context
        this.lastQueryContext = null;
        
        // Reset clinical context
        this.clinicalContext = null;

        // Clear all conversation-related session storage
        sessionStorage.removeItem('followupContext');
        sessionStorage.removeItem('treatmentEvalContext');
        sessionStorage.removeItem('treatmentEvalClinicalContext');
        
        // Clear TreatmentEvaluation context if it exists
        if (typeof TreatmentEvaluation !== 'undefined') {
            TreatmentEvaluation.queryContext = null;
            TreatmentEvaluation.clinicalContext = null;
            TreatmentEvaluation.currentMessageId = null;
        }
        
        // Clear API conversation context if available
        if (typeof api !== 'undefined' && api.clearConversationContext) {
            api.clearConversationContext();
        }

        // Reset input
        const messageInput = document.getElementById('messageInput');
        if (messageInput) {
            messageInput.value = '';
            messageInput.focus();
        }
        
        // Also check for chatInput (alternative input element name)
        const chatInput = document.getElementById('chatInput');
        if (chatInput) {
            chatInput.value = '';
            chatInput.focus();
        }

        // Show welcome message or reset UI
        const heroExamples = document.querySelector('.hero-examples');
        if (heroExamples) {
            heroExamples.style.display = 'block';
        }

        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });

        console.log('[InChatModules] Started new conversation - all context cleared');
    },
    
    /**
     * Show prompt for study search when no query context is available
     */
    showStudySearchPrompt(existingMessageId) {
        const messageId = existingMessageId || this.addLoadingMessage('Find Studies');
        const self = this;
        
        const html = `
            <div class="study-search-prompt">
                <p>What type of studies are you looking for?</p>
                <div class="form-group">
                    <textarea id="study-search-input-${messageId}" 
                              class="form-input" 
                              rows="2" 
                              placeholder="e.g., Stage III NSCLC immunotherapy trials, or HER2+ breast cancer treatment studies"></textarea>
                </div>
                <div class="validation-message" id="search-validation-${messageId}" style="display: none; color: var(--danger); font-size: 0.875rem; margin-bottom: 0.5rem;"></div>
                <button type="button" class="btn btn-accent" id="submit-search-${messageId}">
                    Search Studies
                </button>
            </div>
        `;
        
        this.updateMessage(messageId, 'Find Studies', html);
        
        // Initialize submit handler
        const submitBtn = document.getElementById('submit-search-' + messageId);
        const inputField = document.getElementById('study-search-input-' + messageId);
        const validationMsg = document.getElementById('search-validation-' + messageId);
        
        if (submitBtn && inputField) {
            submitBtn.addEventListener('click', async function handleSubmitSearch() {
                const searchText = inputField.value.trim();
                
                if (searchText.length < 10) {
                    if (validationMsg) {
                        validationMsg.textContent = 'Please provide more details (at least 10 characters).';
                        validationMsg.style.display = 'block';
                    }
                    return;
                }
                
                submitBtn.disabled = true;
                submitBtn.textContent = 'Searching...';
                
                // Store context and execute search
                self.setQueryContext(searchText, null);
                sessionStorage.setItem('followupContext', JSON.stringify({
                    query: searchText,
                    timestamp: Date.now()
                }));
                
                await self.executePatientMatching(searchText);
            });
            
            // Allow Enter key to submit
            inputField.addEventListener('keydown', function handleKeyDown(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    submitBtn.click();
                }
            });
        }
    },
    
    /**
     * Show patient matching prompt with option to set criteria
     */
    showPatientMatchingPrompt(query) {
        console.log('[InChatModules] showPatientMatchingPrompt called with query:', query);
        const messageId = this.addLoadingMessage('Patient Matching');
        const self = this;
        
        // Default criteria weights
        const defaultWeights = {
            cancer_type: 1.5,
            stage: 1.25,
            biomarkers: 1.0,
            histology: 1.0,
            age: 0.75,
            performance_status: 0.75
        };
        
        // Store weights for later use
        this._matchCriteriaWeights = this._matchCriteriaWeights || defaultWeights;
        
        let html = '<div class="patient-matching-prompt">';
        html += '<p>Would you like to adjust the match criteria before searching?</p>';
        html += '<div class="match-criteria-toggle">';
        html += '<button type="button" class="btn btn-outline btn-sm" id="show-criteria-' + messageId + '">Set Match Criteria</button>';
        html += '<button type="button" class="btn btn-accent btn-sm" id="run-match-default-' + messageId + '">Find Matching Studies</button>';
        html += '</div>';
        
        // Hidden criteria panel
        html += '<div class="match-criteria-panel" id="criteria-panel-' + messageId + '" style="display: none;">';
        html += '<p class="criteria-intro">Adjust how important each criterion is for matching:</p>';
        html += '<div class="criteria-sliders">';
        
        const criteriaLabels = {
            cancer_type: 'Cancer Type',
            stage: 'Stage',
            biomarkers: 'Biomarkers',
            histology: 'Histology',
            age: 'Age',
            performance_status: 'Performance Status (ECOG)'
        };
        
        Object.keys(this._matchCriteriaWeights).forEach(function renderSlider(key) {
            const label = criteriaLabels[key] || key;
            const value = self._matchCriteriaWeights[key];
            html += '<div class="criteria-slider-row">' +
                '<div class="criteria-slider-label">' + label + '</div>' +
                '<input type="range" class="criteria-slider-input" data-criteria="' + key + '" ' +
                    'min="0" max="2" step="0.25" value="' + value + '">' +
                '<span class="criteria-slider-value">' + value.toFixed(2) + 'x</span>' +
            '</div>';
        });
        
        html += '</div>';
        html += '<button type="button" class="btn btn-accent btn-sm" id="run-match-criteria-' + messageId + '" style="width: 100%; margin-top: 1rem;">Find Matching Studies</button>';
        html += '</div>';
        html += '</div>';
        
        this.updateMessage(messageId, 'Patient Matching', html);
        
        // Initialize handlers
        const showCriteriaBtn = document.getElementById('show-criteria-' + messageId);
        const runDefaultBtn = document.getElementById('run-match-default-' + messageId);
        const runCriteriaBtn = document.getElementById('run-match-criteria-' + messageId);
        const criteriaPanel = document.getElementById('criteria-panel-' + messageId);
        
        if (showCriteriaBtn && criteriaPanel) {
            showCriteriaBtn.addEventListener('click', function handleShowCriteria() {
                criteriaPanel.style.display = criteriaPanel.style.display === 'none' ? 'block' : 'none';
                showCriteriaBtn.textContent = criteriaPanel.style.display === 'none' ? 'Set Match Criteria' : 'Hide Criteria';
            });
        }
        
        // Initialize slider interactions
        const container = document.getElementById(messageId);
        if (container) {
            container.querySelectorAll('.criteria-slider-input').forEach(function initSlider(slider) {
                slider.addEventListener('input', function handleSliderChange() {
                    const val = parseFloat(this.value);
                    const key = this.dataset.criteria;
                    self._matchCriteriaWeights[key] = val;
                    this.nextElementSibling.textContent = val.toFixed(2) + 'x';
                });
            });
        }
        
        // Run with default weights
        if (runDefaultBtn) {
            runDefaultBtn.addEventListener('click', async function handleRunDefault() {
                runDefaultBtn.disabled = true;
                runDefaultBtn.textContent = 'Searching...';
                if (showCriteriaBtn) showCriteriaBtn.disabled = true;
                
                await self.executePatientMatching(query);
            });
        }
        
        // Run with custom criteria
        if (runCriteriaBtn) {
            runCriteriaBtn.addEventListener('click', async function handleRunCriteria() {
                runCriteriaBtn.disabled = true;
                runCriteriaBtn.textContent = 'Searching...';
                if (showCriteriaBtn) showCriteriaBtn.disabled = true;
                if (runDefaultBtn) runDefaultBtn.disabled = true;
                
                await self.executePatientMatching(query);
            });
        }
    },
    
    /**
     * Run patient matching and open results in SplitPanel
     */
    async runPatientMatchingWithPanel(query, messageId) {
        const self = this;
        
        // Update the chat message to show we're searching
        this.updateMessage(messageId, 'Patient Matching', 
            '<div class="loading"><div class="spinner"></div><span>Finding matching studies...</span></div>');
        
        try {
            // Store the criteria weights in SplitPanel if available
            if (typeof SplitPanel !== 'undefined') {
                SplitPanel._matchCriteriaWeights = this._matchCriteriaWeights;
            }
            
            // Store context for the page transfer
            sessionStorage.setItem('patientMatchContext', JSON.stringify({
                query: query,
                criteria: this._matchCriteriaWeights,
                timestamp: Date.now()
            }));
            
            // Open SplitPanel with results (skip the criteria setup since we already did it)
            if (typeof SplitPanel !== 'undefined') {
                // Initialize panel first
                SplitPanel.init();
                
                // Set up panel header
                const panel = document.getElementById('splitPanel');
                const title = document.getElementById('splitPanelTitle');
                const exploreLink = document.getElementById('splitPanelExploreMore');
                const chatContainer = document.querySelector('.chat-container');
                
                if (title) title.textContent = 'Patient Matching';
                if (exploreLink) {
                    exploreLink.href = 'patient-matching.html';
                    
                    // Set up context transfer for Explore More
                    const newLink = exploreLink.cloneNode(true);
                    exploreLink.parentNode.replaceChild(newLink, exploreLink);
                    
                    newLink.addEventListener('click', function handleExploreMore() {
                        sessionStorage.setItem('patientMatchContext', JSON.stringify({
                            query: query,
                            criteria: self._matchCriteriaWeights,
                            timestamp: Date.now()
                        }));
                        sessionStorage.setItem('followupContext', JSON.stringify({
                            query: query,
                            timestamp: Date.now()
                        }));
                        console.log('[InChatModules] Stored patient match context for Explore More');
                    });
                }
                
                // Open the panel
                if (panel) {
                    panel.classList.add('open');
                    panel.setAttribute('aria-hidden', 'false');
                    SplitPanel.isOpen = true;
                    SplitPanel.currentModule = 'patient-matching';
                    SplitPanel._lastQuery = query;
                }
                if (chatContainer && !SplitPanel._isMobile()) {
                    chatContainer.classList.add('split-active');
                }
                
                // Call the internal run function directly
                await SplitPanel._runPatientMatchingWithCriteria(query);
                
                // Update chat message to show completion
                this.updateMessage(messageId, 'Patient Matching', 
                    '<div class="module-complete">' +
                        '<p>Results are displayed in the side panel.</p>' +
                        '<button type="button" class="btn btn-outline btn-sm" onclick="SplitPanel.open(\'patient-matching\', \'' + escapeHtml(query).replace(/'/g, "\\'") + '\')">Reopen Panel</button>' +
                    '</div>');
            } else {
                // Fallback to inline execution
                await this.executePatientMatching(query);
            }
        } catch (error) {
            console.error('[InChatModules] Patient matching error:', error);
            this.updateMessage(messageId, 'Patient Matching', 
                '<div class="alert alert-danger">Error: ' + escapeHtml(error.message) + '</div>');
        }
    },
    
    /**
     * Show study review form with currently selected studies
     */
    showStudyReviewForm(query) {
        console.log('[InChatModules] showStudyReviewForm called with query:', query);
        const messageId = this.addLoadingMessage('Review Studies');
        
        // Get studies from comparison tray
        const trayStudies = typeof getComparisonTray === 'function' ? getComparisonTray() : [];
        
        let html = '<div class="study-review-form">';
        
        if (trayStudies.length > 0) {
            html += '<p>Your selected studies for review:</p>';
            html += '<div class="study-review-list" id="study-review-list-' + messageId + '">';
            trayStudies.forEach(function renderStudyItem(study, idx) {
                const title = study.title || 'Unknown Study';
                const year = study.year ? ' (' + study.year + ')' : '';
                const docId = study.doc_id || '';
                html += '<div class="study-review-item">' +
                    '<label class="study-review-checkbox">' +
                        '<input type="checkbox" checked data-doc-id="' + escapeHtml(docId) + '" data-title="' + escapeHtml(title) + '" data-year="' + escapeHtml(study.year || '') + '" data-doi="' + escapeHtml(study.doi || '') + '">' +
                        '<span class="study-review-title">' + escapeHtml(title) + year + '</span>' +
                    '</label>' +
                    '<button type="button" class="btn-remove-study" data-doc-id="' + escapeHtml(docId) + '" title="Remove from selection">x</button>' +
                '</div>';
            });
            html += '</div>';
        } else {
            html += '<div class="alert alert-info">No studies selected yet. Use the buttons below to find and add studies to your review.</div>';
        }
        
        html += '<div class="study-review-actions">';
        html += '<button type="button" class="btn btn-outline btn-sm" id="edit-selection-' + messageId + '">Edit Selection</button>';
        html += '<button type="button" class="btn btn-outline btn-sm" id="find-more-' + messageId + '">Find More Studies</button>';
        html += '<button type="button" class="btn btn-outline btn-sm" id="find-similar-' + messageId + '">Find Similar Studies</button>';
        if (trayStudies.length >= 2) {
            html += '<button type="button" class="btn btn-accent btn-sm" id="compare-selected-' + messageId + '">Compare Selected (' + trayStudies.length + ')</button>';
        }
        html += '</div>';
        html += '</div>';
        
        this.updateMessage(messageId, 'Review Studies', html);
        this.initStudyReviewHandlers(messageId, query);
    },
    
    /**
     * Show study selection popup modal
     */
    showStudySelectionModal() {
        // Remove existing modal if any
        const existing = document.getElementById('studySelectionModal');
        if (existing) existing.remove();

        const trayStudies = typeof getComparisonTray === 'function' ? getComparisonTray() : [];
        
        let studyListHtml = '';
        if (trayStudies.length > 0) {
            trayStudies.forEach(function renderModalStudy(study, idx) {
                const title = study.title || 'Unknown Study';
                const year = study.year ? ' (' + study.year + ')' : '';
                const docId = study.doc_id || '';
                studyListHtml += `
                    <div class="modal-study-item" data-doc-id="${escapeHtml(docId)}">
                        <div class="modal-study-info">
                            <span class="modal-study-title">${escapeHtml(title)}${year}</span>
                        </div>
                        <button type="button" class="btn btn-outline btn-sm btn-remove-modal-study" data-doc-id="${escapeHtml(docId)}">Remove</button>
                    </div>
                `;
            });
        } else {
            studyListHtml = '<p class="modal-empty-message">No studies selected. Add studies from search results or study details.</p>';
        }

        const overlay = document.createElement('div');
        overlay.id = 'studySelectionModal';
        overlay.className = 'study-selection-overlay';
        overlay.innerHTML = `
            <div class="study-selection-modal">
                <div class="modal-header">
                    <h3>Edit Study Selection</h3>
                    <button type="button" class="modal-close-btn" id="closeStudyModal">x</button>
                </div>
                <div class="modal-body">
                    <p class="modal-subtitle">Studies in your review queue (${trayStudies.length}):</p>
                    <div class="modal-study-list" id="modalStudyList">
                        ${studyListHtml}
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-outline btn-sm" id="clearAllStudiesBtn">Clear All</button>
                    <button type="button" class="btn btn-accent btn-sm" id="doneEditingBtn">Done</button>
                </div>
            </div>
        `;

        document.body.appendChild(overlay);

        const self = this;
        
        // Close button
        document.getElementById('closeStudyModal').addEventListener('click', function handleCloseModal() {
            overlay.remove();
        });
        
        // Done button
        document.getElementById('doneEditingBtn').addEventListener('click', function handleDoneEditing() {
            overlay.remove();
            // Refresh the study review form if it exists
            const reviewForm = document.querySelector('.study-review-form');
            if (reviewForm) {
                const messageId = reviewForm.closest('.message')?.id;
                if (messageId) {
                    self.showStudyReviewForm(null);
                }
            }
        });
        
        // Clear All button
        document.getElementById('clearAllStudiesBtn').addEventListener('click', function handleClearAll() {
            if (typeof saveComparisonTray === 'function') {
                saveComparisonTray([]);
            }
            document.getElementById('modalStudyList').innerHTML = '<p class="modal-empty-message">No studies selected.</p>';
        });
        
        // Remove individual study buttons
        overlay.querySelectorAll('.btn-remove-modal-study').forEach(function attachRemoveHandler(btn) {
            btn.addEventListener('click', function handleRemoveStudy() {
                const docId = btn.dataset.docId;
                if (docId && typeof getComparisonTray === 'function' && typeof saveComparisonTray === 'function') {
                    let tray = getComparisonTray();
                    tray = tray.filter(function filterStudy(s) { return s.doc_id !== docId; });
                    saveComparisonTray(tray);
                    btn.closest('.modal-study-item').remove();
                    
                    // Update count
                    const subtitle = overlay.querySelector('.modal-subtitle');
                    if (subtitle) {
                        subtitle.textContent = 'Studies in your review queue (' + tray.length + '):';
                    }
                    
                    if (tray.length === 0) {
                        document.getElementById('modalStudyList').innerHTML = '<p class="modal-empty-message">No studies selected.</p>';
                    }
                }
            });
        });
        
        // Close on overlay click
        overlay.addEventListener('click', function handleOverlayClick(e) {
            if (e.target === overlay) {
                overlay.remove();
            }
        });
    },
    
    /**
     * Initialize study review form handlers
     */
    initStudyReviewHandlers(messageId, query) {
        const self = this;
        
        // Get effective query from multiple sources
        let effectiveQuery = query;
        if (!effectiveQuery || effectiveQuery.length < 10) {
            if (self.lastQueryContext && self.lastQueryContext.query) {
                effectiveQuery = self.lastQueryContext.query;
            }
            if (!effectiveQuery || effectiveQuery.length < 10) {
                try {
                    const followup = sessionStorage.getItem('followupContext');
                    if (followup) {
                        const parsed = JSON.parse(followup);
                        if (parsed.query && parsed.query.length >= 10) {
                            effectiveQuery = parsed.query;
                        }
                    }
                } catch (e) {
                    console.warn('[InChatModules] Failed to parse followup context:', e);
                }
            }
        }
        
        // Edit Selection button - opens the modal
        const editSelectionBtn = document.getElementById('edit-selection-' + messageId);
        if (editSelectionBtn) {
            editSelectionBtn.addEventListener('click', function handleEditSelection() {
                self.showStudySelectionModal();
            });
        }
        
        // Remove study buttons in the list
        const container = document.getElementById(messageId);
        if (container) {
            container.querySelectorAll('.btn-remove-study').forEach(function attachRemoveHandler(btn) {
                btn.addEventListener('click', function handleRemoveStudy(e) {
                    e.preventDefault();
                    const docId = btn.dataset.docId;
                    if (docId && typeof getComparisonTray === 'function' && typeof saveComparisonTray === 'function') {
                        let tray = getComparisonTray();
                        tray = tray.filter(function filterStudy(s) { return s.doc_id !== docId; });
                        saveComparisonTray(tray);
                        btn.closest('.study-review-item').remove();
                        
                        // Update compare button
                        const compareBtn = document.getElementById('compare-selected-' + messageId);
                        if (compareBtn && tray.length < 2) {
                            compareBtn.remove();
                        } else if (compareBtn) {
                            compareBtn.textContent = 'Compare Selected (' + tray.length + ')';
                        }
                    }
                });
            });
        }
        
        // Find More Studies button
        const findMoreBtn = document.getElementById('find-more-' + messageId);
        if (findMoreBtn) {
            findMoreBtn.addEventListener('click', async function handleFindMore() {
                // Enrich query with clinical context
                let searchQuery = effectiveQuery;
                const clinicalContext = self.getClinicalContext();
                
                // Build enriched query from clinical context if available
                if (clinicalContext) {
                    let contextParts = [];
                    if (clinicalContext.cancerType) {
                        contextParts.push(clinicalContext.cancerType);
                    }
                    if (clinicalContext.cancerStage) {
                        contextParts.push('stage ' + clinicalContext.cancerStage);
                    }
                    if (clinicalContext.histology) {
                        contextParts.push(clinicalContext.histology);
                    }
                    
                    if (contextParts.length > 0) {
                        // If we have clinical context, use it as the base query
                        searchQuery = contextParts.join(' ');
                        // Add original query if it has meaningful content
                        if (effectiveQuery && effectiveQuery.length >= 10) {
                            searchQuery = effectiveQuery + ' ' + contextParts.join(' ');
                        }
                        console.log('[InChatModules] Find More Studies - enriched query:', searchQuery);
                    }
                }
                
                // Check if we have a valid query
                if (!searchQuery || searchQuery.length < 5) {
                    // Show input prompt for query
                    self.showStudySearchPrompt(messageId);
                    return;
                }
                
                findMoreBtn.disabled = true;
                findMoreBtn.textContent = 'Searching...';
                
                try {
                    await self.executePatientMatching(searchQuery);
                } catch (error) {
                    console.error('[InChatModules] Find More Studies error:', error);
                    self.addModuleError('patient-matching', error.message);
                } finally {
                    findMoreBtn.disabled = false;
                    findMoreBtn.textContent = 'Find More Studies';
                }
            });
        }
        
        // Find Similar Studies button
        const findSimilarBtn = document.getElementById('find-similar-' + messageId);
        if (findSimilarBtn) {
            findSimilarBtn.addEventListener('click', function handleFindSimilar() {
                self.showFindSimilarStudiesForm();
            });
        }
        
        // Compare Selected Studies button
        const compareBtn = document.getElementById('compare-selected-' + messageId);
        if (compareBtn) {
            compareBtn.addEventListener('click', async function handleCompare() {
                // Get checked studies from tray (not checkboxes)
                const trayStudies = typeof getComparisonTray === 'function' ? getComparisonTray() : [];
                
                if (trayStudies.length < 2) {
                    alert('Please select at least 2 studies to compare.');
                    return;
                }
                
                compareBtn.disabled = true;
                compareBtn.textContent = 'Comparing...';
                
                // Run comparison with selected studies
                await self.executeStudyComparisonWithIds(trayStudies);
            });
        }
        
        // Update compare button state when checkboxes change
        const reviewList = document.getElementById('study-review-list-' + messageId);
        if (reviewList && compareBtn) {
            reviewList.addEventListener('change', function handleCheckboxChange() {
                const checkedCount = reviewList.querySelectorAll('input[type="checkbox"]:checked').length;
                compareBtn.disabled = checkedCount < 2;
                compareBtn.textContent = checkedCount < 2 ? 'Select at least 2 studies' : 'Compare Selected Studies';
            });
        }
    },
    
    /**
     * Execute study comparison with specific study IDs - opens in SplitPanel
     */
    async executeStudyComparisonWithIds(studies) {
        const messageId = this.addLoadingMessage('Comparing Studies');
        const self = this;

        try {
            const studyIds = studies.map(s => s.doc_id).filter(id => id);

            console.log('[InChatModules] Comparing studies:', studyIds);

            if (studyIds.length < 2) {
                this.updateMessage(messageId, 'Study Comparison',
                    '<div class="alert alert-warning">Please select at least 2 studies to compare.</div>');
                return;
            }

            // Render comparison results in chat
            const api = new PaxisAPI();
            const comparisonResult = await api.compareStudies(studyIds);
            console.log('[InChatModules] Comparison result:', comparisonResult);

            let html = '';

            // Studies being compared
            html += '<div class="module-section">' +
                '<h4>Comparing ' + studies.length + ' Studies</h4>' +
                '<div class="compared-studies-list">' +
                    studies.map((s, i) => '<div class="compared-study">' + (i + 1) + '. ' + escapeHtml(s.title || 'Unknown') + (s.year ? ' (' + s.year + ')' : '') + '</div>').join('') +
                '</div>' +
            '</div>';

            // Narrative summary
            if (comparisonResult.narrative) {
                html += '<div class="module-section">' +
                    '<h4>Comparison Summary</h4>' +
                    '<div class="comparison-narrative">' + markdownToHtml(comparisonResult.narrative) + '</div>' +
                '</div>';
            }
                
                // Comparison categories
                if (comparisonResult.categories && comparisonResult.categories.length > 0) {
                    html += '<div class="module-section">' +
                        '<h4>Side-by-Side Comparison</h4>' +
                        this.renderComparisonCategories(comparisonResult.categories, comparisonResult.studies) +
                    '</div>';
                }
                
                // Continue research section
                html += '<div class="continue-research-section">' +
                    '<p class="continue-research-label">Continue your research</p>' +
                    '<div class="continue-research-buttons">' +
                    '<button type="button" class="btn btn-outline btn-sm continue-btn" data-action="treatment-eval">Evaluate Treatment Options</button>' +
                    '<button type="button" class="btn btn-outline btn-sm continue-btn" data-action="studies">Review Studies</button>' +
                    '<button type="button" class="btn btn-outline btn-sm continue-btn" data-action="analytics">Explore Advanced Analytics</button>' +
                '</div>' +
                '<div class="new-conversation-divider">' +
                    '<span>or</span>' +
                '</div>' +
                '<button type="button" class="btn btn-outline btn-sm new-conversation-btn">Start a New Conversation</button>' +
            '</div>';
                
                this.updateMessage(messageId, 'Study Comparison', html);
            
                // Render charts after DOM update
                if (comparisonResult.categories && comparisonResult.categories.length > 0) {
                    setTimeout(() => {
                        this.renderComparisonCharts(comparisonResult.categories);
                    }, 100);
                }
            
                // Try to build context from study titles if no other context available
                let contextQuery = '';
                if (this.lastQueryContext && this.lastQueryContext.query) {
                    contextQuery = this.lastQueryContext.query;
                }
                if (!contextQuery || contextQuery.length < 10) {
                    // Build context from study titles
                    const studyTitles = studies.map(s => s.title).filter(t => t).slice(0, 2);
                    if (studyTitles.length > 0) {
                        contextQuery = studyTitles.join(' vs ');
                    }
                }
            
                this.initContinueButtons(messageId, contextQuery);

        } catch (error) {
            console.error('[InChatModules] Study comparison error:', error);
            this.updateMessage(messageId, 'Study Comparison',
                '<div class="alert alert-danger">Error: ' + escapeHtml(error.message) + '</div>');
        }
    },
    
    /**
     * Initialize add-to-review button click handlers
     */
    initAddToReviewButtons(messageId) {
        const container = document.getElementById(messageId);
        if (!container) return;
        
        const self = this;
        container.querySelectorAll('.add-to-review-btn').forEach(btn => {
            btn.addEventListener('click', function handleAddToReview() {
                const docId = btn.dataset.docId;
                const title = btn.dataset.title;
                const doi = btn.dataset.doi;
                const year = btn.dataset.year;
                
                if (!docId) return;
                
                // Get current tray
                let tray = typeof getComparisonTray === 'function' ? getComparisonTray() : [];
                const index = tray.findIndex(s => s.doc_id === docId);
                
                if (index >= 0) {
                    // Remove from tray
                    tray.splice(index, 1);
                    if (typeof saveComparisonTray === 'function') {
                        saveComparisonTray(tray);
                    }
                    btn.textContent = 'Add to Review';
                    btn.classList.remove('btn-primary');
                    btn.classList.add('btn-outline');
                } else {
                    // Add to tray (max 4)
                    if (tray.length >= 4) {
                        self.showReviewQueueModal(docId, title, doi, year);
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
                
                // Update badge if function exists
                if (typeof updateComparisonBadge === 'function') {
                    updateComparisonBadge();
                }
            });
        });
    },
    
    /**
     * Show form to find similar studies from user's uploaded documents
     */
    async showFindSimilarStudiesForm() {
        const messageId = this.addLoadingMessage('Find Similar Studies');
        
        try {
            const api = new PaxisAPI();
            
            // Check if user is logged in
            const token = localStorage.getItem('exueed_token');
            if (!token) {
                this.updateMessage(messageId, 'Find Similar Studies',
                    '<div class="alert alert-info">Please log in to access your uploaded studies.</div>' +
                    '<div style="text-align: center; margin-top: 1rem;">' +
                        '<a href="login.html" class="btn btn-accent btn-sm">Log In</a>' +
                    '</div>');
                return;
            }
            
            // Get user's uploaded study profiles
            const result = await api.getUserStudyProfiles();
            const profiles = result.profiles || [];
            
            if (profiles.length === 0) {
                this.updateMessage(messageId, 'Find Similar Studies',
                    '<div class="alert alert-info">You have not uploaded any studies yet. Upload a study to find similar research in our knowledge base.</div>' +
                    '<div style="text-align: center; margin-top: 1rem;">' +
                        '<a href="upload.html" class="btn btn-accent btn-sm">Upload a Study</a>' +
                    '</div>');
                return;
            }
            
            // Build selection form
            let html = '<div class="find-similar-form">';
            html += '<p>Select one of your uploaded studies to find similar research:</p>';
            html += '<select id="user-study-select-' + messageId + '" class="form-select" style="width: 100%; padding: 0.5rem; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 1rem;">';
            html += '<option value="">-- Select a study --</option>';
            
            profiles.forEach(function(profile) {
                const title = profile.title || profile.filename || 'Untitled';
                const truncatedTitle = title.length > 60 ? title.substring(0, 60) + '...' : title;
                html += '<option value="' + escapeHtml(profile.upload_id) + '" data-title="' + escapeHtml(title) + '">' + escapeHtml(truncatedTitle) + '</option>';
            });
            
            html += '</select>';
            html += '<button type="button" class="btn btn-accent btn-sm" id="find-similar-submit-' + messageId + '" disabled>Find Similar Studies</button>';
            html += '</div>';
            html += '<div id="similar-results-' + messageId + '" style="margin-top: 1rem;"></div>';
            
            this.updateMessage(messageId, 'Find Similar Studies', html);
            
            // Initialize handlers
            const self = this;
            const selectEl = document.getElementById('user-study-select-' + messageId);
            const submitBtn = document.getElementById('find-similar-submit-' + messageId);
            const resultsDiv = document.getElementById('similar-results-' + messageId);
            
            if (selectEl && submitBtn) {
                selectEl.addEventListener('change', function() {
                    submitBtn.disabled = !selectEl.value;
                });
                
                submitBtn.addEventListener('click', async function() {
                    const uploadId = selectEl.value;
                    const selectedOption = selectEl.options[selectEl.selectedIndex];
                    const studyTitle = selectedOption ? selectedOption.dataset.title : '';
                    
                    if (!uploadId) return;
                    
                    submitBtn.disabled = true;
                    submitBtn.textContent = 'Searching...';
                    resultsDiv.innerHTML = '<div class="loading"><div class="spinner"></div><span>Finding similar studies...</span></div>';
                    
                    try {
                        const similarResult = await api.findSimilarStudies(uploadId, 10);
                        self.renderSimilarStudiesResults(resultsDiv, similarResult, studyTitle, messageId);
                    } catch (error) {
                        console.error('[InChatModules] Find similar studies error:', error);
                        resultsDiv.innerHTML = '<div class="alert alert-danger">Error: ' + escapeHtml(error.message) + '</div>';
                    } finally {
                        submitBtn.disabled = false;
                        submitBtn.textContent = 'Find Similar Studies';
                    }
                });
            }
            
        } catch (error) {
            console.error('[InChatModules] Find similar studies form error:', error);
            this.updateMessage(messageId, 'Find Similar Studies',
                '<div class="alert alert-danger">Error: ' + escapeHtml(error.message) + '</div>');
        }
    },
    
    /**
     * Render similar studies results
     */
    renderSimilarStudiesResults(container, result, sourceStudyTitle, messageId) {
        if (!result.similar_studies || result.similar_studies.length === 0) {
            container.innerHTML = '<div class="alert alert-info">No similar studies found in the knowledge base.</div>';
            return;
        }
        
        let html = '';
        
        // Source study reference
        if (sourceStudyTitle) {
            html += '<div style="font-size: 0.85rem; color: #64748b; margin-bottom: 0.75rem;">Similar to: ' + escapeHtml(sourceStudyTitle) + '</div>';
        }
        
        // Comparison summary
        if (result.comparison_summary) {
            html += '<div class="module-section">' +
                '<div class="module-summary">' + escapeHtml(result.comparison_summary) + '</div>' +
            '</div>';
        }
        
        // Similar studies list
        html += '<div class="module-section">' +
            '<h4>Similar Studies (' + result.similar_studies.length + ' found)</h4>' +
            '<div class="module-matches">';
        
        const self = this;
        result.similar_studies.forEach(function(study, index) {
            const scorePct = Math.round((study.relevance_score || 0) * 100);
            const title = study.title || 'Unknown Study';
            const author = study.author || '';
            const year = study.year || '';
            const doi = study.doi || '';
            const pmid = study.pmid || '';
            const docId = study.doc_id || '';
            
            // Study Details button
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
            
            // Add to Review button
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
            
            // DOI link
            let doiLink = '';
            if (doi) {
                doiLink = '<a href="https://doi.org/' + doi + '" target="_blank" rel="noopener" ' +
                    'class="btn btn-outline btn-sm" style="font-size: 0.75rem;">DOI</a>';
            }
            
            html += '<div class="match-item">' +
                '<div class="match-header">' +
                    '<div class="match-title">' + (index + 1) + '. ' + escapeHtml(title) + '</div>' +
                    '<div class="match-score">' + scorePct + '% match</div>' +
                '</div>' +
                '<div class="match-meta">' +
                    (author ? escapeHtml(author) : '') +
                    (year ? ' (' + year + ')' : '') +
                '</div>' +
                '<div class="match-actions">' + studyBtn + reviewBtn + doiLink + '</div>' +
            '</div>';
        });
        
        html += '</div></div>';
        
        // Continue research section
        html += '<div class="continue-research-section">' +
            '<p class="continue-research-label">Continue your research</p>' +
            '<div class="continue-research-buttons">' +
                '<button type="button" class="btn btn-outline btn-sm continue-btn" data-action="treatment-eval">Evaluate Treatment Options</button>' +
                '<button type="button" class="btn btn-outline btn-sm continue-btn" data-action="studies">Review Studies</button>' +
                '<button type="button" class="btn btn-outline btn-sm continue-btn" data-action="analytics">Explore Advanced Analytics</button>' +
            '</div>' +
            '<div class="new-conversation-divider">' +
                '<span>or</span>' +
            '</div>' +
            '<button type="button" class="btn btn-outline btn-sm new-conversation-btn">Start a New Conversation</button>' +
        '</div>';
        
        container.innerHTML = html;
        
        // Initialize continue buttons
        this.initContinueButtons(messageId, '');
        
        // Initialize add-to-review buttons
        this.initAddToReviewButtons(messageId);
    },
    
    /**
     * Show modal when review queue is full (max 4 studies)
     * Allows user to remove studies and auto-adds pending study when done
     */
    showReviewQueueModal(pendingDocId, pendingTitle, pendingDoi, pendingYear) {
        // Remove existing modal if any
        const existing = document.getElementById('reviewQueueModal');
        if (existing) existing.remove();

        const overlay = document.createElement('div');
        overlay.id = 'reviewQueueModal';
        overlay.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.5);z-index:9999;display:flex;align-items:center;justify-content:center;';

        const modal = document.createElement('div');
        modal.style.cssText = 'background:#fff;border-radius:12px;padding:1.5rem;max-width:480px;width:90%;max-height:80vh;overflow-y:auto;box-shadow:0 8px 32px rgba(0,0,0,0.2);';

        const title = document.createElement('h3');
        title.style.cssText = 'margin:0 0 0.25rem 0;font-size:1rem;color:#0f172a;';
        title.textContent = 'Maximum number of studies has been reached.';
        modal.appendChild(title);

        const subtitle = document.createElement('p');
        subtitle.style.cssText = 'margin:0 0 1rem 0;font-size:0.85rem;color:#64748b;';
        subtitle.textContent = 'Choose one or more studies to remove:';
        modal.appendChild(subtitle);

        const list = document.createElement('div');
        list.id = 'reviewQueueModalList';
        list.style.cssText = 'display:flex;flex-direction:column;gap:0.5rem;margin-bottom:1.25rem;';

        function renderModalList() {
            list.innerHTML = '';
            const currentTray = typeof getComparisonTray === 'function' ? getComparisonTray() : [];
            currentTray.forEach(function(study) {
                const row = document.createElement('div');
                row.style.cssText = 'display:flex;align-items:center;gap:0.5rem;padding:0.5rem 0.75rem;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;';

                const removeBtn = document.createElement('button');
                removeBtn.type = 'button';
                removeBtn.textContent = '-';
                removeBtn.style.cssText = 'width:28px;height:28px;border-radius:50%;border:1px solid #e2e8f0;background:#fff;color:#ef4444;font-size:1.1rem;font-weight:700;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:background 0.15s;';
                removeBtn.addEventListener('mouseenter', function() { removeBtn.style.background = '#fef2f2'; });
                removeBtn.addEventListener('mouseleave', function() { removeBtn.style.background = '#fff'; });
                removeBtn.addEventListener('click', function() {
                    let t = typeof getComparisonTray === 'function' ? getComparisonTray() : [];
                    const idx = t.findIndex(function(s) { return s.doc_id === study.doc_id; });
                    if (idx >= 0) {
                        t.splice(idx, 1);
                        if (typeof saveComparisonTray === 'function') {
                            saveComparisonTray(t);
                        }
                        // Update the corresponding page button if visible
                        const pageBtn = document.querySelector('.add-to-review-btn[data-doc-id="' + study.doc_id + '"]');
                        if (pageBtn) {
                            pageBtn.textContent = 'Add to Review';
                            pageBtn.classList.remove('btn-primary');
                            pageBtn.classList.add('btn-outline');
                        }
                        if (typeof updateComparisonBadge === 'function') {
                            updateComparisonBadge();
                        }
                        console.log('[InChatModules] Removed from review queue via modal:', study.doc_id);
                    }
                    renderModalList();
                });

                const label = document.createElement('span');
                label.style.cssText = 'font-size:0.85rem;color:#334155;line-height:1.3;';
                label.textContent = study.title || study.doc_id || 'Unknown Study';

                row.appendChild(removeBtn);
                row.appendChild(label);
                list.appendChild(row);
            });

            if (currentTray.length === 0) {
                const empty = document.createElement('p');
                empty.style.cssText = 'color:#94a3b8;font-size:0.85rem;text-align:center;margin:0.5rem 0;';
                empty.textContent = 'No studies in queue.';
                list.appendChild(empty);
            }
        }

        renderModalList();
        modal.appendChild(list);

        const doneBtn = document.createElement('button');
        doneBtn.type = 'button';
        doneBtn.textContent = 'Done';
        doneBtn.style.cssText = 'width:100%;padding:0.5rem 1rem;background:#2563eb;color:#fff;border:none;border-radius:8px;font-size:0.88rem;font-weight:600;cursor:pointer;transition:background 0.15s;';
        doneBtn.addEventListener('mouseenter', function() { doneBtn.style.background = '#1d4ed8'; });
        doneBtn.addEventListener('mouseleave', function() { doneBtn.style.background = '#2563eb'; });
        doneBtn.addEventListener('click', function() {
            overlay.remove();
            // If there is now room, auto-add the pending study
            let t = typeof getComparisonTray === 'function' ? getComparisonTray() : [];
            if (t.length < 4 && pendingDocId) {
                const alreadyIn = t.some(function(s) { return s.doc_id === pendingDocId; });
                if (!alreadyIn) {
                    t.push({ doc_id: pendingDocId, title: pendingTitle, doi: pendingDoi, year: pendingYear });
                    if (typeof saveComparisonTray === 'function') {
                        saveComparisonTray(t);
                    }
                    const pageBtn = document.querySelector('.add-to-review-btn[data-doc-id="' + pendingDocId + '"]');
                    if (pageBtn) {
                        pageBtn.textContent = 'In Review Queue';
                        pageBtn.classList.remove('btn-outline');
                        pageBtn.classList.add('btn-primary');
                    }
                    if (typeof updateComparisonBadge === 'function') {
                        updateComparisonBadge();
                    }
                    console.log('[InChatModules] Auto-added after modal:', pendingDocId);
                }
            }
        });
        modal.appendChild(doneBtn);

        overlay.appendChild(modal);
        document.body.appendChild(overlay);

        // Close on overlay click (outside modal)
        overlay.addEventListener('click', function(e) {
            if (e.target === overlay) {
                doneBtn.click();
            }
        });
    }
};
