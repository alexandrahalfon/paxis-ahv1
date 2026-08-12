/**
 * Paxis API Client
 * Unified API interface for all backend endpoints
 */

// API base URL - automatically detects production vs development
// Load config (will be included before this file)
const API_BASE_URL = CONFIG.API_BASE + '/rag';

// Default ceiling for long-running clinical queries. The backend allows
// up to 300s per request, so without a client-side limit a stalled
// request leaves the user watching "Thinking..." forever with no way to
// tell whether anything is still happening. Generous enough for a
// complex patient profile, short enough to fail visibly.
const QUERY_TIMEOUT_MS = 120000;

/**
 * fetch() with an abort-based timeout.
 * Throws a readable Error on timeout instead of hanging indefinitely.
 */
async function fetchWithTimeout(url, options = {}, timeoutMs = QUERY_TIMEOUT_MS) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
        return await fetch(url, { ...options, signal: controller.signal });
    } catch (err) {
        if (err && err.name === 'AbortError') {
            throw new Error(
                `Request timed out after ${Math.round(timeoutMs / 1000)}s. ` +
                `The server may be busy. Please try again.`
            );
        }
        throw err;
    } finally {
        clearTimeout(timer);
    }
}

class PaxisAPI {
    constructor(baseUrl = API_BASE_URL) {
        this.baseUrl = baseUrl;
        
        // Initialize conversation context manager for automatic context tracking
        // Requirements: 1.2, 2.2, 5.2
        if (typeof ConversationContextManager !== 'undefined') {
            this.contextManager = new ConversationContextManager();
        } else {
            console.warn('[PaxisAPI] ConversationContextManager not available, context tracking disabled');
            this.contextManager = null;
        }
    }

    _getAuthHeaders() {
        const token = localStorage.getItem('exueed_token');
        return token ? { 'Authorization': `Bearer ${token}` } : {};
    }

    setToken(token) {
        if (token) {
            localStorage.setItem('exueed_token', token);
        } else {
            localStorage.removeItem('exueed_token');
        }
    }

    /**
     * Clear conversation context to start a new conversation
     * Requirements: 1.2, 2.2, 5.2
     */
    clearConversationContext() {
        if (this.contextManager) {
            this.contextManager.clearContext();
        }
    }

    /**
     * Check if there is any conversation context
     * @returns {boolean} True if context has at least one entry
     * Requirements: 1.2, 2.2, 5.2
     */
    hasConversationContext() {
        return this.contextManager ? this.contextManager.hasContext() : false;
    }

    async register(email, password, firstName, lastName, institution) {
        const response = await fetch(`${CONFIG.API_BASE}/auth/register`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                email, password,
                first_name: firstName,
                last_name: lastName,
                institution,
            })
        });
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Register failed: ${response.status}`);
        }
        return await response.json();
    }

    async login(email, password) {
        const response = await fetch(`${CONFIG.API_BASE}/auth/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ email, password })
        });
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Login failed: ${response.status}`);
        }
        const data = await response.json();
        this.setToken(data.access_token);
        return data;
    }

    async getMe() {
        const response = await fetch(`${CONFIG.API_BASE}/auth/me`, {
            headers: {
                ...this._getAuthHeaders()
            }
        });
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Me failed: ${response.status}`);
        }
        return await response.json();
    }

    /**
     * Query the RAG system
     * @param {string} question - The question to ask
     * @param {string} mode - Query mode: 'basic' (standard), 'concise' (brief), 'detailed' (comprehensive)
     * @param {number} topK - Number of chunks to retrieve
     * @returns {Promise<Object>} Response with answer, sources, and chunks
     */
    async query(question, mode = 'basic', topK = 5, conversationHistory = [], accumulatedContext = null) {
        try {
            const body = {
                question,
                query_mode: mode,
                top_k: topK,
                conversation_history: conversationHistory
            };
            
            // Include accumulated context for conversation continuity
            if (accumulatedContext) {
                body.accumulated_context = accumulatedContext;
            }
            
            // Include serialized conversation context for retrieval boosting
            // Requirements: 1.2, 2.2, 5.2
            if (this.contextManager) {
                const conversationContext = this.contextManager.serializeForRequest();
                if (conversationContext && conversationContext.length > 0) {
                    body.conversation_context = conversationContext;
                }
            }
            
            const response = await fetchWithTimeout(`${this.baseUrl}/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...this._getAuthHeaders()
                },
                body: JSON.stringify(body)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            
            // Store updated context entry from response
            // Requirements: 1.2, 2.2, 5.2
            if (this.contextManager && result.updated_context_entry) {
                this.contextManager.addEntry(result.updated_context_entry);
            }
            
            return result;
        } catch (error) {
            console.error('Query error:', error);
            throw error;
        }
    }

    /**
     * Enhanced query with PTO routing - returns short answer + justification
     * @param {string} question - The question to ask
     * @param {string} mode - Query mode
     * @param {number} topK - Number of chunks to retrieve
     * @param {Array} conversationHistory - Previous conversation messages
     * @param {Object} accumulatedContext - Accumulated structured context from previous queries
     * @param {Object} options - Additional options (useStudyFocused, maxStudies, chunksPerStudy)
     * @returns {Promise<Object>} Response with short_answer, justification, pto_frames, retrieval_results
     */
    async queryEnhanced(question, mode = 'basic', topK = 5, conversationHistory = [], accumulatedContext = null, options = {}) {
        try {
            const body = {
                question,
                query_mode: mode,
                top_k: topK,
                conversation_history: conversationHistory
            };
            
            // Include accumulated context for conversation continuity
            if (accumulatedContext) {
                body.accumulated_context = accumulatedContext;
            }
            
            // Include study-focused retrieval options
            if (options.useStudyFocused !== undefined) {
                body.use_study_focused = options.useStudyFocused;
            }
            if (options.maxStudies !== undefined) {
                body.max_studies = options.maxStudies;
            }
            if (options.chunksPerStudy !== undefined) {
                body.chunks_per_study = options.chunksPerStudy;
            }
            
            // Include serialized conversation context for retrieval boosting
            // Requirements: 1.2, 2.2, 5.2
            if (this.contextManager) {
                const conversationContext = this.contextManager.serializeForRequest();
                if (conversationContext && conversationContext.length > 0) {
                    body.conversation_context = conversationContext;
                }
            }
            
            const response = await fetchWithTimeout(`${this.baseUrl}/query/enhanced`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...this._getAuthHeaders()
                },
                body: JSON.stringify(body)
            });

            if (!response.ok) {
                let detail = `HTTP error! status: ${response.status}`;
                try {
                    const errBody = await response.json();
                    if (errBody.detail) detail = errBody.detail;
                } catch (_) {}
                throw new Error(detail);
            }

            const result = await response.json();

            // Store updated context entry from response
            if (this.contextManager && result.updated_context_entry) {
                this.contextManager.addEntry(result.updated_context_entry);
            }

            return result;
        } catch (error) {
            console.error('[PaxisAPI] Enhanced query error:', error);
            throw error;
        }
    }

    /**
     * Match patient characteristics to studies (structured profile)
     * @param {Object} patientProfile - Patient profile object
     * @param {Object} [criteriaWeights] - Optional criteria weight multipliers
     * @returns {Promise<Object>} Response with matching studies
     */
    async matchPatient(patientProfile, criteriaWeights) {
        try {
            const body = { ...patientProfile };
            if (criteriaWeights) {
                body.criteria_weights = criteriaWeights;
            }
            const response = await fetchWithTimeout(`${this.baseUrl}/patient/match`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...this._getAuthHeaders()
                },
                body: JSON.stringify(body)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Patient matching error:', error);
            throw error;
        }
    }

    /**
     * Match patient from free-text description (unstructured)
     * Enhanced with conversation context support for faster follow-up queries.
     * 
     * @param {string} description - Free-text patient description
     * @param {number} topK - Number of studies to return (default 15)
     * @param {Object} [criteriaWeights] - Optional criteria weight multipliers
     * @param {Object} [cachedProfile] - Optional previously extracted profile to skip LLM extraction
     * @returns {Promise<Object>} Response with matches, extracted_profile, patient_summary
     */
    async matchPatientUnstructured(description, topK = 15, criteriaWeights, cachedProfile = null) {
        try {
            const body = {
                unstructured_description: description,
                top_k: topK
            };
            if (criteriaWeights) {
                body.criteria_weights = criteriaWeights;
            }
            
            // Include cached profile if provided (skips LLM extraction)
            if (cachedProfile) {
                body.cached_profile = cachedProfile;
            }
            
            // Include conversation context for retrieval boosting
            if (this.contextManager) {
                const conversationContext = this.contextManager.serializeForRequest();
                if (conversationContext && conversationContext.length > 0) {
                    body.conversation_context = conversationContext;
                }
            }
            
            const response = await fetchWithTimeout(`${this.baseUrl}/patient/match/unstructured`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...this._getAuthHeaders()
                },
                body: JSON.stringify(body)
            });

            if (!response.ok) {
                const errBody = await response.json().catch(() => ({}));
                const detail = errBody.detail;
                const msg = typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map(d => d.msg || d).join(' ') : `HTTP error! status: ${response.status}`;
                throw new Error(msg || `HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            
            // Store context entry for future queries
            if (this.contextManager && result.extracted_profile) {
                const docIds = (result.matches || []).slice(0, 5).map(m => m.doc_id).filter(Boolean);
                const docTitles = (result.matches || []).slice(0, 5).map(m => m.title).filter(Boolean);
                this.contextManager.addEntry({
                    query: description,
                    action_type: 'patient_match',
                    doc_ids: docIds,
                    doc_titles: docTitles,
                    timestamp: Date.now(),
                    extracted_profile: result.extracted_profile
                });
            }
            
            return result;
        } catch (error) {
            console.error('Unstructured patient matching error:', error);
            throw error;
        }
    }

    /**
     * Search ClinicalTrials.gov (external only)
     * @param {Object} payload - Trial search request payload
     * @returns {Promise<Object>} Matching trials
     */
    async searchClinicalTrials(payload) {
        const base = this.baseUrl.replace('/rag', '');
        const response = await fetch(`${base}/trials/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...this._getAuthHeaders()
            },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Trial search failed: ${response.status}`);
        }
        
        return await response.json();
    }

    /**
     * Compare two treatments
     * @param {string} treatmentA - First treatment name
     * @param {string} treatmentB - Second treatment name
     * @param {string} cancerType - Optional cancer type
     * @param {string} stage - Optional cancer stage
     * @returns {Promise<Object>} Response with treatment comparisons
     */
    async compareTreatments(treatmentA, treatmentB, cancerType = null, stage = null) {
        try {
            const response = await fetch(`${this.baseUrl}/comparison/treatments`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...this._getAuthHeaders()
                },
                body: JSON.stringify({
                    treatment_a: treatmentA,
                    treatment_b: treatmentB,
                    cancer_type: cancerType,
                    stage: stage
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Treatment comparison error:', error);
            throw error;
        }
    }

    /**
     * Visual treatment comparison with charts
     * @param {string} query - Natural language comparison query
     * @param {number} topK - Number of evidence chunks to retrieve
     * @returns {Promise<Object>} Response with summary, detailed_analysis, charts, retrieval_results
     */
    async visualComparison(query, topK = 15) {
        try {
            const response = await fetch(`${this.baseUrl}/compare/visual`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...this._getAuthHeaders()
                },
                body: JSON.stringify({
                    query: query,
                    top_k: topK
                })
            });

            if (!response.ok) {
                const errBody = await response.json().catch(() => ({}));
                throw new Error(errBody.detail || `HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Visual comparison error:', error);
            throw error;
        }
    }

    /**
     * Get health status
     * @returns {Promise<Object>} Health status
     */
    async getHealth() {
        try {
            const response = await fetch(`${this.baseUrl}/health`);
            return await response.json();
        } catch (error) {
            console.error('Health check error:', error);
            throw error;
        }
    }

    /**
     * Get collection statistics
     * @returns {Promise<Object>} Collection stats
     */
    async getStats() {
        try {
            const response = await fetch(`${this.baseUrl}/stats`);
            return await response.json();
        } catch (error) {
            console.error('Stats error:', error);
            throw error;
        }
    }

    /**
     * Generate and download a PDF report
     * @param {string} reportType - 'patient-match' | 'treatment-comparison' | 'query'
     * @param {Object} payload - Full result payload for that mode
     * @param {string} filename - Suggested download filename
     * @returns {Promise<void>}
     */
    async downloadReport(reportType, payload, filename = 'report.pdf') {
        const base = this.baseUrl.replace('/rag', '');
        const url = `${base}/report/${reportType}`;
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...this._getAuthHeaders()
            },
            body: JSON.stringify(payload),
        });
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Report failed: ${response.status}`);
        }
        const blob = await response.blob();
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = filename;
        a.click();
        URL.revokeObjectURL(a.href);
    }

    /**
     * Classify a clinical query into structured data
     * @param {string} query - Free-text clinical description
     * @param {boolean} includeSqlFilters - Whether to include SQL filter conditions
     * @returns {Promise<Object>} Structured query data with demographics, diagnosis, staging, etc.
     */
    async classifyQuery(query, includeSqlFilters = false) {
        const base = this.baseUrl.replace('/rag', '');
        const response = await fetch(`${base}/query-classifier/classify`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...this._getAuthHeaders()
            },
            body: JSON.stringify({
                query,
                include_sql_filters: includeSqlFilters
            })
        });
        
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Classification failed: ${response.status}`);
        }
        
        return await response.json();
    }

    /**
     * Search studies using a classified clinical query
     * @param {string} query - Free-text clinical description
     * @param {number} limit - Maximum number of studies to return
     * @returns {Promise<Object>} Matching studies with extracted fields
     */
    async searchStudiesWithQuery(query, limit = 20) {
        const base = this.baseUrl.replace('/rag', '');
        const response = await fetch(`${base}/query-classifier/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...this._getAuthHeaders()
            },
            body: JSON.stringify({
                query,
                limit,
                include_qdrant_search: true
            })
        });
        
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Search failed: ${response.status}`);
        }
        
        return await response.json();
    }

    // ============================================
    // Saved Cases API
    // ============================================

    /**
     * Save a patient case from a clinical query with optional full response
     * @param {string} query - Free-text clinical description
     * @param {string} caseName - Optional friendly name for the case
     * @param {Object} responseData - Optional full response data
     * @param {string} responseData.answer - The AI response text
     * @param {Array} responseData.sources - List of source citations
     * @param {Object} responseData.metadata - Additional metadata (matching trials, etc.)
     * @returns {Promise<Object>} Saved case info with id, case_name, query_summary
     */
    async saveCase(query, caseName = null, responseData = null) {
        const base = this.baseUrl.replace('/rag', '');
        const body = {
            query,
            case_name: caseName
        };
        
        // Add response data if provided
        if (responseData) {
            if (responseData.answer) body.response_answer = responseData.answer;
            if (responseData.sources) body.response_sources = responseData.sources;
            if (responseData.metadata) body.response_metadata = responseData.metadata;
        }
        
        console.log('[API.saveCase] Sending request:', body);
        
        const response = await fetch(`${base}/saved-cases`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...this._getAuthHeaders()
            },
            body: JSON.stringify(body)
        });
        
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            console.error('[API.saveCase] Error response:', err);
            throw new Error(err.detail || `Save case failed: ${response.status}`);
        }
        
        return await response.json();
    }

    /**
     * Get all saved cases for the current user
     * @param {number} limit - Maximum number of cases to return
     * @param {boolean} includeArchived - Whether to include archived cases
     * @returns {Promise<Object>} List of saved cases
     */
    async getSavedCases(limit = 20, includeArchived = false) {
        const base = this.baseUrl.replace('/rag', '');
        const params = new URLSearchParams({
            limit: limit.toString(),
            include_archived: includeArchived.toString()
        });
        
        const response = await fetch(`${base}/saved-cases?${params}`, {
            headers: {
                ...this._getAuthHeaders()
            }
        });
        
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Get cases failed: ${response.status}`);
        }
        
        return await response.json();
    }

    /**
     * Get a specific saved case by ID
     * @param {number} caseId - Case ID
     * @returns {Promise<Object>} The saved case
     */
    async getCase(caseId) {
        const base = this.baseUrl.replace('/rag', '');
        const response = await fetch(`${base}/saved-cases/${caseId}`, {
            headers: {
                ...this._getAuthHeaders()
            }
        });
        
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Get case failed: ${response.status}`);
        }
        
        return await response.json();
    }

    /**
     * Use a saved case for a new query (increments use count)
     * @param {number} caseId - Case ID
     * @returns {Promise<Object>} The case with context_prompt for chat
     */
    async useCase(caseId) {
        const base = this.baseUrl.replace('/rag', '');
        const response = await fetch(`${base}/saved-cases/${caseId}/use`, {
            method: 'POST',
            headers: {
                ...this._getAuthHeaders()
            }
        });
        
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Use case failed: ${response.status}`);
        }
        
        return await response.json();
    }

    /**
     * Update a saved case (rename or archive)
     * @param {number} caseId - Case ID
     * @param {Object} updates - { case_name, is_archived }
     * @returns {Promise<Object>} Updated case
     */
    async updateCase(caseId, updates) {
        const base = this.baseUrl.replace('/rag', '');
        const response = await fetch(`${base}/saved-cases/${caseId}`, {
            method: 'PATCH',
            headers: {
                'Content-Type': 'application/json',
                ...this._getAuthHeaders()
            },
            body: JSON.stringify(updates)
        });
        
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Update case failed: ${response.status}`);
        }
        
        return await response.json();
    }

    /**
     * Delete a saved case permanently
     * @param {number} caseId - Case ID
     * @returns {Promise<Object>} Success message
     */
    async deleteCase(caseId) {
        const base = this.baseUrl.replace('/rag', '');
        const response = await fetch(`${base}/saved-cases/${caseId}`, {
            method: 'DELETE',
            headers: {
                ...this._getAuthHeaders()
            }
        });
        
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Delete case failed: ${response.status}`);
        }
        
        return await response.json();
    }

    // ============================================
    // Case Alerts API
    // ============================================

    /**
     * Enable alerts for a saved case
     * @param {number} caseId - Case ID
     * @param {boolean} emailNotifications - Whether to send email notifications
     * @returns {Promise<Object>} Alert settings
     */
    async enableCaseAlerts(caseId, emailNotifications = true) {
        const base = this.baseUrl.replace('/rag', '');
        const response = await fetch(`${base}/saved-cases/${caseId}/alerts`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...this._getAuthHeaders()
            },
            body: JSON.stringify({ email_notifications: emailNotifications })
        });
        
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Enable alerts failed: ${response.status}`);
        }
        
        return await response.json();
    }

    /**
     * Disable alerts for a saved case
     * @param {number} caseId - Case ID
     * @returns {Promise<Object>} Success message
     */
    async disableCaseAlerts(caseId) {
        const base = this.baseUrl.replace('/rag', '');
        const response = await fetch(`${base}/saved-cases/${caseId}/alerts`, {
            method: 'DELETE',
            headers: {
                ...this._getAuthHeaders()
            }
        });
        
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Disable alerts failed: ${response.status}`);
        }
        
        return await response.json();
    }

    /**
     * Get alert settings for a case
     * @param {number} caseId - Case ID
     * @returns {Promise<Object>} Alert settings
     */
    async getCaseAlertSettings(caseId) {
        const base = this.baseUrl.replace('/rag', '');
        const response = await fetch(`${base}/saved-cases/${caseId}/alerts`, {
            headers: {
                ...this._getAuthHeaders()
            }
        });
        
        if (!response.ok) {
            if (response.status === 404) return null;
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Get alert settings failed: ${response.status}`);
        }
        
        return await response.json();
    }

    /**
     * Get new trial matches for a case alert
     * @param {number} caseId - Case ID
     * @returns {Promise<Array>} List of matching trials
     */
    async getCaseAlertMatches(caseId) {
        const base = this.baseUrl.replace('/rag', '');
        const response = await fetch(`${base}/saved-cases/${caseId}/alerts/matches`, {
            headers: {
                ...this._getAuthHeaders()
            }
        });
        
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Get matches failed: ${response.status}`);
        }
        
        return await response.json();
    }

    /**
     * Mark all new matches as viewed for a case
     * @param {number} caseId - Case ID
     * @returns {Promise<Object>} Success message
     */
    async markCaseMatchesViewed(caseId) {
        const base = this.baseUrl.replace('/rag', '');
        const response = await fetch(`${base}/saved-cases/${caseId}/alerts/matches/viewed`, {
            method: 'POST',
            headers: {
                ...this._getAuthHeaders()
            }
        });
        
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Mark viewed failed: ${response.status}`);
        }
        
        return await response.json();
    }

    /**
     * Get all alerts for the current user
     * @returns {Promise<Array>} List of alert settings
     */
    async getAllUserAlerts() {
        const base = this.baseUrl.replace('/rag', '');
        const response = await fetch(`${base}/saved-cases/alerts/all`, {
            headers: {
                ...this._getAuthHeaders()
            }
        });
        
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Get alerts failed: ${response.status}`);
        }
        
        return await response.json();
    }

    // ============================================
    // Citation Count API
    // ============================================

    /**
     * Get citation count for a single paper from Semantic Scholar
     * @param {string} doi - DOI of the paper
     * @param {string} pmid - PubMed ID of the paper
     * @returns {Promise<Object>} Citation count info
     */
    async getCitationCount(doi = null, pmid = null) {
        const base = this.baseUrl.replace('/rag', '');
        const response = await fetch(`${base}/user-preferences/citation-count`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...this._getAuthHeaders()
            },
            body: JSON.stringify({ doi, pmid })
        });
        
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Citation count failed: ${response.status}`);
        }
        
        return await response.json();
    }

    /**
     * Get citation counts for multiple papers (batch)
     * @param {Array<{doi?: string, pmid?: string}>} papers - Array of paper identifiers
     * @returns {Promise<Object>} Batch citation count results
     */
    async getCitationCountsBatch(papers) {
        const base = this.baseUrl.replace('/rag', '');
        const response = await fetch(`${base}/user-preferences/citation-counts/batch`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...this._getAuthHeaders()
            },
            body: JSON.stringify({ papers })
        });
        
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Batch citation count failed: ${response.status}`);
        }
        
        return await response.json();
    }

    // ============================================
    // Saved Studies API
    // ============================================

    /**
     * Save a study to the user's collection
     * @param {string} studyId - Unique study identifier
     * @param {string} title - Study title
     * @param {string} doi - DOI (optional)
     * @param {string} pmid - PMID (optional)
     * @returns {Promise<Object>} Saved study record
     */
    async saveStudy(studyId, title = null, doi = null, pmid = null) {
        const base = this.baseUrl.replace('/rag', '');
        const response = await fetch(`${base}/saved-studies`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...this._getAuthHeaders()
            },
            body: JSON.stringify({ study_id: studyId, title, doi, pmid })
        });
        
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Save study failed: ${response.status}`);
        }
        
        return await response.json();
    }

    /**
     * Get all saved studies for the current user
     * @param {number} limit - Maximum number of studies to return
     * @returns {Promise<Object>} List of saved studies
     */
    async getSavedStudies(limit = 50) {
        const base = this.baseUrl.replace('/rag', '');
        const response = await fetch(`${base}/saved-studies?limit=${limit}`, {
            headers: {
                ...this._getAuthHeaders()
            }
        });
        
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Get saved studies failed: ${response.status}`);
        }
        
        return await response.json();
    }

    /**
     * Check if a study is saved
     * @param {string} studyId - Study identifier
     * @returns {Promise<Object>} { saved: boolean }
     */
    async isStudySaved(studyId) {
        const base = this.baseUrl.replace('/rag', '');
        const response = await fetch(`${base}/saved-studies/check/${encodeURIComponent(studyId)}`, {
            headers: {
                ...this._getAuthHeaders()
            }
        });
        
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Check study failed: ${response.status}`);
        }
        
        return await response.json();
    }

    /**
     * Remove a study from saved collection
     * @param {string} studyId - Study identifier
     * @returns {Promise<Object>} Success message
     */
    async deleteSavedStudy(studyId) {
        const base = this.baseUrl.replace('/rag', '');
        const response = await fetch(`${base}/saved-studies/${encodeURIComponent(studyId)}`, {
            method: 'DELETE',
            headers: {
                ...this._getAuthHeaders()
            }
        });
        
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Delete study failed: ${response.status}`);
        }
        
        return await response.json();
    }

    // ============================================
    // User Uploads API
    // ============================================

    /**
     * Upload and process a PDF document
     * @param {File} file - PDF file to upload
     * @returns {Promise<Object>} Processing result with chunks (for session) or success status
     */
    async processUserUpload(file) {
        const base = this.baseUrl.replace('/rag', '');
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${base}/user-uploads/process`, {
            method: 'POST',
            headers: {
                ...this._getAuthHeaders()
            },
            body: formData
        });
        
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Upload failed: ${response.status}`);
        }
        
        return await response.json();
    }

    /**
     * Get all uploads for the current user
     * @param {number} limit - Maximum number of uploads to return
     * @returns {Promise<Object>} List of user uploads
     */
    async getUserUploads(limit = 50) {
        const base = this.baseUrl.replace('/rag', '');
        const response = await fetch(`${base}/user-uploads?limit=${limit}`, {
            headers: {
                ...this._getAuthHeaders()
            }
        });
        
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Get uploads failed: ${response.status}`);
        }
        
        return await response.json();
    }

    /**
     * Delete a user upload
     * @param {string} uploadId - Upload identifier
     * @returns {Promise<Object>} Success message
     */
    async deleteUserUpload(uploadId) {
        const base = this.baseUrl.replace('/rag', '');
        const response = await fetch(`${base}/user-uploads/${encodeURIComponent(uploadId)}`, {
            method: 'DELETE',
            headers: {
                ...this._getAuthHeaders()
            }
        });
        
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Delete upload failed: ${response.status}`);
        }
        
        return await response.json();
    }

    /**
     * Migrate a session upload to user's account
     * @param {Object} uploadData - Upload data with embeddings and metadata
     * @returns {Promise<Object>} Success message
     */
    async migrateUserUpload(uploadData) {
        const base = this.baseUrl.replace('/rag', '');
        const response = await fetch(`${base}/user-uploads/migrate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...this._getAuthHeaders()
            },
            body: JSON.stringify({
                upload_id: uploadData.upload_id,
                filename: uploadData.filename,
                title: uploadData.title,
                doc_meta: uploadData.doc_meta || {},
                embeddings: uploadData.embeddings || [],
                chunk_metadata: uploadData.chunk_metadata || []
            })
        });
        
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Migration failed: ${response.status}`);
        }
        
        return await response.json();
    }

    // ============================================
    // Study Details API
    // ============================================

    /**
     * Get study details including abstract (Task 7)
     * @param {Object} payload - {doi, pmid, title, doc_id}
     * @returns {Promise<Object>} Study details including abstract
     */
    async getStudyDetails(payload) {
        const base = this.baseUrl.replace('/rag', '');
        const response = await fetch(`${base}/study-details`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...this._getAuthHeaders()
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Study details failed: ${response.status}`);
        }

        return await response.json();
    }

    // Multi-Study Comparison API
    // ============================================

    /**
     * Compare multiple studies with visualizations
     * @param {Array<string>} studyIds - Array of study doc_ids (2-4 studies)
     * @returns {Promise<Object>} Comparison results with charts and narrative
     */
    async compareStudies(studyIds) {
        const base = this.baseUrl.replace('/rag', '');
        const response = await fetch(`${base}/study-details/compare`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...this._getAuthHeaders()
            },
            body: JSON.stringify({ study_ids: studyIds })
        });
        
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Comparison failed: ${response.status}`);
        }
        
        return await response.json();
    }
    
    /**
     * Get all study profiles for user's uploaded documents
     * @returns {Promise<Object>} List of study profiles
     */
    async getUserStudyProfiles() {
        const base = this.baseUrl.replace('/rag', '');
        const response = await fetch(`${base}/user-uploads/study-profiles/all`, {
            headers: {
                ...this._getAuthHeaders()
            }
        });
        
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Get study profiles failed: ${response.status}`);
        }
        
        return await response.json();
    }
    
    /**
     * Get study profile for a specific user upload
     * @param {string} uploadId - Upload identifier
     * @returns {Promise<Object>} Study profile data
     */
    async getUserStudyProfile(uploadId) {
        const base = this.baseUrl.replace('/rag', '');
        const response = await fetch(`${base}/user-uploads/${encodeURIComponent(uploadId)}/study-profile`, {
            headers: {
                ...this._getAuthHeaders()
            }
        });
        
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Get study profile failed: ${response.status}`);
        }
        
        return await response.json();
    }
    
    /**
     * Find similar studies from the knowledge base for a user's uploaded study
     * @param {string} uploadId - Upload identifier
     * @param {number} limit - Maximum number of similar studies to return
     * @returns {Promise<Object>} Similar studies with comparison summary
     */
    async findSimilarStudies(uploadId, limit = 10) {
        const base = this.baseUrl.replace('/rag', '');
        const response = await fetch(`${base}/user-uploads/${encodeURIComponent(uploadId)}/similar-studies?limit=${limit}`, {
            headers: {
                ...this._getAuthHeaders()
            }
        });
        
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Find similar studies failed: ${response.status}`);
        }
        
        return await response.json();
    }

    /**
     * Analyze query intent - detects if user provided patient/case info without explicit question
     * Returns structured patient profile, matching trials, and follow-up options
     * @param {string} query - User's input text
     * @param {boolean} forceTrialMatch - When true, always extract patient profile and find matching trials
     * @returns {Promise<Object>} Intent analysis with patient_profile, matching_trials, formatted_response
     */
    async analyzeIntent(query, forceTrialMatch = false) {
        const response = await fetch(`${this.baseUrl}/analyze-intent`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...this._getAuthHeaders()
            },
            body: JSON.stringify({ 
                query,
                force_trial_match: forceTrialMatch
            })
        });
        
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Intent analysis failed: ${response.status}`);
        }
        
        return await response.json();
    }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PaxisAPI;
}
