/**
 * ConversationContextManager
 * Manages conversation context storage in browser sessionStorage for zero-latency access.
 * Provides context entries for follow-up queries and retrieval boosting.
 * 
 * Requirements: 2.2, 2.3, 3.3, 5.1, 5.3, 6.1
 */

class ConversationContextManager {
    /**
     * Create a ConversationContextManager instance
     * @param {Object} config - Configuration options
     * @param {string} config.storageKey - Key for sessionStorage (default: 'exueed_conversation_context')
     * @param {number} config.maxEntries - Maximum context entries to store (default: 10)
     */
    constructor(config = {}) {
        this.storageKey = config.storageKey || 'exueed_conversation_context';
        this.maxEntries = config.maxEntries || 10;
        
        // In-memory fallback when sessionStorage is unavailable
        this._memoryFallback = null;
        this._useMemoryFallback = false;
        
        // Check sessionStorage availability
        this._checkStorageAvailability();
    }

    /**
     * Check if sessionStorage is available and working
     * Falls back to in-memory storage if not
     * @private
     */
    _checkStorageAvailability() {
        try {
            const testKey = '__storage_test__';
            sessionStorage.setItem(testKey, 'test');
            sessionStorage.removeItem(testKey);
            this._useMemoryFallback = false;
        } catch (e) {
            console.error('[ConversationContext] sessionStorage unavailable, using in-memory fallback:', e);
            this._useMemoryFallback = true;
            this._memoryFallback = { entries: [] };
        }
    }

    /**
     * Get raw storage data
     * @private
     * @returns {Object} Storage data with entries array
     */
    _getStorageData() {
        if (this._useMemoryFallback) {
            return this._memoryFallback;
        }

        try {
            const data = sessionStorage.getItem(this.storageKey);
            if (!data) {
                return { entries: [] };
            }
            const parsed = JSON.parse(data);
            // Validate structure
            if (!parsed || !Array.isArray(parsed.entries)) {
                console.error('[ConversationContext] Invalid storage data, clearing');
                this.clearContext();
                return { entries: [] };
            }
            return parsed;
        } catch (e) {
            console.error('[ConversationContext] JSON parse error, clearing storage:', e);
            this.clearContext();
            return { entries: [] };
        }
    }

    /**
     * Save storage data
     * @private
     * @param {Object} data - Storage data with entries array
     */
    _setStorageData(data) {
        if (this._useMemoryFallback) {
            this._memoryFallback = data;
            return;
        }

        try {
            sessionStorage.setItem(this.storageKey, JSON.stringify(data));
        } catch (e) {
            // Handle quota exceeded - remove oldest entries and retry
            if (e.name === 'QuotaExceededError' || e.code === 22) {
                console.warn('[ConversationContext] Storage quota exceeded, removing oldest entries');
                this._handleQuotaExceeded(data);
            } else {
                console.error('[ConversationContext] Storage error, switching to memory fallback:', e);
                this._useMemoryFallback = true;
                this._memoryFallback = data;
            }
        }
    }

    /**
     * Handle storage quota exceeded by removing oldest entries
     * @private
     * @param {Object} data - Storage data to save
     */
    _handleQuotaExceeded(data) {
        // Remove oldest entries until we can save
        while (data.entries.length > 0) {
            data.entries.shift(); // Remove oldest
            try {
                sessionStorage.setItem(this.storageKey, JSON.stringify(data));
                return; // Success
            } catch (e) {
                // Continue removing entries
            }
        }
        // If still failing with empty entries, switch to memory fallback
        console.error('[ConversationContext] Cannot save even empty context, using memory fallback');
        this._useMemoryFallback = true;
        this._memoryFallback = data;
    }

    /**
     * Get current context entries
     * @returns {Array<Object>} Array of ContextEntry objects
     */
    getContext() {
        const data = this._getStorageData();
        return data.entries || [];
    }

    /**
     * Add a new entry to the conversation context
     * Enforces maxEntries limit by removing oldest entries
     * @param {Object} entry - ContextEntry object
     * @param {string} entry.query - Raw user query text
     * @param {string} entry.action_type - Action type (query, eval_treatment, patient_match, study_comparison, followup)
     * @param {Array<string>} entry.doc_ids - Doc IDs from retrieval results
     * @param {Array<string>} entry.doc_titles - Document titles for display
     * @param {number} entry.timestamp - Unix timestamp in milliseconds
     * @param {Array<string>} [entry.treatments] - Optional treatments for eval_treatment actions
     * @param {Object} [entry.extracted_profile] - Optional extracted patient profile for patient_match actions
     */
    addEntry(entry) {
        // Validate required fields
        if (!entry || typeof entry.query !== 'string') {
            console.error('[ConversationContext] Invalid entry: missing or invalid query');
            return;
        }

        // Ensure required fields have defaults
        const normalizedEntry = {
            query: entry.query,
            action_type: entry.action_type || 'query',
            doc_ids: Array.isArray(entry.doc_ids) ? entry.doc_ids : [],
            doc_titles: Array.isArray(entry.doc_titles) ? entry.doc_titles : [],
            timestamp: typeof entry.timestamp === 'number' ? entry.timestamp : Date.now()
        };

        // Include treatments if present (for eval_treatment actions)
        if (Array.isArray(entry.treatments) && entry.treatments.length > 0) {
            normalizedEntry.treatments = entry.treatments;
        }
        
        // Include extracted_profile if present (for patient_match actions)
        if (entry.extracted_profile && typeof entry.extracted_profile === 'object') {
            normalizedEntry.extracted_profile = entry.extracted_profile;
        }

        const data = this._getStorageData();
        data.entries.push(normalizedEntry);

        // Enforce max entries limit - remove oldest entries
        while (data.entries.length > this.maxEntries) {
            data.entries.shift();
        }

        this._setStorageData(data);
    }

    /**
     * Clear all conversation context
     * Used when starting a new conversation
     */
    clearContext() {
        if (this._useMemoryFallback) {
            this._memoryFallback = { entries: [] };
            return;
        }

        try {
            sessionStorage.removeItem(this.storageKey);
        } catch (e) {
            console.error('[ConversationContext] Error clearing storage:', e);
            // Reset memory fallback as well
            this._memoryFallback = { entries: [] };
        }
    }

    /**
     * Get unique doc_ids from all previous context entries
     * Used for retrieval boosting in follow-up queries
     * @returns {Array<string>} Array of unique doc_ids
     */
    getPreviousDocIds() {
        const entries = this.getContext();
        const docIdSet = new Set();
        
        for (const entry of entries) {
            if (Array.isArray(entry.doc_ids)) {
                for (const docId of entry.doc_ids) {
                    if (docId) {
                        docIdSet.add(docId);
                    }
                }
            }
        }
        
        return Array.from(docIdSet);
    }

    /**
     * Serialize context for API request
     * Returns the context entries in the format expected by the backend
     * @returns {Object} Object with entries array for API request
     */
    serializeForRequest() {
        const entries = this.getContext();
        
        // Return only the most recent maxEntries
        const recentEntries = entries.slice(-this.maxEntries);
        
        return recentEntries;
    }

    /**
     * Check if there is any conversation context
     * @returns {boolean} True if context has at least one entry
     */
    hasContext() {
        const entries = this.getContext();
        return entries.length > 0;
    }

    /**
     * Get the number of context entries
     * @returns {number} Number of entries in context
     */
    getEntryCount() {
        const entries = this.getContext();
        return entries.length;
    }
}

// Export for use in other modules
// Using window global for vanilla JS compatibility
window.ConversationContextManager = ConversationContextManager;
