/**
 * ConversationManager - Manages conversation persistence with sessionStorage.
 * 
 * Uses sessionStorage to provide:
 * - Persistence during page navigation within the same session
 * - Automatic clearing when browser/tab is closed
 * - Well-defined conversation entry structure
 * - Cross-page restoration within session
 * 
 * Requirements: 6.1, 7.2, 7.3, 7.4, 7.5, 7.6
 */
class ConversationManager {
    /**
     * Create a ConversationManager instance
     * @param {Object} config - Configuration options
     * @param {string} config.storageKey - Key for sessionStorage (default: 'exueed_conversation_history')
     * @param {number} config.maxEntries - Maximum conversation entries to store (default: 20)
     */
    constructor(config = {}) {
        this.storageKey = config.storageKey || 'exueed_conversation_history';
        this.maxEntries = config.maxEntries || 20;
        this._memoryFallback = null;
        this._useMemoryFallback = false;
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
            console.error('[ConversationManager] sessionStorage unavailable:', e);
            this._useMemoryFallback = true;
            this._memoryFallback = { entries: [], version: 1 };
        }
    }

    /**
     * Get storage data with validation
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
                return { entries: [], version: 1 };
            }

            const parsed = JSON.parse(data);

            // Validate structure
            if (!this._validateStorageData(parsed)) {
                console.error('[ConversationManager] Invalid data structure, clearing');
                this.clearConversation();
                return { entries: [], version: 1 };
            }

            return parsed;
        } catch (e) {
            console.error('[ConversationManager] Parse error, clearing:', e);
            this.clearConversation();
            return { entries: [], version: 1 };
        }
    }

    /**
     * Validate storage data structure
     * Ensures data has correct format and all entries have required fields
     * @private
     * @param {Object} data - Data to validate
     * @returns {boolean} True if data structure is valid
     */
    _validateStorageData(data) {
        if (!data || typeof data !== 'object') return false;
        if (!Array.isArray(data.entries)) return false;

        // Validate each entry has required fields
        for (const entry of data.entries) {
            if (typeof entry.query !== 'string') return false;
            if (typeof entry.timestamp !== 'number') return false;
        }

        return true;
    }

    /**
     * Save storage data with quota handling
     * @private
     * @param {Object} data - Storage data to save
     */
    _setStorageData(data) {
        if (this._useMemoryFallback) {
            this._memoryFallback = data;
            return;
        }

        try {
            sessionStorage.setItem(this.storageKey, JSON.stringify(data));
        } catch (e) {
            if (e.name === 'QuotaExceededError' || e.code === 22) {
                console.warn('[ConversationManager] Quota exceeded, removing oldest');
                this._handleQuotaExceeded(data);
            } else {
                console.error('[ConversationManager] Storage error:', e);
                this._useMemoryFallback = true;
                this._memoryFallback = data;
            }
        }
    }

    /**
     * Handle quota exceeded by removing oldest entries
     * Progressively removes oldest entries until storage succeeds
     * @private
     * @param {Object} data - Storage data to save
     */
    _handleQuotaExceeded(data) {
        while (data.entries.length > 0) {
            data.entries.shift();
            try {
                sessionStorage.setItem(this.storageKey, JSON.stringify(data));
                return;
            } catch (e) {
                // Continue removing
            }
        }
        // If still failing with empty entries, switch to memory fallback
        this._useMemoryFallback = true;
        this._memoryFallback = data;
    }

    /**
     * Add a conversation entry with field normalization
     * @param {Object} entry - Conversation entry
     * @param {string} entry.query - User query text
     * @param {string} entry.response - AI response text
     * @param {string} entry.action_type - Action type (query, module_execution, followup)
     * @param {Array<string>} entry.doc_ids - Document IDs from retrieval
     * @param {Array<string>} entry.doc_titles - Document titles
     * @param {number} [entry.timestamp] - Unix timestamp (auto-generated if not provided)
     * Requirements: 6.5, 7.1
     */
    addEntry(entry) {
        if (!entry || typeof entry.query !== 'string') {
            console.error('[ConversationManager] Invalid entry: missing query');
            return;
        }

        const normalizedEntry = {
            query: entry.query,
            response: entry.response || '',
            action_type: entry.action_type || 'query',
            doc_ids: Array.isArray(entry.doc_ids) ? entry.doc_ids : [],
            doc_titles: Array.isArray(entry.doc_titles) ? entry.doc_titles : [],
            timestamp: typeof entry.timestamp === 'number' ? entry.timestamp : Date.now()
        };

        const data = this._getStorageData();
        data.entries.push(normalizedEntry);

        // Enforce max entries (Requirement 7.2)
        while (data.entries.length > this.maxEntries) {
            data.entries.shift();
        }

        this._setStorageData(data);
    }

    /**
     * Get all conversation entries
     * @returns {Array<Object>} Array of conversation entries
     * Requirements: 6.5
     */
    getEntries() {
        const data = this._getStorageData();
        return data.entries || [];
    }

    /**
     * Check if conversation history exists
     * @returns {boolean} True if there are conversation entries
     * Requirements: 6.5
     */
    hasHistory() {
        return this.getEntries().length > 0;
    }

    /**
     * Get entry count
     * @returns {number} Number of conversation entries
     * Requirements: 6.5
     */
    getEntryCount() {
        return this.getEntries().length;
    }

    /**
     * Get unique doc_ids from all entries for context boosting
     * @returns {Array<string>} Array of unique document IDs
     * Requirements: 6.7
     */
    getPreviousDocIds() {
        const entries = this.getEntries();
        const docIdSet = new Set();

        for (const entry of entries) {
            if (Array.isArray(entry.doc_ids)) {
                entry.doc_ids.forEach(id => {
                    if (id) docIdSet.add(id);
                });
            }
        }

        return Array.from(docIdSet);
    }

    /**
     * Serialize conversation for API request
     * Returns entries formatted for backend conversation context
     * @returns {Array<Object>} Array of conversation entries for API
     * Requirements: 7.1
     */
    serializeForRequest() {
        return this.getEntries().slice(-this.maxEntries);
    }

    /**
     * Clear all conversation history
     * Used when starting a new conversation
     * Requirements: 6.7
     */
    clearConversation() {
        if (this._useMemoryFallback) {
            this._memoryFallback = { entries: [], version: 1 };
            return;
        }

        try {
            sessionStorage.removeItem(this.storageKey);
        } catch (e) {
            console.error('[ConversationManager] Error clearing:', e);
        }
    }

    /**
     * Restore conversation to chat UI
     * Renders all stored conversation entries in the chat container
     * @param {HTMLElement} chatContainer - The chat messages container
     * @param {Function} renderMessage - Function to render a message (query, response, isUser)
     * Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6
     */
    restoreToUI(chatContainer, renderMessage) {
        if (!chatContainer || typeof renderMessage !== 'function') {
            console.error('[ConversationManager] Invalid restore parameters');
            return;
        }

        const entries = this.getEntries();
        if (entries.length === 0) return;

        // Clear existing messages
        chatContainer.innerHTML = '';

        // Render each entry in order (entries are already sorted by timestamp)
        for (const entry of entries) {
            // Render user message
            renderMessage(entry.query, null, true);

            // Render AI response with entry data (preserves module action buttons)
            if (entry.response) {
                renderMessage(entry.response, entry, false);
            }
        }

        // Scroll to bottom after restoration
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
}

// Export singleton class
window.ConversationManager = ConversationManager;

// Create default instance
window.conversationManager = new ConversationManager();
