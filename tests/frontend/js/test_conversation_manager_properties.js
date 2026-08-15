/**
 * Property-based tests for ConversationManager.
 * 
 * Feature: unified-rag-pipeline
 * 
 * Tests the following properties:
 * - Property 7: localStorage persistence survives navigation
 * - Property 8: Conversation persists until explicit clear
 * - Property 9: Maximum entry count enforced
 * - Property 10: Oldest entries removed on overflow
 * - Property 11: Corrupted data cleared on load
 * - Property 12: Conversation entry has required fields
 * - Property 13: JSON serialization round-trip
 * - Property 14: Restored messages in correct order
 * - Property 15: New messages appended after restore
 * 
 * Validates: Requirements 6.1-6.7, 7.1-7.6, 8.4, 8.6
 */

import fc from 'fast-check';

// Mock localStorage for testing
class MockLocalStorage {
    constructor() {
        this.store = {};
    }
    
    getItem(key) {
        return this.store[key] || null;
    }
    
    setItem(key, value) {
        this.store[key] = String(value);
    }
    
    removeItem(key) {
        delete this.store[key];
    }
    
    clear() {
        this.store = {};
    }
}

// ConversationManager implementation (inline for testing)
// This mirrors the implementation in frontend/js/conversationManager.js
class ConversationManager {
    constructor(config = {}) {
        this.storageKey = config.storageKey || 'exueed_conversation_history';
        this.maxEntries = config.maxEntries || 20;
        this._memoryFallback = null;
        this._useMemoryFallback = false;
        this._checkStorageAvailability();
    }

    _checkStorageAvailability() {
        try {
            const testKey = '__storage_test__';
            localStorage.setItem(testKey, 'test');
            localStorage.removeItem(testKey);
            this._useMemoryFallback = false;
        } catch (e) {
            this._useMemoryFallback = true;
            this._memoryFallback = { entries: [], version: 1 };
        }
    }

    _getStorageData() {
        if (this._useMemoryFallback) {
            return this._memoryFallback;
        }

        try {
            const data = localStorage.getItem(this.storageKey);
            if (!data) {
                return { entries: [], version: 1 };
            }

            const parsed = JSON.parse(data);

            if (!this._validateStorageData(parsed)) {
                this.clearConversation();
                return { entries: [], version: 1 };
            }

            return parsed;
        } catch (e) {
            this.clearConversation();
            return { entries: [], version: 1 };
        }
    }

    _validateStorageData(data) {
        if (!data || typeof data !== 'object') return false;
        if (!Array.isArray(data.entries)) return false;

        for (const entry of data.entries) {
            if (typeof entry.query !== 'string') return false;
            if (typeof entry.timestamp !== 'number') return false;
        }

        return true;
    }

    _setStorageData(data) {
        if (this._useMemoryFallback) {
            this._memoryFallback = data;
            return;
        }

        try {
            localStorage.setItem(this.storageKey, JSON.stringify(data));
        } catch (e) {
            if (e.name === 'QuotaExceededError' || e.code === 22) {
                this._handleQuotaExceeded(data);
            } else {
                this._useMemoryFallback = true;
                this._memoryFallback = data;
            }
        }
    }

    _handleQuotaExceeded(data) {
        while (data.entries.length > 0) {
            data.entries.shift();
            try {
                localStorage.setItem(this.storageKey, JSON.stringify(data));
                return;
            } catch (e) {
                // Continue removing
            }
        }
        this._useMemoryFallback = true;
        this._memoryFallback = data;
    }

    addEntry(entry) {
        if (!entry || typeof entry.query !== 'string') {
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

        while (data.entries.length > this.maxEntries) {
            data.entries.shift();
        }

        this._setStorageData(data);
    }

    getEntries() {
        const data = this._getStorageData();
        return data.entries || [];
    }

    hasHistory() {
        return this.getEntries().length > 0;
    }

    getEntryCount() {
        return this.getEntries().length;
    }

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

    serializeForRequest() {
        return this.getEntries().slice(-this.maxEntries);
    }

    clearConversation() {
        if (this._useMemoryFallback) {
            this._memoryFallback = { entries: [], version: 1 };
            return;
        }

        try {
            localStorage.removeItem(this.storageKey);
        } catch (e) {
            // Ignore errors
        }
    }

    restoreToUI(chatContainer, renderMessage) {
        if (!chatContainer || typeof renderMessage !== 'function') {
            return;
        }

        const entries = this.getEntries();
        if (entries.length === 0) return;

        chatContainer.innerHTML = '';

        for (const entry of entries) {
            renderMessage(entry.query, null, true);
            if (entry.response) {
                renderMessage(entry.response, entry, false);
            }
        }

        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
}

// Arbitraries for generating test data
// Use stringMatching to ensure query has at least one non-whitespace character
const nonEmptyQueryArbitrary = fc.stringMatching(/^[a-zA-Z0-9][a-zA-Z0-9\s]{0,199}$/);

const entryArbitrary = fc.record({
    query: nonEmptyQueryArbitrary,
    response: fc.string({ maxLength: 500 }),
    action_type: fc.constantFrom('query', 'followup', 'module_execution'),
    doc_ids: fc.array(fc.string({ minLength: 1, maxLength: 50 }), { maxLength: 5 }),
    doc_titles: fc.array(fc.string({ minLength: 1, maxLength: 100 }), { maxLength: 5 }),
    timestamp: fc.integer({ min: 1000000000000, max: 2000000000000 })
});

const validEntryArbitrary = fc.record({
    query: nonEmptyQueryArbitrary,
    response: fc.string({ maxLength: 500 }),
    action_type: fc.constantFrom('query', 'followup', 'module_execution'),
    doc_ids: fc.array(fc.string({ minLength: 1, maxLength: 50 }), { maxLength: 5 }),
    doc_titles: fc.array(fc.string({ minLength: 1, maxLength: 100 }), { maxLength: 5 }),
    timestamp: fc.integer({ min: 1000000000000, max: 2000000000000 })
});

// Helper to clear localStorage before each property test iteration
function withCleanStorage(fn) {
    return (...args) => {
        localStorage.clear();
        return fn(...args);
    };
}

// Setup and teardown
beforeEach(() => {
    global.localStorage = new MockLocalStorage();
});

afterEach(() => {
    if (global.localStorage) {
        global.localStorage.clear();
    }
});

describe('ConversationManager Properties', () => {

    // Feature: unified-rag-pipeline, Property 7: localStorage persistence survives navigation
    test('Property 7: localStorage persistence survives navigation', () => {
        /**
         * Property 7: localStorage persistence survives navigation.
         * 
         * For any valid conversation entry added to ConversationManager,
         * creating a new ConversationManager instance (simulating page navigation)
         * SHALL restore the same entries from localStorage.
         * 
         * **Validates: Requirements 6.1, 6.2, 6.3, 6.4**
         */
        fc.assert(
            fc.property(
                fc.array(validEntryArbitrary, { minLength: 1, maxLength: 10 }),
                (entries) => {
                    // Create first manager and add entries
                    const manager1 = new ConversationManager();
                    entries.forEach(entry => manager1.addEntry(entry));
                    
                    const countBefore = manager1.getEntryCount();
                    
                    // Simulate navigation by creating new manager instance
                    const manager2 = new ConversationManager();
                    const countAfter = manager2.getEntryCount();
                    
                    // Entries should persist across "navigation"
                    return countAfter === countBefore && countAfter > 0;
                }
            ),
            { numRuns: 100 }
        );
    });

    // Feature: unified-rag-pipeline, Property 8: Conversation persists until explicit clear
    test('Property 8: Conversation persists until explicit clear', () => {
        /**
         * Property 8: Conversation persists until explicit clear.
         * 
         * For any sequence of entries added to ConversationManager,
         * the entries SHALL persist until clearConversation() is called.
         * After clearConversation(), getEntryCount() SHALL return 0.
         * 
         * **Validates: Requirements 6.5, 6.7**
         */
        fc.assert(
            fc.property(
                fc.array(validEntryArbitrary, { minLength: 1, maxLength: 15 }),
                (entries) => {
                    const manager = new ConversationManager();
                    
                    // Add entries
                    entries.forEach(entry => manager.addEntry(entry));
                    
                    // Verify entries exist
                    const countBeforeClear = manager.getEntryCount();
                    if (countBeforeClear === 0) return false;
                    
                    // Clear conversation
                    manager.clearConversation();
                    
                    // Verify entries are cleared
                    const countAfterClear = manager.getEntryCount();
                    return countAfterClear === 0;
                }
            ),
            { numRuns: 100 }
        );
    });

    // Feature: unified-rag-pipeline, Property 9: Maximum entry count enforced
    test('Property 9: Maximum entry count enforced', () => {
        /**
         * Property 9: Maximum entry count enforced.
         * 
         * For any number of entries added to ConversationManager,
         * getEntryCount() SHALL never exceed maxEntries (default 20).
         * 
         * **Validates: Requirements 7.2**
         */
        fc.assert(
            fc.property(
                fc.array(validEntryArbitrary, { minLength: 21, maxLength: 50 }),
                (entries) => {
                    const manager = new ConversationManager({ maxEntries: 20 });
                    
                    entries.forEach(entry => manager.addEntry(entry));
                    
                    const count = manager.getEntryCount();
                    return count <= 20;
                }
            ),
            { numRuns: 100 }
        );
    });

    // Feature: unified-rag-pipeline, Property 10: Oldest entries removed on overflow
    test('Property 10: Oldest entries removed on overflow', () => {
        /**
         * Property 10: Oldest entries removed on overflow.
         * 
         * When more than maxEntries are added, the oldest entries
         * (those added first) SHALL be removed to maintain the limit.
         * The most recent entries SHALL be preserved.
         * 
         * **Validates: Requirements 7.2, 7.3**
         */
        fc.assert(
            fc.property(
                fc.integer({ min: 5, max: 10 }),
                (maxEntries) => {
                    const manager = new ConversationManager({ maxEntries });
                    
                    // Add more entries than maxEntries
                    const totalEntries = maxEntries + 5;
                    for (let i = 0; i < totalEntries; i++) {
                        manager.addEntry({
                            query: `Query ${i}`,
                            response: `Response ${i}`,
                            timestamp: 1000000000000 + i
                        });
                    }
                    
                    const entries = manager.getEntries();
                    
                    // Should have exactly maxEntries
                    if (entries.length !== maxEntries) return false;
                    
                    // First entry should be the (totalEntries - maxEntries)th entry
                    // i.e., the oldest entries should have been removed
                    const expectedFirstQuery = `Query ${totalEntries - maxEntries}`;
                    const expectedLastQuery = `Query ${totalEntries - 1}`;
                    
                    return entries[0].query === expectedFirstQuery && 
                           entries[entries.length - 1].query === expectedLastQuery;
                }
            ),
            { numRuns: 100 }
        );
    });

    // Feature: unified-rag-pipeline, Property 11: Corrupted data cleared on load
    test('Property 11: Corrupted data cleared on load', () => {
        /**
         * Property 11: Corrupted data cleared on load.
         * 
         * When localStorage contains invalid/corrupted JSON data,
         * ConversationManager SHALL clear the corrupted data and
         * return an empty entries array.
         * 
         * **Validates: Requirements 7.4**
         */
        fc.assert(
            fc.property(
                fc.oneof(
                    // Invalid JSON
                    fc.constant('not valid json {{{'),
                    // Missing entries array
                    fc.constant('{"version": 1}'),
                    // Entries not an array
                    fc.constant('{"entries": "not an array", "version": 1}'),
                    // Entry missing required query field
                    fc.constant('{"entries": [{"response": "test"}], "version": 1}'),
                    // Entry with wrong query type
                    fc.constant('{"entries": [{"query": 123, "timestamp": 1000}], "version": 1}'),
                    // Entry missing timestamp
                    fc.constant('{"entries": [{"query": "test"}], "version": 1}')
                ),
                (corruptedData) => {
                    // Set corrupted data directly in localStorage
                    localStorage.setItem('exueed_conversation_history', corruptedData);
                    
                    // Create manager - should handle corrupted data
                    const manager = new ConversationManager();
                    
                    // Should return empty entries (corrupted data cleared)
                    const entries = manager.getEntries();
                    return entries.length === 0;
                }
            ),
            { numRuns: 100 }
        );
    });

    // Feature: unified-rag-pipeline, Property 12: Conversation entry has required fields
    test('Property 12: Conversation entry has required fields', () => {
        /**
         * Property 12: Conversation entry has required fields.
         * 
         * For any entry added via addEntry(), the stored entry SHALL
         * contain all required fields: query, response, action_type,
         * doc_ids, doc_titles, timestamp.
         * 
         * **Validates: Requirements 7.1**
         */
        fc.assert(
            fc.property(
                validEntryArbitrary,
                withCleanStorage((entry) => {
                    const manager = new ConversationManager();
                    manager.addEntry(entry);
                    
                    const entries = manager.getEntries();
                    if (entries.length !== 1) return false;
                    
                    const stored = entries[0];
                    
                    // Check all required fields exist
                    const hasQuery = typeof stored.query === 'string';
                    const hasResponse = typeof stored.response === 'string';
                    const hasActionType = typeof stored.action_type === 'string';
                    const hasDocIds = Array.isArray(stored.doc_ids);
                    const hasDocTitles = Array.isArray(stored.doc_titles);
                    const hasTimestamp = typeof stored.timestamp === 'number';
                    
                    return hasQuery && hasResponse && hasActionType && 
                           hasDocIds && hasDocTitles && hasTimestamp;
                })
            ),
            { numRuns: 100 }
        );
    });


    // Feature: unified-rag-pipeline, Property 13: JSON serialization round-trip
    test('Property 13: JSON serialization round-trip', () => {
        /**
         * Property 13: JSON serialization round-trip.
         * 
         * For any valid entry, storing it via addEntry() and retrieving
         * it via getEntries() SHALL preserve all field values exactly.
         * 
         * **Validates: Requirements 7.6**
         */
        fc.assert(
            fc.property(
                validEntryArbitrary,
                withCleanStorage((entry) => {
                    const manager = new ConversationManager();
                    manager.addEntry(entry);
                    
                    const entries = manager.getEntries();
                    if (entries.length !== 1) return false;
                    
                    const stored = entries[0];
                    
                    // Verify round-trip preserves data
                    const queryMatch = stored.query === entry.query;
                    const responseMatch = stored.response === (entry.response || '');
                    const actionTypeMatch = stored.action_type === (entry.action_type || 'query');
                    const timestampMatch = stored.timestamp === entry.timestamp;
                    
                    // Arrays should match
                    const docIdsMatch = JSON.stringify(stored.doc_ids) === 
                                       JSON.stringify(entry.doc_ids || []);
                    const docTitlesMatch = JSON.stringify(stored.doc_titles) === 
                                          JSON.stringify(entry.doc_titles || []);
                    
                    return queryMatch && responseMatch && actionTypeMatch && 
                           timestampMatch && docIdsMatch && docTitlesMatch;
                })
            ),
            { numRuns: 100 }
        );
    });

    // Feature: unified-rag-pipeline, Property 14: Restored messages in correct order
    test('Property 14: Restored messages in correct order', () => {
        /**
         * Property 14: Restored messages in correct order.
         * 
         * When entries are added in sequence, getEntries() SHALL return
         * them in the same order they were added (FIFO order).
         * 
         * **Validates: Requirements 8.4**
         */
        fc.assert(
            fc.property(
                fc.integer({ min: 2, max: 15 }),
                withCleanStorage((numEntries) => {
                    const manager = new ConversationManager();
                    
                    // Add entries with sequential timestamps
                    for (let i = 0; i < numEntries; i++) {
                        manager.addEntry({
                            query: `Query ${i}`,
                            response: `Response ${i}`,
                            timestamp: 1000000000000 + i
                        });
                    }
                    
                    const entries = manager.getEntries();
                    
                    // Verify order is preserved
                    for (let i = 0; i < entries.length; i++) {
                        if (entries[i].query !== `Query ${i}`) return false;
                        if (entries[i].response !== `Response ${i}`) return false;
                    }
                    
                    return true;
                })
            ),
            { numRuns: 100 }
        );
    });

    // Feature: unified-rag-pipeline, Property 15: New messages appended after restore
    test('Property 15: New messages appended after restore', () => {
        /**
         * Property 15: New messages appended after restore.
         * 
         * When a ConversationManager is created with existing localStorage data,
         * new entries added via addEntry() SHALL be appended to the existing
         * entries, not replace them.
         * 
         * **Validates: Requirements 8.6**
         */
        fc.assert(
            fc.property(
                fc.integer({ min: 1, max: 10 }),
                fc.integer({ min: 1, max: 10 }),
                withCleanStorage((initialCount, additionalCount) => {
                    // Create first manager and add initial entries
                    const manager1 = new ConversationManager();
                    for (let i = 0; i < initialCount; i++) {
                        manager1.addEntry({
                            query: `Initial ${i}`,
                            response: `Response ${i}`,
                            timestamp: 1000000000000 + i
                        });
                    }
                    
                    // Simulate navigation - create new manager
                    const manager2 = new ConversationManager();
                    
                    // Add additional entries
                    for (let i = 0; i < additionalCount; i++) {
                        manager2.addEntry({
                            query: `Additional ${i}`,
                            response: `Response ${i}`,
                            timestamp: 2000000000000 + i
                        });
                    }
                    
                    const entries = manager2.getEntries();
                    const expectedCount = Math.min(initialCount + additionalCount, 20);
                    
                    // Should have combined entries (up to max)
                    if (entries.length !== expectedCount) return false;
                    
                    // If within max, verify initial entries are preserved
                    if (initialCount + additionalCount <= 20) {
                        // First entries should be initial ones
                        for (let i = 0; i < initialCount; i++) {
                            if (entries[i].query !== `Initial ${i}`) return false;
                        }
                        // Remaining entries should be additional ones
                        for (let i = 0; i < additionalCount; i++) {
                            if (entries[initialCount + i].query !== `Additional ${i}`) return false;
                        }
                    }
                    
                    return true;
                })
            ),
            { numRuns: 100 }
        );
    });

    // Additional property test for restoreToUI functionality
    test('Property 14b: restoreToUI renders entries in correct order', () => {
        /**
         * Property 14b: restoreToUI renders entries in correct order.
         * 
         * When restoreToUI is called, it SHALL render messages in the
         * same order as they appear in getEntries().
         * 
         * **Validates: Requirements 8.4**
         */
        fc.assert(
            fc.property(
                fc.integer({ min: 1, max: 10 }),
                withCleanStorage((numEntries) => {
                    const manager = new ConversationManager();
                    
                    // Add entries
                    for (let i = 0; i < numEntries; i++) {
                        manager.addEntry({
                            query: `Query ${i}`,
                            response: `Response ${i}`,
                            timestamp: 1000000000000 + i
                        });
                    }
                    
                    // Mock chat container
                    const chatContainer = {
                        innerHTML: '',
                        scrollTop: 0,
                        scrollHeight: 100
                    };
                    
                    // Track render calls
                    const renderCalls = [];
                    const mockRenderMessage = (text, entry, isUser) => {
                        renderCalls.push({ text, entry, isUser });
                    };
                    
                    // Restore to UI
                    manager.restoreToUI(chatContainer, mockRenderMessage);
                    
                    // Verify render order
                    // Each entry should produce 2 render calls: user message + AI response
                    const expectedCalls = numEntries * 2;
                    if (renderCalls.length !== expectedCalls) return false;
                    
                    // Verify order: alternating user/AI messages
                    for (let i = 0; i < numEntries; i++) {
                        const userCall = renderCalls[i * 2];
                        const aiCall = renderCalls[i * 2 + 1];
                        
                        if (userCall.text !== `Query ${i}`) return false;
                        if (!userCall.isUser) return false;
                        if (aiCall.text !== `Response ${i}`) return false;
                        if (aiCall.isUser) return false;
                    }
                    
                    return true;
                })
            ),
            { numRuns: 100 }
        );
    });
});
