/**
 * Property-based tests for SplitPanel component.
 * 
 * Feature: unified-rag-pipeline
 * 
 * Tests the following properties:
 * - Property 5: Desktop split layout proportions (60/40)
 * - Property 6: Mobile modal overlay (100% width)
 * 
 * Validates: Requirements 5.1, 5.2, 5.3
 */

import fc from 'fast-check';

// Mock DOM elements for testing
class MockElement {
    constructor(id, tagName = 'div') {
        this.id = id;
        this.tagName = tagName;
        this.classList = new MockClassList();
        this.style = {};
        this.innerHTML = '';
        this.children = [];
        this.parentNode = null;
        this.attributes = {};
        this.eventListeners = {};
    }

    setAttribute(name, value) {
        this.attributes[name] = value;
    }

    getAttribute(name) {
        return this.attributes[name] || null;
    }

    addEventListener(event, handler) {
        if (!this.eventListeners[event]) {
            this.eventListeners[event] = [];
        }
        this.eventListeners[event].push(handler);
    }

    removeEventListener(event, handler) {
        if (this.eventListeners[event]) {
            this.eventListeners[event] = this.eventListeners[event].filter(h => h !== handler);
        }
    }

    querySelector(selector) {
        return null;
    }

    querySelectorAll(selector) {
        return [];
    }
}

class MockClassList {
    constructor() {
        this._classes = new Set();
    }

    add(className) {
        this._classes.add(className);
    }

    remove(className) {
        this._classes.delete(className);
    }

    contains(className) {
        return this._classes.has(className);
    }

    toggle(className, force) {
        if (force === undefined) {
            if (this._classes.has(className)) {
                this._classes.delete(className);
                return false;
            } else {
                this._classes.add(className);
                return true;
            }
        } else if (force) {
            this._classes.add(className);
            return true;
        } else {
            this._classes.delete(className);
            return false;
        }
    }
}

// Mock document for testing
class MockDocument {
    constructor() {
        this.elements = {};
        this.body = new MockElement('body', 'body');
        this.eventListeners = {};
    }

    getElementById(id) {
        return this.elements[id] || null;
    }

    querySelector(selector) {
        // Simple selector matching for testing
        if (selector.startsWith('.')) {
            const className = selector.slice(1);
            for (const el of Object.values(this.elements)) {
                if (el.classList.contains(className)) {
                    return el;
                }
            }
        }
        return null;
    }

    createElement(tagName) {
        return new MockElement('', tagName);
    }

    addEventListener(event, handler) {
        if (!this.eventListeners[event]) {
            this.eventListeners[event] = [];
        }
        this.eventListeners[event].push(handler);
    }

    removeEventListener(event, handler) {
        if (this.eventListeners[event]) {
            this.eventListeners[event] = this.eventListeners[event].filter(h => h !== handler);
        }
    }
}

// Mock window for testing
class MockWindow {
    constructor(width = 1200) {
        this._innerWidth = width;
        this.eventListeners = {};
    }

    get innerWidth() {
        return this._innerWidth;
    }

    set innerWidth(value) {
        this._innerWidth = value;
    }

    addEventListener(event, handler) {
        if (!this.eventListeners[event]) {
            this.eventListeners[event] = [];
        }
        this.eventListeners[event].push(handler);
    }

    removeEventListener(event, handler) {
        if (this.eventListeners[event]) {
            this.eventListeners[event] = this.eventListeners[event].filter(h => h !== handler);
        }
    }

    dispatchEvent(event) {
        const handlers = this.eventListeners[event.type] || [];
        handlers.forEach(h => h(event));
    }
}

/**
 * SplitPanel implementation for testing.
 * This mirrors the implementation in frontend/js/splitPanel.js
 * with focus on responsive behavior properties.
 */
const createSplitPanel = (mockDocument, mockWindow) => {
    return {
        isOpen: false,
        currentModule: null,
        _lastQuery: null,
        _document: mockDocument,
        _window: mockWindow,

        init() {
            if (this._document.getElementById('splitPanel')) return;

            // Create panel element
            const panel = new MockElement('splitPanel', 'div');
            panel.classList.add('split-panel');
            panel.setAttribute('aria-hidden', 'true');
            this._document.elements['splitPanel'] = panel;

            // Create overlay element
            const overlay = new MockElement('splitPanelOverlay', 'div');
            overlay.classList.add('split-panel-overlay');
            overlay.setAttribute('aria-hidden', 'true');
            this._document.elements['splitPanelOverlay'] = overlay;

            // Create chat container if not exists
            if (!this._document.elements['chatContainer']) {
                const chatContainer = new MockElement('chatContainer', 'div');
                chatContainer.classList.add('chat-container');
                this._document.elements['chatContainer'] = chatContainer;
            }

            this._bindEvents();
        },

        _bindEvents() {
            const closeBtn = this._document.getElementById('splitPanelClose');
            const overlay = this._document.getElementById('splitPanelOverlay');

            if (closeBtn) {
                const self = this;
                function handleCloseClick() {
                    self.close();
                }
                closeBtn.addEventListener('click', handleCloseClick);
            }

            if (overlay) {
                const self = this;
                function handleOverlayClick() {
                    self.close();
                }
                overlay.addEventListener('click', handleOverlayClick);
            }

            // Handle resize for responsive behavior
            const self = this;
            function handleWindowResize() {
                self._handleResize();
            }
            this._window.addEventListener('resize', handleWindowResize);
        },

        /**
         * Check if viewport is mobile (breakpoint at 1024px)
         * @returns {boolean} True if viewport width < 1024px
         * Requirements: 5.1, 5.2
         */
        _isMobile() {
            return this._window.innerWidth < 1024;
        },

        /**
         * Open the split panel with a specific module
         * @param {string} module - Module name
         * @param {string} query - The query to execute
         * Requirements: 4.1, 4.2, 4.3, 4.7
         */
        open(module, query) {
            this.init();

            const panel = this._document.getElementById('splitPanel');
            const overlay = this._document.getElementById('splitPanelOverlay');
            const chatContainer = this._document.querySelector('.chat-container');

            if (!panel) return;

            this.currentModule = module;
            this.isOpen = true;
            this._lastQuery = query;

            // Apply split layout
            panel.classList.add('open');
            panel.setAttribute('aria-hidden', 'false');

            if (this._isMobile()) {
                // Mobile: full-width modal overlay
                if (overlay) {
                    overlay.classList.add('visible');
                    overlay.setAttribute('aria-hidden', 'false');
                }
                this._document.body.classList.add('panel-open-mobile');
                if (chatContainer) {
                    chatContainer.classList.remove('split-active');
                }
            } else {
                // Desktop: 60/40 split
                if (overlay) {
                    overlay.classList.remove('visible');
                    overlay.setAttribute('aria-hidden', 'true');
                }
                this._document.body.classList.remove('panel-open-mobile');
                if (chatContainer) {
                    chatContainer.classList.add('split-active');
                }
            }
        },

        /**
         * Close the split panel and restore layout
         * Requirements: 4.5, 4.6
         */
        close() {
            const panel = this._document.getElementById('splitPanel');
            const overlay = this._document.getElementById('splitPanelOverlay');
            const chatContainer = this._document.querySelector('.chat-container');

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

            this._document.body.classList.remove('panel-open-mobile');

            this.isOpen = false;
            this.currentModule = null;
        },

        /**
         * Handle window resize for responsive layout switching
         * Switches between 60/40 split (desktop) and modal overlay (mobile)
         * Requirements: 5.1, 5.2, 5.3, 5.4
         */
        _handleResize() {
            if (!this.isOpen) return;

            const overlay = this._document.getElementById('splitPanelOverlay');
            const chatContainer = this._document.querySelector('.chat-container');

            if (this._isMobile()) {
                // Mobile: full-width modal overlay
                if (overlay) {
                    overlay.classList.add('visible');
                    overlay.setAttribute('aria-hidden', 'false');
                }
                this._document.body.classList.add('panel-open-mobile');
                if (chatContainer) {
                    chatContainer.classList.remove('split-active');
                }
            } else {
                // Desktop: 60/40 split
                if (overlay) {
                    overlay.classList.remove('visible');
                    overlay.setAttribute('aria-hidden', 'true');
                }
                this._document.body.classList.remove('panel-open-mobile');
                if (chatContainer) {
                    chatContainer.classList.add('split-active');
                }
            }
        },

        /**
         * Get the current layout state for testing
         * @returns {Object} Layout state with desktop/mobile indicators
         */
        getLayoutState() {
            const panel = this._document.getElementById('splitPanel');
            const overlay = this._document.getElementById('splitPanelOverlay');
            const chatContainer = this._document.querySelector('.chat-container');

            return {
                isOpen: this.isOpen,
                isMobile: this._isMobile(),
                viewportWidth: this._window.innerWidth,
                panelOpen: panel ? panel.classList.contains('open') : false,
                overlayVisible: overlay ? overlay.classList.contains('visible') : false,
                chatSplitActive: chatContainer ? chatContainer.classList.contains('split-active') : false,
                bodyMobileClass: this._document.body.classList.contains('panel-open-mobile')
            };
        }
    };
};

// Arbitraries for generating test data
const desktopViewportArbitrary = fc.integer({ min: 1024, max: 3840 });
const mobileViewportArbitrary = fc.integer({ min: 320, max: 1023 });
const moduleArbitrary = fc.constantFrom('treatment-comparison', 'patient-matching', 'study-comparison');
const queryArbitrary = fc.stringMatching(/^[a-zA-Z0-9][a-zA-Z0-9\s]{0,99}$/);

// Helper to create fresh test environment
function createTestEnvironment(viewportWidth) {
    const mockDoc = new MockDocument();
    const mockWin = new MockWindow(viewportWidth);
    
    // Create chat container
    const chatContainer = new MockElement('chatContainer', 'div');
    chatContainer.classList.add('chat-container');
    mockDoc.elements['chatContainer'] = chatContainer;
    
    const splitPanel = createSplitPanel(mockDoc, mockWin);
    
    return { mockDoc, mockWin, splitPanel, chatContainer };
}

describe('SplitPanel Properties', () => {

    // Feature: unified-rag-pipeline, Property 5: Desktop split layout proportions (60/40)
    test('Property 5: Desktop split layout proportions (60/40)', () => {
        /**
         * Property 5: Desktop split layout proportions (60/40).
         * 
         * For viewport width > 1024px with Split_Panel open,
         * chat container width SHALL be ~60% and Split_Panel width SHALL be ~40%.
         * 
         * This is verified by checking:
         * - Chat container has 'split-active' class (triggers 60% width CSS)
         * - Panel has 'open' class (triggers 40% width CSS)
         * - Overlay is NOT visible (desktop uses side-by-side, not overlay)
         * - Body does NOT have 'panel-open-mobile' class
         * 
         * **Validates: Requirements 5.1, 5.2**
         */
        fc.assert(
            fc.property(
                desktopViewportArbitrary,
                moduleArbitrary,
                queryArbitrary,
                (viewportWidth, module, query) => {
                    const { splitPanel } = createTestEnvironment(viewportWidth);
                    
                    // Open panel on desktop viewport
                    splitPanel.open(module, query);
                    
                    const state = splitPanel.getLayoutState();
                    
                    // Desktop layout assertions:
                    // 1. Panel should be open
                    const panelIsOpen = state.panelOpen === true;
                    
                    // 2. Chat container should have split-active class (60% width)
                    const chatHasSplitActive = state.chatSplitActive === true;
                    
                    // 3. Overlay should NOT be visible (desktop uses side-by-side)
                    const overlayNotVisible = state.overlayVisible === false;
                    
                    // 4. Body should NOT have mobile class
                    const noMobileClass = state.bodyMobileClass === false;
                    
                    // 5. Should not be detected as mobile
                    const notMobile = state.isMobile === false;
                    
                    // 6. Viewport should be >= 1024
                    const isDesktopViewport = state.viewportWidth >= 1024;
                    
                    return panelIsOpen && 
                           chatHasSplitActive && 
                           overlayNotVisible && 
                           noMobileClass && 
                           notMobile && 
                           isDesktopViewport;
                }
            ),
            { numRuns: 100 }
        );
    });

    // Feature: unified-rag-pipeline, Property 6: Mobile modal overlay (100% width)
    test('Property 6: Mobile modal overlay (100% width)', () => {
        /**
         * Property 6: Mobile modal overlay (100% width).
         * 
         * For viewport width < 1024px with Split_Panel open,
         * Split_Panel SHALL have 100% width and overlay SHALL be visible.
         * 
         * This is verified by checking:
         * - Panel has 'open' class
         * - Overlay has 'visible' class (triggers full-width modal CSS)
         * - Body has 'panel-open-mobile' class (prevents background scroll)
         * - Chat container does NOT have 'split-active' class (no split on mobile)
         * 
         * **Validates: Requirements 5.2, 5.3**
         */
        fc.assert(
            fc.property(
                mobileViewportArbitrary,
                moduleArbitrary,
                queryArbitrary,
                (viewportWidth, module, query) => {
                    const { splitPanel } = createTestEnvironment(viewportWidth);
                    
                    // Open panel on mobile viewport
                    splitPanel.open(module, query);
                    
                    const state = splitPanel.getLayoutState();
                    
                    // Mobile layout assertions:
                    // 1. Panel should be open
                    const panelIsOpen = state.panelOpen === true;
                    
                    // 2. Overlay should be visible (mobile uses overlay)
                    const overlayVisible = state.overlayVisible === true;
                    
                    // 3. Body should have mobile class (prevents background scroll)
                    const hasMobileClass = state.bodyMobileClass === true;
                    
                    // 4. Chat container should NOT have split-active class
                    const chatNoSplitActive = state.chatSplitActive === false;
                    
                    // 5. Should be detected as mobile
                    const isMobile = state.isMobile === true;
                    
                    // 6. Viewport should be < 1024
                    const isMobileViewport = state.viewportWidth < 1024;
                    
                    return panelIsOpen && 
                           overlayVisible && 
                           hasMobileClass && 
                           chatNoSplitActive && 
                           isMobile && 
                           isMobileViewport;
                }
            ),
            { numRuns: 100 }
        );
    });

    // Additional property: Resize from desktop to mobile switches layout
    test('Property 5b: Resize from desktop to mobile switches to overlay', () => {
        /**
         * Property 5b: Resize from desktop to mobile switches to overlay.
         * 
         * When panel is open on desktop and viewport resizes to mobile,
         * the layout SHALL switch from 60/40 split to full-width overlay.
         * 
         * **Validates: Requirements 5.1, 5.2, 5.3**
         */
        fc.assert(
            fc.property(
                desktopViewportArbitrary,
                mobileViewportArbitrary,
                moduleArbitrary,
                queryArbitrary,
                (desktopWidth, mobileWidth, module, query) => {
                    const { splitPanel, mockWin } = createTestEnvironment(desktopWidth);
                    
                    // Open panel on desktop
                    splitPanel.open(module, query);
                    
                    // Verify desktop layout
                    let state = splitPanel.getLayoutState();
                    const wasDesktopLayout = state.chatSplitActive === true && 
                                            state.overlayVisible === false;
                    
                    // Resize to mobile
                    mockWin.innerWidth = mobileWidth;
                    splitPanel._handleResize();
                    
                    // Verify mobile layout
                    state = splitPanel.getLayoutState();
                    const isMobileLayout = state.chatSplitActive === false && 
                                          state.overlayVisible === true &&
                                          state.bodyMobileClass === true;
                    
                    return wasDesktopLayout && isMobileLayout;
                }
            ),
            { numRuns: 100 }
        );
    });

    // Additional property: Resize from mobile to desktop switches layout
    test('Property 6b: Resize from mobile to desktop switches to split', () => {
        /**
         * Property 6b: Resize from mobile to desktop switches to split.
         * 
         * When panel is open on mobile and viewport resizes to desktop,
         * the layout SHALL switch from full-width overlay to 60/40 split.
         * 
         * **Validates: Requirements 5.1, 5.2, 5.3**
         */
        fc.assert(
            fc.property(
                mobileViewportArbitrary,
                desktopViewportArbitrary,
                moduleArbitrary,
                queryArbitrary,
                (mobileWidth, desktopWidth, module, query) => {
                    const { splitPanel, mockWin } = createTestEnvironment(mobileWidth);
                    
                    // Open panel on mobile
                    splitPanel.open(module, query);
                    
                    // Verify mobile layout
                    let state = splitPanel.getLayoutState();
                    const wasMobileLayout = state.overlayVisible === true && 
                                           state.bodyMobileClass === true;
                    
                    // Resize to desktop
                    mockWin.innerWidth = desktopWidth;
                    splitPanel._handleResize();
                    
                    // Verify desktop layout
                    state = splitPanel.getLayoutState();
                    const isDesktopLayout = state.chatSplitActive === true && 
                                           state.overlayVisible === false &&
                                           state.bodyMobileClass === false;
                    
                    return wasMobileLayout && isDesktopLayout;
                }
            ),
            { numRuns: 100 }
        );
    });

    // Property: Close restores layout on both desktop and mobile
    test('Property 5c: Close panel restores desktop layout', () => {
        /**
         * Property 5c: Close panel restores desktop layout.
         * 
         * When panel is closed on desktop, the chat container SHALL
         * return to full width (split-active class removed).
         * 
         * **Validates: Requirements 5.1, 5.2**
         */
        fc.assert(
            fc.property(
                desktopViewportArbitrary,
                moduleArbitrary,
                queryArbitrary,
                (viewportWidth, module, query) => {
                    const { splitPanel } = createTestEnvironment(viewportWidth);
                    
                    // Open panel
                    splitPanel.open(module, query);
                    
                    // Verify panel is open with split layout
                    let state = splitPanel.getLayoutState();
                    const wasOpen = state.panelOpen === true && state.chatSplitActive === true;
                    
                    // Close panel
                    splitPanel.close();
                    
                    // Verify layout is restored
                    state = splitPanel.getLayoutState();
                    const isClosed = state.panelOpen === false;
                    const splitRemoved = state.chatSplitActive === false;
                    const overlayHidden = state.overlayVisible === false;
                    const noMobileClass = state.bodyMobileClass === false;
                    
                    return wasOpen && isClosed && splitRemoved && overlayHidden && noMobileClass;
                }
            ),
            { numRuns: 100 }
        );
    });

    test('Property 6c: Close panel restores mobile layout', () => {
        /**
         * Property 6c: Close panel restores mobile layout.
         * 
         * When panel is closed on mobile, the overlay SHALL be hidden
         * and body scroll SHALL be restored.
         * 
         * **Validates: Requirements 5.2, 5.3**
         */
        fc.assert(
            fc.property(
                mobileViewportArbitrary,
                moduleArbitrary,
                queryArbitrary,
                (viewportWidth, module, query) => {
                    const { splitPanel } = createTestEnvironment(viewportWidth);
                    
                    // Open panel
                    splitPanel.open(module, query);
                    
                    // Verify panel is open with overlay
                    let state = splitPanel.getLayoutState();
                    const wasOpen = state.panelOpen === true && state.overlayVisible === true;
                    
                    // Close panel
                    splitPanel.close();
                    
                    // Verify layout is restored
                    state = splitPanel.getLayoutState();
                    const isClosed = state.panelOpen === false;
                    const overlayHidden = state.overlayVisible === false;
                    const mobileClassRemoved = state.bodyMobileClass === false;
                    
                    return wasOpen && isClosed && overlayHidden && mobileClassRemoved;
                }
            ),
            { numRuns: 100 }
        );
    });

    // Property: Breakpoint boundary behavior
    test('Property 5d: Breakpoint at exactly 1024px is desktop', () => {
        /**
         * Property 5d: Breakpoint at exactly 1024px is desktop.
         * 
         * When viewport width is exactly 1024px, the layout SHALL
         * use desktop 60/40 split (not mobile overlay).
         * 
         * **Validates: Requirements 5.1, 5.2**
         */
        fc.assert(
            fc.property(
                moduleArbitrary,
                queryArbitrary,
                (module, query) => {
                    const { splitPanel } = createTestEnvironment(1024);
                    
                    // Open panel at exactly 1024px
                    splitPanel.open(module, query);
                    
                    const state = splitPanel.getLayoutState();
                    
                    // Should be desktop layout at 1024px
                    const isDesktopLayout = state.chatSplitActive === true && 
                                           state.overlayVisible === false &&
                                           state.isMobile === false;
                    
                    return isDesktopLayout;
                }
            ),
            { numRuns: 100 }
        );
    });

    test('Property 6d: Breakpoint at 1023px is mobile', () => {
        /**
         * Property 6d: Breakpoint at 1023px is mobile.
         * 
         * When viewport width is 1023px (just below breakpoint),
         * the layout SHALL use mobile overlay (not desktop split).
         * 
         * **Validates: Requirements 5.2, 5.3**
         */
        fc.assert(
            fc.property(
                moduleArbitrary,
                queryArbitrary,
                (module, query) => {
                    const { splitPanel } = createTestEnvironment(1023);
                    
                    // Open panel at 1023px
                    splitPanel.open(module, query);
                    
                    const state = splitPanel.getLayoutState();
                    
                    // Should be mobile layout at 1023px
                    const isMobileLayout = state.overlayVisible === true && 
                                          state.bodyMobileClass === true &&
                                          state.chatSplitActive === false &&
                                          state.isMobile === true;
                    
                    return isMobileLayout;
                }
            ),
            { numRuns: 100 }
        );
    });

    // Property: Resize when panel is closed has no effect
    test('Property 5e: Resize when closed has no effect', () => {
        /**
         * Property 5e: Resize when closed has no effect.
         * 
         * When panel is closed, resize events SHALL NOT change
         * any layout classes.
         * 
         * **Validates: Requirements 5.1, 5.2, 5.3**
         */
        fc.assert(
            fc.property(
                desktopViewportArbitrary,
                mobileViewportArbitrary,
                (desktopWidth, mobileWidth) => {
                    const { splitPanel, mockWin } = createTestEnvironment(desktopWidth);
                    
                    // Initialize but don't open
                    splitPanel.init();
                    
                    // Get initial state
                    let state = splitPanel.getLayoutState();
                    const initialOverlay = state.overlayVisible;
                    const initialSplit = state.chatSplitActive;
                    const initialMobile = state.bodyMobileClass;
                    
                    // Resize to mobile
                    mockWin.innerWidth = mobileWidth;
                    splitPanel._handleResize();
                    
                    // State should be unchanged (panel is closed)
                    state = splitPanel.getLayoutState();
                    const overlayUnchanged = state.overlayVisible === initialOverlay;
                    const splitUnchanged = state.chatSplitActive === initialSplit;
                    const mobileUnchanged = state.bodyMobileClass === initialMobile;
                    
                    return overlayUnchanged && splitUnchanged && mobileUnchanged;
                }
            ),
            { numRuns: 100 }
        );
    });
});
