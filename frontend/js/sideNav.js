/**
 * Side Navigation Component
 * Reusable side navigation panel for all pages
 */

/**
 * Link sets. The clinician set is the original nav; the patient set was
 * added with the patient portal. Both render through the same component
 * so the two interfaces share one sidebar implementation rather than
 * maintaining a second copy that drifts.
 *
 * icon values are raw SVG path content.
 */
const SIDE_NAV_SETS = {
    clinician: null,   // null = use the built-in clinician markup below

    patient: {
        home: 'patient-home.html',
        sections: [
            {
                title: 'Your care',
                links: [
                    { id: 'home', href: 'patient-home.html', label: 'Home',
                      icon: '<path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>' },
                    { id: 'ask', href: 'patient-qa.html', label: 'Ask a Question',
                      icon: '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>' },
                    { id: 'care-team', href: 'patient-connect.html', label: 'My Care Team',
                      icon: '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/>' },
                ],
            },
            {
                title: 'My health',
                links: [
                    { id: 'dashboard', href: 'patient-dashboard.html', label: 'My Health',
                      icon: '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>' },
                    { id: 'documents', href: 'patient-documents.html', label: 'My Documents',
                      icon: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="8" y1="13" x2="16" y2="13"/><line x1="8" y1="17" x2="13" y2="17"/>' },
                    { id: 'timeline', href: 'patient-timeline.html', label: 'Timeline',
                      icon: '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>' },
                ],
            },
            {
                title: 'Tools',
                links: [
                    { id: 'medication', href: 'patient-medication.html', label: 'About My Medication',
                      icon: '<path d="M10.5 20.5 3.5 13.5a5 5 0 0 1 7-7l7 7a5 5 0 0 1-7 7z"/><line x1="8.5" y1="8.5" x2="15.5" y2="15.5"/>' },
                    { id: 'report', href: 'patient-report.html', label: 'Explain My Report',
                      icon: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="8" y1="13" x2="16" y2="13"/><line x1="8" y1="17" x2="13" y2="17"/>' },
                    { id: 'symptoms', href: 'patient-symptoms.html', label: 'Symptom Diary',
                      icon: '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>' },
                ],
            },
        ],
    },
};

const SideNav = {
    /**
     * Initialize the side navigation on the current page
     * @param {string} activePage - The current page identifier (e.g., 'home', 'upload', 'analytics')
     * @param {string} [variant] - 'clinician' (default) or 'patient'
     */
    init(activePage, variant) {
        this.variant = variant || 'clinician';
        this.injectHTML(activePage);
        this.injectStyles();
        this.bindEvents();
        this.restoreState();
        if (this.variant === 'clinician') this.loadInboxCount();
    },

    /** Build the markup for a configured link set (patient interface). */
    buildFromSet(set, activePage) {
        const sections = set.sections.map(sec => `
                <div class="side-nav-section">
                    <div class="side-nav-section-title">${sec.title}</div>
                    ${sec.links.map(l => `
                    <a href="${l.href}" class="side-nav-link ${activePage === l.id ? 'active' : ''}">
                        <svg class="side-nav-icon-svg" viewBox="0 0 24 24" fill="none"
                             stroke="currentColor" stroke-width="2">${l.icon}</svg>
                        <span class="side-nav-text">${l.label}</span>
                    </a>`).join('')}
                </div>`).join('');

        return `
        <nav class="side-nav-panel" id="sideNavPanel">
            <div class="side-nav-header">
                <a href="${set.home}" class="side-nav-logo">
                    <img src="assets/paxis-mark.png" alt="Paxis" class="side-nav-logo-img">
                </a>
                <button class="side-nav-toggle" id="sideNavToggle" title="Toggle navigation">
                    <svg class="toggle-icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                        <path d="M15 18l-6-6 6-6"/>
                    </svg>
                </button>
            </div>
            <div class="side-nav-content">${sections}</div>
            <div class="side-nav-footer">
                <div class="side-nav-user" id="sideNavUser" style="display:none;">
                    <div class="side-nav-user-avatar" id="sideNavUserAvatar"></div>
                    <span class="side-nav-text" id="sideNavUserEmail"></span>
                </div>
            </div>
        </nav>`;
    },

    /**
     * Show a count of patient questions waiting on this clinician.
     * Fails silently: a badge is not worth breaking navigation over,
     * and signed-out or patient accounts simply get no badge.
     */
    async loadInboxCount() {
        try {
            const token = localStorage.getItem('exueed_token');
            if (!token) return;
            const base = (typeof CONFIG !== 'undefined' && CONFIG.API_BASE) ? CONFIG.API_BASE : '/api';
            const [escRes, reqRes] = await Promise.all([
                fetch(`${base}/portal/clinician/escalations`, { headers: { 'Authorization': `Bearer ${token}` } }),
                fetch(`${base}/portal/clinician/link-requests`, { headers: { 'Authorization': `Bearer ${token}` } })
            ]);
            if (!escRes.ok || !reqRes.ok) return;
            const esc = await escRes.json();
            const req = await reqRes.json();
            const total = (esc.escalations || []).length + (req.requests || []).length;
            const badge = document.getElementById('sideNavInboxBadge');
            if (badge && total > 0) {
                badge.textContent = total > 99 ? '99+' : String(total);
                badge.style.display = '';
            }
        } catch (_) { /* non-fatal */ }
    },

    /**
     * Inject the side nav HTML into the page
     */
    injectHTML(activePage) {
        const set = SIDE_NAV_SETS[this.variant];
        if (set) {
            const wrapper = document.createElement('div');
            wrapper.innerHTML = this.buildFromSet(set, activePage);
            document.body.insertBefore(wrapper.firstElementChild, document.body.firstChild);
            return;
        }
        const navHTML = `
        <nav class="side-nav-panel" id="sideNavPanel">
            <div class="side-nav-header">
                <a href="index.html" class="side-nav-logo">
                    <img src="assets/paxis-mark.png" alt="Paxis" class="side-nav-logo-img">
                </a>
                <button class="side-nav-toggle" id="sideNavToggle" title="Toggle navigation">
                    <svg class="toggle-icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                        <path d="M15 18l-6-6 6-6"/>
                    </svg>
                </button>
            </div>
            <div class="side-nav-content">
                <div class="side-nav-section">
                    <div class="side-nav-section-title">Pages</div>
                    <a href="index.html" class="side-nav-link ${activePage === 'home' ? 'active' : ''}">
                        <svg class="side-nav-icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
                            <polyline points="9 22 9 12 15 12 15 22"/>
                        </svg>
                        <span class="side-nav-text">Home</span>
                    </a>
                    <a href="my-saves.html" class="side-nav-link ${activePage === 'saves' ? 'active' : ''}">
                        <svg class="side-nav-icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/>
                        </svg>
                        <span class="side-nav-text">My Collections</span>
                    </a>
                    <a href="patient-inbox.html" class="side-nav-link ${activePage === 'inbox' ? 'active' : ''}">
                        <svg class="side-nav-icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M4 4h16v12H5.17L4 17.17z"/>
                        </svg>
                        <span class="side-nav-text">Patient Inbox</span>
                        <span class="side-nav-badge" id="sideNavInboxBadge" style="display:none;"></span>
                    </a>
                    <a href="upload.html" class="side-nav-link ${activePage === 'upload' ? 'active' : ''}">
                        <svg class="side-nav-icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                            <polyline points="17 8 12 3 7 8"/>
                            <line x1="12" y1="3" x2="12" y2="15"/>
                        </svg>
                        <span class="side-nav-text">Upload</span>
                    </a>
                    <a href="trial-search.html" class="side-nav-link ${activePage === 'trials' ? 'active' : ''}">
                        <svg class="side-nav-icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="11" cy="11" r="8"/>
                            <line x1="21" y1="21" x2="16.65" y2="16.65"/>
                        </svg>
                        <span class="side-nav-text">Trial Finder</span>
                    </a>
                </div>
                <div class="side-nav-section">
                    <div class="side-nav-section-title">Tools</div>
                    <a href="patient-matching.html" class="side-nav-link ${activePage === 'patient-matching' ? 'active' : ''}">
                        <svg class="side-nav-icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
                            <circle cx="12" cy="7" r="4"/>
                        </svg>
                        <span class="side-nav-text">Patient Matching</span>
                    </a>
                    <a href="treatment-comparison.html" class="side-nav-link ${activePage === 'treatment-comparison' ? 'active' : ''}">
                        <svg class="side-nav-icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <line x1="18" y1="20" x2="18" y2="10"/>
                            <line x1="12" y1="20" x2="12" y2="4"/>
                            <line x1="6" y1="20" x2="6" y2="14"/>
                        </svg>
                        <span class="side-nav-text">Treatment Comparison</span>
                    </a>
                    <a href="study-comparison.html" class="side-nav-link ${activePage === 'study-comparison' ? 'active' : ''}">
                        <svg class="side-nav-icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="3" y="3" width="7" height="7"/>
                            <rect x="14" y="3" width="7" height="7"/>
                            <rect x="14" y="14" width="7" height="7"/>
                            <rect x="3" y="14" width="7" height="7"/>
                        </svg>
                        <span class="side-nav-text">Review Studies</span>
                    </a>
                    <a href="analytics.html" class="side-nav-link ${activePage === 'analytics' ? 'active' : ''}">
                        <svg class="side-nav-icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M21.21 15.89A10 10 0 1 1 8 2.83"/>
                            <path d="M22 12A10 10 0 0 0 12 2v10z"/>
                        </svg>
                        <span class="side-nav-text">Analytics</span>
                    </a>
                </div>
            </div>
            <div class="side-nav-footer">
                <div class="side-nav-patient-wrap">
                    <a href="patient-qa.html" class="side-nav-link side-nav-patient-btn ${activePage === 'patient-qa' ? 'active' : ''}">
                        <svg class="side-nav-icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>
                        </svg>
                        <span class="side-nav-text">For Patients</span>
                    </a>
                </div>
                <div class="side-nav-user" id="sideNavUser" style="display: none;">
                    <div class="side-nav-user-avatar" id="sideNavUserAvatar"></div>
                    <span class="side-nav-text side-nav-user-email" id="sideNavUserEmail"></span>
                </div>
            </div>
        </nav>`;

        // Insert at the beginning of body
        document.body.insertAdjacentHTML('afterbegin', navHTML);
    },

    /**
     * Inject additional styles for the improved side nav
     */
    injectStyles() {
        const styleId = 'side-nav-injected-styles';
        if (document.getElementById(styleId)) return;

        const styles = document.createElement('style');
        styles.id = styleId;
        styles.textContent = `
            /* Patient Inbox unread badge */
            .side-nav-badge {
                margin-left: auto;
                background: var(--primary, #e11d48);
                color: #fff;
                font-size: 0.7rem;
                font-weight: 800;
                line-height: 1;
                padding: 0.2rem 0.4rem;
                border-radius: 999px;
                min-width: 18px;
                text-align: center;
            }
            /* Side Nav Logo */
            .side-nav-logo {
                display: flex;
                align-items: center;
                text-decoration: none;
            }
            .side-nav-logo-img {
                width: 32px;
                height: 32px;
                object-fit: contain;
                transition: transform 0.2s ease;
            }
            .side-nav-logo:hover .side-nav-logo-img {
                transform: scale(1.05);
            }
            .side-nav-panel.collapsed .side-nav-logo-img {
                width: 28px;
                height: 28px;
            }

            /* SVG Icons */
            .side-nav-icon-svg {
                width: 20px;
                height: 20px;
                flex-shrink: 0;
                stroke: currentColor;
                transition: stroke 0.15s ease;
            }
            .toggle-icon-svg {
                width: 13px;
                height: 13px;
                transition: transform 0.2s ease;
            }

            /* Footer with user info */
            .side-nav-footer {
                padding: 0.75rem;
                border-top: 1px solid var(--gray-200);
                margin-top: auto;
            }
            .side-nav-user {
                display: flex;
                align-items: center;
                gap: 0.75rem;
                padding: 0.5rem;
                border-radius: var(--radius-md);
                background: var(--gray-50);
            }
            .side-nav-user-avatar {
                width: 32px;
                height: 32px;
                border-radius: 50%;
                background: var(--primary);
                color: white;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 600;
                font-size: 0.8rem;
                flex-shrink: 0;
            }
            .side-nav-user-email {
                font-size: 0.75rem;
                color: var(--gray-600);
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }
            .side-nav-panel.collapsed .side-nav-user {
                justify-content: center;
                padding: 0.5rem;
            }
            .side-nav-panel.collapsed .side-nav-user-email {
                display: none;
            }

            /* For Patients button — teal accent to stand apart from clinical tools */
            .side-nav-patient-wrap {
                padding: 0.5rem 0.75rem 0.25rem;
            }
            .side-nav-patient-btn {
                background: #ecfdf5 !important;
                color: #065f46 !important;
                border-radius: 8px;
                border: 1px solid #a7f3d0 !important;
                font-weight: 700 !important;
            }
            .side-nav-patient-btn:hover {
                background: #d1fae5 !important;
                color: #047857 !important;
            }
            .side-nav-patient-btn.active {
                background: #059669 !important;
                color: white !important;
                border-color: #059669 !important;
            }
            .side-nav-panel.collapsed .side-nav-patient-wrap {
                padding: 0.5rem 0.5rem 0.25rem;
            }
        `;
        document.head.appendChild(styles);
    },

    /**
     * Bind event handlers
     */
    bindEvents() {
        const panel = document.getElementById('sideNavPanel');
        const toggle = document.getElementById('sideNavToggle');

        if (toggle && panel) {
            toggle.addEventListener('click', function handleToggleClick() {
                panel.classList.toggle('collapsed');
                localStorage.setItem('sideNavCollapsed', panel.classList.contains('collapsed'));
                SideNav.updateBodyPadding();
            });
        }

        // Update user info if logged in
        this.updateUserInfo();
        
        // Add scroll listener to dynamically adjust height
        this.adjustHeightOnScroll();
        window.addEventListener('scroll', () => this.adjustHeightOnScroll());
        window.addEventListener('resize', () => this.adjustHeightOnScroll());
    },

    /**
     * Adjust side nav height dynamically to avoid covering header/footer
     */
    adjustHeightOnScroll() {
        const panel = document.getElementById('sideNavPanel');
        const header = document.querySelector('.header');
        const footer = document.querySelector('.footer');
        
        if (!panel) return;

        // In fullscreen chat mode the header is hidden — sidebar should start at top: 0
        const headerIsHidden = !header || window.getComputedStyle(header).display === 'none';
        const headerRect = header ? header.getBoundingClientRect() : { bottom: 0 };
        const headerBottom = headerIsHidden ? 0 : Math.max(headerRect.bottom, 0);

        // Calculate top position (below header, or 0 when header is hidden)
        const topPosition = Math.max(headerBottom, 0);
        panel.style.top = topPosition + 'px';
        
        // Calculate available height
        let availableHeight = window.innerHeight - topPosition;
        
        // If footer is visible, reduce height to not cover it
        if (footer) {
            const footerRect = footer.getBoundingClientRect();
            if (footerRect.top < window.innerHeight) {
                const footerVisibleHeight = window.innerHeight - footerRect.top;
                availableHeight = availableHeight - footerVisibleHeight;
            }
        }
        
        panel.style.height = Math.max(availableHeight, 100) + 'px';
    },

    /**
     * Restore collapsed state from localStorage
     */
    restoreState() {
        const panel = document.getElementById('sideNavPanel');
        const isCollapsed = localStorage.getItem('sideNavCollapsed') === 'true';
        if (isCollapsed && panel) {
            panel.classList.add('collapsed');
        }
        this.updateBodyPadding();
    },

    /**
     * Update body padding based on nav state (fallback for browsers without :has())
     */
    updateBodyPadding() {
        const panel = document.getElementById('sideNavPanel');
        if (!panel) return;

        const isCollapsed = panel.classList.contains('collapsed');
        const paddingLeft = isCollapsed ? '56px' : '220px';

        // Check if :has() is supported
        let hasSupported = true;
        try {
            document.querySelector(':has(*)');
        } catch (e) {
            hasSupported = false;
        }

        if (!hasSupported) {
            // Only apply padding to sections, not header or footer
            const sections = document.querySelectorAll('.section');
            sections.forEach(s => s.style.paddingLeft = paddingLeft);
        }
    },

    /**
     * Update user info in the footer
     */
    async updateUserInfo() {
        const userSection = document.getElementById('sideNavUser');
        const avatarEl = document.getElementById('sideNavUserAvatar');
        const emailEl = document.getElementById('sideNavUserEmail');

        // Check for logged in user
        const token = localStorage.getItem('exueed_token');
        if (!token || !userSection || !avatarEl || !emailEl) return;

        try {
            const base = (typeof CONFIG !== 'undefined' && CONFIG.API_BASE) ? CONFIG.API_BASE : '/api';
            const res = await fetch(`${base}/auth/me`, { headers: { 'Authorization': `Bearer ${token}` } });
            if (!res.ok) return;
            const me = await res.json();
            const name = [me.first_name, me.last_name].filter(Boolean).join(' ');
            const display = name || me.email || '';
            if (!display) return;

            userSection.style.display = 'flex';
            avatarEl.textContent = display.charAt(0).toUpperCase();
            emailEl.textContent = display;
        } catch (e) {
            // Non-fatal — sidebar just won't show the user pill.
        }
    }
};
