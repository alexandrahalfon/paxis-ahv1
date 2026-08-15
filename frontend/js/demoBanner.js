/**
 * Demo Banner Component
 * Dismissible announcement bar at the very top of the site linking to the
 * product walkthrough demo (demo_journey.html). Shown by default on every
 * page that calls DemoBanner.init(); once a visitor dismisses it, that
 * choice is remembered (localStorage) and it stays hidden on this browser.
 *
 * Must be initialized BEFORE SideNav.init() on the same page, so the
 * sidebar's header-height measurement already accounts for the banner.
 */
const DemoBanner = {
    STORAGE_KEY: 'paxis_demo_banner_dismissed',

    init() {
        if (localStorage.getItem(this.STORAGE_KEY) === '1') return;
        this.injectStyles();
        this.injectHTML();
        this.bindEvents();
    },

    injectStyles() {
        if (document.getElementById('demoBannerStyles')) return;
        const style = document.createElement('style');
        style.id = 'demoBannerStyles';
        style.textContent = `
            #demoBanner {
                position: relative;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 0.6rem;
                background: var(--primary, #e11d48);
                color: #fff;
                font-size: 0.85rem;
                font-weight: 600;
                padding: 0.6rem 2.75rem;
                text-align: center;
                line-height: 1.4;
            }
            #demoBanner .demo-banner-link {
                color: #fff;
                font-weight: 800;
                text-decoration: underline;
                text-underline-offset: 2px;
                white-space: nowrap;
            }
            #demoBanner .demo-banner-link:hover { opacity: 0.9; }
            #demoBannerClose {
                position: absolute;
                right: 10px;
                top: 50%;
                transform: translateY(-50%);
                background: none;
                border: none;
                color: rgba(255,255,255,0.85);
                cursor: pointer;
                font-size: 1.25rem;
                line-height: 1;
                padding: 4px 8px;
                border-radius: 6px;
            }
            #demoBannerClose:hover { background: rgba(255,255,255,0.15); color: #fff; }
        `;
        document.head.appendChild(style);
    },

    injectHTML() {
        if (document.getElementById('demoBanner')) return;
        const el = document.createElement('div');
        el.id = 'demoBanner';
        el.innerHTML = `
            <span>New here? See Paxis handle a full patient case, start to finish.</span>
            <a class="demo-banner-link" href="demo.html" target="_blank" rel="noopener">Watch the demo</a>
            <button id="demoBannerClose" aria-label="Dismiss">&times;</button>
        `;
        document.body.insertBefore(el, document.body.firstChild);
    },

    bindEvents() {
        const closeBtn = document.getElementById('demoBannerClose');
        if (!closeBtn) return;
        closeBtn.addEventListener('click', () => {
            localStorage.setItem(this.STORAGE_KEY, '1');
            const banner = document.getElementById('demoBanner');
            if (banner) banner.remove();
            // Sidebar height/position is derived from the header's rendered
            // position (see sideNav.js adjustHeightOnScroll). Removing the
            // banner changes that position, so nudge the existing resize
            // listener to recompute it rather than duplicating that logic here.
            window.dispatchEvent(new Event('resize'));
        });
    },
};
