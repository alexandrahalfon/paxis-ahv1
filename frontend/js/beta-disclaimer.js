/**
 * Paxis Beta Disclaimer System
 * 
 * Self-contained module for beta disclaimers and terms.
 * All text is centralized here for easy editing.
 * 
 * Usage:
 *   - Auto-shows on first visit (any page that includes this script)
 *   - Call BetaDisclaimer.showForRegistration() during account creation
 *   - Call BetaDisclaimer.reset() to clear acceptance (testing)
 */

const BetaDisclaimer = (() => {
    const STORAGE_KEY = 'exueed_beta_accepted';
    const VERSION = '1.1'; // bump this to re-show after major text changes — bumped for the Paxis rebrand + welcome copy

    // ─── ALL DISCLAIMER TEXT LIVES HERE ─────────────────────────
    // Edit these strings to update what users see.

    const BETA_HEADING = 'Welcome to Paxis';

    const WHAT_IS_PAXIS_HEADING = 'What is Paxis?';
    const WHAT_IS_PAXIS = `
        Paxis helps oncologists match patient cases to relevant clinical evidence —
        submit a patient profile and get back matched studies, trial eligibility,
        and literature-backed answers, all searchable through one AI-assisted
        research tool. Thanks for being one of our first users — you're helping
        shape where this goes next.
    `;

    const BETA_INTRO = `
        Paxis is currently in <strong>beta</strong>. Features are actively being
        developed and may change. We appreciate your feedback as we continue to
        improve the platform.
    `;

    const MEDICAL_DISCLAIMER_HEADING = 'Medical Disclaimer';
    const MEDICAL_DISCLAIMER = `
        Paxis is an AI-powered medical literature search tool designed to assist
        healthcare professionals in reviewing published oncology research.
        It is <strong>not</strong> a substitute for professional clinical judgment,
        medical advice, diagnosis, or treatment.
        <br><br>
        All results, analyses, and recommendations are derived from publicly
        available literature and AI-generated summaries. They may contain
        inaccuracies, omissions, or outdated information and should be
        <strong>independently verified</strong> by qualified healthcare
        professionals before any clinical decision-making.
    `;

    const LEGAL_DISCLAIMER_HEADING = 'Terms of Use';
    const LEGAL_DISCLAIMER = `
        By using Paxis you acknowledge and agree that:
        <ul>
            <li>This is a beta product. The system may contain errors or produce inaccurate results.</li>
            <li>Paxis does not provide medical advice. All information is for research and educational purposes only.</li>
            <li>No warranty is made regarding the completeness, accuracy, or reliability of any output.</li>
            <li>You will not hold Paxis, its developers, or affiliates liable for any decisions made based on the platform's output.</li>
            <li>Patient descriptions entered are processed solely to generate search results and are not stored beyond the session unless you explicitly save them.</li>
            <li>You are responsible for complying with all applicable regulations, including patient privacy laws, when using this tool.</li>
        </ul>
    `;

    const ACCEPT_BUTTON_TEXT = 'I understand and agree';
    const REGISTRATION_HEADING = 'Before you create an account';
    // ─── END EDITABLE TEXT ──────────────────────────────────────

    function injectStyles() {
        if (document.getElementById('beta-disclaimer-styles')) return;
        const style = document.createElement('style');
        style.id = 'beta-disclaimer-styles';
        style.textContent = `
            .beta-disclaimer-overlay {
                position: fixed;
                inset: 0;
                background: rgba(0, 0, 0, 0.6);
                backdrop-filter: blur(4px);
                z-index: 10000;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 1rem;
                animation: betaFadeIn 0.25s ease-out;
            }
            @keyframes betaFadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            .beta-disclaimer-modal {
                background: white;
                border-radius: 1rem;
                max-width: 560px;
                width: 100%;
                max-height: 85vh;
                overflow-y: auto;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                animation: betaSlideUp 0.3s ease-out;
            }
            @keyframes betaSlideUp {
                from { transform: translateY(20px); opacity: 0; }
                to { transform: translateY(0); opacity: 1; }
            }
            .beta-disclaimer-modal .bdm-header {
                padding: 1.75rem 2rem 0;
                text-align: center;
            }
            .beta-disclaimer-modal .bdm-header h2 {
                margin: 0 0 0.25rem;
                font-size: 1.5rem;
                color: var(--gray-900, #111827);
            }
            .beta-disclaimer-modal .bdm-badge {
                display: inline-block;
                background: #fef3c7;
                color: #92400e;
                font-size: 0.75rem;
                font-weight: 700;
                padding: 0.2rem 0.6rem;
                border-radius: 9999px;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 0.75rem;
            }
            .beta-disclaimer-modal .bdm-body {
                padding: 1.25rem 2rem;
                font-size: 0.9rem;
                line-height: 1.65;
                color: var(--gray-700, #374151);
            }
            .beta-disclaimer-modal .bdm-body p {
                margin: 0 0 1rem;
            }
            .beta-disclaimer-modal .bdm-section-title {
                font-weight: 700;
                font-size: 0.95rem;
                color: var(--gray-900, #111827);
                margin: 1.25rem 0 0.5rem;
            }
            .beta-disclaimer-modal .bdm-body ul {
                margin: 0.5rem 0 1rem 1.25rem;
                padding: 0;
            }
            .beta-disclaimer-modal .bdm-body li {
                margin-bottom: 0.4rem;
            }
            .beta-disclaimer-modal .bdm-footer {
                padding: 0 2rem 1.75rem;
                text-align: center;
            }
            .beta-disclaimer-modal .bdm-accept-btn {
                display: inline-block;
                background: var(--primary, #3b82f6);
                color: white;
                font-weight: 600;
                font-size: 1rem;
                padding: 0.75rem 2rem;
                border: none;
                border-radius: 0.5rem;
                cursor: pointer;
                transition: background 0.15s;
                width: 100%;
                max-width: 320px;
            }
            .beta-disclaimer-modal .bdm-accept-btn:hover {
                background: var(--primary-dark, #2563eb);
            }
        `;
        document.head.appendChild(style);
    }

    function buildModalHTML(heading, showCheckbox) {
        const checkboxHTML = showCheckbox ? `
            <label style="display: flex; align-items: center; gap: 0.5rem; margin-top: 1rem; font-size: 0.85rem; cursor: pointer;">
                <input type="checkbox" id="bdmAgreeCheckbox">
                <span>I have read and agree to the terms above</span>
            </label>
        ` : '';

        return `
            <div class="beta-disclaimer-overlay" id="betaDisclaimerOverlay">
                <div class="beta-disclaimer-modal" role="dialog" aria-modal="true" aria-labelledby="bdmHeading">
                    <div class="bdm-header">
                        <span class="bdm-badge">Beta</span>
                        <h2 id="bdmHeading">${heading}</h2>
                    </div>
                    <div class="bdm-body">
                        <div class="bdm-section-title" style="margin-top: 0;">${WHAT_IS_PAXIS_HEADING}</div>
                        <p>${WHAT_IS_PAXIS}</p>

                        <p>${BETA_INTRO}</p>

                        <div class="bdm-section-title">${MEDICAL_DISCLAIMER_HEADING}</div>
                        <p>${MEDICAL_DISCLAIMER}</p>

                        <div class="bdm-section-title">${LEGAL_DISCLAIMER_HEADING}</div>
                        <p>${LEGAL_DISCLAIMER}</p>

                        ${checkboxHTML}
                    </div>
                    <div class="bdm-footer">
                        <button class="bdm-accept-btn" id="bdmAcceptBtn"
                            ${showCheckbox ? 'disabled' : ''}>
                            ${ACCEPT_BUTTON_TEXT}
                        </button>
                    </div>
                </div>
            </div>
        `;
    }

    function showModal(heading, onAccept, requireCheckbox) {
        injectStyles();

        if (document.getElementById('betaDisclaimerOverlay')) return;

        const wrapper = document.createElement('div');
        wrapper.innerHTML = buildModalHTML(heading, requireCheckbox);
        document.body.appendChild(wrapper.firstElementChild);

        const overlay = document.getElementById('betaDisclaimerOverlay');
        const acceptBtn = document.getElementById('bdmAcceptBtn');
        const checkbox = document.getElementById('bdmAgreeCheckbox');

        if (checkbox) {
            checkbox.addEventListener('change', () => {
                acceptBtn.disabled = !checkbox.checked;
            });
        }

        acceptBtn.addEventListener('click', () => {
            overlay.style.animation = 'betaFadeIn 0.2s ease-out reverse';
            setTimeout(() => {
                overlay.remove();
                if (onAccept) onAccept();
            }, 180);
        });

        document.addEventListener('keydown', function escHandler(e) {
            if (e.key === 'Escape' && !requireCheckbox) {
                overlay.remove();
                if (onAccept) onAccept();
                document.removeEventListener('keydown', escHandler);
            }
        });
    }

    function hasAccepted() {
        try {
            const stored = localStorage.getItem(STORAGE_KEY);
            if (!stored) return false;
            const parsed = JSON.parse(stored);
            return parsed.version === VERSION;
        } catch {
            return false;
        }
    }

    function markAccepted() {
        localStorage.setItem(STORAGE_KEY, JSON.stringify({
            version: VERSION,
            accepted_at: new Date().toISOString()
        }));
    }

    return {
        /**
         * Show the first-visit beta disclaimer if not already accepted.
         * Call this on page load for any page that should gate usage.
         */
        showIfNeeded() {
            if (hasAccepted()) return;
            showModal(BETA_HEADING, () => { markAccepted(); }, false);
        },

        /**
         * Show the disclaimer during account registration.
         * Returns a Promise that resolves when user accepts.
         * If already accepted (e.g. from first-visit popup), resolves immediately.
         */
        showForRegistration() {
            if (hasAccepted()) return Promise.resolve();
            return new Promise((resolve) => {
                showModal(REGISTRATION_HEADING, () => {
                    markAccepted();
                    resolve();
                }, true);
            });
        },

        /**
         * Reset acceptance (for testing or version bumps).
         */
        reset() {
            localStorage.removeItem(STORAGE_KEY);
        }
    };
})();

// Auto-show on first visit when this script loads
document.addEventListener('DOMContentLoaded', () => {
    BetaDisclaimer.showIfNeeded();
});
