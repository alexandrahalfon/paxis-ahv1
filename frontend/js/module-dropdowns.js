/**
 * Module Dropdown Menus
 *
 * Adds hover-activated feature lists under each module navigation button.
 * Self-contained: injects styles and wraps existing buttons on DOMContentLoaded.
 * Edit MODULE_FEATURES to change the text shown under each module.
 */

(function () {
    // ─── FEATURE LISTS PER MODULE ──────────────────────────────
    // Keys match the href of the module buttons.

    const MODULE_FEATURES = {
        'patient-matching.html': {
            heading: 'Query Clinical Evidence',
            features: [
                'Find studies matching your patient',
                'Match by cancer type, stage, and biomarkers',
                'Semantic similarity scoring',
                'Structured or free-text input'
            ]
        },
        'treatment-comparison.html': {
            heading: 'Evaluate Treatment Options',
            features: [
                'Side-by-side evidence analysis',
                'Visual charts and data comparison',
                'Guideline alignment',
                'Cancer prognostic insights'
            ]
        },
        'study-comparison.html': {
            heading: 'Review Studies',
            features: [
                'Compare multiple studies head-to-head',
                'Use guideline as base study',
                'Study selection and collection',
                'Key endpoint comparison'
            ]
        },
        'analytics.html': {
            heading: 'Advanced Analytics',
            features: [
                'Survival and outcome statistics',
                'Dose distribution analysis',
                'Technique frequency across studies',
                'Meta-analysis and forest plots'
            ]
        },
        'trial-search.html': {
            heading: 'Clinical Trial Finder',
            features: [
                'Search ClinicalTrials.gov',
                'Filter by phase, status, and location',
                'Match trials to patient profile',
                'Active recruiting trials'
            ]
        }
    };

    // ─── STYLES ────────────────────────────────────────────────

    function injectStyles() {
        if (document.getElementById('module-dropdown-styles')) return;
        const style = document.createElement('style');
        style.id = 'module-dropdown-styles';
        style.textContent = `
            .module-dropdown-wrap {
                position: relative;
                display: inline-flex;
            }
            .module-dropdown {
                display: none;
                position: absolute;
                top: calc(100% + 6px);
                left: 50%;
                transform: translateX(-50%);
                background: #fff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                box-shadow: 0 4px 16px rgba(0,0,0,0.10);
                padding: 0.65rem 0.85rem;
                min-width: 210px;
                z-index: 100;
                white-space: nowrap;
            }
            .module-dropdown::before {
                content: '';
                position: absolute;
                top: -6px;
                left: 50%;
                transform: translateX(-50%);
                width: 10px;
                height: 10px;
                background: #fff;
                border-left: 1px solid #e2e8f0;
                border-top: 1px solid #e2e8f0;
                rotate: 45deg;
            }
            .module-dropdown-wrap:hover .module-dropdown {
                display: block;
            }
            .module-dropdown-heading {
                font-size: 0.7rem;
                font-weight: 700;
                color: #94a3b8;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                margin-bottom: 0.4rem;
            }
            .module-dropdown ul {
                list-style: none;
                margin: 0;
                padding: 0;
            }
            .module-dropdown li {
                font-size: 0.78rem;
                color: #475569;
                padding: 0.2rem 0;
                line-height: 1.4;
            }
            .module-dropdown li::before {
                content: '+';
                color: #f59e0b;
                font-weight: 700;
                margin-right: 0.4rem;
            }
        `;
        document.head.appendChild(style);
    }

    // ─── WRAP BUTTONS ──────────────────────────────────────────

    function init() {
        const containers = document.querySelectorAll('.feature-nav, .hero-actions');
        if (containers.length === 0) return;

        injectStyles();

        containers.forEach(container => {
            const links = container.querySelectorAll('a[href]');
            links.forEach(link => {
                const href = link.getAttribute('href');
                const config = MODULE_FEATURES[href];
                if (!config) return;

                // Wrap the link
                const wrap = document.createElement('div');
                wrap.className = 'module-dropdown-wrap';
                link.parentNode.insertBefore(wrap, link);
                wrap.appendChild(link);

                // Build dropdown
                const dropdown = document.createElement('div');
                dropdown.className = 'module-dropdown';
                dropdown.innerHTML =
                    '<div class="module-dropdown-heading">' + config.heading + '</div>' +
                    '<ul>' +
                    config.features.map(f => '<li>' + f + '</li>').join('') +
                    '</ul>';
                wrap.appendChild(dropdown);
            });
        });
    }

    document.addEventListener('DOMContentLoaded', init);
})();
