/**
 * Utility functions for Paxis frontend
 */

/**
 * Escape HTML special characters to prevent XSS
 * @param {string} text - Text to escape
 * @returns {string} Escaped text
 */
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Convert markdown to HTML
 */
function markdownToHtml(text) {
    if (!text) return '';
    
    let html = text;
    
    // Split into lines for processing
    const lines = html.split('\n');
    const processedLines = [];
    let inList = false;
    let listItems = [];
    let listType = null; // 'ol' or 'ul'
    
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        
        // Headers
        if (line.startsWith('### ')) {
            if (inList) {
                processedLines.push(closeList(listItems, listType));
                listItems = [];
                inList = false;
            }
            processedLines.push('<h4>' + line.substring(4) + '</h4>');
            continue;
        }
        if (line.startsWith('## ')) {
            if (inList) {
                processedLines.push(closeList(listItems, listType));
                listItems = [];
                inList = false;
            }
            processedLines.push('<h3>' + line.substring(3) + '</h3>');
            continue;
        }
        if (line.startsWith('# ')) {
            if (inList) {
                processedLines.push(closeList(listItems, listType));
                listItems = [];
                inList = false;
            }
            processedLines.push('<h2>' + line.substring(2) + '</h2>');
            continue;
        }
        
        // Numbered lists
        const numberedMatch = line.match(/^(\d+)\.\s+(.+)$/);
        if (numberedMatch) {
            if (!inList || listType !== 'ol') {
                if (inList) {
                    processedLines.push(closeList(listItems, listType));
                }
                listItems = [];
                listType = 'ol';
                inList = true;
            }
            listItems.push('<li>' + processInlineMarkdown(numberedMatch[2]) + '</li>');
            continue;
        }
        
        // Bullet lists
        const bulletMatch = line.match(/^[-*]\s+(.+)$/);
        if (bulletMatch) {
            if (!inList || listType !== 'ul') {
                if (inList) {
                    processedLines.push(closeList(listItems, listType));
                }
                listItems = [];
                listType = 'ul';
                inList = true;
            }
            listItems.push('<li>' + processInlineMarkdown(bulletMatch[1]) + '</li>');
            continue;
        }
        
        // Empty line - close list if open
        if (line === '') {
            if (inList) {
                processedLines.push(closeList(listItems, listType));
                listItems = [];
                inList = false;
            }
            processedLines.push('');
            continue;
        }
        
        // Regular paragraph
        if (inList) {
            processedLines.push(closeList(listItems, listType));
            listItems = [];
            inList = false;
        }
        processedLines.push('<p>' + processInlineMarkdown(line) + '</p>');
    }
    
    // Close any open list
    if (inList) {
        processedLines.push(closeList(listItems, listType));
    }
    
    return processedLines.join('\n');
}

function closeList(items, type) {
    if (items.length === 0) return '';
    const tag = type === 'ol' ? 'ol' : 'ul';
    return '<' + tag + '>' + items.join('') + '</' + tag + '>';
}

function processInlineMarkdown(text) {
    // Bold (**text**)
    text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    // Italic (*text*)
    text = text.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    return text;
}

/**
 * Format a single source citation for display
 * @param {Object} source - Source object with title, author, citation, etc.
 * @returns {string} Formatted citation string
 */
function formatCitation(source) {
    if (!source) return 'Unknown source';
    
    // Get the title
    const title = source.title || '';
    
    // Get author string
    const authors = source.authors || source.author || '';
    const authorStr = Array.isArray(authors) ? authors.join(', ') : authors;
    
    // Get year
    const year = source.publication_date || source.year || '';
    
    // Build a clean citation: Title - Author (Year)
    let citation = '';
    
    if (title) {
        citation = `<strong>${title}</strong>`;
    }
    
    if (authorStr) {
        citation += citation ? ` - ${authorStr}` : authorStr;
    }
    
    if (year) {
        citation += ` (${year})`;
    }
    
    // Add patient count if available (useful for population sorting)
    if (source.number_of_patients) {
        citation += ` <span class="patient-count">[n=${source.number_of_patients}]</span>`;
    }
    
    // Add DOI if available
    if (source.doi) {
        citation += ` <a href="https://doi.org/${source.doi}" target="_blank" rel="noopener">[DOI]</a>`;
    }
    
    return citation || 'Unknown source';
}

/**
 * Sort sources based on user preference
 * @param {Array} sources - Array of source objects
 * @param {string} sortBy - Sort preference (relevance, population, date, citations, outcomes)
 * @returns {Array} Sorted sources
 */
function sortSources(sources, sortBy = 'relevance') {
    if (!sources || sources.length === 0) return sources;
    
    const sorted = [...sources]; // Don't mutate original
    
    switch (sortBy) {
        case 'date':
            // Sort by year, newest first
            sorted.sort((a, b) => {
                const yearA = parseInt(a.year) || 0;
                const yearB = parseInt(b.year) || 0;
                return yearB - yearA; // Descending (newest first)
            });
            break;
            
        case 'citations':
            // Sort by citation count if available
            sorted.sort((a, b) => {
                const citesA = parseInt(a.citation_count) || 0;
                const citesB = parseInt(b.citation_count) || 0;
                return citesB - citesA; // Descending
            });
            break;
            
        case 'population':
            // Sort by patient count if available
            // Check if any sources have population data
            const hasPopulationData = sources.some(s => 
                s.number_of_patients || s.patient_count || s.n_patients
            );
            
            if (!hasPopulationData) {
                console.log('[sortSources] No population data available - falling back to relevance sort');
                // Fall back to relevance sort
                sorted.sort((a, b) => {
                    const scoreA = parseFloat(a.score) || 0;
                    const scoreB = parseFloat(b.score) || 0;
                    return scoreB - scoreA;
                });
            } else {
                sorted.sort((a, b) => {
                    const popA = parseInt(a.number_of_patients || a.patient_count || a.n_patients) || 0;
                    const popB = parseInt(b.number_of_patients || b.patient_count || b.n_patients) || 0;
                    return popB - popA; // Descending (high to low)
                });
            }
            break;
            
        case 'relevance':
        default:
            // Sort by score (already sorted by backend, but ensure it)
            sorted.sort((a, b) => {
                const scoreA = parseFloat(a.score) || 0;
                const scoreB = parseFloat(b.score) || 0;
                return scoreB - scoreA; // Descending
            });
            break;
    }
    
    return sorted;
}

/**
 * Get current sort preference from preferences manager or localStorage
 * @returns {string} Sort preference
 */
function getCurrentSortPreference() {
    // Try to get from preferences manager - use collectFilters directly to get sort_by
    // even when filters_active is false
    if (window.preferencesManager) {
        // First try getActivePreferences (when filters are active)
        const activePrefs = window.preferencesManager.getActivePreferences();
        if (activePrefs && activePrefs.sort_by) {
            console.log('[getCurrentSortPreference] Got from active preferences:', activePrefs.sort_by);
            return activePrefs.sort_by;
        }
        
        // If filters not active, still get sort_by from collectFilters
        if (typeof window.preferencesManager.collectFilters === 'function') {
            const allFilters = window.preferencesManager.collectFilters();
            if (allFilters && allFilters.sort_by) {
                console.log('[getCurrentSortPreference] Got from collectFilters:', allFilters.sort_by);
                return allFilters.sort_by;
            }
        }
    }
    
    // Fallback to localStorage
    const savedPrefs = localStorage.getItem('userPreferences');
    if (savedPrefs) {
        try {
            const prefs = JSON.parse(savedPrefs);
            if (prefs.sort_by) {
                console.log('[getCurrentSortPreference] Got from localStorage:', prefs.sort_by);
                return prefs.sort_by;
            }
        } catch (e) {
            console.error('Error parsing preferences:', e);
        }
    }
    
    console.log('[getCurrentSortPreference] Using default: relevance');
    return 'relevance';
}

/**
 * Format source list for display with deduplication and study details button
 * @param {Array} sources - Array of source objects
 * @param {Object} options - Optional: { preserveCitationOrder: true } to keep order matching inline [1], [2] citations
 */
function formatSources(sources, options = {}) {
    if (!sources || sources.length === 0) return '';
    
    let sourcesToShow;
    if (options.preserveCitationOrder) {
        // Keep original order so [1], [2] in the answer match the reference list
        sourcesToShow = sources;
    } else {
        // Get sort preference and apply sorting
        const sortBy = getCurrentSortPreference();
        console.log(`[formatSources] Sorting ${sources.length} sources by: ${sortBy}`);
        sourcesToShow = sortSources(sources, sortBy);
    }
    
    // Debug: log first few sources with their sort key
    const sortBy = getCurrentSortPreference();
    if (!options.preserveCitationOrder && sortBy === 'date') {
        console.log('[formatSources] First 3 sources by year:', 
            sourcesToShow.slice(0, 3).map(s => ({ title: s.title?.substring(0, 30), year: s.year }))
        );
    } else if (!options.preserveCitationOrder && sortBy === 'population') {
        console.log('[formatSources] First 3 sources by population:', 
            sourcesToShow.slice(0, 3).map(s => ({ 
                title: s.title?.substring(0, 30), 
                number_of_patients: s.number_of_patients 
            }))
        );
    } else if (!options.preserveCitationOrder && sortBy === 'citations') {
        console.log('[formatSources] First 3 sources by citations:', 
            sourcesToShow.slice(0, 3).map(s => ({ 
                title: s.title?.substring(0, 30), 
                citation_count: s.citation_count 
            }))
        );
    }
    
    // Deduplicate sources by DOI or title (skip when preserving citation order so [1],[2] match)
    const seen = new Set();
    const uniqueSources = options.preserveCitationOrder ? sourcesToShow : sourcesToShow.filter(source => {
        const key = source.doi || source.title || JSON.stringify(source);
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
    });
    
    return uniqueSources.map((source, index) => {
        const citation = formatCitation(source);
        
        // Extract identifiers for study details lookup
        const doc_id = source.doc_id || '';
        const pmid = source.pmid || '';
        const doi = source.doi || '';
        const title = source.title || '';
        
        // Create study details button if we have an identifier
        let studyDetailsBtn = '';
        if (doc_id || pmid || doi) {
            // Use data attributes to store the source info - safely escape values
            const escapedDocId = (doc_id || '').replace(/'/g, "\\'").replace(/"/g, '&quot;');
            const escapedPmid = (pmid || '').replace(/'/g, "\\'").replace(/"/g, '&quot;');
            const escapedDoi = (doi || '').replace(/'/g, "\\'").replace(/"/g, '&quot;');
            const escapedTitle = (title || '').replace(/'/g, "\\'").replace(/"/g, '&quot;');
            
            studyDetailsBtn = `
                <button class="study-details-btn" 
                        onclick="openStudyDetailsFromSource({doc_id:'${escapedDocId}',pmid:'${escapedPmid}',doi:'${escapedDoi}',title:'${escapedTitle}'})" 
                        title="View detailed study information">
                    Study Details
                </button>
            `;
        }
        
        // Patient-match badge — patient_match_scorer output (0–100):
        // weighted overlap of the patient's ClinicalProfile axes
        // against the study's doc_level_* metadata. This is the
        // per-document "how well does this study's enrolled
        // population fit my patient" number. Only shown when
        // populated; non-patient queries leave this null and the
        // badge does not render. Rendered next to the Study Details
        // button so each citation row carries its own match score.
        //
        // Bucketed for display — the underlying scorer has fake
        // precision (62% vs 65% are computationally distinguishable
        // but practically indistinguishable). Show a qualitative
        // tier and put the exact value in the tooltip for power
        // users / audit.
        //
        // Tiers:
        //   Strong  ≥ 70%   (deep blue)
        //   Moderate 50–69% (light blue)
        //   Weak    35–49%  (grey)
        //   Limited < 35%   (muted red — wrong-cancer territory,
        //                    lines up with HARD_CANCER_TYPE_CAP=35
        //                    in the scorer)
        //
        // The cross-encoder per-chunk relevance_score is NOT shown
        // per source — it lives at the top of the response as a
        // single "Evidence relevance" badge representing the best
        // chunk-to-query relevance across all cited sources. See
        // frontend/index.html (search for "Evidence relevance").
        let patientMatchBadge = '';
        if (source.patient_match_score != null) {
            const score = Math.round(source.patient_match_score);
            let cls, label;
            if (score >= 70)      { cls = 'patient-match-strong';   label = 'Strong match'; }
            else if (score >= 50) { cls = 'patient-match-moderate'; label = 'Moderate match'; }
            else if (score >= 35) { cls = 'patient-match-weak';     label = 'Weak match'; }
            else                  { cls = 'patient-match-limited';  label = 'Limited match'; }
            // Include the exact % alongside the qualitative tier so readers
            // don't have to hover to see the number. The Find Trials UI
            // shows the % directly; the chat UI used to hide it in a
            // tooltip, which read as "no score" to clinicians at a glance.
            const tooltip = `Patient–study match: ${score}% (weighted overlap of patient profile against study population)`;
            patientMatchBadge = `<span class="patient-match-badge ${cls}" title="${tooltip}">${label} · ${score}%</span>`;
        }

        // Evidence-type badge — guideline / landmark trial / trial.
        // Populated by evidence_classifier in the two-track retrieval
        // path. Helps the reader distinguish a guideline reference
        // from a patient-specific trial result.
        let evidenceTypeBadge = '';
        if (source.evidence_type === 'guideline') {
            evidenceTypeBadge = `<span class="evidence-type-badge evidence-type-guideline" title="Clinical practice guideline">Guideline</span>`;
        } else if (source.evidence_type === 'landmark_trial') {
            evidenceTypeBadge = `<span class="evidence-type-badge evidence-type-landmark" title="Landmark clinical trial">Landmark</span>`;
        }

        // Source type badge (Task 9)
        let sourceTypeBadge = '';
        if (source.source_type === 'pubmed') {
            sourceTypeBadge = `<span class="source-type-badge source-type-pubmed">PubMed</span>`;
        } else if (source.source_type === 'clinicaltrials') {
            sourceTypeBadge = `<span class="source-type-badge source-type-clinicaltrials">ClinicalTrials.gov</span>`;
        }

        // Order in the citation row:
        //   "1. Author (2020) Title    [Match 78%] [Guideline] [PubMed]    [Study Details]"
        // patient-match badge comes first (most clinically useful at-
        // a-glance number), then evidence-type and source-type tags,
        // then the Study Details button on the right.
        return `
            <div class="source-item">
                <span class="source-citation">${index + 1}. ${citation}</span>
                ${patientMatchBadge}${evidenceTypeBadge}${sourceTypeBadge}
                ${studyDetailsBtn}
            </div>
        `;
    }).join('');
}

/**
 * Show loading state
 */
function showLoading(element, message = 'Loading...') {
    element.innerHTML = `
        <div class="loading">
            <div class="spinner"></div>
            <span>${message}</span>
        </div>
    `;
}

/**
 * Show error message
 */
function showError(element, message) {
    element.innerHTML = `
        <div class="alert alert-error">
            <strong>Error:</strong> ${message}
        </div>
    `;
}

/**
 * Show success message
 */
function showSuccess(element, message) {
    element.innerHTML = `
        <div class="alert alert-success">
            <strong>Success:</strong> ${message}
        </div>
    `;
}

/**
 * Format match score as percentage
 */
function formatScore(score) {
    return `${(score * 100).toFixed(1)}%`;
}

/**
 * Debounce function calls
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Copy text to clipboard
 */
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        return true;
    } catch (err) {
        console.error('Failed to copy:', err);
        return false;
    }
}

/**
 * Format date
 */
function formatDate(dateString) {
    if (!dateString) return 'Unknown';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
        year: 'numeric', 
        month: 'long', 
        day: 'numeric' 
    });
}

/**
 * Scroll to element smoothly
 */
function scrollToElement(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

/**
 * Get URL parameter
 */
function getUrlParameter(name) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(name);
}

/**
 * Set active navigation link
 */
function setActiveNavLink(page) {
    document.querySelectorAll('.nav-links a').forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === page || link.textContent.toLowerCase() === page.toLowerCase()) {
            link.classList.add('active');
        }
    });
}

/* ============================================
   Collapsible Section Functions
   ============================================ */

/**
 * Split an answer into a short answer (first sentence) and justification (rest)
 * @param {string} answerText - The full answer text
 * @returns {Object} Object with shortAnswer and justification properties
 */
function splitAnswerAndJustification(answerText) {
    if (!answerText) {
        return { shortAnswer: '', justification: '' };
    }
    
    // Try to find the first sentence ending with a period followed by space or end
    // But be careful with abbreviations like "et al." or "Dr." or numbers like "1.5"
    const sentenceEndPattern = /(?<![A-Z][a-z]?)(?<!\bet\s)(?<!\bal)(?<!\d)\.\s+(?=[A-Z])/;
    const match = answerText.match(sentenceEndPattern);
    
    if (match && match.index !== undefined) {
        const splitIndex = match.index + 1; // Include the period
        const shortAnswer = answerText.substring(0, splitIndex).trim();
        const justification = answerText.substring(splitIndex).trim();
        return { shortAnswer, justification };
    }
    
    // Fallback: try splitting on first period followed by space
    const simpleSplit = answerText.indexOf('. ');
    if (simpleSplit > 20 && simpleSplit < answerText.length - 10) {
        const shortAnswer = answerText.substring(0, simpleSplit + 1).trim();
        const justification = answerText.substring(simpleSplit + 2).trim();
        return { shortAnswer, justification };
    }
    
    // If no good split point, return the whole thing as short answer
    return { shortAnswer: answerText, justification: '' };
}

/**
 * Create HTML for a collapsible "More Information" section
 * @param {string} messageId - Unique identifier for the message
 * @param {string} contentHtml - Pre-formatted HTML content (justification + sources)
 * @returns {string} Complete HTML for the collapsible section, or empty string if no content
 */
function createCollapsibleSection(messageId, contentHtml) {
    // Return empty string if contentHtml is empty, null, or undefined
    if (!contentHtml || contentHtml.trim() === '') {
        return '';
    }
    
    const contentId = `details-${messageId}`;
    
    return `
        <div class="collapsible-section">
            <button class="collapsible-toggle" 
                    aria-expanded="false" 
                    aria-controls="${contentId}"
                    type="button">
                <span class="toggle-text">More Information</span>
                <span class="toggle-icon" aria-hidden="true">▼</span>
            </button>
            <div class="collapsible-content" 
                 id="${contentId}" 
                 aria-hidden="true">
                ${contentHtml}
            </div>
        </div>
    `;
}

/**
 * Toggle the expand/collapse state of a collapsible section
 * @param {HTMLElement} toggleElement - The toggle button element
 */
function toggleCollapsible(toggleElement) {
    if (!toggleElement) return;
    
    const isExpanded = toggleElement.getAttribute('aria-expanded') === 'true';
    const contentId = toggleElement.getAttribute('aria-controls');
    const contentElement = document.getElementById(contentId);
    const iconElement = toggleElement.querySelector('.toggle-icon');
    
    if (!contentElement) return;
    
    // Toggle states
    const newExpandedState = !isExpanded;
    
    // Update ARIA attributes
    toggleElement.setAttribute('aria-expanded', newExpandedState.toString());
    contentElement.setAttribute('aria-hidden', (!newExpandedState).toString());
    
    // Toggle CSS classes for animation
    if (newExpandedState) {
        contentElement.classList.add('expanded');
    } else {
        contentElement.classList.remove('expanded');
    }
    
    // Rotate chevron icon
    if (iconElement) {
        if (newExpandedState) {
            iconElement.classList.add('rotated');
        } else {
            iconElement.classList.remove('rotated');
        }
    }
}

/**
 * Initialize event listeners for a collapsible toggle button
 * @param {HTMLElement} toggleElement - The toggle button element
 */
function initCollapsibleToggle(toggleElement) {
    if (!toggleElement) return;
    
    // Click event listener
    toggleElement.addEventListener('click', function(e) {
        e.preventDefault();
        toggleCollapsible(this);
    });
    
    // Keyboard event listener for Enter and Space
    toggleElement.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            toggleCollapsible(this);
        }
    });
}
