/**
 * Layered Response Renderer
 * 
 * Renders the layered response structure with:
 * - Brief answer (always visible)
 * - "More Information" expandable section
 * - Structured details (format-specific rendering)
 * - Explanation text
 * - Sources list
 */

class LayeredResponseRenderer {
    
    /**
     * Render a layered response into HTML
     * @param {Object} response - The LayeredResponse object from backend
     * @param {string} messageId - Unique ID for this message (for expand/collapse)
     * @returns {string} HTML string
     */
    static render(response, messageId) {
        const {
            brief_answer,
            structured_details,
            structured_format,
            explanation,
            sources,
            query_type,
            follow_ups
        } = response;
        
        // Check if we have expandable content
        const hasStructuredDetails = structured_details && Object.keys(structured_details).length > 0;
        const hasExplanation = explanation && explanation.trim().length > 0;
        const hasExpandableContent = hasStructuredDetails || hasExplanation;
        
        return `
            <div class="layered-response" data-query-type="${query_type}">
                <!-- Layer 1: Brief Answer (always visible) -->
                <div class="brief-answer">
                    ${this._formatMarkdown(brief_answer)}
                </div>
                
                ${hasExpandableContent ? `
                <!-- Expand Button -->
                <button class="expand-btn" onclick="LayeredResponseRenderer.toggle('${messageId}')" id="expand-btn-${messageId}">
                    <span class="expand-text">More information</span>
                    <span class="expand-arrow">▼</span>
                </button>
                
                <!-- Expandable Content -->
                <div class="expandable-content hidden" id="expandable-${messageId}">
                    ${hasStructuredDetails ? `
                    <!-- Layer 2: Structured Details -->
                    <div class="structured-details">
                        ${this._renderStructuredDetails(structured_details, structured_format)}
                    </div>
                    ` : ''}
                    
                    ${hasExplanation ? `
                    <!-- Layer 3: Explanation -->
                    <div class="explanation">
                        ${this._formatMarkdown(explanation)}
                    </div>
                    ` : ''}
                </div>
                ` : ''}
                
                <!-- Layer 4: Sources (always visible at bottom) -->
                ${sources && sources.length > 0 ? `
                <div class="sources-section">
                    <div class="sources-header">Sources</div>
                    <div class="sources-list">
                        ${sources.map((source, i) => this._renderSource(source, i + 1)).join('')}
                    </div>
                </div>
                ` : ''}
                
                <!-- Follow-up Suggestions -->
                ${follow_ups && follow_ups.length > 0 ? `
                <div class="follow-ups">
                    ${follow_ups.map(f => `
                        <button class="follow-up-btn" onclick="sendFollowUp('${this._escapeHtml(f)}')">${f}</button>
                    `).join('')}
                </div>
                ` : ''}
            </div>
        `;
    }
    
    /**
     * Toggle expand/collapse state
     */
    static toggle(messageId) {
        const content = document.getElementById(`expandable-${messageId}`);
        const btn = document.getElementById(`expand-btn-${messageId}`);
        
        if (content.classList.contains('hidden')) {
            content.classList.remove('hidden');
            btn.querySelector('.expand-text').textContent = 'Less information';
            btn.querySelector('.expand-arrow').style.transform = 'rotate(180deg)';
        } else {
            content.classList.add('hidden');
            btn.querySelector('.expand-text').textContent = 'More information';
            btn.querySelector('.expand-arrow').style.transform = 'rotate(0deg)';
        }
    }
    
    /**
     * Render structured details based on format type
     */
    static _renderStructuredDetails(details, format) {
        switch (format) {
            case 'prescription':
                return this._renderPrescription(details);
            case 'results_table':
                return this._renderResultsTable(details);
            case 'derivation_steps':
                return this._renderDerivationSteps(details);
            case 'ordered_checklist':
                return this._renderOrderedChecklist(details);
            case 'toxicity_timeline':
                return this._renderToxicityTimeline(details);
            case 'criteria_checklist':
                return this._renderCriteriaChecklist(details);
            case 'comparison_table':
                return this._renderComparisonTable(details);
            case 'eligibility_card':
                return this._renderEligibilityCard(details);
            case 'eligibility_card_with_prose':
                return this._renderEligibilityCardWithProse(details);
            case 'study_matrix':
                return this._renderStudyMatrix(details);
            case 'recommendation_card':
                return this._renderRecommendationCard(details);
            default:
                return this._renderGenericDetails(details);
        }
    }
    
    // ==========================================
    // FORMAT-SPECIFIC RENDERERS
    // ==========================================
    
    static _renderPrescription(details) {
        const { dose, target, technique, concurrent, constraints, boost } = details;
        
        let html = `<div class="prescription-format">`;
        
        // Main fields
        html += `<table class="prescription-table">`;
        if (dose) html += `<tr><th>Dose</th><td>${dose}</td></tr>`;
        if (target) html += `<tr><th>Target</th><td>${target}</td></tr>`;
        if (technique) html += `<tr><th>Technique</th><td>${technique}</td></tr>`;
        if (concurrent) html += `<tr><th>Concurrent</th><td>${concurrent}</td></tr>`;
        if (boost) html += `<tr><th>Boost</th><td>${boost}</td></tr>`;
        html += `</table>`;
        
        // Constraints - handle both string and array formats
        if (constraints) {
            html += `<div class="constraints-section">`;
            html += `<div class="section-label">Dose constraints</div>`;
            
            if (Array.isArray(constraints) && constraints.length > 0) {
                // Array format with structure/constraint objects
                html += `
                    <table class="constraints-table">
                        <thead><tr><th>Structure</th><th>Constraint</th></tr></thead>
                        <tbody>
                            ${constraints.map(c => `<tr><td>${c.structure || ''}</td><td>${c.constraint || ''}</td></tr>`).join('')}
                        </tbody>
                    </table>
                `;
            } else if (typeof constraints === 'string' && constraints.trim()) {
                // String format - display as text
                html += `<div style="font-size: 14px; color: var(--text-secondary, #666);">${constraints}</div>`;
            }
            
            html += `</div>`;
        }
        
        html += `</div>`;
        return html;
    }
    
    static _renderResultsTable(details) {
        const { trial_name, design, primary_endpoint, outcomes, conclusion } = details;
        
        let html = `<div class="results-format">`;
        
        // Header
        html += `
            <div class="trial-header">
                <strong>${trial_name || 'Trial'}</strong>
                ${design ? `<span class="trial-design">${design}</span>` : ''}
            </div>
            ${primary_endpoint ? `<div class="primary-ep">Primary endpoint: ${primary_endpoint}</div>` : ''}
        `;
        
        // Outcomes - handle both string and array formats
        if (outcomes) {
            if (typeof outcomes === 'string' && outcomes.trim()) {
                // String format - display as styled text block
                html += `<div class="primary-ep">${outcomes}</div>`;
            } else if (Array.isArray(outcomes) && outcomes.length > 0) {
                // Array format with outcome objects
                html += `
                    <table class="outcomes-table">
                        <thead>
                            <tr>
                                <th>Outcome</th>
                                <th>${outcomes[0].arm_a_name || 'Arm A'}</th>
                                <th>${outcomes[0].arm_b_name || 'Arm B'}</th>
                                <th>HR (95% CI)</th>
                                <th>p-value</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${outcomes.map(o => `
                                <tr>
                                    <td>${o.outcome || ''}</td>
                                    <td>${o.arm_a || '—'}</td>
                                    <td>${o.arm_b || '—'}</td>
                                    <td>${o.hr || '—'}</td>
                                    <td>${o.p || '—'}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                `;
            }
        }
        
        // Conclusion
        if (conclusion) {
            html += `<div class="trial-conclusion"><strong>Conclusion:</strong> ${conclusion}</div>`;
        }
        
        html += `</div>`;
        return html;
    }
    
    static _renderDerivationSteps(details) {
        const { t_stage, n_stage, m_stage, overall_stage, edition, special_factors } = details;
        
        let html = `<div class="staging-format">`;
        
        // T Stage
        if (t_stage) {
            html += `
                <div class="stage-component">
                    <div class="stage-label">T Stage: <strong>${t_stage.stage}</strong></div>
                    <div class="stage-reasoning">${t_stage.reasoning}</div>
                </div>
            `;
        }
        
        // N Stage
        if (n_stage) {
            html += `
                <div class="stage-component">
                    <div class="stage-label">N Stage: <strong>${n_stage.stage}</strong></div>
                    <div class="stage-reasoning">${n_stage.reasoning}</div>
                </div>
            `;
        }
        
        // M Stage
        if (m_stage) {
            html += `
                <div class="stage-component">
                    <div class="stage-label">M Stage: <strong>${m_stage.stage}</strong></div>
                    <div class="stage-reasoning">${m_stage.reasoning}</div>
                </div>
            `;
        }
        
        // Overall Stage
        html += `
            <div class="overall-stage">
                <strong>Overall Stage: ${overall_stage}</strong>
                ${edition ? `<span class="edition">(AJCC ${edition})</span>` : ''}
            </div>
        `;
        
        // Special factors - handle both string and array formats
        if (special_factors) {
            let factorsText = '';
            if (typeof special_factors === 'string' && special_factors.trim()) {
                factorsText = special_factors;
            } else if (Array.isArray(special_factors) && special_factors.length > 0) {
                factorsText = special_factors.join(', ');
            }
            if (factorsText) {
                html += `
                    <div class="special-factors">
                        <em>Key factors: ${factorsText}</em>
                    </div>
                `;
            }
        }
        
        html += `</div>`;
        return html;
    }
    
    static _renderOrderedChecklist(details) {
        const { sections, steps } = details;
        
        let html = `<div class="checklist-format">`;
        
        // Handle steps array (from workup extraction)
        if (steps && steps.length > 0) {
            html += `<ul class="checklist-items">`;
            steps.forEach((step, i) => {
                if (typeof step === 'string') {
                    html += `<li><span class="checkbox">${i + 1}.</span> <span class="item-text">${step}</span></li>`;
                } else {
                    html += `<li><span class="checkbox">${i + 1}.</span> <span class="item-text">${step.test || ''}${step.purpose ? ` - ${step.purpose}` : ''}</span></li>`;
                }
            });
            html += `</ul>`;
        }
        // Handle sections array (complex format)
        else if (sections && sections.length > 0) {
            sections.forEach((section, i) => {
                html += `<div class="checklist-section">`;
                html += `<div class="section-title">${i + 1}. ${section.title || ''}</div>`;
                if (section.items && Array.isArray(section.items)) {
                    html += `<ul class="checklist-items">`;
                    section.items.forEach(item => {
                        if (typeof item === 'string') {
                            html += `<li><span class="checkbox">-</span> <span class="item-text">${item}</span></li>`;
                        } else {
                            html += `<li><span class="checkbox">-</span> <span class="item-text">${item.text || ''}</span>${item.conditional ? `<span class="conditional">(${item.conditional})</span>` : ''}</li>`;
                        }
                    });
                    html += `</ul>`;
                }
                html += `</div>`;
            });
        }
        
        html += `</div>`;
        return html;
    }
    
    static _renderToxicityTimeline(details) {
        const { acute, late } = details;
        
        let html = `<div class="toxicity-format">`;
        
        // Acute toxicities - handle both string array and object array
        if (acute && acute.length > 0) {
            html += `<div class="toxicity-section">`;
            html += `<div class="section-label">Acute (during treatment to 90 days)</div>`;
            
            if (typeof acute[0] === 'string') {
                // String array format
                html += `<ul style="margin: 0; padding-left: 1.5rem;">`;
                acute.forEach(item => {
                    html += `<li style="margin: 0.5rem 0;">${item}</li>`;
                });
                html += `</ul>`;
            } else {
                // Object array format
                html += `
                    <table class="toxicity-table">
                        <thead>
                            <tr><th>Effect</th><th>Incidence</th><th>Grade 3+</th><th>Management</th></tr>
                        </thead>
                        <tbody>
                            ${acute.map(t => `
                                <tr>
                                    <td>${t.effect || ''}</td>
                                    <td>${t.incidence || ''}</td>
                                    <td>${t.grade3_plus || '—'}</td>
                                    <td>${t.management || '—'}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                `;
            }
            html += `</div>`;
        }
        
        // Late toxicities - handle both string array and object array
        if (late && late.length > 0) {
            html += `<div class="toxicity-section">`;
            html += `<div class="section-label">Late (&gt;90 days post-treatment)</div>`;
            
            if (typeof late[0] === 'string') {
                // String array format
                html += `<ul style="margin: 0; padding-left: 1.5rem;">`;
                late.forEach(item => {
                    html += `<li style="margin: 0.5rem 0;">${item}</li>`;
                });
                html += `</ul>`;
            } else {
                // Object array format
                html += `
                    <table class="toxicity-table">
                        <thead>
                            <tr><th>Effect</th><th>Incidence</th><th>Risk factors</th><th>Management</th></tr>
                        </thead>
                        <tbody>
                            ${late.map(t => `
                                <tr>
                                    <td>${t.effect || ''}</td>
                                    <td>${t.incidence || ''}</td>
                                    <td>${t.risk_factors || '—'}</td>
                                    <td>${t.management || '—'}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                `;
            }
            html += `</div>`;
        }
        
        html += `</div>`;
        return html;
    }
    
    static _renderCriteriaChecklist(details) {
        const { indications, consider_if, not_indicated, guideline_category, guideline_source } = details;
        
        let html = `<div class="criteria-format">`;
        
        // Indications - handle both string array and object array
        if (indications && indications.length > 0) {
            html += `<div class="criteria-section indications">`;
            html += `<div class="section-label">Indications</div>`;
            html += `<ul>`;
            indications.forEach(i => {
                if (typeof i === 'string') {
                    html += `<li><span class="criterion-check">+</span> ${i}</li>`;
                } else {
                    html += `<li><span class="criterion-check">+</span> ${i.criterion || ''}${i.abbreviated ? ` <span class="abbrev">(${i.abbreviated})</span>` : ''}</li>`;
                }
            });
            html += `</ul></div>`;
        }
        
        // Consider if - handle both string array and object array
        if (consider_if && consider_if.length > 0) {
            html += `<div class="criteria-section consider">`;
            html += `<div class="section-label">Consider if</div>`;
            html += `<ul>`;
            consider_if.forEach(c => {
                if (typeof c === 'string') {
                    html += `<li><span class="criterion-consider">~</span> ${c}</li>`;
                } else {
                    html += `<li><span class="criterion-consider">~</span> ${c.criterion || ''}${c.note ? ` <span class="note">(${c.note})</span>` : ''}</li>`;
                }
            });
            html += `</ul></div>`;
        }
        
        // Not indicated - handle both string array and object array
        if (not_indicated && not_indicated.length > 0) {
            html += `<div class="criteria-section exclusions">`;
            html += `<div class="section-label">Not indicated</div>`;
            html += `<ul>`;
            not_indicated.forEach(n => {
                if (typeof n === 'string') {
                    html += `<li><span class="criterion-exclude">-</span> ${n}</li>`;
                } else {
                    html += `<li><span class="criterion-exclude">-</span> ${n.criterion || ''}</li>`;
                }
            });
            html += `</ul></div>`;
        }
        
        // Guideline info
        if (guideline_category || guideline_source) {
            html += `
                <div class="guideline-info">
                    ${guideline_category ? `<span class="category">Category: ${guideline_category}</span>` : ''}
                    ${guideline_source ? `<span class="source">Source: ${guideline_source}</span>` : ''}
                </div>
            `;
        }
        
        html += `</div>`;
        return html;
    }
    
    static _renderComparisonTable(details) {
        const { options, comparison, option_a_favored, option_b_favored, key_evidence } = details;
        
        let html = `<div class="comparison-format">`;
        
        // Comparison - handle both string and array formats
        if (comparison) {
            if (typeof comparison === 'string' && comparison.trim()) {
                // String format - display as styled text block
                html += `<div class="trial-conclusion">${comparison}</div>`;
            } else if (Array.isArray(comparison) && comparison.length > 0) {
                // Array format with comparison objects
                const optionNames = options || ['Option A', 'Option B'];
                html += `
                    <table class="comparison-table">
                        <thead>
                            <tr>
                                <th>Outcome</th>
                                <th>${optionNames[0]}</th>
                                <th>${optionNames[1]}</th>
                                <th>Difference</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${comparison.map(c => `
                                <tr>
                                    <td>${c.outcome || ''}</td>
                                    <td>${c.option_a || '—'}</td>
                                    <td>${c.option_b || '—'}</td>
                                    <td class="difference">${c.difference || '—'}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                `;
            }
        }
        
        // Patient selection - handle both string and array formats
        html += `<div class="patient-selection">`;
        if (option_a_favored) {
            if (typeof option_a_favored === 'string' && option_a_favored.trim()) {
                html += `
                    <div class="favored-section">
                        <strong>${options?.[0] || 'Option A'} favored when:</strong>
                        <ul><li>${option_a_favored}</li></ul>
                    </div>
                `;
            } else if (Array.isArray(option_a_favored) && option_a_favored.length > 0) {
                html += `
                    <div class="favored-section">
                        <strong>${options?.[0] || 'Option A'} favored when:</strong>
                        <ul>${option_a_favored.map(f => `<li>${f}</li>`).join('')}</ul>
                    </div>
                `;
            }
        }
        if (option_b_favored) {
            if (typeof option_b_favored === 'string' && option_b_favored.trim()) {
                html += `
                    <div class="favored-section">
                        <strong>${options?.[1] || 'Option B'} favored when:</strong>
                        <ul><li>${option_b_favored}</li></ul>
                    </div>
                `;
            } else if (Array.isArray(option_b_favored) && option_b_favored.length > 0) {
                html += `
                    <div class="favored-section">
                        <strong>${options?.[1] || 'Option B'} favored when:</strong>
                        <ul>${option_b_favored.map(f => `<li>${f}</li>`).join('')}</ul>
                    </div>
                `;
            }
        }
        html += `</div>`;
        
        html += `</div>`;
        return html;
    }
    
    static _renderEligibilityCard(details) {
        const { study_title, author_year, match_score, criteria, treatment, key_outcome, fit_summary } = details;
        
        let html = `<div class="eligibility-format">`;
        
        // Header
        html += `
            <div class="eligibility-header">
                <div class="study-title">${study_title || 'Study'}</div>
                <div class="author-year">${author_year || ''}</div>
                <div class="match-score">Match: <strong>${match_score || 0}%</strong></div>
            </div>
        `;
        
        // Criteria checklist - handle both string and array formats
        if (criteria) {
            if (typeof criteria === 'string' && criteria.trim()) {
                // String format - display as text
                html += `
                    <div class="eligibility-checklist">
                        <div class="section-label">Eligibility assessment</div>
                        <p style="margin: 0.5rem 0; font-size: 14px;">${criteria}</p>
                    </div>
                `;
            } else if (Array.isArray(criteria) && criteria.length > 0) {
                // Array format with criterion objects
                html += `
                    <div class="eligibility-checklist">
                        <div class="section-label">Eligibility assessment</div>
                        ${criteria.map(c => {
                            if (typeof c === 'string') {
                                return `<div class="criterion-row"><span class="criterion-text">${c}</span></div>`;
                            }
                            const icon = c.status === 'met' ? '+' : 
                                         c.status === 'not_met' || c.status === 'excluded' ? '-' : '?';
                            const statusClass = c.status === 'met' ? 'met' : 
                                               c.status === 'not_met' || c.status === 'excluded' ? 'not-met' : 'unknown';
                            return `
                                <div class="criterion-row ${statusClass}">
                                    <span class="criterion-icon">${icon}</span>
                                    <span class="criterion-text">${c.criterion || ''}</span>
                                    <span class="patient-value">— ${c.patient_value || ''}</span>
                                </div>
                            `;
                        }).join('')}
                    </div>
                `;
            }
        }
        
        // Treatment and outcome
        if (treatment || key_outcome) {
            html += `<div class="study-details">`;
            if (treatment) html += `<div><strong>Treatment:</strong> ${treatment}</div>`;
            if (key_outcome) html += `<div><strong>Key outcome:</strong> ${key_outcome}</div>`;
            html += `</div>`;
        }
        
        // Fit summary
        if (fit_summary) {
            html += `<div class="fit-summary"><strong>Patient fit:</strong> ${fit_summary}</div>`;
        }
        
        html += `</div>`;
        return html;
    }
    
    /**
     * Render eligibility card WITH preserved prose sections
     * This is the combined format for Trial Match that shows:
     * 1. Structured eligibility checklist (new, scannable)
     * 2. Original prose sections (summary, treatment, population, eligibility, patient_fit)
     */
    static _renderEligibilityCardWithProse(details) {
        const { 
            match_header, 
            eligibility_checklist, 
            prose, 
            key_outcome 
        } = details;
        
        // Support both flat and nested structures for backwards compatibility
        const study_title = match_header?.study_title || details.study_title;
        const author_year = match_header?.author_year || details.author_year;
        const match_percent = match_header?.match_percent || details.match_score || details.match_percent;
        const criteria = eligibility_checklist || details.criteria;
        
        let html = `<div class="eligibility-format eligibility-with-prose">`;
        
        // Header
        html += `
            <div class="eligibility-header">
                <div class="study-title">${study_title || 'Study'}</div>
                <div class="author-year">${author_year || ''}</div>
                <div class="match-score">Match: <strong>${match_percent || 0}%</strong></div>
            </div>
        `;
        
        // NEW: Structured eligibility checklist - handle both string and array formats
        if (criteria) {
            if (typeof criteria === 'string' && criteria.trim()) {
                // String format - display as text
                html += `
                    <div class="eligibility-checklist">
                        <div class="section-label">Eligibility assessment</div>
                        <p style="margin: 0.5rem 0; font-size: 14px;">${criteria}</p>
                    </div>
                `;
            } else if (Array.isArray(criteria) && criteria.length > 0) {
                // Array format with criterion objects
                html += `
                    <div class="eligibility-checklist">
                        <div class="section-label">Eligibility assessment</div>
                        ${criteria.map(c => {
                            if (typeof c === 'string') {
                                return `<div class="criterion-row"><span class="criterion-text">${c}</span></div>`;
                            }
                            const icon = c.status === 'met' ? '+' : 
                                         c.status === 'not_met' || c.status === 'excluded' ? '-' : '?';
                            const statusClass = c.status === 'met' ? 'met' : 
                                               c.status === 'not_met' || c.status === 'excluded' ? 'not-met' : 'unknown';
                            return `
                                <div class="criterion-row ${statusClass}">
                                    <span class="criterion-icon">${icon}</span>
                                    <span class="criterion-text">${c.criterion || ''}</span>
                                    <span class="patient-value">— ${c.patient_value || ''}</span>
                                </div>
                            `;
                        }).join('')}
                    </div>
                `;
            }
        }
        
        // Key outcome (if present)
        if (key_outcome) {
            html += `
                <div class="study-outcome">
                    <strong>Key outcome:</strong> ${key_outcome}
                </div>
            `;
        }
        
        // PRESERVED: Original prose sections
        if (prose) {
            html += `<div class="prose-sections">`;
            
            if (prose.summary) {
                html += `
                    <div class="prose-section">
                        <h4>Summary</h4>
                        <p>${prose.summary}</p>
                    </div>
                `;
            }
            
            if (prose.treatment) {
                html += `
                    <div class="prose-section">
                        <h4>Treatment</h4>
                        <p>${prose.treatment}</p>
                    </div>
                `;
            }
            
            if (prose.population) {
                html += `
                    <div class="prose-section">
                        <h4>Population</h4>
                        <p>${prose.population}</p>
                    </div>
                `;
            }
            
            if (prose.eligibility) {
                const { inclusion, exclusion } = prose.eligibility;
                html += `
                    <div class="prose-section">
                        <h4>Eligibility</h4>
                        ${inclusion ? `<p><strong>Inclusion:</strong> ${inclusion}</p>` : ''}
                        ${exclusion ? `<p><strong>Exclusion:</strong> ${exclusion}</p>` : ''}
                    </div>
                `;
            }
            
            if (prose.patient_fit) {
                html += `
                    <div class="prose-section patient-fit-section">
                        <h4>Patient Fit</h4>
                        <p>${prose.patient_fit}</p>
                    </div>
                `;
            }
            
            html += `</div>`;
        }
        
        html += `</div>`;
        return html;
    }
    
    static _renderStudyMatrix(details) {
        const { studies, outcomes, key_differences } = details;
        
        let html = `<div class="study-matrix-format">`;
        
        // Studies - handle both string and array formats
        if (studies) {
            if (typeof studies === 'string' && studies.trim()) {
                // String format - display as text
                html += `<div class="studies-text" style="margin: 1rem 0; font-size: 14px;">${studies}</div>`;
            } else if (Array.isArray(studies) && studies.length > 0) {
                // Array format with study objects
                html += `
                    <table class="studies-table">
                        <thead>
                            <tr>
                                <th>Study</th>
                                <th>Year</th>
                                <th>N</th>
                                <th>Design</th>
                                <th>Population</th>
                                <th>Primary EP</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${studies.map(s => `
                                <tr>
                                    <td>${s.name || ''}</td>
                                    <td>${s.year || '—'}</td>
                                    <td>${s.n || '—'}</td>
                                    <td>${s.design || '—'}</td>
                                    <td>${s.population || '—'}</td>
                                    <td>${s.primary_ep || '—'}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                `;
            }
        }
        
        // Outcomes comparison - handle both string and array formats
        if (outcomes) {
            if (typeof outcomes === 'string' && outcomes.trim()) {
                // String format - display as text
                html += `
                    <div class="section-label" style="margin-top: 16px;">Outcomes comparison</div>
                    <div class="outcomes-text" style="font-size: 14px;">${outcomes}</div>
                `;
            } else if (Array.isArray(outcomes) && outcomes.length > 0) {
                // Array format with outcome objects
                const studyNames = (Array.isArray(studies) && studies.length > 0) 
                    ? studies.map(s => s.name || 'Study') 
                    : ['Study 1', 'Study 2', 'Study 3'];
                html += `
                    <div class="section-label" style="margin-top: 16px;">Outcomes comparison</div>
                    <table class="outcomes-comparison-table">
                        <thead>
                            <tr>
                                <th>Outcome</th>
                                ${studyNames.map(n => `<th>${n}</th>`).join('')}
                            </tr>
                        </thead>
                        <tbody>
                            ${outcomes.map(o => `
                                <tr>
                                    <td>${o.outcome || ''}</td>
                                    <td>${o.study_1 || '—'}</td>
                                    <td>${o.study_2 || '—'}</td>
                                    ${o.study_3 ? `<td>${o.study_3}</td>` : ''}
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                `;
            }
        }
        
        // Key differences - handle both string and array formats
        if (key_differences) {
            if (typeof key_differences === 'string' && key_differences.trim()) {
                html += `
                    <div class="key-differences">
                        <div class="section-label">Key differences</div>
                        <p style="margin: 0.5rem 0;">${key_differences}</p>
                    </div>
                `;
            } else if (Array.isArray(key_differences) && key_differences.length > 0) {
                html += `
                    <div class="key-differences">
                        <div class="section-label">Key differences</div>
                        <ul>${key_differences.map(d => `<li>${d}</li>`).join('')}</ul>
                    </div>
                `;
            }
        }
        
        html += `</div>`;
        return html;
    }
    
    static _renderRecommendationCard(details) {
        const { recommendation, patient_factors, rationale, key_evidence, alternatives,
                // Also handle flat format from backend
                preferred_regimen, alternative, guideline_category, source } = details;
        
        let html = `<div class="recommendation-format">`;
        
        // Handle flat format from backend (preferred_regimen, alternative, etc.)
        if (preferred_regimen || alternative || guideline_category || source) {
            if (preferred_regimen) {
                html += `
                    <div class="recommendation-box">
                        <div class="rec-treatment">${preferred_regimen}</div>
                    </div>
                `;
            }
            
            if (alternative) {
                html += `
                    <div class="rec-section">
                        <div class="section-label">Alternative</div>
                        <p style="margin: 0.5rem 0;">${alternative}</p>
                    </div>
                `;
            }
            
            if (guideline_category || source) {
                html += `<div class="rec-section">`;
                if (guideline_category) {
                    html += `<p style="margin: 0.25rem 0;"><strong>Guideline category:</strong> ${guideline_category}</p>`;
                }
                if (source) {
                    html += `<p style="margin: 0.25rem 0;"><strong>Source:</strong> ${source}</p>`;
                }
                html += `</div>`;
            }
        }
        // Handle nested object format
        else {
            // Recommendation
            if (recommendation) {
                if (typeof recommendation === 'string') {
                    html += `
                        <div class="recommendation-box">
                            <div class="rec-treatment">${recommendation}</div>
                        </div>
                    `;
                } else {
                    html += `
                        <div class="recommendation-box">
                            <div class="rec-treatment">${recommendation.treatment || ''}</div>
                            ${recommendation.duration ? `<div class="rec-duration">${recommendation.duration}</div>` : ''}
                        </div>
                    `;
                }
            }
            
            // Patient factors - handle string or object with items array
            if (patient_factors) {
                if (typeof patient_factors === 'string') {
                    html += `
                        <div class="rec-section">
                            <div class="section-label">For this patient</div>
                            <p style="margin: 0.5rem 0;">${patient_factors}</p>
                        </div>
                    `;
                } else if (patient_factors.items && Array.isArray(patient_factors.items) && patient_factors.items.length > 0) {
                    html += `
                        <div class="rec-section">
                            <div class="section-label">${patient_factors.title || 'For this patient'}</div>
                            <ul>${patient_factors.items.map(i => `<li>${i}</li>`).join('')}</ul>
                        </div>
                    `;
                }
            }
            
            // Rationale - handle string or object with items array
            if (rationale) {
                if (typeof rationale === 'string') {
                    html += `
                        <div class="rec-section">
                            <div class="section-label">Rationale</div>
                            <p style="margin: 0.5rem 0;">${rationale}</p>
                        </div>
                    `;
                } else if (rationale.items && Array.isArray(rationale.items) && rationale.items.length > 0) {
                    html += `
                        <div class="rec-section">
                            <div class="section-label">${rationale.title || 'Rationale'}</div>
                            <ul>${rationale.items.map(i => `<li>${i}</li>`).join('')}</ul>
                        </div>
                    `;
                }
            }
            
            // Key evidence - handle string or object with items array
            if (key_evidence) {
                if (typeof key_evidence === 'string') {
                    html += `
                        <div class="rec-section">
                            <div class="section-label">Key evidence</div>
                            <p style="margin: 0.5rem 0;">${key_evidence}</p>
                        </div>
                    `;
                } else if (key_evidence.items && Array.isArray(key_evidence.items) && key_evidence.items.length > 0) {
                    html += `
                        <div class="rec-section">
                            <div class="section-label">${key_evidence.title || 'Key evidence'}</div>
                            <ul>${key_evidence.items.map(i => `<li>${i}</li>`).join('')}</ul>
                        </div>
                    `;
                }
            }
            
            // Alternatives - handle string or object with items array
            if (alternatives) {
                if (typeof alternatives === 'string') {
                    html += `
                        <div class="rec-section alternatives">
                            <div class="section-label">Alternatives</div>
                            <p style="margin: 0.5rem 0;">${alternatives}</p>
                        </div>
                    `;
                } else if (alternatives.items && Array.isArray(alternatives.items) && alternatives.items.length > 0) {
                    html += `
                        <div class="rec-section alternatives">
                            <div class="section-label">${alternatives.title || 'Alternatives'}</div>
                            <ul>${alternatives.items.map(i => `<li>${i}</li>`).join('')}</ul>
                        </div>
                    `;
                }
            }
        }
        
        html += `</div>`;
        return html;
    }
    
    static _renderGenericDetails(details) {
        // Fallback for unknown formats
        if (typeof details === 'string') {
            return `<div class="generic-details">${this._formatMarkdown(details)}</div>`;
        }
        
        let html = `<div class="generic-details">`;
        for (const [key, value] of Object.entries(details)) {
            html += `<div><strong>${key}:</strong> ${typeof value === 'object' ? JSON.stringify(value) : value}</div>`;
        }
        html += `</div>`;
        return html;
    }
    
    // ==========================================
    // HELPER METHODS
    // ==========================================
    
    static _renderSource(source, index) {
        const title = source.title || source.name || 'Source';
        const author = source.author || '';
        const year = source.year || '';
        const doi = source.doi;
        
        return `
            <div class="source-item">
                <span class="source-num">${index}</span>
                <span class="source-text">
                    ${title}
                    ${author ? ` - ${author}` : ''}
                    ${year ? ` (${year})` : ''}
                    ${doi ? ` <a href="https://doi.org/${doi}" target="_blank" class="doi-link">[DOI]</a>` : ''}
                </span>
            </div>
        `;
    }
    
    static _formatMarkdown(text) {
        if (!text) return '';
        
        // Basic markdown formatting
        return text
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.+?)\*/g, '<em>$1</em>')
            .replace(/`(.+?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
    }
    
    static _escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = LayeredResponseRenderer;
}
