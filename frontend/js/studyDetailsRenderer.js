/**
 * Study Details Renderer - Complete Field Display
 * Dynamically displays ALL non-null fields with evidence quotes in italics
 */

// ============================================
// Dynamic Field Renderer
// ============================================

class StudyDetailsRenderer {
    constructor() {
        this.sectionOrder = [
            { key: 'study_details', title: 'Study Details' },
            { key: 'patient_characteristics', title: 'Patient Characteristics' },
            { key: 'diagnosis', title: 'Diagnosis' },
            { key: 'staging', title: 'Staging' },
            { key: 'treatment', title: 'Treatment' },
            { key: 'outcomes', title: 'Outcomes' },
            { key: 'biomarkers', title: 'Biomarkers' },
            { key: 'toxicity', title: 'Toxicity' },
            { key: 'dose_constraints', title: 'Dose Constraints' }
        ];
    }
    
    /**
     * Render complete study details with all non-null fields
     * @param {Object} data - Study details from API
     * @returns {string} HTML string
     */
    render(data) {
        if (!data) return '<div class="error">No data available</div>';
        
        let html = '';
        
        // Render each section
        for (const section of this.sectionOrder) {
            let sectionData = data[section.key];
            if (!sectionData) continue;
            
            // For study_details section, add DOI and PMID from top-level
            if (section.key === 'study_details') {
                sectionData = { ...sectionData };
                if (data.doi) {
                    sectionData['doi'] = { label: 'DOI', value: data.doi, evidence_quote: null };
                }
                if (data.pmid) {
                    sectionData['pmid'] = { label: 'PMID', value: data.pmid, evidence_quote: null };
                }
            }
            
            const sectionHtml = this.renderSection(section.title, sectionData);
            if (sectionHtml) {
                html += sectionHtml;
            }
        }
        
        return html || '<div class="no-data">No study details available</div>';
    }
    
    /**
     * Render a section (handles both objects and arrays)
     * @param {string} title - Section title
     * @param {Object|Array} data - Section data
     * @returns {string} HTML string
     */
    renderSection(title, data) {
        if (!data) return '';
        
        // Check if section has any content
        const hasContent = Array.isArray(data) 
            ? data.length > 0 
            : Object.keys(data).some(key => {
                const value = data[key];
                if (Array.isArray(value)) return value.length > 0;
                if (typeof value === 'object' && value !== null) {
                    return value.value !== null && value.value !== undefined;
                }
                return value !== null && value !== undefined;
            });
        
        if (!hasContent) return '';
        
        const sectionId = `section-${title.toLowerCase().replace(/\s+/g, '-')}`;
        
        let contentHtml = '';
        
        if (Array.isArray(data)) {
            // Render array items
            contentHtml = this.renderArrayItems(data);
        } else {
            // Render object fields
            contentHtml = this.renderObjectFields(data);
        }
        
        if (!contentHtml) return '';
        
        return `
            <div class="detail-section" id="${sectionId}">
                <div class="section-header" onclick="studyDetailsRenderer.toggleSection('${sectionId}')">
                    <h4 class="section-title">
                        <span class="expand-icon">▼</span>
                        ${escapeHtml(title)}
                    </h4>
                </div>
                <div class="section-content expanded">
                    ${contentHtml}
                </div>
            </div>
        `;
    }
    
    /**
     * Render object fields (handles both field objects and plain values)
     * Skips duplicate nested arrays that are rendered separately
     * @param {Object} obj - Object to render
     * @returns {string} HTML string
     */
    renderObjectFields(obj) {
        let html = '';
        
        // Track which keys to skip (they're rendered as separate sub-sections)
        const skipKeys = ['inclusion_criteria', 'exclusion_criteria', 'stage_distribution', 
                         'staging_components', 'study_arms', 'chemotherapy_regimens', 
                         'radiation_details', 'surgery_details'];
        
        for (const [key, value] of Object.entries(obj)) {
            if (value === null || value === undefined) continue;
            
            // Skip keys that are rendered separately to avoid duplication
            if (skipKeys.includes(key)) continue;
            
            // Handle arrays (like inclusion_criteria, exclusion_criteria)
            if (Array.isArray(value)) {
                if (value.length === 0) continue;
                
                const label = this.formatLabel(key);
                html += `
                    <div class="field-group">
                        <div class="field-label">${escapeHtml(label)}</div>
                        <div class="array-items">
                            ${this.renderArrayItems(value)}
                        </div>
                    </div>
                `;
                continue;
            }
            
            // Handle field objects with value and evidence_quote
            if (typeof value === 'object' && value.value !== null && value.value !== undefined) {
                // Skip study_name/title entirely (already shown in panel header)
                if (key === 'study_name' || key === 'title') continue;
                
                const fieldHtml = this.renderField(value.label || this.formatLabel(key), value.value, value.evidence_quote);
                if (fieldHtml) html += fieldHtml;
                continue;
            }
            
            // Handle plain values
            if (typeof value !== 'object') {
                const fieldHtml = this.renderField(this.formatLabel(key), value);
                if (fieldHtml) html += fieldHtml;
            }
        }
        
        // Now render the nested arrays as sub-sections
        for (const key of skipKeys) {
            const value = obj[key];
            if (!value || !Array.isArray(value) || value.length === 0) continue;
            
            const label = this.formatLabel(key);
            html += `
                <div class="field-group">
                    <div class="field-label">${escapeHtml(label)}</div>
                    <div class="array-items">
                        ${this.renderArrayItems(value)}
                    </div>
                </div>
            `;
        }
        
        return html;
    }
    
    /**
     * Render array items
     * @param {Array} items - Array of items
     * @returns {string} HTML string
     */
    renderArrayItems(items) {
        if (!items || items.length === 0) return '';
        
        return items.map((item, index) => {
            // Handle simple strings (some LLMs return criteria as plain strings)
            if (typeof item === 'string') {
                return this.renderCriterion(item, null);
            }
            
            // Handle simple objects with criterion/evidence
            if (item.criterion) {
                return this.renderCriterion(item.criterion, item.evidence_quote);
            }
            
            // Handle objects with 'value' field (alternative format)
            if (item.value && typeof item.value === 'string') {
                return this.renderCriterion(item.value, item.evidence_quote);
            }
            
            // Handle toxicity
            if (item.toxicity_type) {
                return this.renderToxicity(item);
            }
            
            // Handle biomarkers
            if (item.biomarker_name) {
                return this.renderBiomarker(item);
            }
            
            // Handle study arms
            if (item.arm_name) {
                return this.renderStudyArm(item);
            }
            
            // Handle chemotherapy regimens
            if (item.regimen_name) {
                return this.renderChemoRegimen(item);
            }
            
            // Handle radiation details
            if (item.radiation_type) {
                return this.renderRadiationDetail(item);
            }
            
            // Handle surgery details
            if (item.surgery_type) {
                return this.renderSurgeryDetail(item);
            }
            
            // Handle stage distribution
            if (item.stage_category) {
                return this.renderStageDistribution(item);
            }
            
            // Handle staging components
            if (item.component_name) {
                return this.renderStagingComponent(item);
            }
            
            // Handle dose constraints
            if (item.organ_at_risk) {
                return this.renderDoseConstraint(item);
            }
            
            // Generic object rendering
            return this.renderGenericItem(item, index);
        }).join('');
    }
    
    /**
     * Render a single field with optional evidence
     * @param {string} label - Field label
     * @param {any} value - Field value
     * @param {string} evidence - Evidence quote (optional)
     * @returns {string} HTML string
     */
    renderField(label, value, evidence = null) {
        if (value === null || value === undefined) return '';
        
        return `
            <div class="field-row">
                <div class="field-label">${escapeHtml(label)}</div>
                <div class="field-value">
                    ${escapeHtml(String(value))}
                    ${evidence ? `<div class="evidence-quote">${escapeHtml(evidence)}</div>` : ''}
                </div>
            </div>
        `;
    }
    
    /**
     * Render criterion (inclusion/exclusion)
     */
    renderCriterion(criterion, evidence) {
        return `
            <div class="criterion-item">
                <div class="criterion-text">${escapeHtml(criterion)}</div>
                ${evidence ? `<div class="evidence-quote">${escapeHtml(evidence)}</div>` : ''}
            </div>
        `;
    }
    
    /**
     * Render toxicity item - bullet format
     */
    renderToxicity(item) {
        const lines = [];
        if (item.toxicity_type) lines.push(`<strong>${escapeHtml(item.toxicity_type)}</strong>`);
        if (item.grade) lines.push(`• Grade: ${escapeHtml(item.grade)}`);
        if (item.frequency) lines.push(`• Frequency: ${escapeHtml(item.frequency)}`);
        if (item.number_of_patients) lines.push(`• Patients: ${escapeHtml(String(item.number_of_patients))}`);
        if (item.timing) lines.push(`• Timing: ${escapeHtml(item.timing)}`);
        
        return `
            <div class="list-item">
                <div class="item-content-bullets">${lines.join('<br>')}</div>
                ${item.evidence_quote ? `<div class="evidence-quote">${escapeHtml(item.evidence_quote)}</div>` : ''}
            </div>
        `;
    }
    
    /**
     * Render biomarker item - bullet format
     */
    renderBiomarker(item) {
        const lines = [];
        if (item.biomarker_name) lines.push(`<strong>${escapeHtml(item.biomarker_name)}</strong>`);
        if (item.biomarker_type) lines.push(`• Type: ${escapeHtml(item.biomarker_type)}`);
        if (item.measurement_method) lines.push(`• Method: ${escapeHtml(item.measurement_method)}`);
        if (item.baseline_value) lines.push(`• Baseline: ${escapeHtml(item.baseline_value)}`);
        if (item.change_from_baseline) lines.push(`• Change: ${escapeHtml(item.change_from_baseline)}`);
        if (item.significance) lines.push(`• Significance: ${escapeHtml(item.significance)}`);
        
        return `
            <div class="list-item">
                <div class="item-content-bullets">${lines.join('<br>')}</div>
                ${item.evidence_quote ? `<div class="evidence-quote">${escapeHtml(item.evidence_quote)}</div>` : ''}
            </div>
        `;
    }
    
    /**
     * Render study arm - bullet format
     */
    renderStudyArm(item) {
        const lines = [];
        if (item.arm_name) lines.push(`<strong>${escapeHtml(item.arm_name)}</strong>`);
        if (item.number_of_patients) lines.push(`• N=${escapeHtml(String(item.number_of_patients))}`);
        if (item.description) lines.push(`• ${escapeHtml(item.description)}`);
        
        return `
            <div class="list-item">
                <div class="item-content-bullets">${lines.join('<br>')}</div>
                ${item.evidence_quote ? `<div class="evidence-quote">${escapeHtml(item.evidence_quote)}</div>` : ''}
            </div>
        `;
    }
    
    /**
     * Render chemotherapy regimen - bullet format
     */
    renderChemoRegimen(item) {
        const lines = [];
        if (item.regimen_name) lines.push(`<strong>${escapeHtml(item.regimen_name)}</strong>`);
        if (item.number_of_cycles) lines.push(`• ${escapeHtml(String(item.number_of_cycles))} cycles`);
        if (item.drugs && item.drugs.length > 0) {
            const drugList = Array.isArray(item.drugs) ? item.drugs.join(', ') : item.drugs;
            lines.push(`• Drugs: ${escapeHtml(drugList)}`);
        }
        if (item.doxorubicin_dose) lines.push(`• Doxorubicin: ${escapeHtml(item.doxorubicin_dose)}`);
        if (item.doxorubicin_schedule) lines.push(`• Schedule: ${escapeHtml(item.doxorubicin_schedule)}`);
        if (item.cyclophosphamide_dose) lines.push(`• Cyclophosphamide: ${escapeHtml(item.cyclophosphamide_dose)}`);
        if (item.dosage_info) lines.push(`• ${escapeHtml(item.dosage_info)}`);
        if (item.schedule_info) lines.push(`• ${escapeHtml(item.schedule_info)}`);
        
        return `
            <div class="list-item">
                <div class="item-content-bullets">${lines.join('<br>')}</div>
                ${item.evidence_quote ? `<div class="evidence-quote">${escapeHtml(item.evidence_quote)}</div>` : ''}
            </div>
        `;
    }
    
    /**
     * Render radiation detail - bullet format
     */
    renderRadiationDetail(item) {
        const lines = [];
        if (item.radiation_type) lines.push(`<strong>${escapeHtml(item.radiation_type)}</strong>`);
        if (item.total_dose) lines.push(`• Dose: ${escapeHtml(item.total_dose)}`);
        if (item.fractionation) lines.push(`• Fractionation: ${escapeHtml(item.fractionation)}`);
        if (item.technique) lines.push(`• Technique: ${escapeHtml(item.technique)}`);
        if (item.target_volume) lines.push(`• Target: ${escapeHtml(item.target_volume)}`);
        
        return `
            <div class="list-item">
                <div class="item-content-bullets">${lines.join('<br>')}</div>
                ${item.evidence_quote ? `<div class="evidence-quote">${escapeHtml(item.evidence_quote)}</div>` : ''}
            </div>
        `;
    }
    
    /**
     * Render surgery detail - bullet format
     */
    renderSurgeryDetail(item) {
        const lines = [];
        if (item.surgery_type) lines.push(`<strong>${escapeHtml(item.surgery_type)}</strong>`);
        if (item.description) lines.push(`• ${escapeHtml(item.description)}`);
        
        return `
            <div class="list-item">
                <div class="item-content-bullets">${lines.join('<br>')}</div>
                ${item.evidence_quote ? `<div class="evidence-quote">${escapeHtml(item.evidence_quote)}</div>` : ''}
            </div>
        `;
    }
    
    /**
     * Render stage distribution - bullet format
     */
    renderStageDistribution(item) {
        const lines = [];
        if (item.stage_category) lines.push(`<strong>${escapeHtml(item.stage_category)}</strong>`);
        if (item.number_of_patients) lines.push(`• ${escapeHtml(String(item.number_of_patients))} patients`);
        if (item.percentage) lines.push(`• ${escapeHtml(item.percentage)}`);
        
        return `
            <div class="list-item">
                <div class="item-content-bullets">${lines.join('<br>')}</div>
                ${item.evidence_quote ? `<div class="evidence-quote">${escapeHtml(item.evidence_quote)}</div>` : ''}
            </div>
        `;
    }
    
    /**
     * Render staging component - bullet format
     */
    renderStagingComponent(item) {
        return `
            <div class="list-item">
                <div class="item-content-bullets">
                    <strong>${escapeHtml(item.component_name || 'Component')}:</strong> ${escapeHtml(item.component_value || '')}
                </div>
                ${item.evidence_quote ? `<div class="evidence-quote">${escapeHtml(item.evidence_quote)}</div>` : ''}
            </div>
        `;
    }
    
    /**
     * Render dose constraint - bullet format
     */
    renderDoseConstraint(item) {
        const lines = [];
        if (item.organ_at_risk) lines.push(`<strong>${escapeHtml(item.organ_at_risk)}</strong>`);
        if (item.constraint_type) lines.push(`• Type: ${escapeHtml(item.constraint_type)}`);
        if (item.dose_limit) lines.push(`• Dose: ${escapeHtml(item.dose_limit)}`);
        if (item.volume_limit) lines.push(`• Volume: ${escapeHtml(item.volume_limit)}`);
        
        return `
            <div class="list-item">
                <div class="item-content-bullets">${lines.join('<br>')}</div>
                ${item.evidence_quote ? `<div class="evidence-quote">${escapeHtml(item.evidence_quote)}</div>` : ''}
            </div>
        `;
    }
    
    /**
     * Render generic item (fallback) - bullet format
     */
    renderGenericItem(item, index) {
        const lines = Object.entries(item)
            .filter(([key, value]) => value !== null && value !== undefined && key !== 'evidence_quote')
            .map(([key, value]) => {
                if (Array.isArray(value)) {
                    return `• ${this.formatLabel(key)}: ${value.join(', ')}`;
                }
                return `• ${this.formatLabel(key)}: ${escapeHtml(String(value))}`;
            });
        
        return `
            <div class="list-item">
                <div class="item-content-bullets">${lines.join('<br>')}</div>
                ${item.evidence_quote ? `<div class="evidence-quote">${escapeHtml(item.evidence_quote)}</div>` : ''}
            </div>
        `;
    }
    
    /**
     * Format field label (convert snake_case to Title Case)
     * @param {string} str - String to format
     * @returns {string} Formatted string
     */
    formatLabel(str) {
        return str
            .replace(/_/g, ' ')
            .replace(/\b\w/g, l => l.toUpperCase());
    }
    
    /**
     * Toggle section expand/collapse
     * @param {string} sectionId - Section element ID
     */
    toggleSection(sectionId) {
        const section = document.getElementById(sectionId);
        if (!section) return;

        const content = section.querySelector('.section-content');
        const icon = section.querySelector('.expand-icon');
        
        if (content.classList.contains('expanded')) {
            content.classList.remove('expanded');
            icon.textContent = '▶';
        } else {
            content.classList.add('expanded');
            icon.textContent = '▼';
        }
    }
}

// Global instance
const studyDetailsRenderer = new StudyDetailsRenderer();

// ============================================
// Query Criteria Match Tags
// ============================================

/**
 * Compare the last query's extracted clinical criteria against a study's data
 * and generate informational match tags (no score, just which criteria match).
 * @param {Object} queryStructure - The query_structure from the last RAG response
 * @param {Object} studyData - The study details data from PostgreSQL
 * @returns {string} HTML string of match tags, or empty string if no matches
 */
function generateQueryMatchTags(queryStructure, studyData) {
    if (!queryStructure || !studyData) return '';

    const matchedTags = [];
    const cancer = queryStructure.cancer || {};
    const treatment = queryStructure.treatment || {};
    const patient = queryStructure.patient || {};

    // Helper: case-insensitive substring check
    function containsAny(haystack, needles) {
        if (!haystack) return false;
        const h = haystack.toLowerCase();
        return needles.some(n => n && h.includes(n.toLowerCase()));
    }

    // Helper: extract field value from study data section
    function getFieldValue(section, fieldKey) {
        if (!section) return null;
        const field = section[fieldKey];
        if (!field) return null;
        if (typeof field === 'string') return field;
        if (field.value) return field.value;
        return null;
    }

    // 1. Cancer site / location
    if (cancer.site) {
        const studyLocation = getFieldValue(studyData.diagnosis, 'cancer_location');
        if (studyLocation) {
            const siteTerms = [cancer.site];
            if (cancer.site_detail) siteTerms.push(cancer.site_detail);
            if (containsAny(studyLocation, siteTerms)) {
                matchedTags.push({ label: cancer.site_detail || cancer.site, category: 'site' });
            }
        }
    }

    // 2. Histology
    if (cancer.histology) {
        const studyHistology = getFieldValue(studyData.diagnosis, 'histopathologic_type');
        if (studyHistology && containsAny(studyHistology, [cancer.histology])) {
            matchedTags.push({ label: cancer.histology, category: 'histology' });
        }
    }

    // 3. Stage
    if (cancer.stage) {
        let stageMatched = false;
        // Check stage_distribution array
        const stageDist = studyData.staging?.stage_distribution;
        if (Array.isArray(stageDist) && stageDist.length > 0) {
            const queryStage = cancer.stage.toLowerCase().replace(/\s+/g, '');
            for (const sd of stageDist) {
                const cat = (sd.stage_category || '').toLowerCase().replace(/\s+/g, '');
                if (cat.includes(queryStage) || queryStage.includes(cat)) {
                    stageMatched = true;
                    break;
                }
            }
        }
        // Also check staging components or risk stratification
        if (!stageMatched) {
            const riskStrat = getFieldValue(studyData.staging, 'risk_stratification');
            if (riskStrat && containsAny(riskStrat, [cancer.stage])) {
                stageMatched = true;
            }
        }
        if (stageMatched) {
            matchedTags.push({ label: 'Stage ' + cancer.stage, category: 'stage' });
        }
    }

    // 4. Treatment modality
    if (treatment.modality) {
        const mod = treatment.modality.toLowerCase();
        let treatmentMatched = false;
        let treatmentLabel = treatment.modality;

        if ((mod.includes('rt') || mod.includes('radiation') || mod.includes('radiotherapy'))
            && studyData.treatment?.radiation_details?.length > 0) {
            treatmentMatched = true;
            treatmentLabel = 'Radiation Therapy';
        }
        if ((mod.includes('chemo') || mod.includes('chemotherapy'))
            && studyData.treatment?.chemotherapy_regimens?.length > 0) {
            treatmentMatched = true;
            treatmentLabel = 'Chemotherapy';
        }
        if ((mod.includes('surg') || mod.includes('surgery'))
            && studyData.treatment?.surgery_details?.length > 0) {
            treatmentMatched = true;
            treatmentLabel = 'Surgery';
        }
        if (mod.includes('immuno') || mod.includes('immunotherapy')) {
            // Check chemo regimens for immunotherapy drugs
            const immunoDrugs = ['pembrolizumab', 'nivolumab', 'atezolizumab', 'durvalumab', 'ipilimumab', 'avelumab', 'cemiplimab'];
            const regimens = studyData.treatment?.chemotherapy_regimens || [];
            for (const r of regimens) {
                const drugs = (r.drugs || r.regimen_name || '').toLowerCase();
                if (immunoDrugs.some(d => drugs.includes(d))) {
                    treatmentMatched = true;
                    treatmentLabel = 'Immunotherapy';
                    break;
                }
            }
        }
        if (treatmentMatched) {
            matchedTags.push({ label: treatmentLabel, category: 'treatment' });
        }
    }

    // 5. Receptor / molecular subtype
    if (cancer.receptor_status) {
        const studySubtype = getFieldValue(studyData.diagnosis, 'molecular_subtype');
        if (studySubtype && containsAny(studySubtype, [cancer.receptor_status])) {
            matchedTags.push({ label: cancer.receptor_status, category: 'molecular' });
        }
    }

    // 6. Grade
    if (cancer.grade) {
        const studyGrade = getFieldValue(studyData.diagnosis, 'tumor_grade');
        if (studyGrade && containsAny(studyGrade, [cancer.grade])) {
            matchedTags.push({ label: 'Grade ' + cancer.grade, category: 'grade' });
        }
    }

    if (matchedTags.length === 0) return '';

    // Build HTML
    const tagColors = {
        site: '#dbeafe',       // blue-100
        histology: '#fce7f3',  // pink-100
        stage: '#fef3c7',      // amber-100
        treatment: '#d1fae5',  // green-100
        molecular: '#ede9fe',  // violet-100
        grade: '#e0e7ff',      // indigo-100
    };
    const tagTextColors = {
        site: '#1e40af',
        histology: '#9d174d',
        stage: '#92400e',
        treatment: '#065f46',
        molecular: '#5b21b6',
        grade: '#3730a3',
    };

    const tagsHtml = matchedTags.map(t => {
        const bg = tagColors[t.category] || '#f3f4f6';
        const color = tagTextColors[t.category] || '#374151';
        return `<span style="display:inline-block;padding:0.2rem 0.6rem;margin:0.15rem;border-radius:9999px;font-size:0.75rem;font-weight:600;background:${bg};color:${color};">${escapeHtml(t.label)}</span>`;
    }).join('');

    return `
        <div class="query-match-tags" style="padding:0.75rem 1rem;margin-bottom:0.5rem;background:#f8fafc;border-bottom:1px solid #e2e8f0;">
            <span style="font-size:0.75rem;font-weight:600;color:#64748b;margin-right:0.5rem;">Matches your query:</span>
            ${tagsHtml}
        </div>
    `;
}

// ============================================
// Integration with Existing SplitViewManager
// ============================================

async function enhancedRenderStudyDetails(splitViewManager, data) {
    // Re-fetch panel element to ensure we have the correct reference
    if (!splitViewManager.panelElement) {
        splitViewManager.panelElement = document.getElementById('study-details-panel');
    }
    
    if (!splitViewManager.panelElement) {
        console.error('[StudyDetails] enhancedRenderStudyDetails: Panel element not found');
        return;
    }

    const title = data.title || data.study_name || 'Study Details';
    const metaInfo = [];
    
    if (data.doc_id) metaInfo.push(`<span class="meta-item">ID: ${escapeHtml(data.doc_id.substring(0, 20))}...</span>`);
    if (data.pmid) metaInfo.push(`<span class="meta-item">PMID: ${escapeHtml(data.pmid)}</span>`);
    if (data.doi) metaInfo.push(`<span class="meta-item">DOI: ${escapeHtml(data.doi)}</span>`);
    
    // Create study identifier for saving
    const studyId = data.doc_id || data.pmid || data.doi || '';
    
    // Check saved status (async for logged-in users)
    let isSaved = isStudySavedInSession(studyId);
    const token = localStorage.getItem('exueed_token');
    if (token && typeof api !== 'undefined') {
        try {
            const result = await api.isStudySaved(studyId);
            isSaved = result.saved;
        } catch (e) {
            console.warn('Failed to check saved status:', e);
        }
    }
    
    const sectionsHtml = studyDetailsRenderer.render(data);
    
    // Generate query match tags if we have a stored query structure
    const queryMatchTagsHtml = generateQueryMatchTags(window._lastQueryStructure, data);

    // Check if study is in comparison tray (if comparison tray exists)
    const inComparisonTray = typeof window.isStudyInComparisonTray === 'function' 
        ? window.isStudyInComparisonTray(studyId) 
        : false;
    
    splitViewManager.panelElement.innerHTML = `
        <div class="panel-header">
            <div class="panel-title-area">
                <h3 class="full-title">${escapeHtml(title)}</h3>
                <div class="panel-meta-row">
                    <div class="panel-meta">
                        ${metaInfo.join('')}
                    </div>
                    <div class="panel-actions">
                        <button class="save-study-btn ${isSaved ? 'saved' : ''}" 
                                data-study-id="${escapeHtml(studyId)}"
                                data-study-title="${escapeHtml(title)}"
                                data-study-doi="${escapeHtml(data.doi || '')}"
                                data-study-pmid="${escapeHtml(data.pmid || '')}"
                                onclick="toggleSaveStudy(this)">
                            <span class="save-star">${isSaved ? '★' : '☆'}</span>
                            <span class="save-text">${isSaved ? 'Saved!' : 'Save'}</span>
                        </button>
                        <button class="compare-study-btn ${inComparisonTray ? 'in-tray' : ''}" 
                                data-study-id="${escapeHtml(studyId)}"
                                data-study-title="${escapeHtml(title)}"
                                data-study-doi="${escapeHtml(data.doi || '')}"
                                data-study-year="${escapeHtml(data.year || '')}"
                                onclick="toggleCompareStudy(this)"
                                title="Add to comparison">
                            <span class="compare-icon">${inComparisonTray ? '✓' : '⊕'}</span>
                            <span class="compare-text">${inComparisonTray ? 'In Compare' : 'Compare'}</span>
                        </button>
                    </div>
                </div>
            </div>
            <button class="close-btn" onclick="splitViewManager.close()" title="Close panel">&times;</button>
        </div>
        <div class="panel-content">
            ${queryMatchTagsHtml}
            ${sectionsHtml}
        </div>
        <div class="panel-footer">
            <button class="study-qa-btn" 
                    data-study-id="${escapeHtml(studyId)}"
                    data-study-title="${escapeHtml(title)}"
                    data-study-doi="${escapeHtml(data.doi || '')}"
                    data-study-pmid="${escapeHtml(data.pmid || '')}"
                    onclick="studyQAManager.startStudyQA(this)">
                Have questions about this trial?
            </button>
        </div>
    `;
}

// Session storage key for saved studies (fallback for non-logged-in users)
const SAVED_STUDIES_KEY = 'exueed_saved_studies';

function getSavedStudies() {
    try {
        const data = sessionStorage.getItem(SAVED_STUDIES_KEY);
        return data ? JSON.parse(data) : [];
    } catch {
        return [];
    }
}

function isStudySavedInSession(studyId) {
    if (!studyId) return false;
    const saved = getSavedStudies();
    return saved.some(s => s.id === studyId);
}

async function isStudySaved(studyId) {
    if (!studyId) return false;
    
    const token = localStorage.getItem('exueed_token');
    if (token && typeof api !== 'undefined') {
        try {
            const result = await api.isStudySaved(studyId);
            return result.saved;
        } catch (e) {
            console.warn('Failed to check saved status from API:', e);
        }
    }
    
    // Fallback to session storage
    return isStudySavedInSession(studyId);
}

async function toggleSaveStudy(btn) {
    const studyId = btn.dataset.studyId;
    const studyTitle = btn.dataset.studyTitle;
    const studyDoi = btn.dataset.studyDoi;
    const studyPmid = btn.dataset.studyPmid;
    
    if (!studyId) return;
    
    const token = localStorage.getItem('exueed_token');
    const isSaved = btn.classList.contains('saved');
    
    btn.disabled = true;
    
    try {
        if (token && typeof api !== 'undefined') {
            // Use backend API
            if (isSaved) {
                await api.deleteSavedStudy(studyId);
                btn.classList.remove('saved');
                btn.querySelector('.save-star').textContent = '☆';
                btn.querySelector('.save-text').textContent = 'Save';
            } else {
                await api.saveStudy(studyId, studyTitle, studyDoi, studyPmid);
                btn.classList.add('saved');
                btn.querySelector('.save-star').textContent = '★';
                btn.querySelector('.save-text').textContent = 'Saved!';
            }
        } else {
            // Use session storage
            const saved = getSavedStudies();
            const existingIndex = saved.findIndex(s => s.id === studyId);
            
            if (existingIndex >= 0) {
                saved.splice(existingIndex, 1);
                sessionStorage.setItem(SAVED_STUDIES_KEY, JSON.stringify(saved));
                btn.classList.remove('saved');
                btn.querySelector('.save-star').textContent = '☆';
                btn.querySelector('.save-text').textContent = 'Save';
            } else {
                saved.unshift({
                    id: studyId,
                    title: studyTitle,
                    doi: studyDoi,
                    pmid: studyPmid,
                    saved_at: new Date().toISOString()
                });
                sessionStorage.setItem(SAVED_STUDIES_KEY, JSON.stringify(saved));
                btn.classList.add('saved');
                btn.querySelector('.save-star').textContent = '★';
                btn.querySelector('.save-text').textContent = 'Saved!';
            }
        }
    } catch (error) {
        console.error('Error toggling save:', error);
        alert('Failed to save study. Please try again.');
    } finally {
        btn.disabled = false;
    }
}

// ============================================
// Standalone Render Function for Upload Page
// ============================================

/**
 * Render study details sections into a container element
 * Used by upload.html to display study details after processing
 * @param {HTMLElement} container - Container element to render into
 * @param {Object} data - Study details data
 */
function renderStudyDetailsSections(container, data) {
    if (!container || !data) return;
    
    const sectionsHtml = studyDetailsRenderer.render(data);
    container.innerHTML = sectionsHtml;
}


// ============================================
// Study Q&A Manager
// ============================================

class StudyQAManager {
    constructor() {
        this.isActive = false;
        this.currentStudy = null;
        this.conversationHistory = [];
        this.qaPanelElement = null;
    }

    /**
     * Start Q&A mode for a specific study
     * @param {HTMLElement} btn - The button that was clicked
     */
    startStudyQA(btn) {
        const studyId = btn.dataset.studyId;
        const studyTitle = btn.dataset.studyTitle;
        const studyDoi = btn.dataset.studyDoi;
        const studyPmid = btn.dataset.studyPmid;

        console.log('[StudyQA] Starting Q&A for study:', {
            id: studyId,
            title: studyTitle,
            doi: studyDoi,
            pmid: studyPmid
        });

        this.currentStudy = {
            id: studyId,
            title: studyTitle,
            doi: studyDoi,
            pmid: studyPmid
        };
        this.conversationHistory = [];
        this.isActive = true;

        // Exit fullscreen mode if active so study details remain visible
        const heroChat = document.querySelector('.hero-chat');
        if (heroChat && heroChat.classList.contains('chat-fullscreen')) {
            heroChat.classList.remove('chat-fullscreen');
            document.body.classList.remove('chat-fullscreen-active');
            console.log('[StudyQA] Exited fullscreen mode to show study details');
        }

        // Add study-qa-active class to body for layout swap
        document.body.classList.add('study-qa-active');

        // Create Q&A panel
        this.createQAPanel();
    }

    /**
     * Create the Q&A chat panel
     */
    createQAPanel() {
        // Check if panel already exists
        let qaPanel = document.getElementById('study-qa-panel');
        if (!qaPanel) {
            qaPanel = document.createElement('div');
            qaPanel.id = 'study-qa-panel';
            qaPanel.className = 'study-qa-panel';
            
            // Insert after study-details-panel
            const detailsPanel = document.getElementById('study-details-panel');
            if (detailsPanel) {
                detailsPanel.parentNode.insertBefore(qaPanel, detailsPanel.nextSibling);
            } else {
                document.body.appendChild(qaPanel);
            }
        }
        this.qaPanelElement = qaPanel;

        // Render the Q&A interface
        this.renderQAInterface();
    }

    /**
     * Render the Q&A chat interface
     */
    renderQAInterface() {
        if (!this.qaPanelElement) return;

        const truncatedTitle = this.currentStudy.title.length > 60 
            ? this.currentStudy.title.substring(0, 60) + '...' 
            : this.currentStudy.title;

        this.qaPanelElement.innerHTML = `
            <div class="qa-panel-header">
                <div class="qa-title-area">
                    <h4>Ask about this study</h4>
                    <span class="qa-study-name" title="${escapeHtml(this.currentStudy.title)}">${escapeHtml(truncatedTitle)}</span>
                </div>
                <button class="close-btn" onclick="studyQAManager.closeQA()" title="Close Q&A">&times;</button>
            </div>
            <div class="qa-messages" id="studyQAMessages">
                <div class="qa-welcome-message">
                    <p>Ask me anything about this study. For example:</p>
                    <div class="qa-example-questions">
                        <button class="qa-example-btn" onclick="studyQAManager.askExample(this)">What were the primary endpoints?</button>
                        <button class="qa-example-btn" onclick="studyQAManager.askExample(this)">What were the inclusion criteria?</button>
                        <button class="qa-example-btn" onclick="studyQAManager.askExample(this)">What toxicities were reported?</button>
                    </div>
                </div>
            </div>
            <div class="qa-input-area">
                <textarea 
                    id="studyQAInput" 
                    class="qa-input" 
                    placeholder="Ask a question about this study..."
                    rows="1"
                    onkeydown="studyQAManager.handleKeyDown(event)"
                ></textarea>
                <button class="qa-send-btn" onclick="studyQAManager.sendQuestion()">
                    <span>›</span>
                </button>
            </div>
            <div class="qa-footer">
                <button class="btn btn-outline btn-sm" onclick="studyQAManager.closeQA()">
                    Back to study details
                </button>
            </div>
        `;

        // Auto-resize textarea
        const textarea = document.getElementById('studyQAInput');
        if (textarea) {
            textarea.addEventListener('input', function() {
                this.style.height = 'auto';
                this.style.height = Math.min(this.scrollHeight, 100) + 'px';
            });
            textarea.focus();
        }
    }

    /**
     * Handle keyboard events in the input
     */
    handleKeyDown(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            this.sendQuestion();
        }
    }

    /**
     * Ask an example question
     */
    askExample(btn) {
        const question = btn.textContent;
        const input = document.getElementById('studyQAInput');
        if (input) {
            input.value = question;
            this.sendQuestion();
        }
    }

    /**
     * Send a question about the study
     */
    async sendQuestion() {
        const input = document.getElementById('studyQAInput');
        const question = input?.value?.trim();
        if (!question) return;

        // Clear input
        input.value = '';
        input.style.height = 'auto';

        // Add user message to UI
        this.addMessage(question, 'user');

        // Show loading
        const loadingId = this.addMessage('Thinking...', 'assistant', true);

        try {
            // Call the study-specific Q&A endpoint
            const response = await this.queryStudy(question);

            // Remove loading
            this.removeMessage(loadingId);

            // Add assistant response
            this.addMessage(response.answer, 'assistant');

            // Update conversation history
            this.conversationHistory.push({ role: 'user', content: question });
            this.conversationHistory.push({ role: 'assistant', content: response.answer });

        } catch (error) {
            this.removeMessage(loadingId);
            this.addMessage(`Error: ${error.message}`, 'assistant');
        }
    }

    /**
     * Query the study-specific endpoint
     */
    async queryStudy(question) {
        console.log('[StudyQA] Querying study:', {
            id: this.currentStudy.id,
            title: this.currentStudy.title,
            doi: this.currentStudy.doi,
            pmid: this.currentStudy.pmid,
            question: question
        });
        
        const response = await fetch(`${CONFIG.API_BASE}/rag/query/study`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...(localStorage.getItem('exueed_token') ? 
                    { 'Authorization': `Bearer ${localStorage.getItem('exueed_token')}` } : {})
            },
            body: JSON.stringify({
                question: question,
                study_id: this.currentStudy.id,
                study_doi: this.currentStudy.doi,
                study_pmid: this.currentStudy.pmid,
                study_title: this.currentStudy.title,
                conversation_history: this.conversationHistory
            })
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            console.error('[StudyQA] Error response:', err);
            throw new Error(err.detail || `HTTP error! status: ${response.status}`);
        }

        return await response.json();
    }

    /**
     * Add a message to the Q&A chat
     */
    addMessage(content, role, isLoading = false) {
        const messagesContainer = document.getElementById('studyQAMessages');
        if (!messagesContainer) return null;

        // Remove welcome message if it exists
        const welcomeMsg = messagesContainer.querySelector('.qa-welcome-message');
        if (welcomeMsg) {
            welcomeMsg.remove();
        }

        const messageId = `qa-msg-${Date.now()}`;
        const messageDiv = document.createElement('div');
        messageDiv.id = messageId;
        messageDiv.className = `qa-message ${role}`;

        if (isLoading) {
            messageDiv.innerHTML = `
                <div class="qa-message-content">
                    <div class="loading"><div class="spinner"></div><span>Thinking...</span></div>
                </div>
            `;
        } else {
            const avatar = role === 'user' ? 'You' : '<img src="assets/paxis-mark.png" alt="Paxis" style="width:20px;height:18px;object-fit:contain;">';
            messageDiv.innerHTML = `
                <div class="qa-message-avatar">${avatar}</div>
                <div class="qa-message-content">${this.formatMessage(content)}</div>
            `;
        }

        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        return messageId;
    }

    /**
     * Remove a message by ID
     */
    removeMessage(messageId) {
        const element = document.getElementById(messageId);
        if (element) element.remove();
    }

    /**
     * Format message content (basic markdown support)
     */
    formatMessage(content) {
        if (!content) return '';
        // Basic escaping and formatting
        let formatted = escapeHtml(content);
        // Convert newlines to <br>
        formatted = formatted.replace(/\n/g, '<br>');
        return formatted;
    }

    /**
     * Close the Q&A mode
     */
    closeQA() {
        this.isActive = false;
        this.currentStudy = null;
        this.conversationHistory = [];

        // Remove study-qa-active class
        document.body.classList.remove('study-qa-active');

        // Remove Q&A panel
        if (this.qaPanelElement) {
            this.qaPanelElement.remove();
            this.qaPanelElement = null;
        }
    }
}

// Global instance
const studyQAManager = new StudyQAManager();


// ============================================
// Comparison Tray Functions (for cross-page use)
// ============================================

const COMPARISON_TRAY_KEY = 'exueed_comparison_tray';

/**
 * Get studies in comparison tray from localStorage
 */
function getComparisonTray() {
    try {
        const data = localStorage.getItem(COMPARISON_TRAY_KEY);
        return data ? JSON.parse(data) : [];
    } catch {
        return [];
    }
}

/**
 * Save comparison tray to localStorage
 */
function saveComparisonTray(studies) {
    localStorage.setItem(COMPARISON_TRAY_KEY, JSON.stringify(studies));
}

/**
 * Check if a study is in the comparison tray
 */
function isStudyInComparisonTray(studyId) {
    if (!studyId) return false;
    const tray = getComparisonTray();
    return tray.some(s => s.doc_id === studyId);
}

// Expose globally
window.isStudyInComparisonTray = isStudyInComparisonTray;
window.getComparisonTray = getComparisonTray;
window.saveComparisonTray = saveComparisonTray;

/**
 * Toggle a study in the comparison tray
 */
function toggleCompareStudy(btn) {
    const studyId = btn.dataset.studyId;
    const studyTitle = btn.dataset.studyTitle;
    const studyDoi = btn.dataset.studyDoi;
    const studyYear = btn.dataset.studyYear;
    
    if (!studyId) return;
    
    let tray = getComparisonTray();
    const index = tray.findIndex(s => s.doc_id === studyId);
    
    if (index >= 0) {
        // Remove from tray
        tray.splice(index, 1);
        saveComparisonTray(tray);
        
        btn.classList.remove('in-tray');
        btn.querySelector('.compare-icon').textContent = '+';
        btn.querySelector('.compare-text').textContent = 'Compare';
    } else {
        // Add to tray (max 4)
        if (tray.length >= 4) {
            // Show modal to manage queue instead of alert
            if (typeof InChatModules !== 'undefined' && typeof InChatModules.showReviewQueueModal === 'function') {
                InChatModules.showReviewQueueModal(studyId, studyTitle, studyDoi, studyYear);
            } else {
                alert('Maximum 4 studies can be compared. Remove one first from the Review Studies page.');
            }
            return;
        }
        
        tray.push({
            doc_id: studyId,
            title: studyTitle,
            doi: studyDoi,
            year: studyYear
        });
        saveComparisonTray(tray);
        
        btn.classList.add('in-tray');
        btn.querySelector('.compare-icon').textContent = '✓';
        btn.querySelector('.compare-text').textContent = 'In Compare';
    }
    
    // Update comparison tray count badge if it exists
    updateComparisonBadge();
}

/**
 * Update the comparison badge count in navigation
 */
function updateComparisonBadge() {
    const badge = document.getElementById('comparisonBadge');
    if (badge) {
        const tray = getComparisonTray();
        badge.textContent = tray.length;
        badge.style.display = tray.length > 0 ? 'inline-flex' : 'none';
    }
}

// Expose updateComparisonBadge globally
window.updateComparisonBadge = updateComparisonBadge;

// Initialize badge on page load
document.addEventListener('DOMContentLoaded', updateComparisonBadge);
