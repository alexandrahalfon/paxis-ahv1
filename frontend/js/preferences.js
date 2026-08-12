/**
 * Preferences Management
 * Handles sidebar UI, saving/loading preferences, and applying filters
 */

class PreferencesManager {
    constructor() {
        this.sidebar = document.getElementById('preferencesSidebar');
        this.overlay = document.getElementById('sidebarOverlay');
        this.openBtn = document.getElementById('openPreferences');
        this.closeBtn = document.querySelector('.btn-close-sidebar');
        this.saveBtn = document.getElementById('savePreferences');
        this.clearBtn = document.getElementById('clearPreferences');
        
        // Toggle elements
        this.inlineToggle = document.getElementById('preferencesActiveToggle');
        this.sidebarToggle = document.getElementById('filtersActive');
        this.statusIndicator = document.getElementById('preferencesStatus');
        
        // Selected values for autocomplete fields
        this.selectedCountries = [];
        this.selectedInstitutions = [];
        
        this.initEventListeners();
        this.initAutocomplete();
        this.loadPreferences();
    }
    
    initEventListeners() {
        // Open sidebar
        this.openBtn?.addEventListener('click', () => this.openSidebar());
        
        // Close sidebar
        this.closeBtn?.addEventListener('click', () => this.closeSidebar());
        this.overlay?.addEventListener('click', () => this.closeSidebar());
        
        // Save preferences
        this.saveBtn?.addEventListener('click', () => this.savePreferences());
        
        // Clear preferences
        this.clearBtn?.addEventListener('click', () => this.clearAllFilters());
        
        // ESC key to close
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.sidebar?.classList.contains('open')) {
                this.closeSidebar();
            }
        });
        
        // Sync inline toggle with sidebar toggle
        if (this.inlineToggle) {
            this.inlineToggle.addEventListener('change', () => this.handleInlineToggleChange());
        }
        
        if (this.sidebarToggle) {
            this.sidebarToggle.addEventListener('change', () => this.handleSidebarToggleChange());
        }
    }
    
    initAutocomplete() {
        // Country autocomplete
        const countryInput = document.getElementById('countrySearch');
        const countryDropdown = document.getElementById('countryDropdown');
        
        if (countryInput && countryDropdown) {
            countryInput.addEventListener('input', (e) => {
                this.searchOptions('countries', e.target.value, countryDropdown);
            });
            
            countryInput.addEventListener('focus', () => {
                this.searchOptions('countries', countryInput.value, countryDropdown);
            });
            
            // Close dropdown when clicking outside
            document.addEventListener('click', (e) => {
                if (!e.target.closest('.autocomplete-container')) {
                    countryDropdown.classList.remove('show');
                }
            });
        }
        
        // Institution autocomplete
        const institutionInput = document.getElementById('institutionSearch');
        const institutionDropdown = document.getElementById('institutionDropdown');
        
        if (institutionInput && institutionDropdown) {
            institutionInput.addEventListener('input', (e) => {
                this.searchOptions('institutions', e.target.value, institutionDropdown);
            });
            
            institutionInput.addEventListener('focus', () => {
                this.searchOptions('institutions', institutionInput.value, institutionDropdown);
            });
            
            document.addEventListener('click', (e) => {
                if (!e.target.closest('.autocomplete-container')) {
                    institutionDropdown.classList.remove('show');
                }
            });
        }
    }
    
    async searchOptions(type, query, dropdown) {
        try {
            const response = await fetch(`${CONFIG.API_BASE}/user-preferences/${type}?search=${encodeURIComponent(query)}`);
            if (!response.ok) throw new Error('Failed to fetch options');
            
            const data = await response.json();
            this.renderDropdown(type, data.options, dropdown);
        } catch (error) {
            console.error(`Error fetching ${type}:`, error);
            dropdown.innerHTML = '<div class="dropdown-item disabled">Error loading options</div>';
            dropdown.classList.add('show');
        }
    }
    
    renderDropdown(type, options, dropdown) {
        const selectedList = type === 'countries' ? this.selectedCountries : this.selectedInstitutions;
        
        if (options.length === 0) {
            dropdown.innerHTML = '<div class="dropdown-item disabled">No results found</div>';
            dropdown.classList.add('show');
            return;
        }
        
        dropdown.innerHTML = options.map(opt => {
            const isSelected = selectedList.includes(opt.value);
            return `
                <div class="dropdown-item ${isSelected ? 'selected' : ''}" 
                     data-value="${this.escapeHtml(opt.value)}"
                     data-type="${type}">
                    <span class="item-label">${this.escapeHtml(opt.label)}</span>
                    <span class="item-count">${opt.count} studies</span>
                    ${isSelected ? '<span class="item-check">✓</span>' : ''}
                </div>
            `;
        }).join('');
        
        // Add click handlers
        dropdown.querySelectorAll('.dropdown-item:not(.disabled)').forEach(item => {
            item.addEventListener('click', () => {
                const value = item.dataset.value;
                const itemType = item.dataset.type;
                this.toggleSelection(itemType, value);
                
                // Update dropdown display
                const input = itemType === 'countries' 
                    ? document.getElementById('countrySearch')
                    : document.getElementById('institutionSearch');
                this.searchOptions(itemType, input?.value || '', dropdown);
            });
        });
        
        dropdown.classList.add('show');
    }
    
    toggleSelection(type, value) {
        const list = type === 'countries' ? this.selectedCountries : this.selectedInstitutions;
        const index = list.indexOf(value);
        
        if (index === -1) {
            list.push(value);
        } else {
            list.splice(index, 1);
        }
        
        this.renderSelectedTags(type);
    }
    
    renderSelectedTags(type) {
        const list = type === 'countries' ? this.selectedCountries : this.selectedInstitutions;
        const container = document.getElementById(type === 'countries' ? 'selectedCountries' : 'selectedInstitutions');
        
        if (!container) return;
        
        if (list.length === 0) {
            container.innerHTML = '<span class="no-selection">None selected</span>';
            return;
        }
        
        container.innerHTML = list.map(value => `
            <span class="selected-tag">
                ${this.escapeHtml(value)}
                <button type="button" class="tag-remove" data-type="${type}" data-value="${this.escapeHtml(value)}">×</button>
            </span>
        `).join('');
        
        // Add remove handlers
        container.querySelectorAll('.tag-remove').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const tagType = btn.dataset.type;
                const tagValue = btn.dataset.value;
                this.toggleSelection(tagType, tagValue);
            });
        });
    }
    
    escapeHtml(str) {
        if (!str) return '';
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }
    
    /**
     * Handle inline toggle change (quick toggle next to Preferences button)
     */
    handleInlineToggleChange() {
        const isActive = this.inlineToggle?.checked || false;
        
        // Sync with sidebar toggle
        if (this.sidebarToggle) {
            this.sidebarToggle.checked = isActive;
        }
        
        // Update status indicator
        this.updateStatusIndicator(isActive);
        
        // Auto-save the toggle state
        this.saveToggleState(isActive);
    }
    
    /**
     * Handle sidebar toggle change
     */
    handleSidebarToggleChange() {
        const isActive = this.sidebarToggle?.checked || false;
        
        // Sync with inline toggle
        if (this.inlineToggle) {
            this.inlineToggle.checked = isActive;
        }
        
        // Update status indicator
        this.updateStatusIndicator(isActive);
    }
    
    /**
     * Update the status indicator text
     */
    updateStatusIndicator(isActive) {
        if (!this.statusIndicator) return;
        
        if (isActive) {
            this.statusIndicator.textContent = 'On';
            this.statusIndicator.classList.add('active');
        } else {
            this.statusIndicator.textContent = 'Off';
            this.statusIndicator.classList.remove('active');
        }
    }
    
    /**
     * Save just the toggle state (quick save without full preferences)
     */
    async saveToggleState(isActive) {
        try {
            const token = localStorage.getItem('exueed_token');
            if (!token) {
                // Just update localStorage for non-logged-in users
                const saved = localStorage.getItem('userPreferences');
                if (saved) {
                    const prefs = JSON.parse(saved);
                    prefs.filters_active = isActive;
                    localStorage.setItem('userPreferences', JSON.stringify(prefs));
                }
                return;
            }
            
            // Get current preferences and update just the toggle
            const filters = this.collectFilters();
            filters.filters_active = isActive;
            
            const response = await fetch(`${CONFIG.API_BASE}/user-preferences`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify(filters)
            });
            
            if (response.ok) {
                localStorage.setItem('userPreferences', JSON.stringify(filters));
                console.log('[Preferences] Toggle state saved:', isActive ? 'On' : 'Off');
            }
        } catch (error) {
            console.error('[Preferences] Error saving toggle state:', error);
        }
    }
    
    openSidebar() {
        this.sidebar?.classList.add('open');
        this.overlay?.classList.add('active');
        document.body.style.overflow = 'hidden';
    }
    
    closeSidebar() {
        this.sidebar?.classList.remove('open');
        this.overlay?.classList.remove('active');
        document.body.style.overflow = '';
        
        // Close any open dropdowns
        document.querySelectorAll('.autocomplete-dropdown').forEach(d => d.classList.remove('show'));
    }
    
    /**
     * Collect current filter values from UI
     */
    collectFilters() {
        const filters = {
            study_types: [],
            study_phases: [],
            cancer_types: [],
            min_patients: null,
            max_patients: null,
            analysis_types: [],
            treatment_modalities: [],
            countries: this.selectedCountries,
            institutions: this.selectedInstitutions,
            race_ethnicities: [],
            include_unknown_race: true,
            min_publication_year: null,
            max_publication_year: null,
            require_peer_reviewed: false,
            min_followup_months: null,
            required_outcomes: [],
            sort_by: 'relevance',
            sort_order: 'desc',
            filters_active: true,
            results_per_page: 20
        };
        
        // Study types (checkboxes)
        document.querySelectorAll('input[name="study_type"]:checked').forEach(cb => {
            filters.study_types.push(cb.value);
        });
        
        // Study phases (checkboxes)
        document.querySelectorAll('input[name="study_phase"]:checked').forEach(cb => {
            filters.study_phases.push(cb.value);
        });
        
        // Cancer types (multi-select)
        const cancerSelect = document.querySelector('select[name="cancer_type"]');
        if (cancerSelect) {
            filters.cancer_types = Array.from(cancerSelect.selectedOptions)
                .map(opt => opt.value)
                .filter(v => v !== '');
        }
        
        // Patient count range
        const minPatients = document.querySelector('input[name="min_patients"]');
        const maxPatients = document.querySelector('input[name="max_patients"]');
        filters.min_patients = minPatients?.value ? parseInt(minPatients.value) : null;
        filters.max_patients = maxPatients?.value ? parseInt(maxPatients.value) : null;
        
        // Analysis types (checkboxes)
        document.querySelectorAll('input[name="analysis_type"]:checked').forEach(cb => {
            filters.analysis_types.push(cb.value);
        });
        
        // Treatment modalities (checkboxes)
        document.querySelectorAll('input[name="treatment_modality"]:checked').forEach(cb => {
            filters.treatment_modalities.push(cb.value);
        });
        
        // Race/ethnicity filter (checkboxes)
        document.querySelectorAll('input[name="race_ethnicity"]:checked').forEach(cb => {
            filters.race_ethnicities.push(cb.value);
        });
        
        // Include unknown race toggle
        const includeUnknownRace = document.querySelector('input[name="include_unknown_race"]');
        filters.include_unknown_race = includeUnknownRace?.checked ?? true;
        
        // Publication year range
        const minYear = document.querySelector('input[name="min_publication_year"]');
        const maxYear = document.querySelector('input[name="max_publication_year"]');
        filters.min_publication_year = minYear?.value ? parseInt(minYear.value) : null;
        filters.max_publication_year = maxYear?.value ? parseInt(maxYear.value) : null;
        
        // Evidence quality
        const peerReviewed = document.querySelector('input[name="require_peer_reviewed"]');
        filters.require_peer_reviewed = peerReviewed?.checked || false;
        
        const minFollowup = document.querySelector('input[name="min_followup_months"]');
        filters.min_followup_months = minFollowup?.value ? parseInt(minFollowup.value) : null;
        
        // Required outcomes (checkboxes)
        document.querySelectorAll('input[name="required_outcomes"]:checked').forEach(cb => {
            filters.required_outcomes.push(cb.value);
        });
        
        // Sorting
        const sortSelect = document.querySelector('select[name="sort_by"]');
        if (sortSelect) {
            filters.sort_by = sortSelect.value;
        }
        
        // Filters active toggle
        const activeToggle = document.getElementById('filtersActive');
        if (activeToggle) {
            filters.filters_active = activeToggle.checked;
        }
        
        // Include user uploads toggle
        const includeUploadsToggle = document.getElementById('includeUserUploads');
        if (includeUploadsToggle) {
            filters.include_user_uploads = includeUploadsToggle.checked;
        }
        
        return filters;
    }
    
    /**
     * Apply filter values to UI
     */
    applyFiltersToUI(filters) {
        // Study types
        filters.study_types?.forEach(value => {
            const checkbox = document.querySelector(`input[name="study_type"][value="${value}"]`);
            if (checkbox) checkbox.checked = true;
        });
        
        // Study phases
        filters.study_phases?.forEach(value => {
            const checkbox = document.querySelector(`input[name="study_phase"][value="${value}"]`);
            if (checkbox) checkbox.checked = true;
        });
        
        // Cancer types
        const cancerSelect = document.querySelector('select[name="cancer_type"]');
        if (cancerSelect && filters.cancer_types) {
            Array.from(cancerSelect.options).forEach(opt => {
                opt.selected = filters.cancer_types.includes(opt.value);
            });
        }
        
        // Patient counts
        if (filters.min_patients !== null) {
            const minInput = document.querySelector('input[name="min_patients"]');
            if (minInput) minInput.value = filters.min_patients;
        }
        if (filters.max_patients !== null) {
            const maxInput = document.querySelector('input[name="max_patients"]');
            if (maxInput) maxInput.value = filters.max_patients;
        }
        
        // Analysis types
        filters.analysis_types?.forEach(value => {
            const checkbox = document.querySelector(`input[name="analysis_type"][value="${value}"]`);
            if (checkbox) checkbox.checked = true;
        });
        
        // Treatment modalities
        filters.treatment_modalities?.forEach(value => {
            const checkbox = document.querySelector(`input[name="treatment_modality"][value="${value}"]`);
            if (checkbox) checkbox.checked = true;
        });
        
        // Countries
        this.selectedCountries = filters.countries || [];
        this.renderSelectedTags('countries');
        
        // Institutions
        this.selectedInstitutions = filters.institutions || [];
        this.renderSelectedTags('institutions');
        
        // Race/ethnicity filters
        filters.race_ethnicities?.forEach(value => {
            const checkbox = document.querySelector(`input[name="race_ethnicity"][value="${value}"]`);
            if (checkbox) checkbox.checked = true;
        });
        
        // Include unknown race toggle
        const includeUnknownRace = document.querySelector('input[name="include_unknown_race"]');
        if (includeUnknownRace) {
            includeUnknownRace.checked = filters.include_unknown_race !== false;
        }
        
        // Publication year range
        if (filters.min_publication_year !== null) {
            const minYear = document.querySelector('input[name="min_publication_year"]');
            if (minYear) minYear.value = filters.min_publication_year;
        }
        if (filters.max_publication_year !== null) {
            const maxYear = document.querySelector('input[name="max_publication_year"]');
            if (maxYear) maxYear.value = filters.max_publication_year;
        }
        
        // Evidence quality
        const peerReviewed = document.querySelector('input[name="require_peer_reviewed"]');
        if (peerReviewed) peerReviewed.checked = filters.require_peer_reviewed || false;
        
        if (filters.min_followup_months !== null) {
            const minFollowup = document.querySelector('input[name="min_followup_months"]');
            if (minFollowup) minFollowup.value = filters.min_followup_months;
        }
        
        // Required outcomes
        filters.required_outcomes?.forEach(value => {
            const checkbox = document.querySelector(`input[name="required_outcomes"][value="${value}"]`);
            if (checkbox) checkbox.checked = true;
        });
        
        // Sorting
        const sortSelect = document.querySelector('select[name="sort_by"]');
        if (sortSelect && filters.sort_by) {
            sortSelect.value = filters.sort_by;
        }
        
        // Active toggle
        const activeToggle = document.getElementById('filtersActive');
        if (activeToggle) {
            activeToggle.checked = filters.filters_active !== false;
        }
        
        // Include user uploads toggle
        const includeUploadsToggle = document.getElementById('includeUserUploads');
        if (includeUploadsToggle) {
            includeUploadsToggle.checked = filters.include_user_uploads !== false;
        }
        
        // Sync inline toggle and status indicator
        const isActive = filters.filters_active !== false;
        if (this.inlineToggle) {
            this.inlineToggle.checked = isActive;
        }
        this.updateStatusIndicator(isActive);
    }
    
    /**
     * Save preferences to backend
     */
    async savePreferences() {
        const filters = this.collectFilters();
        
        try {
            const response = await fetch(`${CONFIG.API_BASE}/user-preferences`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('exueed_token')}`
                },
                body: JSON.stringify(filters)
            });
            
            if (!response.ok) {
                throw new Error('Failed to save preferences');
            }
            
            // Store in localStorage as fallback
            localStorage.setItem('userPreferences', JSON.stringify(filters));
            
            // Sync toggles and status
            const isActive = filters.filters_active !== false;
            if (this.inlineToggle) {
                this.inlineToggle.checked = isActive;
            }
            this.updateStatusIndicator(isActive);
            
            // Show success message
            this.showNotification('Preferences saved', 'success');
            
        } catch (error) {
            console.error('[Preferences] Error saving:', error);
            // Still save to localStorage even if server fails
            localStorage.setItem('userPreferences', JSON.stringify(filters));
            this.showNotification('Saved locally (login to sync)', 'info');
        }
    }
    
    /**
     * Load preferences from backend or localStorage
     */
    async loadPreferences() {
        let filtersActive = true; // Default to active
        
        try {
            const token = localStorage.getItem('exueed_token');
            if (token) {
                const response = await fetch(`${CONFIG.API_BASE}/user-preferences`, {
                    headers: {
                        'Authorization': `Bearer ${token}`
                    }
                });
                
                if (response.ok) {
                    const filters = await response.json();
                    this.applyFiltersToUI(filters);
                    filtersActive = filters.filters_active !== false;
                    this.updateStatusIndicator(filtersActive);
                    return;
                }
            }
        } catch (error) {
            console.error('[Preferences] Error loading from server:', error);
        }
        
        // Fallback to localStorage
        const savedFilters = localStorage.getItem('userPreferences');
        if (savedFilters) {
            const filters = JSON.parse(savedFilters);
            this.applyFiltersToUI(filters);
            filtersActive = filters.filters_active !== false;
        }
        
        // Always update status indicator
        this.updateStatusIndicator(filtersActive);
    }
    
    /**
     * Get current preferences (for use in queries)
     */
    getActivePreferences() {
        const filters = this.collectFilters();
        if (!filters.filters_active) {
            return null;
        }
        return filters;
    }
    
    /**
     * Clear all filters
     */
    clearAllFilters() {
        // Uncheck all checkboxes
        document.querySelectorAll('.preferences-sidebar input[type="checkbox"]').forEach(cb => {
            if (cb.id !== 'filtersActive') {
                cb.checked = false;
            }
        });
        
        // Clear number inputs
        document.querySelectorAll('.preferences-sidebar input[type="number"]').forEach(input => {
            input.value = '';
        });
        
        // Clear text inputs (search fields)
        document.querySelectorAll('.preferences-sidebar input[type="text"]').forEach(input => {
            input.value = '';
        });
        
        // Reset select elements
        document.querySelectorAll('.preferences-sidebar select').forEach(select => {
            if (select.name === 'sort_by') {
                select.value = 'relevance';
            } else {
                select.selectedIndex = 0;
            }
        });
        
        // Clear selected countries and institutions
        this.selectedCountries = [];
        this.selectedInstitutions = [];
        this.renderSelectedTags('countries');
        this.renderSelectedTags('institutions');
        
        this.showNotification('All filters cleared', 'info');
    }
    
    /**
     * Show notification toast
     */
    showNotification(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `notification notification-${type}`;
        toast.textContent = message;
        
        const bgColor = type === 'success' ? 'var(--secondary)' : 
                       type === 'error' ? '#dc3545' : 'var(--primary)';
        
        toast.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 1rem 1.5rem;
            background: ${bgColor};
            color: white;
            border-radius: 8px;
            box-shadow: var(--shadow-lg);
            z-index: 10000;
            animation: slideIn 0.3s ease;
        `;
        
        document.body.appendChild(toast);
        
        setTimeout(() => {
            toast.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    window.preferencesManager = new PreferencesManager();
});
