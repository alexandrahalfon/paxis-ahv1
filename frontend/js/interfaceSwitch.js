/**
 * Interface switch (added 2026-08-08)
 *
 * Paxis now has two front doors:
 *   - the patient interface  (patient-home.html, the default landing page)
 *   - the clinician interface (index.html and every existing tool page)
 *
 * This module injects the switch between them and handles sending a user
 * to the right side after login. It is deliberately self-contained and
 * additive, following the same pattern as sideNav.js and demoBanner.js:
 * a page opts in with one script tag plus one init call, and no existing
 * page markup has to change.
 */
const InterfaceSwitch = {
    PATIENT_HOME: 'patient-home.html',
    CLINICIAN_HOME: 'index.html',

    /** Cached role from the last /auth/me, so we don't refetch per page. */
    _cachedRole: null,

    /**
     * @param {'patient'|'clinician'} side which interface this page belongs to
     */
    init(side) {
        this.side = side || 'clinician';
        this.injectStyles();
        this.injectSwitch();
    },

    injectStyles() {
        if (document.getElementById('interfaceSwitchStyles')) return;
        const css = `
            .iface-switch {
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                padding: 0.35rem 0.75rem;
                border: 1px solid var(--primary, #e11d48);
                border-radius: 999px;
                color: var(--primary, #e11d48);
                background: transparent;
                font-size: 0.82rem;
                font-weight: 700;
                text-decoration: none;
                white-space: nowrap;
                transition: background 0.15s ease, color 0.15s ease;
            }
            .iface-switch:hover {
                background: var(--primary, #e11d48);
                color: #fff;
            }
            .iface-switch svg { width: 14px; height: 14px; }
            @media (max-width: 640px) {
                .iface-switch .iface-switch-label-long { display: none; }
            }
        `;
        const el = document.createElement('style');
        el.id = 'interfaceSwitchStyles';
        el.textContent = css;
        document.head.appendChild(el);
    },

    injectSwitch() {
        // Drop the switch into the page's existing nav list if there is
        // one, so it inherits the header layout rather than floating.
        const navList = document.querySelector('.nav-links');
        if (!navList) return;

        const goingToPatient = this.side === 'clinician';
        const href = goingToPatient ? this.PATIENT_HOME : this.CLINICIAN_HOME;
        const label = goingToPatient ? 'For Patients' : 'For Clinicians';

        const li = document.createElement('li');
        li.className = 'iface-switch-item';
        li.innerHTML = `
            <a href="${href}" class="iface-switch" id="ifaceSwitchLink">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
                     stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M8 3 4 7l4 4"></path><path d="M4 7h16"></path>
                    <path d="m16 21 4-4-4-4"></path><path d="M20 17H4"></path>
                </svg>
                <span class="iface-switch-label-long">${label}</span>
            </a>
        `;
        // Insert before the auth controls so it reads as navigation.
        const firstAuth = navList.querySelector('.nav-auth');
        if (firstAuth) navList.insertBefore(li, firstAuth);
        else navList.appendChild(li);
    },

    /**
     * Fetch the signed-in user's role, or null if signed out.
     * Safe to call anywhere; never throws.
     */
    async getRole() {
        if (this._cachedRole !== null) return this._cachedRole;
        const token = localStorage.getItem('exueed_token');
        if (!token) return null;
        try {
            const base = (typeof CONFIG !== 'undefined' && CONFIG.API_BASE)
                ? CONFIG.API_BASE : '/api';
            const res = await fetch(`${base}/auth/me`, {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (!res.ok) return null;
            const user = await res.json();
            this._cachedRole = user.role || 'physician';
            return this._cachedRole;
        } catch (_) {
            return null;
        }
    },

    /**
     * Send a freshly-logged-in user to the interface matching their role.
     * Called from the login page. Falls back to the clinician home, which
     * is where every pre-existing account belongs.
     */
    async redirectByRole(fallback) {
        const role = await this.getRole();
        if (role === 'patient') {
            window.location.href = this.PATIENT_HOME;
        } else {
            window.location.href = fallback || this.CLINICIAN_HOME;
        }
    },

    /**
     * On a clinician page, nudge a patient account back to their own side.
     * Advisory only: the real enforcement is the role guard on the API,
     * this just avoids showing a patient a UI full of 403s.
     */
    async guardClinicianPage() {
        const role = await this.getRole();
        if (role === 'patient') window.location.href = this.PATIENT_HOME;
    }
};

if (typeof window !== 'undefined') window.InterfaceSwitch = InterfaceSwitch;
