/**
 * Shared auth UI helpers (login/logout link state)
 */

function getStoredToken() {
    return localStorage.getItem('exueed_token');
}

function clearStoredToken() {
    localStorage.removeItem('exueed_token');
}

function updateAuthLinks() {
    const authLink = document.getElementById('authLink');
    const authLinkNav = authLink ? authLink.closest('.nav-auth') : null;
    const logoutNav = document.getElementById('logoutNav');
    const authBadge = document.getElementById('authBadge');
    const token = getStoredToken();

    // Hide/show the Create account / Login button based on login state
    if (authLinkNav) {
        authLinkNav.style.display = token ? 'none' : 'list-item';
    } else if (authLink) {
        authLink.style.display = token ? 'none' : 'inline-flex';
    }

    // Show/hide logout button
    if (logoutNav) {
        logoutNav.style.display = token ? 'inline-flex' : 'none';
    }
    
    // Reset auth badge (will be populated by hydrateBadge if logged in)
    if (authBadge) {
        authBadge.style.display = 'none';
        authBadge.textContent = '';
    }
}

function bindLogout() {
    const logoutLink = document.getElementById('logoutLink');
    if (!logoutLink) return;
    logoutLink.addEventListener('click', (e) => {
        e.preventDefault();
        clearStoredToken();
        updateAuthLinks();
        window.location.href = 'index.html';
    });
}

window.addEventListener('DOMContentLoaded', () => {
    updateAuthLinks();
    bindLogout();
    hydrateBadge();
});

async function hydrateBadge() {
    const token = getStoredToken();
    const authBadge = document.getElementById('authBadge');
    if (!token || !authBadge) return;
    try {
        const response = await fetch('/api/auth/me', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        if (!response.ok) {
            clearStoredToken();
            updateAuthLinks();
            return;
        }
        const me = await response.json();
        const name = [me.first_name, me.last_name].filter(Boolean).join(' ');
        // Older accounts created before name fields existed may not have
        // them set — fall back to email so the badge is never blank.
        authBadge.textContent = name || me.email || 'Logged in';
        authBadge.style.display = 'inline-flex';
    } catch {
        authBadge.style.display = 'none';
    }
}
