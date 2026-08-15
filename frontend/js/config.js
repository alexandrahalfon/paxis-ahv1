/**
 * Frontend configuration for API endpoints
 * Automatically detects production vs development environment
 */

const CONFIG = (() => {
    const hostname = window.location.hostname;
    const isDevelopment = hostname === 'localhost' || hostname === '127.0.0.1';
    
    return {
        isDevelopment,
        API_BASE: isDevelopment ? 'http://localhost:8000/api' : '/api',
        API_DOCS: isDevelopment ? 'http://localhost:8000/docs' : '/docs',
        API_HEALTH: isDevelopment ? 'http://localhost:8000/api/rag/health' : '/api/rag/health',
        API_STATS: isDevelopment ? 'http://localhost:8000/api/rag/stats' : '/api/rag/stats'
    };
})();
