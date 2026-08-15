# Paxis Frontend

A modern, unified frontend for the Paxis medical literature AI platform.

## Features

- **Unified Design System** - Consistent styling across all pages
- **AI Query Interface** - Multiple query modes (basic, agent, naive, local, global, hybrid)
- **Patient Matching** - Find studies matching patient characteristics
- **Treatment Comparison** - Side-by-side treatment analysis
- **Responsive Design** - Works on desktop, tablet, and mobile
- **Modern UI/UX** - Clean, professional interface

## Structure

```
frontend/
├── css/
│   └── styles.css          # Unified design system
├── js/
│   ├── api.js              # API client for backend
│   └── utils.js            # Utility functions
├── index.html              # Landing page
├── query.html              # AI query interface
├── patient-matching.html   # Patient matching interface
├── treatment-comparison.html # Treatment comparison interface
└── README.md               # This file
```

## Getting Started

### Option 1: Simple HTTP Server (Recommended)

```bash
cd frontend
python -m http.server 8080
```

Then open: `http://localhost:8080`

### Option 2: Using Node.js

```bash
cd frontend
npx serve -p 8080
```

### Option 3: Using PHP

```bash
cd frontend
php -S localhost:8080
```

## Configuration

The frontend connects to the backend API at `http://localhost:8000` by default.

To change the API URL, edit `frontend/js/api.js`:

```javascript
const API_BASE_URL = 'http://your-api-url:8000/api/rag';
```

## Pages

### 1. Home (`index.html`)
- Landing page with feature overview
- Quick start guide
- Links to all features

### 2. AI Query (`query.html`)
- Chat interface for asking questions
- Mode selector (basic, agent, naive, local, global, hybrid)
- Example questions
- Source citations

### 3. Patient Matching (`patient-matching.html`)
- Form for patient characteristics
- Age, gender, cancer stage, histology
- Molecular markers, performance status
- Matching studies with scores

### 4. Treatment Comparison (`treatment-comparison.html`)
- Compare two treatments
- Optional cancer type and stage filters
- Side-by-side comparison results
- Statistical significance indicators

## Design System

The frontend uses a unified design system with:

- **Colors**: Primary blue, secondary green, accent orange
- **Typography**: System fonts for optimal performance
- **Spacing**: Consistent spacing scale
- **Components**: Reusable card, button, form components
- **Responsive**: Mobile-first design

## API Integration

All pages use the `PaxisAPI` class from `js/api.js`:

```javascript
const api = new PaxisAPI();

// Query
const result = await api.query('What are pembrolizumab side effects?', 'agent', 5);

// Patient matching
const matches = await api.matchPatient({
    age: 65,
    gender: 'male',
    cancer_stage: 'III'
});

// Treatment comparison
const comparison = await api.compareTreatments('pembrolizumab', 'chemotherapy', 'lung cancer', 'III');
```

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Safari (latest)
- Mobile browsers (iOS Safari, Chrome Mobile)

## Development

### Making Changes

1. **Styles**: Edit `css/styles.css` - all styles are in one file
2. **API**: Edit `js/api.js` for API changes
3. **Utilities**: Edit `js/utils.js` for helper functions
4. **Pages**: Edit individual HTML files

### Testing

1. Start the backend API server:
   ```bash
   python run_api.py
   ```

2. Start a simple HTTP server for the frontend:
   ```bash
   cd frontend
   python -m http.server 8080
   ```

3. Open `http://localhost:8080` in your browser

## Features Comparison

| Feature | Old Frontend | New Frontend |
|---------|-------------|--------------|
| Design System | Inconsistent | Unified |
| Styles | Per-page | Shared CSS |
| API Client | Scattered | Centralized |
| Responsive | Partial | Full |
| Code Quality | Mixed | Clean |
| Maintainability | Low | High |

## Next Steps

- Add authentication (if needed)
- Add user preferences/settings
- Add export functionality (PDF, CSV)
- Add advanced filtering options
- Add visualization charts for comparisons
