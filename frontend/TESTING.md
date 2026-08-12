# Testing the Paxis Frontend

## Quick Start Guide

### Step 1: Start the Backend API Server

Open a terminal and run:

```bash
cd /Users/aysha/Desktop/Paxis-backend
python run_api.py
```

You should see:
```
======================================================================
Paxis Medical Literature API
======================================================================
Starting server on http://0.0.0.0:8000
API docs available at http://localhost:8000/docs
======================================================================
```

**Keep this terminal window open** - the API server needs to keep running.

### Step 2: Start the Frontend Server

Open a **new terminal window** and run:

```bash
cd /Users/aysha/Desktop/Paxis-backend/frontend
python -m http.server 8080
```

You should see:
```
Serving HTTP on 0.0.0.0 port 8080 (http://0.0.0.0:8080/) ...
```

### Step 3: Open in Browser

Open your web browser and go to:

```
http://localhost:8080
```

You should see the Paxis home page with:
- Header navigation
- "AI Agent" and "Features" tabs
- Chat interface in the AI Agent tab

## Testing Features

### 1. AI Agent Chat (Home Page)

1. Make sure you're on the home page (`http://localhost:8080`)
2. The "AI Agent" tab should be active by default
3. Try different query modes:
   - Click "Basic", "Agent", "Naive", "Local", "Global", or "Hybrid"
4. Type a question in the chat input, for example:
   - "What are the side effects of pembrolizumab?"
   - "Compare pembrolizumab vs chemotherapy for lung cancer"
   - "What treatments are effective for HER2-positive breast cancer?"
5. Click "Send" or press Enter
6. Wait for the AI response with citations

### 2. Features Tab

1. Click the "Features" tab on the home page
2. You'll see three feature cards:
   - AI-Powered Query
   - Patient Matching
   - Treatment Comparison
3. Click any button to navigate to that feature's dedicated page

### 3. Patient Matching

1. Click "Match Patients" button or go to `http://localhost:8080/patient-matching.html`
2. Fill in patient characteristics:
   - Age: 65
   - Gender: Male
   - Cancer Stage: III
   - Histology: adenocarcinoma
   - Molecular Markers: PD-L1+
   - Performance Status: ECOG 1
   - Smoking Status: former
3. Click "Find Matching Studies"
4. View matching studies with confidence scores

### 4. Treatment Comparison

1. Click "Compare Treatments" button or go to `http://localhost:8080/treatment-comparison.html`
2. Enter two treatments:
   - Treatment A: pembrolizumab
   - Treatment B: chemotherapy
   - (Optional) Cancer Type: lung cancer
   - (Optional) Stage: III
3. Click "Compare Treatments"
4. View side-by-side comparison with statistical significance

### 5. Dedicated Query Page

1. Click "AI Query" in navigation or go to `http://localhost:8080/query.html`
2. Similar to home page chat but with example questions
3. Test all query modes

## Troubleshooting

### API Server Not Running

**Error:** "Failed to connect" or "API not available"

**Solution:**
- Make sure the backend API server is running on port 8000
- Check terminal for errors
- Verify `.env` file exists with correct API keys

### Frontend Not Loading

**Error:** Page doesn't load or shows errors

**Solution:**
- Make sure the frontend server is running on port 8080
- Check browser console for errors (F12)
- Verify all files are in the `frontend/` directory

### CORS Errors

**Error:** "CORS policy" or "blocked by CORS"

**Solution:**
- The backend should already have CORS enabled
- If issues persist, check `src/api/main.py` for CORS configuration

### No Results Returned

**Error:** "No matching studies" or empty results

**Solution:**
- This is normal if the query doesn't match any documents
- Try broader queries or different search terms
- Check that documents have been ingested into Qdrant

## Alternative: Using Node.js Server

If you prefer Node.js:

```bash
cd /Users/aysha/Desktop/Paxis-backend/frontend
npx serve -p 8080
```

## Alternative: Using PHP Server

If you have PHP installed:

```bash
cd /Users/aysha/Desktop/Paxis-backend/frontend
php -S localhost:8080
```

## Testing Checklist

- [ ] Backend API server running on port 8000
- [ ] Frontend server running on port 8080
- [ ] Home page loads with tabs
- [ ] AI Agent chat works
- [ ] Can switch between query modes
- [ ] Patient matching form works
- [ ] Treatment comparison works
- [ ] Citations display correctly
- [ ] Navigation links work
- [ ] All pages load without errors

## Quick Test Commands

Test API health:
```bash
curl http://localhost:8000/api/rag/health
```

Test API query:
```bash
curl -X POST "http://localhost:8000/api/rag/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the side effects of pembrolizumab?", "mode": "basic", "top_k": 5}'
```

## Notes

- The frontend connects to `http://localhost:8000` by default
- To change the API URL, edit `frontend/js/api.js` and update `API_BASE_URL`
- All pages use the unified design system
- No emojis or gradients - clean, professional design
- Fully responsive for mobile and desktop
