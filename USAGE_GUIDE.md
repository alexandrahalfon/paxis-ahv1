# Paxis Platform Usage Guide

A comprehensive walkthrough of the Paxis medical literature search platform for oncology research.

## Table of Contents

1. [Getting Started](#getting-started)
2. [User Accounts](#user-accounts)
3. [Search Preferences](#search-preferences)
4. [Q&A Mode](#qa-mode)
5. [Conversation Mode](#conversation-mode)
6. [Trial Match Mode](#trial-match-mode)
7. [Trial Finder](#trial-finder)
8. [Single Study Query](#single-study-query)
9. [Patient Matching](#patient-matching)
10. [Treatment Comparison](#treatment-comparison)
11. [Study Comparison](#study-comparison)
12. [Document Upload](#document-upload)
13. [Knowledge Base Analytics](#knowledge-base-analytics)
14. [My Saves](#my-saves)
15. [Study Details Panel](#study-details-panel)
16. [Tips and Best Practices](#tips-and-best-practices)

---

## Getting Started

Paxis is an AI-powered platform that helps oncology professionals find evidence-based answers from peer-reviewed research. The platform searches through thousands of clinical studies and synthesizes relevant information to answer your questions.

### Using Without an Account

All platform features are available without creating an account:
- Search and Q&A
- Trial Match mode
- Trial Finder (ClinicalTrials.gov search)
- Patient Matching
- Treatment and Study Comparison
- Document Upload
- Preferences and Filters
- Saving queries and favoriting studies

*However*, without an account, your data is stored only in your browser's local storage and will be lost when you close your browser or clear your cache. Creating an account ensures your data persists across sessions and devices.

### Navigation

The main navigation bar includes:
- **Home** - Main search interface
- **My Saves** - Your saved queries, cases, favorite studies, and uploads
- **Upload** - Upload your own PDFs to your personal knowledge base
- **Trial Finder** - Search ClinicalTrials.gov for matching clinical trials
- **About** - Platform information

### Creating an Account

You can use the platform without an account, but creating one unlocks:
- Personal document uploads that persist
- Saved queries and cases
- Favorite studies
- Search preferences that persist across sessions
- Alert notifications for new matching trials

Click "Create account / Login" in the top right to get started.

---

## User Accounts

### With vs Without an Account

| Feature | Without Account | With Account |
|---------|-----------------|--------------|
| Search & Q&A | Yes | Yes |
| Preferences | Yes (session only) | Yes (persistent) |
| Document Upload | Yes (session only) | Yes (persistent) |
| Saved Queries | Yes (session only) | Yes (persistent) |
| Favorite Studies | Yes (session only) | Yes (persistent) |
| Trial Alerts | No | Yes |
| Cross-device Access | No | Yes |

Without an account, all features work normally but data is stored in your browser's local storage. When you close your browser or clear cache, this data is lost. With an account, everything syncs to our secure servers and persists indefinitely.

### Why Create an Account?

With an account, you get:
- **Personal Knowledge Base**: Upload your own PDFs that get included in all searches
- **Saved Searches**: Save queries and their results for quick reference
- **Favorite Studies**: Bookmark studies you reference frequently
- **Custom Preferences**: Set filters and sorting that apply to all your searches
- **Trial Alerts**: Get notified when new trials match your saved patient cases

### Account Features

Once logged in, you'll see your email in the top right corner with a "Logout" button. All your data is securely stored and associated with your account.

---

## Search Preferences

### Accessing Preferences

Click the "Preferences" button below the navigation bar on the Home page. You'll see a toggle switch next to it showing whether your preferences are currently active (On/Off).

### Quick Toggle

The toggle next to the Preferences button lets you quickly enable or disable your saved preferences without opening the full panel. This is useful when you want to temporarily search without filters.

### Preference Options

Open the Preferences panel to configure:

**Study Filters:**
- Study Types - *multi-select* (Prospective, Retrospective, RCT, Meta-Analysis)
- Study Phases - *multi-select* (Phase I, I/II, II, II/III, III)
- Cancer Types - *multi-select* (filter to specific tumor types)
- Patient Count Range - *single value* (minimum/maximum patients)
- Treatment Modalities - *multi-select* (Surgery, Radiation, Chemotherapy, Immunotherapy, Targeted Therapy, Hormone Therapy, Brachytherapy, SBRT/SRS)
- Analysis Types - *multi-select* (Intent-to-treat, Per-protocol, Subgroup)

**Geographic Filters:**
- Countries - *multi-select with autocomplete* (search and select multiple)
- Institutions - *multi-select with autocomplete* (search and select multiple)

**Demographics:**
- Race/Ethnicity - *multi-select* (filter by reported demographics)
- Include Unknown Race - *toggle* (include studies that don't report demographics)

**Temporal Filters:**
- Publication Year Range - *single value each* (min year, max year)
- Minimum Follow-up Duration - *single value* (months)

**Evidence Quality:**
- Require Peer-Reviewed - *toggle*
- Required Outcomes - *multi-select* (OS, PFS, Local Control, DFS)

**Sorting Options** - *single select*:
- Relevance (default)
- Patient Match (best fit first based on patient characteristics)
- Population Size (larger studies ranked higher)
- Publication Date (newer studies ranked higher)
- Citation Count (highly-cited studies ranked higher)
- Outcomes (studies with OS/PFS data ranked higher)

**User Uploads:**
- Include my uploaded studies - *toggle* (include/exclude your uploads from search)

### Saving Preferences

Click "Save Preferences" to store your settings. They will automatically apply to all future searches when the toggle is On.

If you're logged in, preferences sync to your account. Without an account, preferences are saved to local storage and will be lost when you close your browser.

Click "Clear All Filters" to reset all filters to defaults.

---

## Q&A Mode

The default search mode for asking medical questions.

### How to Use

1. Type your query in the search box on the home page
2. Press Enter or click the arrow button
3. Receive a synthesized answer with supporting evidence

### What You Get

- **Short Answer**: A concise 1-2 sentence direct answer
- **Detailed Justification**: Full explanation with citations (expandable)
- **Supporting Evidence**: List of relevant studies with:
  - Study title and citation
  - Relevance score
  - Key excerpts
  - Click to view full study details

### Example Questions

- "What is the standard RT dose for breast cancer?"
- "Compare pembrolizumab vs chemotherapy for lung cancer"
- "What are the survival rates for stage III NSCLC?"
- "What is the role of SBRT in oligometastatic disease?"

### Viewing Results

Each source in the results can be clicked to open the Study Details Panel where you can:
- View full study information
- Favorite the study
- Add it to comparison

---

## Conversation Mode

Continue asking follow-up questions in context.

### Entering Conversation Mode

After receiving an answer in Q&A mode, simply type another question. The system automatically enters conversation mode, maintaining context from previous exchanges.

### How It Works

- Previous questions and answers are used as context
- Follow-up questions can reference previous topics
- The AI suggests relevant follow-up questions

### Example Flow

1. "What is the standard dose for prostate cancer SBRT?"
2. Follow-up: "What about for high-risk patients?"
3. Follow-up: "Are there any dose constraints for the rectum?"

### Returning to Q&A Mode

Click "Ask a new question" to clear the conversation and start fresh. This returns you to single Q&A mode.

### Fullscreen Mode

Click the expand icon in the chat header to enter fullscreen mode for easier reading of long conversations.

---

## Trial Match Mode

Specialized mode for matching patient cases to clinical trials from the Paxis knowledge base.

### Enabling Trial Match

Toggle the "Trial Match" switch near the search box before submitting your query.

### Input Format

Describe your patient case with relevant clinical details:
- Diagnosis and staging
- Prior treatments
- Biomarker status
- Performance status
- Relevant comorbidities

### Example Input

```
65-year-old male with newly diagnosed stage IIIA NSCLC, EGFR wild-type, PD-L1 50%, ECOG 1, no prior treatment
```

### What You Get

Trial Match mode provides a different response format:
- **Patient Summary**: Extracted clinical profile
- **Matching Trials**: Studies from the knowledge base that match the patient criteria
- **Eligibility Analysis**: Why each trial matches
- **Treatment Recommendations**: Evidence-based options

### Saving Trial Matches

Trial match results can be saved to "My Saves" with the option to set up alerts for new matching trials.

---

## Trial Finder

Search ClinicalTrials.gov for active clinical trials matching your patient.

### Accessing Trial Finder

Click "Trial Finder" in the top navigation bar (available on every page).

### How to Use

1. **Describe your patient** in the text box - include cancer type, stage, biomarkers, age, ECOG performance status, and any prior treatments
2. The system uses AI to **extract structured details** automatically
3. Optionally expand **Advanced fields & filters** to override or add details
4. Click **Search Clinical Trials** to query ClinicalTrials.gov
5. Results are **ranked by how well each trial matches** your patient profile

### Patient Description Example

```
65-year-old male with metastatic NSCLC, EGFR+, ECOG 1, progressed after osimertinib.
```

### Advanced Fields & Filters

Expand the advanced panel to manually specify or override:
- Cancer Type
- Anatomical Site
- Stage
- Histology
- Biomarkers (comma-separated)
- Gender
- Age
- Performance Status (ECOG 0-3)
- Smoking Status
- Prior Treatments (comma-separated)
- Location (City, State, Country)
- Recruiting Status (multi-select: Recruiting, Not yet recruiting, Active not recruiting, Completed)
- Phase (multi-select: Phase I, II, III, IV)

### Results

Each matching trial shows:
- Trial title and NCT number
- Match score
- Phase and status
- Brief summary
- Eligibility criteria highlights
- Link to full ClinicalTrials.gov entry

### Trial Finder vs Trial Match

| Feature | Trial Match | Trial Finder |
|---------|-------------|--------------|
| Data Source | Paxis knowledge base | ClinicalTrials.gov |
| Content | Published studies | Active/recruiting trials |
| Best For | Finding evidence | Finding enrollment opportunities |

---

## Single Study Query

Ask questions about a specific study.

### How to Access

1. Click on any study in your search results to open the Study Details Panel
2. Look for "Have questions about this trial?" section
3. Type your question about that specific study

### Use Cases

- "What were the inclusion criteria?"
- "What was the median follow-up?"
- "Were there any grade 3+ toxicities?"
- "What was the local control rate?"

### How It Works

Your question is answered using only information from that specific study, providing focused, accurate responses.

---

## Patient Matching

Find trials for a specific patient profile from the knowledge base.

### Accessing Patient Matching

Click "Patient Matching" from the home page feature buttons or navigate via the feature navigation bar.

### Input

Enter a detailed patient description including:
- Demographics (age, sex)
- Diagnosis with staging
- Histology and biomarkers
- Prior treatments
- Performance status
- Relevant medical history

### Output

- List of matching clinical trials from the knowledge base
- Match score for each trial
- Key eligibility criteria
- Treatment arms and outcomes

### Using Your Uploads

If you've uploaded documents to your personal knowledge base, they are optionally included in patient matching searches (controlled by the "Include my uploaded studies" preference).

---

## Treatment Comparison

Compare different treatment approaches.

### Accessing Treatment Comparison

Click "Treatment Comparison" from the home page feature buttons or feature navigation bar.

### How to Use

1. Enter the treatments you want to compare (e.g., "SBRT vs conventional fractionation")
2. Optionally specify the disease context
3. Submit to see a structured comparison

### Output Format

- Side-by-side comparison table
- Key outcomes for each treatment
- Supporting evidence for each claim
- Statistical comparisons where available

### Example Comparisons

- "Prostatectomy vs radiation for localized prostate cancer"
- "Concurrent vs sequential chemoradiation for NSCLC"
- "Hypofractionation vs conventional fractionation for breast cancer"

---

## Study Comparison

Compare specific studies side-by-side.

### Accessing Study Comparison

Click "Study Comparison" from the home page feature buttons or feature navigation bar.

### Input Options

You can add studies to compare in several ways:
1. **Search for Studies**: Type a study name or topic to find studies
2. **From Favorites**: Add studies you've previously favorited
3. **From Uploads**: Add your uploaded documents
4. **From Study Panel**: Click "Add to Compare" on any study details panel

### Finding Similar Studies

Select one of your uploaded studies and click "Find Similar" to automatically discover studies in the knowledge base that are similar to yours.

### Output Format

- Structured comparison table
- Key differences highlighted
- Patient populations compared
- Outcomes compared
- Methodology differences noted

### Use Cases

- Compare your institution's outcomes to published literature
- Evaluate different treatment protocols
- Identify gaps in existing research

---

## Document Upload

Build your personal knowledge base.

### Accessing Upload

Click "Upload" in the navigation bar.

### Supported Formats

- PDF documents (research papers, protocols, reports)

### How to Upload

1. Drag and drop files onto the upload area, OR
2. Click to browse and select files
3. Wait for processing to complete

### Processing

Each uploaded document goes through:
1. OCR text extraction (using Mistral AI)
2. Chunking and embedding
3. Study profile extraction
4. Storage in your personal knowledge base

### What Happens to Your Uploads

Your uploaded documents are:
- **Private**: Only visible to your account (or session if not logged in)
- **Searchable**: Included in all your searches (when preference enabled)
- **Integrated**: Used in Q&A, Trial Match, Patient Matching, and Comparisons
- **Persistent**: Stored securely with your account (or in local storage for session-only use)

Note: Without an account, uploaded documents are processed and available for your current session, but will be lost when you close your browser. Create an account to keep your uploads permanently.

### Managing Uploads

On the Upload page, you can:
- View all your uploaded documents
- See processing status
- Delete uploads
- Find similar studies to your uploads
- Add uploads to study comparison

---

## Knowledge Base Analytics

Explore aggregate statistics and meta-analysis across the entire knowledge base.

### Accessing Analytics

Click "Analytics" in the feature navigation bar on any feature page, or via the footer link on the home page.

### Overview Stats

The top of the page shows key metrics:
- Total studies in the knowledge base
- Number of cancer types covered
- Average overall survival rate
- Average follow-up duration
- Average radiation dose
- Number of RT techniques

### Custom Aggregate

Build custom queries to explore patterns:
1. **Select Metric**: OS rate, PFS rate, local control, patient count, follow-up, median age, etc.
2. **Group By**: Study phase, cancer type, study type, randomized status, metastatic status, country
3. **Aggregation**: Average, median, min, or max
4. **Filter**: Optionally filter by cancer type

Click "Run" to generate a bar chart with the results.

### Dose Distribution

View the distribution of radiation doses across studies:
- Filter by cancer type
- See histogram of total doses (Gy)

### Technique Frequency

See which radiation techniques are most common:
- Filter by cancer type
- Horizontal bar chart of studies per technique

### Outcomes by Stage

Compare outcomes across cancer stages:
- Select metric (OS, PFS, local control)
- Filter by cancer type
- See average outcomes by stage with study counts

### Meta-Analysis / Forest Plot

Generate forest plots for survival data:
- Select endpoint (OS or PFS)
- Filter by cancer type
- View per-study survival rates or hazard ratios with confidence intervals
- See pooled estimates when HR data is available

---

## My Saves

Your personal library of saved items.

### Accessing My Saves

Click "My Saves" in the navigation bar.

### Session vs Account Storage

- **With Account**: All saves sync to your account and persist forever
- **Without Account**: Saves are stored in browser local storage and lost when you close your browser or clear cache

### What You Can Save

**Saved Queries (Q&A Mode):**
- The question you asked
- Quick reference for common searches

**Saved Cases (Trial Match Mode):**
- Patient profile
- Query and response
- Supporting sources
- Option to set alerts

**Favorite Studies:**
- Studies you've bookmarked
- Quick access for reference
- Can be added to comparisons

### Setting Up Alerts

For saved trial match cases, you can enable alerts:
1. Save a trial match result
2. Toggle "Enable Alerts" on the saved case
3. Receive email notifications when new matching trials are found

Alerts would be triggered when:
- New studies are added to the knowledge base
- New trials are discovered via API that match your saved profile
(not fully implemented yet)

### Managing Saves

- Click on any saved item to view details
- Delete items you no longer need

---

## Study Details Panel

The slide-out panel for viewing full study information.

### Opening the Panel

Click on any study in search results, comparisons, or your saves.

### Panel Contents

- **Study Title and Citation**
- **Publication Details** (journal, year, DOI)
- **Study Design** (type, phase, patient count)
- **Patient Population** (eligibility, demographics)
- **Treatment Details** (arms, doses, schedules)
- **Outcomes** (survival, response rates, toxicity)
- **Key Findings**

### Panel Actions

**Favorite Button:**
- Click the heart/star icon to add to your favorites
- Favorited studies appear in My Saves
- Favorites can be used to boost search results (coming soon)

**Add to Compare:**
- Click "Add to Compare" to add the study to your comparison list
- Navigate to Study Comparison to see your comparison

**Ask Questions:**
- Use "Have questions about this trial?" to ask specific questions
- Get answers based only on that study's data

### Closing the Panel

Click outside the panel or click the X button to close.

---

## Tips and Best Practices

### Getting Better Results

1. **Be Specific**: Include relevant clinical details in your questions
2. **Use Medical Terminology**: The system understands standard oncology terms
3. **Specify Context**: Mention the disease site, stage, or treatment setting

### Using Preferences Effectively

1. **Start Broad**: Begin with minimal filters
2. **Refine Gradually**: Add filters if you're getting too many irrelevant results
3. **Toggle Off for Exploration**: Disable preferences when exploring new topics

### Building Your Knowledge Base

1. **Upload Relevant Papers**: Add papers you frequently reference
2. **Include Your Institution's Data**: Upload your own outcomes data
3. **Keep It Organized**: Your uploads are automatically categorized

### Saving Time

1. **Save Common Queries**: Save searches you run frequently
2. **Favorite Key Studies**: Bookmark studies you reference often
3. **Use Conversation Mode**: Follow up without re-entering context

### For Clinical Use

1. **Verify Critical Information**: Always verify important clinical decisions
2. **Check Original Sources**: Click through to view original study data
3. **Note Limitations**: AI synthesis may not capture all nuances

---

## Keyboard Shortcuts

- **Enter**: Submit query
- **Escape**: Close panels and modals
- **Tab**: Navigate between form fields

---

## Getting Help

If you encounter issues or have questions:
- Check the About page for contact information
- Report bugs through the feedback mechanism
- Consult your institution's support resources

---

## Privacy and Data

- Your uploaded documents are private to your account
- Search queries are not shared with other users
- Saved items are encrypted and securely stored
- You can delete your data at any time

---

*Paxis - Evidence-based answers from peer-reviewed oncology research*
