# Comprehensive Test Suite Guide

## Overview

The comprehensive test suite (`comprehensive_test_suite.py`) tests all features of the Paxis platform including:

- **RAG Queries**: Various question types with different complexities
- **Query Modes**: All available query modes (naive, local, global, hybrid)
- **Enhanced Queries**: Short answer generation
- **Deep Dive Queries**: Site-specific comprehensive searches
- **Patient Matching**: Various patient profiles
- **Treatment Comparison**: Side-by-side treatment analysis
- **Utility Endpoints**: Health checks, modes, sites

## Prerequisites

1. **Backend API Running**: Make sure the API server is running
   ```bash
   python run_api.py
   ```
   Or if using the frontend server:
   ```bash
   python run_api.py  # Terminal 1
   python -m http.server 3000 --directory frontend  # Terminal 2
   ```

2. **Dependencies**: Ensure `requests` is installed
   ```bash
   pip install requests
   ```

## Running the Tests

### Basic Usage

```bash
# Test against default localhost:8000
python comprehensive_test_suite.py

# Test against custom URL
python comprehensive_test_suite.py http://localhost:8000

# Test against production/staging
python comprehensive_test_suite.py https://your-api-url.com
```

### What Gets Tested

#### 1. RAG Queries (50+ test cases)
- Simple questions
- Complex questions with abbreviations (NSCLC, HER2+, etc.)
- Dose-specific questions
- Outcome questions (OS, PFS, DFS)
- Treatment recommendation questions
- Comparison questions
- Staging questions
- Indication questions
- Very detailed/complex questions

#### 2. Query Modes (4 tests)
- Naive mode
- Local mode
- Global mode
- Hybrid mode (recommended)

#### 3. Site Inference (3 tests)
- Queries with automatic site inference enabled

#### 4. Enhanced Queries (4 tests)
- Short answer generation
- Justification with citations

#### 5. Deep Dive Queries (3 tests)
- Site-specific comprehensive searches
- Auto-inference of tumor sites

#### 6. Patient Matching (6 test cases)
- Early-stage breast cancer patient
- Advanced lung cancer patient
- Prostate cancer patient
- Head and neck cancer patient
- Minimal profile patient
- Complex profile with comorbidities

#### 7. Treatment Comparison (6 test cases)
- IMRT vs 3D-CRT for prostate
- Adjuvant RT vs observation for breast
- SBRT vs conventional RT for lung
- Chemoradiation vs surgery for rectal
- IMRT vs VMAT for head and neck
- General comparisons

#### 8. Utility Endpoints (5 tests)
- Health check
- Query modes endpoint
- Available sites endpoint
- Root endpoint
- General health endpoint

**Total: ~80+ comprehensive test cases**

## Test Report

After running, the script generates two files:

1. **HTML Report**: `comprehensive_test_report_YYYYMMDD_HHMMSS.html`
   - Interactive report with expandable test cases
   - Color-coded pass/fail indicators
   - Full request/response details
   - Answers displayed for each query
   - Summary statistics

2. **JSON Report**: `comprehensive_test_report_YYYYMMDD_HHMMSS.json`
   - Machine-readable format
   - Complete test data
   - Suitable for CI/CD integration

## Report Features

### HTML Report Includes:
- **Summary Dashboard**: Total tests, pass/fail counts, average response time
- **Categorized Results**: Tests grouped by feature
- **Expandable Test Cases**: Click to see full details
- **Answer Display**: Shows generated answers for each query
- **Error Details**: Full error messages for failed tests
- **Response Metadata**: Query types, expanded queries, evidence counts
- **Timing Information**: Response times for each test

### Viewing the Report

Simply open the HTML file in any web browser:
```bash
open comprehensive_test_report_*.html
# or
python -m http.server 8001
# Then open http://localhost:8001/comprehensive_test_report_*.html
```

## Test Coverage

The test suite covers:

✅ **All API Endpoints**
- `/api/rag/query`
- `/api/rag/query/enhanced`
- `/api/rag/deep-dive`
- `/api/rag/patient/match`
- `/api/rag/comparison/treatments`
- `/api/rag/health`
- `/api/rag/modes`
- `/api/rag/sites`

✅ **All Query Types**
- Dose questions
- Outcome questions
- Treatment recommendations
- Comparisons
- Staging questions
- Indication questions

✅ **All Query Modes**
- Naive
- Local
- Global
- Hybrid

✅ **Patient Matching**
- Various cancer types
- Different stages
- With/without molecular markers
- With/without comorbidities

✅ **Treatment Comparison**
- Different treatment pairs
- With/without cancer type filters
- Various stages

## Expected Duration

- **Full Test Suite**: ~5-10 minutes (depending on API response times)
- **Individual Categories**: 30 seconds - 2 minutes each

## Troubleshooting

### Tests Failing

1. **Check API is running**: Ensure `python run_api.py` is running
2. **Check URL**: Verify the base URL matches your API server
3. **Check Environment Variables**: Ensure all API keys are set
4. **Check Network**: Verify connectivity to Qdrant and OpenAI

### Slow Response Times

- Normal: 2-10 seconds per query
- Deep dive queries: 10-30 seconds
- Patient matching: 5-15 seconds
- Treatment comparison: 10-20 seconds

### Rate Limiting

The script includes 0.5 second delays between tests to avoid rate limiting. If you encounter rate limits:
- Increase delays in the script
- Run tests in smaller batches
- Check API rate limits

## Customization

You can modify the test suite to:
- Add custom test cases
- Change test parameters
- Adjust timeouts
- Modify report format

Edit `comprehensive_test_suite.py` to customize.

## Integration with CI/CD

The JSON report can be integrated into CI/CD pipelines:

```bash
# Run tests and check exit code
python comprehensive_test_suite.py
if [ $? -eq 0 ]; then
    echo "Tests passed"
else
    echo "Tests failed"
    exit 1
fi
```

## Notes

- Tests are designed to be comprehensive but not exhaustive
- Some tests may fail if the knowledge base doesn't contain relevant documents
- Response times vary based on API load and network conditions
- The report shows both successful and failed tests for debugging
