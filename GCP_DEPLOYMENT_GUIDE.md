# Paxis Backend - GCP Cloud Run Deployment Guide

## Prerequisites

1. **GCP Project Setup**
   - GCP project created
   - Billing enabled
   - Cloud Run API enabled
   - Container Registry or Artifact Registry enabled

2. **Required Tools**
   ```bash
   # Install gcloud CLI
   # https://cloud.google.com/sdk/docs/install
   
   # Install Docker
   # https://docs.docker.com/get-docker/
   ```

3. **Environment Variables**
   Required for deployment:
   - `OPENAI_API_KEY`: OpenAI API key
   - `MISTRAL_API_KEY`: Mistral API key
   - `QDRANT_URL`: Qdrant vector database URL
   - `QDRANT_API_KEY`: Qdrant API key
   - `QDRANT_COLLECTION`: Qdrant collection name (default: exueed_kb_latest)
   - `ALLOWED_ORIGINS`: CORS allowed origins (default: *)

---

## Quick Deploy (Recommended)

### Step 1: Set Variables
```bash
# Set your GCP project
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"  # or your preferred region
export SERVICE_NAME="exueed-backend"

# Set your secrets (NEVER commit these!)
export OPENAI_API_KEY="sk-..."
export MISTRAL_API_KEY="..."
export QDRANT_URL="https://..."
export QDRANT_API_KEY="..."
export QDRANT_COLLECTION="exueed_kb_latest"
```

### Step 2: Build and Push Docker Image
```bash
# Configure Docker for GCP
gcloud auth configure-docker

# Build the image
docker build -t gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest .

# Push to Container Registry
docker push gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest
```

### Step 3: Deploy to Cloud Run
```bash
gcloud run deploy ${SERVICE_NAME} \
  --image gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --min-instances 1 \
  --max-instances 10 \
  --set-env-vars "OPENAI_API_KEY=${OPENAI_API_KEY},MISTRAL_API_KEY=${MISTRAL_API_KEY},QDRANT_URL=${QDRANT_URL},QDRANT_API_KEY=${QDRANT_API_KEY},QDRANT_COLLECTION=${QDRANT_COLLECTION},ALLOWED_ORIGINS=*"
```

### Step 4: Verify Deployment
```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
  --region ${REGION} \
  --format 'value(status.url)')

echo "Service deployed at: ${SERVICE_URL}"

# Test health endpoint
curl ${SERVICE_URL}/health

# Test API endpoint
curl ${SERVICE_URL}/api/rag/health

# Open in browser
open ${SERVICE_URL}
```

---

## Detailed Deployment Steps

### 1. Prepare Environment

Create a `.env.production` file (DO NOT commit this):
```bash
OPENAI_API_KEY=sk-...
MISTRAL_API_KEY=...
QDRANT_URL=https://...
QDRANT_API_KEY=...
QDRANT_COLLECTION=exueed_kb_latest
ALLOWED_ORIGINS=*
PORT=8080
```

### 2. Test Locally with Docker

```bash
# Build image
docker build -t exueed-backend:latest .

# Run locally
docker run -p 8080:8080 \
  --env-file .env.production \
  exueed-backend:latest

# Test
curl http://localhost:8080/health
curl http://localhost:8080/api/rag/health
open http://localhost:8080
```

### 3. Push to GCP Container Registry

```bash
# Tag for GCR
docker tag exueed-backend:latest gcr.io/${PROJECT_ID}/exueed-backend:latest

# Push
docker push gcr.io/${PROJECT_ID}/exueed-backend:latest
```

### 4. Deploy with Secret Manager (More Secure)

First, create secrets:
```bash
# Create secrets in Secret Manager
echo -n "${OPENAI_API_KEY}" | gcloud secrets create openai-api-key --data-file=-
echo -n "${MISTRAL_API_KEY}" | gcloud secrets create mistral-api-key --data-file=-
echo -n "${QDRANT_API_KEY}" | gcloud secrets create qdrant-api-key --data-file=-
```

Then deploy with secrets:
```bash
gcloud run deploy exueed-backend \
  --image gcr.io/${PROJECT_ID}/exueed-backend:latest \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --min-instances 1 \
  --max-instances 10 \
  --set-secrets "OPENAI_API_KEY=openai-api-key:latest,MISTRAL_API_KEY=mistral-api-key:latest,QDRANT_API_KEY=qdrant-api-key:latest" \
  --set-env-vars "QDRANT_URL=${QDRANT_URL},QDRANT_COLLECTION=${QDRANT_COLLECTION},ALLOWED_ORIGINS=*"
```

---

## Configuration Options

### Memory and CPU
- **Default:** 2Gi RAM, 2 CPU
- **Light load:** 1Gi RAM, 1 CPU
- **Heavy load:** 4Gi RAM, 4 CPU

```bash
--memory 2Gi --cpu 2
```

### Scaling
- **Min instances:** Keep 1 to avoid cold starts
- **Max instances:** 10 for burst capacity

```bash
--min-instances 1 --max-instances 10
```

### Timeout
- **Default:** 300 seconds (5 minutes)
- For long-running queries

```bash
--timeout 300
```

### CORS
- **Development:** `ALLOWED_ORIGINS=*`
- **Production:** `ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com`

---

## Monitoring and Debugging

### View Logs
```bash
gcloud run services logs read exueed-backend \
  --region ${REGION} \
  --limit 100
```

### Stream Logs
```bash
gcloud run services logs tail exueed-backend \
  --region ${REGION}
```

### Check Service Status
```bash
gcloud run services describe exueed-backend \
  --region ${REGION}
```

### Debug Endpoints
```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe exueed-backend \
  --region ${REGION} \
  --format 'value(status.url)')

# Test all endpoints
curl ${SERVICE_URL}/
curl ${SERVICE_URL}/health
curl ${SERVICE_URL}/api/rag/health
curl ${SERVICE_URL}/docs

# Test query
curl -X POST ${SERVICE_URL}/api/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is radiation therapy?", "top_k": 5}'
```

---

## Troubleshooting

### Issue: Container fails to start
**Check:**
1. View logs: `gcloud run services logs read ...`
2. Verify all env vars are set
3. Test container locally first

### Issue: 502 Bad Gateway
**Causes:**
1. Container not listening on PORT env var
2. Health check failing
3. Container crashed

**Fix:**
- Verify Dockerfile exposes correct port
- Check health endpoint works
- Review startup logs

### Issue: CORS errors
**Fix:**
```bash
# Update ALLOWED_ORIGINS
gcloud run services update exueed-backend \
  --region ${REGION} \
  --set-env-vars "ALLOWED_ORIGINS=https://yourdomain.com"
```

### Issue: Timeout errors
**Fix:**
```bash
# Increase timeout
gcloud run services update exueed-backend \
  --region ${REGION} \
  --timeout 600
```

### Issue: Out of memory
**Fix:**
```bash
# Increase memory
gcloud run services update exueed-backend \
  --region ${REGION} \
  --memory 4Gi
```

---

## Custom Domain Setup

### 1. Map Custom Domain
```bash
gcloud run domain-mappings create \
  --service exueed-backend \
  --domain yourdomain.com \
  --region ${REGION}
```

### 2. Update DNS
Follow the instructions from the command output to add DNS records.

### 3. Update CORS
```bash
gcloud run services update exueed-backend \
  --region ${REGION} \
  --set-env-vars "ALLOWED_ORIGINS=https://yourdomain.com"
```

---

## CI/CD Pipeline (Optional)

### GitHub Actions Example
Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Cloud Run

on:
  push:
    branches: [main]

env:
  PROJECT_ID: your-project-id
  SERVICE_NAME: exueed-backend
  REGION: us-central1

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - uses: google-github-actions/setup-gcloud@v0
        with:
          service_account_key: ${{ secrets.GCP_SA_KEY }}
          project_id: ${{ env.PROJECT_ID }}
      
      - name: Build and Push
        run: |
          gcloud auth configure-docker
          docker build -t gcr.io/$PROJECT_ID/$SERVICE_NAME:$GITHUB_SHA .
          docker push gcr.io/$PROJECT_ID/$SERVICE_NAME:$GITHUB_SHA
      
      - name: Deploy
        run: |
          gcloud run deploy $SERVICE_NAME \
            --image gcr.io/$PROJECT_ID/$SERVICE_NAME:$GITHUB_SHA \
            --region $REGION \
            --platform managed
```

---

## Cost Optimization

### Estimated Costs (US Central1)
- **Min 1 instance:** ~$30-50/month
- **Min 0 instances:** Pay per request (~$0.50/million requests)
- **Storage:** Negligible for code
- **Egress:** $0.12/GB after first 1GB

### Tips to Reduce Costs
1. Set min-instances to 0 for dev/staging
2. Use Secret Manager for secrets (cheaper than env vars)
3. Set appropriate timeout (don't overallocate)
4. Use Cloud Build triggers instead of manual builds

---

## Security Best Practices

1. **Use Secret Manager** for sensitive data
2. **Enable Authentication** for admin endpoints
3. **Restrict CORS** to specific domains
4. **Use HTTPS** only (Cloud Run enforces this)
5. **Implement Rate Limiting** for API endpoints
6. **Regular Security Scans** of Docker images

---

## Health Checks

The service includes these health endpoints:

- `/health` - Basic health check
- `/api/rag/health` - RAG system health with connection tests

Cloud Run automatically uses `/health` for container health checks.

---

## Backup and Recovery

### Database (Qdrant)
- Ensure Qdrant has regular backups
- Document collection schema
- Keep embeddings reproducible

### Application State
- Cloud Run is stateless
- Use Cloud Storage for uploads
- Keep configs in version control

---

## Support and Resources

- **Cloud Run Docs:** https://cloud.google.com/run/docs
- **FastAPI Docs:** https://fastapi.tiangolo.com
- **Qdrant Docs:** https://qdrant.tech/documentation/

---

*Last updated: 2026-01-18*
