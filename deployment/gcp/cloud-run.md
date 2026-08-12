# Deploy to Google Cloud Run

## Prerequisites

- Google Cloud account
- gcloud CLI installed
- Docker installed

## Steps

1. **Set up GCP project**
   ```bash
   gcloud config set project YOUR_PROJECT_ID
   gcloud services enable cloudbuild.googleapis.com
   gcloud services enable run.googleapis.com
   ```

2. **Build and push image**
   ```bash
   # Build
   docker build -t gcr.io/YOUR_PROJECT_ID/exueed-api .
   
   # Push
   docker push gcr.io/YOUR_PROJECT_ID/exueed-api
   ```

3. **Deploy to Cloud Run**
   ```bash
   gcloud run deploy exueed-api \
     --image gcr.io/YOUR_PROJECT_ID/exueed-api \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --port 8000 \
     --memory 4Gi \
     --cpu 2 \
     --timeout 300 \
     --set-env-vars="OPENAI_API_KEY=...,MISTRAL_API_KEY=...,QDRANT_URL=..."
   ```

4. **Deploy frontend to Cloud Storage + Cloud CDN**
   ```bash
   gsutil -m cp -r frontend/* gs://YOUR_BUCKET_NAME/
   gsutil web set -m index.html -e index.html gs://YOUR_BUCKET_NAME
   ```

## Environment Variables

Set via Cloud Run console or CLI:
- `OPENAI_API_KEY`
- `MISTRAL_API_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY`
- `QDRANT_COLLECTION`
