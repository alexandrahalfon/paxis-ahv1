# GCP DEPLOY — Updated for Paxis v2

## 1. Upload zipped file to Cloud Shell

Upload your zip to Cloud Shell, then:

```bash
unzip "filename.zip"
```

Run `ls` to check inflated content.

## 2. Add missing packages (if any)

```bash
echo "asyncpg==0.30.0" >> requirements/prod.txt
echo "reportlab>=4.0.0" >> requirements/prod.txt
echo "PyMuPDF>=1.23.0" >> requirements/prod.txt
```

## 3. Set + export secrets

```bash
export PROJECT_ID=$(gcloud config get-value project)

export MISTRAL_API_KEY=${MISTRAL_API_KEY}
export MISTRAL_MODEL=pixtral-large-latest
export MISTRAL_OCR_MODEL=mistral-ocr-latest

export OPENAI_API_KEY=${OPENAI_API_KEY}
export OPENAI_MODEL=gpt-4o
export STUDY_PROFILE_MODEL=gpt-4o-mini

export QDRANT_URL=https://ffbfc406-406a-4f3a-a0e4-8e3fa754d378.us-east4-0.gcp.cloud.qdrant.io
export QDRANT_API_KEY=${QDRANT_API_KEY}
export QDRANT_COLLECTION=exueed_kb_latest

export POSTGRES_HOST=34.21.60.224
export POSTGRES_PORT=5432
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
export POSTGRES_DATABASE=display-study-details

export CACHE_POSTGRES_HOST=34.21.60.224
export CACHE_POSTGRES_PORT=5432
export CACHE_POSTGRES_USER=postgres
export CACHE_POSTGRES_PASSWORD=${CACHE_POSTGRES_PASSWORD}
export CACHE_POSTGRES_DATABASE=exueed_cache

export PATIENTS_POSTGRES_HOST=34.21.60.224
export PATIENTS_POSTGRES_PORT=5432
export PATIENTS_POSTGRES_USER=postgres
export PATIENTS_POSTGRES_PASSWORD=${PATIENTS_POSTGRES_PASSWORD}
export PATIENTS_POSTGRES_DATABASE=exueed-patients

export HF_TOKEN=${HF_TOKEN}
export GCP_BUCKET_NAME=processed-documents_01-2026
export AUTO_SYNC_GCP=true

export EMBED_MODEL=text-embedding-3-large
export EMBED_DIM=3072
export MAX_TOKENS_PER_CHUNK=600
export EMBED_BATCH_SIZE=64
export QDRANT_BATCH_SIZE=512
export AUTO_INGEST=true

export AUTH_SECRET_KEY=${AUTH_SECRET_KEY}
```

## 4. Upload to Cloud Build

```bash
gcloud builds submit --tag gcr.io/paxis-prod/paxis-deploy:latest .
```

## 5. Run Cloud Deploy

```bash
gcloud run deploy paxis-deploy \
  --image gcr.io/paxis-prod/paxis-deploy:latest \
  --platform managed \
  --region us-east4 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 4Gi \
  --cpu 4 \
  --timeout 600 \
  --min-instances 1 \
  --max-instances 10 \
  --set-env-vars OPENAI_API_KEY=${OPENAI_API_KEY},OPENAI_MODEL=${OPENAI_MODEL},STUDY_PROFILE_MODEL=${STUDY_PROFILE_MODEL},MISTRAL_API_KEY=${MISTRAL_API_KEY},MISTRAL_MODEL=${MISTRAL_MODEL},MISTRAL_OCR_MODEL=${MISTRAL_OCR_MODEL},QDRANT_URL=${QDRANT_URL},QDRANT_API_KEY=${QDRANT_API_KEY},QDRANT_COLLECTION=${QDRANT_COLLECTION},EMBED_MODEL=${EMBED_MODEL},EMBED_DIM=${EMBED_DIM},MAX_TOKENS_PER_CHUNK=${MAX_TOKENS_PER_CHUNK},EMBED_BATCH_SIZE=${EMBED_BATCH_SIZE},QDRANT_BATCH_SIZE=${QDRANT_BATCH_SIZE},POSTGRES_HOST=${POSTGRES_HOST},POSTGRES_PORT=${POSTGRES_PORT},POSTGRES_USER=${POSTGRES_USER},POSTGRES_PASSWORD=${POSTGRES_PASSWORD},POSTGRES_DATABASE=${POSTGRES_DATABASE},CACHE_POSTGRES_HOST=${CACHE_POSTGRES_HOST},CACHE_POSTGRES_PORT=${CACHE_POSTGRES_PORT},CACHE_POSTGRES_USER=${CACHE_POSTGRES_USER},CACHE_POSTGRES_PASSWORD=${CACHE_POSTGRES_PASSWORD},CACHE_POSTGRES_DATABASE=${CACHE_POSTGRES_DATABASE},PATIENTS_POSTGRES_HOST=${PATIENTS_POSTGRES_HOST},PATIENTS_POSTGRES_PORT=${PATIENTS_POSTGRES_PORT},PATIENTS_POSTGRES_USER=${PATIENTS_POSTGRES_USER},PATIENTS_POSTGRES_PASSWORD=${PATIENTS_POSTGRES_PASSWORD},PATIENTS_POSTGRES_DATABASE=${PATIENTS_POSTGRES_DATABASE},AUTH_SECRET_KEY=${AUTH_SECRET_KEY},HF_TOKEN=${HF_TOKEN},GCP_BUCKET_NAME=${GCP_BUCKET_NAME},AUTO_SYNC_GCP=${AUTO_SYNC_GCP},AUTO_INGEST=${AUTO_INGEST},GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcp-key.json,ALLOWED_ORIGINS=* \
  --set-secrets=/secrets/gcp-key.json=gcp-service-account-key:latest
```

## Notes

- The `exueed-patients` database must exist on the production Postgres server (`34.21.60.224`). If it doesn't, create it before deploying:
  ```sql
  CREATE DATABASE "exueed-patients";
  ```
  The app's `ensure_schema()` will create all tables on first request.

- The deploy uses `34.21.60.224` for all three databases (the original production server from your previous deploys). Cloud Run can reach this IP because it's on the same GCP VPC — your local machine can't because your IP isn't authorized in Cloud SQL networking.
