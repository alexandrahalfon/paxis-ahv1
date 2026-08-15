# Deployment Guide

This guide covers deploying Paxis to production.

## Quick Deploy Options

### Option 1: Docker Compose (Recommended for VPS/Cloud)

```bash
# 1. Set environment variables
cp .env.example .env
# Edit .env with your production API keys

# 2. Build and start
docker-compose up -d

# 3. Access
# Frontend: http://your-domain.com
# API: http://your-domain.com/api
# API Docs: http://your-domain.com/docs
```

### Option 2: GCP Cloud Run

```bash
# 1. Build Docker image
docker build -t gcr.io/YOUR_PROJECT_ID/exueed-api .

# 2. Push to GCR
docker push gcr.io/YOUR_PROJECT_ID/exueed-api

# 3. Deploy to Cloud Run
gcloud run deploy exueed-api \
  --image gcr.io/YOUR_PROJECT_ID/exueed-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="OPENAI_API_KEY=...,MISTRAL_API_KEY=...,QDRANT_URL=..."
```

### Option 3: AWS EC2/ECS

See `deployment/aws/` for detailed instructions.

### Option 4: Heroku

```bash
# 1. Install Heroku CLI
# 2. Login
heroku login

# 3. Create app
heroku create exueed-app

# 4. Set environment variables
heroku config:set OPENAI_API_KEY=...
heroku config:set MISTRAL_API_KEY=...
heroku config:set QDRANT_URL=...
heroku config:set QDRANT_API_KEY=...

# 5. Deploy
git push heroku main
```

## Production Configuration

### Environment Variables

Create `.env.production`:

```env
# API Keys
OPENAI_API_KEY=your_production_key
MISTRAL_API_KEY=your_production_key
QDRANT_URL=https://your-qdrant-instance.qdrant.io
QDRANT_API_KEY=your_qdrant_key
QDRANT_COLLECTION=exueed_kb_latest

# Models
OPENAI_MODEL=gpt-4o
MISTRAL_MODEL=pixtral-large-latest
MISTRAL_OCR_MODEL=mistral-ocr-latest

# Embedding
EMBED_MODEL=text-embedding-3-large
EMBED_DIM=3072

# RAG
RAG_TOP_K=5
RAG_TEMPERATURE=0.1

# Processing
ENABLE_PIXTRAL=true
ENABLE_FIGURE_EXTRACTION=true
ENABLE_TABLE_EXTRACTION=true
```

### Frontend Configuration

The frontend automatically detects production vs development:
- Development: Uses `http://localhost:8000`
- Production: Uses relative URLs (`/api/...`)

No changes needed - it works automatically!

## Deployment Checklist

- [ ] Set all environment variables in production
- [ ] Update Qdrant collection name if needed
- [ ] Test API endpoints
- [ ] Test frontend on production URL
- [ ] Set up SSL/HTTPS (use Let's Encrypt)
- [ ] Configure domain name
- [ ] Set up monitoring/logging
- [ ] Backup Qdrant database
- [ ] Test document upload/processing
- [ ] Test RAG queries

## Post-Deployment

1. **Test API Health**
   ```bash
   curl https://your-domain.com/api/rag/health
   ```

2. **Test Frontend**
   - Visit https://your-domain.com
   - Test AI query
   - Test patient matching
   - Test treatment comparison

3. **Monitor Logs**
   ```bash
   docker-compose logs -f api
   ```

## Troubleshooting

### API Not Responding
- Check if API container is running: `docker ps`
- Check logs: `docker-compose logs api`
- Verify environment variables

### Frontend Can't Connect to API
- Check nginx configuration
- Verify API_BASE_URL in browser console
- Check CORS settings

### PDF Processing Fails
- Verify poppler-utils is installed in container
- Check Mistral API key
- Review processing logs

## Security Notes

1. **Never commit `.env` files**
2. **Use environment variables** for all secrets
3. **Enable HTTPS** in production
4. **Restrict admin page** access (add authentication)
5. **Rate limit** API endpoints
6. **Monitor** API usage and costs
