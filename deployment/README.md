# Deployment Options

## Quick Start: Docker Compose

The easiest way to deploy is using Docker Compose:

```bash
# 1. Set up environment
cp .env.production.example .env
# Edit .env with your production keys

# 2. Deploy
./deploy.sh

# 3. Access
# Frontend: http://your-server-ip
# API: http://your-server-ip:8000
# API Docs: http://your-server-ip:8000/docs
```

## Platform-Specific Guides

### GCP Cloud Run
See `gcp/cloud-run.md`

### AWS EC2/ECS
See `aws/ec2-deploy.md`

### Heroku
See `heroku/deploy.md`

### DigitalOcean App Platform
See `digitalocean/app-platform.md`

## Production Checklist

- [ ] Environment variables configured
- [ ] SSL/HTTPS enabled
- [ ] Domain name configured
- [ ] Qdrant collection accessible
- [ ] API keys valid
- [ ] Frontend accessible
- [ ] API health check passing
- [ ] Document upload working
- [ ] RAG queries working
