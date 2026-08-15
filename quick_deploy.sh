#!/bin/bash
#
# Quick Deployment Script for Paxis Backend to GCP Cloud Run
# 
# Usage: ./quick_deploy.sh
#
# Prerequisites:
# - gcloud CLI installed and authenticated
# - Docker installed
# - Environment variables set (see below)
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Paxis Backend - GCP Cloud Run Deployment${NC}"
echo -e "${GREEN}========================================${NC}\n"

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    if ! command -v gcloud &> /dev/null; then
        echo -e "${RED}❌ gcloud CLI not found. Please install: https://cloud.google.com/sdk/docs/install${NC}"
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker not found. Please install: https://docs.docker.com/get-docker/${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ All prerequisites met${NC}\n"
}

# Get configuration
get_config() {
    echo -e "${YELLOW}Configuration:${NC}"
    
    # GCP Project
    if [ -z "$PROJECT_ID" ]; then
        PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
        if [ -z "$PROJECT_ID" ]; then
            echo -e "${RED}❌ No GCP project set. Run: gcloud config set project YOUR_PROJECT_ID${NC}"
            exit 1
        fi
    fi
    echo "  Project ID: $PROJECT_ID"
    
    # Region
    REGION="${REGION:-us-central1}"
    echo "  Region: $REGION"
    
    # Service name
    SERVICE_NAME="${SERVICE_NAME:-exueed-backend}"
    echo "  Service Name: $SERVICE_NAME"
    
    # Check required env vars
    if [ -z "$OPENAI_API_KEY" ]; then
        echo -e "${RED}❌ OPENAI_API_KEY not set${NC}"
        exit 1
    fi
    
    if [ -z "$QDRANT_URL" ]; then
        echo -e "${RED}❌ QDRANT_URL not set${NC}"
        exit 1
    fi
    
    if [ -z "$QDRANT_API_KEY" ]; then
        echo -e "${RED}❌ QDRANT_API_KEY not set${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Configuration valid${NC}\n"
}

# Build Docker image
build_image() {
    echo -e "${YELLOW}Building Docker image...${NC}"
    
    IMAGE_TAG="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"
    
    docker build -t ${IMAGE_TAG} . || {
        echo -e "${RED}❌ Docker build failed${NC}"
        exit 1
    }
    
    echo -e "${GREEN}✅ Image built: ${IMAGE_TAG}${NC}\n"
}

# Push image to GCR
push_image() {
    echo -e "${YELLOW}Pushing image to Google Container Registry...${NC}"
    
    # Configure Docker
    gcloud auth configure-docker --quiet || {
        echo -e "${RED}❌ Failed to configure Docker${NC}"
        exit 1
    }
    
    # Push
    docker push ${IMAGE_TAG} || {
        echo -e "${RED}❌ Docker push failed${NC}"
        exit 1
    }
    
    echo -e "${GREEN}✅ Image pushed to GCR${NC}\n"
}

# Deploy to Cloud Run
deploy_service() {
    echo -e "${YELLOW}Deploying to Cloud Run...${NC}"
    
    # Set default values for optional env vars
    MISTRAL_API_KEY="${MISTRAL_API_KEY:-}"
    QDRANT_COLLECTION="${QDRANT_COLLECTION:-exueed_kb_latest}"
    ALLOWED_ORIGINS="${ALLOWED_ORIGINS:-*}"
    
    gcloud run deploy ${SERVICE_NAME} \
        --image ${IMAGE_TAG} \
        --platform managed \
        --region ${REGION} \
        --allow-unauthenticated \
        --port 8080 \
        --memory 2Gi \
        --cpu 2 \
        --timeout 300 \
        --min-instances 1 \
        --max-instances 10 \
        --set-env-vars "OPENAI_API_KEY=${OPENAI_API_KEY},MISTRAL_API_KEY=${MISTRAL_API_KEY},QDRANT_URL=${QDRANT_URL},QDRANT_API_KEY=${QDRANT_API_KEY},QDRANT_COLLECTION=${QDRANT_COLLECTION},ALLOWED_ORIGINS=${ALLOWED_ORIGINS}" || {
        echo -e "${RED}❌ Deployment failed${NC}"
        exit 1
    }
    
    echo -e "${GREEN}✅ Service deployed${NC}\n"
}

# Get service URL
get_service_url() {
    SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
        --region ${REGION} \
        --format 'value(status.url)' 2>/dev/null)
    
    if [ -z "$SERVICE_URL" ]; then
        echo -e "${RED}❌ Could not get service URL${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Deployment Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Service URL: ${SERVICE_URL}"
    echo ""
    echo "Test endpoints:"
    echo "  Health: ${SERVICE_URL}/health"
    echo "  RAG Health: ${SERVICE_URL}/api/rag/health"
    echo "  API Docs: ${SERVICE_URL}/docs"
    echo "  Frontend: ${SERVICE_URL}/"
    echo ""
    echo -e "${YELLOW}Run verification:${NC}"
    echo "  python verify_deployment.py ${SERVICE_URL}"
    echo ""
}

# Main execution
main() {
    check_prerequisites
    get_config
    build_image
    push_image
    deploy_service
    get_service_url
}

# Run
main
