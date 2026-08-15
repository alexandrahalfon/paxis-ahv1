# Deploy to Heroku

## Prerequisites

- Heroku account
- Heroku CLI installed

## Steps

1. **Login to Heroku**
   ```bash
   heroku login
   ```

2. **Create app**
   ```bash
   heroku create exueed-app
   ```

3. **Set environment variables**
   ```bash
   heroku config:set OPENAI_API_KEY=your_key
   heroku config:set MISTRAL_API_KEY=your_key
   heroku config:set QDRANT_URL=your_url
   heroku config:set QDRANT_API_KEY=your_key
   heroku config:set QDRANT_COLLECTION=exueed_kb_latest
   ```

4. **Add buildpacks**
   ```bash
   heroku buildpacks:add heroku/python
   heroku buildpacks:add https://github.com/heroku/heroku-buildpack-apt
   ```

5. **Create Aptfile for poppler**
   ```bash
   echo "poppler-utils" > Aptfile
   ```

6. **Deploy**
   ```bash
   git push heroku main
   ```

7. **Deploy frontend separately**
   - Use Heroku static buildpack or separate service
   - Or deploy to Netlify/Vercel
