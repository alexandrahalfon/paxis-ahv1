# Paxis Deploy Guide

One-page reference for shipping any change (frontend, backend, or DB) to `patientaxis.net`. Follow this same sequence every time so nothing gets skipped.

**Project:** `paxis-prod` (GCP) &nbsp;•&nbsp; **Service:** `paxis-backend` (Cloud Run) &nbsp;•&nbsp; **Region:** `us-central1` &nbsp;•&nbsp; **Domain:** `patientaxis.net`

---

## 0. Before you start

- [ ] Changes tested locally (or at minimum, read through the diff — `git diff`)
- [ ] No secrets, API keys, or `credentials/` files staged (`git status` should never show these — they're gitignored)
- [ ] If you touched anything DB-related: see [DB changes](#db-changes) below *before* deploying
- [ ] Docker Desktop is open and running (check the whale icon in the menu bar — steady, not animating)

---

## 1. Commit and push

```bash
cd ~/Desktop/exueedrevamp
git status                     # sanity check — see what you're about to ship
git add -A
git commit -m "Describe the change here"
git push
```

You may see a notice that the repo "moved" to `paxisv1.git` — that's just informational, the push still succeeds against the old remote. No action needed (though at some point it's worth updating the remote with `git remote set-url origin git@github.com:aysha-alv/paxisv1.git`).

---

## 2. Build the Docker image

```bash
docker build --platform linux/amd64 -t us-central1-docker.pkg.dev/paxis-prod/paxis-repo/paxis-backend:latest .
```

`--platform linux/amd64` is required if you're on Apple Silicon (M1/M2/M3) — Cloud Run needs an amd64 image, and Docker on ARM Macs defaults to arm64. Takes 2–5 minutes depending on what's cached.

**Important:** `docker build` copies your entire working directory (minus `.dockerignore`), not just what's committed to git. Always commit first anyway, so the image and the git history stay in sync.

---

## 3. Push the image

```bash
docker push us-central1-docker.pkg.dev/paxis-prod/paxis-repo/paxis-backend:latest
```

---

## 4. Deploy to Cloud Run

```bash
gcloud run deploy paxis-backend \
  --image us-central1-docker.pkg.dev/paxis-prod/paxis-repo/paxis-backend:latest \
  --region us-central1 \
  --min-instances 1
```

`--min-instances 1` keeps one instance warm to avoid cold-start delay on the first request after idle. `patientaxis.net` is mapped directly to this Cloud Run service (not a specific revision), so the new revision goes live under the real domain automatically once it's serving 100% of traffic — no separate domain step needed.

---

## 5. Verify

- [ ] `patientaxis.net` loads and looks right
- [ ] Whatever you specifically changed actually shows up (hard-refresh / incognito to rule out caching)
- [ ] Check logs for errors: `gcloud run services logs read paxis-backend --region us-central1 --limit 50`

---

## Rollback

If something's broken, Cloud Run keeps prior revisions. Roll back traffic without rebuilding:

```bash
gcloud run revisions list --service paxis-backend --region us-central1
gcloud run services update-traffic paxis-backend --region us-central1 --to-revisions REVISION_NAME=100
```

(Or do this from the Cloud Run Console: Service → Revisions → select the good one → Manage Traffic.)

---

## DB changes

The app connects to Postgres (Cloud SQL) via a Unix socket in production, configured through Cloud Run's own environment variables — **not** your local `.env`. Your local `.env` and production config are entirely separate; editing `.env` locally never touches prod.

Before deploying anything that touches schema or queries:
- [ ] Test the change against a local/dev DB connection first if at all possible
- [ ] Schema changes (new columns, tables, migrations) should be additive/backward-compatible where possible, so the old code path doesn't break mid-deploy while the new revision is still rolling out
- [ ] Double check `src/core/config.py` if you're touching connection settings — there's a fallback pattern (`cache_postgres_host or postgres_host`, etc.) that's easy to accidentally break

Production Postgres env vars (host, user, port, etc.) live in the Cloud Run service config under **Variables & Secrets** in the GCP Console — not in this repo.

---

## Where things live

| What | Where |
|---|---|
| Frontend static files | `frontend/` — served directly by FastAPI (`src/api/main.py`, mounted at `/`) |
| Backend/API code | `src/api/` |
| GCP service account key | `credentials/` (gitignored, never commit) |
| Production secrets/env vars | Cloud Run Console → `paxis-backend` → Variables & Secrets |
| Local dev env vars | `.env` (repo root, gitignored) — has **no effect** on production |
