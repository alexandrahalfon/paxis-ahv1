# Paxis — Dorm Room Fund Pitch Package
**Date:** April 2026 | **Stage:** Pre-seed | **Ask:** $20,000

---

## EXECUTIVE SUMMARY

**Paxis** is an AI-powered clinical decision support platform for oncology. We give oncologists instant, cited answers from thousands of peer-reviewed studies — replacing hours of manual literature search with a single query. Our platform combines semantic search, large language models, and a proprietary oncology knowledge base to help cancer doctors find the right evidence, match patients to trials, and compare treatment approaches in seconds.

**The problem:** Oncologists spend 15–30% of their working time searching for evidence to support clinical decisions. PubMed has 35M+ papers. Existing tools return lists of links — not answers. Patients wait. Decisions get made without the best evidence.

**Our solution:** Ask Paxis a question in plain English ("What are outcomes for EGFR-mutant stage III NSCLC with SBRT?"), get a synthesized, cited answer from peer-reviewed oncology literature in under 10 seconds.

**Traction:** Live product deployed on GCP. Indexed knowledge base of oncology peer-reviewed studies. Three co-founders with technical + medical domain expertise.

**Team:** Aysha Allahverdiyeva (Co-Founder & Developer), Alexandra Halfon (Co-Founder & Engineer), Dr. Jinyu Xue (Co-Founder & Medical Advisor)

---

## THE PROBLEM

### Oncologists are drowning in evidence — and still can't find what they need fast enough.

- **35 million papers** on PubMed. ~500,000 new ones published every year.
- An oncologist treating a patient with a rare mutation combination may need to synthesize data from 20+ papers across different cancer types, treatment phases, and biomarker profiles.
- Current workflows: Google Scholar → skim abstracts → download PDFs → read manually. This can take **hours per question**.
- Existing tools (UpToDate, PubMed, clinical guidelines) are either too generic, too slow, or require the clinician already know what they're looking for.
- **The gap costs lives:** Treatment decisions made without the latest evidence lead to suboptimal outcomes. Clinical trial recruitment is slow because matching patients to eligibility criteria is manual.

### The numbers behind the pain:
- There are ~16,000 oncologists in the US and ~100,000+ globally
- 1.9 million new cancer diagnoses in the US per year
- Clinical trial enrollment fails 80% of the time due to poor patient-to-trial matching
- ~$50B is spent annually on cancer clinical trials; poor enrollment is a major cost driver

---

## OUR SOLUTION

### Paxis: The AI research assistant built for oncologists.

**Core product:** A natural language interface over a curated oncology knowledge base, powered by retrieval-augmented generation (RAG) with GPT-4o.

**What makes us different from "just ChatGPT for medicine":**

1. **Specialized oncology knowledge base** — we index, process, and normalize peer-reviewed oncology studies. Every answer is grounded in real sources, not hallucinated.

2. **Advanced RAG pipeline** — we don't just search and summarize. Our pipeline uses:
   - Query expansion (medical abbreviation resolution)
   - Cross-encoder neural reranking for relevance
   - 8-type query classification with specialized generation prompts
   - Evidence packing that groups results by study
   - Semantic chunking with 3072-dimensional embeddings

3. **Patient-to-trial matching** — paste a patient's profile in plain text, get a ranked list of matching clinical studies with eligibility scoring. No more manual protocol reviews.

4. **Treatment comparison** — side-by-side outcomes data for different treatment approaches (e.g., SBRT vs. conventional fractionation for lung cancer).

5. **Personal document upload** — doctors can upload their own PDFs (unpublished studies, institutional data) and search across them alongside the knowledge base.

6. **Knowledge base analytics** — aggregate queries, forest plots, dose distribution charts, outcomes by stage. Research-grade analytics in a clinical interface.

### What a user sees:
```
Query: "What is the local control rate for SBRT in oligometastatic 
        NSCLC with liver mets?"

Answer: "Based on 12 studies in our knowledge base, SBRT achieves 
         local control rates of 85–92% at 1 year for oligometastatic 
         NSCLC liver metastases [PMID 12345, 23456, 34567]..."

Sources: [Study 1] [Study 2] [Study 3] → expandable panels with 
         full study profiles, patient populations, and key data
```

---

## MARKET OPPORTUNITY

### Three distinct paths to revenue — all large.

**Tier 1 — Individual Clinicians (Bottom-up)**
- ~16,000 oncologists in the US; ~100,000+ globally
- Adjacent: radiation oncologists, oncology fellows, clinical researchers
- SaaS subscription: $100–$300/month per clinician
- TAM at 10% US oncologist penetration: ~$32M ARR

**Tier 2 — Hospital / Cancer Center Licenses (Enterprise)**
- ~1,500 NCI-designated cancer centers and academic medical centers in the US
- Enterprise deals: $50,000–$250,000/year per institution
- TAM at 5% penetration: $37.5M–$187.5M ARR

**Tier 3 — Pharma / Clinical Trial Operations (High-value)**
- Patient-to-trial matching is a $2B+ problem for pharmaceutical companies
- Trial recruitment failure = ~$8M average cost per delayed trial
- B2B contracts for automated patient identification and protocol matching
- This is the highest-margin, most scalable revenue path

**Total Addressable Market:** $15B+ (clinical decision support software + oncology AI)

---

## PRODUCT & TECHNOLOGY

### Why this is hard to replicate:

| Layer | What we built |
|-------|--------------|
| **Data** | Curated, indexed oncology knowledge base (PostgreSQL + Qdrant vector DB) |
| **Retrieval** | Semantic search with cross-encoder reranking, 3072-dim embeddings |
| **Intelligence** | 8 query-type classifiers, specialized generation prompts per query type |
| **Medical NLP** | Biomarker normalization, TNM staging inference, clinical entity extraction |
| **Matching** | Patient profile extraction → eligibility scoring against trial criteria |
| **Analytics** | Aggregate queries, meta-analysis forest plots, outcomes visualizations |
| **Infrastructure** | GCP Cloud Run, auto-scaling, Redis caching, production-ready |

### Current tech stack:
- **Backend:** Python/FastAPI (async)
- **Vector DB:** Qdrant Cloud (3072-dim, `text-embedding-3-large`)
- **Relational DB:** PostgreSQL on GCP Cloud SQL (3 databases)
- **LLM:** OpenAI GPT-4o (synthesis) + GPT-4o-mini (intermediate steps)
- **OCR:** Mistral OCR + Pixtral (for PDF processing with tables/figures)
- **Deployment:** GCP Cloud Run (serverless, auto-scaling)

### Product maturity:
- Full-stack application live and deployed
- 18 frontend pages (query, patient matching, study comparison, analytics, uploads, saved cases, auth)
- 23 API route groups, 60+ backend services
- Comprehensive test coverage
- Production CI/CD pipeline

---

## TRACTION & VALIDATION

- **Live product** deployed on GCP, accessible now
- **Knowledge base** of oncology peer-reviewed studies indexed and searchable
- **Core features** all functional: RAG query, patient matching, treatment comparison, study comparison, document upload, analytics
- **Medical advisor** on the founding team (Dr. Jinyu Xue) providing domain validation
- **Iterative development** across multiple product phases with real feature refinement
- **Target users identified** and product built around real clinical workflows

*Note: We are at the pre-revenue stage and actively working toward first paying users / pilot institutions.*

---

## BUSINESS MODEL

### Go-to-market strategy:

**Phase 1 (Now → 6 months): Individual clinician adoption**
- Free tier for limited queries/month
- Pro tier: $149/month (unlimited queries, document upload, saved cases)
- Target: oncology fellows and residents at academic medical centers (early adopters, high openness to new tools)

**Phase 2 (6–18 months): Institutional sales**
- Leverage individual user champions to pitch department heads
- Hospital/cancer center licenses: $75K–$200K/year
- HIPAA compliance roadmap (architecture already privacy-conscious)

**Phase 3 (18 months+): Pharma partnerships**
- Patient-to-trial matching as a service for clinical trial sponsors
- API access for trial recruitment optimization
- High-value contracts ($500K–$2M+)

---

## TEAM

### Why us?

**Aysha Allahverdiyeva** — Co-Founder & Lead Developer
- Built the full-stack platform from scratch (FastAPI backend, RAG pipeline, frontend)
- Architected the data ingestion pipeline, vector search system, and all core services
- Deep expertise in AI/ML infrastructure, NLP, and production system design

**Alexandra Halfon** — Co-Founder & Engineer
- Full-stack engineering contributions across frontend and backend
- Feature development, testing infrastructure, and deployment

**Dr. Jinyu Xue** — Co-Founder & Medical Advisor
- Clinical domain expertise in oncology
- Validates product decisions against real clinical workflows
- Bridges the gap between AI capabilities and medical utility
- Essential for credibility with clinician users and institutional buyers

### Why now?
- GPT-4o made high-quality synthesis from large document corpora economically viable
- Oncology is the highest-stakes, highest-complexity medical domain — and the one most starved for good decision support tooling
- Clinical AI is moving from "interesting demo" to "institutional procurement" — we are building now to capture this shift

---

## THE ASK

**Requesting: $20,000 from Dorm Room Fund**

### Use of funds:
| Category | Amount | Purpose |
|----------|--------|---------|
| Infrastructure | $5,000 | GCP, Qdrant, OpenAI API costs for 6 months of active user testing |
| Knowledge base expansion | $4,000 | Licensing/processing costs to expand indexed study database |
| User acquisition | $4,000 | Reach oncology fellows at 5 target academic medical centers |
| Legal / compliance groundwork | $3,000 | Entity formation, basic IP protection, HIPAA review |
| Operational runway | $4,000 | Conference attendance, clinical advisor access, misc |

### What we will achieve with this funding:
- **10 pilot users** (oncologists/fellows) actively using the platform within 3 months
- **1 letter of intent** from a cancer center or academic department within 6 months
- **HIPAA compliance roadmap** completed
- **Series of user interviews** to validate and sharpen the product
- Foundation to raise a $500K–$1M pre-seed round within 12 months

---

## WHY DORM ROOM FUND?

Beyond the capital, we want to join the DRF network because:

1. **Community** — access to other student founders who've navigated early healthcare/AI companies
2. **Warm intros** — DRF's First Round network has deep connections to healthcare VCs (a16z Bio, GV, Rock Health portfolio)
3. **Credibility signal** — DRF backing helps open doors to institutional pilots (hospitals take it more seriously)
4. **Mentorship** — we are first-time founders and want genuine guidance, not just a check

---

## APPENDIX

### Key risks and mitigations:

| Risk | Mitigation |
|------|-----------|
| Regulatory (FDA SaMD) | Clinical decision *support* (not autonomous decision-making); physician always makes final call. Monitoring FDA guidance on AI/ML SaMD closely. |
| Hallucination / accuracy | Every answer grounded in cited sources. Users see primary evidence. Not replacing clinical judgment. |
| Competition (large incumbents) | UpToDate ($400M+ revenue) is a static reference tool, not AI-native. Epic/Epic Cosmos lacks oncology-specific depth and NLP interface. We are purpose-built. |
| Clinician adoption | Starting with fellows/residents (most tech-forward); building trust bottom-up before institutional sales. Medical advisor on team for credibility. |
| Data access / licensing | Focus on open-access and permissively licensed oncology literature as foundation; partnerships with publishers as we scale. |

### Comparable companies:
- **Consensus** (AI research search, ~$10M ARR) — general science, not clinical
- **Elicit** (AI literature review) — research, not clinical decision support
- **Tempus** (oncology AI, $1B+ valuation) — genomics-focused, not literature search
- **Viz.ai** (radiology AI) — imaging, not literature
- **Paxis** fills the evidence synthesis gap that none of these address for practicing oncologists

### Contact:
- **Aysha Allahverdiyeva** — [contact via DRF application]
- **Live demo available upon request**

---

## ANTICIPATED Q&A — PREP GUIDE

### On the product

**Q: How is this different from just using ChatGPT or Claude?**
> General LLMs hallucinate medical facts and have no reliable citations. Paxis is grounded — every answer is retrieved from our indexed knowledge base of real peer-reviewed studies. Users can click into the source, see the exact patient population, read the methodology. We also have domain-specific features general LLMs don't: patient profile extraction, staging inference, biomarker normalization, trial eligibility scoring. The difference is like Google vs. a medical database — except ours actually synthesizes the answer.

**Q: Why not just use PubMed or UpToDate?**
> PubMed returns 10,000 links when you ask a clinical question. UpToDate gives you a static, manually curated article that may be 2 years out of date. Neither one gives you a synthesized, cited answer to a specific clinical question in plain English in under 10 seconds. Paxis does. We're not replacing the primary literature — we're making it accessible in the middle of a clinical workflow.

**Q: Is the AI making clinical decisions?**
> No. Paxis is clinical decision *support* — it surfaces evidence for the physician to interpret. The doctor makes every decision. This is the same model as UpToDate, IBM Micromedex, or any clinical reference tool. We make it explicit that results should be interpreted by a qualified clinician.

**Q: How accurate is it?**
> Our pipeline is designed around accuracy over speed: we retrieve more candidates than needed, rerank with a cross-encoder neural model, and only synthesize from the top-k most relevant chunks. Every claim in an answer is backed by a specific source the user can expand and verify. We do not summarize information that is not in the knowledge base — if we don't have a source for it, we don't say it.

**Q: How big is your knowledge base right now?**
> We have a live Qdrant vector database with indexed oncology studies. The current focus is radiation oncology and solid tumor treatments — a well-defined subset where we can ensure depth and quality. We expand systematically, not by scraping everything at once.

---

### On the market / competition

**Q: Epic already does this for hospitals. Why wouldn't they just build it?**
> Epic's strength is EHR data — patient records, orders, billing. Their literature search is minimal and not AI-native. They also face enormous integration complexity — any new feature must work across thousands of hospital configurations. We're purpose-built for oncology literature synthesis, which means we can move faster and go deeper. When we have traction, we become an acquisition target or integration partner, not a loser in a head-to-head.

**Q: What about Tempus, Flatiron, or other oncology AI companies?**
> Tempus and Flatiron are genomics and real-world evidence companies — they process patient data, not published literature. They don't synthesize peer-reviewed research for clinical decision support. There's no direct competitor doing what we do at this quality level for practicing oncologists. The closest is Consensus or Elicit, but those are general academic research tools, not clinical.

**Q: This seems like a crowded AI space. What's your moat?**
> Three things: (1) Domain depth — building oncology-specific pipelines (staging inference, biomarker normalization, query classification for 8 oncology query types) is not something a general AI company will prioritize. (2) Data curation — our knowledge base is curated and normalized, not scraped. Quality takes time to build. (3) Clinical trust — adoption in medicine is relationship-driven. The trust we build with early oncologist users is not easily replicated.

---

### On the business

**Q: Why haven't you gotten paying users yet?**
> We've been heads-down building to the point where the product can stand on its own in a demo to a skeptical oncologist. We're confident it's there now. The DRF funding is specifically to cover the infrastructure costs and outreach effort to get the first 10 paying users or pilot agreements.

**Q: How will you get your first customers?**
> Our medical advisor Dr. Xue provides direct access to oncology networks. We're targeting oncology fellows at academic medical centers — they're the most tech-forward, most likely to be doing literature search daily, and have institutional email addresses that unlock academic licensing conversations. Word of mouth in tight clinical communities moves fast once you have one champion.

**Q: What does a sales cycle look like?**
> Individual clinicians: self-serve, potentially same-day conversion. Institutions: 3–9 months, typically requires a champion inside the department, a pilot period, and IT/legal review. We plan to drive institutional deals through bottom-up individual adoption first.

**Q: What's the path to $1M ARR?**
> Either 556 individual subscribers at $149/month, or 5–10 institutional licenses, or 2–3 pharma partnerships. The most realistic near-term path: 50–100 individual users generating buzz + 2–3 institutional pilot agreements converting to paid in year 2.

**Q: How are you thinking about HIPAA?**
> Currently, the platform works with published literature — no patient data enters the system except what a clinician voluntarily types into the patient matching form. The architecture is already privacy-conscious (no PII stored in the knowledge base). Formal HIPAA compliance (BAA, security audit) is on the near-term roadmap and we've scoped the work. This is a known requirement and a known path.

---

### On the team

**Q: You're students — why should we trust you to build a healthcare company?**
> We've already built it. The product is live, deployed on GCP, with a production-grade architecture — 60+ backend services, a full RAG pipeline, 18 frontend pages, real users able to query it right now. We're not pitching a concept; we're pitching a working product that needs go-to-market resources. The technical risk is substantially de-risked. What we need is help with distribution and the first commercial relationships.

**Q: What's Dr. Xue's involvement?**
> Dr. Xue is a co-founder, not an advisor in name only. They validate our product decisions against real clinical workflows, help us understand what oncologists actually need vs. what sounds good to an engineer, and provide direct access to the clinical community for our go-to-market. Having a credible clinical voice on the founding team is essential for institutional sales and for avoiding costly product mistakes.

**Q: What happens if one founder leaves?**
> The codebase is well-structured and documented (multiple steering guides, testing guides, deployment docs). The critical IP is in the RAG pipeline and data architecture, which Aysha owns end-to-end. We've discussed this and are aligned on commitment. That said, DRF's due diligence should satisfy itself that the team is stable.

---

### On the ask

**Q: Why $20,000? What does it actually unlock?**
> It covers 6 months of infrastructure (GCP + Qdrant + OpenAI API runs ~$800–1,200/month at active user testing scale), plus the outreach and travel needed to get in front of oncologists at 3–5 academic medical centers. The specific goal is: 10 active pilot users and 1 letter of intent in 6 months — enough to anchor a $500K–$1M pre-seed raise from healthcare-focused investors.

**Q: Are you talking to other investors?**
> We are in early conversations. DRF is our top choice for the first check because of the student founder community, the First Round network in healthcare, and the mentorship model — not just the capital.

---

*Prepared April 2026 | Paxis | Confidential*
