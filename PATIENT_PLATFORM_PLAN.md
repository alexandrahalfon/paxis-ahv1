# Patient Platform Plan

Plan for making Paxis patient-first while keeping the entire physician product intact behind a "For Physicians" entry point. Written against the current codebase, with a modular build order so nothing that works today gets rewritten or broken.

The doctor's insight is a good one and worth stating plainly: oncologists are drowning in patient questions, most of which are not urgent and not complicated, but all of which cost time and emotional energy. A tool that answers the answerable ones well, and routes the genuinely concerning ones to the physician with context attached, is valuable to both sides. That framing (reduce physician burden, not just serve patients) is also what makes it sellable to the physician, who is still your buyer.

---

## 1. What already exists

### The patient side today is thin

`frontend/patient-qa.html` plus `src/api/routes/patient_query.py`. One page, one endpoint, roughly 140 lines total. A patient types a question, it runs the full retrieval pipeline, and GPT-4o answers with a patient-friendly system prompt plus a fixed disclaimer.

The system prompt is actually good and worth keeping as the foundation. It already instructs: plain language, no recommending new treatments, no contradicting the care team, mention when to contact the care team, admit uncertainty.

What it does not do:

- **No accounts.** The endpoint has no auth at all, so there is no patient identity.
- **No connection to anything.** It does not know the patient's diagnosis, their physician, or their history. Every question is answered cold.
- **No memory.** Conversation history is passed in from the browser and never persisted.
- **No escalation.** There is no path from "I am worried about this" to a physician.
- **No safety layer.** Nothing detects a medical emergency or a distressed patient.
- **No gap-filling.** It answers whatever is asked without ever asking the patient a clarifying question.

Three implementation issues to fix while rebuilding: the OpenAI call is not threaded (blocks the event loop, same class of bug the audit found elsewhere), it constructs a fresh OpenAI client per request, and the 500 handler returns a full stack trace to the caller, which should not ship on a public unauthenticated endpoint.

### What is reusable, which is most of it

This is the good news, and it is why this can be modular rather than a rewrite:

| Existing piece | Reuse for patients |
|---|---|
| Retrieval pipeline (`comprehensive_retrieval`, `enhanced_rag_service`) | Same evidence base, unchanged. Only the generation prompt differs. |
| `patient_signal_service` | Already extracts clinical facts from free text. This is exactly the gap-filling engine the patient chat needs. |
| `clinical_inference.py` | Maps implicit statements to clinical concepts. Works on patient phrasing too. |
| `patients` + related tables | Already model a patient clinically. Needs a link to a patient user account. |
| `patients.physician_id` | The physician relationship is already modeled. |
| Auth (`account_db`, `auth.py`) | Extend with a role rather than build a second auth system. |
| `saved_studies_service` | Becomes the patient's saved answers and resources. |

### The one structural gap

Today a "patient" is a **record a physician owns**. There is no patient user.

- `users` has id, email, password_hash, first_name, last_name, institution, and **no role column**. Every account is implicitly a physician.
- `patients` has `physician_id` pointing at a user, but no `user_id` pointing at the patient's own account.

So the central modeling change is: a patient becomes both a clinical record (already exists) and a user account (new), joined by a nullable `user_id`. Everything else follows from that.

---

## 2. Two things to decide before building

### 2.1 The physician dropdown is a privacy hole as described

"Pick your physician from a dropdown" means any person who signs up can attach themselves to any oncologist, and then route messages into that physician's inbox. That creates spam, and worse, it lets someone assert a clinical relationship that does not exist.

Use an invitation model instead, which is barely more work and is what patients will expect anyway:

- The physician invites the patient from inside the physician app, generating a single-use code or link.
- The patient signs up and enters the code, or lands on the link pre-filled.
- The link is established, verified from the physician's side.

Keep the dropdown as a fallback for a patient who arrives without a code, but treat it as a **request** that lands in the physician's inbox for approval, not an automatic connection. Same UX, safe semantics.

### 2.2 Patient-facing changes your risk posture

A physician using Paxis can sanity-check a wrong answer. A frightened patient at 11pm cannot. Two implications:

**Safety engineering is not optional here.** You need a real triage layer, covered in section 4.

**Get a regulatory read before launch.** Software that supports clinical decisions has historically had more room when it is aimed at a professional who can independently review the basis for the recommendation. Patient-directed tools generally do not sit in the same place. I am not a lawyer and this is not legal advice, but the difference between "educational information about your treatment" and "guidance about your care" is exactly the line that matters, and it is worth a healthcare regulatory attorney looking at your scope and copy before you launch broadly. Scoping the product tightly to explaining things the care team has already decided, which the current prompt does well, is the defensible position.

---

## 3. Architecture: modular, additive, nothing breaks

### 3.1 Data model changes (all additive)

```sql
-- users: role, defaulting to physician so every existing account is unchanged
ALTER TABLE users ADD COLUMN IF NOT EXISTS role TEXT NOT NULL DEFAULT 'physician';
  -- 'physician' | 'patient' | 'admin'

-- patients: link a clinical record to a patient's own login
ALTER TABLE patients ADD COLUMN IF NOT EXISTS user_id UUID;      -- nullable
ALTER TABLE patients ADD COLUMN IF NOT EXISTS invite_code TEXT;  -- single use
ALTER TABLE patients ADD COLUMN IF NOT EXISTS link_status TEXT DEFAULT 'unlinked';
  -- 'unlinked' | 'invited' | 'pending_approval' | 'linked'
CREATE UNIQUE INDEX IF NOT EXISTS patients_user_id_idx ON patients (user_id)
  WHERE user_id IS NOT NULL;
```

Nullable `user_id` is what makes this modular: every physician-created patient record keeps working exactly as it does now, and linking is purely opt-in on top.

Two new tables:

```sql
-- Patient conversations, persisted (today nothing is saved)
CREATE TABLE patient_conversations (
    id UUID PRIMARY KEY,
    patient_user_id UUID NOT NULL,
    patient_record_id UUID,          -- null until linked
    title TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE patient_messages (
    id UUID PRIMARY KEY,
    conversation_id UUID NOT NULL REFERENCES patient_conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,              -- 'patient' | 'assistant' | 'physician'
    content TEXT NOT NULL,
    safety_flag TEXT,                -- null | 'emergency' | 'clinical' | 'distress'
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Escalations: the physician inbox
CREATE TABLE patient_escalations (
    id UUID PRIMARY KEY,
    patient_user_id UUID NOT NULL,
    patient_record_id UUID,
    physician_id UUID NOT NULL,
    conversation_id UUID,
    question TEXT NOT NULL,
    ai_draft_answer TEXT,            -- what Paxis would have said
    context_summary TEXT,            -- what Paxis knows about this patient
    urgency TEXT NOT NULL,           -- 'routine' | 'soon' | 'urgent'
    status TEXT NOT NULL DEFAULT 'open',  -- 'open' | 'answered' | 'closed'
    physician_response TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    answered_at TIMESTAMPTZ
);
```

The `ai_draft_answer` field is the detail that makes this genuinely save physician time. The physician does not get a raw question, they get a question plus a draft they can approve, edit, or discard. That turns a five minute reply into a fifteen second one, which is the entire value proposition to them.

### 3.2 New service modules

Keep everything new in its own files so the physician product is untouched:

```
src/api/services/patient_portal/
    __init__.py
    patient_link_service.py      # invites, claims, approvals
    patient_chat_service.py      # patient conversation + gap-filling
    patient_safety_service.py    # triage and escalation rules
    patient_vocab.py             # lay language to clinical concept mapping
    escalation_service.py        # physician inbox
src/api/routes/
    patient_portal.py            # all patient-authenticated routes
```

Only three existing files get touched, all additively: `account_db.py` (role column), `patient_db.py` (new tables), and `auth.py` (role in the token and registration).

### 3.3 Routing and navigation

Flip the default while keeping every physician page where it is:

- `/` becomes the patient home.
- `/physicians` (or the existing `index.html` moved) becomes the physician home, reachable from a clear "For Physicians" button in the header.
- After login, redirect by role: patients to the patient home, physicians to the physician home. A physician who lands on the patient side sees a "switch to physician view" prompt rather than being blocked.
- Every existing physician page keeps its URL, so nothing bookmarked breaks and no internal links need rewriting.

Guard the physician routes with a role check. That is a small dependency added to the existing auth dependency, not a new system.

---

## 4. The patient chat: gap-filling and safety

This is the part that needs the most design thought, because it is where patient-facing genuinely differs from physician-facing.

### 4.1 Patients do not know their clinical details

A physician types "recurrent HNSCC, CPS 100, progressing on pembro." A patient types "I have throat cancer and the immunotherapy stopped working I think."

Three layers handle this:

**Layer 1: a lay-language vocabulary map** (`patient_vocab.py`, new). A dictionary from how patients talk to what the literature calls it. This is real work but very tractable, and it is the same shape as the existing `clinical_inference.py` inference map, so the pattern already exists in the codebase:

- "the red medicine", "red devil" to doxorubicin
- "chemo pills" to oral chemotherapy, capecitabine and similar
- "immunotherapy", "the drug ending in -mab" to checkpoint inhibitor
- "the big scan", "the lighting-up scan" to PET/CT
- "they said it spread" to metastatic
- "they can't operate" to unresectable
- "radiation burns" to radiation dermatitis

**Layer 2: conversational gap-filling.** Instead of answering a vague question badly, the assistant asks one clarifying question at a time. The rules that matter:

- Ask at most one question per turn. A form disguised as a chat feels worse than a form.
- Never block on an answer. If the patient does not know, say that is completely fine and answer at the level you can.
- Only ask for what changes the answer. Do not collect a full history for the sake of it.
- Offer easy outs: "if you have a copy of your pathology report, the cancer type is usually near the top, but do not worry if you cannot find it."
- Reuse `patient_signal_service` to extract facts from whatever they do say, so the profile builds passively across the conversation rather than through interrogation.

**Layer 3: use the linked record when it exists.** When the patient is linked to a physician-created record, Paxis already knows the diagnosis, biomarkers, and treatment history. Then it should not ask at all, it should confirm gently: "I can see you are on pembrolizumab, is that what you are asking about?" This is the strongest argument for the linking feature, and the thing no standalone patient app can do.

### 4.2 Safety triage on every message

Run a fast classifier on every patient message before answering. Three categories:

**Emergency.** Chest pain, trouble breathing, uncontrolled bleeding, high fever on chemotherapy (neutropenic fever is a genuine emergency and patients frequently do not know that), stroke symptoms, suicidal ideation. Response: stop, do not answer the question as asked, show clear instructions to call emergency services or their care team's urgent line, and flag the conversation.

**Clinical decision.** "Should I stop taking this?", "Can I skip a dose?", "Is this treatment right for me?", "What are my chances?" Response: explain generally what is known, then explicitly route to the physician. This is the escalation path and should feel like a feature rather than a refusal: "This one is really a question for Dr. X. I can send it to them with a summary of what you asked, if you like."

**General information.** Everything else. Answer normally.

Build this as rules plus a small model call, not model-only. A regex list for the unambiguous emergencies is faster and more reliable than an LLM, and it fails safe. Use the model for the ambiguous middle.

Prognosis questions deserve their own handling. "How long do I have" is common, deeply human, and never appropriate for an AI to answer from literature statistics. Route to the physician with warmth.

### 4.3 Escalation flow

1. Patient asks something clinical or says they want to ask their doctor.
2. Paxis composes: the question, a draft answer, and a short context summary of what it knows.
3. Patient confirms and sends.
4. Physician sees it in an inbox in the physician app, with approve / edit / write-my-own.
5. Answer goes back into the patient's conversation, clearly attributed to the physician rather than to Paxis.

Set expectations in the UI: this is not a messaging service and not for emergencies, and responses come when the physician is available.

---

## 5. Build order

### Phase 1: Foundation (no user-visible change)
Role column, `user_id` on patients, role in the auth token, role-aware redirects, route guards. Everything existing keeps working. Ship and verify this alone before anything else.

### Phase 2: Patient account and home
Patient registration with role, the new patient home page, "For Physicians" button, physician app relocated behind it. Patient chat still the existing simple Q&A at this stage.

### Phase 3: The real patient chat
Persisted conversations, the vocabulary map, gap-filling follow-ups, and the safety triage layer. Also fix the three implementation issues in the current endpoint (thread the OpenAI call, reuse the client, stop leaking stack traces). This is the biggest single phase and the core of the product.

### Phase 4: Linking and escalation
Physician invites, patient claim flow, approval inbox, escalation queue with AI drafts, physician response path. This is what makes it a two-sided product rather than two products.

### Phase 5: Patient versions of the tools
Carefully, and only the ones that make sense:
- **Trial finder:** high value to patients, genuinely wanted, and relatively safe if framed as "trials to discuss with your doctor."
- **Saved information:** their answers and resources in one place.
- **Treatment explainer:** a patient-appropriate version of treatment comparison, framed as understanding options the care team has raised, not choosing between them.
- **Study matching:** probably not directly. Handing raw trial literature to patients invites misinterpretation. Better surfaced as plain-language summaries.

### Phase 6: Community forums
Deliberately last. Forums are a moderation and liability commitment, not a feature. Patients giving each other medical advice under your brand is a real risk, and an unmoderated cancer forum can cause harm. When you do it: topic-scoped by cancer type, clear rules, moderation tooling from day one, and a plan for who actually moderates. Consider physician-moderated AMAs or curated Q&A as a lighter first step.

---

## 6. Risks worth tracking

**Physician adoption is the gate.** Patients arrive by referral, so if physicians do not invite, nothing happens. This is why the escalation inbox must save them time from the first week, not add to their load. If the inbox feels like another thing to answer, they will stop referring.

**Escalation volume.** If too much routes to the physician, you have rebuilt their inbox problem inside your product. Track the escalation rate as a core metric and tune the triage. The goal is a high share of questions answered well without the physician.

**Answer quality bar is higher.** A physician spots a subtly wrong answer. A patient does not. This raises the stakes on retrieval quality, which is another reason to finish the beta optimization work first.

**Scope creep into advice.** The current prompt's discipline (explain, never recommend, never contradict) is the product's safety backbone. Every new patient feature should be checked against it.

**Do not start this before beta ships.** The physician product is what proves the evidence engine works, and the beta is where you find the physicians who will refer patients. Build Phase 1 whenever, since it is invisible and additive, but the real work should follow beta.

---

## 7. Why this is a strong direction

The competitive point is real. Existing patient oncology tools (Outcomes4Me, BelongAI, TrialJectory and similar) are standalone consumer apps. They do not know the patient's actual care team and cannot route anything to them.

Paxis would be answering patients from the same evidence base their oncologist is using, about their actual documented case, with a path back to their actual physician. That is a different product from a cancer chatbot, and it is only possible because the physician side already exists. The two sides make each other more valuable, which is the strongest version of this pitch.
