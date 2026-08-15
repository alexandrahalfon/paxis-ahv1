# EHR Integration Plan

How Paxis would connect to hospital EHRs, what it actually takes, and in what order. Written against the current Paxis architecture (manual patient entry, `patients` / `patient_diagnosis` / `patient_biomarkers` / `patient_treatment_history` / `patient_timeline_events` in the `paxis_patients` database).

The headline: the technical work is the easy part. Access, compliance, and finding a health system willing to sponsor you are the hard parts, and they run on much longer timelines. Plan around that, not around the code.

---

## 1. The single most important scoping decision

Paxis only needs to **read** patient data. It does not need to write to the chart.

That matters enormously, because broad write access to clinical records is not offered to third-party applications by Epic at all. Products that need write-back face a much harder path. Paxis is a read-only decision-support tool, which is the easiest category to integrate and the one with the most established pattern.

Hold that line. If a future feature tempts you into writing back to the chart, understand you are changing integration difficulty by an order of magnitude. The one exception worth considering much later is attaching a generated evidence summary as a document, which is a narrower and more achievable form of write.

---

## 2. Where Paxis already stands

Genuinely helpful things that are already true:

- The patient schema is normalized along the same axes FHIR uses, so the mapping is mostly mechanical rather than a redesign.
- Patient records are already separated from literature and accounts into their own database, which simplifies the access-control story you will have to explain during security review.
- Every patient mutation already writes a timeline event, which is most of an audit trail. Security reviewers will ask for exactly this.
- Retrieval already accepts a structured profile, not just free text, so a FHIR-sourced profile can feed the existing pipeline without touching the retrieval core.

Real gaps that EHR integration will expose:

- **Coded data versus narrative.** This is the biggest technical item. Paxis's ontology and inference layers were built to read clinical narrative ("no longer a surgical candidate"). FHIR delivers structured codes instead: ICD-10 and SNOMED for conditions, LOINC for labs and observations, RxNorm for medications. A code-to-concept mapping layer is new work, not a config change.
- **No comorbidities field** on the patient record, which the audit already flagged. FHIR will hand you a full Condition list, so this gap becomes immediately visible.
- **PHI leaving your infrastructure.** Covered in section 5. This is the item most likely to be underestimated.
- **No deletion capability**, no encryption at rest, and no SSL enforced on the DB connection. All three become blocking issues once real PHI arrives.

---

## 3. Technical phases

### Phase 0: Make the internal model FHIR-native (do this first, needs no partner)

Write a mapping layer between FHIR resources and the existing patient schema:

| FHIR resource | Paxis destination |
|---|---|
| `Patient` | `patients` (name, DOB, sex, MRN) |
| `Condition` | `patient_diagnosis` plus a new comorbidities concept |
| `Observation` | `patient_biomarkers` (CPS, PD-L1, labs) |
| `MedicationRequest` / `MedicationStatement` | `patient_treatment_history` |
| `Procedure` | `patient_treatment_history` (surgery, radiation) |
| `Condition.stage` / staging observations | diagnosis staging fields |

Build it as a standalone `fhir_patient_mapper.py` that takes a FHIR bundle and returns the same profile dict `patient_collection_seeder` already consumes. That keeps the retrieval pipeline untouched and testable.

You can build and test all of this today against synthetic data. No health system, no contracts. Use the public sandboxes: Epic on FHIR and Cerner/Oracle Health both publish open sandboxes with test patients, and Synthea generates realistic synthetic FHIR bundles at volume.

Alongside it, build the terminology layer: map ICD-10 and SNOMED condition codes and RxNorm drug codes onto the canonical concepts your ontology already uses. Start narrow, oncology diagnoses and oncology drugs only, and expand from real data.

### Phase 1: File-based bridge (fastest real-world value)

Before any API integration, most health systems can export a patient summary as a CCDA document or a FHIR bundle. Accepting an uploaded bundle and parsing it through the Phase 0 mapper gives physicians a working "import my patient" flow with zero integration overhead on the hospital side. This is a very cheap way to prove value during beta and to learn what the real data looks like.

### Phase 2: SMART on FHIR app (the main event)

SMART on FHIR is the standard for launching a third-party app from inside the EHR. The flow is:

1. The physician has a patient open in Epic and clicks Paxis from within the chart.
2. Epic launches Paxis with a launch context and an OAuth2 authorization step.
3. Paxis receives a token scoped to that patient plus the physician's identity.
4. Paxis reads `Patient`, `Condition`, `Observation`, `MedicationRequest`, and `Procedure` for that patient.
5. The Phase 0 mapper turns that into a profile, and the existing pipeline runs.

The important property: because SMART is a standard, the same app code works across any SMART-enabled EHR. You build once and the incremental cost per additional health system is contractual rather than engineering.

Practically this means adding an OAuth2 client, a launch endpoint, token handling, and scope management. It is a well-trodden path with good libraries.

### Phase 3: CDS Hooks (proactive rather than pull)

CDS Hooks lets the EHR call you at clinically meaningful moments, for example when a physician opens a patient or signs an order, and lets you return a card in their workflow. Research has found that contextually relevant CDS Hooks prompts significantly increase SMART app usage, because you are no longer relying on the physician remembering to open your tool.

This is where the "never stops watching" part of the Paxis pitch becomes real inside the EHR rather than only inside your product. It is a natural fit, but it comes after Phase 2 and after you have earned trust, because unsolicited alerts are held to a much higher bar than a tool someone chose to open.

### Phase 4: Limited write-back (much later, optional)

The realistic version is attaching a Paxis evidence summary to the chart as a document. Treat this as a separate project with its own approval path.

---

## 4. The access path, which is the actual bottleneck

The order that matters:

1. **Find a design partner health system first.** Epic access is effectively gated by customer demand. A health system that wants you is what unlocks the process. Your beta is where you find this person, ideally an oncologist with institutional pull, or a service line chief.
2. **Register with Epic Vendor Services** for production API access, and create a listing on Epic's Showroom / Connection Hub so hospital IT can find and vet you. Many Epic customer sites only approve integrations for applications that are listed.
3. **Pass technical review.** Epic examines functionality, performance, and conformance to FHIR and SMART.
4. **Pass security review.** Expect penetration testing, vulnerability assessment, and detailed scrutiny of data handling and HIPAA compliance.
5. **Negotiate with the health system directly:** BAA, security questionnaire, and often their own separate IT review.

Budget generously for time. Review queues are reported as backlogged, with early submission beating late submission by something like six to ten weeks, and that is on top of the hospital's own procurement cycle, which is frequently six to twelve months on its own.

**Middleware alternative.** Integration vendors such as Redox, Health Gorilla, 1upHealth, and Metriport sit between you and many EHRs, and can substantially reduce per-system engineering and sometimes shorten access timelines. You pay for that in recurring cost and in some loss of control. For a small team chasing the first two or three health systems, this is worth pricing seriously rather than dismissing. Build Phase 0 so it does not care whether the bundle came from Epic directly or from middleware.

---

## 5. Compliance, and the one item most likely to bite you

Once real PHI flows in, several things change from "should fix" to "cannot ship without":

- **PHI in LLM prompts.** Paxis currently sends patient narratives to OpenAI. With manually typed pseudonymous data that is one risk posture. With real identified EHR data it is a different one entirely, and it requires either a BAA with the model provider plus zero-retention terms, or de-identification before the call. Decide this early, because "de-identify before sending" is an architectural constraint that touches the whole pipeline, not a switch you flip later.
- **BAA with your cloud provider.** GCP will sign one and supports HIPAA workloads, but the covered-services configuration has to be done deliberately.
- **Encryption in transit and at rest.** The DB pool currently requests no SSL, and DOB and MRN are stored in plaintext. Both need fixing.
- **Deletion.** There is still no delete path anywhere in the product. Under a BAA you will be contractually required to have one.
- **Audit logging.** The timeline table gets you most of the way, but you will need access logging too, meaning who viewed which patient and when.
- **SOC 2 Type II.** Not legally required, but health system security reviews ask for it routinely, and not having it adds months. Starting the observation window early is a cheap way to avoid being blocked later.

---

## 6. Recommended sequencing

Do not start EHR integration before beta. You need clinical validation and a design partner first, and both come out of beta.

1. **Now, during beta prep:** close the PHI-readiness gaps that are good hygiene regardless (SSL on the DB, deletion path, encryption of DOB and MRN). Decide the LLM-and-PHI question in principle.
2. **During beta:** build Phase 0 against Synthea and the public sandboxes. Costs you nothing externally and de-risks everything later. Use beta relationships to identify a design partner.
3. **Immediately after beta:** ship Phase 1 file import, which gives real value while you negotiate. Start SOC 2. Begin Epic Vendor Services registration in parallel, since queue time is the constraint.
4. **With a signed design partner:** build Phase 2 SMART on FHIR against their sandbox, go through security review, pilot on a single service line.
5. **After a successful pilot:** consider Phase 3 CDS Hooks and expansion to additional systems.

The realistic timeline from "decide to do this" to "live in one health system" is nine to eighteen months, and the majority of that is not engineering. The engineering you can start this month.

---

## Sources

- [Epic EHR Integration Guide: SMART on FHIR and App Orchard](https://www.tactionsoft.com/blog/epic-ehr-integration-guide/)
- [Epic EHR Integration: APIs, Costs, and Best Practices](https://arkenea.com/blog/integrating-healthcare-app-with-epic-ehr/)
- [Epic App Store Integration Developer Guide](https://lifebit.ai/blog/epic-app-store-integration/)
- [SMART on FHIR official documentation](https://docs.smarthealthit.org/)
- [Using CDS Hooks to increase SMART on FHIR app utilization: a cluster-randomized trial](https://pubmed.ncbi.nlm.nih.gov/35641136/)
- [SMART on FHIR: Guide to CDS Hooks](https://intuitionlabs.ai/articles/smart-on-fhir-cds-hooks-coverage-guide)
