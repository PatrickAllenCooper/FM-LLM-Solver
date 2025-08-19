# FM-LLM - Experimental Design & System Design (MVP)

**Status**: Core MVP functionality implemented with acceptance terminology. Two-stage acceptance protocol operational with AcceptanceService replacing legacy VerificationService. Current deployment uses Firestore with planned PostgreSQL migration.

**Goal:** rigorous, reproducible evaluation of LLMs for proposing **Lyapunov function candidates** and **barrier function candidates** on **continuous** and **discrete** systems, with strict input schemas, acceptance checks, authenticated usage, and complete provenance.

## 1) Research Design (pre-registerable)

### Primary questions

- **Success rate** vs classical baselines (paired, equal budget).
- **Time-to-acceptance** and compute.
- **Quality**: margins, accepted region size, candidate complexity.
- **Robustness**: prompt variants, temperature, param perturbations.

### Conditions

- LLM modes: **direct expression**, **basis+coeffs**, **structure+constraints**.
- Budgets (fixed per attempt): max LLM calls, tokens, solver CPU time, restarts.
- Splits: **Dev** for prompt tuning; **Test** frozen before report.

### Stats

- Success: **McNemar** (paired).
- Time/strength: **Wilcoxon signed-rank** (paired). Report median [IQR], 95% CI (bootstrap).

## 2) User-Facing System DSL (strict, typed)

### 2.1 SystemSpec (JSON)

```json
{
  "spec_version": "1.0",
  "system_id": "string",
  "time_type": "continuous|discrete",
  "state": { "dimension": 2, "vars": ["x1","x2"] },
  "parameters": {
    "decl": {"mu":{"type":"real","range":[-2,2]}},
    "assign": {"mu": 1.0}
  },
  "dynamics": {
    "form": "symbolic|linear|polynomial",
    "rhs": ["x2", "mu*(1 - x1**2)*x2 - x1"], // continuous
    "update": null // discrete: ["...", "..."]
  },
  "domain": { "type":"semialgebraic", "ineq":["16 - x1**2 - x2**2"] },
  "initial_set": { "type":"box", "lb":[-0.2,-0.2], "ub":[0.2,0.2] },
  "safe_set": { "type":"semialgebraic", "ineq":["25 - x1**2 - x2**2"] },
  "unsafe_set": { "type":"semialgebraic", "ineq":["(x1-2)**2 + x2**2 - 0.25"] },
  "equilibrium": { "value":[0,0] },
  "control_policy": null,
  "notes": ""
}
```

**Allowed set types:** box, polytope(H,b), ellipsoid(center,P), semialgebraic(list of g_i(x)≤). Internally normalize to **semialgebraic**.

**Validation (hard errors):** dimension consistency; parseable real expressions; finite on domain; equilibrium for Lyapunov; unsafe_set for barrier.

### 2.2 LLM I/O Schemas

#### Input

```json
{
  "task": "lyapunov|barrier",
  "time_type": "continuous|discrete",
  "system": { /* SystemSpec */ },
  "constraints": {
    "max_degree": 4,
    "allowed_ops": ["+","-","*","^"],
    "target_form": "polynomial|rational|quadratic",
    "prefer_radial": true,
    "symmetry_hints": []
  }
}
```

#### Output (strict JSON)

```json
{
  "method": "lyapunov|barrier",
  "form": "direct|basis",
  "expression": "x1**2 + x2**2 + 0.1*x1*x2",
  "basis": ["x1**2","x2**2","x1*x2"],
  "coeffs": [1.0,1.0,0.1],
  "metadata": { "degree": 2, "notes": "" }
}
```

Reject anything non-JSON or outside syntax/ops; canonicalize (CAS) and simplify before checks.

## 3) Acceptance Protocol (rigor)

### 3.1 Stage A - Numeric (always)

- **Sampling:** Sobol/LHS on domain; oversample near B(x)=0, set boundaries, and spheres around equilibrium.
- **AD gradients:** compute ∇V, ∇B, V̇∇V·f, Δ=V∘f−V, L_fB=∇B·f.
- **Margins (tunable, stored with run):**
  - Positivity: V(x) ≥ε_pos‖x‖p (x≠).
  - Decrease: V̇x) ≤−ε_dec‖x‖p (ct) or Δ(x) ≤−ε_dec‖x‖p (dt).
  - Barrier: B≤−m_init on initial, B≥m_unsafe on unsafe, L_fB≤−m_inv on B≈.
- **Adaptive refinement + adversarial search** (maximize violation; keep counterexamples).

### 3.2 Stage B - Formal (selectively, when eligible)

- **SOS/SDP** for polynomial cases (Lyapunov & barrier multipliers).
- **SMT/δ-sat (e.g., dReal)** for non-polynomial constraints; record δ.
- **Reachability cross-check** (Flow*/CORA) on a subset.

**Acceptance Criteria:** Stage A pass **and** Stage B pass (when Stage B enabled). Persist **margins**, **tool versions**, and **artifacts**.

## 4) Metrics (per attempt)

- **Success** (binary), **time-to-acceptance** (breakdown by LLM, Stage A/B).
- **Strength:** Lyapunov: min_x V̇‖x‖p; largest accepted sublevel Ω_c. Barrier: m_init, m_unsafe, m_inv; Monte-Carlo estimate of accepted safe volume.
- **Complexity:** degree, term count, AST size.
- **Robustness:** param jitter pass rate; gradient-based violation bounds.
- **Budget used:** tokens, solver iters, wall-clock.
- **Counterexamples** until success.

## 5) UI/UX (Material 3 + CU Boulder branding)

**Auth first** (no model/solver pre-auth).

- Roles: admin (provision users), researcher, viewer.

### Flows

1. **Define System** (wizard): time type →state/dynamics (live parse/grad preview) →sets (builders + 2D/3D slice viz) →parameters/equilibrium →validate →save version.
2. **Configure Experiment:** task, LLM mode, constraints (degree/ops), budgets, margins, Stage B tools, baselines, seed.
3. **Run Monitor:** stream of candidates (normalized expr + LaTeX), acceptance checks, violation heatmaps/level sets, formal status.
4. **Result Detail (immutable):** final expression (symbolic + LaTeX), acceptance artifacts, margins, coverage estimates, complexity, prompts/model versions, timeline; export **PDF/JSON/LaTeX**.

**Guardrails:** schema-driven validation; margin warnings; disabled "Run" until green.

## 6) Persistence (SQL) - minimal DDL

**Implementation Note**: Current MVP uses simplified schema with Firestore backend. The detailed schema below represents the planned PostgreSQL architecture. Current implementation consolidates acceptance status directly in candidates table with fields: `acceptance_status ('pending'|'accepted'|'failed'|'timeout')`, `accepted_at`, `acceptance_duration_ms`.

```sql
create table users (
  id bigserial primary key,
  email text unique not null,
  role text check (role in ('admin','researcher','viewer')) not null,
  created_at timestamptz default now(),
  active boolean default true
);

create table systems (
  id bigserial primary key,
  owner_user_id bigint references users(id),
  name text not null,
  spec_version text not null,
  spec_json jsonb not null,
  hash text not null,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create table experiments (
  id bigserial primary key,
  system_id bigint references systems(id),
  task text check (task in ('lyapunov','barrier')) not null,
  config_json jsonb not null,
  llm_model text not null,
  seed bigint,
  created_by bigint references users(id),
  created_at timestamptz default now()
);

create table runs (
  id bigserial primary key,
  experiment_id bigint references experiments(id),
  status text check (status in ('pending','running','succeeded','failed','canceled')) not null,
  started_at timestamptz, 
  ended_at timestamptz,
  compute_env jsonb, 
  code_commit_sha text
);

create table candidates (
  id bigserial primary key,
  run_id bigint references runs(id),
  mode text check (mode in ('direct','basis','structure')) not null,
  expr_raw text not null,
  expr_canonical text not null,
  latex text not null,
  ast_json jsonb,
  degree int, 
  term_count int,
  gen_time_ms int
);

create table numeric_checks (
  id bigserial primary key,
  candidate_id bigint references candidates(id),
  sampler_config jsonb,
  n_samples int,
  n_violations int,
  min_margin_vals_json jsonb,
  coverage_estimate real,
  lipschitz_bounds_json jsonb,
  passed boolean not null
);

create table formal_checks (
  id bigserial primary key,
  candidate_id bigint references candidates(id),
  tool text, 
  tool_version text,
  settings_json jsonb,
  delta real, 
  sos_degree int,
  result text, 
  artifact_uri text
);

create table counterexamples (
  id bigserial primary key,
  candidate_id bigint references candidates(id),
  x_json jsonb not null,
  context text,
  violation_metrics_json jsonb
);

create table audit_events (
  id bigserial primary key,
  user_id bigint references users(id),
  action text, 
  entity_type text, 
  entity_id bigint,
  at timestamptz default now(),
  ip text, 
  user_agent text
);
```

## 7) GCP Architecture (serverless, reproducible)

### Resources

- **Cloud Run**: fmgen-ui (SPA) + fmgen-api (REST/WebSocket).
- **Cloud SQL (Postgres)**: relational store.
- **GCS**: artifacts (SOS certs, SMT logs, reports).
- **Secret Manager**: ANTHROPIC_API_KEY (Claude), other keys.
- **Identity Platform** (or IAP): email/password or SAML/OIDC; JWT to API.
- **Cloud Load Balancer (HTTPS, global)**: front fmgen.net, www.fmgen.net.
- **Cloud Logging/Monitoring**; **Error Reporting**.
- Optional: **Workload Identity**; **VPC Serverless Connector** (if needed).

### Networking & Domain

- Domain: **fmgen.net** (purchased).
- Current DNS:
  - @ A 34.55.217.224
  - www A 34.55.217.224
  - _domainconnect CNAME _domainconnect.domains.squarespace.com (harmless; unused by this stack).

**Recommendation:** attach your HTTPS Load Balancer to the **same static global IPv4** (or reserve another and update A records). Issue **Google-managed cert** for fmgen.net and www.fmgen.net. Redirect www →apex.

## 8) Deployment: concise commands (you run these)

Replace PROJECT, REGION, and keep secrets out of code. If the IP 34.55.217.224 is already reserved to your project, bind the LB to it; otherwise reserve a new static IP and update DNS.

### 8.1 Secrets

```bash
gcloud secrets create ANTHROPIC_API_KEY --replication-policy="automatic"
printf "%s" "<REDACTED_CLAUDE_KEY>" | gcloud secrets versions add ANTHROPIC_API_KEY --data-file=-
```

### 8.2 Cloud SQL (Postgres)

```bash
gcloud sql instances create fmgen-pg --database-version=POSTGRES_15 --tier=db-custom-2-7680 --region=us-central1
gcloud sql databases create fmgen --instance=fmgen-pg
# Create a user; store password in Secret Manager as FMGEN_DB_PASSWORD
```

### 8.3 Cloud Run services

```bash
# UI (static build served by a minimal container or Cloud Run static)
gcloud run deploy fmgen-ui --source=. --region=us-central1 --allow-unauthenticated

# API (requires auth)
gcloud run deploy fmgen-api --source=. --region=us-central1 \
  --set-secrets=ANTHROPIC_API_KEY=ANTHROPIC_API_KEY:latest \
  --set-env-vars=DB_INSTANCE="PROJECT:us-central1:fmgen-pg",DB_NAME="fmgen" \
  --ingress=all --allow-unauthenticated=false
```

### 8.4 Identity & Auth (Identity Platform →JWT)

- Enable **Identity Platform**; enable **email/password** (admin invites).
- API validates **Firebase/Identity-Platform JWT** on every endpoint; **no LLM/solver routes without JWT**.

### 8.5 HTTPS LB (Serverless NEG →Cloud Run)

```bash
# Create serverless NEGs for both services
gcloud compute network-endpoint-groups create fmgen-api-neg \
  --region=us-central1 --network-endpoint-type=serverless \
  --cloud-run-service=fmgen-api

gcloud compute backend-services create fmgen-backend --global \
  --load-balancing-scheme=EXTERNAL_MANAGED

gcloud compute backend-services add-backend fmgen-backend --global \
  --network-endpoint-group=fmgen-api-neg \
  --network-endpoint-group-region=us-central1

# URL map (/* →api; /app/* or static path →ui as needed)
gcloud compute url-maps create fmgen-urlmap --default-service=fmgen-backend

# Managed cert
gcloud compute ssl-certificates create fmgen-cert \
  --domains=fmgen.net,www.fmgen.net --global

# HTTPS proxy & forwarding rule
gcloud compute target-https-proxies create fmgen-https-proxy \
  --ssl-certificates=fmgen-cert --url-map=fmgen-urlmap

gcloud compute addresses create fmgen-ipv4 --global

gcloud compute forwarding-rules create fmgen-https-fr \
  --address=fmgen-ipv4 --global --target-https-proxy=fmgen-https-proxy \
  --ports=443
```

Ensure your **A records** (@, www) point at the IP printed by fmgen-ipv4. If you must keep 34.55.217.224, bind the forwarding rule to that address if it's reserved in your project.

## 9) API surfaces (auth-gated)

**Current Implementation**: MVP operates with Firebase/Firestore backend and simplified API structure. Planned endpoints below represent full PostgreSQL-based architecture.

**Implemented Endpoints**:
- POST /api/system-specs - create & validate system specifications
- GET /api/system-specs - list system specifications with pagination
- POST /api/certificates/generate - generate certificate candidates via LLM
- GET /api/certificates - list candidates with acceptance status
- Auth endpoints: POST /api/auth/login, /api/auth/register
- Admin endpoints: POST /api/admin/emails (email authorization)

**Planned Endpoints**:
- POST /v1/systems - validate & store SystemSpec, return immutable system_id + hash.
- POST /v1/experiments - config (task, LLM mode, budgets, margins, StageB flags, seed).
- POST /v1/runs - start run for experiment; server streams attempts (SSE/WebSocket).
- GET /v1/runs/{id} - status + results.
- GET /v1/candidates/{id} - expression, LaTeX, metrics, artifacts.
- **No endpoint** accessible without valid JWT. Admin-only: user lifecycle.

## 10) LLM usage (Claude 4 generation models)

- Provider: Anthropic Claude 4 (public API).
- **Key handling:** load from Secret Manager at runtime; **never** embed or log.
- **Budget controls per attempt:** temperature ∈{0.0, 0.2}, max tokens, max attempts.
- **Prompt scaffolding:** strict JSON output; terminate on non-JSON.

You supplied a Claude key. Don't paste it into code. Store it exactly once in **Secret Manager** (ANTHROPIC_API_KEY) and reference via Cloud Run secret mounts/env.

## 9.5) Current Service Architecture (Implemented)

**Core Services**:
- **AcceptanceService**: Replaces planned VerificationService. Implements two-stage acceptance protocol with `acceptCandidate()` method. Performs numerical validation (Stage A) and formal verification (Stage B when enabled).
- **LLMService**: Anthropic Claude integration with structured prompting, JSON validation, and error handling.
- **MathService**: Mathematical computation engine for Lyapunov/barrier condition checking, expression evaluation, and gradient computation.
- **BaselineService**: Classical method implementations (SOS, SDP, quadratic templates) for comparative analysis.
- **AuthService**: JWT-based authentication with Firebase/Firestore integration.
- **EmailAuthorizationService**: Admin email management for user access control.

**Key Implementation Details**:
- **Acceptance Protocol**: Stage A always runs (numerical sampling, margin checks). Stage B (formal SOS/SMT) planned for future implementation.
- **Candidate Processing**: Background acceptance checking with status updates (`pending` → `accepted`/`failed`).
- **Error Handling**: Comprehensive counterexample tracking and violation reporting.
- **Audit Trail**: Complete provenance logging for research reproducibility.

## 11) Baselines

- **SOS/SDP** (degree-matched) for polynomial dynamics.
- **Quadratic templates** around equilibria.
- **Energy heuristics** (mechanical systems).
- Same margins/budgets; paired per system.

## 12) Reproducibility & Provenance (hard requirements)

Log with every attempt:

- **SystemSpec hash**, spec_version; **experiment config**.
- LLM: provider, model/version, temperature, seed, full prompts/outputs (encrypted at rest), token counts.
- Verifiers: sampler type+seed, counts, margins, δ, SOS degree, solver versions.
- Timestamps, durations, hardware; code commit SHA.
- Immutable artifacts (GCS); runs are **append-only** after completion.

## 13) Security

- **Auth-first** gating; principle of least privilege (RBAC).
- Secrets in **Secret Manager** only; short-lived access tokens; **no** plaintext keys in logs.
- Managed TLS; HSTS; CSRF for any state-changing web flows; re-auth before launch.
- Full **audit_events** table (see DDL).

## 14) UI Implementation (Material 3 + CU Boulder branding)

**Current Implementation**: Full-featured React application with comprehensive user interface.

**Implemented Pages**:
- **About Page**: Comprehensive technical documentation including mathematical foundations (Lyapunov/barrier theory), acceptance protocol details, research methodology, system architecture, and LLM integration details.
- **Login/Register**: CU Boulder themed authentication with email authorization system.
- **Dashboard**: Statistics overview, recent candidates, system specs summary with acceptance status displays.
- **System Specs**: Table view with create/edit wizard, specification details pages.
- **Certificates**: Full candidate management with acceptance status filtering, detailed view with LaTeX rendering.
- **Profile**: User account management.
- **Admin**: Email authorization management for user provisioning.

**Planned UI Features**:
- **Experiments**: config form (task, mode, budgets, margins, baselines); seeded & pinned prompts.
- **Run view**:
  - Left: attempt list with pass/fail chips & degree/term count.
  - Right: tabs →**Expression** (normalized + LaTeX), **Numeric** (heatmaps/level sets), **Formal** (SOS/SMT status), **Artifacts**.
  - Prominent "Candidate accepted" banner with margins and tool versions.
- **Exports**: PDF, LaTeX, JSON bundle buttons.

**Design System**: Material Design 3 with CU Boulder gold/black branding, consistent typography (academic-header, academic-subheader, academic-body), and comprehensive status badge system for acceptance states.

## 15) Example test systems (ready to seed)

- **Van der Pol (ct)** mu=1, domain radius 4; unsafe disk at (2,0), r=0.5.
- **Damped oscillator (dt)** Δ=0.1, domain box [-4,4]^2, unsafe disk at (2,-2), r=0.8.

(Use the JSON patterns from §2.)

## 16) Checklists

### Before run

- SystemSpec parsed, validated, hashed.
- Experiment config frozen; budgets & margins set.
- Baselines enabled (or justified).
- Seeds set; model versions pinned.

### On acceptance

- Stage A passed with margins; adversarial search found no counterexamples.
- Stage B (if enabled) produced artifacts; stored in GCS.
- Results exported; run made immutable.

### Notes on fmgen.net DNS you provided

- @ A 34.55.217.224, www A 34.55.217.224 already point to a Google-owned IPv4. Attach your HTTPS LB to **that** IP (if reserved in your project) or reserve a new global static IP and update the two A records.
- _domainconnect CNAME is benign and unrelated to this stack.

## 17) Current Implementation Status & Next Steps

**What is Currently Implemented**:
- ✅ Authenticated, schema-driven system input with strict LLM outputs
- ✅ Core AcceptanceService with numerical validation (Stage A of acceptance protocol)
- ✅ Complete Material 3 UI with CU Boulder branding and comprehensive About page
- ✅ Firebase/Firestore backend with real-time candidate processing
- ✅ Full candidate lifecycle: generation → acceptance checking → status display
- ✅ Email authorization system and role-based access control
- ✅ Mathematical computation engine with Lyapunov/barrier condition checking
- ✅ Anthropic Claude 4 integration with structured prompting, JSON validation, and refusal handling

**Terminology Update**: System now uses "accepted" instead of "verified" throughout to reflect the cautious nature of numerical acceptance checks. Candidates are "accepted" when they pass our rigorous but not absolute numerical and formal validation processes.

**Planned Enhancements**:
- Stage B formal verification (SOS/SMT integration)
- PostgreSQL migration from Firestore
- Full experiment management and statistical analysis features
- Advanced visualization (heatmaps, level sets, 3D plots)
- PDF/LaTeX export functionality
- Domain fmgen.net deployment with Google-managed TLS

**Available for Implementation**:
- JSON Schemas for SystemSpec and LLM output (partially implemented)
- Terraform infrastructure templates
- PDF report templates for accepted candidates
- Advanced statistical analysis pipeline