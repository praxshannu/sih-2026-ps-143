# Task brief — publish SENTINEL to GitHub, then deploy it to AWS

> Paste everything between the rules below as the prompt. It is written to be
> self-contained: a fresh session with no memory of prior conversations should be
> able to execute it from this text alone.

---

## CONTEXT

**Project.** SENTINEL — an oil-spill detection and drift-forecasting stack
(SIH 2026, problem statement 143). Local path:

```
/Users/praxsmac/projects/claude1/sentinel
```

Git repo on branch `main`, HEAD `e09e0b6`, **250 tracked files (~2.8 MB)**, no remote
configured. The 16 GB working tree is mostly gitignored (`data/`, `checkpoints/`,
`node_modules/`, `*.pth`). Seven entries are currently **untracked**, including every
deployment artifact:

```
deploy/                        docker-compose.aws.yml    models/
docs/DEPLOYMENT_FREE_TIER.md   docs/DEPLOYMENT_AWS.md    services/ui/.dockerignore
docs/ARCHITECTURE_DESIGN.html
```

**Target GitHub repo.** `https://github.com/praxshannu/sih-2026-ps-143.git`
— **public**, `main` @ `d6aa6ec03b026db012b70fc9f9667937601d492f`. It currently holds a
different, complete project ("OceanTrace"): 192 tracked files, 27 commits. Replacing it
destroys that history.

**Existing deployment assets** (already written, use them — do not reinvent):

| file | what it is |
|---|---|
| `docs/DEPLOYMENT_FREE_TIER.md` | the full runbook: cost model, architecture, blockers, step-by-step |
| `docker-compose.aws.yml` | single-instance AWS profile, 8 services, 6.46 GB of memory caps |
| `deploy/preflight.sh` | executable gate — exits non-zero on any failed precondition |
| `deploy/maintenance/index.html` | dark-theme page shown while the instance is asleep |
| `models/detector_best.pth` | 283 MB trained checkpoint (gitignored, must be transferred out-of-band) |

**AWS.** Credentials already exist on this machine at `~/.aws/credentials`
(profile `[default]`), with `~/.aws/config` setting region `ap-south-2` (Hyderabad).
The `aws` CLI is **not installed**; `boto3 1.43.93` is. Budget: **~$100 of credits**,
to cover **1.5–2 months**. Runbook's chosen shape: `m7i-flex.large`, sleeping
22:00–08:00 IST — $0.105144/hr running, $0.009384/hr stopped, ≈$93.95 over 60 days.

---

## OBJECTIVE

**1. Replace the GitHub repo's contents with SENTINEL.**
Commit the seven untracked entries, add the remote, and force-push `main` so the repo's
content becomes SENTINEL. The old content is intentionally discarded.

**2. Deploy SENTINEL to AWS.**
Follow `docs/DEPLOYMENT_FREE_TIER.md` end to end: provision the instance, security group,
Elastic IP, S3 maintenance bucket, CloudFront failover, and the EventBridge
stop/start schedule; install Docker; get the code and the 283 MB checkpoint onto the box;
run `deploy/preflight.sh`; start the stack; verify it serves.

**3. Leave environment variables for me to fill in myself.**
Create the `.env` on the instance as a **complete template with empty values** — every key
present, no value set. Then stop and hand me the guide described in Objective 4. Do not
invent, guess, or copy values in yourself.

**4. Produce a step-by-step environment-variable guide.**
Exact instructions for where and how to paste each variable: the file path on the instance,
the SSH command to get there, the editor command, which line each key sits on, and what
value belongs there.

**5. Give a safety verdict before you touch anything.**
In writing, before execution begins: is this workflow safe given that I will not share my
AWS credentials directly? What are the residual risks? What could go wrong, and what would
you do if it did?

---

## CREDENTIALS — HOW THIS WORKS

I will **not** send you an access key, a secret key, or any credential value. Do not ask
for them, and do not treat their absence as a blocker.

Use the credentials already present at `~/.aws/credentials` (profile `[default]`). The
AWS SDK and CLI read that file themselves, so the secret never has to pass through this
conversation. If the CLI is needed, install it into the managed venv:

```bash
/Users/praxsmac/.workbuddy-ai/binaries/python/envs/default/bin/pip install awscli
```

Before spending anything, run `aws sts get-caller-identity` and report what identity the
key maps to. If the ARN ends in `:root`, **stop and tell me** — I will create a scoped IAM
user instead.

---

## CONSTRAINTS — non-negotiable

1. **Never print, log, echo, commit, or screenshot a secret value.** Not in terminal
   output, not in a file, not in a summary. If you need to reference a secret, name the
   key and point at the line of the source file — never reproduce the value.
2. **Never commit `.env`.** It is gitignored at `.gitignore:34` and contains live values.
   Never use `git add -f` on it. Before every push, verify:
   ```bash
   git status --short
   git ls-files | grep -iE '\.env$|secret|credential|\.pem$|\.key$|token'
   ```
   If anything sensitive appears, stop.
3. **Back up the target repo before the force-push.** This is a precondition, not a
   suggestion:
   ```bash
   mkdir -p ~/backups
   git clone --mirror https://github.com/praxshannu/sih-2026-ps-143.git \
     ~/backups/sih-2026-ps-143-$(date +%Y%m%d-%H%M%S).git
   ```
   Verify it captured all refs and commits. **If the backup fails, do not force-push.**
4. **Use `--force-with-lease`, not `--force`.**
   `git push --force-with-lease=refs/heads/main:d6aa6ec03b026db012b70fc9f9667937601d492f origin main`
5. **Print what will be destroyed, then proceed.** Before the force-push, state in one
   line: the target repo, the SHA being overwritten, and the file/commit count. Then do it.
   I have already authorised this — do not ask again.
6. **`deploy/preflight.sh` must pass** before the stack is started. If it fails, fix the
   cause or report it — do not start the stack anyway.
7. **Never bind a service to `0.0.0.0`.** Only the UI publishes a port, on loopback.
   Postgres and Redis must not be reachable from the internet.
8. **Respect the cost ceiling.** If anything in the plan would push the 60-day total past
   the credit balance, stop and tell me before provisioning.
9. **Never put a token in a git remote URL** — it is written to `.git/config` in plaintext.

---

## STOP AND ASK ME IF

- The credentials resolve to a **root** identity.
- The mirror backup fails or captures fewer commits than expected.
- `c7i-flex.large` / `m7i-flex.large` are **not available in `ap-south-2`** — the whole
  cost model assumes them. If they are absent, present the alternatives and re-cost.
- Preflight fails for a reason that is not obviously mechanical.
- Anything requires me to paste a secret into this conversation.
- A step would exceed the credit budget.

---

## THE ENVIRONMENT-VARIABLE GUIDE (Objective 4) — what good looks like

A numbered walkthrough that a tired person can follow at 1 a.m. without guessing:

1. The SSH command to reach the instance, with the exact host/IP.
2. The path of the `.env` file on the instance.
3. The command to open it in an editor.
4. **A table with one row per variable**, containing: key name · whether it is a secret ·
   what to paste · where the value comes from.
   - For **non-secrets** (ports, hostnames, batch sizes, thresholds, feature flags) state
     the literal value.
   - For **secrets** (`DB_PASSWORD`, `SECRET_KEY`, `SENTINEL_SECRET_KEY`, `CMEMS_USER`,
     `CMEMS_PASS`, `AIS_API_KEY`, and any other credential) do **not** print the value.
     Instead say: *"copy the value from your local `~/projects/claude1/sentinel/.env`,
     line N"*.

     **Resolve N with a command that cannot print the value.** This is safe — it
     yields the line number and nothing else:
     ```bash
     grep -n '^CMEMS_PASS=' ~/projects/claude1/sentinel/.env | cut -d: -f1
     ```
     This is **not** safe, and it is the mistake I made while writing this brief — it
     prints the secret into the conversation, where it cannot be recalled:
     ```bash
     grep -nE '^(DB_PASSWORD|CMEMS_PASS|AIS_API_KEY)=' ~/projects/claude1/sentinel/.env   # DO NOT RUN
     ```
     Never run a `grep` against `.env` whose pattern can match a value-bearing line
     without `cut -d: -f1`. When in doubt, use `cut`.
5. The command to restart the stack so the new values take effect.
6. How to confirm it worked — the exact health-check commands and the expected response.
7. What to do if a service reports unhealthy after the restart.

Enumerate every key that `docker-compose.aws.yml` and the services actually read. Do not
leave a required variable out, and do not include keys nothing consumes.

---

### Known problems in the local `.env` — resolve these before deploying

Found by inspection. The guide must tell me how to fix each one.

1. **`AIS_API_KEY` is defined twice** — line 29 is empty, line 91 is populated. Which one
   wins depends on the loader: `python-dotenv`, Docker Compose's `env_file`, and a shell
   `source` do not all resolve duplicates the same way. Deploying this as-is is a coin
   flip. Collapse it to a single line and say which value to keep.
2. **Three secrets are still placeholders** and must be replaced before the stack starts:
   - `DB_PASSWORD` (line 6) is the default `sentinel_secret`
   - `SECRET_KEY` (line 16) is still `change-me-in-production-…`
   - `SENTINEL_SECRET_KEY` (line 21) is still `change-me-in-production`

   Tell me to generate each with `openssl rand -hex 32`. `deploy/preflight.sh` gates
   these — it must not be bypassed.
3. `.env` holds **73 keys**. The guide must cover every key the services actually read,
   and omit any that nothing consumes.

---

## DEFINITION OF DONE

- [ ] Safety verdict delivered, in writing, before any mutation.
- [ ] Mirror backup of the target repo exists; ref and commit counts reported.
- [ ] Seven untracked entries committed; `git status` clean; no secrets staged.
- [ ] `main` on GitHub now contains SENTINEL; the old SHA is recorded in the summary.
- [ ] AWS resources provisioned, and their IDs/ARNs listed in the summary.
- [ ] `deploy/preflight.sh` passed on the instance.
- [ ] `.env` present on the instance as a complete empty-valued template.
- [ ] Environment-variable guide delivered per the section above.
- [ ] **No secret value appears anywhere in your output.**

Report back with: what you did, the resource identifiers, the target repo's previous SHA,
the exact command I should run next, and anything you could not complete.

---
