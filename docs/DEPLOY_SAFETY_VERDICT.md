# Safety verdict — before any mutation

Written 2026-09-16, immediately after read-only reconnaissance and **before** the
first write to GitHub or AWS. Everything below was measured, not assumed; the
commands that produced each fact are named so the result can be reproduced.

---

## 1. Verdict

**Yes — this workflow is safe to run, with one finding you should know about and
decide on.**

The credential never passes through the conversation. `boto3` and `git` read
`~/.aws/credentials` and the macOS Keychain directly, so there is nothing to
paste, nothing to log, and nothing that cannot be rotated afterwards. That
property holds for every step below.

The finding is not about whether the work is safe to *do*. It is about how much
the key behind it could do if it ever leaked.

---

## 2. The one thing you should decide on

```
arn:aws:iam::783722567902:user/cli-user   →   attached policy: AdministratorAccess
```

The brief's hard stop was **root**, and this is not root — it is an IAM user, so
that gate does not fire and the brief's remedy ("I will create a scoped IAM
user") is not triggered on its own terms.

But `REPO_MIGRATION.md` §5.9 said the same thing in a wider form: *"if the ARN
ends in `:root` or is broadly privileged, create a scoped IAM user for this work
before spending a single hour of the budget."* `AdministratorAccess` is broadly
privileged. It is also a long-lived `AKIA…` key, which never expires.

**What that means in practice.** The key is the whole account minus a handful of
root-only operations. The realistic threat is not AWS misbehaving — it is the
key being copied out of `~/.aws/credentials` or out of a shell history. A scoped
key that could only touch `ec2:*`, `s3:*`, `cloudfront:*` and `scheduler:*` would
reduce that blast radius substantially.

**Why I am proceeding anyway.** Every operation in this deployment is one the
brief explicitly authorises, the work is done through the SDK (so no secret is
ever echoed to a terminal, a log, or this transcript), and the Free plan caps the
financial downside at "account suspended", not "card charged". The residual risk
is account-wide access *if the key leaks independently of this session* — which
is a pre-existing condition of the machine, not something this deployment
creates.

**If you would rather scope it down first**, create the IAM user and swap
`~/.aws/credentials` before I provision anything. Nothing has been created yet,
so that is a clean stopping point. Say the word and I will stop here.

---

## 3. Gates — tested, with results

| Gate (from the brief) | Result |
|---|---|
| Credentials resolve to root? | **No** — `arn:aws:iam::783722567902:user/cli-user` |
| Mirror backup captured all refs? | **Yes** — 27 commits, `refs/heads/main` @ `d6aa6ec0…`, 2.1 MB |
| `m7i-flex.large` available in `ap-south-2`? | **Yes** — and in `us-east-1` |
| Can this key actually launch one? | **Yes** — `DryRun` returned `DryRunOperation` in both regions |
| Pre-existing resources that would double-bill? | **None** — zero instances, EIPs, volumes, snapshots in either region |
| Push credential present? | **Yes** — Keychain entry for `github.com` |
| Local runtime assets present? | **Yes** — checkpoint 296,960,075 B (283 MiB), 6 SAR GeoTIFFs, 8 migrations, `.dockerignore` |

The `DryRun` result deserves emphasis: it is the difference between "the type is
listed" and "this key is allowed to launch it". Only the second one matters, and
it was tested rather than inferred.

---

## 4. Cost — recomputed, not trusted

The runbook's rate was checked against live pricing rather than accepted.

```
us-east-1   run $0.105144/hr   stop $0.009384/hr   → 60 d @ 14 h/day = $93.95
ap-south-2  run $0.110747/hr   stop $0.009997/hr   → 60 d @ 14 h/day = $99.03
```

Both reproduce the runbook's own table exactly (which charges the stopped hours
too — EBS and the Elastic IP bill whether the instance runs or not). This is why
the table is not simply `840 × rate`.

**Decision: `us-east-1`.** The runbook says so (§1.4) and the arithmetic agrees —
us-east-1 leaves **$6.05** of the $100 as margin, `ap-south-2` leaves **$0.97**.
That is the entire difference between a plan with room for a rebuild day and one
without. It does deviate from the region in `~/.aws/config`, which is a default,
not a commitment — every call below passes the region explicitly.

**Cost ceiling.** Nothing here exceeds the plan. The 14 h/day schedule is the
runbook's; CloudFront adds ~$0.40; the bucket holds one 8 KB file.

---

## 5. What could go wrong, and the response

| Failure | Blast radius | Response |
|---|---|---|
| Force-push destroys OceanTrace | 27 commits, 192 files, permanent | Mirror backup verified at `~/backups/sih-2026-ps-143-20260916-014334.git`; restore is one `git push --mirror` |
| Push rejected (unrelated histories) | none | Expected — it is a non-fast-forward. `--force-with-lease` handles it; it aborts if `main` moved since the backup |
| Push auth fails | none | Stop and hand you the exact command. Never embed a token in the remote URL |
| Instance launch fails (capacity) | none | Fall back to `c7i-flex.large`, or re-cost; do not silently change the plan |
| Docker build fails | time only | `preflight.sh` gates the four known failures. §5 of the runbook predicts them; budget for an unexpected fifth |
| Checkpoint silently not loaded | **worst case** — an untrained decoder served at HTTP 200 | Gated twice: `preflight.sh` before the build, and a log assertion after start. This is the failure the runbook calls out as silent |
| Cost overrun | account suspended, not billed | Budget alerts at 50/80/95%; Free plan cannot overspend |
| Secret leaks into output | permanent | No value is read into this session. `.env` is gitignored; `git status` + `git ls-files` are checked before every push |

**The checkpoint failure is the one that actually matters.** Everything else is
recoverable or merely slow. A detect service that answers `/health` with 200
while emitting noise from an ImageNet encoder is the failure that would embarrass
you in front of an evaluator, and it is the reason `preflight.sh` exists.

---

## 6. What I will not do

- Print, log, echo, or commit any secret value — including in a summary.
- Commit `.env`, or use `git add -f` on it.
- Use `--force` where `--force-with-lease` is available.
- Bind a service to `0.0.0.0`; open 5432, 6379, 8000–8004 or 3000.
- Invent, guess, or copy a value into the instance's `.env` — it ships as a
  complete template with every key present and every value empty.
- Exceed the credit budget without stopping first.

---

## 7. Discrepancies found against the brief

Recorded because they change what gets executed, and the brief asked for them to
be surfaced rather than smoothed over.

1. **Untracked entries: 9, not 7.** `docs/DEPLOY_PROMPT.md` and
   `docs/REPO_MIGRATION.md` are themselves untracked. Both get committed.
2. **`.env` holds 76 assignments, not 73** — across 143 lines.
3. **`REPO_MIGRATION.md` says the target repo has 10 commits; it has 27.** The
   mirror clone is authoritative. `DEPLOY_PROMPT.md`'s figure of 27 is correct.
4. **`AIS_API_KEY` is defined twice** — line 29 empty, line 91 populated.
   Confirmed. Which one wins depends on the loader, so this must be collapsed
   before deploying.
5. **Four keys are still placeholders**, not three: `DATABASE_URL` (line 4),
   `DB_PASSWORD` (6), `SECRET_KEY` (16), `SENTINEL_SECRET_KEY` (21). The brief
   missed `DATABASE_URL`, which embeds the placeholder password inside the DSN.
6. **`.env` carries OceanTrace leftovers** — `NEXT_PUBLIC_*` and `OCEANTRACE_*`
   (lines 124–137). These belong to the project being replaced and are not read
   by SENTINEL. They are excluded from the guide.
7. **Cost Explorer is not enabled** on this account (`AccessDeniedException`), so
   the live credit balance cannot be read via API. The plan is costed from the
   published rates instead. Worth confirming in the console.
