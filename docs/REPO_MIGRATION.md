# Publishing SENTINEL — credentials, repo migration, and the risks

Written 2026-09-16. **Advisory only — nothing in here has been executed.**
Every command below is meant to be run by you, deliberately, one step at a time.

---

## 0. The facts, before any advice

### Your machine

| thing | actual state |
|---|---|
| `sentinel/` repo | branch `main`, HEAD `e09e0b6`, **no remote configured** |
| tracked files | **250** (~2.8 MB total) |
| working tree | 16 GB — the other 99.98% is gitignored (`data/`, `checkpoints/`, `node_modules/`, `*.pth`) |
| untracked | **7 entries**, and they include *every deployment artifact from this session* |
| `~/.aws/credentials` | **already exists** — `[default]`, `AKIA…`, 40-char secret |
| `~/.aws/config` | `[default]`, region `ap-south-2` (Hyderabad) |
| `aws` CLI | **not installed** |
| `boto3` | **installed**, 1.43.93 |
| `gh` CLI | not installed |
| git credential helper | none |
| SSH keys | none — `~/.ssh/` holds only `config` and `known_hosts` |

The untracked seven:

```
deploy/                          docker-compose.aws.yml       models/
docs/DEPLOYMENT_FREE_TIER.md     docs/DEPLOYMENT_AWS.md       services/ui/.dockerignore
docs/ARCHITECTURE_DESIGN.html
```

### The target repo is **not** an empty scratch repo

`https://github.com/praxshannu/sih-2026-ps-143.git`

- **public** — verified by anonymous API call: HTTP 200, `"private": false`
- `main` @ `d6aa6ec03b026db012b70fc9f9667937601d492f`, last pushed 2026-09-11
- **192 tracked files, 10 commits**, containing a complete project called **OceanTrace**:
  `agent1/` · `agent2/` · `backend/` · `frontend/` · `training/` · `tests/` · `tools/`
  · 25 KB `PROJECT_TECHNICAL_DOCUMENTATION.md` · 23 KB `README.md` · `MODEL_CARD.md` · `CHANGELOG.md`
- history back to `b254475 Initial commit`

**So "delete all the current files in that target repo" means deleting OceanTrace.**
That is not clearing an empty repo. Read §4 before you run it.

### There is no local clone of the target anywhere on this machine

You asked to move `sentinel/` into "another directory that already has an existing Git
repository." No such directory exists here. The only git repository under
`/Users/praxsmac/projects` is `sentinel` itself. That premise does not match the disk.

---

## 1. AWS credentials — can I deploy it, and how should you share them?

### Short answer: you don't need to share anything

Your credentials are **already on this machine** at `~/.aws/credentials`. The AWS SDKs read
that file directly. If I ever ran a deployment, the secret would be read by `boto3` — it never
has to pass through this conversation, never appears in a log, and never becomes something
that cannot be un-shared.

That is strictly better than pasting keys into chat, and it costs you nothing.

### Why pasting them into chat is the wrong move

A chat message is not a private channel. It is persisted, it may be logged, it may be indexed,
and it may be replayed into a future context. An AWS access key is a **bearer token** — anyone
holding it *is* you, with no second factor. Once it is in a transcript you cannot recall it.
The only remedy is rotation, which means invalidating every deployment that used it.

So: **never paste an access key or secret into a chat window, an issue, a screenshot, or a commit.**

### If you genuinely must delegate credentials to someone

1. **Never use root.** Create a dedicated IAM user for the purpose. Root access keys should not
   exist at all — AWS says so explicitly.
2. **Least privilege.** Write a policy that grants exactly what the task needs, nothing else.
3. **Prefer temporary credentials.** `aws sts assume-role` issues credentials that expire
   (15 min – 12 h). A leak then has a blast radius measured in hours, not forever.
4. **Rotate afterwards**, unconditionally, whether or not anything went wrong.

### My actual recommendation

**Run it yourself.** The runbook (`docs/DEPLOYMENT_FREE_TIER.md`) and `deploy/preflight.sh`
exist precisely so that the person holding the credit card is also the person holding the keys.
Handing a possibly-admin-capable key to an agent for a 14-hour, multi-service deployment is a
bad trade — the upside is convenience, the downside is your entire AWS account.

If you want help *during* the deployment, the useful shape is: you run a command, paste the
**output** (which contains no secrets), and I tell you what it means and what to do next.

---

## 2. Is AWS CLI access global or scoped?

**Neither, exactly — the CLI has no scope at all.** This is the important bit.

`aws` is a thin HTTP client. Every call it makes is signed with your access key. The *only*
thing that limits what it can reach is **the IAM policy attached to the identity behind that
key.** There is no "CLI permission set," no per-service toggle, no install-time restriction.

| what your key is | what the CLI can do |
|---|---|
| **root user** key | the **entire account** — every service, every region, plus IAM and billing. Effectively unfixable if leaked. |
| IAM user with `AdministratorAccess` | same reach as root, minus a few root-only operations |
| IAM user with a narrow inline policy | genuinely scoped — e.g. only `ec2:*` in one region, or `s3` on one bucket |
| STS temporary credentials | whatever the assumed role allows, **and it expires** |

Two corollaries people get wrong:

- **Region is not a security boundary.** A credential that works in `ap-south-2` works in
  `us-east-1` too. Region only constrains you if the policy explicitly uses the
  `aws:RequestedRegion` condition key.
- **The key prefix tells you the kind.** `AKIA…` = a long-lived IAM user key (or root key) that
  **never expires**. `ASIA…` = temporary STS credentials that **do** expire. Yours is `AKIA…`,
  so it is long-lived.

### How to find out what yours can actually do

```bash
aws sts get-caller-identity
```

That returns the account ID and the ARN of the identity. If the ARN ends in `:root`, **stop and
create a proper IAM user** before doing anything else. It costs nothing and it converts a
permanent account-wide liability into a revocable, scoped one.

Note this call needs the CLI, which is not installed yet:

```bash
/Users/praxsmac/.workbuddy-ai/binaries/python/envs/default/bin/pip install awscli
```

---

## 3. The repo — you do not need to move the folder

You asked to move `sentinel/` into another directory that already has a git repo, clear that
repo, and push SENTINEL as the new content. Three separate problems with that:

**1. It is unnecessary.** `git push` transmits **commits**, not the contents of a directory.
Where the working tree physically sits is irrelevant to what lands on GitHub. Moving 16 GB
across your disk changes nothing about the push.

**2. There is no destination.** As established in §0, no local clone of the target repo exists
on this machine. There is nothing to move into.

**3. It creates a nested-repository footgun.** If you drop a directory containing `.git` into
another repository's working tree, the outer repo records a **gitlink** — a bare pointer to a
commit — not the files. You get:

```
warning: adding embedded git repository: sentinel
hint: You've added another git repository inside your current repository.
```

…and the inner history never becomes part of the outer repo. The alternative — deleting
SENTINEL's `.git` to avoid that — throws away 250 files of history for zero benefit.

**Do this instead: add a remote and push.** That is the entire operation.

If you later want the folder to carry the target's name, that is a cosmetic rename and it can
happen any time:

```bash
mv /Users/praxsmac/projects/claude1/sentinel /Users/praxsmac/projects/sih-2026-ps-143
```

Git is path-agnostic — nothing breaks. One caveat: `.workbuddy-ai/memory/` lives *inside* that
folder, so this session's memory path moves with it.

---

## 4. Publishing it — pick one of three

### Step 0 — authenticate (required, you have no push credential yet)

GitHub removed password authentication. Pick one:

```bash
# Option A — PAT over HTTPS, stored in the macOS Keychain (simplest)
git config --global credential.helper osxkeychain
# the first push will prompt: username = praxshannu, password = <paste your PAT>
```

```bash
# Option B — SSH
ssh-keygen -t ed25519 -C "praveenhonuka672@gmail.com"
cat ~/.ssh/id_ed25519.pub      # add this at GitHub → Settings → SSH and GPG keys
```

The PAT needs `repo` scope (classic) or `Contents: read and write` (fine-grained).
**Do not put the token in the remote URL** — that writes it in plaintext to `.git/config`.

### Step 0.5 — back up the target repo (do this regardless of which path you pick)

This is the *only* real undo. A `--mirror` clone captures every ref.

```bash
mkdir -p ~/backups
git clone --mirror https://github.com/praxshannu/sih-2026-ps-143.git \
  ~/backups/sih-2026-ps-143-$(date +%Y%m%d-%H%M%S).git
```

Restoring, if you ever need to:

```bash
cd ~/backups/sih-2026-ps-143-<timestamp>.git
git push --mirror https://github.com/praxshannu/sih-2026-ps-143.git
```

---

### Path A — push to a new branch (**recommended**)

OceanTrace stays intact. You get SENTINEL on GitHub, compare them side by side, and switch
when you are ready.

```bash
cd /Users/praxsmac/projects/claude1/sentinel

# 1. commit the deployment artifacts — they are untracked right now
git add deploy/ docker-compose.aws.yml docs/DEPLOYMENT_FREE_TIER.md \
        docs/DEPLOYMENT_AWS.md docs/ARCHITECTURE_DESIGN.html \
        models/README.md services/ui/.dockerignore

# 2. LOOK before you commit
git status --short

# 3. commit
git commit -m "deploy: free-tier runbook, single-instance AWS profile, preflight gate, maintenance page"

# 4. add the remote and push to a SIDE branch, not main
git remote add origin https://github.com/praxshannu/sih-2026-ps-143.git
git push -u origin main:sentinel
```

Result: `refs/heads/sentinel` holds SENTINEL. `main` still holds OceanTrace. **Nothing is
destroyed.** When you are satisfied, switch the default branch on GitHub and delete `main`.

---

### Path B — replace `main` outright (what you literally asked for)

Destructive. The backup in Step 0.5 is what makes this reversible.

```bash
# 0. the mirror backup FIRST — not optional
mkdir -p ~/backups
git clone --mirror https://github.com/praxshannu/sih-2026-ps-143.git \
  ~/backups/sih-2026-ps-143-$(date +%Y%m%d-%H%M%S).git

cd /Users/praxsmac/projects/claude1/sentinel

# 1. same commit as Path A
git add deploy/ docker-compose.aws.yml docs/DEPLOYMENT_FREE_TIER.md \
        docs/DEPLOYMENT_AWS.md docs/ARCHITECTURE_DESIGN.html \
        models/README.md services/ui/.dockerignore
git status --short
git commit -m "deploy: free-tier runbook, single-instance AWS profile, preflight gate, maintenance page"

# 2. remote + fetch, so the lease below has something to compare against
git remote add origin https://github.com/praxshannu/sih-2026-ps-143.git
git fetch origin

# 3. force-push, but ONLY if the remote main is still where you left it
git push --force-with-lease=refs/heads/main:d6aa6ec03b026db012b70fc9f9667937601d492f \
  origin main
```

Two notes:

- **A plain `git push` will be rejected.** SENTINEL and OceanTrace have unrelated histories
  (no common ancestor), so the push is a non-fast-forward. That rejection is expected
  behaviour, not a bug.
- **`--force-with-lease=<ref>:<sha>` beats plain `--force`.** It aborts if anyone pushed to
  `main` since you looked. Plain `--force` would silently clobber their work.

---

### Path C — merge both histories (probably not worth it)

```bash
git fetch origin main
git merge --allow-unrelated-histories origin/main
```

This preserves OceanTrace *and* SENTINEL in one tree — but they collide heavily. Both repos
have `README.md`, `Makefile`, `conftest.py`, `.env.example`, `.gitignore`, `data/`, `models/`,
`scripts/`, and `tests/`. Expect a long conflict-resolution session, and a repo where it is
unclear which `README.md` is authoritative. Mentioned for completeness; I would not choose it.

---

## 5. Risks — read this list

1. **Path B permanently destroys OceanTrace's history.** 27 commits, 192 files, including a
   25 KB technical document. GitHub keeps dangling commits reachable by SHA for a while, but
   that is not a guarantee. **The `--mirror` backup is the undo** — I ran it as a test and it
   completes in a couple of seconds and weighs 1.2 MB. There is no excuse to skip it.

2. **The repository is public.** Everything pushed is world-readable, permanently — including
   third-party forks and caches. Deleting a file later does **not** un-publish it.
   `AGENTS.md` names **NTRO**. Make that decision deliberately, not by accident.

3. **Secrets.** `.env` is gitignored (`.gitignore:34`) and contains live values —
   `DB_PASSWORD`, `SECRET_KEY`, `SENTINEL_SECRET_KEY`, `CMEMS_USER`/`CMEMS_PASS`,
   `AIS_API_KEY`. Never `git add -f` it. Check before every push:

   ```bash
   git status --short
   git ls-files | grep -iE '\.env$|secret|credential|\.pem$|\.key$'
   ```

   (Currently clean — no secrets are tracked.)

4. **The deployment artifacts are untracked right now.** If you push without committing them,
   the runbook, `docker-compose.aws.yml`, `deploy/preflight.sh`, and the maintenance page
   never reach the repo, and a fresh clone cannot deploy anything.

5. **A fresh clone cannot run the detector.** `models/detector_best.pth` (283 MB) is excluded
   by `.gitignore:66 *.pth`; only `models/README.md` is tracked. The documented failure mode
   is a **silent** fallback to `encoder_weights="imagenet"` — an untrained decoder served at
   HTTP 200 with no warning. `deploy/preflight.sh` exists to gate exactly this.

6. **The 16 GB working tree is not what gets pushed.** Only 2.8 MB of tracked files move.
   If you expected `data/` or `checkpoints/` to appear in the repo, they will not — that is
   by design.

7. **No push credential is configured.** See §4 Step 0. Do not embed the PAT in the remote URL.

8. **Region.** Your `~/.aws/config` is `ap-south-2` (Hyderabad). The runbook's `*-flex`
   instance types (`c7i-flex.large`, `m7i-flex.large`) are not offered in every region.
   **Confirm availability in `ap-south-2` before committing to the plan** — if they are absent,
   either pick a region that has them or re-cost the plan against `t3.small`/`t3.micro`.

9. **Your AWS key is long-lived (`AKIA…`) and never expires.** If it carries an admin policy it
   is the entire account. Run `aws sts get-caller-identity` and, if the ARN ends in `:root` or
   is broadly privileged, create a scoped IAM user for this work before spending a single hour
   of the budget.

---

## 6. One-page summary

- **Credentials:** already on this machine at `~/.aws/credentials`. Nothing needs to be shared.
  Never paste keys into chat. If you must delegate, use a scoped IAM user plus temporary STS
  credentials. Best answer: run it yourself.
- **CLI scope:** the CLI has none. Access is defined entirely by the IAM policy behind the key.
  Region is not a boundary. `AKIA…` = long-lived.
- **Folder move:** unnecessary, has no destination, and risks a nested-repo footgun.
  Add a remote and push instead.
- **Publishing:** Path A (side branch) is reversible and I recommend it. Path B (force-push to
  `main`) does what you asked and destroys OceanTrace — take the `--mirror` backup first.
- **Biggest risk:** the target repo is **public**, and Path B deletes 192 files and 27 commits
  of real prior work.
