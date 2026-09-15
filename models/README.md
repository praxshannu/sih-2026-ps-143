# models/

This directory holds the **deployed** model checkpoint. It is deliberately not
tracked — `*.pth` and `checkpoints/` are both gitignored, so a clone arrives
with this README and nothing else.

## What goes here

One file:

```
models/detector_best.pth      283 MB
```

Stage it from a training run. `20260915T180945Z-real-10ep-512-holdout-7e9d57` is
run C, the best checkpoint this repo has:

```bash
cd /Users/praxsmac/projects/claude1/sentinel
mkdir -p models
cp checkpoints/20260915T180945Z-real-10ep-512-holdout-7e9d57/detector_best.pth \
   models/detector_best.pth
ls -lh models/detector_best.pth      # expect ~283M
```

## Why this directory exists at all, rather than mounting `checkpoints/`

`checkpoints/` is 5.5 GB — 14 runs × 2 files each. Mounting it would put all of
that on a 40 GB volume for no reason. The container needs exactly one file.

## The failure this README is here to prevent

`docker-compose.aws.yml` mounts this directory read-only:

```yaml
    volumes:
      - ./models:/app/models:ro
    environment:
      MODEL_CHECKPOINT: /app/models/detector_best.pth
```

If `models/` is **missing**, Docker does not complain — it creates an empty
directory, the mount succeeds, and the container starts. Then
`services/detect/app/main.py` finds no file at `MODEL_CHECKPOINT` and takes this
branch:

```python
    else:
        logger.warning(
            f"Checkpoint not found at {MODEL_CHECKPOINT}, loading pretrained encoder only"
        )
        _model = UNetPlusPlusSCSE(encoder_weights="imagenet", ...)
```

That is an **untrained decoder**. The service answers `/health` with 200, every
endpoint responds, and the segmentation output is meaningless. Nothing crashes
and nothing looks wrong — which is exactly why it is worth a directory that
exists by default and a README that says so.

The file is small and cheap to check. Verify it, and verify the log line, before
you demo anything:

```bash
ls -lh models/detector_best.pth
docker compose -f docker-compose.aws.yml logs sentinel-detect \
  | grep -i -E 'Loading model|Checkpoint not found'
# want: "Loading model from /app/models/detector_best.pth"
# bad:  "Checkpoint not found ... loading pretrained encoder only"
```

`deploy/preflight.sh` runs this check for you.

## Checkpoint metadata

Read from the checkpoint's own payload rather than assumed:

| Field | Value |
|---|---|
| `in_channels` | **2** — `main.py` defaults to 6, which does not load |
| `encoder` | `resnet34` |
| `image_size` | 512 |
| `band_names` | `['VV', 'VH']` — labelling is inferred, see `LIMITATIONS.md` B12 |
| `best_val_iou` | 0.517951 |
| `provenance` | `real` |
| parameters | 24,738,772 |

Because `in_channels` is recorded in the payload, `docker-compose.aws.yml` pins
`IN_CHANNELS: "2"` explicitly. Loading a 2-channel checkpoint into a 6-channel
model raises a size mismatch on `encoder.conv1.weight` (`[64,2,7,7]` vs
`[64,6,7,7]`), and `strict=False` does **not** suppress that — it only tolerates
missing and unexpected keys.

## On the number to quote

`best_val_iou` is 0.517951. That is a **validation** figure, selected on the same
split it is reported from. The held-out test IoU for this run is **0.1447**, on 6
acquisitions, and the model over-predicts on unseen acquisitions. See
`docs/DEPLOYMENT_FREE_TIER.md` §10.1 and `HANDOVER.md` before presenting either
number.
