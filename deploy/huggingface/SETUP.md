# Publishing the live demo on Hugging Face Spaces

Takes about two minutes. Nothing here needs a token in CI — the Space builds
from the already-public GHCR image.

## 1. Create the Space

Go to https://huggingface.co/new-space and set:

| Field | Value |
|---|---|
| Owner | your account |
| Space name | `insurance-fraud-detection` |
| License | MIT |
| SDK | **Docker** → *Blank* |
| Hardware | CPU basic (free) |
| Visibility | Public |

## 2. Push the two files in this folder

```bash
git clone https://huggingface.co/spaces/<your-username>/insurance-fraud-detection hf-space
cp deploy/huggingface/Dockerfile deploy/huggingface/README.md hf-space/
cd hf-space
git add -A && git commit -m "Serve the published fraud API image" && git push
```

Hugging Face will ask for your username and an access token as the password —
create one at https://huggingface.co/settings/tokens with **write** scope. That
token goes to Hugging Face only.

## 3. Check it

The Space builds in a couple of minutes, then:

- `https://<your-username>-insurance-fraud-detection.hf.space/ui/` — analyst UI
- `https://<your-username>-insurance-fraud-detection.hf.space/health` — health check
- `https://<your-username>-insurance-fraud-detection.hf.space/docs` — OpenAPI docs

## Why it pulls the image instead of building from source

The Space runs the exact image the pipeline published and smoke-tested. Building
from source inside the Space would produce a second artifact that can drift from
the tested one — same code, different bytes, and no smoke test in front of it.

To ship an update: merge to `main`, let the publish workflow run, then use the
Space's **Factory rebuild** to pull the new `latest`.

## Free-tier caveat

A free Space sleeps after a period of inactivity and takes a few seconds to wake
on the next request. That is fine for a portfolio link; it is not a production
SLA, and the README should not claim otherwise.
