# Stock Media Generation Automation

This repository contains a daily automation for a stock-safe AI image generation workflow.

## What the automation does

The automation runs once per day at **10:00 AM Europe/Istanbul** and creates a batch under `generated/YYYY-MM-DD/`.
It is configured for **5 concepts**, **4 images per concept**, and a target of **20 final images per day**.

Pipeline stages:

1. **Scheduler service** starts the workflow from GitHub Actions.
2. **Research service** performs trend-informed ideation, converts trends into stock-safe concepts, and scores each concept.
3. **Prompt service** creates prompt packets, technical specs, aspect-ratio decisions, and Gemini/Nano Banana 2 payloads.
4. **Generation service** calls Gemini/Nano Banana 2 and stores generated outputs.
5. **Review service** emails an image review request. Approve images by moving them into the batch `approved/` folder.
6. **Metadata service** calls OpenAI with the approved image plus prompt/topic context to generate Adobe Stock-oriented metadata.
7. **Export service** writes an Adobe Stock metadata CSV package.
8. **Notification service** emails image review, metadata review, and final batch-ready alerts.

## Stock-safety guardrails

Every stage is instructed to avoid:

- editorial/news framing;
- real people, celebrities, public figures, politicians, athletes, or influencers;
- brands, logos, trademarks, copyrighted characters, franchises, or protected product designs;
- misleading metadata that describes the prompt instead of the actual generated image.

The metadata step uses prompt data as hints only. It must describe the approved image itself.

## Troubleshooting GitHub Actions

If the workflow fails, open the failed run and expand **Run stock image automation**. The script emits GitHub error annotations for missing secrets, API failures, and image-generation responses that do not contain image bytes.

The workflow opts JavaScript actions into Node.js 24 with `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24=true` to avoid the GitHub-hosted runner warning about Node.js 20 action deprecation.

## Required secrets

Add these repository or environment secrets before enabling the workflow:

- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `STOCK_AUTOMATION_EMAIL_TO`
- `STOCK_AUTOMATION_EMAIL_FROM`
- `STOCK_AUTOMATION_SMTP_HOST`
- `STOCK_AUTOMATION_SMTP_PORT`
- `STOCK_AUTOMATION_SMTP_USERNAME`
- `STOCK_AUTOMATION_SMTP_PASSWORD`

## Manual run

```bash
python scripts/stock_image_automation.py --config config/stock_image_automation.json
```

For a specific batch date:

```bash
python scripts/stock_image_automation.py --config config/stock_image_automation.json --date 2026-06-09
```

## Human approval workflow

After generation, review images in:

```text
generated/YYYY-MM-DD/images/
```

Move approved images into:

```text
generated/YYYY-MM-DD/approved/
```

The next automation run will detect approved images, generate metadata from each approved image and its prompt context, and create:

```text
generated/YYYY-MM-DD/exports/adobe_stock_metadata.csv
```
