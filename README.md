# Stock Media Generation Automation

This repository contains a daily automation for a stock-safe AI image generation workflow.

## What the automation does

The automation runs once per day at **10:00 AM Europe/Istanbul** and creates a batch under `generated/YYYY-MM-DD/`.
It is configured for **5 concepts**, **4 images per concept**, and a target of **20 final images per day**. Generated image files are requested directly from Nano Banana 2 as native **4K PNG** files; the automation does not resize or upscale generated images after download.

Pipeline stages:

1. **Scheduler service** starts the workflow from GitHub Actions.
2. **Research service** performs trend-informed ideation, converts trends into stock-safe concepts, and scores each concept.
3. **Prompt service** creates prompt packets, technical specs, aspect-ratio decisions, and Gemini/Nano Banana 2 payloads.
4. **Generation service** calls Gemini/Nano Banana 2 and stores generated outputs.
5. **Review service** uses OpenAI to automatically approve or reject generated images against stock-safety and quality criteria.
6. **Metadata service** calls OpenAI with each auto-approved image plus prompt/topic context to generate Adobe Stock-oriented metadata.
7. **Export service** writes an Adobe Stock metadata CSV package.
8. **Notification service** sends one email only when the full batch is complete and ready.

## Stock-safety guardrails

Every stage is instructed to avoid:

- editorial/news framing;
- real people, celebrities, public figures, politicians, athletes, or influencers;
- brands, logos, trademarks, copyrighted characters, franchises, or protected product designs;
- misleading metadata that describes the prompt instead of the actual generated image.

The automated review step approves only commercially useful, non-editorial, stock-safe images. The metadata step uses prompt data as hints only and must describe the approved image itself.

## Troubleshooting GitHub Actions

If the workflow fails, open the failed run and expand **Run stock image automation**. The script emits GitHub error annotations for missing secrets, API failures, image-generation responses that do not contain image bytes, and image responses that are not PNG files. Gemini image generation uses Nano Banana 2 (`gemini-3.1-flash-image`) with `responseFormat.image.imageSize` set to `4K`.

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

Paste only the raw secret values. Do not include labels such as `OPENAI_API_KEY=`, quotes, or extra lines. The workflow strips surrounding whitespace but rejects secrets that still contain embedded line breaks because they can make API authorization headers invalid.

## Manual run

```bash
python scripts/stock_image_automation.py --config config/stock_image_automation.json
```

For a specific batch date:

```bash
python scripts/stock_image_automation.py --config config/stock_image_automation.json --date 2026-06-09
```

## Automated approval workflow

After generation, images are stored in:

```text
generated/YYYY-MM-DD/images/
```

Generated files use `.png` extensions and are written directly from the PNG bytes returned by Nano Banana 2. The automation does not resize them.

OpenAI reviews each generated image automatically. Approved images are copied into:

```text
generated/YYYY-MM-DD/approved/
```

The run writes an automated review summary to:

```text
generated/YYYY-MM-DD/review/automated_review_summary.json
```

Metadata and the Adobe Stock CSV are created in the same run for the auto-approved images:

```text
generated/YYYY-MM-DD/exports/adobe_stock_metadata.csv
```

You receive a single email only after review, metadata, and export steps are complete.
