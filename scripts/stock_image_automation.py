#!/usr/bin/env python3
"""Daily stock image generation automation.

Pipeline stages:
1. Research: trend research, stock-safe concept conversion, concept scoring.
2. Prompt: prompt packets, technical specs, Nano Banana JSON payloads.
3. Generation: calls Gemini/Nano Banana and stores outputs.
4. Review: emails image approval instructions.
5. Metadata: for images moved to approved/, calls OpenAI with image + prompt context.
6. Export: writes Adobe Stock-oriented CSV package.
7. Notification: emails metadata/final status.
"""
from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import json
import mimetypes
import os
import re
import smtplib
import ssl
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Any

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
GEMINI_GENERATE_URL = "https://generativelanguage.googleapis.com/v1/models/{model}:generateContent"
SUPPORTED_IMAGE_SUFFIXES = {".png"}
DEFAULT_OUTPUT_FORMAT = "PNG"
DEFAULT_OUTPUT_EXTENSION = ".png"
DEFAULT_IMAGE_SIZE = "4K"


@dataclass(frozen=True)
class BatchPaths:
    root: Path
    research: Path
    prompts: Path
    images: Path
    review: Path
    approved: Path
    metadata: Path
    exports: Path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def image_output_settings(config: dict[str, Any]) -> tuple[str, str]:
    image_settings = config.get("image_output", {})
    output_format = str(image_settings.get("format", DEFAULT_OUTPUT_FORMAT)).upper()
    extension = str(image_settings.get("extension", DEFAULT_OUTPUT_EXTENSION)).lower()
    if output_format == "JPG":
        output_format = "JPEG"
    if not extension.startswith("."):
        extension = f".{extension}"
    return output_format, extension


def image_size_setting(config: dict[str, Any]) -> str:
    return str(config.get("image_output", {}).get("image_size", DEFAULT_IMAGE_SIZE)).upper()


def approved_image_paths(paths: BatchPaths) -> list[Path]:
    return sorted(path for path in paths.approved.iterdir() if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES)


def keep_directory(path: Path) -> None:
    (path / ".gitkeep").touch(exist_ok=True)


def detect_image_extension(image_bytes: bytes, mime_type: str | None = None) -> str | None:
    normalized_mime = (mime_type or "").split(";", 1)[0].strip().lower()
    if normalized_mime == "image/png" or image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if normalized_mime in {"image/jpeg", "image/jpg"} or image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if normalized_mime == "image/webp" or image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return ".webp"
    return None


def write_direct_png(image_bytes: bytes, mime_type: str | None, destination: Path) -> dict[str, Any]:
    detected_extension = detect_image_extension(image_bytes, mime_type)
    if detected_extension != ".png":
        raise RuntimeError(
            f"Gemini returned {mime_type or 'an unknown image type'} ({detected_extension or 'unrecognized bytes'}) "
            "instead of a PNG. The automation did not rename or convert it; adjust the model/output settings and rerun."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(image_bytes)
    return {
        "source_mime_type": mime_type or "image/png",
        "output_format": "PNG",
        "extension": destination.suffix,
        "direct_from_model": True,
        "resized": False,
    }


def today_iso() -> str:
    return dt.datetime.now(dt.UTC).date().isoformat()


def batch_paths(output_root: str, approval_folder_name: str, batch_date: str) -> BatchPaths:
    root = Path(output_root) / batch_date
    paths = BatchPaths(
        root=root,
        research=root / "research",
        prompts=root / "prompts",
        images=root / "images",
        review=root / "review",
        approved=root / approval_folder_name,
        metadata=root / "metadata",
        exports=root / "exports",
    )
    for path in paths.__dict__.values():
        path.mkdir(parents=True, exist_ok=True)
    keep_directory(paths.approved)
    return paths


def safe_url_for_error(url: str) -> str:
    return re.sub(r"([?&]key=)[^&]+", r"\1<redacted>", url)


def github_error(title: str, message: str) -> None:
    escaped = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::error title={title}::{escaped}", file=sys.stderr)


def secret_from_env(name: str) -> str:
    """Read and validate a secret from the environment before it is used.

    GitHub masks secret values in tracebacks as `***`, so a pasted key with a
    trailing newline otherwise fails later as a vague `Invalid header value
    b'***'` error when urllib builds the Authorization header. Stripping common
    surrounding whitespace keeps pasted secrets usable, while rejecting any
    remaining control characters gives the user a clear, actionable message.
    """
    value = os.environ[name].strip()
    if not value:
        raise KeyError(name)
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise RuntimeError(
            f"{name} contains a line break or other control character. "
            "Edit the GitHub Actions secret and paste only the raw key/password value."
        )
    return value


def optional_secret_from_env(name: str) -> str | None:
    if name not in os.environ:
        return None
    return secret_from_env(name)


def validate_http_headers(headers: dict[str, str]) -> None:
    for name, value in headers.items():
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise RuntimeError(
                f"HTTP header {name} contains a line break or other control character. "
                "Check the related GitHub Actions secret and paste only the raw value."
            )


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    validate_http_headers(request_headers)
    request = urllib.request.Request(
        url,
        data=data,
        headers=request_headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {safe_url_for_error(url)} failed with {error.code}: {body}") from error


def extract_json_object(text: str) -> Any:
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = min([pos for pos in [text.find("{"), text.find("[")] if pos != -1], default=-1)
        if start == -1:
            raise
        end_char = "}" if text[start] == "{" else "]"
        end = text.rfind(end_char)
        return json.loads(text[start : end + 1])


def openai_text(model: str, instructions: str, user_content: str, api_key: str) -> str:
    payload = {
        "model": model,
        "instructions": instructions,
        "input": user_content,
        "temperature": 0.7,
    }
    response = post_json(
        OPENAI_RESPONSES_URL,
        payload,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    if response.get("output_text"):
        return response["output_text"]
    chunks: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                chunks.append(content["text"])
    return "\n".join(chunks)


def openai_image_metadata(model: str, image_path: Path, prompt_context: dict[str, Any], api_key: str) -> dict[str, Any]:
    media_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "model": model,
        "instructions": (
            "Generate Adobe Stock metadata for the actual image. Use the prompt, topic, and concept only as hints. "
            "Avoid brands, real persons, public figures, editorial framing, trademarks, and unverifiable claims. "
            "Return strict JSON with title, description, keywords array of 25-45 terms, category, and releases_needed."
        ),
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": json.dumps(prompt_context, ensure_ascii=False)},
                    {"type": "input_image", "image_url": f"data:{media_type};base64,{image_b64}"},
                ],
            }
        ],
        "temperature": 0.4,
    }
    response = post_json(
        OPENAI_RESPONSES_URL,
        payload,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    text = response.get("output_text", "")
    if not text:
        parts: list[str] = []
        for item in response.get("output", []):
            for content in item.get("content", []):
                if content.get("text"):
                    parts.append(content["text"])
        text = "\n".join(parts)
    return extract_json_object(text)


def research_concepts(config: dict[str, Any], paths: BatchPaths, api_key: str) -> list[dict[str, Any]]:
    output = paths.research / "concepts.json"
    if output.exists():
        return load_json(output)
    count = config["batch"]["concepts_per_day"]
    instructions = "You are a commercial stock image research strategist for Adobe Stock-safe AI image batches."
    request = f"""
Create {count} commercially useful, non-editorial stock image concepts for today's batch.
For each concept, perform trend-informed ideation, convert it into a stock-safe concept, and score it.
Hard exclusions: real persons, brands, logos, trademarks, copyrighted characters, public figures, news/editorial framing.
Return strict JSON array. Each item must include id, trend_signal, topic, stock_safe_concept, buyer_use_cases,
visual_elements, risk_notes, score_0_to_100, and score_reason.
""".strip()
    concepts = extract_json_object(openai_text(config["models"]["research_model"], instructions, request, api_key))
    write_json(output, concepts)
    return concepts


def prompt_packets(config: dict[str, Any], paths: BatchPaths, concepts: list[dict[str, Any]], api_key: str) -> list[dict[str, Any]]:
    output = paths.prompts / "prompt_packets.json"
    if output.exists():
        return load_json(output)
    image_model = config["models"]["image_model"]
    per_concept = config["batch"]["images_per_concept"]
    instructions = "You create production-ready AI image prompt packets for Adobe Stock-safe generated assets."
    request = f"""
Create {per_concept} image prompt packets per concept from this JSON:
{json.dumps(concepts, ensure_ascii=False)}

Each packet must choose the best aspect ratio for stock usefulness and include:
id, concept_id, topic, aspect_ratio, aspect_ratio_reason, positive_prompt, negative_prompt, technical_specs,
nano_banana_payload.
The nano_banana_payload should be a minimal Gemini generateContent REST object with only contents.parts.text.
Do not include model_hint, tools, or other unsupported API options.
The automation will rebuild the final request for {image_model} with responseFormat.image.aspectRatio and imageSize=4K.
Ensure prompts ban logos, brands, text, watermarks, real people, public figures,
editorial/news scenes, and recognizable protected designs.
Return strict JSON array only.
""".strip()
    packets = extract_json_object(openai_text(config["models"]["research_model"], instructions, request, api_key))
    write_json(output, packets)
    return packets


def build_gemini_prompt_text(packet: dict[str, Any]) -> str:
    positive_prompt = str(packet.get("positive_prompt") or "").strip()
    if not positive_prompt:
        raise RuntimeError(f"Packet {packet.get('id', '<unknown>')} is missing positive_prompt")

    prompt_parts = [positive_prompt]
    prompt_parts.append(
        "Create a native 4K PNG commercial stock image suitable for Adobe Stock submission. Do not upscale or resize; generate at 4K resolution directly."
    )
    aspect_ratio = str(packet.get("aspect_ratio") or "").strip()
    if aspect_ratio:
        prompt_parts.append(
            f"Compose the final stock image for a {aspect_ratio} aspect ratio. "
            f"Use framing, subject placement, and copy space appropriate for {aspect_ratio} stock usage."
        )

    technical_specs = packet.get("technical_specs")
    if technical_specs:
        prompt_parts.append(f"Technical specs: {technical_specs}")

    negative_prompt = str(packet.get("negative_prompt") or "").strip()
    stock_safety = (
        "Stock safety requirements: non-editorial commercial stock image; no real people or public figures; "
        "no brands, logos, trademarks, copyrighted characters, protected product designs, readable text, or watermarks."
    )
    prompt_parts.append(stock_safety)
    if negative_prompt:
        prompt_parts.append(f"Negative prompt: {negative_prompt}")
    return "\n\n".join(prompt_parts)


def build_gemini_payload(packet: dict[str, Any], image_size: str = DEFAULT_IMAGE_SIZE) -> dict[str, Any]:
    aspect_ratio = str(packet.get("aspect_ratio") or "1:1").strip()
    # Build the request ourselves instead of trusting model-generated
    # nano_banana_payload content. Gemini 3.1 Flash Image (Nano Banana 2)
    # supports native image sizing through generationConfig.responseFormat.image.
    return {
        "contents": [{"parts": [{"text": build_gemini_prompt_text(packet)}]}],
        "generationConfig": {
            "responseModalities": ["Image"],
            "responseFormat": {
                "image": {
                    "aspectRatio": aspect_ratio,
                    "imageSize": image_size,
                }
            },
        },
    }


def generate_images(config: dict[str, Any], paths: BatchPaths, packets: list[dict[str, Any]], api_key: str) -> list[Path]:
    generated: list[Path] = []
    model = config["models"]["image_model"]
    output_format, extension = image_output_settings(config)
    image_size = image_size_setting(config)
    if output_format != "PNG" or extension != ".png":
        raise RuntimeError("Image output must be configured as native PNG with a .png extension.")
    url = GEMINI_GENERATE_URL.format(model=model)
    for packet in packets:
        image_id = packet["id"]
        image_path = paths.images / f"{image_id}{extension}"
        context_path = paths.images / f"{image_id}.json"
        if image_path.exists():
            generated.append(image_path)
            continue
        payload = build_gemini_payload(packet, image_size)
        response = post_json(url, payload, headers={"x-goog-api-key": api_key})
        inline_image = first_inline_image(response)
        if not inline_image:
            write_json(paths.images / f"{image_id}.response.json", redact_binary(response))
            raise RuntimeError(f"No image bytes returned for packet {image_id}; saved redacted response for debugging")
        image_bytes, mime_type = inline_image
        image_info = write_direct_png(image_bytes, mime_type, image_path)
        write_json(
            context_path,
            {
                "packet": packet,
                "request_payload": payload,
                "image_output": image_info,
                "response_metadata": redact_binary(response),
            },
        )
        generated.append(image_path)
    return generated


def redact_binary(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: ("<base64 image omitted>" if key in {"data", "inlineData"} else redact_binary(val)) for key, val in value.items()}
    if isinstance(value, list):
        return [redact_binary(item) for item in value]
    return value


def first_inline_image(response: dict[str, Any]) -> tuple[bytes, str | None] | None:
    for candidate in response.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                mime_type = inline.get("mimeType") or inline.get("mime_type")
                return base64.b64decode(inline["data"]), mime_type
    return None


def send_email(subject: str, body: str) -> None:
    to_addr = optional_secret_from_env("STOCK_AUTOMATION_EMAIL_TO")
    from_addr = optional_secret_from_env("STOCK_AUTOMATION_EMAIL_FROM")
    host = optional_secret_from_env("STOCK_AUTOMATION_SMTP_HOST")
    username = optional_secret_from_env("STOCK_AUTOMATION_SMTP_USERNAME")
    password = optional_secret_from_env("STOCK_AUTOMATION_SMTP_PASSWORD")
    port = int((os.getenv("STOCK_AUTOMATION_SMTP_PORT") or "587").strip())
    if not all([to_addr, from_addr, host, username, password]):
        print(f"Email not sent; missing SMTP environment. Subject: {subject}")
        return
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = from_addr
    message["To"] = to_addr
    message.set_content(body)
    context = ssl.create_default_context()
    with smtplib.SMTP(host, port, timeout=60) as server:
        server.starttls(context=context)
        server.login(username, password)
        server.send_message(message)


def create_review_sheet(paths: BatchPaths, generated: list[Path]) -> None:
    review_file = paths.review / "image_review_request.md"
    lines = [
        "# Image review request",
        "",
        f"Generated images: {len(generated)}",
        "",
        "Review each image in `images/`. Move approved images into `approved/` to continue metadata/export processing.",
        "Rejected images can remain in `images/` or be deleted.",
        "",
        "## Files",
    ]
    lines.extend(f"- `{path.name}`" for path in generated)
    review_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def metadata_for_approved(config: dict[str, Any], paths: BatchPaths, packets: list[dict[str, Any]], api_key: str) -> list[dict[str, Any]]:
    packet_by_id = {packet["id"]: packet for packet in packets}
    rows: list[dict[str, Any]] = []
    for image_path in approved_image_paths(paths):
        image_id = image_path.stem
        metadata_path = paths.metadata / f"{image_id}.json"
        if metadata_path.exists():
            metadata = load_json(metadata_path)
        else:
            metadata = openai_image_metadata(
                config["models"]["metadata_model"],
                image_path,
                packet_by_id.get(image_id, {"image_id": image_id}),
                api_key,
            )
            write_json(metadata_path, metadata)
        rows.append({"filename": image_path.name, **metadata})
    return rows


def export_csv(paths: BatchPaths, rows: list[dict[str, Any]]) -> Path | None:
    if not rows:
        return None
    export_path = paths.exports / "adobe_stock_metadata.csv"
    with export_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "title", "description", "keywords", "category", "releases_needed"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "filename": row.get("filename", ""),
                    "title": row.get("title", ""),
                    "description": row.get("description", ""),
                    "keywords": ", ".join(row.get("keywords", [])) if isinstance(row.get("keywords"), list) else row.get("keywords", ""),
                    "category": row.get("category", ""),
                    "releases_needed": row.get("releases_needed", ""),
                }
            )
    return export_path


def process_approved_batch(config: dict[str, Any], paths: BatchPaths, batch_date: str, openai_key: str) -> None:
    prompts_path = paths.prompts / "prompt_packets.json"
    if not prompts_path.exists():
        return
    approved_images = approved_image_paths(paths)
    if not approved_images:
        return
    packets = load_json(prompts_path)
    rows = metadata_for_approved(config, paths, packets, openai_key)
    if rows:
        send_email(f"Metadata review ready: {batch_date}", f"Metadata generated for {len(rows)} approved images in `{paths.metadata}`.")
        export_path = export_csv(paths, rows)
        send_email(f"Final stock batch ready: {batch_date}", f"Export package is ready: `{export_path}`")


def process_all_approved_batches(config: dict[str, Any], openai_key: str) -> None:
    output_root = Path(config["batch"]["output_root"])
    if not output_root.exists():
        return
    for batch_dir in sorted(path for path in output_root.iterdir() if path.is_dir() and path.name != "approved"):
        paths = batch_paths(config["batch"]["output_root"], config["batch"]["approval_folder_name"], batch_dir.name)
        process_approved_batch(config, paths, batch_dir.name, openai_key)


def run(config_path: Path, batch_date: str) -> None:
    config = load_json(config_path)
    paths = batch_paths(config["batch"]["output_root"], config["batch"]["approval_folder_name"], batch_date)
    openai_key = secret_from_env("OPENAI_API_KEY")
    gemini_key = secret_from_env("GEMINI_API_KEY")

    concepts = research_concepts(config, paths, openai_key)
    packets = prompt_packets(config, paths, concepts, openai_key)
    generated = generate_images(config, paths, packets, gemini_key)
    create_review_sheet(paths, generated)
    send_email(
        f"Stock image review ready: {batch_date}",
        f"{len(generated)} images are ready. Review `{paths.images}` and move approved images into `{paths.approved}`.",
    )

    process_all_approved_batches(config, openai_key)
    if not approved_image_paths(paths):
        print(f"No approved images found in {paths.approved}; metadata/export will run after approvals are moved there.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the daily stock image generation automation.")
    parser.add_argument("--config", type=Path, default=Path("config/stock_image_automation.json"))
    parser.add_argument("--date", default=today_iso(), help="Batch date folder in YYYY-MM-DD format.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run(args.config, args.date)
    except KeyError as error:
        github_error("Missing environment variable", f"Required environment variable is missing: {error}")
        return 2
    except RuntimeError as error:
        github_error("Stock automation failed", str(error))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
