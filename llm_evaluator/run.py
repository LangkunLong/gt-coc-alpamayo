"""
CoC Reasoning Trace → Ontology Parser

Parses reasoning traces into a structured driving ontology using an LLM
with guaranteed structured output (Anthropic tool_use / OpenAI json_schema).

Usage:
    python run_pipeline.py \
        --input traces.csv \
        --output parsed_ontology.csv \
        --provider anthropic \
        --model claude-sonnet-4-20250514 \
        --concurrency 5

Supported providers: anthropic, openai
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

from schema import (
    SCHEMA,
    ALL_FIELDS,
    get_anthropic_tool_definition,
    get_openai_response_format,
)
from rag import (
    build_system_prompt,
    build_user_prompt,
    build_user_prompt_openai,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM Client with Structured Output
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Wrapper around LLM APIs that enforces structured output.

    - Anthropic: tool_use forces the model to return JSON matching the schema.
    - OpenAI: json_schema response_format constrains token generation.
    """

    def __init__(self, provider: str, model: str, api_key: str | None = None):
        self.provider = provider
        self.model = model
        self._async_client = None
        self._init_client(api_key)

    def _init_client(self, api_key: str | None):
        if self.provider == "anthropic":
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                raise ImportError("pip install anthropic")
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            self._async_client = AsyncAnthropic(api_key=key)

        elif self.provider == "openai":
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError("pip install openai")
            key = api_key or os.environ.get("OPENAI_API_KEY")
            self._async_client = AsyncOpenAI(api_key=key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def call(self, system_prompt: str, user_prompt: str) -> dict:
        """Async structured call. Returns parsed dict."""
        if self.provider == "anthropic":
            return await self._call_anthropic(system_prompt, user_prompt)
        else:
            return await self._call_openai(system_prompt, user_prompt)

    async def _call_anthropic(self, system_prompt: str, user_prompt: str) -> dict:
        tool = get_anthropic_tool_definition()
        response = await self._async_client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            tools=[tool],
            tool_choice={"type": "tool", "name": "submit_ontology_parse"},
            temperature=0.0,
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_ontology_parse":
                return block.input
        raise ValueError("No tool call found in response")

    async def _call_openai(self, system_prompt: str, user_prompt: str) -> dict:
        response = await self._async_client.chat.completions.create(
            model=self.model,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=get_openai_response_format(),
            temperature=0.0,
        )
        return json.loads(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_field(field: str, raw_value: str) -> tuple[str, list[str]]:
    """Validate comma-separated values against the closed vocabulary."""
    if field not in SCHEMA:
        return raw_value, []

    allowed = set(SCHEMA[field])
    values = [
        v.strip().lower().replace(" ", "_").replace("-", "_")
        for v in raw_value.split(",")
    ]

    cleaned, warnings = [], []
    for v in values:
        if v in allowed:
            cleaned.append(v)
        else:
            # Fuzzy: substring match
            match = next((a for a in allowed if v in a or a in v), None)
            if match:
                cleaned.append(match)
                warnings.append(f"{field}: '{v}' → fuzzy matched to '{match}'")
            else:
                cleaned.append(v)
                warnings.append(f"{field}: '{v}' NOT in vocabulary")

    return ",".join(cleaned) if cleaned else "none", warnings


def validate_row(row: dict) -> tuple[dict, list[str]]:
    """Validate all fields in a parsed row."""
    cleaned, all_warnings = {}, []
    for field in SCHEMA:
        raw = row.get(field, "none") or "none"
        value, warnings = validate_field(field, raw.strip())
        cleaned[field] = value
        all_warnings.extend(warnings)

    cleaned["confidence"] = row.get("confidence", "low")
    cleaned["confidence_notes"] = row.get("confidence_notes", "")

    if all_warnings:
        warn_str = "; ".join(all_warnings)
        existing = cleaned["confidence_notes"]
        cleaned["confidence_notes"] = (
            f"{existing}; VALIDATION: {warn_str}" if existing
            else f"VALIDATION: {warn_str}"
        )
        if cleaned["confidence"] == "high":
            cleaned["confidence"] = "medium"

    return cleaned, all_warnings


# ---------------------------------------------------------------------------
# Single Trace Processing
# ---------------------------------------------------------------------------

async def process_trace(
    client: LLMClient,
    system_prompt: str,
    clip_id: str,
    trace: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> list[dict]:
    """Process one trace with concurrency control and retries."""

    user_prompt = (
        build_user_prompt(clip_id, trace)
        if client.provider == "anthropic"
        else build_user_prompt_openai(clip_id, trace)
    )

    for attempt in range(1, max_retries + 1):
        try:
            async with semaphore:
                parsed = await client.call(system_prompt, user_prompt)

            rows = parsed.get("rows", [])
            if not rows:
                logger.warning(f"[{clip_id}] Empty rows, attempt {attempt}")
                continue

            validated = []
            for row in rows:
                cleaned, _ = validate_row(row)
                cleaned["clip_id"] = clip_id
                validated.append(cleaned)
            return validated

        except Exception as e:
            logger.warning(f"[{clip_id}] Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)

    logger.error(f"[{clip_id}] Failed after {max_retries} attempts")
    return [{
        **{f: "PARSE_ERROR" for f in ALL_FIELDS},
        "clip_id": clip_id,
        "confidence": "low",
        "confidence_notes": f"Failed after {max_retries} attempts",
    }]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

async def run_pipeline(args):
    """Read CSV → evaluate each trace → write output CSV."""

    client = LLMClient(args.provider, args.model, args.api_key)
    system_prompt = build_system_prompt()

    # Read input
    input_path = Path(args.input)
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for col in [args.id_column, args.trace_column]:
            if col not in reader.fieldnames:
                logger.error(f"Column '{col}' not found. Available: {reader.fieldnames}")
                sys.exit(1)
        input_rows = list(reader)

    logger.info(f"Loaded {len(input_rows)} traces from {input_path}")

    # Process all traces concurrently
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        process_trace(
            client, system_prompt,
            row[args.id_column], row[args.trace_column],
            semaphore, args.max_retries,
        )
        for row in input_rows
        if row[args.trace_column].strip()
    ]

    results = await asyncio.gather(*tasks)
    all_rows = [row for result in results for row in result]

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_FIELDS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({f: row.get(f, "none") for f in ALL_FIELDS})

    # Summary
    conf = {"high": 0, "medium": 0, "low": 0}
    errors = 0
    for row in all_rows:
        conf[row.get("confidence", "low")] = conf.get(row.get("confidence", "low"), 0) + 1
        if row.get("maneuver_lat") == "PARSE_ERROR":
            errors += 1

    logger.info(f"Done. {len(all_rows)} rows → {output_path}")
    logger.info(f"Confidence: {conf} | Errors: {errors}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Parse CoC traces into driving ontology.")
    p.add_argument("--input", "-i", required=True, help="Input CSV path.")
    p.add_argument("--output", "-o", default="parsed_ontology.csv", help="Output CSV path.")
    p.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    p.add_argument("--model", default="claude-sonnet-4-20250514")
    p.add_argument("--api-key", default=None, help="API key (overrides env var).")
    p.add_argument("--id-column", default="clip_id")
    p.add_argument("--trace-column", default="reasoning_trace")
    p.add_argument("--concurrency", type=int, default=5, help="Max parallel requests.")
    p.add_argument("--max-retries", type=int, default=3)
    asyncio.run(run_pipeline(p.parse_args()))


if __name__ == "__main__":
    main()