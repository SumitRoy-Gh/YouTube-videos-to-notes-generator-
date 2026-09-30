"""Extractor agent (CrewAI). Runs once per TranscriptChunk.

Reads the chunk's raw text and pulls out what is actually worth noting:
key points, definitions, examples, and step-by-step processes, organized
by topic. The chunk's timestamp range is attached by this code; the model
is never asked to compute timestamps.

Pipeline position:
    transcription.get_transcript(url)
      -> translation.get_english_transcript()   (English TranscriptResult)
      -> segmenter.chunk_transcript()           (token-budgeted chunks)
      -> extract_chunks()  (THIS MODULE, one crew per chunk, in parallel)
      -> synthesizer (merges every ChunkExtractionResult into final notes)

One crew per chunk: a single Agent + Task whose response is parsed and
validated against the ChunkExtraction pydantic schema. A malformed
response earns one stricter repair attempt on the same key; a transport
or auth failure rotates straight to the next key (a second key only
plausibly fixes transient/auth failures -- same convention as the
translation stage).

Model choice is architecture, not deployment config: the model name
lives HERE in code (EXTRACTOR_MODEL), never in .env. Only the API keys
come from .env (NVIDIA_API_KEY, FALLBACK_NVIDIA_API_KEY).
"""

from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional

from crewai import Agent, Crew, LLM, Task
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from app.agents.segmenter import TranscriptChunk
from config import (
    EXTRACTOR_TEMPERATURE,
    MAX_PARALLEL_CHUNKS,
    MAX_REPAIR_ATTEMPTS,
    EXTRACTOR_MODEL
)

load_dotenv()

logger = logging.getLogger("extractor_agent")

# The model both agents share, routed through LiteLLM's nvidia_nim
# provider (CrewAI's LLM layer) to NVIDIA's OpenAI-compatible endpoint.
# Extraction is structural work, not deep reasoning, but per the pipeline
# decision the extractor uses the same model as the synthesizer.


class ExtractionError(Exception):
    """Raised when a chunk could not be extracted after all keys/retries."""


# ---------------------------------------------------------------------------
# Output schema: everything the crew returns is validated against this.
# Missing keys default to [] so sparse chunks still validate; a completely
# wrong shape raises ValidationError, which the retry loop understands.
# ---------------------------------------------------------------------------


class Definition(BaseModel):
    term: str
    explanation: str


class Process(BaseModel):
    name: str
    steps: list[str] = Field(default_factory=list)


class TopicExtraction(BaseModel):
    title: str = Field(description="Short, specific title for this topic.")
    key_points: list[str] = Field(default_factory=list)
    definitions: list[Definition] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    processes: list[Process] = Field(default_factory=list)


class ChunkExtraction(BaseModel):
    topics: list[TopicExtraction]


@dataclass
class ChunkExtractionResult:
    """One chunk's validated extraction plus its identity and time range."""

    chunk_index: int
    start_time: float
    end_time: float
    extraction: ChunkExtraction


# ---------------------------------------------------------------------------
# LLM candidates: one CrewAI LLM per configured key, primary first.
# ---------------------------------------------------------------------------


def candidate_llms(temperature: float) -> list[tuple[str, LLM]]:
    """
    Builds one LLM instance per configured key, in rotation order.
    Unset keys are skipped; an empty list means "no key configured"
    and callers raise their own descriptive error.
    """
    candidates: list[tuple[str, LLM]] = []
    for label, variable in (
        ("primary", "NVIDIA_API_KEY"),
        ("fallback", "FALLBACK_NVIDIA_API_KEY"),
    ):
        key = os.environ.get(variable, "")
        if key:
            candidates.append(
                (label, LLM(model=EXTRACTOR_MODEL, api_key=key, temperature=temperature))
            )
    return candidates


_JSON_SCHEMA_HINT = (
    "Respond with ONLY a JSON object of this exact shape, no commentary, "
    "no markdown code fences:\n"
    "{\n"
    '  "topics": [\n'
    "    {\n"
    '      "title": "short specific topic title",\n'
    '      "key_points": ["...", "..."],\n'
    '      "definitions": [{"term": "...", "explanation": "..."}],\n'
    '      "examples": ["...", "..."],\n'
    '      "processes": [{"name": "...", "steps": ["...", "..."]}]\n'
    "    }\n"
    "  ]\n"
    "}"
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _build_task_description(chunk_text: str, repair: bool) -> str:
    """Prompt for the crew: extract substance, skip filler, obey the schema."""
    prefix = (
        "Your previous response was not valid JSON matching the required "
        "shape. Re-read the instructions carefully and respond again, "
        "following the schema exactly.\n\n"
        if repair
        else ""
    )
    return (
        f"{prefix}"
        "Identify the distinct topics covered in the transcript below, and "
        "for each one extract its key points, any definitions given, any "
        "examples used, and any step-by-step processes described. Skip "
        "filler, tangents, and restatements -- extract substance only. "
        "Write definitions and explanations concisely, in your own words "
        "(one or two sentences each), never as verbatim transcript quotes. "
        "List each distinct key point, definition, example, or step only "
        "once per topic. If "
        "the transcript contains nothing substantive, return an empty "
        "topics list.\n\n"
        f"{_JSON_SCHEMA_HINT}\n\n"
        f"Transcript:\n{chunk_text}"
    )


def _build_agent(llm: Any) -> Agent:
    """Single extractor agent: sees one chunk, emits machine-grade JSON."""
    return Agent(
        role="Video transcript extractor",
        goal=(
            "Extract the topics, key points, definitions, examples and "
            "step-by-step processes actually present in one transcript "
            "chunk -- completely and precisely, never inventing content."
        ),
        backstory=(
            "You are one extractor crewmate in a multi-agent notes "
            "pipeline that turns YouTube videos into study notes. You "
            "see exactly one chunk of the transcript; other extractor "
            "agents handle the remaining chunks in parallel, and a "
            "synthesizer agent later merges everyone's findings into "
            "one outline. Your output is consumed as JSON by machines, "
            "so you respond with schema-exact JSON and nothing else."
        ),
        llm=llm,
        allow_delegation=False,
    )


def _crew_text(result: Any) -> str:
    """Unwraps a CrewAI kickoff result into its raw text."""
    raw = getattr(result, "raw", None)
    return raw if isinstance(raw, str) else str(result)


def _parse_extraction(raw: str) -> ChunkExtraction:
    """
    Strips accidental markdown code fences, then parses and validates
    against the ChunkExtraction schema. Raises json.JSONDecodeError or
    pydantic.ValidationError, which the retry loop understands.
    """
    match = _FENCE_RE.search(raw)
    cleaned = match.group(1).strip() if match else raw.strip()
    return ChunkExtraction.model_validate(json.loads(cleaned))


def _extract_once(chunk: TranscriptChunk, llm: Any, repair: bool) -> ChunkExtraction:
    """Runs one single-agent crew over the chunk and validates its reply."""
    agent = _build_agent(llm)
    task = Task(
        description=_build_task_description(chunk.text, repair),
        expected_output=(
            "A single JSON object with one key, topics: a list of objects "
            "each with title, key_points, definitions, examples and "
            "processes. No markdown fences, no text outside the JSON."
        ),
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task])
    return _parse_extraction(_crew_text(crew.kickoff()))


def extract_chunk(
    chunk: TranscriptChunk,
    llms: Optional[list[tuple[str, Any]]] = None,
) -> ChunkExtractionResult:
    """
    Runs the extractor crew over one chunk and returns its extraction
    with the chunk's identity and timestamp range attached. Invalid
    JSON gets up to MAX_REPAIR_ATTEMPTS stricter retries per key;
    transport errors rotate immediately to the next key.
    """
    candidates = llms if llms is not None else candidate_llms(EXTRACTOR_TEMPERATURE)
    if not candidates:
        raise ExtractionError(
            "No NVIDIA API key is set. Add NVIDIA_API_KEY (and optionally "
            "FALLBACK_NVIDIA_API_KEY) to your .env file."
        )

    last_error: Exception | None = None
    for label, llm in candidates:
        for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
            try:
                extraction = _extract_once(chunk, llm, repair=attempt > 0)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                logger.warning(
                    "chunk %d: %s key returned invalid JSON (attempt %d/%d): %s",
                    chunk.index,
                    label,
                    attempt + 1,
                    MAX_REPAIR_ATTEMPTS + 1,
                    exc,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "chunk %d: %s key call failed: %s", chunk.index, label, exc
                )
                break
            else:
                logger.info(
                    "chunk %d: %d topic(s) extracted (%s key)",
                    chunk.index,
                    len(extraction.topics),
                    label,
                )
                return ChunkExtractionResult(
                    chunk_index=chunk.index,
                    start_time=chunk.start_time,
                    end_time=chunk.end_time,
                    extraction=extraction,
                )
        logger.warning("chunk %d: %s key exhausted, trying next key", chunk.index, label)

    raise ExtractionError(
        f"chunk {chunk.index}: extraction failed on all available keys "
        f"(last error: {last_error})"
    )


def extract_chunks(
    chunks: list[TranscriptChunk],
    max_workers: int = MAX_PARALLEL_CHUNKS,
    llms: Optional[list[tuple[str, Any]]] = None,
) -> list[ChunkExtractionResult]:
    """
    Extracts every chunk in parallel (independent crews on a thread
    pool), returning results in chunk order. Raises ExtractionError
    at the first chunk that fails on every key: a partial notes
    document is worse than a clear failure, since the synthesizer
    downstream assumes complete coverage of the video.
    """
    if not chunks:
        return []
    if llms is None and not candidate_llms(EXTRACTOR_TEMPERATURE):
        raise ExtractionError(
            "No NVIDIA API key is set. Add NVIDIA_API_KEY (and optionally "
            "FALLBACK_NVIDIA_API_KEY) to your .env file."
        )
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        return list(pool.map(lambda c: extract_chunk(c, llms), chunks))


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python -m app.agents.extractor <youtube_url>")
        sys.exit(1)

    from app.agents.segmenter import chunk_transcript
    from app.transcription.translation import get_english_transcript

    transcript = get_english_transcript(sys.argv[1])
    results = extract_chunks(chunk_transcript(transcript))

    for r in results:
        print(f"\n=== Chunk {r.chunk_index} [{r.start_time:.0f}s-{r.end_time:.0f}s] ===")
        for topic in r.extraction.topics:
            print(f"\n  # {topic.title}")
            for point in topic.key_points:
                print(f"    - {point}")
            for d in topic.definitions:
                print(f"    * {d.term}: {d.explanation}")
            for ex in topic.examples:
                print(f"    e.g. {ex}")
            for p in topic.processes:
                print(f"    steps [{p.name}]: " + " -> ".join(p.steps))
