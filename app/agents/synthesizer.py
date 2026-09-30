"""Synthesizer agent (CrewAI). Stage 3 of the notes pipeline.

Runs exactly once over ALL chunk-level extractions produced by the
extractor: merges topics that got split across chunk boundaries,
removes redundant restatements, and arranges the material into one
coherent hierarchical outline (headings, subheadings, bullets) with
timestamp ranges so the outline maps back onto the video.

Pipeline position:
    transcription.get_transcript(url)
      -> translation.get_english_transcript()   (English TranscriptResult)
      -> segmenter.chunk_transcript()           (token-budgeted chunks)
      -> extractor.extract_chunks()             (one ChunkExtractionResult per chunk)
      -> synthesize_notes()  (THIS MODULE, single crew over everything)

This is the reasoning-heavy step -- only this agent sees every chunk's
findings together, so only it can merge across boundaries and
deduplicate globally. It runs on the same model as the extractor.

Model choice is architecture, not deployment config: the model name
lives HERE in code (SYNTHESIZER_MODEL), never in .env. Only the API
keys come from .env (NVIDIA_API_KEY, FALLBACK_NVIDIA_API_KEY).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from crewai import Agent, Crew, LLM, Task
from dotenv import load_dotenv

from app.agents.extractor import ChunkExtractionResult
from config import SYNTHESIZER_TEMPERATURE, SYNTHESIZER_MODEL

load_dotenv()

logger = logging.getLogger("synthesizer_agent")

# Same model as the extractor, routed through LiteLLM's nvidia_nim
# provider (CrewAI's LLM layer) to NVIDIA's OpenAI-compatible endpoint.


class SynthesisError(Exception):
    """Raised when the outline could not be synthesized on any key."""


def candidate_llms(temperature: float) -> list[tuple[str, LLM]]:
    """
    Builds one LLM instance per configured key, in rotation order:
    primary first, then fallback. Unset keys are skipped; an empty
    list means "no key configured" and the caller raises its own
    descriptive error.
    """
    candidates: list[tuple[str, LLM]] = []
    for label, variable in (
        ("primary", "NVIDIA_API_KEY"),
        ("fallback", "FALLBACK_NVIDIA_API_KEY"),
    ):
        key = os.environ.get(variable, "")
        if key:
            candidates.append(
                (label, LLM(model=SYNTHESIZER_MODEL, api_key=key, temperature=temperature))
            )
    return candidates


_TASK_DESCRIPTION = (
    "You are given JSON extracted from every chunk of one video, in "
    "chronological order. Each chunk reports its timestamps (start_time "
    "and end_time in seconds) and the topics found in it, each with "
    "key_points, definitions, examples and processes.\n\n"
    "EXTRACTIONS:\n"
    '"""\n{extractions_json}\n"""\n\n'
    "Write the final study outline for the entire video:\n"
    "- Merge topics that continue across chunk boundaries into one "
    "section; never present the same topic twice.\n"
    "- Remove redundant restatements of the same idea; keep the "
    "clearest wording, stated exactly once -- this applies within a "
    "section too, so a definition, example, or process step must never "
    "appear twice under the same heading.\n"
    "- Keep chronological order unless the material clearly groups "
    "better thematically.\n"
    "- Never invent content that is not present in the extractions.\n"
    "- Give every major section heading a timestamp range, formatted "
    "like [MM:SS - MM:SS], derived from the contributing chunks' "
    "start_time and end_time.\n\n"
    "Output plain Markdown only: one top-level heading (a descriptive "
    "title for the video), second-level headings per merged topic with "
    "its timestamp range, third-level subheadings where a topic has "
    "distinct sub-areas, and bullet lists for the points, definitions, "
    "examples and process steps. No preamble, no closing remarks."
)


def _build_agent(llm: Any) -> Agent:
    """Single synthesizer agent: sees everything, emits the final outline."""
    return Agent(
        role="Study notes synthesizer",
        goal=(
            "Turn the per-chunk extractions from an entire video into one "
            "coherent, deduplicated hierarchical study outline."
        ),
        backstory=(
            "You are the final agent in a multi-agent notes pipeline. "
            "Extractor agents mined every transcript chunk for topics, "
            "key points, definitions, examples and processes; only you "
            "see all of their findings at once, so only you can merge "
            "topics that straddle chunk boundaries and cut redundant "
            "restatements. Students will study directly from your "
            "outline, so it must be organized, complete and free of "
            "repetition."
        ),
        llm=llm,
        allow_delegation=False,
    )


def _crew_text(result: Any) -> str:
    """Unwraps a CrewAI kickoff result into its raw text."""
    raw = getattr(result, "raw", None)
    return raw if isinstance(raw, str) else str(result)


def _extractions_json(results: list[ChunkExtractionResult]) -> str:
    """Serializes every chunk extraction (with its time range) for the prompt."""
    return json.dumps(
        [
            {
                "chunk_index": r.chunk_index,
                "start_time": r.start_time,
                "end_time": r.end_time,
                "topics": [topic.model_dump() for topic in r.extraction.topics],
            }
            for r in results
        ],
        indent=2,
    )


def synthesize_notes(
    results: list[ChunkExtractionResult],
    llms: Optional[list[tuple[str, Any]]] = None,
) -> str:
    """
    Runs the synthesizer crew once over every chunk extraction and
    returns the merged hierarchical outline as Markdown. A failed call
    rotates to the fallback NVIDIA key before giving up.
    """
    if not results:
        raise ValueError("results is empty -- nothing to synthesize")

    prompt = _TASK_DESCRIPTION.format(extractions_json=_extractions_json(results))

    candidates = llms if llms is not None else candidate_llms(SYNTHESIZER_TEMPERATURE)
    if not candidates:
        raise SynthesisError(
            "No NVIDIA API key is set. Add NVIDIA_API_KEY (and optionally "
            "FALLBACK_NVIDIA_API_KEY) to your .env file."
        )

    last_error: Exception | None = None
    for label, llm in candidates:
        try:
            agent = _build_agent(llm)
            task = Task(
                description=prompt,
                expected_output=(
                    "A hierarchical Markdown outline: one top-level title "
                    "heading, second-level topic sections with timestamp "
                    "ranges in their headings, occasional third-level "
                    "subheadings, and bullet points covering all key "
                    "points, definitions, examples and step-by-step "
                    "processes. Plain Markdown text, nothing else."
                ),
                agent=agent,
            )
            crew = Crew(agents=[agent], tasks=[task])
            outline = _crew_text(crew.kickoff()).strip()
            logger.info("outline synthesized (%s key)", label)
            return outline
        except Exception as exc:
            last_error = exc
            logger.warning("synthesis with %s key failed: %s", label, exc)

    raise SynthesisError(
        f"synthesis failed on all available keys (last error: {last_error})"
    )


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python -m app.agents.synthesizer <youtube_url>")
        sys.exit(1)

    from app.agents.extractor import extract_chunks
    from app.agents.segmenter import chunk_transcript
    from app.transcription.translation import get_english_transcript

    transcript = get_english_transcript(sys.argv[1])
    results = extract_chunks(chunk_transcript(transcript))
    print(synthesize_notes(results))
