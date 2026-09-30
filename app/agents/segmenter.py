from __future__ import annotations
import os 
from dataclasses import dataclass 

from app.transcription.transcription import TranscriptResult,TranscriptSegment
from config import DEFAULT_MAX_CHUNK_TOKENS,TOKENS_PER_WORD_ESTIMATE

@dataclass 
class TranscriptChunk:
    index: int
    start_time: float
    end_time: float
    segments: list[TranscriptSegment] 
    estimated_tokens: int

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments)


def _estimate_tokens(text: str) -> int:
    return int(len(text.split()) * TOKENS_PER_WORD_ESTIMATE)


def chunk_transcript(
    result: TranscriptResult,
    max_chunk_tokens: int=DEFAULT_MAX_CHUNK_TOKENS,
    ) -> list[TranscriptChunk]:
    " "
    if not result.segments:
        return []
    total_tokens = sum(_estimate_tokens(s.text) for s in result.segments)

    if total_tokens <= max_chunk_tokens:
        return [
            TranscriptChunk(
                index=0,
                start_time=result.segments[0].start,
                end_time=result.segments[-1].end,
                segments=result.segments,
                estimated_tokens=total_tokens,
            )
        ]


# Split into the smallest number of chunks that fits the budget,
    # then divide the transcript as evenly as possible across them --
    # avoids the "3 full chunks + 1 tiny leftover chunk" shape you'd
    # get from greedily filling each chunk to the max.
    num_chunks = -(-total_tokens // max_chunk_tokens)  # ceil division
    target_tokens_per_chunk = total_tokens / num_chunks

    chunks: list[TranscriptChunk] = []
    current: list[TranscriptSegment] = []
    current_tokens = 0

    for seg in result.segments:
        current.append(seg)
        current_tokens += _estimate_tokens(seg.text)

        is_last_segment = seg is result.segments[-1]
        chunks_remaining = num_chunks - len(chunks)
        if (
            not is_last_segment
            and chunks_remaining > 1
            and current_tokens >= target_tokens_per_chunk
        ):
            chunks.append(
                TranscriptChunk(
                    index=len(chunks),
                    start_time=current[0].start,
                    end_time=current[-1].end,
                    segments=current,
                    estimated_tokens=current_tokens,
                )
            )
            current = []
            current_tokens = 0

    if current:
        chunks.append(
            TranscriptChunk(
                index=len(chunks),
                start_time=current[0].start,
                end_time=current[-1].end,
                segments=current,
                estimated_tokens=current_tokens,
            )
        )

    return chunks


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m notes_gen.agents.segmentor <youtube_url>")
        sys.exit(1)

    from app.transcription.translation import get_english_transcript

    transcript = get_english_transcript(sys.argv[1])
    chunks = chunk_transcript(transcript)

    print(f"\n{len(chunks)} chunk(s) (budget: {DEFAULT_MAX_CHUNK_TOKENS:,} tokens each):\n")
    for c in chunks:
        print(
            f"  [{c.start_time:7.1f}s - {c.end_time:7.1f}s] "
            f"~{c.estimated_tokens:,} tokens, {len(c.segments)} segments"
        )      