from __future__ import annotations

import logging
import os
import re
from typing import Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from config import (
    GROQ_TRANSLATOR_LLM_MODEL,
    SEGMENTS_PER_BATCH,
    WORDS_PER_WINDOW,
)

from .transcription import (
    TranscriptResult,
    TranscriptSegment,
    get_transcript,
)

load_dotenv()

logger = logging.getLogger("translation_service")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
FALLBACK_GROQ_API_KEY = os.environ.get("FALLBACK_GROQ_API_KEY", "")


class TranslationError(Exception):
    """Raised when a non-English transcript cannot be translated."""


_LINE_RE = re.compile(r"^(\d+)[.)]\s+(.+)$")


def is_english_language(language_code: Optional[str]) -> bool:
    """ """
    if not language_code:
        return False
    code = language_code.strip().lower()
    if code.startswith("asr-"):
        code = code[len("asr-"):]
    return code == "en" or code.startswith("en-") or code.startswith("en_")


def _clean_numbered_line(raw_line: str) -> str:
    """ """
    return raw_line.strip().strip("*#>`-").strip()


def _build_llm(api_key: str) -> ChatGroq:
    """Builds a LangChain ChatGroq client bound to one API key.

    max_retries=0 so HTTP 429 surfaces immediately and our own
    primary -> fallback rotation in _chat_text_with_fallback applies.
    """
    return ChatGroq(
        model=GROQ_TRANSLATOR_LLM_MODEL,
        temperature=0.1,
        groq_api_key=api_key,
        max_retries=0,
    )


def _is_fallbackable(exc: Exception) -> bool:
    """True for auth / rate-limit failures worth retrying with the fallback key."""
    status = getattr(exc, "status_code", None)
    if status in (401, 429):
        return True
    msg = str(exc).lower()
    return "429" in msg or "401" in msg or "rate_limit" in msg or "rate limit" in msg


def _extract_text(response) -> str:
    """Unwraps a LangChain AIMessage into plain text."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", block)))
            else:
                parts.append(str(getattr(block, "text", block)))
        return "\n".join(parts)
    return str(content)


def _chat_text_with_fallback(prompt: str) -> str:
    """
    Sends a translation prompt via LangChain ChatGroq with the primary API
    key first. On auth/rate-limit errors retries once with the fallback key.
    Mirrors the key rotation previously done with the raw Groq SDK.
    """
    keys = [(GROQ_API_KEY, "primary"), (FALLBACK_GROQ_API_KEY, "fallback")]

    for api_key, label in keys:
        if not api_key:
            continue
        try:
            llm = _build_llm(api_key)
            return _extract_text(llm.invoke(prompt))
        except Exception as exc:
            if _is_fallbackable(exc) and label == "primary" and FALLBACK_GROQ_API_KEY:
                logger.warning(
                    "translation LLM: %s key hit %s, falling back to fallback key",
                    label,
                    getattr(exc, "status_code", exc),
                )
                continue
            raise

    raise TranslationError("No valid GROQ API keys configured. Check your .env file.")


# Backward-compatible alias: previously returned a raw Groq completion object,
# now returns the plain text string via LangChain.
def _chat_completion_with_fallback(messages: list[dict]) -> str:
    prompt = "\n".join(m.get("content", "") for m in messages if m.get("content"))
    return _chat_text_with_fallback(prompt)


def _translate_numbered_lines(lines: list[str]) -> Optional[list[str]]:
    """
    Sends a batch of numbered transcript lines to the LLM and returns the
    translated lines in order. Returns None if the model's output cannot
    be mapped one-to-one back onto the input lines -- the caller then
    falls back to whole-text translation.
    """
    numbered = "\n".join(f"{i + 1}. {line}" for i, line in enumerate(lines))
    prompt = (
        "Translate every numbered line below into English.\n"
        "Rules:\n"
        "- Output the same number of lines, with the same numbers, in the same order.\n"
        "- Each output line must be exactly: <line number>. <English translation>\n"
        "- Do not merge, split, skip, or add lines. Do not add any commentary.\n\n"
        f"{numbered}"
    )

    content = _chat_text_with_fallback(prompt)

    translations: dict[int, str] = {}
    for raw_line in content.splitlines():
        line = _clean_numbered_line(raw_line)
        match = _LINE_RE.match(line)
        if match:
            number = int(match.group(1))
            text = match.group(2).strip()
            if number not in translations and text:
                translations[number] = text

    if len(translations) != len(lines):
        return None

    ordered: list[str] = []
    for i in range(1, len(lines) + 1):
        text = translations.get(i)
        if text is None:
            return None
        ordered.append(text)
    return ordered


def _translate_segments(
    result: TranscriptResult,
) -> Optional[list[TranscriptSegment]]:
    """
    Translates the transcript segment-by-segment in batches, keeping each
    segment's original start/end timestamps. Returns None (and logs why)
    if any batch cannot be aligned back onto its segments.
    """
    translated: list[TranscriptSegment] = []
    total = len(result.segments)

    for start in range(0, total, SEGMENTS_PER_BATCH):
        batch = result.segments[start : start + SEGMENTS_PER_BATCH]
        try:
            translated_lines = _translate_numbered_lines([s.text for s in batch])
        except Exception as exc:
            logger.warning("batch translation call failed: %s", exc)
            return None

        if translated_lines is None:
            logger.warning(
                "video_id=%s segments %d-%d: model output could not be aligned "
                "-- aborting segment-aligned translation",
                result.video_id,
                start + 1,
                start + len(batch),
            )
            return None

        for seg, text in zip(batch, translated_lines):
            translated.append(TranscriptSegment(start=seg.start, end=seg.end, text=text))

        logger.info(
            "video_id=%s translated segments %d/%d",
            result.video_id,
            min(start + SEGMENTS_PER_BATCH, total),
            total,
        )

    return translated


def _translate_text_windows(text: str) -> str:
    """
    Fallback path: translates the raw text in word-counted windows,
    without segment alignment.
    """
    words = text.split()
    if not words:
        raise TranslationError("transcript text is empty -- nothing to translate")

    windows = [
        " ".join(words[i : i + WORDS_PER_WINDOW])
        for i in range(0, len(words), WORDS_PER_WINDOW)
    ]

    parts: list[str] = []
    for j, window in enumerate(windows):
        content = _chat_text_with_fallback(
            "Translate the following transcript text into English. "
            "Output only the translation, with no commentary:\n\n" + window
        ).strip()
        if not content:
            raise TranslationError(
                f"LLM returned an empty translation for window {j + 1}/{len(windows)}"
            )
        parts.append(content)
        logger.info("translated text window %d/%d", j + 1, len(windows))

    return " ".join(parts)


def ensure_english(result: TranscriptResult) -> TranscriptResult:
    """
    Returns an English TranscriptResult for the given transcript.

    English input passes through unchanged at zero cost. Non-English
    input is translated with the Groq LLM via LangChain; timestamps are
    preserved by translating segment-by-segment, falling back to
    whole-text translation if the model's output cannot be aligned.
    """
    if is_english_language(result.detected_language):
        logger.info(
            "video_id=%s transcript already in English (%s) -- no translation needed",
            result.video_id,
            result.detected_language,
        )
        return result

    language = result.detected_language or "unknown"
    logger.info(
        "video_id=%s transcript is %r -- translating to English via %s (langchain-groq)",
        result.video_id,
        language,
        GROQ_TRANSLATOR_LLM_MODEL,
    )

    if not GROQ_API_KEY and not FALLBACK_GROQ_API_KEY:
        raise TranslationError(
            "No GROQ API key is set -- cannot translate a non-English transcript. "
            "Add GROQ_API_KEY (and optionally FALLBACK_GROQ_API_KEY) to your .env file."
        )
    if not result.text.strip():
        raise TranslationError("transcript text is empty -- nothing to translate")

    if result.segments:
        translated_segments = _translate_segments(result)
        if translated_segments is not None:
            return TranscriptResult(
                text=" ".join(s.text for s in translated_segments).strip(),
                segments=translated_segments,
                source=result.source,
                detected_language="en",
                video_id=result.video_id,
                was_translated=True,
            )
        logger.warning(
            "video_id=%s falling back to whole-text translation "
            "(segment texts will remain in the original language)",
            result.video_id,
        )

    try:
        translated_text = _translate_text_windows(result.text)
    except TranslationError:
        raise
    except Exception as exc:
        raise TranslationError(f"LLM translation failed: {exc}") from exc

    return TranscriptResult(
        text=translated_text,
        segments=result.segments,
        source=result.source,
        detected_language="en",
        video_id=result.video_id,
        was_translated=True,
    )


def get_english_transcript(url: str) -> TranscriptResult:
    """
    Full pipeline entry point for the rest of the app: runs the
    transcription fallback chain (captions -> TranscriptAPI -> Whisper)
    and then guarantees the returned transcript is English.
    """
    result = get_transcript(url)
    return ensure_english(result)
