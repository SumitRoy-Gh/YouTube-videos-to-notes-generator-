"""YouTube transcription service with a 3-layer fallback chain.

How this script works (big picture):
    The single public entry point is get_transcript(url). Give it any
    YouTube URL and it returns a TranscriptResult. Internally it tries
    three layers in order -- cheapest/fastest first -- and returns as
    soon as one layer succeeds:

    Layer 1 -- YouTube captions (FREE): _try_captions()
        Reads the video's existing caption tracks via youtube-transcript-api.
        Prefers a native English track; otherwise asks YouTube's own
        translation layer to translate the available track to English.
        No audio is downloaded, no paid API is touched.

    Layer 2 -- TranscriptAPI (PAID proxy): _try_transcriptapi()
        Calls the third-party TranscriptAPI service, which can fetch
        captions even when direct YouTube access is blocked (IP blocks,
        PoToken requirements, etc.). Transient failures (HTTP 408/429/503
        or network errors) are retried with exponential backoff. Note this
        layer may return NON-English text -- English is guaranteed only by
        the downstream translation stage.

    Layer 3 -- Groq Whisper (LAST RESORT, paid inference): _try_whisper()
        Downloads the audio with yt-dlp (16 kHz mono mp3), splits it into
        <= MAX_CHUNK_SIZE_MB pieces with ffmpeg, and sends each chunk to
        Groq's Whisper *translations* endpoint, which always outputs
        English. Primary/fallback API keys are tried per chunk. Any
        failure here raises TranscriptionError -- there is nothing left
        to fall back to.

    Every layer returns None (layers 1-2) on failure so the caller falls
    through; only layer 3 raises. Use TranscriptSource on the result to
    see which layer produced the transcript.

Configuration lives in config.py (TranscriptAPI key/URL/timeouts, Groq
model name, chunk-size and duration limits) plus GROQ_API_KEY /
FALLBACK_GROQ_API_KEY env vars loaded from .env.
"""

from __future__ import annotations

import logging
from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled, VideoUnavailable, NoTranscriptFound
from dotenv import load_dotenv
import os 
import re
import math
import requests
import time
import tempfile
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator,Optional
from enum import Enum
load_dotenv()

logger=logging.getLogger("transcript_service")

#configs

from config import TRANSCRIPTAPI_KEY,TRANSCRIPTAPI_TIMEOUT,TRANSCRIPTAPI_MAX_RETRIES,TRANSCRIPTAPI_URL,GROQ_WHISPER_MODEL,MAX_CHUNK_SIZE_MB,MAX_AUDIO_SECONDS,_RETRYABLE_STATUSES

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
FALLBACK_GROQ_API_KEY = os.environ.get("FALLBACK_GROQ_API_KEY", "")

class TranscriptSource(str, Enum):
    """Records which pipeline layer produced a transcript.

    CAPTIONS_NATIVE_EN: an English caption track straight from YouTube.
    CAPTIONS_TRANSLATED: a non-English caption track translated to English
        by YouTube's own translation layer (still no audio download).
    TRANSCRIPT_API: fetched via the paid TranscriptAPI proxy service;
        text may be non-English.
    WHISPER_TRANSLATED: audio transcribed via Groq Whisper's translations
        endpoint; output is English by construction.
    """

    CAPTIONS_NATIVE_EN = "captions_native_en"
    CAPTIONS_TRANSLATED = "captions_translated"
    TRANSCRIPT_API = "transcript_api"
    WHISPER_TRANSLATED = "whisper_translated"

@dataclass
class TranscriptSegment:
    """One timestamped piece of a transcript.

    Attributes:
        text: the spoken words for this slice of the video.
        start: when the slice begins, in seconds from video start.
        end: when the slice ends, in seconds from video start.
    """

    text: str
    start: float
    end: float

@dataclass 
class TranscriptResult:
    """The final output of the transcription pipeline.

    Attributes:
        text: the full transcript as a single string (segments joined
            with spaces).
        source: which TranscriptSource layer produced this result.
        segments: timestamped TranscriptSegment pieces in video order.
        detected_language: language code reported by the source layer
            ("en" for captions/Whisper paths; whatever TranscriptAPI
            reported for that path).
        video_id: the 11-character YouTube video id.
        was_translated: reserved flag for the downstream translation
            stage; always False at this point in the pipeline.
    """

    text: str
    source: TranscriptSource
    segments: list[TranscriptSegment]
    detected_language: Optional[str]
    video_id: str
    was_translated: bool = False

class TranscriptionError(Exception):
    """Raised when every fallback in the chain has been exhausted.

    Only the last-resort Whisper layer raises this; layers 1-2 return
    None on failure so get_transcript() can fall through to the next
    layer. Catching this error therefore means no transcript could be
    obtained from ANY layer.
    """  

#url parsing
# These regexes cover the four common YouTube URL shapes. Each one
# captures the 11-character video id; extract_video_url() tries them
# in order and raises ValueError if none match.

_YT_ID_PATTERNS = (
    re.compile(r"(?:youtu\.be/)([A-Za-z0-9_-]{11})"),
    re.compile(r"(?:v=)([A-Za-z0-9_-]{11})"),
    re.compile(r"(?:youtube\.com/embed/)([A-Za-z0-9_-]{11})"),
    re.compile(r"(?:youtube\.com/shorts/)([A-Za-z0-9_-]{11})"),
)

def extract_video_url(url: str) -> str:
    """Pulls the 11-character video id out of any supported YouTube URL.

    Handles watch (v=...), youtu.be short, /embed/ and /shorts/ links;
    extra query params (e.g. ?si=...) are ignored.

    Args:
        url: the full YouTube URL pasted by the user.

    Returns:
        The 11-character video id (letters, digits, '-' and '_').

    Raises:
        ValueError: if the string is not a recognisable YouTube URL.
    """
    for pattern in _YT_ID_PATTERNS:
        match = pattern.search(url)
        if match:
            return match.group(1)
    raise ValueError(f"Invalid Youtube URL{url!r}")

#caption based transcript retrieval
# LAYER 1 of the fallback chain: free YouTube captions. Nothing is
# downloaded and no paid service is called here.

def _try_captions(video_id: str) -> Optional[TranscriptResult]:
    """Attempts a transcript from the video's existing YouTube captions.

    Strategy, in order:
      1. List the video's caption tracks. Known-absent cases
         (TranscriptsDisabled / VideoUnavailable / NoTranscriptFound)
         and unexpected listing errors (IP blocks, PoTokenRequired --
         exactly what the paid layer 2 exists to absorb) give up
         immediately by returning None.
      2. Prefer a native/auto-generated English track
         (CAPTIONS_NATIVE_EN).
      3. Otherwise take the first available track and ask YouTube's own
         translation layer for English (CAPTIONS_TRANSLATED). If the
         only track is non-translatable, give up and return None.
      4. Fetch the chosen track, convert snippets to TranscriptSegments,
         and join them into the full text. Empty text counts as failure.

    Args:
        video_id: the 11-character YouTube video id.

    Returns:
        A TranscriptResult on success, or None on ANY failure so the
        caller (get_transcript) falls through to layer 2.
    """
    try:
        transcript_list = YouTubeTranscriptApi().list(video_id)
    except (TranscriptsDisabled, VideoUnavailable, NoTranscriptFound):
        logger.info("video_id=%s no caption tracks available", video_id)
        return None
    except Exception:
        # Includes IpBlocked / RequestBlocked / PoTokenRequired -- exactly
        # the failures the paid TranscriptAPI layer exists to absorb.
        logger.exception("video_id=%s unexpected error listing transcripts", video_id)
        return None

    # Prefer a manually-created or auto-generated English track if one exists.
    try:
        transcript = transcript_list.find_transcript(["en", "en-US", "en-GB"])
        source = TranscriptSource.CAPTIONS_NATIVE_EN
        detected_language = "en"
    except NoTranscriptFound:
        # No English track. Grab whatever track exists and ask YouTube's
        # own translation layer for English -- this is still just an API
        # call against YouTube, no audio download, no Whisper.
        try:
            available = next(iter(transcript_list))
        except StopIteration:
            return None

        if not available.is_translatable:
            logger.info(
                "video_id=%s only non-English, non-translatable track found "
                "(lang=%s) -- falling through",
                video_id,
                available.language_code,
            )
            return None

        try:
            transcript = available.translate("en")
            source = TranscriptSource.CAPTIONS_TRANSLATED
            # The text is English after YouTube's translation layer.
            detected_language = "en"
            logger.info(
                "video_id=%s using YouTube's own translation of the %s caption track",
                video_id,
                available.language_code,
            )
        except Exception:
            logger.exception(
                "video_id=%s translation of caption track failed", video_id
            )
            return None

    try:
        raw = transcript.fetch()
    except Exception:
        logger.exception("video_id=%s fetching caption track failed", video_id)
        return None

    # v1.x of youtube-transcript-api returns dataclass snippets with
    # .text/.start/.duration attributes.
    segments = [
        TranscriptSegment(start=item.start, end=item.start + item.duration, text=item.text)
        for item in raw
    ]
    full_text = " ".join(s.text for s in segments).strip()

    if not full_text:
        return None

    return TranscriptResult(
        text=full_text,
        segments=segments,
        source=source,
        detected_language=detected_language,
        video_id=video_id,
    )    

#transcriptapi
# LAYER 2 of the fallback chain: the paid TranscriptAPI proxy service.
# Helpers below deal with its error bodies, retry timing, and payload
# parsing; _try_transcriptapi() ties them together.

def _error_detail(response: requests.Response) -> str:
    """Extracts a human-readable message from a TranscriptAPI error body."""
    try:
        payload = response.json()
    except ValueError:
        return (response.text or "")[:200]
    detail = payload.get("detail")
    if isinstance(detail, dict):
        return str(detail.get("message") or detail)
    return str(detail or payload)[:200]


def _retry_delay_seconds(response: Optional[requests.Response], attempt: int) -> float:
    """Respects the Retry-After header when present, else exponential backoff."""
    if response is not None:
        header = response.headers.get("Retry-After")
        if header:
            try:
                return max(0.0, float(header))
            except ValueError:
                pass
    return float(2 ** attempt)


def _parse_transcriptapi_payload(
    video_id: str, payload: dict
) -> Optional[TranscriptResult]:
    """Converts a TranscriptAPI JSON body into a TranscriptResult.

    Expects payload["transcript"] to be a list of {"text", "start",
    "duration"} dicts. Blank/ malformed entries are skipped; if nothing
    usable remains, None is returned so the caller falls through to
    layer 3. Unlike the captions layer, the text here may be non-English
    -- the payload's own "language" value is passed through untouched.

    Args:
        video_id: the 11-character YouTube video id (for logging).
        payload: the decoded JSON body from a HTTP-200 TranscriptAPI call.

    Returns:
        A TranscriptResult with source TRANSCRIPT_API, or None when the
        payload holds no usable transcript segments.
    """
    raw = payload.get("transcript")
    if not isinstance(raw, list) or not raw:
        logger.warning("video_id=%s TranscriptAPI returned no transcript segments", video_id)
        return None

    segments: list[TranscriptSegment] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = (item.get("text") or "").strip()
        if not text:
            continue
        start = float(item.get("start") or 0.0)
        duration = float(item.get("duration") or 0.0)
        segments.append(TranscriptSegment(start=start, end=start + duration, text=text))

    if not segments:
        logger.warning("video_id=%s TranscriptAPI transcript had no usable text", video_id)
        return None

    return TranscriptResult(
        text=" ".join(s.text for s in segments),
        segments=segments,
        source=TranscriptSource.TRANSCRIPT_API,
        detected_language=payload.get("language"),
        video_id=video_id,
    )


def _try_transcriptapi(video_id: str) -> Optional[TranscriptResult]:
    """
    Fetches the transcript via TranscriptAPI. Returns None on any
    failure so the caller falls through to the Whisper layer; details
    are logged. Retries transient failures (network errors, 408/429/503)
    with exponential backoff, honouring the Retry-After header.
    """
    if not TRANSCRIPTAPI_KEY:
        logger.warning(
            "Youtube_to_Transcript_API_KEY is not set -- skipping TranscriptAPI layer"
        )
        return None

    headers = {"Authorization": f"Bearer {TRANSCRIPTAPI_KEY}"}
    params = {"video_url": video_id, "format": "json", "include_timestamp": "true"}

    max_attempts = TRANSCRIPTAPI_MAX_RETRIES + 1
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(
                TRANSCRIPTAPI_URL,
                params=params,
                headers=headers,
                timeout=TRANSCRIPTAPI_TIMEOUT,
            )
        except requests.RequestException as exc:
            if attempt < max_attempts:
                delay = _retry_delay_seconds(None, attempt)
                logger.warning(
                    "video_id=%s TranscriptAPI network error (attempt %d/%d): %s "
                    "-- retrying in %.0fs",
                    video_id, attempt, max_attempts, exc, delay,
                )
                time.sleep(delay)
                continue
            logger.warning(
                "video_id=%s TranscriptAPI network error, retries exhausted: %s",
                video_id, exc,
            )
            return None

        if response.status_code == 200:
            try:
                payload = response.json()
            except ValueError:
                logger.warning(
                    "video_id=%s TranscriptAPI returned invalid JSON", video_id
                )
                return None
            return _parse_transcriptapi_payload(video_id, payload)

        if response.status_code in _RETRYABLE_STATUSES and attempt < max_attempts:
            delay = _retry_delay_seconds(response, attempt)
            logger.warning(
                "video_id=%s TranscriptAPI HTTP %d (attempt %d/%d) "
                "-- retrying in %.0fs",
                video_id, response.status_code, attempt, max_attempts, delay,
            )
            time.sleep(delay)
            continue

        # Non-retryable (400/401/402/404/422) or retries exhausted:
        # log loudly, then fall through to Whisper. 402 = out of credits,
        # 401 = bad key, 404 = no captions anywhere -- the last one is
        # exactly what the Whisper layer exists for.
        logger.warning(
            "video_id=%s TranscriptAPI failed: HTTP %d -- %s",
            video_id, response.status_code, _error_detail(response),
        )
        return None

    return None


# --------------------------------------------------------------------------
# Layer 3: Groq Whisper fallback (audio download + cloud inference)
# --------------------------------------------------------------------------

@contextmanager
def _downloaded_audio(video_id: str) -> Iterator[str]:
    """
    Downloads the audio-only stream as a compressed mp3 to a temp directory.
    Yields the path to the downloaded mp3 file.
    Cleanup is guaranteed by TemporaryDirectory on context exit.
    """
    import yt_dlp

    with tempfile.TemporaryDirectory(prefix="yt_audio_") as tmp_dir:
        out_template = os.path.join(tmp_dir, "%(id)s.%(ext)s")
        ydl_opts = {
            "format": "worstaudio/bestaudio",
            "outtmpl": out_template,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "5",
                }
            ],
            # Downsample to 16kHz mono -- Whisper's native format.
            # Cuts file size dramatically for long videos.
            "postprocessor_args": ["-ar", "16000", "-ac", "1"],
        }

        url = f"https://www.youtube.com/watch?v={video_id}"
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
        except Exception as exc:
            raise TranscriptionError(f"Audio download failed for {video_id}: {exc}") from exc

        duration = info.get("duration") or 0
        if duration and duration > MAX_AUDIO_SECONDS:
            raise TranscriptionError(
                f"Video duration {duration}s exceeds MAX_AUDIO_SECONDS="
                f"{MAX_AUDIO_SECONDS}s limit for this deployment."
            )

        # Find the mp3 file yt-dlp produced
        mp3_path = os.path.join(tmp_dir, f"{info['id']}.mp3")
        if not os.path.exists(mp3_path):
            candidates = [f for f in os.listdir(tmp_dir) if f.endswith(".mp3")]
            if not candidates:
                raise TranscriptionError("Audio postprocessing did not produce an .mp3 file.")
            mp3_path = os.path.join(tmp_dir, candidates[0])

        yield mp3_path


def _ffprobe_duration_seconds(audio_path: str) -> float:
    """Returns the audio file's duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def _split_audio_into_chunks(audio_path: str, tmp_dir: str) -> list[str]:
    """
    Splits an audio file into chunks of approximately MAX_CHUNK_SIZE_MB
    each. Returns a list of file paths to the chunk files.

    Chunking serves two constraints at once: Groq's 25 MB upload cap on
    the free tier, and the small ephemeral disks of free-tier deployments
    (Render etc.) when long videos would otherwise produce very large
    audio files.

    Splitting is done by streaming through ffmpeg (never loading the
    file into RAM), estimating how many seconds of audio fit into
    MAX_CHUNK_SIZE_MB based on the file's actual bitrate.
    """
    file_size_bytes = os.path.getsize(audio_path)

    if file_size_bytes <= MAX_CHUNK_SIZE_MB * 1024 * 1024:
        # File is already small enough, no splitting needed
        logger.info(
            "audio file %.1f MB, under %d MB limit -- no chunking needed",
            file_size_bytes / (1024 * 1024),
            MAX_CHUNK_SIZE_MB,
        )
        return [audio_path]

    duration_seconds = _ffprobe_duration_seconds(audio_path)

    # Calculate how many seconds of audio fit in one chunk
    bytes_per_second = file_size_bytes / duration_seconds
    seconds_per_chunk = (MAX_CHUNK_SIZE_MB * 1024 * 1024) / bytes_per_second
    num_chunks = math.ceil(duration_seconds / seconds_per_chunk)

    logger.info(
        "audio file %.1f MB (%.0fs), splitting into %d chunks of ~%.0fs each",
        file_size_bytes / (1024 * 1024),
        duration_seconds,
        num_chunks,
        seconds_per_chunk,
    )

    chunk_paths: list[str] = []
    for i in range(num_chunks):
        start = i * seconds_per_chunk
        end = min((i + 1) * seconds_per_chunk, duration_seconds)
        chunk_path = os.path.join(tmp_dir, f"chunk_{i:04d}.mp3")

        # Stream-copy the slice; mp3 frame granularity (~26 ms) is plenty
        # accurate for transcript stitching, and -c copy is near-instant.
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", audio_path,
                "-ss", f"{start:.3f}",
                "-to", f"{end:.3f}",
                "-c", "copy",
                chunk_path,
            ],
            check=True,
        )
        chunk_paths.append(chunk_path)
        logger.info(
            "chunk %d/%d exported: %.1f MB",
            i + 1,
            num_chunks,
            os.path.getsize(chunk_path) / (1024 * 1024),
        )

    return chunk_paths


def _groq_transcribe_chunk(chunk_path: str, api_key: str):
    """
    Sends a single audio chunk to Groq's Whisper *translations* endpoint,
    which returns English text regardless of the source language.
    Returns the raw API response object (verbose_json: .text, .segments,
    .duration).
    """
    from groq import Groq

    client = Groq(api_key=api_key)

    with open(chunk_path, "rb") as audio_file:
        response = client.audio.translations.create(
            model=GROQ_WHISPER_MODEL,
            file=audio_file,
            response_format="verbose_json",
        )

    return response


def _transcribe_chunk_with_fallback(chunk_path: str, chunk_index: int):
    """
    Tries translating a chunk with the primary API key first. If it hits
    a rate-limit (429) or auth error (401), retries with the fallback key.
    """
    from groq import APIStatusError

    keys = [(GROQ_API_KEY, "primary"), (FALLBACK_GROQ_API_KEY, "fallback")]

    for api_key, label in keys:
        if not api_key:
            continue
        try:
            logger.info("chunk %d: attempting Groq translation with %s key", chunk_index, label)
            result = _groq_transcribe_chunk(chunk_path, api_key)
            logger.info("chunk %d: success with %s key", chunk_index, label)
            return result
        except APIStatusError as exc:
            if exc.status_code in (401, 429) and label == "primary" and FALLBACK_GROQ_API_KEY:
                logger.warning(
                    "chunk %d: %s key returned HTTP %d, falling back to %s key",
                    chunk_index,
                    label,
                    exc.status_code,
                    "fallback",
                )
                continue
            raise TranscriptionError(
                f"Groq API error for chunk {chunk_index}: HTTP {exc.status_code} - {exc.message}"
            ) from exc
        except Exception as exc:
            raise TranscriptionError(
                f"Groq API call failed for chunk {chunk_index}: {exc}"
            ) from exc

    raise TranscriptionError("No valid GROQ API keys configured. Check your .env file.")


def _try_whisper(video_id: str) -> TranscriptResult:
    """
    Last-resort path for videos with no captions at all: download audio,
    chunk it, and translate each chunk to English via Groq's cloud Whisper
    API (translations endpoint, so the output is English by construction).
    Raises TranscriptionError on any failure -- there's nothing left to
    fall back to after this.
    """
    with _downloaded_audio(video_id) as audio_path:
        tmp_dir = os.path.dirname(audio_path)
        chunk_paths = _split_audio_into_chunks(audio_path, tmp_dir)

        all_segments: list[TranscriptSegment] = []
        all_text_parts: list[str] = []
        time_offset = 0.0  # cumulative offset for stitching chunk timestamps
        source_language: Optional[str] = None

        for i, chunk_path in enumerate(chunk_paths):
            response = _transcribe_chunk_with_fallback(chunk_path, i)

            reported_language = getattr(response, "language", None)
            if reported_language:
                source_language = reported_language

            chunk_text = response.text if hasattr(response, "text") else ""
            if chunk_text:
                all_text_parts.append(chunk_text.strip())

            # Extract segments with timestamps if available
            chunk_segments = getattr(response, "segments", None) or []
            for seg in chunk_segments:
                seg_start = getattr(seg, "start", 0.0) or 0.0
                seg_end = getattr(seg, "end", 0.0) or 0.0
                seg_text = getattr(seg, "text", "") or ""
                all_segments.append(
                    TranscriptSegment(
                        start=seg_start + time_offset,
                        end=seg_end + time_offset,
                        text=seg_text.strip(),
                    )
                )

            # Calculate this chunk's duration so the next chunk's timestamps
            # are offset correctly
            chunk_duration = getattr(response, "duration", None)
            if chunk_duration:
                time_offset += chunk_duration
            elif chunk_segments:
                # Fallback: use the last segment's end time as chunk duration
                last_seg = chunk_segments[-1]
                time_offset += getattr(last_seg, "end", 0.0) or 0.0

    full_text = " ".join(all_text_parts).strip()
    if not full_text:
        raise TranscriptionError(f"Groq Whisper produced empty output for video_id={video_id}")

    if source_language:
        logger.info(
            "video_id=%s Whisper detected source language: %s (output is English)",
            video_id,
            source_language,
        )

    return TranscriptResult(
        text=full_text,
        segments=all_segments,
        source=TranscriptSource.WHISPER_TRANSLATED,
        detected_language="en",  # translations endpoint always outputs English
        video_id=video_id,
    )


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def get_transcript(url: str) -> TranscriptResult:
    """
    The one function the transcription stage exposes. Runs the full
    fallback chain (captions -> TranscriptAPI -> Whisper) and returns as
    soon as one layer succeeds.

    Note: the returned text may be non-English (only possible via the
    TranscriptAPI layer). Use notes_gen.transcript_pipeline.translation.ensure_english, or
    the combined pipeline notes_gen.transcript_pipeline.translation.get_english_transcript,
    to guarantee English output.
    """
    video_id = extract_video_url(url)
    logger.info("video_id=%s starting transcript resolution", video_id)

    captions_result = _try_captions(video_id)
    if captions_result is not None:
        logger.info(
            "video_id=%s resolved via %s (free, no Whisper needed)",
            video_id,
            captions_result.source.value,
        )
        return captions_result

    logger.info("video_id=%s captions unavailable -- trying TranscriptAPI", video_id)
    api_result = _try_transcriptapi(video_id)
    if api_result is not None:
        logger.info(
            "video_id=%s resolved via TranscriptAPI (language=%s)",
            video_id,
            api_result.detected_language,
        )
        return api_result

    logger.info("video_id=%s TranscriptAPI unavailable -- falling back to Groq Whisper", video_id)
    return _try_whisper(video_id)
