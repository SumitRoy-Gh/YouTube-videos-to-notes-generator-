import os

# transcription

TRANSCRIPTAPI_KEY = os.getenv("Youtube_to_Transcript_API_KEY")
TRANSCRIPTAPI_URL = "https://transcriptapi.com/api/v2/youtube/transcript"
TRANSCRIPTAPI_TIMEOUT = 30
TRANSCRIPTAPI_MAX_RETRIES = 2
GROQ_WHISPER_MODEL = "whisper-large-v3"
MAX_CHUNK_SIZE_MB = 20
MAX_AUDIO_SECONDS = 30 * 60 * 60  
_RETRYABLE_STATUSES = {408, 429, 503}

# translation

GROQ_TRANSLATOR_LLM_MODEL = "openai/gpt-oss-120b"
SEGMENTS_PER_BATCH = 40
WORDS_PER_WINDOW = 2500

#segmenter

DEFAULT_MAX_CHUNK_TOKENS = 700000
TOKENS_PER_WORD_ESTIMATE = 1.3

#extractor

MAX_PARALLEL_CHUNKS = 4
MAX_REPAIR_ATTEMPTS = 1
EXTRACTOR_TEMPERATURE = 0.2
# DeepSeek V4.1 Flash via NVIDIA NIM: ~8B active / ~16B output at 500B+
# stored knowledge -- speed play for the structural extraction step.
# Synthesizer stays on the ultra model (reasoning-heavy merge step).
EXTRACTOR_MODEL = "nvidia_nim/deepseek-ai/deepseek-v4.1-flash"

#synthesizer

SYNTHESIZER_TEMPERATURE = 0.3
SYNTHESIZER_MODEL = "nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b"