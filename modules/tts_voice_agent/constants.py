"""Named constants for tts-voice-agent. Change them here only and quote them in the report."""

# A scam result at or above this confidence is escalated as "high confidence".
# Below it (including 0.0 = unknown) is still escalated, but for review.
ESCALATE_THRESHOLD = 0.7

# Hard cap on tokens in CallState.context(). Must stay below vLLM's --max-model-len, with
# room left for the prompt and the answer. The count is PyThaiNLP words
# (default_count_tokens), not model tokens: Thai words are roughly 2-3 model tokens each, so
# 1500 is about 3000-4500 model tokens, well under qwen3.8's 100K-token window.
CONTEXT_TOKEN_BUDGET = 1500

# Newest turns kept word for word; older turns are folded into the summary first.
RECENT_TURNS_VERBATIM = 6

# Leak check: this many Thai/English digit words in a row count as a spoken number.
MIN_DIGIT_WORD_RUN = 2

# Longest LLM reply (characters) we accept before falling back to a template.
MAX_REPLY_CHARS = 200

# Default (no model) summarizer keeps only the last this many characters.
SUMMARY_FALLBACK_CHARS = 600
