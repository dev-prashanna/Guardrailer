"""
constants.py
Shared keyword and pattern lists for prompt security scoring.
Used by security_engine.py, scoring.py, ingest_precomputed.py, feedback_processor.py.
"""

SPARSE_KEYWORDS = [
    "ignore previous", "override", "bypass", "jailbreak", "system prompt",
    "your instructions", "forget", "disregard", "dan", "do anything now",
    "act as", "roleplay", "pretend you", "hypothetical", "in theory",
    "markdown injection", "code comment", "readme", "yaml", "json payload",
    "<script>", "]]>", "```", "<!--", "-->", "eval(", "exec(",
    "base64", "rot13", "hex encoded", "obfuscated",
    "ignore all", "new instructions", "you are now", "persona",
    "reveal", "output", "display", "print", "show", "expose",
    "previous instructions", "earlier instructions", "initial instructions",
    "developer mode", "debug mode", "admin mode", "root mode",
    "you must", "you will", "you shall", "comply", "obey",
    "no restrictions", "no rules", "no limits", "unrestricted",
    "evil", "uncensored", "unfiltered", "without guidelines",
]

ATTACK_PATTERNS = [
    "ignore previous", "ignore all", "bypass", "jailbreak", "override",
    "new instructions", "you are now", "do anything now", "dan",
    "system prompt", "forget", "disregard", "act as", "roleplay",
    "pretend you", "eval(", "exec(", "base64",
    "reveal", "output your", "show me your", "display your",
    "what are your rules", "how do you work", "what instructions",
    "previous instructions", "earlier instructions", "initial instructions",
    "developer mode", "debug mode", "admin mode", "root mode",
    "you must", "you will", "you shall", "comply", "obey",
    "no restrictions", "no rules", "no limits", "unrestricted",
    "evil", "uncensored", "unfiltered", "without guidelines",
    "hidden instruction", "secret instruction", "override safety",
    "without restrictions", "without limitations", "without limits",
    # Additional direct injection patterns
    "admin rights", "giving admin", "grant admin", "elevate privileges",
    "escalate privileges", "root access", "superuser", "sudo access",
    "override system", "replace instructions", "new system prompt",
    "your new instructions", "disregard all", "forget everything",
    "ignore safety", "bypass safety", "disable safety",
    "override safety", "ignore rules", "ignore guidelines",
    "you are now a", "act as if", "pretend to be",
    "simulate being", "roleplay as", "impersonate",
    "hypothetically", "in a hypothetical", "if you were",
    "imagine you are", "assume you are",
]

SUBTLE_PATTERNS = [
    "your rules", "your instructions", "your guidelines",
    "how do you", "what do you", "can you tell me",
    "what text", "what makes up", "underlying",
    "this is urgent", "i am the", "i am a developer",
    "debugging", "testing", "security audit",
]

BENIGN_PATTERNS = [
    "what is", "how do", "can you", "please", "thank you",
    "hello", "hi there", "good morning", "good evening",
    "weather", "time", "date", "translate", "explain",
    "describe", "list", "define", "summarize", "compare",
    "write", "create", "help me", "tell me", "show me",
    "i need", "i want", "i would like", "could you",
    "would you", "do you", "does", "is there", "are there",
    "what are", "what does", "how many", "how much",
    "where is", "when did", "why do", "who is",
]
