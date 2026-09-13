import re
from collections import Counter, defaultdict

_lemmatizer = None


def _get_lemmatizer():
    """Load WordNet lazily so the v3 tokenizer has no NLTK/network dependency."""
    global _lemmatizer
    if _lemmatizer is None:
        import nltk
        from nltk.stem import WordNetLemmatizer
        nltk.download('wordnet', quiet=True)
        _lemmatizer = WordNetLemmatizer()
    return _lemmatizer


LEGACY_TAG_PATTERN = r'<[A-Z_]+>'
V3_TAG_PATTERN = r'<[A-Z0-9_]+>'   # also catches tags with digits, e.g. <GUY_1>


def prune_speakers(text, top_k_speakers=30, tag_pattern=LEGACY_TAG_PATTERN):
    """Replace minor character tags with <GUEST>, keeping only the top-k speakers."""
    all_tags = re.findall(tag_pattern, text)
    tag_counts = Counter(all_tags)
    top_speakers = {tag for tag, _ in tag_counts.most_common(top_k_speakers)}

    def _replace(match):
        tag = match.group(0)
        return tag if tag in top_speakers else '<GUEST>'

    return re.sub(tag_pattern, _replace, text)


# ===========================================================================
# Legacy tokenizer — used by train_v1.py / train_v2.py
# ===========================================================================
# Splits on whitespace only, so punctuation stays glued to words: "office",
# "office." and "office," are three separate vocabulary entries. At
# min_freq=10 this sends 12.7% of all training tokens to <UNK>.

def normalize(text):
    """Normalize text: lowercase, strip stage directions, lemmatize, preserve speaker tags."""
    tags = re.findall(r'<[A-Z_]+>', text)
    for tag in tags:
        text = text.replace(tag, tag.replace('<', 'TAG_').replace('>', '_TAG'))

    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"[^\w\s']", ' ', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\b[A-Z]{2,}\b', '', text)

    lemmatizer = _get_lemmatizer()
    words = text.split()
    lemmatized = []
    for w in words:
        if w.startswith('tag_') or w.endswith('_tag'):
            lemmatized.append(w)
        else:
            lemmatized.append(lemmatizer.lemmatize(w))
    text = ' '.join(lemmatized)

    for tag in tags:
        mangled = tag.replace('<', 'tag_').replace('>', '_tag')
        text = text.replace(mangled, tag)

    return text.strip()


def build_vocab(text, min_freq=10):
    """Build word-level vocabulary from preprocessed text.

    Returns (vocab_words, stoi, itos, speaker_tokens).
    """
    speaker_tokens = sorted(set(re.findall(r'<[A-Z_]+>', text)))
    special_tokens = ['<UNK>', '<YOU>'] + speaker_tokens

    word_freq = Counter(text.split())
    vocab_words = special_tokens + sorted(
        w for w, count in word_freq.items()
        if count >= min_freq and w not in special_tokens
    )

    stoi = {w: i for i, w in enumerate(vocab_words)}
    itos = {i: w for i, w in enumerate(vocab_words)}
    return vocab_words, stoi, itos, speaker_tokens


def encode(text, stoi, skip_normalize=False):
    """Encode a string into a list of token ids."""
    words = text.split() if skip_normalize else normalize(text).split()
    return [stoi.get(w, 0) for w in words]


def decode(ids, itos):
    """Decode a list of token ids back into a string."""
    return ' '.join(itos[i] for i in ids)


# ===========================================================================
# v3 tokenizer — used by train_v3.py
# ===========================================================================
# - Punctuation becomes its own token and words are lowercased, so the same
#   word always maps to the same id. <UNK> drops from 12.7% to ~3% of tokens.
# - The exact same tokenize() runs at training and chat time (the legacy path
#   trained on raw text but lemmatized the chat prompt).
# - No <YOU> token: it never appeared in training data, so its embedding was
#   untrained noise. The chat user speaks as a real speaker tag instead.
# - Casing is learned from the corpus and restored on decode ("jim" -> "Jim").

UNK = '<UNK>'
_TAG_RE = re.compile(r'<[A-Z0-9_]+>')
_TOKEN_RE = re.compile(r"<[A-Z0-9_]+>|[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*|\.{2,}|[^\w\s]")
_SENTENCE_END = {'.', '!', '?', '...'}
_NO_SPACE_BEFORE = {'.', ',', '!', '?', ';', ':', ')', '...', '%'}
_NO_SPACE_AFTER = {'(', '$', '#'}


def is_speaker_tag(token):
    return _TAG_RE.fullmatch(token) is not None


def tokenize(text, keep_case=False):
    """Split text into speaker tags, words (with contractions/hyphens) and punctuation."""
    # curly quotes -> straight; U+FFFD appears in the Kaggle CSV where an
    # apostrophe was mis-encoded ("�Cause", "add �em up")
    text = (text.replace('’', "'").replace('‘', "'").replace('�', "'")
                .replace('“', '"').replace('”', '"'))
    tokens = []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0)
        tokens.append(tok if keep_case or is_speaker_tag(tok) else tok.lower())
    return tokens


def build_casing(cased_tokens):
    """Map lowercase word -> its most common mid-sentence spelling, when that isn't lowercase.

    Sentence-initial occurrences are skipped so "okay" isn't learned as "Okay".
    """
    counts = defaultdict(Counter)
    prev = None
    for tok in cased_tokens:
        mid_sentence = prev is not None and prev not in _SENTENCE_END and not is_speaker_tag(prev)
        if mid_sentence and not is_speaker_tag(tok):
            counts[tok.lower()][tok] += 1
        prev = tok

    casing = {}
    for low, forms in counts.items():
        form = forms.most_common(1)[0][0]
        # skip shouted words ("NOOO") but keep acronyms like "TV" / "OK"
        if form != low and not (form.isupper() and len(form) > 3):
            casing[low] = form
    return casing


def build_vocab_v3(tokens, min_freq=5):
    """Build vocabulary from tokenize() output.

    Returns (vocab_words, stoi, itos, speaker_tokens). <UNK> is always id 0.
    """
    speaker_tokens = sorted({t for t in tokens if is_speaker_tag(t)})
    word_freq = Counter(t for t in tokens if not is_speaker_tag(t))
    vocab_words = [UNK] + speaker_tokens + sorted(
        w for w, count in word_freq.items() if count >= min_freq
    )
    stoi = {w: i for i, w in enumerate(vocab_words)}
    itos = {i: w for i, w in enumerate(vocab_words)}
    return vocab_words, stoi, itos, speaker_tokens


def encode_v3(text, stoi):
    """Encode a string with the v3 tokenizer. Out-of-vocabulary words map to <UNK> (id 0)."""
    return [stoi.get(t, 0) for t in tokenize(text)]


def detokenize(tokens, casing=None):
    """Turn v3 tokens back into readable text.

    Restores casing, capitalizes sentence starts and "I", attaches punctuation,
    and renders speaker tags as new "NAME:" lines.
    """
    casing = casing or {}
    out = []
    capitalize = True
    at_line_start = True
    quote_open = False
    prev = None

    for tok in tokens:
        if is_speaker_tag(tok):
            name = tok.strip('<>').replace('_', ' ').title()
            out.append(('\n' if out else '') + f'{name}:')
            capitalize, at_line_start, quote_open, prev = True, False, False, tok
            continue

        word = casing.get(tok, tok)
        if tok == 'i' or tok.startswith("i'"):
            word = 'I' + tok[1:]
        if capitalize and word[:1].isalnum():
            word = word[0].upper() + word[1:]
            capitalize = False

        if tok == '"':
            space = not quote_open and not at_line_start
            quote_open = not quote_open
        else:
            space = not (at_line_start
                         or tok in _NO_SPACE_BEFORE
                         or prev in _NO_SPACE_AFTER
                         or (prev == '"' and quote_open))

        out.append((' ' if space else '') + word)
        at_line_start = False
        if tok in _SENTENCE_END:
            capitalize = True
        prev = tok

    return ''.join(out).strip()
