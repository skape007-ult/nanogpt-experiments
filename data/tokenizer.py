import re
from collections import Counter
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet', quiet=True)
_lemmatizer = WordNetLemmatizer()


def prune_speakers(text, top_k_speakers=30):
    """Replace minor character tags with <GUEST>, keeping only the top-k speakers."""
    all_tags = re.findall(r'<[A-Z_]+>', text)
    tag_counts = Counter(all_tags)
    top_speakers = {tag for tag, _ in tag_counts.most_common(top_k_speakers)}

    def _replace(match):
        tag = match.group(0)
        return tag if tag in top_speakers else '<GUEST>'

    return re.sub(r'<[A-Z_]+>', _replace, text)


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

    words = text.split()
    lemmatized = []
    for w in words:
        if w.startswith('tag_') or w.endswith('_tag'):
            lemmatized.append(w)
        else:
            lemmatized.append(_lemmatizer.lemmatize(w))
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
