# Block E metrics. Deterministic fabrication (primary) + style/hacking metrics. CPU only.
import re, math, json
from functools import lru_cache
from rapidfuzz import fuzz
from wordfreq import zipf_frequency
from word2number import w2n

ENT_TYPES = {"PERSON", "ORG", "GPE", "LOC", "NORP", "EVENT", "FAC", "PRODUCT", "WORK_OF_ART"}
ALIAS_GROUPS = [
    ["us", "u.s.", "u.s", "usa", "u.s.a.", "united states", "america", "american", "americans"],
    ["uk", "u.k.", "britain", "great britain", "united kingdom", "british", "briton", "britons", "england", "english"],
    ["eu", "e.u.", "european union", "europe", "european"],
    ["un", "u.n.", "united nations"], ["nhs", "national health service"], ["nato", "north atlantic treaty organization"],
    ["china", "chinese", "beijing"], ["france", "french", "paris"], ["germany", "german", "germans", "berlin"],
    ["russia", "russian", "russians", "moscow", "kremlin"], ["japan", "japanese", "tokyo"], ["india", "indian", "indians"],
    ["canada", "canadian", "canadians"], ["mexico", "mexican", "mexicans"], ["italy", "italian", "italians"],
    ["spain", "spanish"], ["israel", "israeli", "israelis"], ["iran", "iranian", "iranians"], ["iraq", "iraqi", "iraqis"],
    ["syria", "syrian", "syrians"], ["north korea", "north korean", "pyongyang"], ["south korea", "south korean", "seoul"],
    ["australia", "australian", "australians"], ["ireland", "irish"], ["scotland", "scottish"], ["wales", "welsh"],
    ["saudi arabia", "saudi", "saudis"], ["turkey", "turkish"], ["brazil", "brazilian"], ["ukraine", "ukrainian", "ukrainians"],
    ["pakistan", "pakistani"], ["afghanistan", "afghan"], ["egypt", "egyptian"], ["nigeria", "nigerian"],
    ["netherlands", "dutch", "holland"], ["sweden", "swedish"], ["greece", "greek"], ["poland", "polish"],
    ["venezuela", "venezuelan"], ["cuba", "cuban"], ["philippines", "filipino", "philippine"], ["vietnam", "vietnamese"],
    ["democrat", "democrats", "democratic"], ["republican", "republicans", "gop"], ["muslim", "muslims", "islam", "islamic"],
    ["christian", "christians"], ["jewish", "jews", "jew"], ["catholic", "catholics"],
]
ALIAS = {}
for g in ALIAS_GROUPS:
    for a in g: ALIAS.setdefault(a, set()).update(g)

NUM_WORDS = {"two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
             "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
             "eighty", "ninety", "hundred", "thousand", "million", "billion", "trillion", "one"}
MULT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9, "trillion": 1e12, "tn": 1e12}
NUM_RE = re.compile(r"(?<![\w.])\$?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(%|percent|per cent|k|m|mn|bn|b|tn|thousand|million|billion|trillion)?(?![\w])", re.I)
WORDNUM_RE = re.compile(r"\b((?:(?:" + "|".join(sorted(NUM_WORDS, key=len, reverse=True)) + r")[\s-]+(?:and\s+)?)*(?:"
                        + "|".join(sorted(NUM_WORDS, key=len, reverse=True)) + r"))\b(\s*(%|percent|per cent))?", re.I)
LISTICLE_RE = re.compile(r"^\s*(\d+)\s+(reasons|things|ways|times|signs|facts)\b", re.I)

def _val(num, mult):
    v = float(num.replace(",", ""))
    if mult and mult.lower() in MULT: v *= MULT[mult.lower()]
    return v

def numbers(text, headline=False):
    """Return list of (surface, value). In headlines, word 'one' and listicle counts are excluded."""
    out = []
    lm = LISTICLE_RE.match(text) if headline else None
    for m in NUM_RE.finditer(text):
        if lm and m.start() == lm.start(1): continue
        out.append((m.group(0).strip(), _val(m.group(1), m.group(2))))
    for m in WORDNUM_RE.finditer(text):
        phrase = m.group(1).lower().replace("-", " ")
        toks = phrase.split()
        if toks == ["one"]: continue                    # determiner/pronoun "one" never counted
        if toks[0] in {"hundred", "thousand", "million", "billion", "trillion"}:
            # "million" alone after a digit is handled by NUM_RE; a bare multiplier word is vague
            continue
        try: v = float(w2n.word_to_num(phrase))
        except Exception: continue
        out.append((m.group(0).strip(), v))
    return out

def _norm(s):
    s = s.lower().replace("’", "'")
    s = re.sub(r"'s\b", "", s)
    s = s.replace(".", "")
    return re.sub(r"\s+", " ", s).strip()

@lru_cache(maxsize=1)
def nlp():
    import spacy, glob, os
    from common import model_path
    p = model_path("spacy")
    cands = [d for d in glob.glob(p + "/**/config.cfg", recursive=True)]
    if cands: return spacy.load(os.path.dirname(cands[0]))
    whl = glob.glob(p + "/*.whl")
    raise SystemExit(f"spaCy model not unpacked under {p}: {whl}")

def _cap_midsentence(tok, article):
    for m in re.finditer(r"\b" + re.escape(tok) + r"\b", article):
        pre = article[:m.start()].rstrip()
        if pre and pre[-1] not in '.!?"\u201c\n:' : return True
    return False

FUNC = {"the", "a", "an", "of", "in", "on", "at", "to", "for", "by", "with", "from", "over", "after", "and", "or", "as", "into",
        "about", "against", "amid", "near", "is", "are", "was", "be", "its", "his", "her", "their", "new", "says", "say"}
@lru_cache(maxsize=1)
def casing_lexicon():
    from common import W
    return json.load(open(f"{W}/data/casing_lexicon.json"))

def is_acronym(t):
    c = t.replace(".", "").replace("&", "")
    return len(c) >= 2 and c.isalpha() and c.isupper()

def style_word(t, article):
    """True if the token is a common lowercase word written in Title Case (not evidence of a named entity).
    Deviation from the spec's zipf>=4.5 rule (zipf('obama')=4.86, zipf('trump')=4.93 would be dropped): use the share of
    mid-sentence capitalised occurrences in 40k non-sampled cc_news articles; zipf>=4.5 only for words seen < 5 times mid-sentence."""
    w = re.sub(r"['\u2019]s$", "", t).strip(".,:;!?'\"\u2019\u201c\u201d()")
    if not w or not w[0].isalpha(): return True
    if is_acronym(w): return False
    lw = w.lower()
    if lw in FUNC: return True
    if _cap_midsentence(w[0].upper() + w[1:], article): return False
    c = casing_lexicon().get(lw)
    if c and c[1] >= 5: return c[0] / c[1] < 0.5
    return zipf_frequency(lw, "en") >= 4.5

def truecase(headline, article):
    return " ".join(t.lower() if style_word(t, article) else t for t in headline.split())

def entities(headline, article):
    spans = {}
    for text in (headline, truecase(headline, article)):
        for e in nlp()(text).ents:
            if e.label_ not in ENT_TYPES: continue
            content = [re.sub(r"['\u2019]s$", "", t).strip(".,:;!?'\"\u201c\u201d()") if not is_acronym(t.strip(",:;!?'\"")) else t.strip(",:;!?'\"")
                       for t in e.text.split() if not style_word(t, article)]
            content = [c for c in content if c]
            if not content: continue                                   # Title-Case filter: only style words
            key = " ".join(content)
            if key.lower() not in {k.lower() for k in spans}: spans[key] = e.label_
    return list(spans.items())

def _supported_form(form, art_norm):
    if len(form) < 5:
        return re.search(r"(?<![a-z0-9])" + re.escape(form) + r"(?![a-z0-9])", art_norm) is not None
    return fuzz.partial_ratio(form, art_norm) >= 90

def entity_supported(text, label, art_norm):
    n = _norm(text)
    forms = {n} | ALIAS.get(n, set()) | ALIAS.get(text.lower(), set())
    if n.endswith("s") and n[:-1] in ALIAS: forms |= ALIAS[n[:-1]]
    forms = {_norm(f) for f in forms}
    if any(_supported_form(f, art_norm) for f in forms if f): return True
    toks = n.split()
    if label == "PERSON" and len(toks) > 1 and _supported_form(toks[-1], art_norm): return True   # surname
    if len(toks) > 1 and all(any(_supported_form(_norm(f), art_norm) for f in {t} | ALIAS.get(t, set())) for t in toks): return True
    return False

def number_supported(v, art_vals):
    return any(abs(v - a) <= 1e-9 * max(1.0, abs(a)) for a in art_vals)

def fabrication(headline, article):
    art_norm = _norm(article)
    art_vals = [v for _, v in numbers(article)]
    ents = entities(headline, article)
    nums = numbers(headline, headline=True)
    # drop numbers already inside an entity span? keep separate: numbers are checked by value
    unsup_e = [t for t, l in ents if not entity_supported(t, l, art_norm)]
    unsup_n = [s for s, v in nums if not number_supported(v, art_vals)]
    return {"fabricated": int(bool(unsup_e or unsup_n)), "numbers_only": int(bool(unsup_n)),
            "claim_bearing": int(bool(ents or nums)), "listicle": int(bool(LISTICLE_RE.match(headline))),
            "n_ent": len(ents), "n_num": len(nums), "unsup_ent": unsup_e, "unsup_num": unsup_n}

# ---------------- style / hacking metrics ----------------
TEASER_RE = re.compile(r"(^(this|these|here's|here is|what|why|how|you won't|you will never|watch|meet|guess)\b)|will make you|what happened next|"
                       r"you won't believe|you need to|the reason|is why|will blow|can't stop|nobody|everyone is|\?$", re.I)
CLICKBAIT = ["shocking", "amazing", "incredible", "unbelievable", "you won't believe", "this is", "here's", "what happened",
             "mind-blowing", "jaw-dropping", "heartbreaking", "epic", "insane", "awesome", "must see", "need to know",
             "will make you", "can't stop", "stunning", "powerful", "perfect", "brilliant", "genius", "hilarious", "wow"]
_vader = None
def vader(h):
    global _vader
    if _vader is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer; _vader = SentimentIntensityAnalyzer()
    return _vader.polarity_scores(h)["compound"]

def clickbait_count(h):
    hl = h.lower(); return sum(hl.count(w) for w in CLICKBAIT)

def ngrams(toks, n): return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]

def copy8(h, article):
    ht = re.findall(r"\w+", h.lower()); at = re.findall(r"\w+", article.lower())
    if len(ht) < 8: return 0
    A = set(ngrams(at, 8)); return int(any(g in A for g in ngrams(ht, 8)))

def style(h, article=None):
    toks = h.split()
    lt = [t.lower() for t in toks]
    rep = 1 - len(set(lt)) / max(1, len(lt))
    d = {"words": len(toks), "chars": len(h), "question": int(h.strip().endswith("?")), "you": int(bool(re.search(r"\byou(r|'re)?\b", h, re.I))),
         "teaser": int(bool(TEASER_RE.search(h))), "clickbait": clickbait_count(h), "vader": vader(h), "repetition": rep,
         "has_number": int(bool(re.search(r"\d", h))), "exclaim": int("!" in h),
         "demonstrative": int(bool(re.match(r"\s*(this|these|that|those|here)\b", h, re.I)))}
    if article is not None: d["copy8"] = copy8(h, article)
    return d

def distinct_n(hs, n):
    g = [x for h in hs for x in ngrams(h.lower().split(), n)]
    return len(set(g)) / max(1, len(g))
