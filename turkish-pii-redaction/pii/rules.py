"""Deterministic validators, regex detectors and valid-identifier generators."""
import re
import random

# ------------------------------------------------------------------ validators
def _digits(s):
    return re.sub(r"\D", "", s)


def tckn_valid(s):
    d = _digits(s)
    if len(d) != 11 or d[0] == "0":
        return False
    x = [int(c) for c in d]
    if (sum(x[0:9:2]) * 7 - sum(x[1:8:2])) % 10 != x[9]:
        return False
    return sum(x[:10]) % 10 == x[10]


def vkn_valid(s):
    d = _digits(s)
    if len(d) != 10:
        return False
    x = [int(c) for c in d]
    tot = 0
    for i in range(9):
        t = (x[i] + 9 - i) % 10
        v = (t * (2 ** (9 - i))) % 9
        if t != 0 and v == 0:
            v = 9
        tot += v
    return (10 - tot % 10) % 10 == x[9]


def iban_valid(s):
    r = re.sub(r"\s", "", s).upper()
    if not re.fullmatch(r"TR\d{24}", r):
        return False
    return int("".join(str(int(c, 36)) for c in r[4:] + r[:4])) % 97 == 1


def luhn_valid(s):
    d = _digits(s)
    if not 13 <= len(d) <= 19:
        return False
    tot = 0
    for i, c in enumerate(reversed(d)):
        n = int(c)
        if i % 2:
            n = n * 2 - 9 if n > 4 else n * 2
        tot += n
    return tot % 10 == 0


# ------------------------------------------------------------------- detectors
_TCKN_RE = re.compile(r"(?<!\d)\d{11}(?!\d)")
_VKN_RE = re.compile(r"(?<!\d)\d{10}(?!\d)")
_IBAN_RE = re.compile(r"(?<![A-Za-z0-9])TR\s?\d{2}(?:\s?\d{4}){5}\s?\d{2}(?!\d)", re.I)
_CARD_RE = re.compile(r"(?<!\d)\d(?:[ \-]?\d){12,18}(?!\d)")
_EMAIL_RE = re.compile(r"(?<![\w.+\-])[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+")
_PHONE_RE = re.compile(r"(?<!\d)(?:(?:\+|00)?90[\s\-.]?)?\(?0?[2-5]\d{2}\)?[\s\-.]?\d{3}[\s\-.]?\d{2}[\s\-.]?\d{2}(?!\d)")
_CTX_VKN = re.compile(r"vergi|vkn|v\.k\.n", re.I)
_CTX_CARD = re.compile(r"kart|card|\bpan\b", re.I)
_CTX_PHONE = re.compile(r"tel|gsm|cep|numara|ara[yn]|whatsapp|wp|iletişim|iletisim", re.I)
_PRIO = {"TCKN": 0, "IBAN": 1, "CARD": 2, "EMAIL": 3, "VKN": 4, "PHONE": 5}


def find_structured(text):
    """Checksum/format-validated structured spans: [(start, end, label, score)]."""
    c = []
    for m in _TCKN_RE.finditer(text):
        if tckn_valid(m.group()):
            c.append((m.start(), m.end(), "TCKN", 1.0))
    for m in _IBAN_RE.finditer(text):
        if iban_valid(m.group()):
            c.append((m.start(), m.end(), "IBAN", 1.0))
    for m in _CARD_RE.finditer(text):
        g = m.group()
        grouped = re.fullmatch(r"\d{4}([ \-]?)\d{4}\1\d{4}\1\d{4}", g) is not None
        ctx = _CTX_CARD.search(text[max(0, m.start() - 30):m.start()]) is not None
        if luhn_valid(g) and (grouped or ctx):
            c.append((m.start(), m.end(), "CARD", 0.98))
    for m in _EMAIL_RE.finditer(text):
        e = m.end()
        while text[e - 1] in ".-":
            e -= 1
        c.append((m.start(), e, "EMAIL", 0.99))
    for m in _VKN_RE.finditer(text):
        if vkn_valid(m.group()) and _CTX_VKN.search(text[max(0, m.start() - 30):m.start()]):
            c.append((m.start(), m.end(), "VKN", 0.95))
    for m in _PHONE_RE.finditer(text):
        g = m.group()
        bare = not re.match(r"\s*(\+|00|90|0)", g)
        if bare and not _CTX_PHONE.search(text[max(0, m.start() - 25):m.start()]):
            continue
        c.append((m.start(), m.end(), "PHONE", 0.9))
    c.sort(key=lambda t: (_PRIO[t[2]], -(t[1] - t[0])))
    out = []
    for s, e, l, sc in c:
        if all(e <= s2 or s >= e2 for s2, e2, _, _ in out):
            out.append((s, e, l, sc))
    return sorted(out)


# ------------------------------------------------------------------ generators
def gen_tckn(rng: random.Random):
    d = [rng.randint(1, 9)] + [rng.randint(0, 9) for _ in range(8)]
    d.append((sum(d[0:9:2]) * 7 - sum(d[1:8:2])) % 10)
    d.append(sum(d) % 10)
    return "".join(map(str, d))


def gen_vkn(rng):
    x = [rng.randint(0, 9) for _ in range(9)]
    tot = 0
    for i in range(9):
        t = (x[i] + 9 - i) % 10
        v = (t * (2 ** (9 - i))) % 9
        if t != 0 and v == 0:
            v = 9
        tot += v
    x.append((10 - tot % 10) % 10)
    return "".join(map(str, x))


def gen_iban(rng, style=None):
    bban = rng.choice(["00061", "00010", "00046", "00064", "00134", "00111", "00015", "00205"]) \
        + "0" + "".join(str(rng.randint(0, 9)) for _ in range(16))
    chk = 98 - int("".join(str(int(c, 36)) for c in bban + "TR00")) % 97
    raw = f"TR{chk:02d}{bban}"
    style = style or rng.choice(["spaced", "spaced", "compact", "lower"])
    if style == "spaced":
        return " ".join([raw[:4]] + [raw[i:i + 4] for i in range(4, 24, 4)] + [raw[24:]])
    return raw.lower() if style == "lower" else raw


def gen_card(rng):
    pre = rng.choice(["4", "4", "51", "52", "53", "54", "55", "9792"])
    d = [int(c) for c in pre] + [rng.randint(0, 9) for _ in range(15 - len(pre))]
    tot = 0
    for i, n in enumerate(reversed(d)):
        if i % 2 == 0:
            n = n * 2 - 9 if n > 4 else n * 2
        tot += n
    d.append((10 - tot % 10) % 10)
    s = "".join(map(str, d))
    sep = rng.choice([" ", " ", "-", ""])
    return sep.join(s[i:i + 4] for i in range(0, 16, 4))


def gen_phone(rng):
    pre = rng.choice(["530", "532", "533", "535", "537", "541", "542", "505", "506", "507", "551", "555",
                      "212", "216", "312", "232"])
    a, b, c = f"{rng.randint(0, 999):03d}", f"{rng.randint(0, 99):02d}", f"{rng.randint(0, 99):02d}"
    st = rng.choice(["0{p}{a}{b}{c}", "0{p} {a} {b} {c}", "0 ({p}) {a} {b} {c}", "+90 {p} {a} {b} {c}",
                     "+90{p}{a}{b}{c}", "0{p}-{a}-{b}-{c}", "0090 {p} {a} {b} {c}", "0{p} {a}{b}{c}"])
    return st.format(p=pre, a=a, b=b, c=c)


def gen_plate(rng):
    il = rng.randint(1, 81)
    letters = "".join(rng.choice("ABCDEFGHJKLMNPRSTUVYZ") for _ in range(rng.choice([1, 2, 3])))
    n = rng.randint(10, 9999) if len(letters) == 1 else rng.randint(10, 999)
    return rng.choice([f"{il:02d} {letters} {n}", f"{il:02d}{letters}{n}"])
