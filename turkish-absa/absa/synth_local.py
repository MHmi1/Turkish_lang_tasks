"""Offline rule-based Turkish ABSA edge-case generator.

Covers:
1. Heavy Turkish idioms for getting ripped off / terrible experience ("kazık yedik", "elimizde patladı",
   "paramız çöp oldu", "saç baş yoldurdu", "kan kusturdu", "ömür törpüsü", "fiyasko").
2. Biting sarcasm & irony ("Harika(!) sadece 3 haftada geldi", "Bayıldım(!) kutudan kırık çıktı").
3. Gen-Z & social media Turkish slang ("yok artık", "çöp", "efso", "aşırı iyi", "f/p kralı", "baya iyi",
   "toksik", "cringe", "kargo uçtu resmen", "pil ömrü şaka mı").
4. Formal corporate / polite complaints ("beklentilerimin oldukça altında", "tarafıma kusurlu ulaştı").
5. Contrastive multi-aspect & negation ("kamerası efsane ama bataryası günü çıkarmıyor", "fiyatı hiç de ucuz değil").
6. Implicit aspects (sentiment without naming an explicit aspect span).
"""
import random
import re

from .data import locate_spans

PRODUCTS = ["Samsung S24", "iPhone 15", "Xiaomi Redmi Note 13", "kulaklık", "laptop", "kahve makinesi",
            "spor ayakkabı", "mont", "krem", "protein tozu", "oyuncu faresi", "powerbank", "robot süpürge"]

# Each template uses [[ASP|surface|polarity]] for explicit spans, and optional implicit list.
TEMPLATES = [
    # ---- 1. Sarcasm & Irony (irony, idiom) -----------------------------------
    {
        "register": "sarcastic",
        "flags": ["irony", "idiom"],
        "domain": "ecommerce",
        "tpls": [
            "Bu {p} alarak resmen kazık yedik, [[ASP|fiyatı|negative]] tamamen soygun, [[ASP|performansı|negative]] da yerlerde!",
            "Maşallah [[ASP|kargo|negative]] hızına hayran kaldım (!), sadece 18 günde kapıma teşrif etti.",
            "Harika bir [[ASP|batarya|negative]] yapmışlar (!), sabah yüzde yüz şarj ediyorum öğlen olmadan kapanıyor.",
            "Aldığım {p} elimizde patladı, [[ASP|ekranı|negative]] ikinci günde bozuldu, paramız çöp oldu.",
            "Bayıldım gerçekten (!), [[ASP|paketleme|negative]] o kadar özenliydi ki kutu paramparça geldi.",
            "[[ASP|Müşteri hizmetleri|negative]] sağ olsun (!), yarım saat hatta bekletip yüzüme kapattılar.",
            "Bu [[ASP|fiyat|negative]]a böyle [[ASP|kalite|negative]] (!), {p} resmen şaka gibi, iki günde dağıldı.",
            "Süper (!) [[ASP|kamera|negative]] kalitesi var, gece çekimleri tost makinesiyle çekilmiş gibi.",
            "[[ASP|Teslimat|negative]] süreci tam bir ömür törpüsüydü, kurye adresi bulamayıp iade etmiş (!).",
            "Bravo valla (!), [[ASP|yazılımı|negative]] o kadar stabil ki günde beş kere donup reset atıyor.",
        ],
    },
    # ---- 2. Gen-Z & Informal Slang (gen_z, informal) -------------------------
    {
        "register": "gen_z",
        "flags": ["gen_z", "idiom", "multi_aspect"],
        "domain": "electronics",
        "tpls": [
            "abi {p} [[ASP|kamerası|positive]] efso olmuş, gece modu ateş ediyor ama [[ASP|batarya|negative]] tam çöp ya.",
            "[[ASP|fiyat performans|positive]] kralı bi ürün, [[ASP|kargo|positive]] da ışık hızında geldi valla helal.",
            "ya bu [[ASP|ses kalitesi|negative]] şaka mı?? baslar sıfır, verilen paraya yazık cidden cringe.",
            "[[ASP|tasarımı|positive]] aşırı iyi vibe veriyor, elde duruşu baya premium ama [[ASP|şarj aleti|negative]] kutudan çıkmadı :/ ",
            "kanka [[ASP|ekran yenileme hızı|positive]] yağ gibi akıyor, oyunlarda sıfır drop, [[ASP|ısınma|negative]] biraz var sadece.",
            "ürün tam bir f/p canavarı, [[ASP|malzeme kalitesi|positive]] beklentimin çok üstünde, [[ASP|kutu içeriği|neutral]] standart.",
            "[[ASP|uygulama|negative]] aşırı buglı ya, sürekli crash veriyor, [[ASP|bağlantı|negative]] kopup duruyor çıldırıcam.",
            "valla [[ASP|kumaşı|positive]] pamuk gibi baya rahat, ama [[ASP|kalıbı|negative]] aşırı dar bir beden büyük alın.",
            "[[ASP|teslimat|positive]] uçtu resmen dün sipariş verdim bugün elimde, [[ASP|ürün|positive]] de 10/10.",
            "abi [[ASP|fiyatı|negative]] uçmuş gitmiş, bu paraya değmez, [[ASP|performansı|neutral]] idare eder sadece.",
        ],
    },
    # ---- 3. Metaphorical & Idiomatic Turkish (idiom) -------------------------
    {
        "register": "informal",
        "flags": ["idiom", "contrast"],
        "domain": "ecommerce",
        "tpls": [
            "{p} alırken resmen gözümüzü boyamışlar, [[ASP|reklamı|negative]] başka [[ASP|gerçekteki kalitesi|negative]] bambaşka.",
            "[[ASP|Servis|negative]] süreci bize kan kusturdu, bir [[ASP|parça değişimi|negative]] için kırk dereden su getirdiler.",
            "[[ASP|Motor gücü|positive]] canavar gibi, halıdaki tüm tozu söküp alıyor ama [[ASP|sesi|negative]] kafa ütülüyor.",
            "[[ASP|Kurulumu|negative]] saç baş yoldurdu, [[ASP|kullanım kılavuzu|negative]] hiçbir işe yaramıyor.",
            "Verdiğim her kuruş helal olsun, [[ASP|performansı|positive]] taş çıkartır, [[ASP|dayanıklılığı|positive]] da cabası.",
            "[[ASP|Satıcı|negative]] burnundan kıl aldırmıyor, [[ASP|iade süreci|negative]] tam bir işkenceye dönüştü.",
            "[[ASP|Görüntü kalitesi|positive]] cam gibi, [[ASP|renkler|positive]] canlı fakat [[ASP|hoparlör|negative]] teneke gibi ses veriyor.",
            "Kargocu paketi tekmeleyerek getirmiş sanki, [[ASP|kutu|negative]] haşat olmuş ama [[ASP|cihaz|positive]] sağlam çıktı.",
        ],
    },
    # ---- 4. Formal & Polite Corporate Register (formal) ----------------------
    {
        "register": "formal",
        "flags": ["formal", "contrast"],
        "domain": "ecommerce",
        "tpls": [
            "Tarafıma ulaşan {p} ürününün [[ASP|ambalajı|positive]] özenli olmakla birlikte, [[ASP|teknik performansı|negative]] beklentilerimin oldukça altındadır.",
            "[[ASP|Teslimat süresi|positive]] taahhüt edildiği gibi gerçekleşmiş olup, [[ASP|ürün kalitesi|positive]] fiyatını fazlasıyla hak etmektedir.",
            "Ürünün [[ASP|malzeme işçiliği|negative]] kusurlu bulunmuş, [[ASP|iade talebi|neutral]] oluşturulmuştur.",
            "[[ASP|Müşteri temsilcisi|positive]] son derece ilgili davrandı; ancak [[ASP|kargo firması|negative]] teslimatı geciktirdi.",
            "Cihazın [[ASP|pil ömrü|neutral]] günlük kullanım için yeterli düzeyde, [[ASP|ekran parlaklığı|positive]] ise oldukça başarılı.",
            "[[ASP|Fiyatlandırma|negative]] politikasını makul bulmadığımı, buna karşın [[ASP|satış sonrası destek|positive]] hizmetinden memnun kaldığımı belirtmek isterim.",
        ],
    },
    # ---- 5. Tricky Negation & Double Negation (negation) ---------------------
    {
        "register": "standard",
        "flags": ["negation", "multi_aspect"],
        "domain": "electronics",
        "tpls": [
            "[[ASP|Kamerası|negative]] hiç de anlatıldığı kadar iyi değil, üstelik [[ASP|gece çekimi|negative]] berbat.",
            "[[ASP|Fiyatı|positive]] pahalı sayılmaz, [[ASP|kalitesi|positive]] de fena değil açıkçası memnun kaldım.",
            "[[ASP|Kargo|positive]] gecikmedi değil, tam zamanında geldi ve [[ASP|paketleme|positive]] sorunsuzdu.",
            "Ne [[ASP|ses kalitesi|negative]] tatmin etti ne de [[ASP|mikrofonu|negative]] düzgün çalışıyor.",
            "[[ASP|Isınma sorunu|positive]] yok denecek kadar az, [[ASP|şarj süresi|positive]] de hiç fena değil.",
            "[[ASP|Kumaşı|negative]] beklediğim gibi çıkmadı, yıkandıktan sonra çekmez demişlerdi ama çekti.",
        ],
    },
]

# Implicit-only templates (no explicit aspect span in text; sentiment is clear from idiom/sarcasm)
IMPLICIT_TEMPLATES = [
    ("Bu {p} alarak paramızı sokağa attık, tam bir hayal kırıklığı ve soygun!", "ürün", "negative", "sarcastic", ["idiom", "implicit"]),
    ("Sipariş vereli iki hafta oldu hâlâ yola bile çıkmamış, kaplumbağa hızında!", "teslimat", "negative", "sarcastic", ["irony", "implicit"]),
    ("Gözünüz kapalı alın, verdiğiniz her kuruşun hakkını sonuna kadar veriyor.", "ürün", "positive", "informal", ["idiom", "implicit"]),
    ("Dün akşam sipariş ettim bu sabah kapımdaydı, gerçekten helal olsun.", "kargo", "positive", "informal", ["implicit"]),
    ("Bu fiyata böyle bir performans mucize, rakiplerine göre bedava sayılır.", "fiyat", "positive", "gen_z", ["implicit"]),
    ("Tam anlamıyla elimizde patladı, ikinci gün bozulup çöp oldu.", "kalite", "negative", "informal", ["idiom", "implicit"]),
    ("Ne iyi diyebilirim ne kötü, beklentimi tam karşılamadı ama iş görür.", "ürün", "neutral", "standard", ["implicit"]),
    ("Kutuyu bir açtım her yeri ezik büzük, insan biraz balonlu naylona sarar!", "paketleme", "negative", "informal", ["implicit"]),
]

MARK = re.compile(r"\[\[ASP\|([^|\]]+)\|(positive|negative|neutral)\]\]")


def _parse_marked(marked):
    out, aspects, last, cur = [], [], 0, 0
    for m in MARK.finditer(marked):
        pre = marked[last:m.start()]
        out.append(pre)
        cur += len(pre)
        surf, pol = m.group(1).strip(), m.group(2).strip()
        out.append(surf)
        aspects.append({"span": surf, "start": cur, "end": cur + len(surf), "aspect": surf.lower(), "polarity": pol})
        cur += len(surf)
        last = m.end()
    out.append(marked[last:])
    return "".join(out), aspects


PREFIXES = ["", "", "", "Açıkçası ", "Bence ", "Doğrusunu söylemek gerekirse ", "Yani ", "Valla "]
SUFFIXES_POS = ["", "", "", " Tavsiye ederim.", " Bilginize.", " :)", " 10/10!"]
SUFFIXES_NEG = ["", "", "", " Kesinlikle uzak durun.", " Paranızı çöpe atmayın.", " :/", " !!!"]
SUFFIXES_NEU = ["", "", "", " Bilginize.", " Karar sizin."]


def _pick_suffix(pols, r):
    if "negative" in pols and "positive" not in pols:
        return r.choice(SUFFIXES_NEG)
    if "positive" in pols and "negative" not in pols:
        return r.choice(SUFFIXES_POS)
    return r.choice(SUFFIXES_NEU)


def generate(n=2400, seed=42):
    r = random.Random(seed)
    docs = []
    for i in range(n):
        if i % 6 == 0:
            tpl, asp, pol, reg, flags = r.choice(IMPLICIT_TEMPLATES)
            p = r.choice(PRODUCTS)
            text = r.choice(PREFIXES) + tpl.format(p=p) + _pick_suffix({pol}, r)
            docs.append({
                "id": f"local_imp_{i}",
                "text": text.strip(),
                "domain": "ecommerce",
                "source": "synth_local",
                "register": reg,
                "flags": list(flags),
                "group": f"imp_{IMPLICIT_TEMPLATES.index((tpl, asp, pol, reg, flags))}",
                "aspects": [{"span": None, "start": None, "end": None, "aspect": asp, "polarity": pol}],
            })
            continue
        group_idx = r.randrange(len(TEMPLATES))
        group = TEMPLATES[group_idx]
        ti = r.randrange(len(group["tpls"]))
        tpl = group["tpls"][ti]
        p = r.choice(PRODUCTS)
        pols = {m.group(2) for m in MARK.finditer(tpl)}
        raw = r.choice(PREFIXES) + tpl.format(p=p) + _pick_suffix(pols, r)
        text, asps = _parse_marked(raw.strip())
        asps = locate_spans(text, asps)
        docs.append({
            "id": f"local_{i}",
            "text": text,
            "domain": group["domain"],
            "source": "synth_local",
            "register": group["register"],
            "flags": list(group["flags"]),
            "group": f"g{group_idx}_t{ti}",
            "aspects": asps,
        })
    return docs
