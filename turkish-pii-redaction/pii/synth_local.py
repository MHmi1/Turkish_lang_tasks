"""Rule-based Turkish PII edge-case generator (checksum-valid IDs, suffix-bearing entities,
hard negatives, formatting/OCR noise). Fully offline; used alongside Gemini data."""
import random
import re

from .data import make_doc
from .rules import gen_tckn, gen_vkn, gen_iban, gen_card, gen_phone, gen_plate

FIRST = ["Ahmet", "Mehmet", "Ayşe", "Fatma", "Elif", "Zeynep", "Mustafa", "Emre", "Burak", "Selin", "Deniz",
         "Cem", "Merve", "Gökhan", "Şule", "Özlem", "Hüseyin", "İbrahim", "Yasemin", "Ceren", "Tolga", "Barış",
         "Büşra", "Kübra", "Onur", "Pınar", "Serkan", "Esra", "Volkan", "Gül", "Murat", "Hakan", "Nurcan",
         "Orhan", "Ümit", "Çağla", "Sinan", "Derya", "Kemal", "Aylin", "Ismail", "Ebru", "Tuğba", "Oğuz"]
LAST = ["Yılmaz", "Kaya", "Demir", "Şahin", "Çelik", "Yıldız", "Yıldırım", "Öztürk", "Aydın", "Özdemir", "Arslan",
        "Doğan", "Kılıç", "Aslan", "Çetin", "Kara", "Koç", "Kurt", "Özkan", "Şimşek", "Polat", "Korkmaz", "Çakır",
        "Erdoğan", "Aksoy", "Güneş", "Türk", "Acar", "Bulut", "Tekin", "Karaca", "Uçar", "Balcı", "Ergün"]
PLACES = [("Kadıköy", "İstanbul"), ("Çankaya", "Ankara"), ("Bornova", "İzmir"), ("Nilüfer", "Bursa"),
          ("Muratpaşa", "Antalya"), ("Seyhan", "Adana"), ("Selçuklu", "Konya"), ("Şahinbey", "Gaziantep"),
          ("Melikgazi", "Kayseri"), ("Ortahisar", "Trabzon"), ("Tepebaşı", "Eskişehir"), ("Yenimahalle", "Ankara"),
          ("Beşiktaş", "İstanbul"), ("Karşıyaka", "İzmir"), ("Pendik", "İstanbul"), ("Odunpazarı", "Eskişehir")]
HOODS = ["Atatürk", "Cumhuriyet", "Yeni", "Fatih", "Barbaros", "Gazi", "Kurtuluş", "Mimar Sinan", "Yıldız",
         "Hürriyet", "Çınar", "Zafer", "İnönü", "Bahçelievler", "Esentepe"]
STREETS = ["Gül", "Lale", "Çiçek", "Menekşe", "Papatya", "Akasya", "Ihlamur", "Çam", "Kavak", "Dere", "Bahar", "Gökova"]
DOMAINS = ["gmail.com", "hotmail.com", "outlook.com", "yahoo.com", "example.com.tr", "firma.com.tr", "mail.com"]
MONTHS = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]
FOLD = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")

VOW, BACK, ROUND = "aeıioöuü", "aıou", "oöuü"
DIGIT_WORD = {"0": "sıfır", "1": "bir", "2": "iki", "3": "üç", "4": "dört", "5": "beş", "6": "altı",
              "7": "yedi", "8": "sekiz", "9": "dokuz"}


def suffix(word, case):
    """Turkish case suffix (without apostrophe) following vowel harmony."""
    word = word.rstrip(".,;:!?) ")
    if word and word[-1].isdigit():
        word = DIGIT_WORD[word[-1]]
    elif word and not word[-1].isalpha():
        m = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü]+|\d+", word)
        word = (DIGIT_WORD[m[-1][-1]] if m[-1][-1].isdigit() else m[-1]) if m else "a"
    w = word.translate(str.maketrans({"I": "ı", "İ": "i"})).lower()
    lv = next((c for c in reversed(w) if c in VOW), "e")
    ev, vl = w[-1] in VOW, w[-1] in "çfhkpsşt"
    back, rnd = lv in BACK, lv in ROUND
    v4 = ("u" if rnd else "ı") if back else ("ü" if rnd else "i")
    v2 = "a" if back else "e"
    if case == "gen":
        return ("n" if ev else "") + v4 + "n"
    if case == "dat":
        return ("y" if ev else "") + v2
    if case == "acc":
        return ("y" if ev else "") + v4
    if case == "loc":
        return ("t" if vl else "d") + v2
    if case == "abl":
        return ("t" if vl else "d") + v2 + "n"
    if case == "ins":
        return ("y" if ev else "") + "l" + v2
    return ""


def _name(r):
    n = r.choice(FIRST)
    if r.random() < 0.15:
        n += " " + r.choice(FIRST)
    return f"{n} {r.choice(LAST)}"


def _addr(r):
    ilce, il = r.choice(PLACES)
    hood, st = r.choice(HOODS), r.choice(STREETS)
    k = r.random()
    if k < 0.4:
        return f"{hood} Mah. {st} Sok. No:{r.randint(1, 200)} D:{r.randint(1, 30)}, {ilce}/{il}"
    if k < 0.7:
        return f"{hood} Mh. {st} Cd. No: {r.randint(1, 200)} Kat {r.randint(1, 12)}, {ilce}/{il}"
    if k < 0.85:
        return f"{st} Apt. {hood} Mahallesi {r.randint(1, 90)}. Sokak No {r.randint(1, 99)} {ilce} {il}"
    return f"{hood.lower()} mah {st.lower()} sk no {r.randint(1, 99)} d {r.randint(1, 20)} {ilce.lower()}"


def _email(r, person=None):
    n = (person or _name(r)).translate(FOLD).lower().split()
    sep = r.choice([".", "_", ""])
    return f"{n[0]}{sep}{n[-1]}{r.choice(['', str(r.randint(1, 99)), str(r.randint(1970, 2005))])}@{r.choice(DOMAINS)}"


def _dob(r):
    d, m, y = r.randint(1, 28), r.randint(1, 12), r.randint(1950, 2006)
    return r.choice([f"{d:02d}.{m:02d}.{y}", f"{d}/{m}/{y}", f"{d} {MONTHS[m - 1]} {y}", f"{y}-{m:02d}-{d:02d}",
                     f"{d} {MONTHS[m - 1].lower()} {y}"])


def _rand(r, n, alphabet="0123456789"):
    return "".join(r.choice(alphabet) for _ in range(n))


def _secret(r, kind):
    if kind == "otp":
        return _rand(r, r.choice([4, 5, 6, 6, 6, 8]))
    if kind == "cvv":
        return _rand(r, 3)
    if kind == "pin":
        return _rand(r, 4)
    if kind == "pwd":
        return r.choice(["Kedi", "Elma", "Deniz", "Sifre", "Ankara"]) + _rand(r, 4) + r.choice("!*#.")
    return r.choice(["sk_demo_", "tok_", "key-", "API-DEMO-"]) + _rand(r, 14, "abcdef0123456789")


def _acct(r, kind):
    if kind == "cust":
        return _rand(r, r.choice([7, 8, 9]))
    if kind == "contract":
        return f"SZL-{r.randint(2019, 2026)}-{_rand(r, 6)}"
    if kind == "policy":
        return f"{r.choice(['PL', 'POL', 'KSK'])}{_rand(r, 9)}"
    if kind == "subs":
        return _rand(r, 10)
    return _rand(r, r.choice([8, 10, 12]))


def _url(r):
    host = r.choice(["portal", "musteri", "app", "hesap"]) + "." + r.choice(["firma.com.tr", "example.org", "banka-demo.com"])
    k = r.random()
    if k < 0.5:
        return f"https://{host}/reset?token={_rand(r, 20, 'abcdef0123456789')}"
    if k < 0.8:
        return f"https://{host}/dosya/{_rand(r, 8, 'abcdefghjkmnpqrstuvwxyz23456789')}/{r.randint(1000, 99999)}"
    return f"https://{host}/u/{_name(r).translate(FOLD).lower().replace(' ', '-')}-{r.randint(10, 999)}"


def gen_value(label, kind, r):
    if label == "PERSON":
        return _name(r)
    if label == "ADDRESS":
        return _addr(r)
    if label == "PHONE":
        return gen_phone(r)
    if label == "EMAIL":
        return _email(r)
    if label == "DATE":
        return _dob(r)
    if label == "URL":
        return _url(r)
    if label == "ACCOUNT_NUMBER":
        return _acct(r, kind)
    if label == "SECRET":
        return _secret(r, kind)
    if label == "TCKN":
        v = gen_tckn(r)
        return f"{v[:3]} {v[3:6]} {v[6:9]} {v[9:]}" if r.random() < 0.08 else v
    if label == "VKN":
        return gen_vkn(r)
    if label == "IBAN":
        return gen_iban(r)
    if label == "CARD":
        return gen_card(r)
    if label == "PLATE":
        return gen_plate(r)
    raise KeyError(label)


def gen_distractor(kind, r):
    return {
        "price": r.choice([f"{r.randint(50, 9999)} TL", f"{r.randint(1, 9)}.{r.randint(100, 999)},{r.randint(10, 99)} TL",
                           f"₺{r.randint(10, 999)}"]),
        "qty": f"{r.randint(1, 12)} adet",
        "time": f"{r.randint(8, 22):02d}:{r.choice(['00', '15', '30', '45'])}",
        "holiday": r.choice(["23 Nisan", "29 Ekim Cumhuriyet Bayramı", "19 Mayıs", "1 Mayıs", "30 Ağustos", "Kurban Bayramı"]),
        "campaign": f"{r.randint(1, 28)}.{r.randint(1, 12)}.2026",
        "version": f"v{r.randint(1, 9)}.{r.randint(0, 20)}.{r.randint(0, 9)}",
        "httpcode": r.choice(["404", "500", "502", "503", "429", "401"]),
        "barcode": "869" + _rand(r, 10),
        "stock": f"STK-{r.randint(10000, 99999)}",
        "model": r.choice(["iPhone 15", "Galaxy S24", "Redmi Note 13", "Lenovo IdeaPad 3", "Arçelik 9 Kg", "Philips HD9252"]),
        "pubsite": r.choice(["https://www.turkiye.gov.tr", "https://www.trendyol.com", "www.tcmb.gov.tr", "https://www.google.com"]),
        "city": r.choice(["İstanbul", "Ankara", "İzmir", "Bursa", "Antalya", "Kadıköy", "Çankaya"]),
        "pct": f"%{r.randint(1, 60)}",
        "branch": f"{r.randint(100, 999)} nolu şube",
        "ticket": f"#{r.randint(100000, 999999)}",
        "role": r.choice(["müşteri temsilciniz", "şube müdürü", "yetkili arkadaşımız", "teknik destek ekibi"]),
        "bank": r.choice(["Ziraat Bankası", "Garanti BBVA", "İş Bankası", "Akbank", "QNB", "Yapı Kredi"]),
    }[kind]


# ------------------------------------------------------------------- templates
T = [
    "Merhaba, ben {PERSON}. Kimlik numaram {TCKN}, kartım {CARD} ile ilgili işlem yapmak istiyorum.",
    "{PERSON:gen} IBAN'ı {IBAN}, havale tutarı {~price}. {~role} onay verdi.",
    "Müşteri {PERSON} aradı, geri dönüş için {PHONE} numarasını bıraktı.",
    "Hesap sahibi: {PERSON}\nIBAN: {IBAN}\nTCKN: {TCKN}\nİşlem: {~price}",
    "{PHONE:dat} ulaşılamadı, alternatif olarak {EMAIL} adresine bilgi verildi.",
    "iban {IBAN} tc {TCKN} adım {PERSON} para iadesi bekliyorum",
    "Firma {~bank} hesabı: {IBAN:abl} {~qty} ürün iadesi yapıldı. Vergi no {VKN}.",
    "Kart numaram {CARD}, doğrulama için gelen kod {SECRET.otp} sistem kabul etmiyor.",
    "Şubemizden {PERSON:ins} görüşüldü; müşteri no {ACCOUNT_NUMBER.cust}, e-posta {EMAIL}.",
    "Aramada {PERSON} kendini doğruladı: doğum tarihi {DATE}, telefon {PHONE}.",
    "Güvenlik: CVV {SECRET.cvv} ve kart {CARD} bilgisini kimseyle paylaşmayın. Aranacak hat {~branch}.",
    "Vergi kimlik numarası {VKN} olan firmanın yetkilisi {PERSON}, tel: {PHONE}.",
    "Kargo {PERSON:dat} teslim edilecek. Adres: {ADDRESS}. İletişim {PHONE}.",
    "Teslimat adresi {ADDRESS:dat} değiştirildi, alıcı {PERSON}.",
    "{PERSON} adına {~model} siparişi {ADDRESS} adresine gönderildi, {~price} ödendi.",
    "ALICI: {PERSON}  TEL: {PHONE}  ADRES: {ADDRESS}",
    "kuryeye {PHONE} yı arattım ama açmadı adres {ADDRESS} {~city} sınırları içinde",
    "İade talebi: {PERSON}, müşteri no {ACCOUNT_NUMBER.cust}, ürün {~model}, {~qty}.",
    "Siparişteki araç plakası {PLATE} olan kurye {PERSON:gen} adresine gelecek.",
    "Plakası {PLATE} olan aracı {PERSON:acc} tanıyor musunuz? Ruhsat sahibi {PERSON}.",
    "Ticket {~ticket}: kullanıcı {EMAIL} şifre sıfırlama bağlantısı istedi, link {URL}.",
    "Destek kaydı: {PERSON} ({EMAIL}) giriş yapamıyor. Hata kodu {~httpcode}, sürüm {~version}.",
    "Tek kullanımlık şifre {SECRET.otp} {PHONE:dat} gönderildi.",
    "Kullanıcı şifresini {SECRET.pwd} olarak paylaştı, kayıttan silinmeli.",
    "API anahtarı {SECRET.key} yanlışlıkla ticket'a yapıştırılmış, {PERSON} bilgilendirildi.",
    "{PERSON:gen} hesabı için sözleşme no {ACCOUNT_NUMBER.contract} ve poliçe {ACCOUNT_NUMBER.policy} güncellendi.",
    "Abone no {ACCOUNT_NUMBER.subs}, ad soyad {PERSON}, doğum tarihi {DATE}.",
    "Merhaba {PERSON}, doğrulama kodunuz {SECRET.otp}. Kimseyle paylaşmayınız.",
    "PIN {SECRET.pin} ile giriş yapılamadı, {PERSON} destek hattını aradı.",
    "Gizli bağlantı: {URL} — yalnızca {PERSON} için geçerli.",
    "Kiracı: {PERSON}, T.C. Kimlik No: {TCKN}, adres: {ADDRESS}.",
    "KİMLİK BİLGİLERİ ; ad soyad:{PERSON} ; tc:{TCKN} ; dogum:{DATE} ; tel:{PHONE}",
    "Fatura unvanı {PERSON}, VKN {VKN}, IBAN {IBAN}, fatura tutarı {~price} (KDV {~pct}).",
    "Vekaleten {PERSON:ins} imzalanan sözleşmede {ACCOUNT_NUMBER.contract} numaralı dosya geçmektedir.",
    "Hasta kabul: {PERSON}, doğum {DATE}, iletişim {PHONE}, e-posta {EMAIL}.",
    "Araç ruhsat bilgisi: plaka {PLATE}, sahibi {PERSON}, T.C. {TCKN}.",
    "selam ben {PERSON} numaram {PHONE} bi bakar mısınız kartımdan {~price} çekilmiş",
    "abi {EMAIL:dat} yolladım gelmedi galiba tc de {TCKN} yazıyo yanlış mı",
    "yarın {~time} de {PERSON:ins} görüşeceğim, evim {ADDRESS}",
    "hesap numaram {ACCOUNT_NUMBER.acct} ama iban {IBAN} bunu kullan",
    "Merhaba, {PERSON} ben. {~bank} kartımın son kullanma ve cvv {SECRET.cvv} bilgisi sorulmuş, normal mi?",
    "Whatsapp: {PHONE} | mail: {EMAIL} | isim: {PERSON}",
]

N = [
    "{~role} sizi {~time} saatleri arasında arayacaktır. Kampanya {~campaign} tarihine kadar geçerlidir.",
    "Ürün {~model} için stok kodu {~stock}, barkod {~barcode}, fiyat {~price}.",
    "Sunucu {~httpcode} hatası verdi, sürüm {~version} ile güncellendi, {~ticket} numaralı kayıt incelemede.",
    "{~holiday} nedeniyle şubelerimiz kapalı olacaktır. Detaylar için {~pubsite} adresini ziyaret edin.",
    "{~qty} {~model} sepete eklendi, indirim oranı {~pct}, toplam {~price}.",
    "{~bank} {~branch} hafta sonu {~time} - 17:00 arası hizmet vermektedir.",
    "Kargonuz {~city} dağıtım merkezine ulaştı, teslimat yarın yapılacaktır.",
    "Hesabınıza {~price} yatırıldı. Bilgi için {~pubsite} adresine bakabilirsiniz.",
    "Yeni uygulama sürümü {~version} yayınlandı; {~holiday} kampanyası {~pct} indirim içeriyor.",
    "{~ticket} numaralı talebiniz alındı, {~role} en kısa sürede dönüş yapacaktır.",
    "Barkod {~barcode} okutulamadı, lütfen {~stock} kodunu manuel girin.",
    "Merhaba, kartımla ilgili genel bir soru soracaktım, limit artışı için hangi belgeler gerekli?",
]

_SLOT = re.compile(r"\{(~?)([A-Za-z_]+)(?:\.([a-z]+))?(?::([a-z]+))?\}")


def _render(tpl, r):
    parts, spans, cur, last = [], [], 0, 0
    for m in _SLOT.finditer(tpl):
        pre = tpl[last:m.start()]
        parts.append(pre)
        cur += len(pre)
        last = m.end()
        neg, name, kind, case = m.groups()
        if neg:
            v = gen_distractor(name, r)
            parts.append(v)
            cur += len(v)
            continue
        v = gen_value(name, kind, r)
        parts.append(v)
        spans.append([cur, cur + len(v), name])
        cur += len(v)
        if case and name not in ("EMAIL", "URL", "SECRET"):
            sfx = "'" + suffix(v.split(" ")[-1] if not v[-1].isdigit() else v, case)
            parts.append(sfx)
            cur += len(sfx)
    parts.append(tpl[last:])
    return "".join(parts), spans


def _noise(text, spans, r):
    """Length-preserving perturbations so offsets stay valid."""
    k = r.random()
    if k < 0.12:
        text = text.translate(FOLD)
    elif k < 0.20:
        text = text.translate(str.maketrans({"I": "ı", "İ": "i"})).lower()
    elif k < 0.24:
        text = text.translate(str.maketrans({"i": "İ", "ı": "I"})).upper()
    elif k < 0.34:
        inside = {c for s, e, _ in spans for c in range(s, e)}
        swap = {"i": "ı", "ı": "i", "o": "0", "l": "1", "s": "ş", "g": "ğ", "u": "ü"}
        text = "".join(swap[c] if (i not in inside and c in swap and r.random() < 0.06) else c
                       for i, c in enumerate(text))
    return text


PREFIX = ["", "", "", "", "Merhaba, ", "İyi günler. ", "Not: ", "FW: ", "Müşteri diyor ki: ", "Selamlar, ", ">> "]


def generate(n_pos=4000, n_neg=1200, seed=0):
    r = random.Random(seed)
    docs = []
    for i in range(n_pos + n_neg):
        pos = i < n_pos
        pool = T if pos else N
        ti = r.randrange(len(pool))
        tpl = pool[ti]
        text, spans = _render(tpl, r)
        pre = r.choice(PREFIX)
        if pre:
            text = pre + text
            spans = [[s + len(pre), e + len(pre), l] for s, e, l in spans]
        t2 = _noise(text, spans, r)
        if len(t2) != len(text):
            t2 = text
        d = make_doc(f"local_{i}", t2, spans, "local", case="pos" if pos else "neg")
        d["group"] = f"tpl_{'p' if pos else 'n'}{ti}"
        docs.append(d)
    return docs
