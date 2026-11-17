"""Synthetic, unverified Turkish challenge suite. For diagnostic use only.
Not a native-annotated scientific gold set. Never train on this file.
"""
from .data import locate_spans

# (review, [(verbatim explicit span or None, canonical category, polarity)], flags)
_CASES = [
    ("Üç hafta bekleyip kırık kutu teslim almak da ayrıcalıkmış, teşekkürler (!)\n", [("kutu","paketleme","negative"), (None,"teslimat","negative")], ["irony","contrast"]),
    ("Üç günde bozuldu; verdiğim para resmen buhar oldu.", [(None,"kalite","negative"),(None,"fiyat","negative")], ["idiom","implicit"]),
    ("Kanka bu kulaklığın basları uçuyor ama mikrofon bildiğin patates.", [("basları","ses","positive"),("mikrofon","mikrofon","negative")], ["gen_z","contrast"]),
    ("Kumaşın dokusu harika da kalıbı bildiğin çadır gibi; fotoğraftakiyle alakasız.", [("Kumaşın dokusu","kumaş","positive"),("kalıbı","beden","negative")], ["idiom","contrast"]),
    ("Mükemmel hizmet (!), üç kere aradım tek bir insana ulaşamadım.", [(None,"müşteri hizmetleri","negative")], ["irony","implicit"]),
    ("Güncellemeden sonra uygulama yağ gibi akıyor; eskisine nazaran şahane.", [("uygulama","uygulama","positive")], ["idiom"]),
    ("Abi 15k verdik ekran 60 hz, resmen şaka. Kamerası ise üzmedi.", [("ekran","ekran","negative"),("Kamerası","kamera","positive")], ["gen_z","contrast"]),
    ("Öyle bir paketlemişler ki açmak için tornavida lazım; sapasağlam gelmiş.", [(None,"paketleme","positive")], ["idiom","implicit"]),
    ("Almadığıma pişman değilim, fiyatına bakınca değmezmiş zaten.", [("fiyatına","fiyat","negative")], ["negation"]),
    ("Açıkçası beklentilerimi karşılamayan bir işçilik söz konusu.", [("işçilik","kalite","negative")], ["formal"]),
    ("Pili kötü diyemem, iki gün rahat gidiyor; şarj kablosu vasat.", [("Pili","batarya","positive"),("şarj kablosu","aksesuar","neutral")], ["negation","contrast"]),
    ("Fiyat uygunmuş gibi anlatılıyor ama sepette sürpriz masraf çıktı.", [("Fiyat","fiyat","negative")], ["irony"]),
    ("Sabah aldım, akşam kapımda. Bundan hızlısı ışınlanma herhalde.", [(None,"teslimat","positive")], ["idiom","implicit"]),
    ("Ya nasıl anlatsam... yorumlardaki övgülerle alakası yok, iki kullanışta dağıldı.", [(None,"kalite","negative")], ["informal","implicit"]),
    ("Ürün elime ulaştı, henüz denemedim; bu yüzden performansa dair yorum yok.", [("performansa","performans","neutral")], ["neutral"]),
    ("Hoparlör teneke gibi ötüyor ama ekranın parlaklığı epey iyi.", [("Hoparlör","ses","negative"),("ekranın parlaklığı","ekran","positive")], ["idiom","contrast"]),
    ("Şu fiyatı görünce kalbime indi; özellikleri de 2018'de kalmış.", [("fiyatı","fiyat","negative"),("özellikleri","performans","negative")], ["idiom"]),
    ("Sizden aldığım montun dikişleri ikinci giyişte sökülmüştür; iade talep ediyorum.", [("dikişleri","kalite","negative"),("iade","iade","neutral")], ["formal"]),
    ("Kargo desen yavaş, destek desen yok; ürünün kendisi çalışıyor bari.", [("Kargo","kargo","negative"),("destek","müşteri hizmetleri","negative"),("ürünün kendisi","ürün","positive")], ["contrast"]),
    ("Valla f/p desek olmaz, ucuzluk sadece etikette kalmış.", [(None,"fiyat","negative")], ["gen_z","implicit"]),
    ("Satıcı mesajıma saniyesinde döndü; kırık gelen ürün için değişim de sağlandı.", [("Satıcı","müşteri hizmetleri","positive"),("ürün","ürün","negative")], ["contrast"]),
    ("Dizüstünün fanı resmen uçak motoru, performans ise taş gibi.", [("fanı","ses","negative"),("performans","performans","positive")], ["idiom","contrast"]),
    ("Tam vibe ürün, yalnız kapaktaki boya elde kalıyor :')", [("boya","kalite","negative")], ["gen_z"]),
    ("Muhteşem zamanlama, doğum gününden iki hafta sonra geldi (!)", [(None,"teslimat","negative")], ["irony","implicit"]),
    ("Kamera gece iyi değil desem yalan olur, video sabitleme ortalama.", [("Kamera","kamera","positive"),("video sabitleme","kamera","neutral")], ["negation","contrast"]),
    ("Sözde premium. Düğmeleri bastıkça içeri kaçıyor, kalite kontrol tatile çıkmış.", [("Düğmeleri","kalite","negative")], ["irony","idiom"]),
    ("Uygulamanın arayüzü sade, açılışı biraz hantal; genel olarak idare eder.", [("arayüzü","arayüz","positive"),("açılışı","performans","negative")], ["contrast"]),
    ("Bu ayakkabı ayağıma değil sanki başka gezegene göre tasarlanmış.", [(None,"beden","negative")], ["idiom","implicit"]),
    ("Hiç değilse iade ederken uğraştırmadılar, para iki günde yattı.", [(None,"iade","positive")], ["negation","implicit"]),
    ("Ne iyi ne kötü, bu iş için normal bir seçenek.", [(None,"ürün","neutral")], ["neutral","implicit"]),
    ("Kargosu bir yıldız, kumaş kalitesi beş yıldız; hangisine puan versek?", [("Kargosu","kargo","negative"),("kumaş kalitesi","kumaş","positive")], ["contrast"]),
    ("Ooo teknoloji harikası(!), açılışta beş kere donuyor; kasmaya bayıldık.", [(None,"performans","negative")], ["irony","implicit"]),
]


def curated_stress_set():
    docs=[]
    for i,(text, triples, tags) in enumerate(_CASES):
        text=text.strip()
        raw=[{"span": span, "aspect": category, "polarity":pol} for span,category,pol in triples]
        aligned=locate_spans(text,raw)
        if len(aligned)!=len(raw):
            raise ValueError(f"Curated stress span mismatch at example {i}: {text!r}")
        docs.append({"id":f"stress_{i:03}","text":text,"domain":"ecommerce",
                     "source":"curated_synthetic_unverified","register":"informal",
                     "flags":list(set(tags + (["implicit"] if any(a['span'] is None for a in aligned) else []))),
                     "aspects":aligned})
    return docs
