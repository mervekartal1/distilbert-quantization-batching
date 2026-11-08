**Beklenti.** Dynamic quantization'ın CPU'da gecikmeyi yaklaşık 2–3 kat azaltmasını, model boyutunu %75 küçültmesini ve doğruluğu 1 puandan az düşürmesini; batching'in ise örnek başına maliyeti düşürüp throughput'u artırmasını bekledik.

**Sonuç.** Batch=1'de quantization gecikmeyi p50 17.6 → 10.6 ms, p95 19.8 → 12.3 ms düşürdü (1.66 kat; 3 run). Doğruluk %91.2 → %90.4 (-0.8 puan; McNemar p = 0.454, istatistiksel olarak anlamlı değil; farkın %95 GA [-2.4, 0.8] puan). Pratik değerlendirme: belirsiz: aralık −1.0 puanı içeriyor; 500 örnek karar için yetersiz. Model dosyası 268 → 139 MB küçüldü (%48.2), beklenen %75'in altında. Batch=32, throughput'u FP32'de 56 → 60, INT8'de 95 → 166 örnek/s'ye çıkardı; ancak çağrı başına gecikme arttı.

**Olası nedenler.** INT8 matris çarpımı daha az hesap ve 4 kat daha az bellek trafiği gerektirir. Boyutun %75 küçülmemesinin nedeni, parametrelerin yaklaşık üçte birini oluşturan embedding tablosunun quantize edilmemesidir. Dynamic quantization aktivasyon ölçeğini batch'e göre hesapladığından INT8 doğruluğu batch boyutuyla biraz değişebilir. Batching kazancı dolgu (padding) ve çekirdek sayısıyla sınırlıdır.

**Sonraki adım.** Cümleleri uzunluğa göre gruplayarak dolguyu azaltmak, ONNX Runtime ve statik quantization'ı denemek, 872 örneğin tamamıyla doğruluk belirsizliğini azaltmak ve eş zamanlı istek yükü altında p95'i ölçmek.
