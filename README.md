# ABC Algorithm: Canonical Artificial Bee Colony Optimization Library

Bu kütüphane, **Yapay Arı Kolonisi (Artificial Bee Colony - ABC)** algoritmasının Derviş Karaboğa (2005) tarafından önerilen **%100 Kanonik (Standart)** versiyonunu içeren bir Python paketidir.

Hem saf matematiksel optimizasyon problemleri için bir çözücü (`solver`) hem de Makine Öğrenmesi modelleri için **Scikit-Learn uyumlu** bir hiperparametre optimize edici (`tuner`) içerir.

## 🚀 Özellikler

* **Kanonik İmplementasyon:** Orijinal makaledeki İşçi, Gözcü (Rulet Tekerleği Seçimi) ve Kaşif arı fazlarına sadık kalınmıştır.
* **Scikit-Learn Uyumu:** `GridSearchCV` veya `RandomizedSearchCV` kullanır gibi modelinizi optimize edebilirsiniz.
* **Esnek Yapı:** Her türlü matematiksel fonksiyonu minimize edebilir.
* **Modüler:** Algoritma çekirdeği ve ML arayüzü birbirinden bağımsızdır.

---

## 📦 Kurulum

Bu kütüphaneyi kaynak kodundan kurmak için terminali proje dizininde açın ve aşağıdaki komutu çalıştırın:

```bash
pip install -e .