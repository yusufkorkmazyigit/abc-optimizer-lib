# 🐝 ABC Optimizer Lib: Canonical Artificial Bee Colony Algorithm

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Scikit-Learn](https://img.shields.io/badge/sklearn-compatible-orange)

**ABC Optimizer Lib**, Derviş Karaboğa (2005) tarafından önerilen **Yapay Arı Kolonisi (Artificial Bee Colony)** algoritmasının %100 kanonik (standart) implementasyonunu içeren bir Python kütüphanesidir.

Bu kütüphane iki temel amaç için geliştirilmiştir:
1.  **Matematiksel Optimizasyon:** Herhangi bir fonksiyonun minimum noktasını bulmak.
2.  **Hyperparameter Tuning:** Scikit-Learn uyumlu (LightGBM, XGBoost, vb.) modellerin hiperparametrelerini optimize etmek.

---

## 🚀 Özellikler

* **Scikit-Learn Wrapper:** `GridSearchCV` mantığıyla çalışır. `fit()` ve `predict()` metodlarını destekler.
* **Kanonik Algoritma:** Literatürdeki orijinal İşçi, Gözcü (Rulet Tekerleği) ve Kaşif arı fazlarına sadık kalınmıştır.
* **Hafif ve Hızlı:** Sadece `numpy` ve `scikit-learn` bağımlılığı vardır.
* **Esnek:** Hem sürekli (float) hem ayrık (int/categorical) parametre uzaylarını destekler.

---

## 📦 Kurulum

Bu kütüphaneyi doğrudan GitHub üzerinden `pip` ile kurabilirsiniz:

```bash
pip install git+[https://github.com/yusufkorkmazyigit/abc-optimizer-lib.git](https://github.com/yusufkorkmazyigit/abc-optimizer-lib.git)
```
Geliştirme yapmak (kodu değiştirmek) isterseniz:

```
git clone [https://github.com/yusufkorkmazyigit/abc-optimizer-lib.git](https://github.com/yusufkorkmazyigit/abc-optimizer-lib.git)
cd abc-optimizer-lib
pip install -e .
```
## 📖 Kullanım Örnekleri
1. LightGBM Hiperparametre Optimizasyonu
Makine öğrenmesi modellerinizde en iyi parametreleri bulmak için `ABCSearchCV` sınıfını kullanın:

```
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from abc_algorithm import ABCSearchCV

# Veri ve Model
data = load_breast_cancer()
X, y = data.data, data.target
model = lgb.LGBMClassifier(verbosity=-1)

# Arama Uzayı
param_space = {
    'learning_rate': {'type': 'float', 'range': (0.01, 0.3)},
    'n_estimators':  {'type': 'int',   'range': (50, 500)},
    'num_leaves':    {'type': 'int',   'range': (20, 100)}
}

# Optimizasyon
abc = ABCSearchCV(
    estimator=model,
    param_space=param_space,
    cv=3,
    scoring='accuracy',
    pop_size=20,    # Koloni boyutu
    max_evals=100   # Toplam deneme sayısı
)

abc.fit(X, y)

print("En iyi skor:", abc.best_score_)
print("En iyi parametreler:", abc.best_params_)
```
2. Matematiksel Fonksiyon Minimizasyonu
Sadece bir denklemi çözmek isterseniz `CanonicalABCSolver` kullanın:

```
from abc_algorithm import CanonicalABCSolver

# Hedef: Sphere Fonksiyonu (x^2 toplamı 0 olmalı)
def objective(x):
    return sum(x**2)

solver = CanonicalABCSolver(
    objective_func=objective,
    n_params=3,
    lb=[-10, -10, -10],
    ub=[10, 10, 10],
    max_evals=500
)

best_params, best_cost, _ = solver.solve()
print(f"Sonuç: {best_params}, Maliyet: {best_cost:.5f}")
```
## 🧠 Algoritma Mantığı
ABC algoritması, doğadaki arıların yiyecek arama davranışlarını taklit eder ve üç fazdan oluşur:

**İşçi Arılar (Employed Bees):** Mevcut bir kaynağı (çözümü) komşuluk araştırması ile geliştirmeye çalışır.

**Gözcü Arılar (Onlooker Bees):** İşçi arıların getirdiği nektar bilgisine (fitness) göre Rulet Tekerleği yöntemiyle seçim yapar. İyi kaynaklar daha çok araştırılır.

**Kaşif Arılar (Scout Bees):** Belirli bir süre geliştirilemeyen (`limit`) kaynaklar terk edilir ve rastgele yeni bir çözüm aranır.

---

## 🔬 Gerçek Hayat Uygulaması: Federated Learning Optimizasyonu

Bu kütüphane kullanılarak, **MedMNIST** veriseti üzerinde **Non-IID (Dengesiz) Veri** dağılımına sahip bir **Federated Learning** mimarisi optimize edilmiştir.

**Senaryo:**
* **Veri Seti:** PathMNIST (Bağırsak dokusu sınıflandırma).
* **Problem:** 5 farklı hastaneye (istemciye) dengesiz dağıtılmış veri. Standart `FedAvg` algoritması bu durumda zorlanmaktadır.
* **Çözüm:** `CanonicalABCSolver` kullanılarak Learning Rate ve Momentum parametreleri optimize edilmiştir.

**Sonuçlar:**
ABC ile optimize edilmiş model, standart parametrelere göre daha hızlı yakınsamış ve **%7 daha yüksek doğruluk** elde etmiştir.

![ABC vs Standard FedAvg](./examples/abc_fedavg_final_result.png)

🔗 **[Tüm kodu ve detaylı analizi incelemek için tıklayın](./examples/Federated_Learning_MedMNIST_Optimization.ipynb)**

---

## 📚 Referans
Karaboga, D. (2005). An idea based on honey bee swarm for numerical optimization. Technical report-tr06, Erciyes University, engineering faculty, computer engineering department.

## 📝 Lisans
Bu proje MIT Lisansı ile sunulmuştur.
