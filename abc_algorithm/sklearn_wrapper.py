import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import cross_val_score
from .core import CanonicalABCSolver

class ABCSearchCV(BaseEstimator):
    """
    Scikit-Learn uyumlu ABC Hiperparametre Optimizasyon Sınıfı.
    """
    def __init__(self, estimator, param_space, cv=3, scoring='accuracy', 
                 pop_size=20, max_evals=200, limit=None, verbose=True):
        self.estimator = estimator
        self.param_space = param_space
        self.cv = cv
        self.scoring = scoring
        self.pop_size = pop_size
        self.max_evals = max_evals
        self.limit = limit
        self.verbose = verbose
        
    def fit(self, X, y):
        # Parametre sözlüğünü (dict) vektöre (list) çevirme mantığı
        keys = list(self.param_space.keys())
        lb = [self.param_space[k]['range'][0] for k in keys]
        ub = [self.param_space[k]['range'][1] for k in keys]

        def objective(vector):
            params = {}
            for i, key in enumerate(keys):
                val = vector[i]
                p_type = self.param_space[key]['type']
                if p_type == 'int': params[key] = int(round(val))
                else: params[key] = val
            
            model = clone(self.estimator)
            model.set_params(**params)
            
            try:
                scores = cross_val_score(model, X, y, cv=self.cv, scoring=self.scoring)
                score = np.mean(scores)
                # Minimize edilecek formata çevir (Hata = 1 - Başarı)
                return -score if 'neg_' in self.scoring else 1 - score
            except:
                return float('inf')

        solver = CanonicalABCSolver(objective, len(keys), lb, ub, self.pop_size, self.max_evals, self.limit)
        
        if self.verbose: print("ABC Optimizasyonu başladı...")
        best_vec, min_cost, history = solver.solve()

        # Skoru (Cost'tan geri dönüştürerek) hesapla ve kaydet
        if 'neg_' in self.scoring:
            self.best_score_ = -min_cost
        else:
            self.best_score_ = 1 - min_cost
        
        # En iyi sonuçları kaydet
        self.best_params_ = {}
        for i, key in enumerate(keys):
            val = best_vec[i]
            if self.param_space[key]['type'] == 'int': self.best_params_[key] = int(round(val))
            else: self.best_params_[key] = val
            
        self.best_estimator_ = clone(self.estimator)
        self.best_estimator_.set_params(**self.best_params_)
        self.best_estimator_.fit(X, y)
        self.history_ = history
        
        if self.verbose: print(f"Tamamlandı. Parametreler: {self.best_params_}")
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)
        
    def score(self, X, y):
        return self.best_estimator_.score(X, y)