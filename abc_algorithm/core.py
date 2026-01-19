import numpy as np

class CanonicalABCSolver:
    """
    Kanonik Yapay Arı Kolonisi (Artificial Bee Colony - ABC) Algoritması.
    D. Karaboğa (2005) standartlarına uygun saf implementasyon.
    """
    def __init__(self, objective_func, n_params, lb, ub, pop_size=20, max_evals=1000, limit=None):
        self.objective_func = objective_func
        self.D = n_params
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.SN = pop_size  # Kaynak sayısı (Food Number)
        self.total_evaluation = max_evals
        self.limit = limit if limit else self.SN * self.D
        
        # Geçmiş kayıtları
        self.best_params = None
        self.best_cost = float('inf')
        self.history = [] 

    def _calculate_fitness(self, cost):
        """Maliyetten (Cost) Uygunluk (Fitness) değerine dönüşüm."""
        return 1 / (1 + cost) if cost >= 0 else 1 + abs(cost)

    def _perturb(self, current, neighbour):
        """Kanonik kural: Sadece 1 parametreyi değiştir."""
        j = np.random.randint(0, self.D)
        phi = np.random.uniform(-1, 1)
        new_sol = np.copy(current)
        new_sol[j] = current[j] + phi * (current[j] - neighbour[j])
        new_sol[j] = np.clip(new_sol[j], self.lb[j], self.ub[j])
        return new_sol

    def solve(self):
        # Başlangıç
        foods = self.lb + np.random.rand(self.SN, self.D) * (self.ub - self.lb)
        costs = np.array([self.objective_func(x) for x in foods])
        fitness = np.array([self._calculate_fitness(c) for c in costs])
        trial = np.zeros(self.SN)
        eval_count = self.SN

        # İlk en iyiyi kaydet
        best_idx = np.argmin(costs)
        self.best_cost = costs[best_idx]
        self.best_params = np.copy(foods[best_idx])

        while eval_count < self.total_evaluation:
            # --- 1. İşçi Arılar ---
            for i in range(self.SN):
                if eval_count >= self.total_evaluation: break
                idx = i
                k = np.random.randint(0, self.SN)
                while k == idx: k = np.random.randint(0, self.SN)
                
                new_sol = self._perturb(foods[idx], foods[k])
                new_cost = self.objective_func(new_sol)
                eval_count += 1
                
                if self._calculate_fitness(new_cost) > fitness[idx]:
                    foods[idx] = new_sol
                    costs[idx] = new_cost
                    fitness[idx] = self._calculate_fitness(new_cost)
                    trial[idx] = 0
                else:
                    trial[idx] += 1

            # --- 2. Gözcü Arılar (Rulet Tekerleği) ---
            total_fit = np.sum(fitness)
            prob = fitness / total_fit if total_fit > 0 else np.ones(self.SN)/self.SN
            
            t, i = 0, 0
            while t < self.SN:
                if eval_count >= self.total_evaluation: break
                if np.random.rand() < prob[i]:
                    t += 1
                    k = np.random.randint(0, self.SN)
                    while k == i: k = np.random.randint(0, self.SN)
                    
                    new_sol = self._perturb(foods[i], foods[k])
                    new_cost = self.objective_func(new_sol)
                    eval_count += 1
                    
                    if self._calculate_fitness(new_cost) > fitness[i]:
                        foods[i] = new_sol
                        costs[i] = new_cost
                        fitness[i] = self._calculate_fitness(new_cost)
                        trial[i] = 0
                    else:
                        trial[i] += 1
                i = (i + 1) % self.SN

            # Global En İyiyi Güncelle
            min_cost_idx = np.argmin(costs)
            if costs[min_cost_idx] < self.best_cost:
                self.best_cost = costs[min_cost_idx]
                self.best_params = np.copy(foods[min_cost_idx])
            
            self.history.append((eval_count, self.best_cost))

            # --- 3. Kaşif Arılar ---
            max_trial_idx = np.argmax(trial)
            if trial[max_trial_idx] > self.limit and eval_count < self.total_evaluation:
                foods[max_trial_idx] = self.lb + np.random.rand(self.D) * (self.ub - self.lb)
                costs[max_trial_idx] = self.objective_func(foods[max_trial_idx])
                fitness[max_trial_idx] = self._calculate_fitness(costs[max_trial_idx])
                trial[max_trial_idx] = 0
                eval_count += 1

        return self.best_params, self.best_cost, self.history