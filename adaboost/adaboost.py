import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

class AdaBoost:
    def __init__(self, M, N):
        super().__init__()
        self.classifiers = []
        self.alphas = np.random.rand(M)
        self.sample_weights = np.ones(N) * (1/N)
        self.M = M
        self.N = N
        self.errors = []
        
    def step_error(self, y, pred, w_i):
        return np.sum(w_i * (np.not_equal(y, pred)).astype(int)) / np.sum(w_i)
    
    def update_sample_weights(self, y, pred, alpha):
        self.sample_weights = self.sample_weights * np.exp(alpha * np.not_equal(y, pred))
    
    def fit(self, x, y):
        for i in range(self.M):
            model = DecisionTreeClassifier(random_state=0, max_depth=2)
            model.fit(x, y, sample_weight=self.sample_weights)
            self.classifiers.append(model)
            
            preds = model.predict(x)
            err_m = self.step_error(y, preds, self.sample_weights)
            self.errors.append(err_m)
            self.alphas[i] = np.log((1-err_m)/err_m)
            
            self.update_sample_weights(y, preds, self.alphas[i])
            
        assert len(self.alphas) == len(self.classifiers)
        
    def __call__(self, x):
        preds = np.array([np.array([0 for _ in range(self.M)]) for _ in range(len(x))])
        for i in range(self.M):
            cur_pred = self.alphas[i] * self.classifiers[i].predict(x)
            preds[:, i] = cur_pred
        
        final_pred = np.sign(preds.T.sum(0)).astype(int)
        return final_pred