import pickle
from abc import ABC, abstractmethod
from catboost import CatBoostRegressor
from sklearn.multioutput import MultiOutputRegressor


class BaseModel(ABC):
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def save_model(self):
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self):
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)


class CatBoostModel(BaseModel):
    def __init__(self, model_path, params=None):
        super().__init__(model_path)
        if params is None:
            params = {
                'iterations': 50,
                'learning_rate': 0.1,
                'depth': 6,
                'loss_function': 'RMSE',
                'eval_metric': 'RMSE',
                'random_seed': 42,
                'verbose': 50
            }
        self.params = params
        self.model = MultiOutputRegressor(CatBoostRegressor(**self.params))

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self):
        for i, estimator in enumerate(self.model.estimators_):
            estimator.save_model(f"{self.model_path}_estimator_{i}")

    def load_model(self):
        for i, estimator in enumerate(self.model.estimators_):
            estimator.load_model(f"{self.model_path}_estimator_{i}")