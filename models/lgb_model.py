import lightgbm as lgbm
from .base_model import BaseModel


class LGBModel(BaseModel):

    def __init__(self, params):
        super().__init__()
        self.model = lgbm.LGBMRegressor(**params)

    def fit(self, X_train, y_train, X_test, y_test):
        self.model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])

    def plot_metric(self):
        return lgbm.plot_metric(self.model)

    def plot_importance(self):
        return lgbm.plot_importance(self.model)
