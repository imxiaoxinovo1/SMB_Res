# 02_models/xgboost/model.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import xgboost as xgb
from base_model import BaseSMBModel
from registry import register

@register('xgboost')
class XGBoostSMB(BaseSMBModel):
    name = 'xgboost'

    def __init__(self, **kwargs):
        from config import XGB_PARAMS
        params = {**XGB_PARAMS, **kwargs}
        self.model = xgb.XGBRegressor(**params)
        self.feature_names_ = None

    def fit(self, X_train, y_train, feature_names=None):
        self.feature_names_ = feature_names
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def feature_importance(self):
        return dict(zip(self.feature_names_ or [], self.model.feature_importances_))
