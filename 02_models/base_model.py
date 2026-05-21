# 02_models/base_model.py
"""Abstract base class for all SMB models."""
from abc import ABC, abstractmethod

class BaseSMBModel(ABC):
    @abstractmethod
    def fit(self, X_train, y_train): ...

    @abstractmethod
    def predict(self, X): ...

    @property
    @abstractmethod
    def name(self) -> str: ...
