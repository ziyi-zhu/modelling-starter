from abc import ABC, abstractmethod


class BaseModel(ABC):

    @abstractmethod
    def fit(self, X_train, y_train):
        pass
