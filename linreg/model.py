import sklearn.linear_model as sk
import typing
from ..lib.scorer import ScorerMixin


class LinearRegression(ScorerMixin, sk.LinearRegression):
    def __init__(self, hyperparams: typing.Dict[str, typing.Any], metrics: typing.List[str]):
        super().__init__(**hyperparams)
        self._model = sk.LinearRegression(**hyperparams)
        self.hyperparams = hyperparams

        # TODO: better metrics logic
        self.metrics = metrics

    def get_merged_params(self, deep=True):
        return {**self.hyperparams, **self._model.get_params(deep)}
