import sklearn.linear_model as sk
from sklearn.model_selection import cross_validate
from sklearn.metrics import get_scorer
import typing
from lib.scorer import ScorerMixin


class LogRegression(ScorerMixin, sk.LogisticRegression):
    def __init__(self, hyperparams: typing.Dict[str, typing.Any], metrics: typing.List[str]):
        super().__init__(**hyperparams)
        self.hyperparams = hyperparams
        self.metrics = metrics

        # TODO: better metrics logic
        self.metrics = metrics
