from sklearn.model_selection import cross_validate
from sklearn.metrics import get_scorer
import typing


class ScorerMixin:
    def cv_scores(self, X, y, cv: int = 5) -> typing.Dict[str, typing.List[float]]:
        return cross_validate(self, X, y, scoring=self.metrics, cv=cv)

    def scores(self, X, y) -> typing.Dict[str, float]:
        return {scorer_name: get_scorer(scorer_name)(self, X, y) for scorer_name in self.metrics}
