from enum import Enum

import torch


class CriterionSorting(Enum):
    DESCENDING = 'DESCENDING'
    GROWING = 'GROWING'


class CriterionType(Enum):
    MIN = 'MIN'
    MAX = 'MAX'
    AVG = 'AVG'


class Criterion:
    def __init__(self, criterion_type: CriterionType):
        self._criterion_type: CriterionType = criterion_type

    def aggregate_score_image(self, scores) -> float:
        if self._criterion_type == CriterionType.MIN:
            return 0 if len(scores) == 0 else torch.min(scores).item()
        elif self._criterion_type == CriterionType.MAX:
            return 0 if len(scores) == 0 else torch.max(scores).item()
