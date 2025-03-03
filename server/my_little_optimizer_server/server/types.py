from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Literal, Tuple


class SweepSpaceItem(BaseModel):
    name: str
    min: float
    max: float
    log: bool = False

class SliceVisualizationDatapoint(BaseModel):
    x: float
    y_mean: float
    y_std: float
    params: Dict[str, float]

class SliceVisualization(BaseModel):
    sweep_id: str
    param_name: str
    computed_at: int
    data: List[SliceVisualizationDatapoint]

class Project(BaseModel):
    id: str
    name: str
    created_at: int

class SweepParamType(BaseModel):
    min: float
    max: float
    type: Literal['linear', 'log', 'logit']

    def todict(self):
        return {'min': self.min, 'max': self.max, 'type': self.type}