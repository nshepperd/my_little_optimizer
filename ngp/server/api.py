from __future__ import annotations

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Optional, Literal, TypeVar, Generic, Any
import sqlite3
import json
from datetime import datetime
from uuid import UUID
from contextlib import contextmanager, asynccontextmanager
import threading
from dataclasses import dataclass, asdict
import time

from ngp.server.manager import SweepManager
from ngp.server.types import SweepSpaceItem, SliceVisualization, SliceVisualizationDatapoint


manager: SweepManager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager
    manager = SweepManager()
    yield
    manager.close()


PROXY = True
app = FastAPI(
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)


# Models

T = TypeVar('T')

class ApiResponse(BaseModel, Generic[T]):
    status: Literal['ok', 'error']
    data: Optional[T] = None
    message: Optional[str] = None
    code: Optional[str] = None
    
    @classmethod
    def ok(cls, data: T = None) -> ApiResponse[T]:
        return cls(status='ok', data=data)
    
    @classmethod
    def error(cls, message: str, code: str = None) -> ApiResponse[T]:
        return cls(status='error', message=message, code=code)

class SweepCreate(BaseModel):
    name: str
    parameters: List[SweepSpaceItem]
    objective: Literal['min', 'max'] = 'min'

class SweepParamType(BaseModel):
    min: float
    max: float
    log: bool

class SweepGetResponse(BaseModel):
    id: str
    name: str
    parameters: Dict[str, SweepParamType]
    status: str
    created_at: int
    objective: Literal['min', 'max']
    num_trials: int

class SweepGetTrialsResponse(BaseModel):
    id: str
    parameters: Dict[str, float]
    value: Optional[float]
    status: str
    created_at: int
    completed_at: Optional[int]
    trial_number: int

class SliceVisualizationRequest(BaseModel):
    param_name: str
class SliceVisualizationResponse(BaseModel):
    cached: Optional[SliceVisualization]
    job_id: Optional[UUID]

def todict(xs):
    if isinstance(xs, list):
        return [todict(x) for x in xs]
    else:
        return asdict(xs)


# API Routes


@app.post("/api/sweeps/")
async def sweep_create(sweep: SweepCreate) -> ApiResponse[str]:
    sweep_id = manager.create_sweep(sweep.name, sweep.parameters, sweep.objective)
    return ApiResponse.ok(sweep_id)


class SweepUpdate(BaseModel):
    name: str = None


@app.patch("/api/sweeps/{sweep_id}")
async def sweep_update(sweep_id: str, updates: SweepUpdate) -> ApiResponse[None]:
    manager.update_sweep(sweep_id, updates.name)
    return ApiResponse.ok()


@app.delete("/api/sweeps/{sweep_id}")
async def sweep_delete(sweep_id: str) -> ApiResponse[None]:
    with manager.db.get_cursor() as c:
        query = "DELETE FROM sweeps WHERE id = ?"
        c.execute(query, [sweep_id])
        if c.rowcount == 0:
            return ApiResponse.error("sweep not found")
        query = "DELETE FROM trials WHERE sweep_id = ?"
        c.execute(query, [sweep_id])
    del manager.sweeps[sweep_id]
    return ApiResponse.ok()


@app.get("/api/sweeps/")
async def sweep_list() -> ApiResponse[List[SweepGetResponse]]:
    with manager.db.get_cursor() as c:
        c.execute("SELECT id, name, parameters, status, created_at, objective, num_trials FROM sweeps")
        results = c.fetchall()
        res = []
        for r in results:
            res.append(SweepGetResponse(
                id=r['id'],
                name=r['name'],
                parameters={p['name']: SweepParamType(min=p['min'], max=p['max'], log=p['log']) for p in json.loads(r['parameters'])},
                status=r['status'],
                created_at=r['created_at'],
                objective=r['objective'],
                num_trials=r['num_trials'],
            ))
        return ApiResponse.ok(res)

@app.post("/api/poll/{job_id}")
async def poll_job(job_id: UUID) -> ApiResponse[Dict[str, Any]]:
    if job_id not in manager.tasks:
        return ApiResponse.error("job not found", code='not_found')
    
    status = manager.tasks[job_id].status
    if status == 'done':
        return ApiResponse.ok({'status': 'done', 'result': manager.tasks[job_id].result})
    elif status == 'error':
        return ApiResponse.ok({'status': 'error', 'message': manager.tasks[job_id].message})
    else:
        return ApiResponse.ok({'status': status})


@app.post("/api/sweeps/{sweep_id}/ask")
async def sweep_ask(sweep_id: str, parameters: Dict[str, float] = None) -> ApiResponse[UUID]:
    job_id = manager.ask_params(sweep_id, parameters)
    return ApiResponse.ok(job_id)

@app.post("/api/sweeps/{sweep_id}/visualize/slice")
async def sweep_get_slice_visualization(sweep_id: str, options: SliceVisualizationRequest) -> ApiResponse[SliceVisualizationResponse]:
    with manager.db.get_cursor() as c:
        c.execute("SELECT id, num_trials, objective FROM sweeps WHERE id = ?", (sweep_id,))
        results = c.fetchall()
        if len(results) == 0:
            return ApiResponse.error("sweep not found", code='not_found')
        if results[0]['num_trials'] == 0:
            return ApiResponse.error("sweep has no trials", code='no_data')
    res = manager.get_slice_visualization(sweep_id, options.param_name)
    return ApiResponse.ok(SliceVisualizationResponse(**res))


class TrialCreate(BaseModel):
    parameters: Dict[str, float]
    value: float = None


@app.post("/api/sweeps/{sweep_id}/trials/")
async def trial_create(sweep_id: str, exp: TrialCreate) -> ApiResponse[str]:
    id = manager.create_trial(sweep_id, exp.parameters, exp.value)
    return ApiResponse.ok(id)

@app.post("/api/sweeps/{sweep_id}/trials/{trial_id}/report")
async def report_result(sweep_id: str, trial_id: str, result: Dict[str, float]) -> ApiResponse[None]:
    manager.report_result(sweep_id, trial_id, result['value'])
    return ApiResponse.ok()

@app.get("/api/sweeps/{sweep_id}/trials/")
async def sweep_get_trials(sweep_id: str) -> ApiResponse[List[SweepGetTrialsResponse]]:
    with manager.db.get_cursor() as c:
        c.execute("SELECT id, parameters, value, status, created_at, completed_at, trial_number FROM trials WHERE sweep_id = ?", (sweep_id,))
        results = c.fetchall()
        res = []
        for r in results:
            res.append(SweepGetTrialsResponse(
                id=r['id'],
                parameters=json.loads(r['parameters']),
                value=r['value'],
                status=r['status'],
                created_at=r['created_at'],
                completed_at=r['completed_at'],
                trial_number=r['trial_number'],
            ))
        return ApiResponse.ok(res)

@app.get("/api/sweeps/{sweep_id}")
async def sweep_get(sweep_id: str) -> ApiResponse[SweepGetResponse]:
    with manager.db.get_cursor() as c:
        c.execute("SELECT id, name, parameters, status, created_at, num_trials, objective FROM sweeps WHERE id = ?", (sweep_id,))
        if c.rowcount == 0:
            return ApiResponse.error("sweep not found", code='not_found')
        results = c.fetchall()
        return ApiResponse.ok(SweepGetResponse(
            id=results[0]['id'],
            name=results[0]['name'],
            parameters={p['name']: SweepParamType(min=p['min'], max=p['max'], log=p['log']) for p in json.loads(results[0]['parameters'])},
            status=results[0]['status'],
            created_at=results[0]['created_at'],
            objective=results[0]['objective'],
            num_trials=results[0]['num_trials'],
        ))