from __future__ import annotations

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Optional, Literal
import sqlite3
import json
from datetime import datetime
from uuid import UUID
from contextlib import contextmanager, asynccontextmanager
import threading
from dataclasses import dataclass, asdict
import time

from ngp.server.manager import SweepManager, SweepSpaceItem


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

def todict(xs):
    if isinstance(xs, list):
        return [todict(x) for x in xs]
    else:
        return asdict(xs)


# API Routes
@app.post("/api/sweeps/")
async def sweep_create(sweep: SweepCreate):
    sweep_id = manager.create_sweep(sweep.name, sweep.parameters, sweep.objective)
    return {'status': 'success', 'id': sweep_id}


class SweepUpdate(BaseModel):
    name: str = None


@app.patch("/api/sweeps/{sweep_id}")
async def sweep_update(sweep_id: str, updates: SweepUpdate):
    manager.update_sweep(sweep_id, updates.name)
    return {'status': 'success'}


@app.delete("/api/sweeps/{sweep_id}")
async def sweep_delete(sweep_id: str):
    with manager.db.get_cursor() as c:
        query = "DELETE FROM sweeps WHERE id = ?"
        c.execute(query, [sweep_id])
        if c.rowcount == 0:
            raise HTTPException(status_code=404, detail="sweep not found")
        query = "DELETE FROM experiments WHERE sweep_id = ?"
        c.execute(query, [sweep_id])
    del manager.sweeps[sweep_id]
    return {'status': 'success'}


@app.get("/api/sweeps/")
async def sweep_list() -> List[SweepGetResponse]:
    with manager.db.get_cursor() as c:
        c.execute("SELECT id, name, parameters, status, created_at FROM sweeps")
        results = c.fetchall()
        res = []
        for r in results:
            res.append(SweepGetResponse(
                id=r['id'],
                name=r['name'],
                parameters={p['name']: SweepParamType(min=p['min'], max=p['max'], log=p['log']) for p in json.loads(r['parameters'])},
                status=r['status'],
                created_at=r['created_at'],
            ))
        return res
        # r['parameters'] = [SpaceItem(**p) for p in json.loads(parameters)]
        # return [SweepGetResponse(**r) for r in results]

@app.post("/api/poll/{job_id}")
async def poll_job(job_id: UUID):
    if job_id not in manager.tasks:
        raise HTTPException(status_code=404, detail="job not found")
    
    status = manager.tasks[job_id].status
    if status == 'pending':
        return {'status': 'pending'}
    elif status == 'running':
        return {'status': 'running'}
    elif status == 'done':
        return {'status': 'success', 'result': manager.tasks[job_id].result}
    elif status == 'error':
        return {'status': 'error', 'message': manager.tasks[job_id].message}
    else:
        raise HTTPException(status_code=500, detail={'status': status, 'message': 'unknown status'})


@app.post("/api/sweeps/{sweep_id}/ask")
async def sweep_ask(sweep_id: str, parameters: Dict[str, float] = None):
    job_id = manager.ask_params(sweep_id, parameters)
    return {'status': 'pending', 'job_id': job_id}
    # manager.job_queue.join()
    # task = manager.tasks[job_id]
    # if task.status == 'done':
    #     return {'status': 'success', 'params': task.result}
    # else:
    #     raise HTTPException(status_code=500, detail={'status': task.status, 'message': task.message})

class ExperimentCreate(BaseModel):
    parameters: Dict[str, float]
    value: float = None


@app.post("/api/sweeps/{sweep_id}/experiments/")
async def experiment_create(sweep_id: str, exp: ExperimentCreate):
    id = manager.create_experiment(sweep_id, exp.parameters, exp.value)
    return {'status': 'success', 'id': id}

@app.post("/api/sweeps/{sweep_id}/experiments/{experiment_id}/report")
async def report_result(sweep_id: str, experiment_id: str, result: Dict[str, float]):
    manager.report_result(sweep_id, experiment_id, result['value'])
    return {'status': 'success'}

class SweepGetExperimentsResponse(BaseModel):
    id: str
    parameters: Dict[str, float]
    value: float
    status: str
    created_at: int
    completed_at: Optional[int]

@app.get("/api/sweeps/{sweep_id}/experiments/")
async def sweep_get_experiments(sweep_id: str) -> List[SweepGetExperimentsResponse]:
    with manager.db.get_cursor() as c:
        c.execute("SELECT id, parameters, value, status, created_at, completed_at FROM experiments WHERE sweep_id = ?", (sweep_id,))
        results = c.fetchall()
        res = []
        for r in results:
            res.append(SweepGetExperimentsResponse(
                id=r['id'],
                parameters=json.loads(r['parameters']),
                value=r['value'],
                status=r['status'],
                created_at=r['created_at'],
                completed_at=r['completed_at'],
            ))
        return res

@app.get("/api/sweeps/{sweep_id}")
async def sweep_get() -> SweepGetResponse:
    with manager.db.get_cursor() as c:
        c.execute("SELECT id, name, parameters, status, created_at FROM sweeps WHERE id = ?", (sweep_id,))
        if c.rowcount == 0:
            raise HTTPException(status_code=404, detail="sweep not found")
        results = c.fetchall()
        return SweepGetResponse(
            id=results[0]['id'],
            name=results[0]['name'],
            parameters={p['name']: SweepParamType(min=p['min'], max=p['max'], log=p['log']) for p in json.loads(results[0]['parameters'])},
            status=results[0]['status'],
            created_at=results[0]['created_at'],
        )