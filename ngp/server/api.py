from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Optional
import sqlite3
import json
from datetime import datetime
import uuid
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


def todict(xs):
    if isinstance(xs, list):
        return [todict(x) for x in xs]
    else:
        return asdict(xs)


# API Routes
@app.post("/api/sweeps/")
async def sweep_create(sweep: SweepCreate):
    sweep_id = manager.create_sweep(sweep.name, sweep.parameters)
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
async def sweep_list():
    with manager.db.get_cursor() as c:
        c.execute("SELECT id, name, parameters, status, created_at FROM sweeps")
        results = c.fetchall()
        return {'status': 'success',
                'sweeps': [
            {
                "id": id,
                "name": name,
                "parameters": json.loads(parameters),
                "status": status,
                "created_at": created_at,
            }
            for (id, name, parameters, status, created_at) in results
        ]}


@app.post("/api/sweeps/{sweep_id}/ask")
async def sweep_ask(sweep_id: str, parameters: Dict[str, float] = None):
    job_id = manager.ask_params(sweep_id, parameters)
    manager.job_queue.join()
    task = manager.tasks[job_id]
    if task.status == 'done':
        return {'status': 'success', 'params': task.result}
    else:
        raise HTTPException(status_code=500, detail={'status': task.status, 'message': task.message})

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


@app.get("/api/sweeps/{sweep_id}/results")
async def get_sweep_results(sweep_id: str):
    sweep = manager.sweeps[sweep_id]
    results = []
    for experiment in sweep.experiments.values():
        results.append({
            "parameters": experiment.parameters,
            "value": experiment.value,
            "status": experiment.status,
            "created_at": experiment.created_at,
            "completed_at": experiment.completed_at
        })
    return {'status': 'success', 'results': results}
