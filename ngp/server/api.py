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

# Database connection management
class Database:
    def __init__(self, db_path: str = 'sweeps.db'):
        self.db_path = db_path
        self._connection = None
        self._lock = threading.Lock()
    
    def _get_connection(self):
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
        return self._connection
    
    @contextmanager
    def get_cursor(self):
        with self._lock:  # Thread safety for SQLite
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                yield cursor
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                cursor.close()

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None

# Global database instance
db = Database()

# Database initialization
def init_db():
    with db.get_cursor() as c:
        c.execute('''
            CREATE TABLE IF NOT EXISTS sweeps (
                id TEXT PRIMARY KEY,
                name TEXT,
                parameters JSON,
                status TEXT,
                created_at TIMESTAMP
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                sweep_id TEXT,
                parameters JSON,
                value REAL,
                status TEXT,
                created_at TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (sweep_id) REFERENCES sweeps (id)
            )
        ''')

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield
    db.close()

PROXY = True
app = FastAPI(lifespan=lifespan, docs_url='/api/docs', redoc_url='/api/redoc', openapi_url='/api/openapi.json')


# Models
@dataclass
class SweepSpaceItem(BaseModel):
    name: str
    min: float
    max: float
    log: bool = False

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
    sweep_id = str(uuid.uuid4())
    
    with db.get_cursor() as c:
        c.execute(
            "INSERT INTO sweeps (id, name, parameters, status, created_at) VALUES (?, ?, ?, ?, ?)",
            (sweep_id, sweep.name, json.dumps(todict(sweep.parameters)), "active", datetime.now())
        )
    
    return {"id": sweep_id, "message": "Sweep created successfully"}

class SweepUpdate(BaseModel):
    name: str = None

@app.patch("/sweeps/{sweep_id}")
async def sweep_update(sweep_id: str, updates: SweepUpdate):
    with db.get_cursor() as c:
        if updates.name is not None:
            query = "UPDATE sweeps SET name = ? WHERE id = ?"
            c.execute(query, [updates.name, sweep_id])
            if c.rowcount == 0:
                raise HTTPException(status_code=404, detail="sweep not found")
    return {"message": "sweep updated successfully"}

@app.get("/api/sweeps/")
async def sweep_list():
    with db.get_cursor() as c:
        c.execute("SELECT id, name, parameters, status, created_at FROM sweeps")
        results = c.fetchall()
        return [{"id": id, "name": name, "parameters": json.loads(parameters), "status": status, "created_at": created_at} for (id, name, parameters, status, created_at) in results]

@app.get("/api/sweeps/{sweep_id}/ask")
async def sweep_ask(sweep_id: str, parameters: Dict[str, float] = None):
    raise NotImplementedError
    # with db.get_cursor() as c:
    #     # Get sweep parameters
    #     c.execute("SELECT parameters FROM sweeps WHERE id = ?", (sweep_id,))
    #     result = c.fetchone()
        
    #     if not result:
    #         raise HTTPException(status_code=404, detail="Sweep not found")
            
    #     params = json.loads(result[0])
        
    #     # Simple random sampling for now - could be replaced with more sophisticated methods
    #     import random
    #     experiment_params = {
    #         k: random.uniform(v["min"], v["max"]) 
    #         for k, v in params.items()
    #     }
        
    #     # Create new experiment
    #     experiment_id = str(uuid.uuid4())
    #     c.execute(
    #         """INSERT INTO experiments 
    #         (id, sweep_id, parameters, status, created_at) 
    #         VALUES (?, ?, ?, ?, ?)""",
    #         (experiment_id, sweep_id, json.dumps(experiment_params), "pending", datetime.now())
    #     )
        
    #     return {
    #         "experiment_id": experiment_id,
    #         "parameters": experiment_params
    #     }


class ExperimentCreate(BaseModel):
    sweep_id: str
    parameters: Dict[str, float]
    value: float = None

@app.post("/api/experiments/")
async def experiment_create(exp: ExperimentCreate):
    id = str(uuid.uuid4())
    created_at = int(time.time())
    if exp.value is None:
        status = 'started'
        completed_at = None
    else:
        status = 'complete'
        completed_at = created_at    

    with db.get_cursor() as c:
        c.execute(
            """INSERT INTO experiments (id, sweep_id, parameters, value, status, created_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (id, exp.sweep_id, json.dumps(exp.parameters), exp.value, status, created_at, completed_at))
    return {"id": id, "message": "Experiment created successfully"}

@app.post("/api/experiments/{experiment_id}/report")
async def report_result(experiment_id: str, result: float):
    completed_at = int(time.time())
    with db.get_cursor() as c:
        c.execute(
            """UPDATE experiments 
            SET value = ?, status = ?, completed_at = ?
            WHERE id = ?""",
            (result, "completed", completed_at, experiment_id)
        )
        return {"message": "Results recorded successfully"}

@app.get("/api/sweeps/{sweep_id}/results")
async def get_sweep_results(sweep_id: str):
    with db.get_cursor() as c:
        c.execute(
            """SELECT parameters, value, status, created_at, completed_at 
            FROM experiments WHERE sweep_id = ?""",
            (sweep_id,)
        )
        results = c.fetchall()
        
        return [{
            "parameters": json.loads(parameters),
            "value": value,
            "status": status,
            "created_at": created_at,
            "completed_at": completed_at
        } for (parameters, value, status, created_at, completed_at) in results]
