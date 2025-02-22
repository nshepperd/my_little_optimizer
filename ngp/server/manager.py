from __future__ import annotations

from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import time
import uuid
from uuid import UUID
import sqlite3
from contextlib import contextmanager, asynccontextmanager
from threading import Thread
from queue import Queue
from dataclasses import dataclass, asdict
import json
import traceback

from ngp.server.database import Database
from ngp.optim import Optim, SpaceItem, Trial

# class Task:
#     id: str
#     created_at: float

# class UpdateChainsTask(Task):
#     sweep_id: str
#     def __init__(self, sweep_id: str):
#         self.sweep_id = sweep_id
#         self.id = f'update-chains-{sweep_id}'
#         self.created_at = time.time()

def todict(xs):
    if isinstance(xs, list):
        return [todict(x) for x in xs]
    else:
        return asdict(xs)

@dataclass
class SweepSpaceItem(BaseModel):
    name: str
    min: float
    max: float
    log: bool = False

@dataclass
class SweepInfo:
    id: str
    optim: Optim
    experiments: Dict[str, ExperimentInfo]
    updated_at: int = 0
    inferred_at: int = 0

@dataclass
class ExperimentInfo:
    id: str
    sweep_id: str
    parameters: Dict[str, float]
    value: float
    status: str
    created_at: int
    completed_at: Optional[int]

class SweepManager:
    sweeps: Dict[str, SweepInfo]
    db: Database
    job_queue: Queue
    job_thread: Thread
    tasks: Dict[UUID, Task]

    def __init__(self):
        self.db = Database()
        init_db(self.db)

        with self.db.get_cursor() as c:
            c.execute("SELECT id, parameters FROM sweeps")
            results = c.fetchall()
            self.sweeps = {id:
                SweepInfo(id=id, optim=Optim([SpaceItem(**p) for p in json.loads(parameters)]), experiments={})
                for (id, parameters) in results
            }

        self.job_queue = Queue()
        self.tasks = {}
        self.job_thread = Thread(target=self.job_worker, daemon=True)
        self.job_thread.start()

        # for sweep in self.sweeps.values():
        #     with self.db.get_cursor() as c:
        #         c.execute("SELECT id, parameters, value, status, created_at, completed_at FROM experiments WHERE sweep_id = ?", (sweep.id,))
        #         results = c.fetchall()
        #         sweep = self.sweeps[sweep.id]
        #         sweep.optim.trials = [Trial(params=json.loads(parameters), value=value) for (id, parameters, value, status, created_at, completed_at) in results if value is not None]
        #         sweep.experiments = {id: ExperimentInfo(id=id, sweep_id=sweep.id, parameters=json.loads(parameters), value=value, status=status, created_at=created_at, completed_at=completed_at)
        #                              for (id, parameters, value, status, created_at, completed_at) in results}
        #         if results:
        #             sweep.last_report = max([0] + [completed_at for (id, parameters, value, status, created_at, completed_at) in results if completed_at is not None])

    def job_worker(self):
        while True:
            job = self.job_queue.get()
            if job is None:
                break
            print('Running job', job.id)
            try:
                job()
            except Exception as e:
                job.status = 'error'
                job.message = f'{type(e)}: {str(e)}\n{traceback.format_exc()}'
            self.job_queue.task_done()

    def schedule(self, task: Task):
        id = uuid.uuid4()
        task.id = id
        self.tasks[id] = task
        self.job_queue.put(task)
        print('Scheduled task', id, task)
        return id

    def close(self):
        self.db.close()

    def create_sweep(self, name: str, parameters: List[SweepSpaceItem]):
        sweep_id = str(uuid.uuid4())

        with self.db.get_cursor() as c:
            c.execute(
                "INSERT INTO sweeps (id, name, parameters, status, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    sweep_id,
                    name,
                    json.dumps(todict(parameters)),
                    "active",
                    int(time.time()),
                ),
            )
        
        self.sweeps[sweep_id] = SweepInfo(id=sweep_id, optim=Optim([SpaceItem(**todict(p)) for p in parameters]), experiments={})

        return sweep_id


    def update_sweep(self, sweep_id: str, name: str = None):
        if name is not None:
            with self.db.get_cursor() as c:
                query = "UPDATE sweeps SET name = ? WHERE id = ?"
                c.execute(query, [name, sweep_id])

    def create_experiment(self, sweep_id: str, parameters: Dict[str, float], value: Optional[float]):
        id = str(uuid.uuid4())
        created_at = int(time.time())
        if value is None:
            status = "started"
            completed_at = None
        else:
            status = "complete"
            completed_at = created_at

        with self.db.get_cursor() as c:
            c.execute(
                """INSERT INTO experiments (id, sweep_id, parameters, value, status, created_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    id,
                    sweep_id,
                    json.dumps(parameters),
                    value,
                    status,
                    created_at,
                    completed_at,
                ),
            )
        
        # self.sweeps[sweep_id].experiments[id] = ExperimentInfo(id=id, sweep_id=sweep_id, parameters=parameters, value=value, status=status, created_at=created_at, completed_at=completed_at)
        return id

    def report_result(self, sweep_id: str, experiment_id: str, value: float):
        completed_at = int(time.time())
        with self.db.get_cursor() as c:
            c.execute(
                """UPDATE experiments 
                SET value = ?, status = ?, completed_at = ?
                WHERE sweep_id = ? AND id = ?""",
                (value, "completed", completed_at, sweep_id, experiment_id),
            )
        self.sweeps[sweep_id].updated_at = int(time.time())
        self.schedule(InferTask(self, sweep_id))
    
    def ask_params(self, sweep_id: str, params: Dict[str, float]) -> str:
        print('Asking for params, sweep_id:', sweep_id, 'params:', params)
        return self.schedule(AskTask(self, sweep_id, params))

class Task:
    status: str = 'pending'
    message: str = ''
    id: UUID = None
    result: Any = None

class InferTask(Task):
    def __init__(self, manager: SweepManager, sweep_id: str):
        self.manager = manager
        self.sweep_id = sweep_id
    
    def __call__(self):
        print('Starting task', self.id)
        print('Inferring sweep_id:', self.sweep_id)
        self.status = 'running'
        sweep = self.manager.sweeps[self.sweep_id]
        if sweep.inferred_at > sweep.updated_at:
            return
        with self.manager.db.get_cursor() as c:
            c.execute("SELECT id, parameters, value, status, created_at, completed_at FROM experiments WHERE sweep_id = ?", (sweep.id,))
            results = c.fetchall()
            sweep.optim.trials = [Trial(params=json.loads(parameters), value=value) for (id, parameters, value, status, created_at, completed_at) in results if status == 'completed']
        print('trials:', len(sweep.optim.trials))
        if len(sweep.optim.trials) > 0:
            sweep.optim.infer()
        sweep.inferred_at = int(time.time())
        self.status = 'done'

class AskTask:
    def __init__(self, manager: SweepManager, sweep_id: str, params: Dict[str, float]):
        self.manager = manager
        self.sweep_id = sweep_id
        self.params = params
        self.result = None
    
    def __call__(self):
        print('Starting task', self.id)
        self.status = 'running'
        sweep = self.manager.sweeps[self.sweep_id]
        if sweep.inferred_at < sweep.updated_at:
            InferTask(self.manager, self.sweep_id)()
        ps = sweep.optim.suggest(self.params)
        self.result = {k: float(v) for (k,v) in ps.items()}
        self.status = 'done'


def init_db(db: Database):
    with db.get_cursor() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS sweeps (
                id TEXT PRIMARY KEY,
                name TEXT,
                parameters JSON,
                status TEXT,
                created_at TIMESTAMP
            )
        """
        )
        c.execute(
            """
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
        """
        )