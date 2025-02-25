from __future__ import annotations

from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Literal, Tuple
import time
import uuid
from uuid import UUID
from contextlib import contextmanager, asynccontextmanager
from threading import Thread
from queue import Queue
from dataclasses import dataclass, asdict
import json
import traceback
from cachetools import TTLCache
import jax

from my_little_optimizer.server.database import Database
from my_little_optimizer.optim import Optim, SpaceItem, Trial
from my_little_optimizer.server.types import SweepSpaceItem, SliceVisualization, SliceVisualizationDatapoint

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
    elif isinstance(xs, BaseModel):
        return xs.__dict__
    else:
        return asdict(xs)

@dataclass
class SweepInfo:
    id: str
    optim: Optim
    trials: Dict[str, TrialInfo]
    updated_at: int = 0
    inferred_at: int = 0

@dataclass
class TrialInfo:
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
    viz_cache: TTLCache

    def __init__(self):
        self.db = Database()
        init_db(self.db)

        with self.db.get_cursor() as c:
            c.execute("SELECT id, parameters, objective FROM sweeps")
            results = c.fetchall()
            self.sweeps = {id:
                SweepInfo(id=id, optim=Optim([SpaceItem(**p) for p in json.loads(parameters)], objective=objective), trials={})
                for (id, parameters, objective) in results
            }

        self.viz_cache = TTLCache(maxsize=100, ttl=60*60)
        self.job_queue = Queue()
        self.tasks = {}
        self.job_thread = Thread(target=self.job_worker, daemon=True)
        self.job_thread.start()

        # for sweep in self.sweeps.values():
        #     with self.db.get_cursor() as c:
        #         c.execute("SELECT id, parameters, value, status, created_at, completed_at FROM trials WHERE sweep_id = ?", (sweep.id,))
        #         results = c.fetchall()
        #         sweep = self.sweeps[sweep.id]
        #         sweep.optim.trials = [Trial(params=json.loads(parameters), value=value) for (id, parameters, value, status, created_at, completed_at) in results if value is not None]
        #         sweep.trials = {id: TrialInfo(id=id, sweep_id=sweep.id, parameters=json.loads(parameters), value=value, status=status, created_at=created_at, completed_at=completed_at)
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
                print('Error running job', job.id, job.message)
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

    def create_project(self, name: str) -> str:
        project_id = str(uuid.uuid4())
        created_at = int(time.time())
        
        with self.db.get_cursor() as c:
            # Check if project with this name already exists
            c.execute("SELECT id FROM projects WHERE name = ?", (name,))
            existing = c.fetchone()
            if existing:
                return existing['id']
                
            # Create new project
            c.execute(
                "INSERT INTO projects (id, name, created_at) VALUES (?, ?, ?)",
                (project_id, name, created_at)
            )
        
        return project_id

    def get_project_by_name(self, name: str) -> Optional[str]:
        with self.db.get_cursor() as c:
            c.execute("SELECT id FROM projects WHERE name = ?", (name,))
            result = c.fetchone()
            return result['id'] if result else None
    
    def create_sweep(self, name: str, parameters: List[SweepSpaceItem], objective: Literal['min', 'max'] = 'min', project_id: str = None):
        sweep_id = str(uuid.uuid4())

        with self.db.get_cursor() as c:
            c.execute(
                "INSERT INTO sweeps (id, name, parameters, objective, status, created_at, project_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    sweep_id,
                    name,
                    json.dumps(todict(parameters)),
                    objective,
                    "active",
                    int(time.time()),
                    project_id,
                ),
            )
        
        self.sweeps[sweep_id] = SweepInfo(id=sweep_id, optim=Optim([SpaceItem(**todict(p)) for p in parameters], objective=objective), trials={})

        return sweep_id


    def update_sweep(self, sweep_id: str, name: str = None):
        if name is not None:
            with self.db.get_cursor() as c:
                query = "UPDATE sweeps SET name = ? WHERE id = ?"
                c.execute(query, [name, sweep_id])

    def create_trial(self, sweep_id: str, parameters: Dict[str, float], value: Optional[float]):
        id = str(uuid.uuid4())
        created_at = int(time.time())
        if value is None:
            status = "started"
            completed_at = None
        else:
            status = "complete"
            completed_at = created_at

        with self.db.get_cursor() as c:
            # Get project_id from the sweep
            c.execute("SELECT project_id FROM sweeps WHERE id = ?", (sweep_id,))
            result = c.fetchone()
            project_id = result['project_id'] if result else None
            
            # Atomically increment and get the trial number
            c.execute(
                "UPDATE sweeps SET num_trials = num_trials + 1 WHERE id = ? RETURNING num_trials",
                [sweep_id]
            )
            trial_number = c.fetchone()['num_trials']

            c.execute(
                """INSERT INTO trials (id, sweep_id, parameters, value, status, created_at, completed_at, trial_number, project_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    id,
                    sweep_id,
                    json.dumps(parameters),
                    value,
                    status,
                    created_at,
                    completed_at,
                    trial_number,
                    project_id,
                ),
            )
        
        return id

    def report_result(self, sweep_id: str, trial_id: str, value: float):
        completed_at = int(time.time())
        with self.db.get_cursor() as c:
            c.execute(
                """UPDATE trials 
                SET value = ?, status = ?, completed_at = ?
                WHERE sweep_id = ? AND id = ?""",
                (value, "complete", completed_at, sweep_id, trial_id),
            )
        self.sweeps[sweep_id].updated_at = int(time.time())
        self.schedule(InferTask(self, sweep_id))
    
    def ask_params(self, sweep_id: str, params: Dict[str, float]) -> UUID:
        print('Asking for params, sweep_id:', sweep_id, 'params:', params)
        return self.schedule(AskTask(self, sweep_id, params))

    def get_slice_visualization(self, sweep_id: str, param_name: str):
        cached: SliceVisualization = None
        if ('slice', sweep_id, param_name) in self.viz_cache:
            cached = self.viz_cache[('slice', sweep_id, param_name)]
        
        if cached and cached.computed_at >= self.sweeps[sweep_id].updated_at:
            return {'cached': cached, 'job_id': None}
        elif cached:
            job_id = self.schedule(SliceVisualizationTask(self, sweep_id, param_name))
            return {'cached': cached, 'job_id': job_id}
        else:
            job_id = self.schedule(SliceVisualizationTask(self, sweep_id, param_name))
            return {'cached': None, 'job_id': job_id}


        
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
            c.execute("SELECT id, parameters, value, status, created_at, completed_at FROM trials WHERE sweep_id = ?", (sweep.id,))
            results = c.fetchall()
            sweep.optim.trials = [Trial(params=json.loads(parameters), value=value) for (id, parameters, value, status, created_at, completed_at) in results if status == 'complete']
        print('trials:', len(sweep.optim.trials))
        if len(sweep.optim.trials) > 0:
            sweep.optim.infer()
        sweep.inferred_at = int(time.time())
        self.status = 'done'

class AskTask(Task):
    def __init__(self, manager: SweepManager, sweep_id: str, params: Dict[str, float]):
        self.manager = manager
        self.sweep_id = sweep_id
        self.params = params
        self.result = None
    
    def __call__(self):
        print('Starting task', self.id)
        self.status = 'running'
        sweep = self.manager.sweeps[self.sweep_id]
        if sweep.inferred_at <= sweep.updated_at:
            InferTask(self.manager, self.sweep_id)()
        ps = sweep.optim.suggest(self.params)
        self.result = {k: float(v) for (k,v) in ps.items()}
        self.status = 'done'

class SliceVisualizationTask(Task):
    def __init__(self, manager: SweepManager, sweep_id: str, param_name: str):
        self.manager = manager
        self.sweep_id = sweep_id
        self.param_name = param_name
        self.result = None
    
    def __call__(self):
        print('Starting slice visualization task', self.id)
        self.status = 'running'
        sweep = self.manager.sweeps[self.sweep_id]
        if sweep.inferred_at <= sweep.updated_at:
            InferTask(self.manager, self.sweep_id)()
        space = sweep.optim.space
        param = space.items[self.param_name]
        xs = param.linspace(100)
        def pred(x):
            params = sweep.optim.suggestbest({param.name: x}, method='cma-es')
            mean, std = sweep.optim.fitted.predict(space.normalize(params)[None])
            return mean.squeeze(), std.squeeze(), params
        means, stds, params = jax.vmap(pred)(xs)
        computed_at = int(time.time())
        datapoints = [SliceVisualizationDatapoint(x=x, y_mean=mean, y_std=std, params = {k:v[i].item() for k,v in params.items()}) for i, (x,mean,std) in enumerate(zip(xs.tolist(), means.tolist(), stds.tolist()))]
        self.result = SliceVisualization(sweep_id=self.sweep_id, param_name=self.param_name, computed_at=computed_at, data=datapoints)
        self.manager.viz_cache[('slice', self.sweep_id, self.param_name)] = self.result
        self.status = 'done'
        print('Finished slice visualization task', self.id)

        


def init_db(db: Database):
    with db.get_cursor() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT,
                created_at TIMESTAMP
            )
            """
        )

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS sweeps (
                id TEXT PRIMARY KEY,
                project_id TEXT,
                name TEXT,
                parameters JSON,
                objective TEXT,
                status TEXT,
                created_at TIMESTAMP,
                num_trials INTEGER DEFAULT 0
            )
        """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS trials (
                id TEXT PRIMARY KEY,
                sweep_id TEXT,
                project_id TEXT,
                parameters JSON,
                value REAL,
                status TEXT,
                created_at TIMESTAMP,
                completed_at TIMESTAMP,
                trial_number INTEGER NOT NULL,
                FOREIGN KEY (sweep_id) REFERENCES sweeps (id)
            )
        """
        )