from __future__ import annotations

from typing import List, Dict, Optional, Any
import sqlite3
from contextlib import contextmanager, asynccontextmanager
import threading
from queue import Queue
from dataclasses import dataclass, asdict

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


class SweepManager:
    sweeps: Dict[str, SweepInfo]
    db: Database

    def __init__(self, db: Database):
        self.db = db
        self.init_sweeps()

    def init_sweeps(self):
        with self.db.get_cursor() as c:
            c.execute("SELECT id, parameters FROM sweeps")
            results = c.fetchall()
            self.sweeps = {id:
                SweepInfo(id=id, dirty=True, optim=Optim([SpaceItem(**p) for p in parameters]))
                for (id, parameters) in results
            }
        for sweep in self.sweeps:
            self.update_experiments(sweep.id)
    
    def add_sweep(self, id: str, parameters: List[Dict[str, Any]]):
        assert id not in self.sweeps
        self.sweeps[id] = SweepInfo(id=id, dirty=True, optim=Optim([SpaceItem(**p) for p in parameters]))

    def update_experiments(self, sweep_id: str):
        with self.db.get_cursor() as c:
            c.execute("SELECT parameters, value FROM experiments WHERE sweep_id = ?, status = 'complete'", (sweep_id,))
            results = c.fetchall()
            sweep = self.sweeps[sweep_id]
            sweep.optim.trials = [Trial(parameters=parameters, value=value) for (parameters, value) in results if value is not None]

    # def update_chains(self, sweep_id):
        
    #     with self.db.get_cursor() as c:

@dataclass
class SweepInfo:
    id: str
    dirty: bool
    optim: Optim

