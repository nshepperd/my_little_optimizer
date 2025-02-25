from __future__ import annotations

import requests
import json
from typing import List, Dict
from dataclasses import dataclass, asdict
import time
import numpy as np

from my_little_optimizer.opt.optim import SpaceItem

def req(url, method, data=None):
    r = requests.request(method, url, json=data)
    while r.status_code != 200:
        print(f'Error {method}ing {url}, retrying: {r.status_code} {r.text}')
        time.sleep(1)
    return r

class OptimClient:
    def __init__(self, url):
        self.url = url
    
    def get_sweep(self, id: str):
        return SweepHandle(self, id)
    
    def get_project(self, id: str):
        return ProjectHandle(self, id)
    
    def get_project_by_name(self, name: str):
        return self.new_project(name)

    def new_project(self, name: str):
        data = {"name": name}
        r = req(self.url + '/api/projects/', 'POST', data=data)
        r = r.json()
        assert r['status'] == 'ok'
        return ProjectHandle(self, r['data'])

    def new_sweep(self, name: str, parameters: List[SpaceItem], objective='min', project_id=None, project_name=None):
        if not project_id and not project_name:
            raise ValueError("Either project_id or project_name must be provided")
            
        data = {
            "name": name,
            "parameters": [asdict(p) for p in parameters],
            "objective": objective,
        }
        
        if project_id:
            data["project_id"] = project_id
        else:
            data["project_name"] = project_name
            
        r = req(self.url + '/api/sweeps/', 'POST', data=data)
        r = r.json()
        assert r['status'] == 'ok'
        return SweepHandle(self, r['data'])

@dataclass
class ProjectHandle:
    client: OptimClient
    id: str
    
    def get_sweeps(self):
        r = req(self.client.url + f'/api/projects/{self.id}/sweeps', 'GET')
        r = r.json()
        assert r['status'] == 'ok'
        return [SweepHandle(self.client, sweep['id']) for sweep in r['data']]
    
    def new_sweep(self, name: str, parameters: List[SpaceItem], objective='min'):
        return self.client.new_sweep(name, parameters, objective, project_id=self.id)
    
@dataclass
class SweepHandle:
    client: OptimClient
    id: str

    def ask(self, params: Dict[str, float] = None) -> Dict[str, float]:
        if params is None:
            params = {}
        data = params
        r = req(self.client.url + '/api/sweeps/' + self.id + '/ask', 'POST', data=data)
        r = r.json()
        assert r['status'] == 'ok'
        job_id = r['data']
        print('Waiting for job to complete...', end='', flush=True)
        while True:
            r = req(self.client.url + f'/api/poll/{job_id}', 'POST')
            r = r.json()
            if r['status'] == 'ok':
                task_info = r['data']
                if task_info['status'] == 'done':
                    print()
                    return task_info['result']
                elif task_info['status'] == 'error':
                    raise Exception(task_info['message'])
                elif task_info['status'] in ('running', 'pending'):
                    print('.', end='', flush=True)
            elif r['status'] == 'error':
                if r['code'] == 'not_found':
                    # The server might have restarted, so we need to ask again.
                    r = req(self.client.url + '/api/sweeps/' + self.id + '/ask', 'POST', data=data)
                    r = r.json()
                    assert r['status'] == 'ok'
                    job_id = r['data']
            time.sleep(1)


    def start(self, params: Dict[str, float]) -> TrialHandle:
        data = {
            "sweep_id": self.id,
            "parameters": params
        }
        r = req(self.client.url + f'/api/sweeps/{self.id}/trials/', 'POST', data=data)
        r = r.json()
        assert r['status'] == 'ok'
        return TrialHandle(self.client, self.id, r['data'])

@dataclass
class TrialHandle:
    client: OptimClient
    sweep_id: str
    id: str
    def report(self, value: float):
        data = {'value': value}
        r = req(self.client.url + f'/api/sweeps/{self.sweep_id}/trials/{self.id}/report', 'POST', data=data)
        r = r.json()
        assert r['status'] == 'ok'


if __name__ == '__main__':
    client = OptimClient('http://localhost:8000')
    sweep = client.new_sweep('test_xy_complex', [SpaceItem('x', -1, 1),
                                                SpaceItem('y', -1, 1)
                                                ], objective='min',
                                                project_name='test')
    print('id:', sweep.id)
    # sweep = client.get_sweep('test_random_y')
    for i in range(20):
        # y = float(np.random.uniform(-1, 1))
        params = sweep.ask()
        trial = sweep.start(params)
        x = params['x']
        y = params['y']
        value = float(x**2 + (x-y)**2 + 1)
        trial.report(value)
        print(params, value)
        time.sleep(1)
    