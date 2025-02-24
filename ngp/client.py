import requests
import json
from typing import List, Dict
from dataclasses import dataclass, asdict
import time
import numpy as np

from ngp.optim import SpaceItem

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
        # TODO: check that it exists first
        return SweepHandle(self, id)


    def new_sweep(self, name: str, parameters: List[SpaceItem], objective='min'):
        data = {
            "name": name,
            "parameters": [asdict(p) for p in parameters],
            "objective": objective,
        }
        r = req(self.url + '/api/sweeps/', 'POST', data=data)
        r = r.json()
        assert r['status'] == 'ok'
        return SweepHandle(self, r['data'])
    
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


    def start(self, params: Dict[str, float]) -> Dict[str, float]:
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
                                                ], objective='min')
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
    