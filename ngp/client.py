import requests
import json
from typing import List, Dict
from dataclasses import dataclass, asdict
import time
import numpy as np

from ngp.optim import SpaceItem

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
        r = requests.post(self.url + '/api/sweeps/', json=data)
        if r.status_code != 200:
            if 'detail' in r.json():
                print(r.json()['detail']['message'])
            raise Exception(r.json())
        return SweepHandle(self, r.json()['id'])
    
@dataclass
class SweepHandle:
    client: OptimClient
    id: str

    def ask(self, params: Dict[str, float] = None) -> Dict[str, float]:
        if params is None:
            params = {}
        data = params
        r = requests.post(self.client.url + '/api/sweeps/' + self.id + '/ask', json=data)
        if r.status_code != 200:
            raise Exception(r.json())
        r = r.json()
        if r['status'] == 'error':
            raise Exception(r['message'])
        elif r['status'] == 'pending':
            job_id = r['job_id']
            print('Waiting for job to complete...', end='', flush=True)
            while True:
                r = requests.post(self.client.url + f'/api/poll/{job_id}')
                if r.status_code != 200:
                    print("Error while polling job status:", r)
                    time.sleep(1)
                    continue
                r = r.json()
                if r['status'] == 'success':
                    print()
                    return r['result']
                elif r['status'] == 'error':
                    raise Exception(r['message'])
                elif r['status'] in ('running', 'pending'):
                    print('.', end='', flush=True)
                    time.sleep(1)
                elif r['status'] == 'not found':
                    # The server might have restarted, so we need to ask again.
                    r = requests.post(self.client.url + '/api/sweeps/' + self.id + '/ask', json=data)
                    if r.status_code == 200:
                        job_id = r.json()['job_id']
                        continue


    def start(self, params: Dict[str, float]) -> Dict[str, float]:
        data = {
            "sweep_id": self.id,
            "parameters": params
        }
        r = requests.post(self.client.url + f'/api/sweeps/{self.id}/trials/', json=data)
        if r.status_code != 200:
            if 'detail' in r.json():
                print(r.json()['detail']['message'])
            raise Exception(r.json())
        return TrialHandle(self.client, self.id, r.json()['id'])

@dataclass
class TrialHandle:
    client: OptimClient
    sweep_id: str
    id: str
    def report(self, value: float):
        data = {'value': value}
        r = requests.post(self.client.url + f'/api/sweeps/{self.sweep_id}/trials/{self.id}/report', json=data)
        if r.status_code != 200:
            if 'detail' in r.json():
                print(r.json()['detail']['message'])
            raise Exception(r.json())


# 1 - gamma


if __name__ == '__main__':
    client = OptimClient('http://localhost:8000')
    sweep = client.new_sweep('test_random_y', [SpaceItem('x', -1, 1),
                                                SpaceItem('y', -1, 1)
                                                ], objective='min')
    print('id:', sweep.id)
    # sweep = client.get_sweep('test_random_y')
    for i in range(20):
        y = float(np.random.uniform(-1, 1))
        params = sweep.ask({'y':y})
        trial = sweep.start(params)
        x = params['x']
        value = float((x-y)**2 + 1)
        trial.report(value)
        print(params, value)
        time.sleep(1)
    