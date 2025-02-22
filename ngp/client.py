import requests
import json
from typing import List, Dict
from dataclasses import dataclass, asdict
import time

from ngp.optim import SpaceItem

class OptimClient:
    def __init__(self, url):
        self.url = url
    
    def get_sweep(self, id: str):
        # TODO: check that it exists first
        return SweepHandle(self, id)


    def new_sweep(self, name: str, parameters: List[SpaceItem]):
        data = {
            "name": name,
            "parameters": [asdict(p) for p in parameters]
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
            if 'detail' in r.json():
                print(r.json()['detail']['message'])
            raise Exception(r.json())
        r = r.json()
        return r['params']

    def start(self, params: Dict[str, float]) -> Dict[str, float]:
        data = {
            "sweep_id": self.id,
            "parameters": params
        }
        r = requests.post(self.client.url + f'/api/sweeps/{self.id}/experiments/', json=data)
        if r.status_code != 200:
            if 'detail' in r.json():
                print(r.json()['detail']['message'])
            raise Exception(r.json())
        return ExperimentHandle(self.client, self.id, r.json()['id'])

@dataclass
class ExperimentHandle:
    client: OptimClient
    sweep_id: str
    id: str
    def report(self, value: float):
        data = {'value': value}
        r = requests.post(self.client.url + f'/api/sweeps/{self.sweep_id}/experiments/{self.id}/report', json=data)
        if r.status_code != 200:
            if 'detail' in r.json():
                print(r.json()['detail']['message'])
            raise Exception(r.json())
if __name__ == '__main__':
    client = OptimClient('http://localhost:8000')
    sweep = client.new_sweep('test', [SpaceItem('x', -1, 1)])
    for i in range(10):
        params = sweep.ask()
        experiment = sweep.start(params)
        x = params['x']
        value = float((x-0.2)**2 + 1)
        experiment.report(value)
        print(params, value)
        time.sleep(1)
    
