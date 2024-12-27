import requests
import json
from typing import List
from dataclasses import dataclass, asdict

from ngp.optim import SpaceItem

class OptimClient:
    def __init__(self, url):
        self.url = url
    
    def new_sweep(self, name: str, parameters: List[SpaceItem]):
        data = {
            "name": name,
            "parameters": [asdict(p) for p in parameters]
        }
        r = requests.post(self.url + '/api/sweeps/', json=data)
        print(r.json())
        return Sweep(self, r.json()['id'])
    
class Sweep:
    def __init__(self, client: OptimClient, id: str):
        self.client = client
        self.id = id
    
    def start(self, params: dict):
        data = {
            "sweep_id": self.id,
            "parameters": params
        }
        r = requests.post(self.client.url + '/api/experiments/', json=data)
        print(r.json())
        return Experiment(self.client, r.json()['id'])

class Experiment:
    def __init__(self, client: OptimClient, id: str):
        self.client = client
        self.id = id
    
    def report(self, value: float):
        data = {
            "value": value
        }
        r = requests.post(self.client.url + '/api/experiments/' + self.id + '/report', json=data)
        print(r.status_code)

if __name__ == '__main__':
    client = OptimClient('http://localhost:8000')
    sweep = client.new_sweep('test', [SpaceItem('x', -1, 1)])
    exp = sweep.start({'x': 0.5})
    print(exp.report(0.5))
    
