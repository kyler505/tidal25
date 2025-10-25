import requests
from typing import Optional

class APIConnector:
    def __init__(self, base_url=None, timeout=5):
        self.base = base_url
        self.timeout = timeout

    def get_prompts(self, endpoint=None):
        url = endpoint or (self.base + '/prompts' if self.base else None)
        if not url: 
            raise ValueError("No prompts endpoint configured")
        r = requests.get(url, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def post_reward(self, text, endpoint=None):
        url = endpoint or (self.base + '/score' if self.base else None)
        if not url:
            raise ValueError("No reward endpoint configured")
        r = requests.post(url, json={'text': text}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()
