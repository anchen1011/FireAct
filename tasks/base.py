import os
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")

class BaseTask:
    def __init__(self):
        pass

    def __getitem__(self, idx):
        return None
    
    def __len__(self):
        return None
    
    def evaluate(self, idx, answer):
        return None
    
    def get_prompt(self):
        return None