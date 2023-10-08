import json
import re
import string
from .base import BaseTask, DATA_DIR
from collections import Counter


class StrategyQATask(BaseTask):
    def __init__(self, split):
        assert split in ["train", "test", "dev"]
        path = f"{DATA_DIR}/strategyqa/strategyqa_{split}.json"
        with open(path, 'r') as f:
            self.data = json.load(f)

    def __getitem__(self, idx):
        return self.data[idx]["question"]

    def __len__(self):
        return len(self.data)

    def evaluate(self, idx, answer):
        pred = answer.lower()
        gt = self.data[idx]['answer']
        em = int(pred.startswith('yes') and gt == True or pred.startswith('no') and gt == False)
        return em, {'reward': em, 'em': em, 'gt': gt, 'pred': pred}
    
    def get_prompt(self):
        with open(f"{DATA_DIR}/../prompts/react_strategyqa_google.txt", "r") as fin:
            prompt = fin.read() 
        return prompt