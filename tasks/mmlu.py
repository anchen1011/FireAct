import pandas as pd
from .base import BaseTask, DATA_DIR
from collections import Counter
import os


class MMLUTask(BaseTask):
    def __init__(self, name):
        if name in ["train", "dev", "val", "test"]:
            split = name
            path = f"{DATA_DIR}/mmlu/{split}"
            self.data = []
            for file in os.listdir(path):
                if file.endswith(".csv"):
                    self.data += pd.read_csv(f"{path}/{file}", header=None).values.tolist()
            self.data = [
                {
                    "question": "Question: {}\nA. {}\nB. {}\nC. {}\nD. {}".format(*row[:5]),
                    "answer": row[5]
                }
                for row in self.data
            ]
        else:
            split = name.split("_")[-1]
            assert split in ["train", "dev", "val", "test"]
            path = f"{DATA_DIR}/mmlu/{split}/{name}.csv"
            self.data = pd.read_csv(path, header=None)
            self.data = [
                {
                    "question": "Question: {}\nA. {}\nB. {}\nC. {}\nD. {}".format(*row[:5]),
                    "answer": row[5]
                }
                for row in self.data.values
            ]

    def __getitem__(self, idx):
        return self.data[idx]["question"]

    def __len__(self):
        return len(self.data)

    def evaluate(self, idx, answer):
        pred = answer.lower()
        gt = self.data[idx]['answer'].lower()
        em = int(pred.startswith(gt))
        return em, {'reward': em, 'em': em, 'gt': gt, 'pred': pred}
    
    def get_prompt(self):
        with open(f"{DATA_DIR}/../prompts/react_mmlu_google.txt", "r") as fin:
            prompt = fin.read() 
        return prompt