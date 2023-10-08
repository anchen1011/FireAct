import json
import re
import string
from .base import BaseTask, DATA_DIR
from collections import Counter
import pandas as pd


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


class BamboogleTask(BaseTask):
    def __init__(self, split):
        data_file = f"{DATA_DIR}/bamboogle/{split}.csv"
        # load csv file
        self.data = pd.read_csv(data_file)
        self.data = self.data.to_dict('records')

    def __getitem__(self, idx):
        return self.data[idx]["Question"]

    def __len__(self):
        return len(self.data)

    def evaluate(self, idx, answer):
        pred = normalize_answer(answer)
        gt = normalize_answer(self.data[idx]["Answer"])
        em = (pred == gt)
        f1 = f1_score(pred, gt)[0]
        return em, {'reward': em, 'em': em, 'f1': f1, 'gt': gt, 'pred': pred}
    
    def get_prompt(self):
        with open(f"{DATA_DIR}/../prompts/react_hotpotqa_google.txt", "r") as fin:
            prompt = fin.read() 
        return prompt