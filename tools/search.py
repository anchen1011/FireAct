import os
import json
from serpapi import GoogleSearch
from IPython.utils import io

current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
try:
    current_dict = json.load(open((os.path.join(current_dir, "search.json"))))
except:
    current_dict = {}


def extract_answer(res):
    if "answer_box" in res.keys() and "answer" in res["answer_box"].keys():
        toret = res["answer_box"]["answer"]
    elif "answer_box" in res.keys() and "snippet" in res["answer_box"].keys():
        toret = res["answer_box"]["snippet"]
    elif "answer_box" in res.keys() and "snippet_highlighted_words" in res["answer_box"].keys():
        toret = res["answer_box"]["snippet_highlighted_words"][0]
    elif "organic_results" in res.keys() and "snippet" in res["organic_results"][0].keys():
        toret = res["organic_results"][0]["snippet"]
    else:
        toret = None
    return toret


def search(question):
    if question in current_dict.keys():
        return current_dict[question]
    params = {
        "api_key": os.environ["SERPAPI_API_KEY"],
        "engine": "google",
        "q": question,
        "google_domain": "google.com",
        "gl": "us",
        "hl": "en"
    }

    search = GoogleSearch(params)
    res = search.get_dict()

    toret = extract_answer(res)
    current_dict[question] = toret
    return toret


def search_save():
    with open(os.path.join(current_dir, "search.json"), "w") as fout:
        json.dump(current_dict, fout, indent=2)