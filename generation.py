import json
import argparse
import concurrent.futures
import random
import time
import logging
logging.getLogger().setLevel(logging.ERROR)
from functools import partial

from models.openai import chatgpts, gpts
from models.llama import LlamaInterface

from tasks import get_task
from tools import call_tools
from tools.search import search_save
from datetime import datetime

def get_fewshot_prompt(promptpath, task=None, chatgpt_format=False):
    if len(promptpath) == 0:
        return [] if chatgpt_format else ""
    elif promptpath == "default" and task is not None:
        return task.get_prompt()
    if not chatgpt_format:
        with open(f"./prompts/{promptpath}.txt", "r") as fin:
            prompt = fin.read() 
        return prompt
    else:
        with open(f"./prompts/{promptpath}.json", "r") as fin:
            prompt = json.load(fin)
        return prompt

def prepare_prompt(question):
    return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n### Response:\n"

def prune_thought(prompt):
    if prompt.startswith("Thought:"):
        return prompt[len("Thought:"):].strip()
    return prompt

def run(task, idxs, gpts, evaluate=True, alpaca_format=False, chatgpt_format=True, promptpath='', question_prefix=''):
    fewshot_prompt = get_fewshot_prompt(promptpath, task, chatgpt_format=chatgpt_format)
    questions = [question_prefix + task[idx] for idx in idxs]
    if not chatgpt_format:
        prompts = [fewshot_prompt + question + "\n" for question in questions]
    else:
        prompts = [fewshot_prompt + [{'role': 'user', 'content': question}] for question in questions]
    if alpaca_format:
        prompts = [prepare_prompt(q.rstrip()) for q in questions]

    rs, infos = {}, {}

    iteration = 0
    while iteration < 11:
        iteration += 1
        print(f"Iteration {iteration}")
        reflection = ""
        
        if not chatgpt_format:
            thought_action_pairs = gpts([prompt + f"Thought:{reflection}" for prompt in prompts], stop=[f"\nObservation:"])
        else:
            thought_action_pairs = gpts(prompts, stop=None)


        for _ in range(5):
            bad_ids = [i for i, pair in enumerate(thought_action_pairs) if "Action: " not in pair]
            if not bad_ids: break

            bad_prompts = [prompts[i] for i in bad_ids]
            bad_pairs = gpts(bad_prompts, stop=None)
            for i, pair in zip(bad_ids, bad_pairs):
                thought_action_pairs[i] = pair
                if _ == 4 and "Action: " not in pair:
                    thought_action_pairs[i] = "Thought: failed\nAction: finish[]"

        thoughts, actions, obs, bad_ids, done_ids = [], [], [], [], []
        for i, thought_action in enumerate(thought_action_pairs):

            try:
                if "\nAction: " in thought_action.strip():
                    thought, action = thought_action.strip().split("\nAction: ")[:2]
                elif "Action: " in thought_action.strip():
                    thought = ""
                    action = thought_action[len("Action: "):]
                else: 
                    thought = thought_action.split("\n")[0]
                    action = None
                    bad_ids.append(i)
                if len(reflection) > 0:
                    thought = reflection.strip() + " " + thought
            except:
                continue
            
            thoughts.append(thought)
            actions.append(action)
        if bad_ids: 
            assert not chatgpt_format, "chatgpt_format is not supported for bad_ids for now"
            bad_prompts = [prompts[i] + f"Thought: {prune_thought(thoughts[i])}\nAction:" for i in bad_ids]
            bad_actions = gpts(bad_prompts, stop=[f"\nObservation:"])
            for i, bad_action in zip(bad_ids, bad_actions):
                actions[i] = bad_action.strip()
        
        old_time = time.time()
        threads = []
        results = {}
        for i, action in enumerate(actions):
            try:
                action_type, action_args = action.split('[')[:2]
                action_args = action_args[:-1]
            except:
                continue
                
            if "finish" not in action_type.lower():
                t = (action_type, action_args)
                threads.append((i, t))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_threads = {executor.submit(call_tools, t[0], t[1]): (i, t) for i, t in threads}
            for future in concurrent.futures.as_completed(future_to_threads):
                i, t = future_to_threads[future]
                try:
                    obs = future.result()
                except Exception as exc:
                    obs = '%r generated an exception: %s' % (t, exc)
                    print(obs)
                results[i] = obs
        
        for i, action in enumerate(actions):
            try:
                action_type, action_args = action.split('[')[:2]
                action_args = action_args[:-1]
                obs = results.get(i, "Observation: None")
            except:
                continue

            if "finish" in action_type.lower():
                if evaluate:
                    r, info = task.evaluate(idxs[i], action_args)
                else:
                    r, info = True, {}

                assert obs == "Observation: None", f"action {action} has observation {obs}"
                obs = f"Episode finished, reward = {r}"
                done_ids.append(i)

            
            if obs == "Observation: None": 
                print(f"Warning: action {action} has observation {obs}")

            if not chatgpt_format:
                prompts[i] += f"Thought: {prune_thought(thoughts[i])}\nAction: {action}\nObservation: {obs}\n"
            else:
                prompts[i] += [{"role": "assistant", "content": thought_action_pairs[i]},
                                {"role": "user", "content": f"Observation: {obs}"}]
                
            if "finish" in action_type.lower():
                if not chatgpt_format:
                    traj = prompts[i][len(fewshot_prompt):]	
                    info.update({'prompt': fewshot_prompt, 'traj': traj, 'traj_by_line': traj.split('\n')})
                else:
                    info.update({'prompt': fewshot_prompt, 'traj': prompts[i][len(fewshot_prompt):].copy()})
                
                rs[idxs[i]] = r
                infos[idxs[i]] = info
        
        print(f"Time used for actions: {time.time() - old_time}", flush=True)
        prompts = [prompts[i] for i in range(len(prompts)) if i not in done_ids]
        idxs = [idxs[i] for i in range(len(idxs)) if i not in done_ids]
        if not prompts:
            break
        
    return rs, infos


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, default='gpt-4')
    args.add_argument('--temperature', type=float, default=0.7)

    args.add_argument('--task', type=str, required=True)
    args.add_argument('--task_split', type=str, default='train')
    args.add_argument('--task_start_index', type=int, default=0)
    args.add_argument('--task_end_index', type=int, default=100)

    args.add_argument('--evaluate', action='store_true')
    args.add_argument('--add_lora', action='store_true')
    args.add_argument('--random', action='store_true')
    args.add_argument('--alpaca_format', action='store_true')
    args.add_argument('--chatgpt_format', action='store_true')
    args.add_argument('--question_prefix', type=str, default='')

    args.add_argument('--modelpath', type=str, default='')
    args.add_argument('--peftpath', type=str, default='')
    args.add_argument('--promptpath', type=str, default='')

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    task = get_task(args.task, args.task_split)
    
    modelname = args.backend
    if args.backend == 'llama':
        pathname = args.peftpath.replace('/', '_') if args.add_lora else args.modelpath.replace('/', '_')
        modelname += f"_{pathname}"
    time_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    outfilename = f"trajs/{args.task}_{args.task_split}_{args.task_start_index}_{args.task_end_index}_{modelname}_{args.temperature}_{time_str}.json"
    print(outfilename)
    
    idxs_all = list(range(len(task)))
    if args.random:
        random.Random(233).shuffle(idxs_all)
    idxs = idxs_all[args.task_start_index:args.task_end_index]

    if args.backend == "llama":
        print(args.modelpath, args.peftpath, args.add_lora)
        llama = LlamaInterface(args.modelpath, args.peftpath, args.add_lora)
        model = llama.generate_responses_from_llama
    elif args.chatgpt_format:
        model = partial(chatgpts, model=args.backend, temperature=args.temperature)
    else:
        model = partial(gpts, model=args.backend, temperature=args.temperature)
    
    rs, infos = run(task, idxs, model, evaluate=args.evaluate, \
                    alpaca_format=args.alpaca_format, 
                    chatgpt_format=args.chatgpt_format,
                    promptpath=args.promptpath,
                    question_prefix=args.question_prefix)

    with open(outfilename, "w") as fout:
        json.dump(infos, fout, indent=2)
    em = sum(rs.values()) / len(idxs)
    print("em", em)

    search_save()