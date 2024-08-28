import os
import re
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import datasets
from datasets import Dataset,concatenate_datasets,load_dataset
from tqdm import tqdm
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
import transformers
import argparse
import random
import scipy.stats
import math
import heapq
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers.generation.utils import GenerationConfig
import bitsandbytes as bnb
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    load_peft_weights,
    set_peft_model_state_dict,
)
import torch
import torch.nn.functional as F
import pprint
from inference2 import inference_output_prob ## 计算难度
from util import compute_prob_average, compute_prob_sum, random_sampling, validate
from inference import inference_output_prob_origin ## 获得LLM对于文本的embedding
import json
import time
import io
from datetime import datetime, date



# CUDA_VISIBLE_DEVICES=0 nohup python -u train.py --sort random --select_num 300 > baichuan_random.out 2>&1 &


def dpp(kernel_matrix, max_length, epsilon=1E-10):
    """
    Fast greedy implementation of DPP algorithm.
    From: https://github.com/laming-chen/fast-map-dpp.
    """
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        di2s[selected_item] = -np.inf
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items


def compute_kernel_matrix2(data, epoch, alpha, scale=1.0):
    n_samples = len(data)
    features = []
    scores = []
    max_length = 30  #根据实际输出len(sentence_emb)长度调整

    for example in data:
        sentence_emb = example['conversation'][0][f'prompt_probs_{epoch}']
        if epoch >=1:
            # score = example[f'difficulty_{epoch}'] - alpha*(example[f'difficulty_{epoch}'] - example[f'difficulty_{epoch-1}'])
            score = example[f'difficulty_{epoch}']
        elif epoch ==0:
            score = example[f'difficulty_{epoch}']

        # print("sentence_emb", len(sentence_emb))
        if len(sentence_emb) >= max_length:
            smallest_100 = heapq.nsmallest(max_length, sentence_emb)
        else:
            smallest_100 = sorted(sentence_emb)
        features.append(smallest_100)
        scores.append(score)

    features = [torch.tensor(sentence, dtype=torch.float32) if not isinstance(sentence, torch.Tensor) else sentence for sentence in features]
    features_padded = [F.pad(sentence, (0, max_length - len(sentence)), mode='constant', value=0) for sentence in features]
    features = np.array([feature.numpy() for feature in features_padded])
    scores = np.array(scores)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    features = (features - mean) / std
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    
    similarity_matrix = np.dot(features, features.T)
    quality_matrix = np.diag(scores)
    kernel_matrix = quality_matrix @ similarity_matrix @ quality_matrix
    
    return kernel_matrix


def policy_sampling_SPL_DPP(epoch, NUM_EPOCHS, model, data_raw, select_num, tokenizer, alpha):
    def compute_difficulty(example, epoch):
        prob = example['conversation'][0].get('prob', [])
        if prob:
            difficulty = 1 - sum(prob) / len(prob)
            return {f'difficulty_{epoch}': difficulty}
        else:
            return {f'difficulty_{epoch}': float('inf')}

    def process_policy(example, epoch):
        system = example['conversation'][0]['system']
        prompt = example['conversation'][0]['input']
        solution = example['conversation'][0]['output']
        action_space = re.split(r'\n+', solution)
        action_space = action_space[:-1] if action_space and action_space[-1] == '' else action_space

        messages = [{"role": "system", "content": system}]
        messages.append({"role": "user", "content": prompt})

        action_probs = inference_output_prob(
            model,
            tokenizer,
            messages,
            device=model.device,
            actions=action_space,
            context_len=2048,
            max_batch_size=5
        )

        prompt_probs = inference_output_prob_origin(
            model,
            tokenizer,
            messages,
            device=model.device,
            actions=[prompt],
            context_len=2048,
            max_batch_size=1
        )[0]

        example['conversation'][0]['prob'] = action_probs 
        example['conversation'][0][f'prompt_probs_{epoch}'] = prompt_probs 
        return example

    data_new = data_raw.map(lambda x: process_policy(x, epoch), batched=False)
    data_new = data_new.map(lambda x: compute_difficulty(x, epoch), batched=False)

    print("data_new", data_new)
    print("epoch", epoch)

    #设定难度分位点范围
    start_percentile = 45
    end_percentile = 95
    percentile_increment = (end_percentile - start_percentile) / NUM_EPOCHS



    # 根据当前epoch确定当前难度分位点

    # 仅考虑当前轮次的难度，不考虑难度的变化趋势
    if epoch >=1:
        difficulties = [d[f'difficulty_{epoch}'] for d in data_new]
    elif epoch ==0:
        difficulties = [d[f'difficulty_{epoch}'] for d in data_new]

    current_percentile = start_percentile + percentile_increment * epoch
    difficulty_threshold = np.percentile(difficulties, current_percentile)
    eligible_data = data_new.filter(lambda item: item[f'difficulty_{epoch}'] <= difficulty_threshold)


    ## 考虑难度与前一轮次的差值
    # if epoch >=1:
    #     difficulties = [d[f'difficulty_{epoch}'] + alpha*(d[f'difficulty_{epoch}'] - d[f'difficulty_{epoch-1}']) for d in data_new]
    # elif epoch ==0:
    #     difficulties = [d[f'difficulty_{epoch}'] for d in data_new]

    # if epoch >=1:
    #     current_percentile2 = start_percentile + percentile_increment * (epoch+1)  ## 难度范围用区间控制
    #     current_percentile1 = start_percentile + percentile_increment * (epoch-1)
    #     print("current_percentile2", current_percentile2)
    #     print("current_percentile1", current_percentile1)
    #     difficulty_threshold2 = np.percentile(difficulties, current_percentile2)
    #     difficulty_threshold1 = np.percentile(difficulties, current_percentile1)
    #     print("difficulty_threshold1", difficulty_threshold1)
    #     print("difficulty_threshold2", difficulty_threshold2)
    # elif epoch ==0:
    #     current_percentile = start_percentile + percentile_increment * epoch
    #     difficulty_threshold = np.percentile(difficulties, current_percentile)

    # if epoch >=1:
    #     # eligible_data = data_new.filter(lambda item: difficulty_threshold1 <= item[f'difficulty_{epoch}'] - alpha*(item[f'difficulty_{epoch}'] - item[f'difficulty_{epoch-1}']) <= difficulty_threshold2)
    #     # eligible_data = data_new.filter(lambda item: difficulty_threshold1 <= item[f'difficulty_{epoch}'] <= difficulty_threshold2)
    #     eligible_data = data_new.filter(lambda item: item[f'difficulty_{epoch}'] + alpha*(item[f'difficulty_{epoch}'] - item[f'difficulty_{epoch-1}']) <= difficulty_threshold)
    #     # eligible_data = data_new.filter(lambda item: difficulty_threshold1 <= item[f'difficulty_{epoch}'] + alpha*(item[f'difficulty_{epoch}'] - item[f'difficulty_{epoch-1}']) <= difficulty_threshold2)
    # elif epoch ==0:
    #     eligible_data = data_new.filter(lambda item: item[f'difficulty_{epoch}'] <= difficulty_threshold)



    # DPP for diversity
    selected_data = Dataset.from_dict({k: [] for k in eligible_data.column_names})
    
    while True:
        eligible_data_list = list(eligible_data)
        print("eligible_data_list", eligible_data_list)
        kernel = compute_kernel_matrix2(eligible_data_list, epoch, alpha)
        new_selected_indices = dpp(kernel, select_num - len(selected_data))
        new_selected_data = eligible_data.select(new_selected_indices)
        
        print("new_selected_data", new_selected_indices)
        print("number of new_selected_data", len(new_selected_indices))
        
        selected_data = concatenate_datasets([selected_data, new_selected_data])
        if len(selected_data) >= select_num:
            break
        all_indices = np.arange(len(eligible_data))
        remaining_indices = np.setdiff1d(all_indices, new_selected_indices)
        eligible_data = eligible_data.select(remaining_indices)
    print(f"Epoch {epoch}: Selected dataset size: {len(selected_data)} items")

    return selected_data, data_new


def policy_sampling_SPL(epoch, NUM_EPOCHS, model, data_raw, select_num, tokenizer, alpha):
    def compute_difficulty(example, epoch):
        prob = example['conversation'][0].get('prob', [])
        if prob:
            difficulty = 1 - sum(prob) / len(prob)
            return {f'difficulty_{epoch}': difficulty}
        else:
            return {f'difficulty_{epoch}': float('inf')}

    def process_policy(example, epoch):
        system = example['conversation'][0]['system']
        prompt = example['conversation'][0]['input']
        solution = example['conversation'][0]['output']
        action_space = re.split(r'\n+', solution)
        action_space = action_space[:-1] if action_space and action_space[-1] == '' else action_space

        messages = [{"role": "system", "content": system}]
        messages.append({"role": "user", "content": prompt})

        action_probs = inference_output_prob(
            model,
            tokenizer,
            messages,
            device=model.device,
            actions=action_space,
            context_len=2048,
            max_batch_size=3
        )

        prompt_probs = inference_output_prob_origin(
            model,
            tokenizer,
            messages,
            device=model.device,
            actions=[prompt],
            context_len=2048,
            max_batch_size=1
        )[0]
        # print("answer_probs", prompt_probs)
        # print("answer_probs", len(answer_probs))
        example['conversation'][0]['prob'] = action_probs 
        example['conversation'][0][f'prompt_probs_{epoch}'] = prompt_probs 
        return example

    data_new = data_raw.map(lambda x: process_policy(x, epoch), batched=False)
    data_new = data_new.map(lambda x: compute_difficulty(x, epoch), batched=False)
    print("data_new", data_new)
    print("epoch", epoch)

    # if epoch >=1:
    #     difficulties = [d[f'difficulty_{epoch}'] for d in data_new]
    # elif epoch ==0:
    #     difficulties = [d[f'difficulty_{epoch}'] for d in data_new]

    # start_percentile = 30
    # end_percentile = 95
    # percentile_increment = (end_percentile - start_percentile) / NUM_EPOCHS
    # current_percentile = start_percentile + percentile_increment * epoch
    # difficulty_threshold = np.percentile(difficulties, current_percentile)

    # eligible_data = data_new.filter(lambda item: item[f'difficulty_{epoch}'] <= difficulty_threshold)


    if epoch >=1:
        difficulties = [d[f'difficulty_{epoch}'] + alpha*(d[f'difficulty_{epoch}'] - d[f'difficulty_{epoch-1}']) for d in data_new]
        # difficulties = [d[f'difficulty_{epoch}'] for d in data_new]
        # print("difficulties", difficulties)
    elif epoch ==0:
        difficulties = [d[f'difficulty_{epoch}'] for d in data_new]
        # print("difficulties", difficulties)


    start_percentile = 30
    end_percentile = 95
    percentile_increment = (end_percentile - start_percentile) / NUM_EPOCHS
 
    # if epoch >=1:
    #     current_percentile2 = start_percentile + percentile_increment * (epoch+1)
    #     current_percentile1 = start_percentile + percentile_increment * (epoch-1)
    #     difficulty_threshold2 = np.percentile(difficulties, current_percentile2)
    #     difficulty_threshold1 = np.percentile(difficulties, current_percentile1)
    #     print("difficulty_threshold1", difficulty_threshold1)
    #     print("difficulty_threshold2", difficulty_threshold2)
    # elif epoch ==0:
    #     current_percentile = start_percentile + percentile_increment * epoch
    #     difficulty_threshold = np.percentile(difficulties, current_percentile)

    # current_percentile = start_percentile + percentile_increment * epoch
    # difficulty_threshold = np.percentile(difficulties, current_percentile)

    # if epoch >=1:
        # eligible_data = data_new.filter(lambda item: difficulty_threshold1 <= item[f'difficulty_{epoch}'] + alpha*(item[f'difficulty_{epoch}'] - item[f'difficulty_{epoch-1}']) <= difficulty_threshold2)
        # eligible_data = data_new.filter(lambda item: difficulty_threshold1 <= item[f'difficulty_{epoch}'] <= difficulty_threshold2)
    # elif epoch ==0:
        # eligible_data = data_new.filter(lambda item: item[f'difficulty_{epoch}'] <= difficulty_threshold)

    selected_data = Dataset.from_dict({k: [] for k in eligible_data.column_names})

    print("len(eligible_data)", len(eligible_data), "select_num", select_num)
    if select_num > len(eligible_data):
        selected_data = eligible_data.shuffle(seed=epoch).select(range(min(select_num, len(eligible_data))))
    else:
        selected_data = eligible_data.shuffle(seed=epoch).select(range(select_num))

    print(f"Epoch {epoch}: Selected dataset size: {len(selected_data)} items")

    return selected_data, data_new


def tokenize(prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < CUTOFF_LEN
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    if add_eos_token and len(result["input_ids"]) >= CUTOFF_LEN:
        result["input_ids"][CUTOFF_LEN - 1] = tokenizer.eos_token_id
        result["attention_mask"][CUTOFF_LEN - 1] = 1

    result["labels"] = result["input_ids"].copy()
    return result


def generate_and_tokenize_prompt(data_point):
    data_point = data_point["conversation"][0]
    instruction = data_point['system']
    input_text = data_point["input"]
    input_text = instruction + input_text
    input_text = tokenizer.bos_token + input_text if tokenizer.bos_token != None else input_text
    target_text = data_point["output"] + tokenizer.eos_token
    full_prompt = input_text + target_text
    tokenized_full_prompt = tokenize(full_prompt)
    return tokenized_full_prompt


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit 
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)



def train_model(model, tokenizer, train_data, BATCH_SIZE, VAL_SET_SIZE, OUTPUT_DIR, last_checkpoint=None):
    model.train()
    checkpoint_path = os.path.join(OUTPUT_DIR, f"epoch_{last_checkpoint}_model") if last_checkpoint else None
    print("checkpoint_path: ", checkpoint_path)
    # print("if os.path.exists(checkpoint_path): ", os.path.exists(checkpoint_path))

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = None 
        print("Starting training from scratch.")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,  
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=4e-4,
        weight_decay=1e-4,
        gradient_accumulation_steps=1,
        evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
        save_strategy="steps",
        eval_steps=2000 if VAL_SET_SIZE > 0 else None,
        save_steps=2000,
        logging_steps=20,
        save_total_limit=3,
        load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
        optim="adamw_torch",
        resume_from_checkpoint=checkpoint_path,
        report_to="tensorboard"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        data_collator=data_collator
    )

    trainer.train()

    if last_checkpoint is not None:
        model.save_pretrained(os.path.join(OUTPUT_DIR, f"epoch_{last_checkpoint + 1}_model"))
    else:
        model.save_pretrained(os.path.join(OUTPUT_DIR, "epoch_0_model"))




def train(alpha, sort_type, metric, select_num, NUM_EPOCHS, model, BATCH_SIZE, data, test_data, tokenizer, OUTPUT_DIR, VAL_SET_SIZE, last_checkpoint=None):

    if last_checkpoint is None:
        NUM_EPOCHS_STATRT = 0
    else:
        NUM_EPOCHS_STATRT = last_checkpoint+1

    for epoch in range(NUM_EPOCHS_STATRT, NUM_EPOCHS):
        print("The current epoch is ", epoch)
        random.seed(epoch)
        if sort_type == 'random':
            train_data = random_sampling(data['train'], select_num)
        
        if sort_type == 'total':
            print("select method:", sort_type)
            train_data = data['train']

        if sort_type == 'policy_sampling_SPL':
            print("select method:", sort_type)
            if epoch == 0:
                train_data, data_new = policy_sampling_SPL(epoch, NUM_EPOCHS, model, data['train'], select_num, tokenizer, alpha)
            if epoch >= 1:
                train_data, data_new = policy_sampling_SPL(epoch, NUM_EPOCHS, model, data_new, select_num, tokenizer, alpha)

        if sort_type == 'policy_sampling_SPL_DPP': 
            print("select method:", sort_type)
            if epoch == 0:
                train_data, data_new = policy_sampling_SPL_DPP(epoch, NUM_EPOCHS, model, data['train'], select_num, tokenizer, alpha)
            if epoch >= 1:
                train_data, data_new = policy_sampling_SPL_DPP(epoch, NUM_EPOCHS, model, data_new, select_num, tokenizer, alpha)

        train_data = train_data.shuffle().map(generate_and_tokenize_prompt)
        train_model(model, tokenizer, train_data, BATCH_SIZE, VAL_SET_SIZE, OUTPUT_DIR, last_checkpoint)
        last_checkpoint = epoch

    # validation
    sr = validate(model, tokenizer, test_data)
    print("for epoch", epoch, ", the sr is ", sr)

    return 0



def parse_args():
    parser = argparse.ArgumentParser(description="Process mode and sort type for script execution.")
    parser.add_argument('--sort', type=str, required=False, default='random')
    parser.add_argument('--metric', type=str, required=False, default=None)
    parser.add_argument('--select_num', type=int, required=False, help="Number of items to select in 'select_data' mode", default=10000)
    parser.add_argument('--epoch', type=int, required=False, help="Number of epochs", default=5)
    parser.add_argument('--alpha', type=float, required=False, help="Number of epochs", default=0.1)
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()
    sort_type = args.sort
    select_num = args.select_num
    metric = args.metric
    alpha = args.alpha
    
    NUM_EPOCHS = args.epoch
    CUTOFF_LEN = 2048  
    VAL_SET_SIZE = 0
    BATCH_SIZE = 5
    print("NUM_EPOCHS", NUM_EPOCHS)
    
    DATA_PATH = "/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/yangyingxuan/SFT/math/sft_data/train/algebra.json" 
    TEST_PATH = "/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/yangyingxuan/SFT/math/sft_data/test/algebra.json" 
    OUTPUT_DIR = "/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/yangyingxuan/SFT/math/output/llama/algebra5/"


    OUTPUT_DIR = os.path.join(OUTPUT_DIR, str(sort_type))
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, str(NUM_EPOCHS))

    model_path = "/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/yangyingxuan/model/Llama-3-8B-Instruct-Gradient-1048k"

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True, padding_side='right')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 trust_remote_code=True,
                                                 quantization_config=BitsAndBytesConfig(
                                                     load_in_4bit=True,
                                                     bnb_4bit_compute_dtype=torch.float16,
                                                     bnb_4bit_use_double_quant=True,
                                                     bnb_4bit_quant_type='nf4'
                                                 ),
                                                 use_cache=False,
                                                 device_map="auto")

    config = GenerationConfig(**{
        "assistant_token_id": 196,
        "bos_token_id": 1,
        "do_sample": True,
        "eos_token_id": 2,
        "max_new_tokens": 2048,
        "pad_token_id": 0,
        "repetition_penalty": 1.05,
        "temperature": 0.2,
        "top_k": 5,
        "top_p": 0.85,
        "user_token_id": 195
    })

    model.generation_config = config

    model = prepare_model_for_kbit_training(model)
    modules = find_all_linear_names(model)

    config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.15,
        bias="none",
        target_modules=modules,
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)

    # adapter_weights = load_peft_weights(adapter_path)
    # set_peft_model_state_dict(model, adapter_weights)


    data = load_dataset("json", data_files=DATA_PATH)
    test_data = load_dataset("json", data_files=TEST_PATH)
    last_checkpoint = None
    train(alpha, sort_type, metric, select_num, NUM_EPOCHS, model, BATCH_SIZE, data, test_data, tokenizer, OUTPUT_DIR, VAL_SET_SIZE, last_checkpoint)


