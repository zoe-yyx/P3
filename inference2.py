import gc
import torch
import numpy as np
from typing import List

# From Baichuan2
def build_chat_input(model, tokenizer, messages: List[dict], max_new_tokens: int=0):
    def _parse_messages(messages, split_role="user"):
        system, rounds = "", []
        round = []
        for i, message in enumerate(messages):
            if message["role"] == "system":
                assert i == 0
                system = message["content"]
                continue
            if message["role"] == split_role and round:
                rounds.append(round)
                round = []
            round.append(message)
        if round:
            rounds.append(round)
        return system, rounds

    max_new_tokens = max_new_tokens or model.generation_config.max_new_tokens
    max_input_tokens = model.config.model_max_length - max_new_tokens
    system, rounds = _parse_messages(messages, split_role="user")
    system_tokens = tokenizer.encode(system)
    max_history_tokens = max_input_tokens - len(system_tokens)

    history_tokens = []
    for round in rounds[::-1]:
        round_tokens = []
        for message in round:
            if message["role"] == "user":
                round_tokens.append(model.generation_config.user_token_id)
            else:
                round_tokens.append(model.generation_config.assistant_token_id)
            round_tokens.extend(tokenizer.encode(message["content"]))
        if len(history_tokens) == 0 or len(history_tokens) + len(round_tokens) <= max_history_tokens:
            history_tokens = round_tokens + history_tokens  # concat left
            if len(history_tokens) < max_history_tokens:
                continue
        break

    input_tokens = system_tokens + history_tokens
    if messages[-1]["role"] != "assistant":
        input_tokens.append(model.generation_config.assistant_token_id)
    input_tokens = input_tokens[-max_input_tokens:]  # truncate left
    return input_tokens


# Llama & GPT2  
def build_chat_input_llama(model, tokenizer, messages: List[dict], max_new_tokens: int=0):
    def _parse_messages(messages, split_role="user"):
        system, text = "", ""
        for i, message in enumerate(messages):
            if message["role"] == "system":
                system = message["content"]
            if message["role"] == split_role:
                text = message["content"]
        return system, text

    max_tokens = max_new_tokens or model.generation_config.max_length
    system, text = _parse_messages(messages, split_role="user")
    system_tokens = tokenizer.encode(system)
    max_text_tokens = max_tokens - len(system_tokens)
    text_tokens = tokenizer.encode(text)

    input_tokens = system_tokens + text_tokens
    input_tokens = input_tokens[-max_tokens:]  # truncate left

    return input_tokens




def batch_prob_infer_fn(
    model,
    batch_input_ids, 
    batch_attention_mask, 
    prompt_lengths,  
    context_len=None,
    normalization="root"
):

    if context_len and batch_input_ids.shape[1] > context_len:
        batch_input_ids = batch_input_ids[:, -context_len:]
        batch_attention_mask = batch_attention_mask[:, -context_len:]
        prompt_lengths = [max(0, pl - (batch_input_ids.shape[1] - context_len)) for pl in prompt_lengths]

    model.eval()

    outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
    batch_probs = torch.ones(batch_input_ids.size(0), device=batch_input_ids.device)
    for i in range(batch_input_ids.size(0)):
        prompt_length = prompt_lengths[i]
        output_probs = torch.softmax(outputs.logits[i, prompt_length-1:-1, :], dim=-1)  
        token_indices = batch_input_ids[i, prompt_length:].unsqueeze(-1) 
        if token_indices.max() >= output_probs.shape[-1]:
            print(f"Index out of bounds: {token_indices.max()} not in [0, {output_probs.shape[-1]-1}]")
            continue  
        token_probs = torch.gather(output_probs, 1, token_indices).squeeze(-1) 

        valid_mask = batch_attention_mask[i, prompt_length:]  
        token_probs[~valid_mask.bool()] = 1
        action_prob = token_probs.prod()
 
        num_valid_tokens = valid_mask.sum()
        if num_valid_tokens > 0:
            action_prob **= (1.0 / num_valid_tokens)
        batch_probs[i] = action_prob

    return batch_probs


def batch_loss_infer_fn(model, batch_input_ids, batch_attention_mask, prompt_lengths, context_len=None):
    if context_len and batch_input_ids.shape[1] > context_len:
        batch_input_ids = batch_input_ids[:, -context_len:]
        batch_attention_mask = batch_attention_mask[:, -context_len:]
        prompt_lengths = [max(0, pl - (batch_input_ids.shape[1] - context_len)) for pl in prompt_lengths]

    model.eval()

    labels = batch_input_ids.clone()

    outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask, labels=labels)
    batch_losses = outputs.loss  
    print("batch_losses",batch_losses)

    return batch_losses



###############################
### Llama2
def inference_output_prob(
    model, tokenizer, messages, actions, device, context_len, max_batch_size=1
):
    num_action = len(actions)
    assert not model.config.is_encoder_decoder
    
    prompt_ids = build_chat_input_llama(model, tokenizer, messages)
    tokenized_actions = tokenizer(actions)
    input_ids, attention_mask = [], []
    
    max_action_length = sum(len(ids) for ids in tokenized_actions.input_ids)
    max_length = len(prompt_ids) + max_action_length  
    tokenizer.add_special_tokens({'pad_token': 'null'})

    prompt_len = []
    for i in range(num_action):
        #version 3 只关注当前action内容
        past_input_ids = prompt_ids + [i for list in tokenized_actions.input_ids[:i] for i in list]
        current_input_ids = {'input_ids': past_input_ids + tokenized_actions.input_ids[i]}
        current_input_ids = tokenizer.pad(current_input_ids, max_length=max_length, padding="max_length")
        prompt_len.append(len(past_input_ids))
        input_ids.append(current_input_ids['input_ids'])
        attention_mask.append(current_input_ids['attention_mask'])


    input_ids = torch.as_tensor(input_ids, device=device)
    attention_mask = torch.as_tensor(attention_mask, device=device)

    assert not torch.any(input_ids[:, 0] == tokenizer.pad_token_id)

    try:
        rets = []
        for i_l in range(0, num_action, max_batch_size):
            i_r = min(i_l + max_batch_size, num_action)
            batch_action_probs = batch_prob_infer_fn(
                model, 
                input_ids[i_l:i_r],
                attention_mask[i_l: i_r],
                prompt_len[i_l:i_r],
                context_len
            )
            if batch_action_probs==None:
                return None
            rets.extend(batch_action_probs.cpu().tolist())
            del batch_action_probs
            gc.collect()
            torch.cuda.empty_cache()
        return rets
    except torch.cuda.OutOfMemoryError as err:
        if max_batch_size == 1:
            print('OOM error occurs even batch_size=1')
            # raise err
            return None
        print(f'Batch Size reduced to {max_batch_size//2}')
        return inference_output_prob(
            model, tokenizer, messages, actions, device, context_len, max_batch_size//2)










##########################################################################################################################
# import gc

# def inference_output_prob(
#     model, tokenizer, messages, actions, context_len, max_batch_size=1
# ):
#     num_action = len(actions)
#     assert not model.module.config.is_encoder_decoder  # 使用model.module获取DataParallel中的实际模型对象
    
#     prompt_ids = build_chat_input(model.module, tokenizer, messages)  # 使用model.module获取DataParallel中的实际模型对象
#     tokenized_actions = tokenizer(actions, padding=True)
#     input_ids, attention_mask = [], []
#     for i in range(len(actions)):
#         prompt_attention_mask = [1] * len(prompt_ids)
#         input_ids.append(prompt_ids+tokenized_actions.input_ids[i])
#         attention_mask.append(prompt_attention_mask+tokenized_actions.attention_mask[i])
#     input_ids = torch.as_tensor(input_ids, device=next(model.parameters()).device)
#     attention_mask = torch.as_tensor(attention_mask, device=next(model.parameters()).device)
#     assert not torch.any(input_ids[:, 0] == tokenizer.pad_token_id)

#     try:
#         rets = []
#         for i_l in range(0, num_action, max_batch_size):
#             i_r = min(i_l + max_batch_size, num_action)
#             batch_action_probs = batch_prob_infer_fn(
#                 model.module,  # 使用model.module获取DataParallel中的实际模型对象
#                 input_ids[i_l:i_r],
#                 attention_mask[i_l: i_r],
#                 len(prompt_ids),
#                 context_len
#             )
#             rets.extend(batch_action_probs.cpu().tolist())
#         gc.collect()
#         torch.cuda.empty_cache()
#         return rets
#     except torch.cuda.OutOfMemoryError as err:
#         if max_batch_size == 1:
#             print('OOM error occurs even batch_size=1')
#             raise err
#         print(f'Batch Size reduced to {max_batch_size//2}')
#         return inference_output_prob(
#             model, tokenizer, messages, actions, context_len, max_batch_size//2)
