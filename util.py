
import re
import torch
from tqdm import tqdm

def compute_prob_sum(example):
    probs = example['conversation'][0].get('prob', [])
    if probs:
        return {'sum_prob': sum(probs)}
    else:
        return {'sum_prob': float('-inf')} 


def compute_prob_average(example):
    probs = example['conversation'][0].get('prob', [])
    print(probs)
    if probs:
        return {'avg_prob': sum(probs)/len(probs)}
    else:
        return {'avg_prob': float('-inf')} 
    

    

def random_sampling(train_data,select_num):
    sampled_indices = torch.randperm(len(train_data))[:select_num]  
    new_train_data = train_data.select(sampled_indices)
    return new_train_data


def extract_math_result(text):
    pattern = r'\\boxed{([^}]+)}'
    matches = re.findall(pattern, text)
    if matches:
        return matches
    else:
        return "No match found."


# def validate(model, tokenizer, data_test_raw):   ##for baichuan
#     model.eval()
#     sr = 0
#     data_test_raw = data_test_raw['train']
#     print("len",len(data_test_raw))
#     for j in tqdm(range(len(data_test_raw))):
#         data_test = data_test_raw[j]['conversation'][0]
#         System= data_test['system']
#         messages = [{"role": "system", "content": System}]
#         Input = data_test['input']
#         label = data_test['output']
#         messages.append({"role": "user", "content": Input})
#         # print("messages", messages)
#         response = model.chat(tokenizer, messages)
#         print("response", extract_math_result(response), "label", extract_math_result(label))
#         if extract_math_result(label)[0] in extract_math_result(response):
#             sr +=1
#             print('yes',sr)
#     return sr/len(data_test_raw)





def validate(model, tokenizer, data_test_raw):
    model.eval()
    sr = 0
    data_test_raw = data_test_raw['train']
    print("len",len(data_test_raw))
    for j in tqdm(range(len(data_test_raw))):
        data_test = data_test_raw[j]['conversation'][0]
        System= data_test['system']
        Input = data_test['input']
        label = data_test['output']

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = model.to(device)


        with torch.no_grad():
            input_text = [System + Input]
            input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048).to(device)  # 设置合理的 max_length，确保不超过模型的限制
            try:
                gen_tokens = model.generate(
                    **input_ids,
                    do_sample=True,
                    temperature=0.9,
                    max_new_tokens=300,  # 确保这个值在模型的处理范围内，根据实际任务调整
                )
            except Exception as e:
                print("An error occurred:", str(e))
                
            response_txt = tokenizer.batch_decode(gen_tokens)[0]


        print("\n response", response_txt)
        print("\n response", extract_math_result(response_txt))
        print("\n label", label)
        print("\n label", extract_math_result(label))

        if extract_math_result(label)[0] in extract_math_result(response_txt):
            sr +=1
            print('yes',sr)
    return sr/len(data_test_raw)


      
