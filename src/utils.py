import os
import re
import json
from tqdm import tqdm
from openai import AzureOpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM


class BaseLLM(object):
    def __init__(self, llm_name):
        self.llm_name = llm_name
        if llm_name.lower() in ['llama3.1', 'llama3']:
            self.llm_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
            self.llm_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", device_map='auto')
        elif llm_name.lower() in ['gpt-4-turbo']:
            self.client = AzureOpenAI(
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
                api_key=os.getenv("AZURE_OPENAI_API_KEY"), # Obtained from the team's key manager
                api_version="2024-05-01-preview"
            )
        else:
            print("Not find LLM!")
    
    def __generate_LLM__(self, query, num_tokens_num):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ]

        input_ids = self.llm_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors='pt'
        ).to(self.llm_model.device)

        terminators = [
            self.llm_tokenizer.eos_token_id,
            self.llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        self.llm_model.config.pad_token_id = self.llm_model.config.eos_token_id

        outputs = self.llm_model.generate(
            input_ids,
            max_new_tokens=num_tokens_num,
            eos_token_id=terminators,
            pad_token_id=self.llm_tokenizer.eos_token_id,
        )

        response = outputs[0][input_ids.shape[-1]:]
        generated_text = self.llm_tokenizer.decode(response, skip_special_tokens=True)

        return generated_text

    def __generate_GPT__(self, query, num_tokens_num):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.llm_name, # Model deployment name      
                max_tokens = num_tokens_num,
                messages=messages
            )
        except Exception as e:
            return 'None'

        return response

    
    def generate(self, query, new_tokens_num):

        if self.llm_name in ['llama3.1', 'llama3']:
            return self.__generate_LLM__(query=query, num_tokens_num=new_tokens_num)
        elif self.llm_name in ['gpt-4-turbo']:
            return self.__generate_GPT__()



class QADataset:
    def __init__(self, data, dir="dataset/"):
        self.data = data.lower().split("_")[0]
        benchmark = json.load(open(os.path.join(dir, "benchmark.json")))
        if self.data not in benchmark:
            raise KeyError("{:s} not supported".format(data))
        
        self.dataset = benchmark[self.data]
        self.index = sorted(self.dataset.keys())

    def __process_data__(self, key):
        data = self.dataset[self.index[key]]
        question = data["question"]
        choices = [v for k, v in data["options"].items()]

        options = [" A: ", " B: ", " C: ", " D: "]

        text = question + "\n"
        for j in range(len(choices)):
            text += "{} {}\n".format(options[j], choices[j])

        answer = data["answer"].strip()
        label_index = ord(answer) - ord('A')
        answer_content = choices[label_index]

        return {"text": text, "answer": answer, "answer_index": label_index, "answer_content": answer_content}

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, key):
        if type(key) == int:
            return self.__process_data__(key)
        elif type(key) == slice:
            return [self.__getitem__(i) for i in range(self.__len__())[key]]
        else:
            raise KeyError("Key type not supported.")
    
class MedDDxLoader:
    def __init__(self, data, dir="dataset/"):
        benchmark = self.process_dataset(dir)
        self.data = data
        if self.data not in benchmark:
            raise KeyError("{:s} not supported".format(data))
        self.dataset = benchmark[self.data]
        print(self.dataset)
        self.index = sorted(self.dataset.keys())
    
    def process_dataset(self, dir):       

        benchmark = json.load(open(os.path.join(dir, "MedDDx.json")))
        
        data_dict = {'MedDDx':{}, 'MedDDx-Basic':{}, 'MedDDx-Intermediate':{}, 'MedDDx-Expert': {}}

        for idx, b in enumerate(benchmark):
            if b['sim_level_std'] > 0.04:
                data_dict['MedDDx-Basic'][idx] = b
            elif b['sim_level_std'] < 0.02:
                data_dict['MedDDx-Expert'][idx] = b
            else:
                data_dict['MedDDx-Intermediate'][idx] = b
            data_dict['MedDDx'][idx] = b
        
        return data_dict
        
    def __process_data__(self, key):
        data = self.dataset[self.index[key]]

        answer = data["answer"].strip()
        label_index = ord(answer) - ord('A')
        
        return {"text": data['query'], "answer": answer, "answer_index": label_index}

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, key):
        if type(key) == int:
            return self.__process_data__(key)
        elif type(key) == slice:
            return [self.__getitem__(i) for i in range(self.__len__())[key]]
        else:
            raise KeyError("Key type not supported.")