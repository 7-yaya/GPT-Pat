import json
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import time

class BaiKeDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.data_path = data_path
        self.tokenizer = tokenizer
        if tokenizer.model_max_length > 1e9:
            self.max_length = max_length
        else:
            self.max_length = tokenizer.model_max_length
        self.lines = []
        with open(data_path, 'r', encoding="utf-8") as f:
            for data in json.load(f):
                if(data["human_answers"]!="" and data["generated_answer_human"]!=""):
                    # print("human")
                    self.lines.append({"text":data["human_answers"],"label":0})
                if(data["chatgpt_answer"]!="" and data["generated_answer_gpt"]!=""):
                    # print("chat")
                    self.lines.append({"text":data["chatgpt_answer"],"label":1})
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        text, label = self.lines[index]['text'],self.lines[index]['label']
        text_input = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": text_input['input_ids'].flatten(),
            "attention_mask": text_input['attention_mask'].flatten(),
            "label": torch.tensor(label, dtype=torch.long)
        }
    
class BaiKeDataset_ori(Dataset):
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.data_path = data_path
        self.tokenizer = tokenizer
        if tokenizer.model_max_length > 1e9:
            self.max_length = max_length
        else:
            self.max_length = tokenizer.model_max_length
        self.lines = []
        with open(data_path, 'r', encoding="utf-8") as f:
            self.lines = [(data["human_answers"],data["chatgpt_answer"]) for data in json.load(f)]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        human_answer, chatgpt_answer = self.lines[index]
        human_answer_input = self.tokenizer(
            human_answer,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        chatgpt_answer_input = self.tokenizer(
            chatgpt_answer,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "human_answer_input": human_answer_input['input_ids'].flatten(),
            "human_answer_attention_mask": human_answer_input['attention_mask'].flatten(),
            "human_answer_label": torch.tensor(0, dtype=torch.long),
            "chatgpt_answer_input": chatgpt_answer_input['input_ids'].flatten(),
            "chatgpt_answer_attention_mask": chatgpt_answer_input['attention_mask'].flatten(),
            "chatgpt_answer_label": torch.tensor(1, dtype=torch.long),
        }

class PreClassificationDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.data_path = data_path
        self.tokenizer = tokenizer
        if tokenizer.model_max_length > 1e9:
            self.max_length = max_length
        else:
            self.max_length = tokenizer.model_max_length
        self.lines = []
        with open(data_path, 'r', encoding="utf-8") as f:
            self.lines = json.load(f)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        text, label = self.lines[index]['text'], self.lines[index]['label']
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs['input_ids'].flatten(),
            "attention_mask": inputs['attention_mask'].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }

class PairInputDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.data_path = data_path
        self.tokenizer = tokenizer
        if tokenizer.model_max_length > 1e9:
            self.max_length = max_length
        else:
            self.max_length = tokenizer.model_max_length
        self.lines = []
        with open(data_path, 'r', encoding="utf-8") as f:
            for line in json.load(f):
                text1_a, text1_b = line['human_answers'], line['generated_answer_human']
                text2_a, text2_b = line['chatgpt_answer'], line['generated_answer_gpt']
                if text1_a!="" and text1_b!="":
                    self.lines.append(
                        {"text_pair":(text1_a, text1_b), "label":0}
                    )
                if text2_a!="" and text2_b!="":
                    self.lines.append(
                        {"text_pair":(text2_a, text2_b), "label":1}
                    )

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        (text_a, text_b), label = self.lines[index]['text_pair'], self.lines[index]['label']
        inputs = self.tokenizer(
            text=text_a,
            text_pair=text_b,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs['input_ids'].flatten(),
            "attention_mask": inputs['attention_mask'].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }


class SiameseLinearDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.data_path = data_path
        self.tokenizer = tokenizer
        if tokenizer.model_max_length > 1e9:
            self.max_length = max_length
        else:
            self.max_length = tokenizer.model_max_length
        self.lines = []
        with open(data_path, 'r', encoding="utf-8") as f:
            for line in json.load(f):
                text1_a, text1_b = line['human_answers'], line['generated_answer_human']
                text2_a, text2_b = line['chatgpt_answer'], line['generated_answer_gpt']
                if text1_a!="" and text1_b!="":
                    self.lines.append(
                        {"text_pair":(text1_a, text1_b), "label":0}
                    )
                if text2_a!="" and text2_b!="":
                    self.lines.append(
                        {"text_pair":(text2_a, text2_b), "label":1}
                    )

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        (text_a, text_b), label = self.lines[index]['text_pair'], self.lines[index]['label']
        inputs_a = self.tokenizer(
            text_a,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs_b = self.tokenizer(
            text_b,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs_a['input_ids'].flatten(),
            "input_ids_b": inputs_b['input_ids'].flatten(),
            "attention_mask": inputs_a['attention_mask'].flatten(),
            "attention_mask_b": inputs_b['attention_mask'].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }