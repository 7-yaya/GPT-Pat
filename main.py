'''
@desc: train a Roberta model for text pair classification task in siamese way.
@usage: redifine the parameters and run the script
@others: the Roberta model with a singel linear is slightly better than the one with a RobertaClassificationHead-like layer.
'''
import os
import logging
import torch
import numpy as np
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear, Dropout, CrossEntropyLoss, CosineSimilarity
from transformers import AutoTokenizer, RobertaPreTrainedModel, RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import tokenization_utils
from transformers import TrainingArguments, Trainer
from dataset import SiameseLinearDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO,format="%(asctime)s| %(levelname)s| %(message)s")
tokenization_utils.logger.setLevel('ERROR')

# <parameters start>
model_name = "xlm_roberta-base"
model_save_path = "models/en-siamese-LT"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data_path = "data/HC3_en_train.json"
valid_data_path = "data/HC3_en_valid.json"
test_data_path = "data/HC3_en_test.json"
batch_size = 4
gradient_accumulation_steps = 8
learning_rate = 5e-5
weight_decay = 0
epochs = 20
# <parameters end>

if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions),
        'recall': recall_score(labels, predictions),
        'f1': f1_score(labels, predictions),
    }

class MyRobertaForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dense = torch.nn.Sequential(
            Dropout(classifier_dropout),
            Linear(config.hidden_size*2+1, config.hidden_size),

        )
        self.activate = torch.nn.Tanh()
        self.classifier = torch.nn.Sequential(
            Dropout(classifier_dropout),
            Linear(config.hidden_size, config.num_labels),
        )

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_ids_b: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        attention_mask_b: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        token_type_ids_b: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_ids_b: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs_a = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        outputs_b = self.roberta(
            input_ids_b,
            attention_mask=attention_mask_b,
            token_type_ids=token_type_ids_b,
            position_ids=position_ids_b,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output_a = outputs_a.last_hidden_state[:, 0, :] # get [CLS] embedding
        sequence_output_b = outputs_b.last_hidden_state[:, 0, :]
        sequence_output = torch.cat((sequence_output_a, sequence_output_b), dim=-1)
        cos = CosineSimilarity(dim=1)
        cos_bias = cos(sequence_output_a, sequence_output_b).unsqueeze(-1)
        sequence_output_cos =  torch.cat((sequence_output, cos_bias), dim=-1)

        emb = self.dense(sequence_output_cos)
        emb = self.activate(emb)
        logits = self.classifier(emb)

        # softmax
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        if not return_dict:
            output = (logits,) + outputs_a[2:] + outputs_b[2:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


def fine_tune_and_evaluate():
    model = MyRobertaForSequenceClassification.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # read data
    logging.info("[-] reading train and valid data...")
    train_dataset = SiameseLinearDataset(train_data_path, tokenizer)
    valid_dataset = SiameseLinearDataset(valid_data_path, tokenizer)
    test_dataset = SiameseLinearDataset(test_data_path, tokenizer)
    
    training_args = TrainingArguments(
        report_to = ['wandb'],
        output_dir=model_save_path,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        overwrite_output_dir=False,
        max_steps=5000,
        save_strategy="steps",
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=100,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        optimizers=(optimizer, None),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    logging.info("[-] running test...")
    predict_results = trainer.predict(test_dataset=test_dataset)
    logging.info("[-] metrics: "+str(predict_results.metrics))


def evaluate_model(model_path:str = None, internal_test_data_path:str = None):
    if model_path:
        logging.info(f"[-] evaluating model {model_path}")
        model = MyRobertaForSequenceClassification.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        logging.info(f"[-] evaluating model {model_name}")
        model = MyRobertaForSequenceClassification.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    if not internal_test_data_path:
        internal_test_data_path = test_data_path
    logging.info(f"[-] reading test data {internal_test_data_path}")
    test_dataset = SiameseLinearDataset(internal_test_data_path, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    logging.info("[-] running test...")
    model_preds_logits = []
    ground_truth = []
    test_looper = tqdm(test_dataloader)
    for batch in test_looper:
        label = batch.pop('label')
        batch = {k:v.to(device) for k,v in batch.items()}
        outputs = model(**batch)
        model_preds_logits += outputs.logits.cpu().tolist()
        ground_truth += label.cpu().tolist()
        test_looper.set_description(f'[-] testing')

    metrics = compute_metrics(eval_pred=(model_preds_logits, ground_truth))
    logging.info(f"[-] metrics: {metrics}")
    res = metrics

    model_preds1 = np.argmax(model_preds_logits, axis=-1)
    human_acc = accuracy_score([p for p, t in zip(model_preds1, ground_truth) if t == 0], [t for t in ground_truth if t == 0])
    machine_acc = accuracy_score([p for p, t in zip(model_preds1, ground_truth) if t == 1], [t for t in ground_truth if t == 1])
    # 输出结果
    logging.info(f"Accuracy of Human texts: {human_acc}")  
    logging.info(f"Accuracy of Machine texts: {machine_acc}")

    return res

def test_model(model, tokenizer, text_pair:List):
    model.eval()
    inputs_a = tokenizer(
            text_pair[0],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
    inputs_b = tokenizer(
        text_pair[1],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    outputs = model(input_ids=inputs_a['input_ids'],
                    input_ids_b=inputs_b['input_ids'],
                    attention_mask=inputs_a['attention_mask'],
                    attention_mask_b=inputs_b['attention_mask_b'])
    res = outputs.logits.argmax(dim=1).cpu().tolist()[0]
    res_txt = {0:"human",1:"AI"}[res]
    logging.info(f"[-] predict: {res_txt}, logits: {outputs.logits.tolist()}")
    return res


if __name__ == "__main__":

    chkp = os.path.basename(os.listdir(model_save_path)[0])
    best_model_path = f"{model_save_path}/{chkp}"
    evaluate_model(model_path=best_model_path,internal_test_data_path="data/HC3_en_test.json")
    evaluate_model(model_path=best_model_path,internal_test_data_path="data/wiki.json")
    evaluate_model(model_path=best_model_path,internal_test_data_path="data/cc_news.json")
    evaluate_model(model_path=best_model_path,internal_test_data_path="data/covidcm.json")
    evaluate_model(model_path=best_model_path,internal_test_data_path="data/aclabs.json")

    # evaluate_model(model_path=best_model_path,internal_test_data_path="data/polish/ori_human.json")
    # evaluate_model(model_path=best_model_path,internal_test_data_path="data/polish/human_polish.json")

    # evaluate_model(model_path=best_model_path,internal_test_data_path="data/polish/ori_gpt.json")
    # evaluate_model(model_path=best_model_path,internal_test_data_path="data/polish/gpt_polish.json")
    
    # evaluate_model(model_path=best_model_path,internal_test_data_path="data/translation/Deepl_translated_human.json")
    # evaluate_model(model_path=best_model_path,internal_test_data_path="data/translation/Deepl_ori_human.json")
    # evaluate_model(model_path=best_model_path,internal_test_data_path="data/translation/Deepl_ori_gpt.json")
    # evaluate_model(model_path=best_model_path,internal_test_data_path="data/translation/Deepl_translated_gpt.json")

    # evaluate_model(model_path=best_model_path,internal_test_data_path="data/translation/baidu_ori_human.json")
    # evaluate_model(model_path=best_model_path,internal_test_data_path="data/translation/baidu_translated_human.json")
     # evaluate_model(model_path=best_model_path,internal_test_data_path="data/translation/baidu_ori_gpt.json")
    # evaluate_model(model_path=best_model_path,internal_test_data_path="data/translation/baidu_translated_gpt.json")