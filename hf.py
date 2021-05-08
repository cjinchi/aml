import torch
from transformers import BertForSequenceClassification, AdamW, BertTokenizer,TrainingArguments,Trainer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support,roc_auc_score
import numpy as np

label_encode_dict = {'fit': 0, 'large': 1, 'small': 2}

def to_one_hot(a):
    b = np.zeros((a.size, 3))
    b[np.arange(a.size), a] = 1
    return b

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    preds_one_hot = to_one_hot(preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(np.asarray(labels),preds_one_hot,multi_class='ovr')
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc':auc
    }


class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def read_train_data():
    global label_encode_dict
    df = pd.read_csv('./data/train.txt')
    df['review_summary'] = df['review_summary'].astype(str)
    df['review_text'] = df['review_text'].astype(str)
    review_texts = df[['review_summary', 'review_text']].agg(' '.join, axis=1).to_list()
    review_labels = df['fit'].map(label_encode_dict).to_list()
    return review_texts, review_labels


if __name__ == '__main__':
    texts, labels = read_train_data()
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    train_dataset = ReviewDataset(train_encodings,train_labels)
    val_dataset = ReviewDataset(val_encodings,val_labels)

    training_args = TrainingArguments(
        output_dir='./model/model1',
        num_train_epochs=3,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10
    )

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels = 3)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model()
    result = trainer.evaluate()
    print(result)



    # optimizer = AdamW(model.parameters(),lr=1e-5)

    # encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
    # input_ids = encoding['input_ids']
    # attention_mask = encoding['attention_mask']
    # outputs = model(input_ids,attention_mask=attention_mask,labels=labels)
    # print(input_ids)
    # print(attention_mask)
    # print(labels)
    # loss = outputs.loss
    # loss.backward()
    # optimizer.step()
