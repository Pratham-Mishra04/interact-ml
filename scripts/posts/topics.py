import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

df = pd.read_json('../../data/post_topics.json')

def preprocess(text):
    return text.lower()

df['post'] = df['post'].apply(preprocess)

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['categories'])

X_train, X_val, y_train, y_val = train_test_split(df['post'], y, test_size=0.2, random_state=42)

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# Load pre-trained BERT tokenizer and model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(mlb.classes_))

train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(list(X_val), truncation=True, padding=True, max_length=128)

class MultiLabelDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)
    
train_dataset = MultiLabelDataset(train_encodings, y_train)
val_dataset = MultiLabelDataset(val_encodings, y_val)

# pip install accelerate
    
training_args = TrainingArguments(
    output_dir='./ner',
    num_train_epochs=20,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

# Train the model
trainer.train()

output_dir = "../../models/posts/topics"

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

import pickle

with open(f'{output_dir}/mlb.pickle', 'wb') as f:
    pickle.dump(mlb, f)