import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import os
import logging

# Quelle: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
class YouTubeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
# Laden von Daten
data = pd.read_csv("Extract_Comments/comments_with_label.csv")

# Train Test Split
train_texts, val_texts, train_labels, val_labels = train_test_split(data['Comment'], data['Label'], test_size=0.2)

print(train_texts)

# Konvertieren von Labes in Integer
label_to_int = {'nonbot': 0, 'bot': 1}
train_labels = train_labels.map(label_to_int)
val_labels = val_labels.map(label_to_int)

# Tokenizer laden
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
# Pad Token hinzufügen falls noch nicht vorhanden
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Tokenisieren von Daten
train_encodings = tokenizer(list(train_texts), truncation=True, padding="longest", max_length=512)
val_encodings = tokenizer(list(val_texts), truncation=True, padding="longest", max_length=512)

# Datasets erstellen
train_dataset = YouTubeDataset(train_encodings, list(train_labels))
val_dataset = YouTubeDataset(val_encodings, list(val_labels))

# Modell laden und pad token hinzufügen
model = AutoModelForSequenceClassification.from_pretrained("EleutherAI/gpt-neo-125M", num_labels=2)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

# Ausgabe- und Logging-Verzeichnisse erstellen
output_dir = "EleutherAI_gpt-neo-125M"
logging_dir = "EleutherAI_gpt-neo-125M"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(logging_dir, exist_ok=True)

# Argumenten des Trainings
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3.0,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16, 
    overwrite_output_dir=True,
    save_total_limit=2,
    logging_dir=logging_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=0.0001,
    weight_decay=0.01,
)

# Erstellen des Trainers
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Training des Modells
try:
    trainer.train()
except RuntimeError as e:
    torch.cuda.empty_cache()

# Modell und Tokenizer speichern
model.save_pretrained("EleutherAI_gpt-neo-125M/model_saved")
tokenizer.save_pretrained("EleutherAI_gpt-neo-125M/tokenizer_saved")