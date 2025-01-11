import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import os
# Lade die Daten
data = pd.read_csv("Sprachmodell/comments_with_label.csv")

# Datenaufteilung in Training und Test
train_texts, val_texts, train_labels, val_labels = train_test_split(data['Comment'], data['Label'], test_size=0.2)


# Tokenizer laden
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Add a padding token to the tokenizer
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Tokenisierung der Texte
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)


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

# Dataset-Objekte erstellen
train_dataset = YouTubeDataset(train_encodings, list(train_labels))
val_dataset = YouTubeDataset(val_encodings, list(val_labels))

# Modell für Klassifikation
model = AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=2)  # 2 Klassen: Bot oder Kein Bot

# Resize token embeddings to accommodate new tokens
model.resize_token_embeddings(len(tokenizer))

# Ensure the directories exist
output_dir = os.path.abspath('Sprachmodell/results')
logging_dir = os.path.abspath('Sprachmodell/logs')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(logging_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=None,         # Speichere das Modell
    num_train_epochs=3,             # Anzahl der Epochen
    per_device_train_batch_size=16, # Batch-Größe
    per_device_eval_batch_size=64,
    warmup_steps=500,               # Lernrate langsam erhöhen
    weight_decay=0.01,              # Regulierung
    logging_dir=None,           # Logs speichern
    logging_steps=10,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
