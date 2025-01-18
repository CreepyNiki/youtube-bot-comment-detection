import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import os
import logging

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Load data
data = pd.read_csv("Sprachmodell/comments_with_label.csv")

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(data['Comment'], data['Label'], test_size=0.2)

# Convert labels to integers
label_to_int = {'nonbot': 0, 'bot': 1}
train_labels = train_labels.map(label_to_int)
val_labels = val_labels.map(label_to_int)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
# Add a padding token explicitly
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Tokenize data
train_encodings = tokenizer(list(train_texts), truncation=True, padding="longest", max_length=512)
val_encodings = tokenizer(list(val_texts), truncation=True, padding="longest", max_length=512)
train_dataset = YouTubeDataset(train_encodings, list(train_labels))
val_dataset = YouTubeDataset(val_encodings, list(val_labels))

model = AutoModelForSequenceClassification.from_pretrained("EleutherAI/gpt-neo-125M", num_labels=2)
model.resize_token_embeddings(len(tokenizer))

model.config.pad_token_id = tokenizer.pad_token_id

output_dir = "EleutherAI_gpt-neo-125M"
logging_dir = "EleutherAI_gpt-neo-125M"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(logging_dir, exist_ok=True)

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3.0,
    per_device_train_batch_size=16,  # Reduce batch size
    per_device_eval_batch_size=16,   # Reduce batch size
    overwrite_output_dir=True,
    save_total_limit=2,
    logging_dir=logging_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=0.0001,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train model
try:
    trainer.train()
except RuntimeError as e:
    logger.error(f"RuntimeError: {e}")
    torch.cuda.empty_cache()

model.save_pretrained("EleutherAI_gpt-neo-125M/model_saved")
tokenizer.save_pretrained("EleutherAI_gpt-neo-125M/tokenizer_saved")

# Evaluate model
results = trainer.evaluate()