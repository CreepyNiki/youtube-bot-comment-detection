import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, roc_auc_score

def to_number(labels):
    number_labels = []
    for label in labels:
        if label == 'nonbot':
            number_labels.append(0)
        elif label == 'bot':
            number_labels.append(1)
    return number_labels

data = pd.read_csv("Feature_based_machine_Learning/feature_table.csv", encoding="utf-8")

X = data.drop('Label', axis=1)
y = data['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

X_train_text = X_train['Comment'].astype(str).tolist()
X_test_text = X_test['Comment'].astype(str).tolist()
y_train = np.array(to_number(y_train))
y_test = np.array(to_number(y_test))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train_text)
vocab_size = len(tokenizer.word_index) + 1
tokenized_X_train = tokenizer.texts_to_sequences(X_train_text)
tokenized_X_test = tokenizer.texts_to_sequences(X_test_text)
MAX_LENGTH = max([len(x) for x in tokenized_X_train])
tokenized_X_train = pad_sequences(tokenized_X_train, maxlen=MAX_LENGTH, padding='post')
tokenized_X_test = pad_sequences(tokenized_X_test, maxlen=MAX_LENGTH, padding='post')

# Ensure the input data is a NumPy array
tokenized_X_train = np.array(tokenized_X_train)
tokenized_X_test = np.array(tokenized_X_test)

model = Sequential()
model.add(Input(shape=(MAX_LENGTH,)))
model.add(Embedding(vocab_size, 100))
model.add(Flatten())
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(tokenized_X_train, y_train, epochs=20, batch_size=32, verbose=1)

y_pred = model.predict(tokenized_X_test)
y_pred = [1 if x > 0.5 else 0 for x in y_pred]

# Print comments, true labels, and predicted labels
# for comment, true_label, predicted_label in zip(X_test_text, y_test, y_pred):
#     print(f"Comment: {comment}\nTrue Label: {true_label}, Predicted Label: {predicted_label}\n")

for comment, true_label, predicted_label in zip(X_test_text, y_test, y_pred):
    if true_label != predicted_label:
        print(f"Comment: {comment}\nTrue Label: {true_label}, Predicted Label: {predicted_label}\n")

print(classification_report(y_test, y_pred))