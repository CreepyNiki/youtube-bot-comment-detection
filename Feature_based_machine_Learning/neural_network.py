import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense, Dropout, SimpleRNN, LSTM, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import L1L2, L2
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

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

number_classes = len(y.unique())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

X_train_text = X_train['Comment'].astype(str).tolist()
X_test_text = X_test['Comment'].astype(str).tolist()
y_train = np.array(to_number(y_train))
y_test = np.array(to_number(y_test))

# One-hot encode the target labels
y_train = to_categorical(y_train, num_classes=number_classes)
y_test = to_categorical(y_test, num_classes=number_classes)

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

# Split the training data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(tokenized_X_train, y_train, test_size=0.2, random_state=42)

regularizer = L1L2(l1=1e-5, l2=1e-5)

model = Sequential()
model.add(Input(shape=(MAX_LENGTH,)))
model.add(Embedding(vocab_size, 100))
model.add(Flatten())
model.add(Dense(66, activation='tanh', kernel_regularizer=regularizer, bias_regularizer=L2(1e-5), activity_regularizer=L2(1e-4)))
model.add(Dropout(0.3))
model.add(Dense(33, activation='tanh', kernel_regularizer=regularizer, bias_regularizer=L2(1e-5), activity_regularizer=L2(1e-4)))
model.add(Dropout(0.3))
model.add(Dense(number_classes, activation='softmax'))
model.compile(optimizer=Adam(learning_rate = 0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model using both training and validation data
history = model.fit(X_train_split, y_train_split, epochs=20, batch_size=16, validation_split=0.1, verbose=1)

y_pred = model.predict(tokenized_X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Print only the wrong label assignments
for comment, true_label, predicted_label in zip(X_test_text, y_test_labels, y_pred):
    if true_label != predicted_label:
        print(f"Comment: {comment}\nTrue Label: {true_label}, Predicted Label: {predicted_label}\n")

print(classification_report(y_test_labels, y_pred))

# Plot training and validation accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.show()