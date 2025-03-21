import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense, Dropout, SimpleRNN, LSTM, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import L1L2, L2
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Methode, die die random Baseline berechnet, indem sie den einzelnen Daten entweder 0 oder 1 zuweist
def random_baseline(data):
    random_preds = np.random.randint(0, 2, len(data))
    print(classification_report(y_test_labels, random_preds))

# Methode, die die Labels in Zahlen umwandelt
def to_number(labels):
    number_labels = []
    for label in labels:
        if label == 'bot':
            number_labels.append(0)
        elif label == 'nonbot':
            number_labels.append(1)
    return number_labels

# Vortrainierte Embeddings laden
embeddings_df = pd.read_csv("Embeddings/embeddings.csv")
# Laden aller Werte außer der letzten Spalte, da diese die ID enthält
embedding_matrix = embeddings_df.iloc[:, :-1].values

# Laden der Daten mit verschiedenen Datengrößen
data = pd.read_csv("Feature_based_machine_Learning/feature_table.csv", encoding="utf-8")
# data = pd.concat([data.head(200), data.tail(200)])
# data = pd.concat([data.head(150), data.tail(150)])
# data = pd.concat([data.head(100), data.tail(100)])
# data = pd.concat([data.head(25), data.tail(25)])




# die Spalte 'Label' wird aus den Features entfernt und in y (also dem Target Label) gespeichert
# axis=1 bezieht sich auf Spalten und nicht auf Zeilen
X = data.drop('Label', axis=1)
y = data['Label']

# Anzahl der Klassen im Target Label
number_classes = len(y.unique())

# Train Test Split mit 25% Testdaten und 75% Trainingsdaten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Comment wird zu String umgewandelt und in Listen gespeichert als Grundlage für Tokenisierung
X_train_text = X_train['Comment'].astype(str).tolist()
X_test_text = X_test['Comment'].astype(str).tolist()

# Hier Anwendung der Methode sodass die Labels zu Integers umgewandelt werden
y_train = np.array(to_number(y_train))
y_test = np.array(to_number(y_test))

# One hot encoding damit das Modell die Daten besser verarbeiten kann
y_train = to_categorical(y_train, num_classes=number_classes)
y_test = to_categorical(y_test, num_classes=number_classes)

# Tokenizer wird erstellt und die Daten werden darauf gefittet
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train_text)
vocab_size = len(tokenizer.word_index) + 1

# Die Daten werden in Sequenzen umgewandelt und gepaddet
tokenized_X_train = tokenizer.texts_to_sequences(X_train_text)
tokenized_X_test = tokenizer.texts_to_sequences(X_test_text)
MAX_LENGTH = max([len(x) for x in tokenized_X_train])
tokenized_X_train = pad_sequences(tokenized_X_train, maxlen=MAX_LENGTH, padding='post')
tokenized_X_test = pad_sequences(tokenized_X_test, maxlen=MAX_LENGTH, padding='post')

# Umwandlung der Input Daten 
tokenized_X_train = np.array(tokenized_X_train)
tokenized_X_test = np.array(tokenized_X_test)


# Erneuter Testsplit um auch das Validation Set zu haben
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(tokenized_X_train, y_train, test_size=0.25, random_state=42)

# Quelle: ChatGPT:

# Entfernen der NaN Werte
embeddings_df = embeddings_df.dropna(subset=[embeddings_df.columns[-1]])
embedding_dim = 50
# Dictionary, dass die Embeddings enthält Key: ID Value: Embedding Vector
embedding_dict = {}
for index, row in embeddings_df.iterrows():
    embedding_vector = row[:-1].values  
    embedding_id = int(row[-1])
    embedding_dict[embedding_id] = embedding_vector

# Erstellung einer neuen Embedding Matrix, die die Embeddings enthält
new_embedding_matrix = np.zeros((vocab_size, embedding_dim))

# Die Embeddings werden in die neue Matrix eingefügt
for word, i in tokenizer.word_index.items():
    if i < vocab_size:
        embedding_vector = embedding_dict.get(i)
        if embedding_vector is not None:
            new_embedding_matrix[i] = embedding_vector

regularizer = L1L2(l1=1e-5, l2=1e-5)

# Erstellung des FFNNs und der 2 Hidden Layers
model = Sequential()
model.add(Input(shape=(MAX_LENGTH,)))
model.add(Embedding(input_dim=vocab_size, output_dim=new_embedding_matrix.shape[1], input_length=tokenized_X_train.shape[1], weights=[new_embedding_matrix], trainable=False))
model.add(Flatten())
model.add(Dense(66, activation='tanh', kernel_regularizer=regularizer, bias_regularizer=L2(1e-5), activity_regularizer=L2(1e-4)))
model.add(Dropout(0.3))
model.add(Dense(33, activation='tanh', kernel_regularizer=regularizer, bias_regularizer=L2(1e-5), activity_regularizer=L2(1e-4)))
model.add(Dropout(0.3))
model.add(Dense(number_classes, activation='softmax'))

# Kompilierung des Modells
model.compile(optimizer=Adam(learning_rate = 0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Eigentliches Training des Modells mit Validation Split
history = model.fit(X_train_split, y_train_split, epochs=20, batch_size=16, validation_split=0.1, verbose=1)

# Vorhersage des Modells
y_pred = model.predict(tokenized_X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Alle falschen Labelzuweisungen werden geprintet
# Zip nimmt die Elemente aus den Listen und packt sie in Tupel um dann über sie dann zu iterieren
for comment, true_label, predicted_label in zip(X_test_text, y_test_labels, y_pred):
    if true_label != predicted_label:
        print(f"Comment: {comment}\nTrue Label: {true_label}, Predicted Label: {predicted_label}\n")

# Ausprinten des Classification Reports, um die Evaluationsmetriken auslesen zu können
print(classification_report(y_test_labels, y_pred))

# Anwendung der Baseline Methode
# random_baseline(y_test)