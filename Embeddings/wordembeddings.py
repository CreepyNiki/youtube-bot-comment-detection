import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.manifold import TSNE

# Methode, die die Labels in Zahlen umwandelt
def to_number(labels):
    number_labels = []
    for label in labels:
        if label == 'bot':
            number_labels.append(0)
        elif label == 'nonbot':
            number_labels.append(1)
    return number_labels

# Methode, die die Embeddings visualisiert
def plot_embeddings(embeddings, labels, ids):
    # TSNE um Hochskalierte Daten auf 2 Dimensionen runterzuskalieren Quelle ChatGPT, https://scikit-learn.org/1.5/modules/generated/sklearn.manifold.TSNE.html
    tsne = TSNE(n_components=2, random_state=42)
    # Reduzierung der Dimensionen
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    for i, label in enumerate(labels):
        # Farbliche Unterscheidung der Labels
        color = 'blue' if label == 1 else 'red' # Bot: Rot, Nonbot: Blau
        # Erstellung des Punktes
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], c=color, alpha=0.6)
        # Erstellung des Labels für den Punkt
        plt.text(reduced_embeddings[i, 0] + 0.45, reduced_embeddings[i, 1], str(ids[i]), fontsize=8)

    # Achsenbeschriftung und Titel
    plt.xlabel('y')
    plt.ylabel('x')
    plt.title('Visualisierung der Embeddings')
    plt.show()

# Methode, die die Embeddings speichert
def save_embeddings(embeddings, ids):
    df = pd.DataFrame(embeddings)
    df['ID'] = ids
    df.to_csv('Embeddings/embeddings.csv', index=False)

# Laden und Tokenisieren der Daten
df = pd.read_csv("Extract_Comments/mixed_with_quotes.csv", encoding="utf-8")
train_labels = np.array(to_number(df['Label']))
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['Comment'])
vocab_size = len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences(df['Comment'])

MAX_LENGTH = max([len(sequence) for sequence in sequences])

tokenized_comments = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post')

# Erstellung des Modells
model = Sequential()
model.add(Input(shape=(MAX_LENGTH,)))
model.add(Embedding(vocab_size, 50, input_length=MAX_LENGTH))
model.add(Flatten())
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training des Modells
model.fit(tokenized_comments, train_labels, epochs=20, verbose=1)
model.summary()

# Trainierte Embeddings werden extrahiert
embeddings = model.layers[0].get_weights()[0]

# Visualisierung und Speicherung der Embeddings
plot_embeddings(embeddings, train_labels, df['ID'])
save_embeddings(embeddings, df['ID'])




