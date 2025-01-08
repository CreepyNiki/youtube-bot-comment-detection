import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.manifold import TSNE

def to_number(labels):
    number_labels = []
    for label in labels:
        if label == 'nonbot':
            number_labels.append(0)
        elif label == 'bot':
            number_labels.append(1)
    return number_labels

def plot_embeddings(embeddings, labels, ids):
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    for i, label in enumerate(labels):
        color = 'red' if label == 1 else 'blue' # Bot: Rot, Nonbot: Blau
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], c=color, alpha=0.6)
        plt.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1], str(ids[i]), fontsize=8)

    plt.xlabel('TSNE Dimension 1')
    plt.ylabel('TSNE Dimension 2')
    plt.title('TSNE Visualization of Comment Embeddings')
    plt.show()

def save_embeddings(embeddings, ids):
    df = pd.DataFrame(embeddings)
    df['ID'] = ids
    df.to_csv('Embeddings/embeddings.csv', index=False)



df = pd.read_csv("Extract_Comments/mixed_with_quotes.csv", encoding="utf-8")
train_labels = np.array(to_number(df['Label']))
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['Comment'])
vocab_size = len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences(df['Comment'])

MAX_LENGTH = max([len(sequence) for sequence in sequences])

tokenized_comments = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post')

model = Sequential()
model.add(Input(shape=(MAX_LENGTH,)))
model.add(Embedding(vocab_size, 50, input_length=MAX_LENGTH))
model.add(Flatten())
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(tokenized_comments, train_labels, epochs=20, verbose=1)
model.summary()

embedding_layer_index = 0 
embeddings = model.layers[embedding_layer_index].get_weights()[0]

plot_embeddings(embeddings, train_labels, df['ID'])

save_embeddings(embeddings, df['ID'])




