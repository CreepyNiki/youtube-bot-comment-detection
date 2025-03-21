import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

data = pd.read_csv("Extract_Comments/comments_with_label.csv")
# Herausfiltern von Bot-Kommentaren
botcomments = data[data['Label'] == 'bot']
botcomments = botcomments['Comment'].tolist()

# Methode zum Generieren von N-Grams Quelle: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
def generate_ngrams(comments, n):
    # CountVektorizer (https://scikit-learn.org/1.5/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
    # wird verwendet, um n-grams zu generieren -> ngram_range=(n, n) beschreibt den Parameter für die Anzahl der N-Grams
    vectorizer = CountVectorizer(ngram_range=(n, n))
    X = vectorizer.fit_transform(comments)
    # Wörter werden extrahiert für Transformation in Dictionary
    ngrams = vectorizer.get_feature_names_out()
    # Häufigkeitszählung der Wörter
    counts = X.toarray().sum(axis=0)
    # Fusionieren von Wörtern und deren Häufigkeit in ein Dictionary
    return dict(zip(ngrams, counts))

# Generieren von N-Grams
ngrams_freq = generate_ngrams(botcomments, n=1)

# Wordcloud wird aus den N-Grams erstellt
wordcloud = WordCloud(width=1000, height=600, background_color='white').generate_from_frequencies(ngrams_freq)

# Plotten der Wordcloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
