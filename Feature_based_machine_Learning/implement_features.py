import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import json
import emoji

# Einlesen der Daten
df = pd.read_csv("Extract_Comments/mixed_with_quotes.csv", encoding="utf-8")

# https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
# Funktion zum Entfernen von Emojis aus den Kommentaren
def remove_emojis(text):
    return emoji.replace_emoji(text, replace='')

# Funktion zum Entfernen von Satzzeichen aus den Kommentaren
def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

# Funktion zur Zählung der Wortanzahl der Kommentare
def count_words(text):
    return len(text.split())

# Funktion zur Zählung der Wortlänge der Kommentare
def calculate_average_word_length(text):
    words = text.split()
    return sum(len(word) for word in words) / len(words) if words else 0

# Funktion zur Zählung der Sätze der Kommentare
def count_sentences(text):
    return len([s for s in re.split(r'[.!?]+|\s{2,}', text) if s.strip()])

# Funktion zur Zählung der Emojis in den Kommentaren
def count_emojis(text):
    return emoji.emoji_count(text)

# Funktion zur Extraktion der Uploadzeiten der Videos
def get_video_published_time(video_id):
    return published_video_times.get(video_id, None)

# Funktion zur Überprüfung des Vorkommens des Wortes "Dank" in den Kommentaren
def check_dank_occurrence(comment):
    return 1 if 'Dank' in comment else 0

# Funktion zur Überprüfung des Vorkommens von Ausrufezeichen in den Kommentaren
def check_exclamation_mark_occurrence(comment):
    return 1 if '!' in comment else 0



# Featureextraktion der Kommentarlänge
df['comment_length'] = df['Comment'].apply(len)

# Featureextraktion des Kommentars ohne Emojis
df['Comment_without_emojis'] = df['Comment'].apply(remove_emojis)

# Umwandlung des Kommentars in einen Kommentar ohne Satzzeichen
df['Comment_without_punctuation'] = df['Comment_without_emojis'].apply(remove_punctuation)

# Featureextraktion der Wortanzahl
df['word_count'] = df['Comment_without_emojis'].apply(count_words)

# Featureextraktion der durchschnittlichen Wortlängen in Kommentaren
df['average_word_length'] = df['Comment_without_punctuation'].apply(calculate_average_word_length)

# Featureextraktion der Anzahl der Sätze in Kommentaren
df['sentence_count'] = df['Comment_without_emojis'].apply(count_sentences)

# Featureextraktion der Anzahl der Emojis in Kommentaren
df['emoji_count'] = df['Comment'].apply(count_emojis)

# Einlesen des JSON-Files mit den Uploadzeiten der Videos
with open('Feature_based_machine_Learning/published_times.json', 'r') as f:
    published_video_times = json.load(f)

# Featureextraktion der Uploadzeiten der Videos nach Extraktion aus dem Dictionary der Video-IDs
df['Video_Published_At'] = df['Video_ID'].apply(get_video_published_time)

# Umwandlung der Postzeiten des Kommentars in ein Datetime-Format
df['Published_At'] = pd.to_datetime(df['Published_At'], errors='coerce')

# Umwandlung der Uploadzeiten der Videos in ein Datetime-Format
df['Video_Published_At'] = pd.to_datetime(df['Video_Published_At'], errors='coerce')

# Featureextraktion der Zeitdifferenz zwischen Veröffentlichung des Kommentars und Veröffentlichung des Videos Video
df['Time_Difference'] = df['Published_At'] - df['Video_Published_At']

# Featureextraktion in Bezug auf das Vorkommen des Wortes "Dank" in Kommentaren
df['Dank_occurring'] = df['Comment'].apply(check_dank_occurrence)

# Featureextraktion in Bezug auf das Vorkommen von Ausrufezeichen in Kommentaren
df['!_occurring'] = df['Comment'].apply(check_exclamation_mark_occurrence)



# Droppen der Spalten, die nicht mehr benötigt werden
df.drop(['Published_At', 'Video_Published_At', 'Comment_without_emojis', 'Comment_without_punctuation', 'Video_ID'], axis=1, inplace=True)

# Speichern der Featuretabelle in ein CSV-File
with open('Feature_based_machine_Learning/feature_table.csv', 'w', newline='', encoding='utf-8') as file:
    df.to_csv(file, index=False)

# Berechnung der Mittelwerte und Prozentzahlen der einzelnen Features
comment_length_by_label = df.groupby('Label')['comment_length'].mean()
word_count_by_label = df.groupby('Label')['word_count'].mean()
average_word_length_by_label = df.groupby('Label')['average_word_length'].mean()
average_sentence_count_by_label = df.groupby('Label')['sentence_count'].mean()
emoji_count_by_label = df.groupby('Label')['emoji_count'].mean()
time_difference_by_label = df.groupby('Label')['Time_Difference'].mean()
dank_occurrence_percentage_by_label = df.groupby('Label')['Dank_occurring'].mean() * 100
exclamation_mark_occurrence_percentage_by_label = df.groupby('Label')['!_occurring'].mean() * 100

# Ausgabe der Mittelwerte der Kommentarlänge für die verschiedenen Label der Kommentare
print(comment_length_by_label)
# Ausgabe der Mittelwerte der Satzanzahl für die verschiedenen Label der Kommentare
print(average_sentence_count_by_label)
# Ausgabe der Mittelwerte der Wortanzahl für die verschiedenen Label der Kommentare
print(word_count_by_label)
# Ausgabe der Mittelwerte der Worlänge für die verschiedenen Label der Kommentare
print(average_word_length_by_label)
# Ausgabe der Mittelwerte der Emojianzahl für die verschiedenen Label der Kommentare
print(emoji_count_by_label)
# Ausgabe des Mittelwertes des Zeitunterschiedes für die verschiedenen Label der Kommentare
print(time_difference_by_label)
# Ausgabe der Prozentwerte für die verschiedenen Label der Kommentare, die das Wort "Dank" enthalten
print(dank_occurrence_percentage_by_label)
# Ausgabe der Prozentwerte für die verschiedenen Label der Kommentare, die ein Ausrufezeichen enthalten
print(exclamation_mark_occurrence_percentage_by_label)