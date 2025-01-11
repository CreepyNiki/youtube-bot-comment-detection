import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import json
import emoji

# https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
def remove_emojis(text):
    return emoji.replace_emoji(text, replace='')

df = pd.read_csv("Extract_Comments/mixed_with_quotes.csv", encoding="utf-8")

df['comment_length'] = df['Comment'].apply(len)

df['Comment_without_emojis'] = df['Comment'].apply(remove_emojis)

df['word_count'] = df['Comment_without_emojis'].apply(lambda x: len(x.split()))

with open('Feature_based_machine_Learning/published_times.json', 'r') as f:
    published_video_times = json.load(f)

df['Video_Published_At'] = df['Video_ID'].apply(lambda x: published_video_times.get(x, None))

df['Published_At'] = pd.to_datetime(df['Published_At'], errors='coerce')

df['Video_Published_At'] = pd.to_datetime(df['Video_Published_At'], errors='coerce')

df['Time_Difference'] = df['Published_At'] - df['Video_Published_At']

df['!_occurring'] = df['Comment'].apply(lambda x: 1 if '!' in x else 0)

df['Dank_occurring'] = df['Comment'].apply(lambda x: 1 if 'Dank' in x else 0)

df['sentence_count'] = df['Comment_without_emojis'].apply(lambda x: len([s for s in re.split(r'[.!?]+|\s{2,}', x) if s.strip()]))

df['Comment_without_punctuation'] = df['Comment_without_emojis'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

df['average_word_length'] = df['Comment_without_punctuation'].apply(lambda x: sum(len(word) for word in x.split()) / len(x.split()) if x.split() else 0)

df['emoji_count'] = df['Comment'].apply(emoji.emoji_count)

df.drop(['Published_At', 'Video_Published_At', 'Comment_without_emojis', 'Comment_without_punctuation', 'Video_ID'], axis=1, inplace=True)

with open('Feature_based_machine_Learning/feature_table.csv', 'w', newline='', encoding='utf-8') as file:
    df.to_csv(file, index=False)

dank_occurrence_by_label = df.groupby('Label')['Dank_occurring'].mean()
exclamation_mark_occurrence_by_label = df.groupby('Label')['!_occurring'].mean()
word_count_by_label = df.groupby('Label')['word_count'].mean()
average_word_length_by_label = df.groupby('Label')['average_word_length'].mean()
time_difference_by_label = df.groupby('Label')['Time_Difference'].mean()

print(dank_occurrence_by_label)
print(exclamation_mark_occurrence_by_label)
print(word_count_by_label)
print(average_word_length_by_label)
print(time_difference_by_label)