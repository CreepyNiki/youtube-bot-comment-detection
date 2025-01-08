import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
from collections import defaultdict
import os

def to_big_csv():
    df1 = pd.read_csv("Extract_Comments/fixed_bot.csv", encoding="utf-8")
    # print(df1['Comment'])
    df2 = pd.read_csv("Extract_Comments/fixed_nonbot.csv", encoding="utf-8")
    # print(df2['Comment'])
    with open('Extract_Comments/mixed.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'Comment', 'Published_At', 'Video_ID', 'Label'])
        for index, row in df1.iterrows():
            writer.writerow([index, row['Comment'], row['Published_At'], row['Video_ID'], 'bot'])
        for index, row in df2.iterrows():
            writer.writerow([index + len(df1), row['Comment'], row['Published_At'], row['Video_ID'], 'nonbot'])


    
def fix_csv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)

        for row in reader:
            writer.writerow(row)

def add_double_quotes(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)

        for row in reader:
            writer.writerow(row)

def check_duplicates(input_file):
    duplicates = defaultdict(list)
    
    with open(input_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip the header row
        for row in reader:
            comment = row[1]  # Assuming the comment is in the second column
            duplicates[comment].append(row)
    
    # Print duplicates
    for comment, rows in duplicates.items():
        if len(rows) > 1:
            print(f"Duplicate comment: {comment}")
            for row in rows:
                print(row)
            print()

fix_csv('Extract_Comments/bot.csv', 'Extract_Comments/fixed_bot.csv')
fix_csv('Extract_Comments/nonbot.csv', 'Extract_Comments/fixed_nonbot.csv')
to_big_csv()
add_double_quotes('Extract_Comments/mixed.csv', 'Extract_Comments/mixed_with_quotes.csv')
check_duplicates('Extract_Comments/mixed_with_quotes.csv')
os.remove('Extract_Comments/fixed_bot.csv')
os.remove('Extract_Comments/fixed_nonbot.csv')
os.remove('Extract_Comments/mixed.csv')