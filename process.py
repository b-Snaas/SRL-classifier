import pandas as pd
import csv
from pandas import ExcelWriter
import numpy as np
import spacy
import benepar
import warnings
from nltk import ParentedTree

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
warnings.filterwarnings("ignore", message="<class 'torch_struct.distributions.TreeCRF'> does not define `arg_constraints`.")

dataset = ''

for i in range(2):

    if i == 0:
        dataset = 'test'
    elif i == 1:
        dataset = 'train'

    print(f'Processing {dataset} dataset...')

    # Read the file contents and assign under 'annotations'
    with open(f'en_ewt-up-{dataset}.conllu', mode="r", encoding="utf-8") as data:    
        annotations = data.readlines()

    # Open a new file to write the processed data

    with open(f'processed_{dataset}.csv', 'w', encoding= "utf-8") as f:
        for line in annotations:
            if not line.startswith("#"):
                f.write(line)


    # Open the CSV file for reading
    with open(f'processed_{dataset}.csv', newline='') as csvfile:

        # Create a CSV reader object
        reader = csv.reader(csvfile)

        # Initialize variables
        data = []
        split_data = []

        # Loop through each row in the CSV file
        for row in reader:

            # If the row is blank, save the data to split_data list
            if not row:
                split_data.append(data)
                data = []

            # Otherwise, append the row to the data list
            else:
                data.append(row)

        # Append any remaining data to the split_data list
        if data:
            split_data.append(data)

    # Create a dataframe for each sentence
    sentence_dataframes = []

    for i in split_data:
        try:
            # split the sentences we extracted
            split_sentence = [x[0].split('\t') for x in i]

            # Check the columnlength of this particular sentence
            sentence_checker = 0
            columnlength = len(split_sentence[sentence_checker])

            while columnlength < 12:
                sentence_checker += 1
                columnlength = len(split_sentence[sentence_checker])

            # Create a df with that many columns
            col_names = [f"col{i}" for i in range(columnlength)]
            df = pd.DataFrame(split_sentence, columns=col_names)

            sentence_dataframes.append(df)
        except IndexError:
            # In case a sentence creates an error, do not append it to the list
            pass



    # Create a list of dataframes for each sentence

    new_list = []

    for i in sentence_dataframes:
        i = i.replace(np.nan, "_") #to replace nan with '-'
        while len(i.columns) > 12:
            new_df = i.loc[:, 'col0':'col10'].copy()
            # grab last column of df
            cut_off_col = i.iloc[:, -1]
            # add last column to new df
            new_df = pd.concat([new_df, cut_off_col], axis=1)

            for index, row in new_df.iterrows():
                if row[-1] == 'V':
                    for j in range(index):
                        new_df.at[j, 'col10'] = '_'
                    for j in range(index + 1, len(new_df)):
                        new_df.at[j, 'col10'] = '_'
                    
            # remove last column from original df
            i = i.iloc[:, :-1]
            # # append new df to list
            # new_list.append(new_df)


        for index, row in i.iterrows():
            if row[-1] == 'V':
                for j in range(index):
                    i.at[j, 'col10'] = '_'
                for j in range(index + 1, len(i)):
                    i.at[j, 'col10'] = '_'

        # Remove all rows with punctuation

        i = i[~i['col1'].str.contains('[^\w\s]')]

        new_list.append(i)



    # Go through every dataframe to extract the word position feature

    for i in new_list:
        
        # Create a new column before the 11th column
        i.insert(11, "WORDPOS", "O")

        # Check on which row in the 10th column the V is located
        for index, row in i.iterrows():

            if row.iloc[-1] == 'V':
                
                # Change the O's in the WORDPOS column to B's for all rows before the index
                for j in range(index):
                    i.at[j, 'WORDPOS'] = 'BEFORE'

                # Change the O's in the WORDPOS column to A's for all rows after the index
                for j in range(index + 1, len(i)):
                    i.at[j, 'WORDPOS'] = 'AFTER'

    # Go through every dataframe to extract the named entity feature

    for i in new_list:

        # Go through the whole 'WORD' column and add all words to form a string
        sentence = ''
        for index, row in i.iterrows():
            
            # if row['col1'] is a float, convert it to a string
            if type(row['col1']) == float:
                row['col1'] = str(row['col1'])
            
            sentence += row['col1'] + ' '


        # Parse the sentence and extract the named entities

        doc = nlp(sentence)

        i.insert(11, "ENTITY", "F")

        for ent in doc.ents:
            for index, row in i.iterrows():
                if row['col1'] == ent.text:
                    row['ENTITY'] = 'T'

     

    # In the last column of all dataframes in new_list turn all -'s into O's

    for i in new_list:
        i.iloc[:, -1] = i.iloc[:, -1].str.replace('_', 'O')

    # Turn all columns into ID, WORD, LEMMA, UPOS, XPOS, MORPH, HEAD, BASIC DEP, ENHANCED DEPS, SPACE, PREDICATE, LABELS

    for i in new_list:
        i.columns = ['ID', 'WORD', 'LEMMA', 'UPOS', 'XPOS', 'MORPH', 'HEAD', 'BASIC DEP', 'ENHANCED DEPS', 'SPACE', 'PREDICATE', 'ENTITY', 'WORDPOS', 'LABEL']


    # Merge all dataframes in new_list into one dataframe

    complete_df = pd.concat(new_list)

    # Remove any rows with nan values

    complete_df = complete_df.dropna()

    # If the labels column of a row does not contain a valid target such as O, V or ARG, remove the row

    complete_df = complete_df[complete_df['LABEL'].str.contains('O|V|ARG')]

    # Add a column before the last one named LABEL2

    complete_df.insert(len(complete_df.columns)-1, 'LABEL2', complete_df['LABEL'])

    # In label2, if the value in a column is not O, V or C-V, replace it with 1

    complete_df.loc[~complete_df['LABEL2'].isin(['O', 'V', 'C-V']), 'LABEL2'] = 1

    # In label2, if the value in a column is O, V or C-V, replace it with 0
    complete_df.loc[complete_df['LABEL2'].isin(['V', 'C-V', 'O']), 'LABEL2'] = 0

    # rename label2 to identity

    complete_df.rename(columns={'LABEL2': 'IDENTITY'}, inplace=True)

    # rename label to category

    complete_df.rename(columns={'LABEL': 'CATEGORY'}, inplace=True)

    # Remove all rows with empty values

    complete_df = complete_df.loc[complete_df['WORD'] != '']

    # Write to csv

    with open(f'processed_{dataset}.csv', 'w', encoding= "utf-8") as f:
        complete_df.to_csv(f, index=False)

