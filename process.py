# Preprocessing libs
import pandas as pd
import json
from bs4 import BeautifulSoup
import re
from sklearn.preprocessing import LabelEncoder

# Stemming libs
import nltk
from nltk.corpus import stopwords
import pymorphy3
from pymorphy3 import MorphAnalyzer
from nltk.tokenize import word_tokenize

# Lemmatize libs
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
from pymystem3 import Mystem

# Logging
from tqdm import tqdm

nltk.download('punkt')
nltk.download('stopwords')

# Constants
PATH = 'output_25595.json'


def json_to_pandas(json_file: json) -> pd.DataFrame:
    """
    Converts json file to pandas dataframe
    :param json_file: input json file
    :return: df of pandas dataframe
    """
    with open(json_file, encoding='utf-8') as f:
        data = json.load(f)
        data = clean_nones(data)
        df = pd.DataFrame(data)
        return df


def clean_nones(value: dict[any, any] | list[any]) -> dict[any, any] | list[any]:
    """
    Recursively remove all None values from dictionaries and lists
    :param value: input dictionary
    :return: a new dictionary or list
    """
    if isinstance(value, list):
        return [clean_nones(x) for x in value if x is not None]
    elif isinstance(value, dict):
        return {
            key: clean_nones(val)
            for key, val in value.items()
            if val is not None
        }
    else:
        return value


def preprocess(path: str = PATH) -> pd.DataFrame:
    """
    Preprocesses input json file according to preprocessing rules
    :param path: file path
    :return: dataframe of preprocessed json file
    """
    df = json_to_pandas(path)

    # Remove missing values
    df.dropna(inplace=True)

    # Get value_counts of each category
    counts = df['category'].value_counts()

    # Define columns needed to preprocess
    dateColumns = ['startDate', 'finalDate']
    htmlColumns = ['shortDescriptionRaw', 'socialGroupsRaw', 'aimsRaw', 'tasksRaw', 'qualityResult', 'evaluation']
    removeLabels = counts[counts < 150].index

    # Extract dates
    for column in dateColumns:
        df[column] = df[column].str[:10] # ISO 8601

    # Remove html text from
    for column in htmlColumns:
        df[column] = df[column].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())

    # Remove emissions
    for label in removeLabels:
        df.drop(df.loc[df['category'] == label].index, inplace=True)

    for column in htmlColumns:
        # Remove links
        df[column] = [re.sub(r'^https?:\/\/.*[\r\n]*', '', i, flags=re.MULTILINE)
                      for i in tqdm(df[column])]

        # Reduce words to lowercase
        df[column] = [i.lower() for i in tqdm(df[column])]

        # Remove punctuation marks
        df[column] = [re.sub(r"[^\w\d\s]+", " ", i)
                      for i in tqdm(df[column])]

        # Remove numbers
        df[column] = [re.sub("\d+", "", i) for i in tqdm(df[column])]

        # Remove duplicate words
        df[column] = [re.sub(r'[x]{2,}', "", i) for i in tqdm(df[column])]

        # Remove double spaces
        df[column] = [re.sub(' +', ' ', i) for i in tqdm(df[column])]

    return df


def get_labels(df: pd.DataFrame, column='category') -> tuple[pd.DataFrame, int, dict]:
    """
    Converts dataframe to labels dataframe
    :param df: dataframe
    :param column: labels column
    :return: labels dataframe, number of labels and mapping dictionary
    """
    le = LabelEncoder()
    mapping2label = dict(zip(le.classes_, range(len(le.classes_))))
    df['label'] = le.fit_transform(df[column])
    num_class = len(df.labels.unique())
    return df, num_class, mapping2label


def stemming_process(text: str, punctuation_marks: list[str], stop_words: list[str], morph: MorphAnalyzer) -> list[str]:
    """
    Stemming method
    :param text: input text
    :param stop_words: list of stop words
    :param punctuation_marks: list of punctuation marks
    :param morph: morphological analyzer
    :return: string stemmed text
    """
    tokens = word_tokenize(text.lower())
    preprocessed_text = []
    for token in tokens:
        if token not in punctuation_marks:
            lemma = morph.parse(token)[0].normal_form
            if lemma not in stop_words:
                preprocessed_text.append(lemma)
    return preprocessed_text


def stemming(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Stemming method for dataframe
    :param df: input dataframe
    :param columns: columns of dataframe need to be stemmed
    :return: dataframe with stemmed columns
    """
    punctuation_marks = ['!', ',', '(', ')', ':', '-', '?', '.', '..', '...', '«', '»', ';', '–', '--']
    stop_words = stopwords.words("russian")
    morph = pymorphy3.MorphAnalyzer()

    tqdm.pandas()

    for column in columns:
        df[f'preprocessed_{column}'] = df.progress_apply(
            lambda row: stemming_process(row[column], punctuation_marks, stop_words, morph), axis=1)

    return df


def lemmatize(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Lemmatizing method for dataframe
    :param df: input dataframe
    :param columns: columns of dataframe need to be lemmatized
    :return: dataframe with lemmatized columns
    """
    noise = list(punctuation) + stopwords.words('russian')
    smart_vectorizer = CountVectorizer(stop_words=noise)
    tok = smart_vectorizer.build_tokenizer()
    mystem = Mystem()

    for column in columns:
        df[f'preprocessed_{column}'] = list(
            map(lambda text: " ".join((tok(str(mystem.lemmatize(text))))), tqdm(df[column])))

    return df


def df_to_csv(df: pd.DataFrame, name: str):
    """
    Converts dataframe to csv file
    :param df: dataframe to convert
    :param name: file name
    """
    df.to_csv(f'{name}.csv', sep=';', encoding='UTF-8', index=False)


def csv_to_df(path: str) -> pd.DataFrame:
    """
    Converts csv to dataframe file
    :param path: file path
    :return: dataframe
    """
    return pd.read_csv(path, sep=';', encoding='UTF-8')