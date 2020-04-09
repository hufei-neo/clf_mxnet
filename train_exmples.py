import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files

    positive = open(positive_data_file, "rb").read().decode('utf-8')
    negative = open(negative_data_file, "rb").read().decode('utf-8')

    positive_examples = positive.split('\n')[:-1]
    negative_examples = negative.split('\n')[:-1]

    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = [s.strip() for s in negative_examples]

    # positive_examples = list(open(positive_data_file, "rb").read().decode('utf-8'))
    # positive_examples = [s.strip() for s in positive_examples]
    # negative_examples = list(open(negative_data_file, "rb").read().decode('utf-8'))
    # negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


df = load_data_and_labels(r'C:\Users\15581\Documents\python_dl\文本分类\09.文本分类\英文邮件分类\data\rt-polaritydata\rt-polarity.pos',
r'C:\Users\15581\Documents\python_dl\文本分类\09.文本分类\英文邮件分类\data\rt-polaritydata\rt-polarity.neg')

print(df)
