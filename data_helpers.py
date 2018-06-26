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
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
'''
import numpy as np
import os
import csv
import time
import datetime
import sys
import re
import base64
import random


class DTSData(object):
    """
    DTS data class, include dts id, dts text and packages
    """

    def __init__(self, dts_id, text, packages=None, requirements=None):
        self.dts_id = dts_id
        self.text = text
        self.packages = packages
        self.requirements = requirements

    def set_label(self, label):
        self.label = label

    def set_words_indices(self, x):
        self.x = x


def load_stopwords(stopwords_file):
    """
    load stopwords from stopwords file
    :param stopwords_file: input stopwords file
    :return: list of stopwords
    """
    print("Loading stopwords...")
    stopwords = set()
    with open(stopwords_file, 'r', encoding='UTF-8') as f:
        for line in f:
            stopwords.add(line.strip())
    return stopwords


def clean_text(text ,stopwords = None):
    words = text.split()
    cleaned_words = ''
    punct_pattern = '[\\.:,\"\'\\(\\)\\[\\]|/?!;]+'
    digit_pattern = '[0-9a-f]+|0x[0-9a-f]+'
    for word in words:
        if word not in stopwords and not re.match(punct_pattern, word) and not re.match(digit_pattern, word):
            cleaned_words += word + ' '
    return cleaned_words.strip()


def load_data_and_labels(directory):
    """
    load package data from directory, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    packages = os.listdir(directory)
    x_text = []
    y = []
    i = 0
    for package in packages:
        # Load text of dts records
        examples = list(open(os.path.join(directory, package), "r", encoding='utf-8').readlines())
        x_text += [s.strip() for s in examples]
        # Generate labels
        labels = [[1 if col == i else 0 for col in range(len(packages))] for _ in examples]
        y = labels if i == 0 else np.concatenate([y, labels], 0)
        i += 1
    return [x_text, y]


def load_dts_data_from_csv(csv_file):
    """
    Load dts data from csv_file
    :param csv_file:  input csv_file
    :return: list of dts data, and dict of package_name counts
    """
    csv.field_size_limit(2147483647)
    dts_list = []
    package_counts = {}
    with open(csv_file, encoding='utf-8') as f:
        f_csv = csv.reader(f)
        next(f_csv)
        for row in f_csv:
            packages = row[2].split()
            dts = DTSData(row[0], row[1], packages=packages)
            dts_list.append(dts)
    return dts_list


def load_dts_data_with_min_packages(csv_file, min_package_count=0):
    """
    Load dts data from csv_file
    :param csv_file:  input csv_file
    :return: list of dts data, and dict of package_name counts
    """
    csv.field_size_limit(2147483647)
    dts_list = []
    package_counts = {}
    with open(csv_file, encoding='utf-8') as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            packages = row[2].split()
            dts = DTSData(row[0], row[1], packages=packages)
            dts_list.append(dts)
            for package in packages:
                if package in package_counts:
                    package_counts[package] = package_counts.get(package) + 1
                else:
                    package_counts[package] = 1
    removed_packages = []
    if min_package_count > 0:
        for package in list(package_counts):
            if package_counts[package] < min_package_count:
                removed_packages.append(package)
                del package_counts[package]
    sorted_package_counts = sorted(package_counts.items(), key=lambda x: x[1], reverse=True)
    return dts_list, sorted_package_counts, removed_packages


def allocate_label_for_dts_list(dts_list, package_name):
    """
    Given a dts list and package name, allocate label according to whether dts's packages involves given package
    """
    for dts in dts_list:
        dts.set_label([1, 0] if package_name in dts.packages else [0, 1])


def load_data_and_labels_from_csv(csv_file):
    """
    load package data from csv file
    """
    csv.field_size_limit(2147483647)
    dts_id, x_text, packages_list = [], [], []
    packages_set = set()
    with open(csv_file, encoding='utf-8') as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            dts_id.append(row[0])
            x_text.append(row[1])
            packages = row[2].split()
            packages_list.append(packages)
            packages_set.update(packages)
        packages_list = list(packages_set)
    labels = [[1 if (packages_list[col] in packages) else 0 for col in range(len(packages_set))] for packages in
              packages_list]
    y = np.asarray(labels)
    return [dts_id, x_text, y]


def clean_dts_list(dts_list, removed_packages):
    """
    removed dts whose packages are all included in removed_packages
    :return:cleaned dts list
    """
    cleaned_dts = []
    for dts in dts_list:
        for removed_package in removed_packages:
            if removed_package in dts.packages:
                dts.packages.remove(removed_package)
        if len(dts.packages) != 0:
            cleaned_dts.append(dts)
    return cleaned_dts


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def load_word_vectors(word_vector_lookuptable):
    """
    load word_vectors from given word_vector_lookuptable
    :param word_vector_lookuptable: word_vectors model
    :return: dict of word and related vector
    """
    print("Load word vectors...")
    word_vectors = {}
    hasHeader = False
    total_lines = 0
    with open(word_vector_lookuptable, 'r') as f:
        # check if word_vector_lookuptable exists header line
        firstline = f.readline()
        if " " not in firstline:
            hasHeader = True
        elif len(firstline.split()) < 4:
            hasHeader = True
            total_lines = int(firstline.split()[0])
    with open(word_vector_lookuptable, 'r') as f:
        # if hasHeader is true, skip first line, else keep firt line
        if hasHeader:
            next(f)
        i = 0
        for line in f:
            i += 1
            if i % 10000 == 0:
                print("Loaded " + str(i) + "/" + str(total_lines) + " lines")
            if line:
                split = line.split(" ")
                word = split[0]
                if word.startswith("B64:"):
                    arp = word.replace("B64:", "")
                    word = base64.b64decode(arp).decode('utf-8')
                vector = [float(i) for i in split[1:]]
                word_vectors[word] = vector
    print("Finish loading!")
    return word_vectors


def load_dts_data_and_requirements(csv_file, requirement_file, stopwords_file=None):
    """
    load dts data from csv file and requirments from requirement file
    :param csv_file: each line contains dts number, text and relevant requirements indices
    :param requirement_file: each line is a requirement description
    :return:  dts data list and requirment list
    """
    if stopwords_file:
        stopwords = load_stopwords(stopwords_file)
    else:
        stopwords = []
    csv.field_size_limit(2147483647)
    dts_list = []
    requirement_list = []
    # load requirements
    with open(requirement_file, encoding='utf-8') as r:
        for line in r:
            cleaned_line = clean_text(line, stopwords)
            requirement_list.append(cleaned_line)
    requirments_len = len(requirement_list)
    # load dts data
    with open(csv_file, encoding='utf-8') as f:
        f_csv = csv.reader(f)
        next(f_csv)
        for row in f_csv:
            requirement_indices = [int(index) for index in row[2].split()]
            dts_text = row[1]
            cleaned_text = clean_text(dts_text, stopwords)
            dts = DTSData(row[0], cleaned_text, requirements=requirement_indices)
            # assign labels, each dts has requirments_len labels
            labels = [[[1, 0] if index in requirement_indices else [0, 1]] for index in range(requirments_len)]
            dts.set_label(labels)
            dts_list.append(dts)
    return dts_list, requirement_list

def load_data_from_csv(data_file):
    # Load text of dts records
    contents, labels = [], []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                content, label = line.strip().split(',')
                if content is not None:
                    contents.append(content)
                    labels.append(label)
            except:
                pass
        categry = list(set(labels))
        categry.sort()
        labels = [[1 if (categry[col] in label) else 0 for col in range(len(categry))] for label in
                  labels]
        y = np.asarray(labels)
    return [contents, y]

def load_data_from_txt(data_file):
    output_train_file = r"./data/module/train.csv"
    output_test_file = r"./data/module/test.csv"
    number_0 = 0
    contents_1,contents_0 = [], []
    datas = []
    # Load text of dts records
    contents, labels = [], []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                label, content = line.strip().split(',')
                if content is not None:
                    contents.append(content)
                    labels.append(label)
            except:
                pass
        category = list(set(labels))
        category.sort()
        max_number = 0
        for label in category:
            number = labels.count(label)
            if number > max_number:
                max_number = number
                max_number_name = label
    print(type(max_number_name))
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                label, content = line.strip().split(',')
                if label == max_number_name:
                    data = content.lstrip() + ',' + '1' + '\n'
                    contents_1.append(content.lstrip())
                    datas.append(data)
                else:
                    contents_0.append(content.lstrip())
            except:
                pass
    copy_number = 0
    for line in contents_0:
        if line in contents_1:
            copy_number = copy_number + 1
            continue
        data = line.strip() + ',' + '0' +'\n'
        datas.append(data)
    random.shuffle(datas)
    sample_index = -1 * int(0.1 * float(len(datas)))
    train, test = datas[:sample_index], datas[sample_index:]
    for cell in train:
        open(output_train_file, 'a', encoding='utf-8').write(cell)
    for cell in test:
        open(output_test_file, 'a', encoding='utf-8').write(cell)
    print(len(datas))
    print(len(train))
    print(len(test))

        #labels = [[1 if (labels[col] in label) else 0 for col in range(len(category))] for label in labels]
        #y = np.asarray(labels)
    #return [contents, y] 

def sort_document_list_by_length(document_list):
    sorted_document_list = sorted(document_list, key=lambda x : len(x.split()), reverse=True)
    print([len(requirement.split()) for requirement in sorted_document_list])
    print(sorted_document_list[0])

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


def load_data_and_labels_polarity(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

if __name__ == "__main__":
    # x_text, y = load_data_and_labels("D:\\data\\galaxy-defect-model\\5800\\classifiedRecords")
    # dts_id, x_text, labels = load_data_and_labels_from_csv("D:\\data\\galaxy-defect-model\\5800\\dts-packages.csv")
    # print(labels)
    # word_vectors = load_word_vectors("D:\\data\\20170927\\total_d200e200.lookuptable")
    #dts_list, requirement_list = load_dts_data_and_requirements('.\\data\\requirements-csv\\SMU_relevant_requirements_train_SR.csv', '.\\data\\requirements-csv\\requirement_sr.s', '.\\data\\StopWords.txt')
    #sort_document_list_by_length(requirement_list)
    #sort_document_list_by_length([dts.text for dts in dts_list])
    x_test, y = load_data_from_csv('.\\data\\module\\test.csv')
    print(x_test)
    print(y)
    #x, y = load_data_and_labels_polarity("./data/rt-polaritydata/rt-polarity.neg", "./data/rt-polaritydata/rt-polarity.pos")
    #print(len(x))
    #print(len(y))
    #print(x)
    #print(y)
'''
