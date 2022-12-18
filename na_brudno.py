from copy import copy, deepcopy
from typing import Optional, List, Dict

import numpy as np
import csv
import pandas as pd
from sklearn.model_selection import train_test_split


def extract_data_from_csv_file(file_handle):
    """
    Extracts data from file. Converts it into dataframe pandas type.
    :param file_handle:
    :return: Data represented as pandas DataFrame type.
    """
    dataset = pd.read_csv(file_handle, header=None)
    return dataset


def prepare_data_for_analysis(dataset: pd.DataFrame, column_names: List[str]):
    """
    Adds column names for better code and data analysis.
    :return: Modified data with column names
    """
    dataset.columns = column_names
    return dataset


def divide_dataset_for_target_attribute(dataset: pd.DataFrame, target_attribute: str):
    """
    Divides provided dataset into two:
    1) contains all the attributes (and their values) on which our ID3 algorithm will learn "the pattern" of the data.
    2) contains the target attribute's values, which later will be predicted by the ID3 tree
    :param dataset: provided dataset for analysis
    :param target_attribute: target attribute on account of which the dataset will be analysed
    :return: two datasets (as array and column)
    """
    dataset_to_analyse = dataset.drop([target_attribute], axis=1)
    target_attribute_column = dataset[target_attribute]
    return dataset_to_analyse, target_attribute_column


def information_gain(dataset_entropy: float, attribute_name: str, attribute_average_information: float):
    information_gain = dataset_entropy - attribute_average_information
    return attribute_name, information_gain


def find_best_attribute(attributes_information_gain: Dict[str, float]):
    """
    Method that finds the best attribute based on its information gain.
    :param attributes_information_gain: list of information gains for each attribute in the dataset.
    :return: attribute which provides the highest information gain.
    """
    return max(attributes_information_gain, key=lambda value: attributes_information_gain[value])


def entropy(target_attribute_values: pd.Series):

    class_values_categorized = target_attribute_values.value_counts()
    total_class_number = class_values_categorized.sum()
    entropy = sum([-(distinct_value_counts / total_class_number)*np.log2(distinct_value_counts / total_class_number)
                   for distinct_value_counts in class_values_categorized])
    return entropy


def average_information(data_attribute_subset: pd.DataFrame, target_attribute_values: pd.Series):
    """

    :return:
    """
    attribute_to_analyse = data_attribute_subset.unique()
    total_class_number = data_attribute_subset.value_counts().sum()
    attribute_average_information = 0

    for each_value in attribute_to_analyse:
        value_fraction = data_attribute_subset.value_counts()[each_value].sum()/total_class_number
        selected_rows = data_attribute_subset.where(data_attribute_subset == each_value)
        rows_to_analyse = target_attribute_values.loc[selected_rows.dropna().index.values.tolist()]
        attribute_average_information += value_fraction*entropy(rows_to_analyse)
    return data_attribute_subset.name, attribute_average_information


class ID3Tree:
    def __init__(self, max_depth: int, dataset: pd.DataFrame):
        self._max_depth = max_depth

    def get_max_depth(self):
        return self._max_depth


class ID3Node:
     def __init__(self, attribute_name: str, children:'ID3Node'=None):
         pass

def id3_algorithm():
    """

    :param train_data: provided part of the data on which the model will learn dependencies between data
    :param target_label: label, for which the data set will be analysed
    :return: dictionary which represents decision tree
    """
    pass


data_frame = extract_data_from_csv_file('breast_cancer.csv')
breast_cancer_columns = ["Class", "age", "menopause", "tumor size", "inv nodes", "node caps", "deg malig", "breast", "breast quad", "irradiat"]
formatted_data = prepare_data_for_analysis(data_frame, breast_cancer_columns)


data, target = divide_dataset_for_target_attribute(formatted_data, 'irradiat')
print(data)
print(target)
print(data.columns[0])
print(entropy(target))
pass
attribute_information_gains = {}
for each_column in data.columns:
    # print(data[each_column].name)
    print(average_information(data[each_column], target))
    # attr_name, av_inf = average_information(data[each_column], target)
    attr_name, information = information_gain(entropy(target), average_information(data[each_column], target)[0], average_information(data[each_column], target)[1])
    attribute_information_gains[attr_name] = information
print(find_best_attribute(attribute_information_gains))


