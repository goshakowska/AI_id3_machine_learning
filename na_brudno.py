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
    """
    Method that calculates information gain for the provided attribute of the dataset.
    :param dataset_entropy: calculated entropy for the whole provided (earlier) dataset
    :param attribute_name: analysed attribute
    :param attribute_average_information: calculated earlier average information for the given attribute
    :return: name of the attribute and its information gain
    """
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
    """
    Method that calculates entropy of the provided target attribute based on the formula
    :param target_attribute_values:
    :return: calculated entropy for the provided attribute
    """

    class_values_categorized = target_attribute_values.value_counts()
    total_class_number = class_values_categorized.sum()
    entropy = sum([-(distinct_value_counts / total_class_number)*np.log2(distinct_value_counts / total_class_number)
                   for distinct_value_counts in class_values_categorized])
    return entropy


def calculate_information_gains_for_each_attribute(dataset_to_analyse: pd.DataFrame, target_attribute: pd.Series):
    """
    Wrapper function that calculates information gain for each attribute in provided dataset.
    :param dataset_to_analyse:
    :param target_attribute: target attribute in
    :return: dictionary which stores data in the form: {key = attribute's name: value = information gain of that attribute}
    """
    attribute_information_gains = {}
    for each_column in dataset_to_analyse.columns:
        attr_name, information = information_gain(entropy(target_attribute),
                                                  average_information(dataset_to_analyse[each_column], target_attribute)[0],
                                                  average_information(dataset_to_analyse[each_column], target_attribute)[1])
        attribute_information_gains[attr_name] = information
    return attribute_information_gains


def average_information(data_attribute_subset: pd.DataFrame, target_attribute_values: pd.Series):
    """
    Method that calculates average information obtained from the provided attribute in the dataset
    It iterates through each value (of the given attribute) and calculates entropy of the target attribute
    (for that attribute value).
    :param data_attribute_subset: series of the given attribute
    :param target_attribute_values: target attribute values
    :return: attributes name, average information for the given attribute
    """
    values_to_analyse = data_attribute_subset.unique()
    total_class_number = data_attribute_subset.value_counts().sum()
    attribute_average_information = 0

    for each_value in values_to_analyse:
        value_fraction = data_attribute_subset.value_counts()[each_value].sum()/total_class_number
        selected_rows = data_attribute_subset.where(data_attribute_subset == each_value)
        rows_to_analyse = target_attribute_values.loc[selected_rows.dropna().index.values.tolist()]  # CZY TU JEST BŁĄD? z .copy()??
        attribute_average_information += value_fraction*entropy(rows_to_analyse)
    return data_attribute_subset.name, attribute_average_information


class ID3Node:
    def __init__(self, attribute_name: str, values_of_attribute, children: ['ID3Node'] = None):
        self._attribute_name = attribute_name
        self._values_of_attribute = values_of_attribute
        self._children_nodes = children


class ID3Tree:
    def __init__(self, max_depth: int):
        self._max_depth = max_depth
        self._root = None

    def get_max_depth(self):
        """
        ID3 tree maximum depth getter.
        :return: maximum depth of the ID3 tree.
        """
        return self._max_depth

    def get_root(self):
        """
        ID3 tree root getter.
        :return: root of the ID3 tree.
        """
        return self._root

    def set_root(self, new_root: ID3Node):
        """
        ID3 tree root setter.
        :param new_root: ID3 Node object, which will be the root of the ID3 tree.
        """
        self._root = new_root

    def build_ID3_tree(self, dataset_to_analyse: pd.DataFrame, target_attribute: pd.Series, depth: int):
        """
        Creates ID3 tree structure by recursively finding best attribute to divide the current set. When the given value
        of the attribute has homogeneous target attribute values or the tree reached its maximum depth (provided by user)
        or there are no columns (attributes) to analyse we return the leaf node
        equal to 0,
        :param dataset_to_analyse: provided part of the data on which the model will learn dependencies between data
        :param target_attribute:
        :param depth:
        :return: a ID3 tree with newly created ID3 Nodes
        """
        total_class_number = target_attribute.value_counts()
        if len(total_class_number) == 1 or depth == 0 or len(dataset_to_analyse.columns) == 0:
            return total_class_number[0]
        attributes_information_gain = calculate_information_gains_for_each_attribute(dataset_to_analyse, target_attribute)
        division_attribute = find_best_attribute(attributes_information_gain)
        values_to_analyse = dataset_to_analyse[division_attribute].unique()
        # children_nodes = []
        # for value in values_to_analyse:
        #     new_dataset = dataset_to_analyse[dataset_to_analyse[division_attribute] == value].drop(division_attribute, axis=1)
        #     sliced_rows_of_target_attribute = target_attribute[dataset_to_analyse[division_attribute] == value]
        #     new_depth = depth - 1
        #     new_branch = self.build_ID3_tree(new_dataset, sliced_rows_of_target_attribute, new_depth)
        #     children_nodes.append(new_branch)
        children_nodes = [self.build_ID3_tree(dataset_to_analyse[dataset_to_analyse[division_attribute] == value].drop(division_attribute, axis=1),
                                          target_attribute[dataset_to_analyse[division_attribute] == value], depth - 1) for value in values_to_analyse]
        return ID3Node(division_attribute, values_to_analyse, children_nodes)


data_frame = extract_data_from_csv_file('breast_cancer.csv')
breast_cancer_columns = ["Class", "age", "menopause", "tumor size", "inv nodes", "node caps", "deg malig", "breast", "breast quad", "irradiat"]
formatted_data = prepare_data_for_analysis(data_frame, breast_cancer_columns)
data, target = divide_dataset_for_target_attribute(formatted_data, 'irradiat')

formatted_data_2 = formatted_data.where(formatted_data == 'irradiat')


X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25)
tree = ID3Tree(5)
total_class_number = target.value_counts()
print(total_class_number[0])
tree.set_root(tree.build_ID3_tree(X_train, y_train, tree.get_max_depth()))
# attribute_information_gains = {}
# for each_column in data.columns:
#     # print(data[each_column].name)
#     print(average_information(data[each_column], target))
#     # attr_name, av_inf = average_information(data[each_column], target)
#     attr_name, information = information_gain(entropy(target), average_information(data[each_column], target)[0], average_information(data[each_column], target)[1])
#     attribute_information_gains[attr_name] = information
# print(find_best_attribute(attribute_information_gains))
pass

