from copy import copy, deepcopy
from typing import Optional, List

import numpy as np
import csv
import pandas as pd
from sklearn.model_selection import train_test_split


def extract_data_from_csv_file(file_handle):
    """
    Extracts data from file. Converts it into dataframe pandas type.
    :param file_handle:
    :return: Data represented as dataframe type.
    """
    data_frame = pd.read_csv(file_handle, header=None)
    return data_frame


def prepare_data_for_analysis(dataset: pd.DataFrame, column_names: List[str]):
    """
    Adds column names for better code and data analysis.
    :return: Modified data with column names
    """
    dataset.columns = column_names
    return dataset


# class ID3Tree:
#     def __init__(self, max_depth: int, dataset: pd.DataFrame):
#         self._max_depth = max_depth
#
#     def get_max_depth(self):
#         return self._max_depth



 # class ID3Node:
 #
 #     def __init__(self, attribute_name: str, children):
 #         pass

def id3_algorithm():
    """

    :param train_data: provided part of the data on which the model will learn dependencies between data
    :param target_label: label, for which the data set will be analysed
    :return: dictionary which represents decision tree
    """
    pass

def entropy(data_frame: pd.DataFrame, provided_attribute: str):

    class_values_categorized = data_frame[provided_attribute].value_counts()
    total_class_number = class_values_categorized.sum()
    entropy = sum([-(distinct_value_counts / total_class_number)*np.log2(distinct_value_counts / total_class_number)
                   for distinct_value_counts in class_values_categorized])
    return entropy


def attribute_entropy():
    pass
    # for each_value in class_values_categorized:
#     entropy = - pos_val_ratio*np.log2(pos_val_ratio) - neg_val_ratio*np.log2(neg_val_ratio)
#
#     return entropy
#
def average_information(data_frame_subset: pd.DataFrame, provided_attribute:str):
    """

    :return:
    """
    # class_values_categorized = data_frame_subset.groupby(data_frame_subset.iloc[:, 0]).size()
    print(type(data_frame_subset))
    attribute_to_analyse = pd.Series(data_frame_subset.iloc[:, 0]).unique()
    total_class_number = data_frame_subset.value_counts().sum()
    attribute_value_information = 0

    for each_value in attribute_to_analyse:
        value_fraction = data_frame_subset.value_counts()[each_value].sum()/total_class_number
        # mask = data_frame_subset['game_id'].values == 'g21'
        attribute_value_information += value_fraction*entropy(data_frame_subset[data_frame_subset.iloc[:, 0] == each_value], provided_attribute)
    return attribute_value_information
#
def information_gain():
    # entropy_of_dataset - average_information_for_given_attribute
    pass

def find_best_attribute(dataset: pd.DataFrame):
    """
    For each attribute in provided dataset the entropy is calculated.
    Function returns the attribute, which provides the highest information gain.
    :param dataset: provided data which is to be analysed.
    :return: attribute which provides the highest information gain.
    """

    pass




data_frame = extract_data_from_csv_file('breast_cancer.csv')
breast_cancer_columns = ["Class", "age", "menopause", "tumor size", "inv nodes", "node caps", "deg malig", "breast", "breast quad", "irradiat"]
formatted_data = prepare_data_for_analysis(data_frame, breast_cancer_columns)
#
# class_values_categorized = data_frame['irradiat'].value_counts()
# print(class_values_categorized)
# total_class_number = class_values_categorized.sum()
# print(total_class_number)


print(entropy(formatted_data, 'irradiat'))


attributes_indices = [x for x in range(len(formatted_data.columns))]
print(attributes_indices)
target_attribute_index = formatted_data.columns.get_loc('irradiat')
attributes_indices.remove(target_attribute_index)

list_of_subsets = []
for each_column_index in attributes_indices:
    subset = deepcopy(formatted_data.iloc[:, [each_column_index, target_attribute_index]])
    list_of_subsets.append(subset)
print(type(list_of_subsets[0]))
print(average_information(list_of_subsets[3], 'irradiat'))
for each_subset in list_of_subsets:
    print(each_subset.columns.values)
    print(average_information(each_subset, 'irradiat'))



pass

tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

def divide_dataset_for_target_attribute(dataset: pd.DataFrame, target_attribute: str):
    dataset_to_analyse = dataset.drop([target_attribute])
    target_attribute_column = dataset[target_attribute]
    return dataset_to_analyse, target_attribute_column

