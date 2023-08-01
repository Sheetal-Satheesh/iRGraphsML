import os
import itertools
from abc import ABCMeta, abstractmethod
import rdflib as rdf
import pandas as pd

__all__ = ["AIFBDataset", "MUTAGDataset"]


class RDFGraphDataset(metaclass=ABCMeta):

    def __init__(self, rootpath=None, training_path=None, test_path=None):
        self.rootpath = rootpath
        self.training_path = training_path
        self.test_path = test_path

    def load_rdf_graph(self):
        """ TODO: Rewrite the doc string
        Loading raw RDF dataset

                    Parameters
                    ----------
                    root_path : str
                        Root path containing the data

                    Returns
                    -------
                        Loaded rdf data
        """
        raw_rdf_graphs = []
        for _, filename in enumerate(os.listdir(self.rootpath)):
            fmt = None
            if filename.endswith("nt"):
                fmt = "nt"
            elif filename.endswith("n3"):
                fmt = "n3"
            if fmt is None:
                continue
            g = rdf.Graph()
            print("Parsing file %s ..." % filename)
            g.parse(os.path.join(self.rootpath, filename), format=fmt)
            raw_rdf_graphs.append(g)
        return g

    @abstractmethod
    def get_unique_classes(self, rootpath, training_path):
        pass


class AIFBDataset(RDFGraphDataset):
    ''' TODO: Rewrite the doc string
    AIFB dataset for node classification task

    AIFB DataSet is a Semantic Web (RDF) dataset used as a benchmark in
    data mining.  It records the organizational structure of AIFB at the
    University of Karlsruhe.

    AIFB dataset statistics:
    - Number of Classes: 4
    - Label Split:
        - Train: 140
        - Test: 36
    '''

    def __int__(self, rootpath, training_path, test_path):
        super().__init__(rootpath, training_path, test_path)

    def get_unique_classes(self, rootpath, training_path):
        # Read the TSV file into a DataFrame
        filepath = rootpath + training_path
        df = pd.read_csv(filepath, sep='\t')
        # Get the unique values from the "label" column as a list
        unique_classes = df['label_affiliation'].unique().tolist()
        return unique_classes

    def read_training_data(self):
        '''
        :param rootpath:
        :param training_path:
        :return: dictionary with class labels as key and values as

        Get the number of classes for each class create a nested dictionary
        '''
        # Read the TSV file into a DataFrame
        filepath = self.rootpath + self.training_path
        df = pd.read_csv(filepath, sep='\t')
        data_dict = {}
        # Iterate through the DataFrame and populate the dictionary
        for index, row in df.iterrows():
            label = row['label_affiliation']
            person = row['person']
            data_id = int(row['id'])

            if label not in data_dict:
                data_dict[label] = {}

            data_dict[label][person] = data_id
        return data_dict

    def read_testing_data(self):
        ''' TODO: Rewrite the doc string
        :param rootpath:
        :param test_path:
        :return: dictionary with class labels as keys and corresponding values i.e. the person and id

        Get the number of classes for each class create a nested dictionary.
        '''
        # Read the TSV file into a DataFrame
        filepath = self.rootpath + self.training_path
        df = pd.read_csv(filepath, sep='\t')
        data_dict = {}
        # Iterate through the DataFrame and populate the dictionary
        for index, row in df.iterrows():
            label = row['label_affiliation']
            person = row['person']
            data_id = int(row['id'])

            if label not in data_dict:
                data_dict[label] = {}

            data_dict[label][person] = data_id
        return data_dict


class MUTAGDataset(RDFGraphDataset):
    r"""MUTAG dataset for node classification task

    Mutag dataset statistics:

    - Number of Classes: 2
    - Label Split:
        - Train: 272
        - Test: 68

    Parameters
    -----------
    """

    def __int__(self, rootpath, training_path, test_path):
        name = "mutag-hetero"
        super().__init__(rootpath, training_path, test_path)

    def get_unique_classes(self, rootpath, training_path):
        # Read the TSV file into a DataFrame
        filepath = rootpath + training_path
        df = pd.read_csv(filepath, sep='\t')
        # Get the unique values from the "label" column as a list
        unique_classes = df['label_mutagenic'].unique().tolist()
        return unique_classes

    def read_training_data(self):
        '''
        :param rootpath:
        :param training_path:
        :return:

        Get the number of classes for each class create a nested dictionary
        '''
        # Read the TSV file into a DataFrame
        filepath = self.rootpath + self.training_path
        df = pd.read_csv(filepath, sep='\t')
        data_dict = {}
        # Iterate through the DataFrame and populate the dictionary
        for index, row in df.iterrows():
            label = row['label_mutagenic']
            bond = row['bond']
            data_id = row['id']

            if label not in data_dict:
                data_dict[label] = {}

            data_dict[label][bond] = data_id

        return data_dict

    def read_testing_data(self):
        '''
        :param rootpath:
        :param testing_path:
        :return:

        Get the number of classes for each class create a nested dictionary
        '''
        # Read the TSV file into a DataFrame
        filepath = self.rootpath + self.test_path
        df = pd.read_csv(filepath, sep='\t')
        data_dict = {}
        # Iterate through the DataFrame and populate the dictionary
        for index, row in df.iterrows():
            label = row['label_mutagenic']
            bond = row['bond']
            data_id = row['id']

            if label not in data_dict:
                data_dict[label] = {}

            data_dict[label][bond] = data_id

        return data_dict