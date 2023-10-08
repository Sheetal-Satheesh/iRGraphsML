import os
import itertools
from abc import ABCMeta, abstractmethod

import rdflib
import rdflib as rdf
import pandas as pd

__all__ = ["AIFBDataset", "MUTAGDataset", "DBLPDataset", "BGSDataset"]


class RDFGraphDataset(metaclass=ABCMeta):

    def __init__(self, rootpath=None, training_path=None, test_path=None):
        self.rootpath = rootpath
        self.training_path = training_path
        self.test_path = test_path

    def load_rdf_graph(self):
        """ TODO: Rewrite the doc string
        Loading raw RDF raw_data

                    Parameters
                    ----------
                    root_path : str
                        Root path containing the data

                    Returns
                    -------
                        Loaded rdf data
        """
        raw_rdf_graphs = []
        combined_graph = rdflib.Graph()
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
        for subgraph in raw_rdf_graphs:
            combined_graph += subgraph
        return combined_graph

    @abstractmethod
    def get_unique_classes(self, rootpath, training_path):
        pass

    def get_most_prominent_class(self, outer_to_inner):
        label_counts = {}
        for outer_key in outer_to_inner:
            inner_dict = outer_to_inner[outer_key]
            label_counts[outer_key] = len(inner_dict)
        max_key = max(label_counts, key=label_counts.get)
        return max_key


class AIFBDataset(RDFGraphDataset):
    ''' TODO: Rewrite the doc string
    AIFB raw_data for node classification task

    AIFB DataSet is a Semantic Web (RDF) raw_data used as a benchmark in
    data mining.  It records the organizational structure of AIFB at the
    University of Karlsruhe.

    AIFB raw_data statistics:
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
            if label == 'http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id1instance':
                label = 'id1'
            elif label == 'http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id2instance':
                label = 'id2'
            elif label == 'http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id3instance':
                label = 'id3'
            elif label == 'http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id4instance':
                label = 'id4'
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
        filepath = self.rootpath + self.test_path
        df = pd.read_csv(filepath, sep='\t')
        data_dict = {}
        # Iterate through the DataFrame and populate the dictionary
        for index, row in df.iterrows():
            label = row['label_affiliation']
            if label == 'http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id1instance':
                label = 'id1'
            elif label == 'http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id2instance':
                label = 'id2'
            elif label == 'http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id3instance':
                label = 'id3'
            elif label == 'http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id4instance':
                label = 'id4'
            person = row['person']
            data_id = int(row['id'])

            if label not in data_dict:
                data_dict[label] = {}

            data_dict[label][person] = data_id
        return data_dict


class MUTAGDataset(RDFGraphDataset):
    r"""MUTAG raw_data for node classification task

    Mutag raw_data statistics:

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

class DBLPDataset(RDFGraphDataset):
    def __int__(self, rootpath, training_path, test_path):
        super().__init__(rootpath, training_path, test_path)

    def get_unique_classes(self, rootpath, training_path):
        # Read the TSV file into a DataFrame
        filepath = rootpath + training_path
        df = pd.read_csv(filepath, sep='\t')
        # Get the unique values from the "label" column as a list
        unique_classes = df['cls'].unique().tolist()
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
            label = row['cls']
            instance = row['instance']
            citations = int(row['num_citations'])

            if label not in data_dict:
                data_dict[label] = {}

            data_dict[label][instance] = citations
        return data_dict

    def read_testing_data(self):
        ''' TODO: Rewrite the doc string
        :param rootpath:
        :param test_path:
        :return: dictionary with class labels as keys and corresponding values i.e. the person and id

        Get the number of classes for each class create a nested dictionary.
        '''
        # Read the TSV file into a DataFrame
        filepath = self.rootpath + self.test_path
        df = pd.read_csv(filepath, sep='\t')
        data_dict = {}
        # Iterate through the DataFrame and populate the dictionary
        for index, row in df.iterrows():
            label = row['cls']
            instance = row['instance']
            citations = int(row['num_citations'])

            if label not in data_dict:
                data_dict[label] = {}

            data_dict[label][instance] = citations
        return data_dict


class BGSDataset(RDFGraphDataset):
    r"""MUTAG raw_data for node classification task

    BGS raw_data statistics:

    - Number of Classes: 2
    - Label Split:
        - Train: 272
        - Test: 68

    Parameters
    -----------
    """

    def __int__(self, rootpath, training_path, test_path):
        name = "bgs-hetero"
        super().__init__(rootpath, training_path, test_path)

    def get_unique_classes(self, rootpath, training_path):
        # Read the TSV file into a DataFrame
        filepath = rootpath + training_path
        df = pd.read_csv(filepath, sep='\t')
        # Get the unique values from the "label" column as a list
        unique_classes = df['label_lithogenesis'].unique().tolist()
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
            label = row['label_lithogenesis']
            if label == 'http://data.bgs.ac.uk/id/Lexicon/LithogeneticType/FLUV':
                label = 1
            elif label == 'http://data.bgs.ac.uk/id/Lexicon/LithogeneticType/GLACI':
                label = 0
            rock = row['rock']
            rock_id = row['rock']
            if label not in data_dict:
                data_dict[label] = {}

            data_dict[label][rock] = rock_id
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
            label = row['label_lithogenesis']
            if label == 'http://data.bgs.ac.uk/id/Lexicon/LithogeneticType/FLUV':
                label = 1
            elif label == 'http://data.bgs.ac.uk/id/Lexicon/LithogeneticType/GLACI':
                label = 0
            rock = row['rock']
            rock_id = row['rock']
            if label not in data_dict:
                data_dict[label] = {}

            data_dict[label][rock] = rock_id

        return data_dict


class AMDataset(RDFGraphDataset):

    def __int__(self, rootpath, training_path, test_path):
        super().__init__(rootpath, training_path, test_path)

    def get_unique_classes(self, rootpath, training_path):
        # Read the TSV file into a DataFrame
        filepath = rootpath + training_path
        df = pd.read_csv(filepath, sep='\t')
        # Get the unique values from the "label" column as a list
        unique_classes = df['label_cateogory'].unique().tolist()
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
            label = row['label_cateogory']
            if label == 'http://purl.org/collections/nl/am/t-22503':
                label = 't-22503'
            elif label == 'http://purl.org/collections/nl/am/t-15459':
                label = 't-15459'
            elif label == 'http://purl.org/collections/nl/am/t-15579':
                label = 't-15579'
            elif label == 'http://purl.org/collections/nl/am/t-5504':
                label = 't-5504'
            elif label == 'http://purl.org/collections/nl/am/t-14592':
                label = 't-14592'
            elif label == 'http://purl.org/collections/nl/am/t-22504':
                label = 't-22504'
            elif label == 'http://purl.org/collections/nl/am/t-15606':
                label = 't-15606'
            elif label == 'http://purl.org/collections/nl/am/t-22505':
                label = 't-22505'
            elif label == 'http://purl.org/collections/nl/am/t-22508':
                label = 't-22508'
            elif label == 'http://purl.org/collections/nl/am/t-22506':
                label = 't-22506'
            elif label == 'http://purl.org/collections/nl/am/t-22507':
                label = 't-22507'
            proxy = row['proxy']
            data_id = int(row['id'])

            if label not in data_dict:
                data_dict[label] = {}

            data_dict[label][proxy] = data_id
        return data_dict

    def read_testing_data(self):
        ''' TODO: Rewrite the doc string
        :param rootpath:
        :param test_path:
        :return: dictionary with class labels as keys and corresponding values i.e. the person and id

        Get the number of classes for each class create a nested dictionary.
        '''
        # Read the TSV file into a DataFrame
        filepath = self.rootpath + self.test_path
        df = pd.read_csv(filepath, sep='\t')
        data_dict = {}
        # Iterate through the DataFrame and populate the dictionary
        for index, row in df.iterrows():
            label = row['label_cateogory']
            if label == 'http://purl.org/collections/nl/am/t-22503':
                label = 't-22503'
            elif label == 'http://purl.org/collections/nl/am/t-15459':
                label = 't-15459'
            elif label == 'http://purl.org/collections/nl/am/t-15579':
                label = 't-15579'
            elif label == 'http://purl.org/collections/nl/am/t-5504':
                label = 't-5504'
            elif label == 'http://purl.org/collections/nl/am/t-14592':
                label = 't-14592'
            elif label == 'http://purl.org/collections/nl/am/t-22504':
                label = 't-22504'
            elif label == 'http://purl.org/collections/nl/am/t-15606':
                label = 't-15606'
            elif label == 'http://purl.org/collections/nl/am/t-22505':
                label = 't-22505'
            elif label == 'http://purl.org/collections/nl/am/t-22508':
                label = 't-22508'
            elif label == 'http://purl.org/collections/nl/am/t-22506':
                label = 't-22506'
            elif label == 'http://purl.org/collections/nl/am/t-22507':
                label = 't-22507'
            proxy = row['proxy']
            data_id = int(row['id'])

            if label not in data_dict:
                data_dict[label] = {}

            data_dict[label][proxy] = data_id
        return data_dict