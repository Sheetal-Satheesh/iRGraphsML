import os
from abc import ABCMeta, abstractmethod

import rdflib
import rdflib as rdf
import pandas as pd

__all__ = ["AIFBDataset", "MUTAGDataset", "BGSDataset", "CarcinogenesisDataset"]


class RDFGraphDataset(metaclass=ABCMeta):
    """Abstract base class for RDF graph datasets.

        This class provides a framework for working with RDF graph datasets.
        Parameters
        ----------
        rootpath :
            The root directory path where RDF data is stored.
        training_path :
            The directory path where training  data is stored.
        test_path : str or None, optional
            The directory path where test data is stored.

        Methods
        -------
        load_rdf_graph()
        get_unique_classes(rootpath, training_path)
        get_most_prominent_class(outer_to_inner)

    """
    def __init__(self, rootpath, training_path, test_path):
        self.rootpath = rootpath
        self.training_path = training_path
        self.test_path = test_path

    def load_rdf_graph(self):
        """Load raw RDF data from files in the specified rootpath.

        Returns
        -------
        rdflib.Graph
            A combined RDF graph containing data from all files .
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
    def get_unique_classes(self):
        """Abstract method to be implemented by subclasses.
            Returns
            -------
            dict
                A dictionary mapping unique classes to their respective data.
        """
        pass

    def get_most_prominent_class(self, outer_to_inner):
        """Get the most prominent class from a dictionary mapping outer classes to inner classes.

            Parameters
            ----------
            outer_to_inner : dict
                A dictionary mapping outer classes to inner classes.

            Returns
            -------
            str
                The outer class with the highest count of inner classes.
        """
        label_counts = {}
        for outer_key in outer_to_inner:
            inner_dict = outer_to_inner[outer_key]
            label_counts[outer_key] = len(inner_dict)
        max_key = max(label_counts, key=label_counts.get)
        return max_key


class AIFBDataset(RDFGraphDataset):
    '''
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

    def __init__(self, rootpath, training_path, test_path):
        super().__init__(rootpath, training_path, test_path)

    def get_unique_classes(self):
        # Read the TSV file into a DataFrame
        filepath = self.rootpath + self.training_path
        df = pd.read_csv(filepath, sep='\t')
        # Get the unique values from the "label" column as a list
        unique_classes = df['label_affiliation'].unique().tolist()
        return unique_classes

    def read_training_data(self):
        '''
        Read and process testing data from a TSV file.
        Returns a  dictionary where class labels are keys, and their corresponding values are dictionaries
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
        """Read and process testing data from a TSV file.

            Returns a dictionary where class labels are keys, and their corresponding values are dictionaries
            mapping person names to data IDs.
        """
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

    """

    def __init__(self, rootpath, training_path, test_path):
        super().__init__(rootpath, training_path, test_path)

    def get_unique_classes(self):
        # Read the TSV file into a DataFrame
        filepath = self.rootpath + self.training_path
        df = pd.read_csv(filepath, sep='\t')
        # Get the unique values from the "label" column as a list
        unique_classes = df['label_mutagenic'].unique().tolist()
        return unique_classes

    def read_training_data(self):
        """Read and process training data from a TSV file.

        Returns
        -------
        dict
        """
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
        """Read and process Test data from a TSV file.

        Returns
        -------
        dict
        """
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


class BGSDataset(RDFGraphDataset):
    r"""BGS raw_data for node classification task
    BGS raw_data statistics:
    - Number of Classes: 2
    - Label Split:
        - Train: 117
        - Test: 29
    """

    def __init__(self, rootpath, training_path, test_path):
        super().__init__(rootpath, training_path, test_path)

    def get_unique_classes(self):
        # Read the TSV file into a DataFrame
        filepath = self.rootpath + self.training_path
        df = pd.read_csv(filepath, sep='\t')
        # Get the unique values from the "label" column as a list
        unique_classes = df['label_lithogenesis'].unique().tolist()
        return unique_classes

    def read_training_data(self):
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


class CarcinogenesisDataset(RDFGraphDataset):
    r"""BGS raw_data for node classification task
    BGS raw_data statistics:
    - Number of Classes: 2
    - Label Split:
        - Train: 117
        - Test: 29
    """

    def __init__(self, rootpath, training_path, test_path):
        super().__init__(rootpath, training_path, test_path)

    def get_unique_classes(self):
        # Read the TSV file into a DataFrame
        filepath = self.rootpath + self.training_path
        df = pd.read_csv(filepath, sep='\t')
        # Get the unique values from the "label" column as a list
        unique_classes = df['label'].unique().tolist()
        return unique_classes

    def read_training_data(self):
        """Read and process training data from a TSV file.

        Returns
        -------
        dict
        """
        # Read the TSV file into a DataFrame
        filepath = self.rootpath + self.training_path
        df = pd.read_csv(filepath, sep='\t')
        data_dict = {}
        # Iterate through the DataFrame and populate the dictionary
        for index, row in df.iterrows():
            label = row['label']
            bond = row['bond_id']

            if label not in data_dict:
                data_dict[label] = {}

            data_dict[label][bond] = bond

        return data_dict

    def read_testing_data(self):
        """Read and process Test data from a TSV file.

        Returns
        -------
        dict
        """
        # Read the TSV file into a DataFrame
        filepath = self.rootpath + self.test_path
        df = pd.read_csv(filepath, sep='\t')
        data_dict = {}
        # Iterate through the DataFrame and populate the dictionary
        for index, row in df.iterrows():
            label = row['label']
            bond = row['bond_id']

            if label not in data_dict:
                data_dict[label] = {}

            data_dict[label][bond] = bond

        return data_dict
