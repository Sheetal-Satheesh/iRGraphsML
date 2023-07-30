from rdflib import URIRef
from utils.elements import Elements


class DLSyntaxGenerator:
    concept = []

    @staticmethod
    def get_concept(walk):
        processed_walk = [DLSyntaxGenerator.__get_short_form(item) for item in walk[0]]
        DLSyntaxGenerator.__create_element(processed_walk)

    @staticmethod
    def __get_short_form(uri):
        if isinstance(uri, URIRef):
            return str(uri).split("#")[-1]
        return str(uri)

    @staticmethod
    def __create_element(processed_walk):
        subject = Elements(processed_walk.pop(0))
        DLSyntaxGenerator.concept.append(subject)

        while len(processed_walk) >= 2:
            item = processed_walk.pop(0)
            quantifier = 'existential'
            restriction = processed_walk.pop(0)

            # Create an Element object with the provided values
            element = Elements(item, quantifier, restriction)
            DLSyntaxGenerator.concept.append(element)

