from rdflib import URIRef


class ConceptGenerator:
    """
    A class that generates the concept for the random walks
    """

    def __init__(self, triples):
        self.concept = None
        self.triples = triples
        self.path = []
        self.pattern = self.path
        # self.concept = self.generate_alc_expression(self.path)

    def get_concept(self):
        return self.concept

    def set_concept(self, concept):
        self.concept = concept

    @staticmethod
    def remove_uri_prefix(uri):
        if isinstance(uri, URIRef):
            return str(uri).split("#")[-1]
        return str(uri)

    @staticmethod
    def union(subject, obj):
        # Add union
        alc_expression = f"{subject}⊔{obj}"
        return alc_expression

    @staticmethod
    def conjunction(subject, obj):
        # Add ⊓
        alc_expression = f"{subject}⊓{obj}"
        return alc_expression

    @staticmethod
    def existential_quantifier(predicate, obj):
        # Add existential quantifier
        alc_expression = f"∃{predicate}.{obj}"
        return alc_expression

    def universal_quantifier(self):
        # Add universal quantifier
        pass

    def generate_alc_expression(self, path):
        subject = path.pop(0)

        if len(path) >= 2:
            while len(path) >= 2:
                pred = path.pop(0)
                obj = path.pop(0)
                temp = ConceptGenerator.existential_quantifier(pred, obj)
                new_subject = ConceptGenerator.conjunction(subject, temp)
                subject = new_subject
                concept = ''.join(subject)
        else:
            concept = subject
        self.set_concept(concept)

    def get_path(self):
        return self.pattern

    def generate_path(self):
        for item in self.triples:
            self.path.append(ConceptGenerator.remove_uri_prefix(item))
        return self.path
