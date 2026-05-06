from itext2kg_atom.itext2kg.documents_distiller.documents_distiller import *

"""
Wrapper function that instanciate a new document distiller object
"""
def parse_documents(llm_model):
    parser = DocumentsDistiller(llm_model)