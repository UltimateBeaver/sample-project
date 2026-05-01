import pandas as pd
import ast
import asyncio
import logging
from models.models import *
from sanity_checks.test_config import validate_config

# Import LLM and Embeddings models using LangChain wrappers
from itext2kg_atom.itext2kg.atom import Atom
from itext2kg_atom.itext2kg import Neo4jStorage
from itext2kg_atom.itext2kg.logging_config import setup_logging


# Configure logging to see itext2kg intermediary steps
# logging.basicConfig(
#     filename="logs/app.log",
#     level=logging.INFO
# )
# logger = logging.getLogger(__name__)

# Get default llm model and embedding model
base_llm_model = get_default_model()
base_embeddings_model = get_default_embedding_model()

# Define a helper function to convert the dataframe's atomic facts into a dictionary,
# where keys are observation dates and values are the combined list of atomic facts for that date.
def to_dictionary(df:pd.DataFrame, max_elements: int | None = 20): 

    if isinstance(df['factoids_g_truth'][0], str):
        df["factoids_g_truth"] = df["factoids_g_truth"].apply(lambda x:ast.literal_eval(x))
    grouped_df = df.groupby("date")["factoids_g_truth"].sum().reset_index()[:max_elements]
    return {
        str(date): factoids for date, factoids in grouped_df.set_index("date")["factoids_g_truth"].to_dict().items()
        }


async def main():
    
    # Initialize default logging configuration
    setup_logging(
        log_file="app.log", 
        level="DEBUG",
        console_output=True
    )
    logger = logging.getLogger("itext2kg")

    config_ok = await validate_config()
    if not config_ok:
        logger.error("Configuration validation failed. Exiting.")
        return
    
    # Load the 2020-COVID-NYT dataset pickle (only 10 rows for testing)
    #news_covid = pd.read_pickle("./itext2kg-1.0.0/datasets/atom/nyt_news/2020_nyt_COVID_last_version_ready.pkl")
    news_covid = pd.read_pickle("./small_pickle.pkl")

    # Convert the dataframe into the required dictionary format
    news_covid_dict = to_dictionary(news_covid)

    # Initialize the ATOM pipeline with the LLM and embedding models
    atom = Atom(llm_model=base_llm_model, embeddings_model=base_embeddings_model)

    # Initialize the Neo4j storage with connection details from environment variables
    neo4j = Neo4jStorage(
        uri=neo4j_uri, 
        username=neo4j_username, 
        password=neo4j_password
    )
    # Read the existing graph from Neo4j
    existing_kg = None
    neo4j.delete_graph()
    # commented for testing purposes to start with an empty graph each time
    # existing_kg = neo4j.read_graph()
    # if existing_kg.is_empty():
    #     logger.info("No existing graph found in Neo4j. Starting with an empty graph.")
    #     existing_kg = None

    # Build the knowledge graph across different observation timestamps
    kg = await atom.build_graph_from_different_obs_times(
        atomic_facts_with_obs_timestamps=news_covid_dict,
        existing_knowledge_graph=existing_kg
    )

    # Update the resulting knowledge graph in Neo4j
    logger.info("Connecting to Neo4j and updating graph...")
    neo4j.store_graph(knowledge_graph=kg)
    # Neo4jStorage(
    #     uri=neo4j_uri, 
    #     username=neo4j_username, 
    #     password=neo4j_password
    # ).update_graph(knowledge_graph=kg, merge_function=atom.parallel_atomic_merge, delete_existing_graph=True)
    logger.info("Graph storing complete! Check out " + neo4j_uri + " to visualize the graph.")


if __name__ == "__main__":
    asyncio.run(main())