import pandas as pd
import ast
import asyncio
import logging
from models.models import *

# Import LLM and Embeddings models using LangChain wrappers
from itext2kg.atom import Atom
from itext2kg import Neo4jStorage


# Configure logging to see itext2kg intermediary steps
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get default llm model and embedding model
openai_llm_model = get_default_model()
openai_embeddings_model = get_default_embedding_model()

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
    
    # Load the 2020-COVID-NYT dataset pickle
    news_covid = pd.read_pickle("./itext2kg-1.0.0/datasets/atom/nyt_news/2020_nyt_COVID_last_version_ready.pkl")

    # Convert the dataframe into the required dictionary format
    news_covid_dict = to_dictionary(news_covid)

    # Initialize the ATOM pipeline with the OpenAI models
    atom = Atom(llm_model=openai_llm_model, embeddings_model=openai_embeddings_model)

    # Build the knowledge graph across different observation timestamps
    kg = await atom.build_graph_from_different_obs_times(
        atomic_facts_with_obs_timestamps=news_covid_dict,
        
    )

    # Visualize the resulting knowledge graph using Neo4j
    URI = neo4j_uri
    USERNAME = neo4j_username
    PASSWORD = neo4j_password

    logger.info("Connecting to Neo4j and visualizing graph...")
    Neo4jStorage(uri=URI, username=USERNAME, password=PASSWORD).visualize_graph(knowledge_graph=kg)
    logger.info("Graph visualization complete!")


if __name__ == "__main__":
    asyncio.run(main())