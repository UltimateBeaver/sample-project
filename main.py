import pandas as pd
import ast
import asyncio
import logging
from document_parser.doc_parser import DocumentParser
from models.models import *
from sanity_checks.test_config import validate_config

# Import LLM and Embeddings models using LangChain wrappers
from itext2kg_atom.itext2kg.atom import Atom
from itext2kg_atom.itext2kg import Neo4jStorage
from itext2kg_atom.itext2kg.logging_config import setup_logging
from env_config import *

# Get default llm model and embedding model
base_llm_model = get_default_model()
base_llm_model_no_reasoning = get_default_model_no_reasoning()
base_embeddings_model = get_default_embedding_model()

# Define a helper function to convert the dataframe's atomic facts into a dictionary,
# where keys are observation dates and values are the combined list of atomic facts for that date.
def to_dictionary(df:pd.DataFrame, column_name_atomic_facts: str): 

    if isinstance(df[column_name_atomic_facts][0], str):
        df[column_name_atomic_facts] = df[column_name_atomic_facts].apply(lambda x:ast.literal_eval(x))
    if num_rows_to_process > 0:
        grouped_df = df.groupby(column_name_date)[column_name_atomic_facts].sum().reset_index()[:num_rows_to_process]
    else:
        grouped_df = df.groupby(column_name_date)[column_name_atomic_facts].sum().reset_index()
    return {
        str(date): factoids for date, factoids in grouped_df.set_index(column_name_date)[column_name_atomic_facts].to_dict().items()
        }


## ------------- Document parsing into atomic facts ------------- ##
async def parse_news_paragraphs_into_atomic_facts() -> pd.DataFrame:
    # Covert xlsx file to pkl format
    parser = DocumentParser(
        llm_model=base_llm_model,
        enable_translation=enable_translation
    )
    result_df = await parser.parse_excel(apply_post_processing=True)
    print("\n" + "=" * 70)
    print("📊 EXTRACTION RESULTS")
    print("=" * 70)
    
    for idx, row in result_df.iterrows():
        print(f"\n📅 Date: {row[column_name_date]}")
        print(f"📄 Paragraph: {row[column_name_paragraph][:100]}...")
        print(f"✨ Extracted Atomic Facts ({len(row[column_name_factoids_extracted])} facts):")
        
        if isinstance(row[column_name_factoids_extracted], list) and len(row[column_name_factoids_extracted]) > 0:
            for i, fact in enumerate(row[column_name_factoids_extracted], 1):
                print(f"   {i}. {fact}")
        else:
            print("   [No facts extracted]")
    
    print("\n" + "=" * 70)
    print(f"✅ Results saved to: {doc_parser_output_excel_path}")
    print("=" * 70)
    
    # Display summary statistics
    print("\n📈 SUMMARY STATISTICS")
    print("-" * 70)
    total_factoids = sum(len(facts) if isinstance(facts, list) else 0 
                        for facts in result_df[column_name_factoids_extracted])
    print(f"Total rows processed: {len(result_df)}")
    print(f"Total atomic facts extracted: {total_factoids}")
    print(f"Average factoids per paragraph: {total_factoids / len(result_df):.1f}")

    return result_df


async def main():
    
    # Configure logging with performance optimizations
    # Note: When level="DEBUG", langchain loggers are now set to WARNING by default
    # to avoid excessive debug output from external libraries (which was causing 2x slowdown)
    # You can override this with: langchain_level="DEBUG" to see full debug output from langchain
    
    # Initialize logging configuration
    setup_logging(
        log_file="app.log", 
        level="DEBUG",  # Your itext2kg logs at DEBUG level
        console_output=True,
        langchain_level="WARNING"  # Keep external library logs at WARNING for performance
    )
    logger = logging.getLogger("itext2kg")

    config_ok = await validate_config()
    if not config_ok:
        logger.error("Configuration validation failed. Exiting.")
        return
    
    # Log language configuration
    logger.info(f"🌐 Document processing configuration:")
    logger.info(f"   Translation enabled: {enable_translation}")

    # Extract atomic pieces of information (Atomic facts, aka Factoids) from raw text paragraphs
    df_atomic_facts = await parse_news_paragraphs_into_atomic_facts()
    df_atomic_facts.to_pickle(doc_parser_output_excel_path.replace(".xlsx", ".pkl"))

    # Load the 2020-COVID-NYT dataset pickle (only 10 rows for testing)
    news_covid = pd.read_pickle(doc_parser_output_excel_path.replace(".xlsx", ".pkl"))

    news_covid_dict = news_covid
    # Convert the dataframe into the required dictionary format
    news_covid_dict = to_dictionary(news_covid, column_name_factoids_extracted)

    # Initialize the ATOM pipeline with the LLM (reasoning disabled for quintuples extraction) and embedding models
    atom = Atom(llm_model=base_llm_model_no_reasoning, embeddings_model=base_embeddings_model)

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