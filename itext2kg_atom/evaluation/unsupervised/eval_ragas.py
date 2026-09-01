"""
This script provides an unsupervised evaluation of ATOM framework, using Ragas.io.

By default uses the 2020-NYT-covid-19 english dataset, obtained from supervised evaluation.
You can specifiy --dataset path to use a different one.

It is possible to pass a non-english dataset as well, analyze the translated_sentiment column and compare column_name_paragraph to column_name_translated_paragraph
"""

import os
import sys
import time
import logging
import argparse
import pandas as pd
from pathlib import Path
from ragas import evaluate
from ragas.metrics import Faithfulness, AspectCritic
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.llms import LangchainLLMWrapper

from itext2kg.atom.models.prompts import Prompt
from document_parser.parser_prompt import ParserPrompt

from models.models import get_default_model, get_default_embedding_model
from env_config import (
    eval_output_dataset_path, eval_input_knowledge_graph_path, num_rows_to_process, enable_translation,
    column_name_date, column_name_paragraph, column_name_translated_paragraph,
    column_name_translated_sentiment,
    column_name_factoids_extracted, column_name_quintuples_extracted_from_raw_text, column_name_quintuples_extracted,
    eval_model_postfixes_list
)

# Add the project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

DATASET: pd.DataFrame = None
ragas_llm = LangchainLLMWrapper(get_default_model())


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
    ]
)
logger = logging.getLogger(__name__)

print("🚀 Starting ragas evaluation script...")
logger.info("Setting up API connections...")


# ------------ Define Custom Unsupervised Metrics ------------

# --- Step 1 : atomic fact extraction ------------------------------------------
 
step1_faithfulness = Faithfulness()
"""
Decomposes each atomic fact into sub-claims and verifies each against the source
paragraph using an LLM-as-NLI judge.  Score = fraction of claims supported.
Catches hallucinated facts that were never in the original text.
"""

exhaustivity = AspectCritic(
    name="exhaustivity",
    definition=(
        "Do the extracted atomic facts collectively capture every essential entity, "
        "event, and claim present in the source paragraph, without omitting key information?"
    ),
)
 
atomicity = AspectCritic(
    name="atomicity",
    definition=(
        "Is each extracted fact truly minimal and self-contained — expressing exactly "
        "one distinct claim — without conflating multiple separate assertions into a "
        "single statement?"
    ),
)

# --- Step 2 : quintuple extraction --------------------------------------------

step2_faithfulness = Faithfulness()
"""
Same NLI-based check, but now the context is the atomic facts and the response
is the serialised quintuples.  Catches quintuples invented by the LLM that
have no basis in the atomic facts provided.
"""

temporal_soundness = AspectCritic(
    name="temporal_soundness",
    definition=(
        "Are the t_start and t_end values in the quintuples logically consistent "
        "and explicitly or implicitly supported by the source text?  "
        "Empty temporal fields are acceptable when the text provides no grounding; "
        "fabricated dates are not."
    ),
)

predicate_quality = AspectCritic(
    name="predicate_quality",
    definition=(
        "Are the predicates in the quintuples specific, informative, and expressed as "
        "present-tense verb phrases that precisely capture the directional relationship "
        "between subject and object — avoiding vague predicates such as 'related_to' "
        "or 'associated_with'?"
    ),
)

# --- Cross-step : end-to-end grounding ----------------------------------------
 
e2e_faithfulness = Faithfulness()
"""
Used twice: once with quintuples extracted directly from raw text, once with
quintuples produced via the two-step (atomic facts) route.
Comparing the two scores quantifies the benefit of the ATOM decomposition.
"""
 
# --- Step 4/5 : entity merge decisions ----------------------------------------
 
merge_correctness = AspectCritic(
    name="merge_correctness",
    definition=(
        "Given the names and contexts of Entity A and Entity B, do they unambiguously "
        "refer to the exact same real-world entity, making it safe to merge them into "
        "a single node without introducing false-positive coreference?"
    ),
)

# --- Optional : translation quality -------------------------------------------
 
translation_fidelity = AspectCritic(
    name="translation_fidelity",
    definition=(
        "Does the English translation faithfully preserve all factual content, named "
        "entities (especially proper nouns and surnames), and meaning from the Italian "
        "source — without adding, removing, or distorting information?"
    ),
)

sentiment_alignment = AspectCritic(
    name="sentiment_alignment",
    definition=(
        "Does the emotional tone and sentiment of the English translation match the "
        "Italian original, with no unintended shift toward a more positive or more "
        "negative framing of the described events?"
    ),
)
 
# sentiment_assignment = AspectCritic(
#     name="sentiment_alignment",
#     definition=(
#         "The **sentiment** value represents the general linguistic tone, on a scale from 1 to 5, with increments of 0.5."
#         "**REFERENCE SCALE:**"
#         "- 1 = very negative"
#         "- 2 = negative"
#         "- 3 = neutral"
#         "- 4 = positive"
#         "- 5 = very positive"
#         "**Intermediate values (1.5, 2.5, 3.5, 4.5) are nuances between two adjacent categories**"
#         "Is the sentiment value of the English translation correctly assigned?"
#     ),
# )


# ── Step 1 evaluation ─────────────────────────────────────────────────────────
 
def build_step1_samples(
    dataset: pd.DataFrame, col_paragraph: str, col_factoids: str, col_date: str
) -> list[SingleTurnSample]:
    samples = []
    for _, row in dataset.iterrows():
        obsdate = str(row.get(col_date, "")).strip()
        paragraph = str(row.get(col_paragraph, "")).strip()
        factoids  = str(row.get(col_factoids,  "")).strip()
        if not paragraph or not factoids:
            continue
        samples.append(
            SingleTurnSample(
                user_input=(
                    "Extract minimal, self-contained atomic facts from the provided "
                    "news paragraph.  Each fact must express exactly one claim."

                    #ParserPrompt._create_temporal_system_query(col_date)
                ),
                retrieved_contexts=[paragraph],
                response=factoids,
            )
        )
    return samples
 
 
def evaluate_step1(
    dataset: pd.DataFrame, col_paragraph: str, col_factoids: str, col_date: str
):
    logger.info("▶ Step 1 — atomic fact extraction …")
    samples = build_step1_samples(dataset, col_paragraph, col_factoids, col_date)
    if not samples:
        logger.warning("  No valid samples for Step 1; skipping.")
        return None
    return evaluate(
        dataset=EvaluationDataset(samples),
        metrics=[step1_faithfulness, exhaustivity, atomicity],
        llm=ragas_llm,
    )
 
 
# ── Step 2 evaluation ─────────────────────────────────────────────────────────
 
def build_step2_samples(
    dataset: pd.DataFrame, col_factoids: str, col_quintuples: str, col_date: str
) -> list[SingleTurnSample]:
    samples = []
    for _, row in dataset.iterrows():
        obsdate = str(row.get(col_date, "")).strip()
        factoids   = str(row.get(col_factoids,   "")).strip()
        quintuples = str(row.get(col_quintuples, "")).strip()
        if not obsdate or not factoids or not quintuples:
            continue
        samples.append(
            SingleTurnSample(
                user_input=(
                    "Convert the atomic facts into structured quintuples of the form "
                    "(subject, predicate, object, t_start, t_end), where t_start and "
                    "t_end represent the temporal scope of the relationship."

                    #Prompt.temporal_system_query(obsdate) + Prompt.EXAMPLES.value
                ),
                retrieved_contexts=[factoids],
                response=quintuples,
            )
        )
    return samples
 
 
def evaluate_step2(
    dataset: pd.DataFrame, col_factoids: str, col_quintuples: str, col_obs_date: str
):
    logger.info("▶ Step 2 — quintuple extraction …")
    samples = build_step2_samples(dataset, col_factoids, col_quintuples, col_obs_date)
    if not samples:
        logger.warning("  No valid samples for Step 2; skipping.")
        return None
    return evaluate(
        dataset=EvaluationDataset(samples),
        metrics=[step2_faithfulness, temporal_soundness, predicate_quality],
        llm=ragas_llm,
    )

# ── Cross-step : end-to-end grounding ─────────────────────────────────────────

def evaluate_e2e_grounding(
    dataset: pd.DataFrame,
    col_paragraph: str,
    col_quintuples_raw: str,
    col_quintuples_two_step: str,
) -> dict:
    """
    Compares faithfulness of quintuples extracted directly from raw text
    vs. quintuples produced via the two-step (atomic facts) route.
    Both are measured against the original paragraph as ground context.
    A higher score for the two-step route validates the ATOM decomposition.
    """
    logger.info("▶ Cross-step — end-to-end grounding comparison …")
    samples_raw, samples_two_step = [], []
    for _, row in dataset.iterrows():
        paragraph = str(row.get(col_paragraph, "")).strip()
        q_raw     = str(row.get(col_quintuples_raw,      "")).strip()
        q_2step   = str(row.get(col_quintuples_two_step, "")).strip()
        if not paragraph:
            continue
        base_input = "Extract quintuples (subject, predicate, object, t_start, t_end) from the text."
        #base_input = Prompt.temporal_system_query + Prompt.EXAMPLES

        if q_raw:
            samples_raw.append(SingleTurnSample(
                user_input=base_input + " [direct extraction from raw text]",
                retrieved_contexts=[paragraph],
                response=q_raw,
            ))
        if q_2step:
            samples_two_step.append(SingleTurnSample(
                user_input=base_input + " [via atomic fact decomposition]",
                retrieved_contexts=[paragraph],
                response=q_2step,
            ))
 
    results = {}
    if samples_raw:
        results["e2e_raw_text"] = evaluate(
            dataset=EvaluationDataset(samples_raw),
            metrics=[Faithfulness()],
            llm=ragas_llm,
        )
    if samples_two_step:
        results["e2e_two_step"] = evaluate(
            dataset=EvaluationDataset(samples_two_step),
            metrics=[Faithfulness()],
            llm=ragas_llm,
        )
    return results

# ── Translation evaluation ────────────────────────────────────────────────────
 
def build_translation_samples(
    dataset: pd.DataFrame, col_italian: str, col_english: str
) -> list[SingleTurnSample]:
    samples = []
    for _, row in dataset.iterrows():
        italian = str(row.get(col_italian, "")).strip()
        english = str(row.get(col_english, "")).strip()
        if not italian or not english:
            continue
        samples.append(
            SingleTurnSample(
                user_input=(
                    "Translate the Italian financial news paragraph faithfully into "
                    "English, preserving all named entities, proper nouns, and sentiment."
                ),
                retrieved_contexts=[italian],
                response=english,
            )
        )
    return samples


def evaluate_translation(
    dataset: pd.DataFrame, col_italian: str, col_english: str
):
    logger.info("▶ Translation quality …")
    samples = build_translation_samples(dataset, col_italian, col_english)
    if not samples:
        logger.warning("  No valid translation samples; skipping.")
        return None
    return evaluate(
        dataset=EvaluationDataset(samples),
        metrics=[translation_fidelity, sentiment_alignment],
        llm=ragas_llm,
    )

# ── Entity merge evaluation ───────────────────────────────────────────────────
 
def build_merge_samples(merged_pairs: list[dict]) -> list[SingleTurnSample]:
    """
    merged_pairs: list of dicts with keys:
        entity_a  (str) — name of the first entity
        entity_b  (str) — name of the second entity
        context   (str) — the atomic fact(s) in which both were mentioned
    
    How to populate this from your KG:
        Load the exported KG and iterate over merged node pairs, collecting
        the atomic_facts fields of their incident edges as context.
        Sample a manageable subset (e.g. 50–100 pairs) to keep costs down.
    """
    samples = []
    for pair in merged_pairs:
        a       = pair.get("entity_a", "")
        b       = pair.get("entity_b", "")
        context = pair.get("context",  "No additional context available.")
        if not a or not b:
            continue
        samples.append(
            SingleTurnSample(
                user_input=(
                    f"Entity A: \"{a}\"\n"
                    f"Entity B: \"{b}\"\n"
                    "Do these two entity names refer to the exact same real-world "
                    "entity, such that merging them into a single knowledge-graph node "
                    "is correct and does not introduce a false-positive coreference?"
                ),
                retrieved_contexts=[context],
                # The pipeline decided to merge them; we evaluate that decision.
                response="Yes, they refer to the same real-world entity and should be merged.",
            )
        )
    return samples
 
 
def evaluate_merge_decisions(merged_pairs: list[dict]):
    logger.info(f"▶ Entity merge correctness ({len(merged_pairs)} pairs) …")
    samples = build_merge_samples(merged_pairs)
    if not samples:
        logger.warning("  No merge pairs provided; skipping.")
        return None
    return evaluate(
        dataset=EvaluationDataset(samples),
        metrics=[merge_correctness],
        llm=ragas_llm,
    )

# To check from here and below









"""
temporal_quality = AspectCritic(
    name="temporal_quality",
    definition="Are the t_start and t_end values in the quintuples strictly logically sound and grounded in the source text?"
)

merge_validity = AspectCritic(
    name="merge_validity",
    definition="Given two concepts (Entity A and Entity B), do they represent the exact same real-world entity without false positive overlapping?"
)

def evaluate_atom_pipeline(raw_text: str, atomic_facts: str, quintuples: str):
    
    #Evaluates extraction exhaustivity and quintuple quality reference-free.
    
    # Evaluate Fact Extraction
    step1_sample = SingleTurnSample(
        user_input="Extract minimal, self-contained atomic facts from the provided text.",
        retrieved_contexts=[raw_text],
        response=atomic_facts
    )
    
    # Evaluate Quintuple Structuring
    step2_sample = SingleTurnSample(
        user_input="Extract structured quintuples (subject, predicate, object, t_start, t_end).",
        retrieved_contexts=[atomic_facts], 
        response=quintuples
    )
    
    dataset = EvaluationDataset([step1_sample, step2_sample])
    
    # Execute Ragas evaluation using local LLM
    results = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), exhaustivity, temporal_quality],
        llm=ragas_llm
    )
    
    return results

# Evaluate Graph Merging
def evaluate_merges(entity_a: str, entity_b: str, context: str):
    merge_sample = SingleTurnSample(
        user_input=f"Should '{entity_a}' and '{entity_b}' be merged?",
        retrieved_contexts=[context],
        response="Yes, they represent the same concept." 
    )
    return evaluate(
        dataset=EvaluationDataset([merge_sample]),
        metrics=[merge_validity],
        llm=ragas_llm
    )
"""
# (Only for non english datasets) Evaluate paragraph translation
def evaluate_english_translations():
    if column_name_translated_paragraph not in DATASET.columns:
        return
    # TODO

# (Only for non english datasets) Evaluate sentiment translation
def evaluate_sentiment_translations():
    if column_name_translated_sentiment not in DATASET.columns:
        return
    # TODO

def save_results(results_dict: dict, output_dir: str, postfix: str = ""):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for name, result in results_dict.items():
        if result is None:
            continue
        out_path = Path(output_dir) / f"{name}{postfix}.json"
        df = result.to_pandas()
        df.to_json(out_path, orient="records", indent=2)
        # Log mean scores for a quick sanity-check in the terminal
        numeric_cols = df.select_dtypes("number").columns.tolist()
        summary = {c: round(float(df[c].mean()), 4) for c in numeric_cols}
        logger.info(f"  {name:30s} → {summary}  (saved to {out_path.name})")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Unsupervised Ragas evaluation of the ATOM pipeline')
    parser.add_argument('--model-postfix', '-p', type=str, required=False,
                       help='The postfix representing the backend and model you are executing the test. You can define all supported postfixes inside your .env file, through $EVAL_MODEL_POSTFIXES_LIST variable')
    parser.add_argument('--dataset', '-d', type=str, required=False,
                       help=f"The dataset, in pkl format, used to perform the unsupervised evaluation. If not specified, the default one will be used: ${eval_output_dataset_path}")
    return parser.parse_args()

def main():
    start_time = time.time()
    
    # Parse command line arguments
    args = parse_arguments()
    model_postfix = args.model_postfix if args.model_postfix else ""

    if model_postfix and model_postfix not in eval_model_postfixes_list:
        logger.error(f'Unsupported --model-postfix arg. Supported ones are: {eval_model_postfixes_list}')
        return
    try:
        COL_DATE =                          f"{column_name_date}"
        COL_FACTOIDS_EXTRACTED =            f"{column_name_factoids_extracted}_{model_postfix}"
        COL_QUINTUPLES_RAW_TEXT_EXTRACTED = f"{column_name_quintuples_extracted_from_raw_text}_{model_postfix}"
        COL_QUINTUPLES_EXTRACTED =          f"{column_name_quintuples_extracted}_{model_postfix}"
        COL_ENGLISH_PARAGRAPH =             f"{column_name_translated_paragraph}" if enable_translation else f"{column_name_paragraph}"
        DATASET_PATH = Path(project_root / eval_output_dataset_path) if not args.dataset else Path(project_root / args.dataset)

        print("📊 Loading dataset...")
        DATASET = pd.read_pickle(DATASET_PATH)
        if num_rows_to_process > 0:
            DATASET = DATASET.head(num_rows_to_process)
        logger.info(f"📋 Loaded dataset with {len(DATASET)} rows")

        if COL_DATE not in DATASET.columns:
            raise ValueError(f"Missing dataset column {COL_DATE}")
        if COL_FACTOIDS_EXTRACTED not in DATASET.columns:
            raise ValueError(f"Missing dataset column {COL_FACTOIDS_EXTRACTED}")
        if COL_QUINTUPLES_RAW_TEXT_EXTRACTED not in DATASET.columns:
            raise ValueError(f"Missing dataset column {COL_QUINTUPLES_RAW_TEXT_EXTRACTED}")
        if COL_QUINTUPLES_EXTRACTED not in DATASET.columns:
            raise ValueError(f"Missing dataset column {COL_QUINTUPLES_EXTRACTED}")
        if COL_ENGLISH_PARAGRAPH not in DATASET.columns:
            raise ValueError(f"Missing dataset column {COL_ENGLISH_PARAGRAPH}")
                    

        # ATOM pipeline evaluation
        for row in DATASET:
            evaluate_atom_pipeline(row[COL_ENGLISH_PARAGRAPH], row[COL_FACTOIDS_EXTRACTED], row[COL_QUINTUPLES_EXTRACTED])

        if enable_translation:
            evaluate_english_translations()
            evaluate_sentiment_translations()

    except Exception as e:
        logger.error(f"❌ Error occurred: {str(e)}")
        raise
    finally:
        elapsed_time = time.time() - start_time
        logger.info(f"Processing completed successfully in {elapsed_time:.2f} seconds")
        print(f"Factoid extraction completed successfully in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()