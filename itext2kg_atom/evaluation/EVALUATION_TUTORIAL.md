# ATOM framework evaluation tutorial

## Initial setup

All the scripts have been configured to read environment variables stored in [.env](../../.env) (check out [env_config.py](../../env_config.py) for default values) and an input dataset in pickle format `../datasets/atom/my_test_datasets/dataset_input.pkl`. The input dataset is expected to have the following columns:

- COLUMN_NAME_DATE: the observation date of the news paragraph
- COLUMN_NAME_DATE_TRANSLATED_PARAGRAPH: the observation date plus the news paragraph in English
- COLUMN_NAME_FACTOIDS_GROUND_TRUTH: the ground truth factoids (aka AtomicFacts) as a list of strings, for each dataset row
- COLUMN_NAME_QUINTUPLES_GROUND_TRUTH: the ground truth quintuples as a list of tuples (Subject_entity, Relation, Object_entity, Time_Start, Time_End), for each dataset row
  Make sure you have the following env variables correctly setup before proceeding:

```bash
# Delete this variable or set it to 0 to process all rows
NUM_ROWS_TO_PROCESS=10
# Batch size for document parsing (number of paragraphs to process in parallel)
DOC_PARSER_BATCH_SIZE=2
# Column names in the input Excel file (uncomment them if they differ from the defaults 'date' and 'lead_paragraph')
COLUMN_NAME_DATE=date
COLUMN_NAME_PARAGRAPH=ARTICOLO

COLUMN_NAME_SENTIMENT=SENTIMENTO
COLUMN_NAME_TRANSLATED_PARAGRAPH=translated_paragraph
COLUMN_NAME_TRANSLATED_SENTIMENT=translated_sentiment
COLUMN_NAME_DATE_TRANSLATED_PARAGRAPH=lead_paragraph_observation_date

COLUMN_NAME_FACTOIDS_EXTRACTION_PROMPT_TOKEN_COUNT=factoids_prompt_tokenc
COLUMN_NAME_QUINTUPLES_EXTRACTION_PROMPT_TOKEN_COUNT=quintuples_prompt_tokenc
COLUMN_NAME_QUINTUPLES_RAW_EXTRACTION_PROMPT_TOKEN_COUNT=quintuples_raw_prompt_tokenc
COLUMN_NAME_FACTOIDS_EXTRACTED=factoids_extracted
COLUMN_NAME_QUINTUPLES_EXTRACTED=quintuples_extracted
COLUMN_NAME_QUINTUPLES_EXTRACTED_FROM_RAW_TEXT=quintuples_extracted_from_raw_text
COLUMN_NAME_FACTOIDS_GROUND_TRUTH=factoids_g_truth
COLUMN_NAME_QUINTUPLES_GROUND_TRUTH=quintuples_g_truth
# Supported models postfixes (use space as a separator!)
EVAL_MODEL_POSTFIXES_LIST="llamacpp_gemma4 ollama_gemma4"
EVAL_MODEL_POSTFIXES_TO_PLOT_LIST="llamacpp_gemma4 ollama_gemma4"

# Evaluation settings
EVAL_BASE_PATH=./datasets/atom/my_test_datasets
EVAL_INPUT_DATASET_PATH="${EVAL_BASE_PATH}/dataset_input.pkl"
EVAL_OUTPUT_DATASET_PATH="${EVAL_BASE_PATH}/dataset_output.pkl"
EVAL_OUTPUT_RESULTS_PATH="${EVAL_BASE_PATH}/evaluation_results"
EVAL_CHECKPOINT_FACTOIDS_PATH="${EVAL_BASE_PATH}/factoids_checkpoint.json"
EVAL_CHECKPOINT_QUINTUPLES_PATH="${EVAL_BASE_PATH}/quintuples_checkpoint.json"
```

---

## Execute the tests

The updated dataset will be saved into `dataset_output.pkl` and the the tests results will be saved into `${EVAL_BASE_PATH}/evaluation_results`.  
Open up a terminal and do:

```
# if you are using llama.cpp backend
start-llama-servers
# else, make sure you have your models backend up and running!

cd itext2kg_atom/evaluation
```

Open up [models.py](../../models/models.py) and check if its functions return your desired llm model and embedding model. If no, check out [models_config.py](../../models/models_config.py).  
<br> Depending on which model and backend you are going to use, you shoud edit the `EVAL_MODEL_POSTFIXES_LIST` and `EVAL_MODEL_POSTFIXES_TO_PLOT_LIST` env vars accordingly.

### Exhaustivity - Recall (factoids and quintuples)

Adds the following columns to the output dataset:

- COLUMN_NAME_FACTOIDS_EXTRACTED_model_postfix
- COLUMN_NAME_QUINTUPLES_EXTRACTED_model_postfix
- COLUMN_NAME_QUINTUPLES_EXTRACTED_FROM_RAW_TEXT_model_postfix
- COLUMN_NAME_FACTOIDS_EXTRACTION_PROMPT_TOKEN_COUNT
- COLUMN_NAME_QUINTUPLES_EXTRACTION_PROMPT_TOKEN_COUNT
- COLUMN_NAME_QUINTUPLES_RAW_EXTRACTION_PROMPT_TOKEN_COUNT

<br> Repeat the following command for each model postfix you want to test.  
<br> Make sure all the postfix exists inside `EVAL_MODEL_POSTFIXES_LIST`

```bash
python ./exhaustivity/factoids_extraction_nyt.py -p <your-model-postfix>
# python ./exhaustivity/quintuples_extraction_nyt.py -p <your-model-postfix>
python ./exhaustivity/quintuples_extraction_nyt_from_factoids.py -p <your-model-postfix>
```

Once you are done, you can plot the results through:

```bash
python ./exhaustivity/plot_exhaustivity_factoids.py --force-recalculate
python ./exhaustivity/plot_exhaustivity_quintuples.py --force-recalculate
```

| Output metric name         | Formula         | Description                                                                                                                                                                                                                                                         |
| -------------------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Factoids Semantic Recall   | avg(f_recall)   | `similarity_matrix = cosine_similarity(quintuple_embeddings, gold_quintuple_embeddings)`. <br> Bipartite matching (Hungarian Algorithm): compare each factoids with all g_truth and match the one that minimizes the cost, greater than a threshold (default=0.7).  |
| Factoids Temporal Recall   | avg(f_recall_t) | Same as semantic, but for each factoids, check if t_start and t_end of g_truth and preduction temporally overlaps                                                                                                                                                   |
| Quintuples Semantic Recall | avg(q_recall)   | `similarity_matrix = cosine_similarity(quintuple_embeddings, gold_quintuple_embeddings)`. <br> Greedy match: for each quintuple, look for the g_truth with highest similarity, greater than a threshold (default=0.7) in the corresponding row of similarity_matrix |
| Quintuples Temporal Recall | avg(q_recall_t) | Same as semantic, but for each quintuple, check if t_start and t_end of g_truth and prediction matches, through dateparser.                                                                                                                                         |

### Quintuples quality - Precision

Make sure you have executed all the Exhaustivity scripts and to have all the required columns inside the `dataset_output.pkl`.  
<br>Repeat the following command for each model postfix you want to test.

```bash
python ./quintuples_quality/calculate_quintuples_quality.py -p <your-model-postfix>
```

| Output metric name                     | Formula                                            | Description                                                                                           |
| -------------------------------------- | -------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `total_gold`                           | count(df[`COLUMN_NAME_QUINTUPLES_GROUND_TRUTH`])   | The total number of reference ground truth quintuples                                                 |
| `total_predicted`                      | count(df[`COLUMN_NAME_QUINTUPLES_EXTRACTED`])      | The total number of quintuples extracted by the LLM                                                   |
| `MATCH` - Semantic Recall              | $\frac{MATCH_{count}} {total_{gold}}$              | It tells us how many g_truth have been matched by the extracted quintuples                            |
| `OM` - Omission rate                   | $\frac{OM_{count}} {total\_{gold}} = 1-Recall$     | It tells us how much of the quintuples g_truth was forgotten.                                         |
| `HALL` - Hallucination Rate            | $\frac{HALL_{count}} {total_{gold}} = 1-Precision$ | Represents what percentage of the model's generated output could not be matched to the gold standard. |
| `MATCH_t` - Temporal Recall            | (only computed if positive semantic match)         | The model got the facts right and the time matched                                                    |
| `OM_t` - Temporal Hallucination Rate   | (only computed if positive semantic match)         | The model got the facts right, but missed or left out the time context.                               |
| `HALL_t` - Temporal Hallucination Rate | (only computed if positive semantic match)         | The model got the facts right, but fabricated or severely changed the date.                           |

This test evaluates the quintuples extraction in two cases:

1. Quintuples extracted from raw text: values Precision. It generates fewer tuples, which keeps noise low, but it suffers from poor context extraction, leading to high omission rates on complex data blocks.
2. Quintuples extracted from Atomic Fact decomposition: values Recall/Exhaustivity. It ensures that subtle nuances aren't ignored, resulting in a much more complete quintuples, though it introduces some redundant ones.

**Caution**: the _"Hallucination rate"_ term might be misleading, because the model is NOT generating false information! In this script it simply means "The model generated a true fact that the human annotator didn't bother to include in the gold standard". It could be refactored as "Redundancy rate".

## Test results

You can find every test results inside `EVAL_OUTPUT_RESULTS_PATH`.  
To easily convert from pkl to excel and vice-versa, open a terminal in your dataset directory and execute:

```bash
cd $EVAL_BASE_PATH
python
import pandas as pd
# From PKL to EXCEL
pd.read_pickle("dataset_output.pkl").to_excel("dataset_output.xlsx")
# From EXCEL to PKL
pd.read_excel("dataset_input.xlsx").to_excel("dataset_input.pkl")
quit()
```
