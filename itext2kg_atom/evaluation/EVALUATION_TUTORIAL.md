# ATOM framework evaluation tutorial

## Initial setup
All the scripts have been configure to read environment variables stored in [.env](../../.env) (check out [env_config.py](../../env_config.py) for default values) and an input dataset in pickle format `../datasets/atom/my_test_datasets/dataset_input.pkl`. The input dataset is expected to have the following columns:
- COLUMN_NAME_DATE: the observation date of the news paragraph
- COLUMN_NAME_DATE_TRANSLATED_PARAGRAPH: the observation date plus the news paragraph in English
- COLUMN_NAME_FACTOIDS_GROUND_TRUTH: the ground truth extracted factoids (aka AtomicFacts) as a list of strings for each dataset row
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
COLUMN_NAME_FACTOIDS_EXTRACTED=factoids_extracted
COLUMN_NAME_QUINTUPLES_EXTRACTED=quintuples_extracted
COLUMN_NAME_QUINTUPLES_EXTRACTED_FROM_RAW_TEXT=quintuples_extracted_from_raw_text
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
Depending on which model and backend you are going to use, you shoud edit the `EVAL_MODEL_POSTFIXES_LIST` and `EVAL_MODEL_POSTFIXES_TO_PLOT_LIST` env vars accordingly.

### Exhaustivity - Recall (factoids and quintuples)
Adds the following columns to the output dataset:
- COLUMN_NAME_FACTOIDS_EXTRACTED_model_postfix
- COLUMN_NAME_QUINTUPLES_EXTRACTED_model_postfix
- COLUMN_NAME_QUINTUPLES_EXTRACTED_FROM_RAW_TEXT_model_postfix
- COLUMN_NAME_FACTOIDS_EXTRACTION_PROMPT_TOKEN_COUNT
- COLUMN_NAME_QUINTUPLES_EXTRACTION_PROMPT_TOKEN_COUNT
- COLUMN_NAME_QUINTUPLES_RAW_EXTRACTION_PROMPT_TOKEN_COUNT

Repeat the following command for each model postfix you want to test.  
Make sure all the postfix exists inside `EVAL_MODEL_POSTFIXES_LIST`
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

### Quintuples quality - Precision
Make sure you have executed all the Exhaustivity scripts and to have all the required columns inside the `dataset_output.pkl`.  


---
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