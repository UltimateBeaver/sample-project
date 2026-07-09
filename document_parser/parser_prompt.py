from datetime import datetime, timedelta

class ParserPrompt():
    @staticmethod
    def _create_temporal_system_query_ollama_version(observation_date: str) -> str:
        """
        Create a comprehensive system query for exhaustive atomic facts extraction.
        Uses the AtomicFact schema description as the foundation and adds:
        - Emphasis on EXHAUSTIVE extraction (all facts, including supporting details)
        - Better date conversion examples with explicit mappings
        - Explicit examples of what NOT to extract (meta-information, noise, irrelevant details)
        - Clear prohibition on extracting "The observation date is" facts
        - Instructions to prevent duplicate/near-duplicate facts
        - Strict guidance on filtering generic descriptions vs. substantive information
        
        Note: This query is designed for the LLM at runtime. The AtomicFact schema 
        description serves a complementary role for JSON structure definition.
        
        Args:
            observation_date: The observation date in format YYYY-MM-DD
            
        Returns:
            System query string with comprehensive temporal context
        """
        obs_date = datetime.strptime(observation_date, '%Y-%m-%d')
        
        # Calculate example dates for the specific observation date
        yesterday = (obs_date - timedelta(days=1)).strftime('%Y-%m-%d')
        three_days_ago = (obs_date - timedelta(days=3)).strftime('%Y-%m-%d')
        week_monday = (obs_date - timedelta(days=obs_date.weekday())).strftime('%Y-%m-%d')
        last_week_monday = (obs_date - timedelta(days=obs_date.weekday() + 7)).strftime('%Y-%m-%d')
        month_first = obs_date.replace(day=1).strftime('%Y-%m-%d')
        year_first = obs_date.replace(month=1, day=1).strftime('%Y-%m-%d')
        last_month_first = (obs_date.replace(day=1) - timedelta(days=1)).replace(day=1).strftime('%Y-%m-%d')
        last_year_first = obs_date.replace(year=obs_date.year - 1, month=1, day=1).strftime('%Y-%m-%d')
        
        return f"""
You are an expert EXHAUSTIVE and RELEVANT atomic facts extraction engine. Your PRIMARY goal is to extract ALL distinct factual information from the input paragraph.


# ABSOLUTE PROHIBITIONS (NEVER extract these):

1. **NEVER include meta-information about the observation date itself:**
   - "The observation date is {observation_date}"
   - "Today's date is {observation_date}"
   - "On {observation_date}, the observation date is being recorded"
    The observation_date is ONLY used for temporal normalization, NOT as content to extract.

2. **NEVER extract purely generic corporate/motivational descriptions without specific content:**
   - "Company X is focused on innovation and excellence"
   - "The organization is dedicated to building a better future"
   - "Y is committed to customer satisfaction and continuous improvement"

3. **NEVER extract unrelated noisy information (things mentioned in passing, meta-commentary or promotional content):**
   - "Discover the top 5 tools for remote workers."
   - "Click here for a deeper understanding of the market shifts."
   - "This content is purely for informational purposes and should not be taken as legal or medical advice"

4. **NEVER extract irrelevant information that is not directly related to the main news:**
    ### Example 1:
    > INPUT: "While I was eating a sandwich, I heard from the TV that the town's jewelry store was robbed this morning."
    > RELEVANT FACT: "The town's jewelry store was robbed this morning."
    > NOISE (DO NOT EXTRACT): "I was eating a sandwich", "I heard from the TV"
    > REASONING: The speaker's meal and the source of information (TV) do not change the facts of the robbery, which is the actual news.

    ### Example 2:
    > INPUT: "The CEO of Company X, while enjoying a cup of coffee, announced a new product line yesterday."
    > RELEVANT FACT: "The CEO of Company X announced a new product line yesterday."
    > NOISE (DO NOT EXTRACT): "while enjoying a cup of coffee"
    > REASONING: The CEO's activity of enjoying a cup of coffee does not change the fact of the announcement, which is the main news.

    ### Example 3:
    > INPUT: "Mark flew to Paris to attend a conference where the Minister announced a new 10% tax on digital services."
    > RELEVANT FACT: "The Minister announced a new 10% tax on digital services."
    > NOISE (DO NOT EXTRACT): "Mark flew to Paris to attend a conference"
    > REASONING: The speaker's travel details and the specific event he was attending are "logistical noise." The actual news is the tax announcement.

    ### Example 4:
    > INPUT: "I was walking my dog in the park when I noticed that the city has finally started the bridge reconstruction project."
    > RELEVANT FACT: "The city has started the bridge reconstruction project."
    > NOISE (DO NOT EXTRACT): "I was walking my dog in the park", "I noticed that"
    > REASONING: The speaker’s activity (walking a dog) and their internal state (noticing something) are irrelevant to the status of the bridge.

5. **NEVER extract the same information multiple times in different phrasings:**
   - If you extract "Event X happened", do NOT also extract "X is something that happened"

   
# WHAT TO EXTRACT (EXHAUSTIVELY):

- **Main events and actions**: "X announced Y", "Z declared an emergency"
- **Supporting facts**: "The event occurred because of...", "It impacts..."
- **Background context**: "This is the first time X did Y", "Historical precedent..."
- **Entity descriptions (ONLY if specific/informative)**: "Company X produces cars", "City Y has population Z"
- **Causal relationships**: "A happened because of B"
- **Relationships and connections**: "X is affiliated with Y", "Z works for company W"
- **Quantitative information**: "X increased by 40%", "Z people were affected"
- **Named entities and roles**: "CEO John Smith", "Organization XYZ supplies sport equipment"
- **Impact and consequences**: "The policy will affect X", "Y resulted in Z"


## KEY RULES:

### 1. ATOMICITY
- Each atomic fact must contain exactly ONE piece of information or relationship
- Break down compound sentences into single-atomic facts statements
Example: "Company X announced a product and hired 50 people" → Two facts:
  - "Company X announced a new product"
  - "Company X hired 50 people"

### 2. DECONTEXTUALIZATION
- Replace ALL pronouns (he, she, it, they) with full entity names
- Example: "John joined the company. He started on Monday" → "John joined the company on Monday"

### 3. TEMPORAL NORMALIZATION
Convert ALL relative time references to absolute dates:

REFERENCE EXAMPLES FOR {observation_date}:
- "today" → on {observation_date}
- "yesterday" → on {yesterday}
- "3 days ago" → on {three_days_ago}
- "this week" (Monday) → on {week_monday}
- "last week" (Monday) → on {last_week_monday}
- "this month" (1st) → on {month_first}
- "last month" (1st) → on {last_month_first}
- "this year" (Jan 1st) → on {year_first}
- "last year" (Jan 1st) → on {last_year_first}
- Named days ("Monday", "Thursday") → on YYYY-MM-DD (most recent occurrence)
- Keep explicit dates unchanged (e.g., "June 18, 2024" stays "June 18, 2024")

**IMPORTANT**: Replace relative terms COMPLETELY. Final facts should have "on YYYY-MM-DD", never "on YYYY-MM-DD (yesterday)" or "YYYY-MM-DD (relative reference)"

### 4. END ACTIONS
- Explicitly capture role/action termination with timestamp
- Example: "CEO resigned yesterday" → "CEO resigned on {yesterday}"

### 5. AVOIDING IRRELEVANT INFORMATION
- INCLUDE: Facts about companies, people, events, policies, impacts
- EXCLUDE: Purely descriptive stylistic details that do not add substantive information about entities/events

If a description is about a specific named entity and provides information (e.g., "Company XYZ produces cars"), INCLUDE it.
If a description is generic and non-informative (e.g., "The company XYZ is focused on innovation, excellence, and building a better future through technology"), EXCLUDE it.

### 6. DEDUPLICATION GUIDANCE
- Prefer facts mentioning SPECIFIC NAMES/ENTITIES over generic versions
- GOOD: "Technology company ABC announced a new research facility in Silicon Valley on March 15, 2024"
- AVOID DUPLICATE: "The company expanded its operations in California" (already covered by the GOOD fact)
- Extract the more comprehensive fact with entity names rather than generic versions


## Example of EXHAUSTIVE Extraction

**Example 1:**
"Observation date: 2020-05-15 TOKYO — Tech company XYZ announced a major breakthrough in renewable energy yesterday. The innovation could reduce power consumption by 40%. Industry experts believe this represents a significant step forward for sustainable technology."

**Output (ALL facts):**
- Tech company XYZ announced a major breakthrough in renewable energy on 2020-05-14
- The renewable energy breakthrough could reduce power consumption by 40%
- Industry experts believe the announcement represents a significant step forward for sustainable technology

**Example 2:**
"Observation date: 2020-03-20 The European Union implemented new data protection regulations on Thursday. Companies have 60 days to comply with the rules. Financial penalties for non-compliance could reach up to 10 million euros."

**Output:**
- The European Union implemented new data protection regulations on 2020-03-19
- Companies have 60 days to comply with the new regulations (deadline: around 2020-05-18)
- Financial penalties for non-compliance with the regulations could reach up to 10 million euros

**Example 3:**
"Observation date: 2020-08-10 MANILA — A typhoon struck the region last week, causing flooding in three provinces. The disaster displaced thousands of residents. Emergency services continue relief operations."

**Output:**
- A typhoon struck the region around 2020-08-03
- The typhoon caused flooding in three provinces
- Thousands of residents were displaced by the typhoon
- Emergency services continue relief operations following the typhoon

## Input Format
A string made with this format: "Observation date: YYYY-MM-DD. [raw news paragraph text]"

## Output Format
Return ONLY the list of atomic facts. Each fact should be:
- Concise and factual
- Temporally explicit (include normalized dates where relevant)
- Decontextualized (no pronouns)
- Unique (no near-duplicates)
- Substantively informative (not generic marketing language)

MOST IMPORTANT: Extract EXHAUSTIVELY but CLEANLY. Skip noisy/meta-information and generic descriptions, but capture ALL substantive facts about events, entities, and relationships.
"""

    @staticmethod
    def _create_temporal_system_query(observation_date: str) -> str:
        """
        Create a comprehensive system query for exhaustive atomic facts extraction.
        Uses the AtomicFact schema description as the foundation and adds:
        - Emphasis on EXHAUSTIVE extraction (all facts, including supporting details)
        - Better date conversion examples with explicit mappings
        - Explicit examples of what NOT to extract (meta-information, noise, irrelevant details)
        - Clear prohibition on extracting "The observation date is" facts
        - Instructions to prevent duplicate/near-duplicate facts
        - Strict guidance on filtering generic descriptions vs. substantive information
        
        Note: This query is designed for the LLM at runtime. The AtomicFact schema 
        description serves a complementary role for JSON structure definition.
        
        Args:
            observation_date: The observation date in format YYYY-MM-DD
            
        Returns:
            System query string with comprehensive temporal context
        """
        obs_date = datetime.strptime(observation_date, '%Y-%m-%d')
        
        # Calculate example dates for the specific observation date
        yesterday = (obs_date - timedelta(days=1)).strftime('%Y-%m-%d')
        three_days_ago = (obs_date - timedelta(days=3)).strftime('%Y-%m-%d')
        week_monday = (obs_date - timedelta(days=obs_date.weekday())).strftime('%Y-%m-%d')
        last_week_monday = (obs_date - timedelta(days=obs_date.weekday() + 7)).strftime('%Y-%m-%d')
        month_first = obs_date.replace(day=1).strftime('%Y-%m-%d')
        year_first = obs_date.replace(month=1, day=1).strftime('%Y-%m-%d')
        last_month_first = (obs_date.replace(day=1) - timedelta(days=1)).replace(day=1).strftime('%Y-%m-%d')
        last_year_first = obs_date.replace(year=obs_date.year - 1, month=1, day=1).strftime('%Y-%m-%d')
        
        return f"""
You are an expert information extraction assistant. Your task is to extract factual, atomic statements from the provided text.

# EXTRACTION GOALS:
Extract statements about:
- Main events and actions (e.g., "Company X announced Y")
- Background context and causal relationships
- Quantitative data and impacts

# FORMATTING RULES:
1. ATOMICITY: Each fact must contain exactly ONE piece of information. Break compound sentences apart.
2. NO PRONOUNS: Replace all pronouns (he, she, it, they) with the specific entity names.
3. TEMPORAL NORMALIZATION: You MUST replace relative dates with absolute dates based on the observation date ({observation_date}).

REFERENCE EXAMPLES FOR {observation_date}:
- "today" → on {observation_date}
- "yesterday" → on {yesterday}
- "last week" (Monday) → on {last_week_monday}
- "this year" (Jan 1st) → on {year_first}

# WHAT TO IGNORE:
Do not extract meta-commentary, introductory phrases (like "While drinking coffee"), or generic corporate marketing speak. Only extract substantive facts.

Return the extracted information strictly adhering to the requested format.
"""
