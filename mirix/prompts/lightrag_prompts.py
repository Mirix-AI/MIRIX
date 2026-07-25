"""
LightRAG prompt templates, adapted for MIRIX graph memory.

Source: https://github.com/HKUDS/LightRAG (MIT License) — see prompt.py.
The structure (delimiters, system/user split, examples format) is preserved
verbatim so output parsers can be shared. Entity types are tuned for
MIRIX's conversational corpus (added "Date", "Quantity"; dropped
"NaturalObject" / "Artifact" which rarely appear in chat).
"""

# Delimiters used inside extracted tuples. Must match the parser in
# lightrag_extractor._parse_extraction_output.
TUPLE_DELIMITER = "<|#|>"
COMPLETION_DELIMITER = "<|COMPLETE|>"

# Default entity types — tuned for personal-assistant conversation memory.
DEFAULT_ENTITY_TYPES = [
    "Person",
    "Organization",
    "Location",
    "Event",
    "Concept",
    "Method",
    "Content",
    "Date",
    "Quantity",
    "Other",
]


ENTITY_EXTRACTION_SYSTEM_PROMPT = """---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships from the input text.

---Instructions---
1.  **Entity Extraction & Output:**
    *   **Identification:** Identify clearly defined and meaningful entities in the input text.
    *   **Entity Details:** For each identified entity, extract the following information:
        *   `entity_name`: The name of the entity. If the entity name is case-insensitive, capitalize the first letter of each significant word (title case). Ensure **consistent naming** across the entire extraction process.
        *   `entity_type`: Categorize the entity using one of the following types: `{entity_types}`. If none of the provided entity types apply, do not add new entity type and classify it as `Other`.
        *   `entity_description`: Provide a concise yet comprehensive description of the entity's attributes and activities, based *solely* on the information present in the input text.
    *   **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
        *   Format: `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`

2.  **Relationship Extraction & Output:**
    *   **Identification:** Identify direct, clearly stated, and meaningful relationships between previously extracted entities.
    *   **N-ary Relationship Decomposition:** If a single statement describes a relationship involving more than two entities (an N-ary relationship), decompose it into multiple binary (two-entity) relationship pairs for separate description.
        *   **Example:** For "Alice, Bob, and Carol collaborated on Project X," extract binary relationships such as "Alice collaborated with Project X," "Bob collaborated with Project X," and "Carol collaborated with Project X," or "Alice collaborated with Bob," based on the most reasonable binary interpretations.
    *   **Relationship Details:** For each binary relationship, extract the following fields:
        *   `source_entity`: The name of the source entity. Ensure **consistent naming** with entity extraction. Capitalize the first letter of each significant word (title case) if the name is case-insensitive.
        *   `target_entity`: The name of the target entity. Ensure **consistent naming** with entity extraction. Capitalize the first letter of each significant word (title case) if the name is case-insensitive.
        *   `relationship_keywords`: One or more high-level keywords summarizing the overarching nature, concepts, or themes of the relationship. Multiple keywords within this field must be separated by a comma `,`. **DO NOT use `{tuple_delimiter}` for separating multiple keywords within this field.**
        *   `relationship_description`: A concise explanation of the nature of the relationship between the source and target entities, providing a clear rationale for their connection.
        *   `relationship_strength`: A floating point value between 0.0 and 1.0 estimating how strong/important this relationship is.
    *   **Output Format - Relationships:** Output a total of 6 fields for each relationship, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `relation`.
        *   Format: `relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description{tuple_delimiter}relationship_strength`

3.  **Delimiter Usage Protocol:**
    *   The `{tuple_delimiter}` is a complete, atomic marker and **must not be filled with content**. It serves strictly as a field separator.
    *   **Incorrect Example:** `entity{tuple_delimiter}Tokyo<|location|>Tokyo is the capital of Japan.`
    *   **Correct Example:** `entity{tuple_delimiter}Tokyo{tuple_delimiter}Location{tuple_delimiter}Tokyo is the capital of Japan.`

4.  **Relationship Direction & Duplication:**
    *   Treat all relationships as **undirected** unless explicitly stated otherwise. Swapping the source and target entities for an undirected relationship does not constitute a new relationship.
    *   Avoid outputting duplicate relationships.

5.  **Output Order & Prioritization:**
    *   Output all extracted entities first, followed by all extracted relationships.
    *   Within the list of relationships, prioritize and output those relationships that are **most significant** to the core meaning of the input text first.

6.  **Context & Objectivity:**
    *   Ensure all entity names and descriptions are written in the **third person**.
    *   Explicitly name the subject or object; **avoid using pronouns** such as `this article`, `this paper`, `our company`, `I`, `you`, and `he/she`.

7.  **Language & Proper Nouns:**
    *   The entire output (entity names, keywords, and descriptions) must be written in `{language}`.
    *   Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

8.  **Completion Signal:** Output the literal string `{completion_delimiter}` only after all entities and relationships, following all criteria, have been completely extracted and outputted.

---Examples---
{examples}
"""


ENTITY_EXTRACTION_USER_PROMPT = """---Task---
Extract entities and relationships from the input text in Data to be Processed below.

---Instructions---
1.  **Strict Adherence to Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system prompt.
2.  **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks, explanations, or additional text before or after the list.
3.  **Completion Signal:** Output `{completion_delimiter}` as the final line after all relevant entities and relationships have been extracted and presented.
4.  **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

---Data to be Processed---
<Entity_types>
[{entity_types}]

<Input Text>
```
{input_text}
```

<Output>
"""


# A single conversational example to keep the system prompt small. Adding more
# examples helps consistency but inflates the prompt cost on every chunk.
ENTITY_EXTRACTION_EXAMPLES = [
    """<Entity_types>
["Person","Organization","Location","Event","Concept","Method","Content","Date","Quantity","Other"]

<Input Text>
```
Caroline mentioned that her cousin Melanie just moved to Berlin to start a job at SAP last month. They used to live together in Munich while Caroline was finishing her PhD on quantum optics.
```

<Output>
entity{tuple_delimiter}Caroline{tuple_delimiter}Person{tuple_delimiter}Caroline is the speaker; she previously lived in Munich while pursuing a PhD on quantum optics.
entity{tuple_delimiter}Melanie{tuple_delimiter}Person{tuple_delimiter}Melanie is Caroline's cousin who recently moved to Berlin to start a job at SAP.
entity{tuple_delimiter}Berlin{tuple_delimiter}Location{tuple_delimiter}Berlin is the city Melanie moved to for her new job at SAP.
entity{tuple_delimiter}Munich{tuple_delimiter}Location{tuple_delimiter}Munich is the city where Caroline and Melanie used to live together while Caroline was a PhD student.
entity{tuple_delimiter}SAP{tuple_delimiter}Organization{tuple_delimiter}SAP is the organization where Melanie recently started working.
entity{tuple_delimiter}Quantum Optics{tuple_delimiter}Concept{tuple_delimiter}Quantum optics is the subject of Caroline's PhD research.
relation{tuple_delimiter}Caroline{tuple_delimiter}Melanie{tuple_delimiter}family relation, cohabitation{tuple_delimiter}Caroline and Melanie are cousins who previously lived together in Munich.{tuple_delimiter}0.9
relation{tuple_delimiter}Melanie{tuple_delimiter}Berlin{tuple_delimiter}relocation, residence{tuple_delimiter}Melanie recently moved to Berlin.{tuple_delimiter}0.8
relation{tuple_delimiter}Melanie{tuple_delimiter}SAP{tuple_delimiter}employment, new job{tuple_delimiter}Melanie started a job at SAP.{tuple_delimiter}0.85
relation{tuple_delimiter}Caroline{tuple_delimiter}Munich{tuple_delimiter}past residence, education{tuple_delimiter}Caroline lived in Munich while completing her PhD.{tuple_delimiter}0.7
relation{tuple_delimiter}Caroline{tuple_delimiter}Quantum Optics{tuple_delimiter}academic research, PhD topic{tuple_delimiter}Caroline pursued a PhD on quantum optics.{tuple_delimiter}0.8
{completion_delimiter}
""",
]


# Used by the description merge step in lightrag_merger when an entity or
# relation accumulates more than FORCE_LLM_SUMMARY_ON_MERGE descriptions.
SUMMARIZE_DESCRIPTIONS_PROMPT = """---Role---
You are a Knowledge Graph Specialist, proficient in data curation and synthesis.

---Task---
Your task is to synthesize a list of descriptions of a given {description_type} into a single, comprehensive, and cohesive summary.

---Instructions---
1. Comprehensiveness: The summary must integrate all key information from *every* provided description. Do not omit any important facts or details.
2. Context & Objectivity:
  - Write the summary from an objective, third-person perspective.
  - Explicitly mention the full name of the {description_type} at the beginning of the summary to ensure immediate clarity and context.
3. Conflict Handling:
  - In cases of conflicting or inconsistent descriptions, attempt to reconcile them or present both viewpoints with noted uncertainty.
4. Length Constraint: The summary's total length must not exceed {summary_length} tokens, while still maintaining depth and completeness.
5. Output: Plain text, no markdown fences, no preamble.

---Input---
{description_type} Name: {description_name}

Description List:
```
{description_list}
```

---Output---
"""


KEYWORDS_EXTRACTION_PROMPT = """---Role---
You are an expert keyword extractor, specializing in analyzing user queries for a Retrieval-Augmented Generation (RAG) system. Your purpose is to identify both high-level and low-level keywords in the user's query that will be used for effective document retrieval.

---Goal---
Given a user query, your task is to extract two distinct types of keywords:
1. **high_level_keywords**: for overarching concepts or themes, capturing user's core intent, the subject area, or the type of question being asked.
2. **low_level_keywords**: for specific entities or details, identifying the specific entities, proper nouns, technical jargon, product names, or concrete items.

---Instructions & Constraints---
1. **Output Format**: Your output MUST be a valid JSON object and nothing else. Do not include any explanatory text, markdown code fences (like ```json), or any other text before or after the JSON. It will be parsed directly by a JSON parser.
2. **Source of Truth**: All keywords must be explicitly derived from the user query, with both high-level and low-level keyword categories are required to contain content.
3. **Concise & Meaningful**: Keywords should be concise words or meaningful phrases. Prioritize multi-word phrases when they represent a single concept. For example, from "latest financial report of Apple Inc.", you should extract "latest financial report" and "Apple Inc." rather than "latest", "financial", "report", and "Apple".
4. **Handle Edge Cases**: For queries that are too simple, vague, or nonsensical (e.g., "hello", "ok", "asdfghjkl"), you must return a JSON object with empty lists for both keyword types.
5. **Language**: All extracted keywords MUST be in {language}. Proper nouns (e.g., personal names, place names, organization names) should be kept in their original language.

---Examples---
Example 1:
Query: "How does international trade influence global economic stability?"
Output:
{{"high_level_keywords": ["International trade", "Global economic stability", "Economic impact"], "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]}}

Example 2:
Query: "Where did Caroline live during her PhD?"
Output:
{{"high_level_keywords": ["Past residence", "Academic life"], "low_level_keywords": ["Caroline", "PhD", "Munich"]}}

Example 3:
Query: "What is the role of education in reducing poverty?"
Output:
{{"high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"], "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]}}

---Real Data---
User Query: {query}

---Output---
"""


def render_keywords_extraction_prompt(query: str, language: str = "English") -> str:
    return KEYWORDS_EXTRACTION_PROMPT.format(query=query, language=language)


def render_extraction_system_prompt(
    entity_types: list[str] | None = None,
    language: str = "English",
) -> str:
    """Render the system prompt with entity types and example bodies inlined."""
    types = entity_types or DEFAULT_ENTITY_TYPES
    types_str = ", ".join(types)
    example_ctx = {
        "tuple_delimiter": TUPLE_DELIMITER,
        "completion_delimiter": COMPLETION_DELIMITER,
    }
    examples = "\n".join(ex.format(**example_ctx) for ex in ENTITY_EXTRACTION_EXAMPLES)
    return ENTITY_EXTRACTION_SYSTEM_PROMPT.format(
        entity_types=types_str,
        tuple_delimiter=TUPLE_DELIMITER,
        completion_delimiter=COMPLETION_DELIMITER,
        language=language,
        examples=examples,
    )


def render_extraction_user_prompt(
    input_text: str,
    entity_types: list[str] | None = None,
    language: str = "English",
) -> str:
    types = entity_types or DEFAULT_ENTITY_TYPES
    return ENTITY_EXTRACTION_USER_PROMPT.format(
        entity_types=", ".join(types),
        completion_delimiter=COMPLETION_DELIMITER,
        language=language,
        input_text=input_text,
    )


def render_summarize_descriptions_prompt(
    description_type: str,
    description_name: str,
    description_list: list[str],
    summary_length: int = 500,
) -> str:
    return SUMMARIZE_DESCRIPTIONS_PROMPT.format(
        description_type=description_type,
        description_name=description_name,
        description_list="\n".join(f"- {d}" for d in description_list),
        summary_length=summary_length,
    )
