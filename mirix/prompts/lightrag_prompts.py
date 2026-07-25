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


