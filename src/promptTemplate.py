
##identify medical terminologies prompt template
generation_prompt_template = """
Given the following multiple-choice query {query}, extract all relevant medical entities contained within the question stem.

Identify and extract all medical entities, such as diseases, proteins, genes, drugs, phenotypes, anatomical regions, treatments, or other relevant medical entities.

Ensure that the extracted entities are specific and medically relevant.

If no medical entities are found in a particular part, return an empty list for that section.

Only return the extracted entities in JSON format with the key "medical_terminologies" and the value is a list of extracted entities.

Query: {query}

"""

##identify triplets prompt template
triplet_prompt_template = """
Given the following query stem {query_stem}, medical terminologies {mt}, and options {option}, generate a set of related undirected triplets. Each triplet should consist of a head entity, a relation, and a tail entity. The relations should describe meaningful interactions or associations between the entities, particularly in a medical or biomedical context.

Use the query stem and the medical entities contained each option to extract triplets that are relevant to the query and can answer the query correctly.

Each triplet should be in the format: (Head Entity, Relationship, Tail Entity). Since the triplets are undirected, the order of Head Entity and Tail Entity does not imply any directional relationship between them.

The relationship should be one of the following: ['protein_protein', 'carrier', 'enzyme', 'target', 'transporter', 'contraindication', 'indication', 'off-label use', 'synergistic interaction', 'associated with', 'parent-child', 'phenotype absent', 'phenotype present', 'side effect', 'interacts with', 'linked to', 'expression present', 'expression absent'].

Ensure that each entity in the triplet is specific and concise, such as diseases, proteins, conditions, symptoms, drugs, treatments, anatomical parts, or other relevant medical entities.

Generate 1-3 triplets for each option, focusing on the ones most relevant to answering the query.

Only return the generated triplets in a structured JSON format with the key as "Triplets" and the value as a list of triplets. The format should be:
{
    "Triplets": [(Head Entity, Relationship, Tail Entity), (Head Entity, Relationship, Tail Entity)]
}

Question: {query_stem}

Medical_Terminologies: {mt}

Options: {option}

"""

##identify triplets prompt template
triplet_prompt_template_for_binary_or_maybe = """
Given the following query stem {query_stem}, and medical terminologies {mt}, generate a set of related undirected triplets. Each triplet should consist of a head entity, a relation, and a tail entity. The relations should describe meaningful interactions or associations between the entities, particularly in a medical or biomedical context.

Use the query stem and the content of each option to generate triplets that are relevant to the query and could answer the query correctly.

Each triplet should be in the format: (Head Entity, Relationship, Tail Entity). Since the triplets are undirected, the order of Head Entity and Tail Entity does not imply any directional relationship between them.

The relationship should be one of the following: ['protein_protein', 'carrier', 'enzyme', 'target', 'transporter', 'contraindication', 'indication', 'off-label use', 'synergistic interaction', 'associated with', 'parent-child', 'phenotype absent', 'phenotype present', 'side effect', 'interacts with', 'linked to', 'expression present', 'expression absent'].

Ensure that each entity in the triplet is specific and concise, such as diseases, proteins, conditions, symptoms, drugs, treatments, anatomical parts, or other relevant medical entities.

Generate 1-3 triplets that are most relevant to answering the query.

Only return the generated triplets in a structured JSON format with the key as "Triplets" and the value as a list of triplets. The format should be:
{
    "Triplets": [(Head Entity, Relationship, Tail Entity),(Head Entity, Relationship, Tail Entity)]
}

Question: {query_stem}

Medical_Terminologies: {mt}

"""

answer_generation_prompt_template = """
The following is a multiple-choice medical question and a list of triplets related to this question. Let's answer this question based on provided triplets and your own knowledge. If no triplets are provided, please answer this question based on your own knowledge.
Please directly select and provide the correct answer from options 'A', 'B, 'C' or 'D'. Only return the correct answer in a structured JSON format with the key as "Answer". The format should be:
{
    "Answer": 
}

Triplets: {t}

Question: {q}

"""


##modification module prompt template
modification_prompt_template = """
Given the following triplet consisting of a head entity, relation, and tail entity, please review and revise the triplet to ensure it is correct and helpful for answering given question. The revision should focus on correcting the head entity, relation, or tail entity as needed to make the triplet accurate and relevant.

The triplet should follow the format (head entity, relation, tail entity).

Ensure that the revised triplet is factually accurate and contextually appropriate.

The relation should clearly define the relationship between the head entity and the tail entity.

If no changes are necessary, return the original triplet.

Only return the revised triplet in JSON format with the key 'Revised_Triplets' and the value as the corrected triplet. The format should be:
{
    "Revised_Triplets": [(Head Entity, Relationship, Tail Entity)]
}

Triplets: {t}

Questions: {q}

"""