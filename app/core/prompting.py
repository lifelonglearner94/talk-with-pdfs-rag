from __future__ import annotations
from langchain_core.prompts import PromptTemplate


AUTHOR_YEAR_CITATION_INSTRUCTIONS = (
    "System: Du bist ein wissenschaftlicher Assistent.\n"
    "Antworte ausschließlich auf Basis des bereitgestellten Kontexts. Falls die Antwort nicht im Kontext enthalten ist, antworte exakt: 'Ich kann es im Kontext nicht finden.'\n"
    "Die Quellen sind im Kontext im Format [Autor Jahr] markiert.\n"
    "Jeder Satz muss mindestens eine Quellenangabe im Format (Autor Jahr) enthalten. Bei mehreren Quellen: (Autor1 Jahr; Autor2 Jahr).\n"
    "Formatiere die Antwort in Absätzen und vermeide Aufzählungen, es sei denn, die Frage erfordert sie.\n"
    "Sprache der Ausgabe: Deutsch."
)

def get_prompt(version: str = "v1") -> PromptTemplate:
    # Simple versioning scaffold
    if version == "v1":
        template = (
            "System: Du bist ein wissenschaftlicher Assistent.\n"\
            "Nutze ausschließlich den bereitgestellten Kontext. Antworte 'Ich kann es im Kontext nicht finden', wenn die Antwort fehlt.\n\n"\
            "Kontext:\n{context}\n\n"\
            "Benutzerfrage (original): {original_question}\n"\
            "Frage (für Suche/Antwort): {question}\n\n"\
            "Antwort (zitiere Quellen mit (QuelleNamen)):"  # ursprüngliche einfache Variante
        )
    elif version == "v2":
        template = (
            AUTHOR_YEAR_CITATION_INSTRUCTIONS +
            "\nKontext:\n{context}\n\n"\
            "Benutzerfrage (original): {original_question}\n"\
            "Frage (für Suche/Antwort): {question}\n\n"\
            "Antwort:"  # neue präzisere Variante
        )
    elif version == "v3_json":
        # Escape outer JSON braces ({{ }}) so PromptTemplate doesn't treat them as
        # template variables. Inner quotes remain as-is.
        template = (
            AUTHOR_YEAR_CITATION_INSTRUCTIONS +
            "\nErzeuge eine JSON Antwort mit folgendem Schema ohne zusätzliche Erklärungen:\n"\
            "{{\n  \"direct_answer\": string,\n  \"supporting_facts\": [ {{ \"sentence\": string, \"citations\": [string] }} ],\n  \"citations\": [ {{ \"label\": string, \"snippet\": string }} ],\n  \"confidence\": number\n}}\n"\
            "Kontext:\n{context}\n\n"\
            "Benutzerfrage (original): {original_question}\n"\
            "Frage (für Suche/Antwort): {question}\n\n"\
            "JSON Antwort:"  # structured output
        )
    else:
        template = (
            "Kontext:\n{context}\n"\
            "Benutzerfrage (original): {original_question}\n"\
            "Frage (für Suche/Antwort): {question}\nAntwort:"
        )  # Fallback
    return PromptTemplate.from_template(template)
