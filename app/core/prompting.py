from __future__ import annotations
from langchain_core.prompts import PromptTemplate


AUTHOR_YEAR_CITATION_INSTRUCTIONS = (
    "System: Du bist ein wissenschaftlicher Assistent.\n"
    "Antworte ausschließlich auf Basis des bereitgestellten Kontexts. Falls die Antwort nicht im Kontext enthalten ist, antworte exakt: 'Ich kann es im Kontext nicht finden.'\n"
    "Die Quellen sind im Format [Autor Jahr] angegeben.\n"
    "Zitiere jeden Satz im Fließtext mit (Autor Jahr). Gib Zitate auch dann an, wenn mehrere Sätze dieselbe Quelle nutzen.\n"
    "Für einen Satz, der auf mehreren Quellen beruht, nutze das Format (Autor1 Jahr; Autor2 Jahr).\n"
    "Formatiere die Antwort in Absätzen und vermeide Aufzählungen, es sei denn, die Frage erfordert sie explizit.\n"
    "Sprache der Ausgabe: Deutsch."
)

def get_prompt(version: str = "v1") -> PromptTemplate:
    # Simple versioning scaffold
    if version == "v1":
        template = (
            "System: Du bist ein wissenschaftlicher Assistent.\n"\
            "Nutze ausschließlich den bereitgestellten Kontext. Antworte 'Ich kann es im Kontext nicht finden', wenn die Antwort fehlt.\n\n"\
            "Kontext:\n{context}\n\nFrage: {question}\n\nAntwort (zitiere Quellen mit (QuelleNamen)):"  # ursprüngliche einfache Variante
        )
    elif version == "v2":
        template = (
            AUTHOR_YEAR_CITATION_INSTRUCTIONS +
            "\nKontext:\n{context}\n\nFrage: {question}\n\nAntwort:"  # neue präzisere Variante
        )
    else:
        template = "Kontext:\n{context}\nFrage: {question}\nAntwort:"  # Fallback
    return PromptTemplate.from_template(template)
