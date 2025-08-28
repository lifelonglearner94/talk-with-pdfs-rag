from __future__ import annotations
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from .prompting import get_prompt
from .config import RAGConfig
from .logging import logger

class AnswerGenerator:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.llm = ChatGoogleGenerativeAI(model=config.llm_model, temperature=0.1)
        # Use JSON prompt version if answer_mode requests structured output
        prompt_version = config.prompt_version
        if config.answer_mode == "json" and prompt_version != "v3_json":
            prompt_version = "v3_json"
        self.prompt = get_prompt(prompt_version)
        try:
            template_preview = getattr(self.prompt, "template", str(self.prompt))
        except Exception:
            template_preview = str(self.prompt)
        logger.info("AnswerGenerator init prompt_version=%s answer_mode=%s prompt_preview=%s", prompt_version, config.answer_mode, (template_preview[:200] if isinstance(template_preview, str) else str(type(template_preview))))
        # Simple enhancer prompt: rewrite/expand a user query to improve
        # retrieval for a RAG system. Kept deliberately short and deterministic.
        self.enhancer_prompt = PromptTemplate.from_template(
            """Sie sind ein prägnanter Abfrageverbesserer für die Dokumentensuche. Formulieren Sie die Frage des Benutzers
konkreter und fügen Sie relevante Schlüsselwörter hinzu! Geben Sie nur die verbesserte Abfrage zurück.

Benutzerfrage: {question}
"""
        )

    def enhance_query(self, question: str) -> str:
        """Return an enhanced query produced by the LLM. On failure, return the original question."""
        try:
            # Build a tiny runnable chain that maps the raw string into the prompt
            logger.info("enhance_query: invoking enhancer for question=%s", question[:120])
            chain = ( {"question": RunnablePassthrough()} | self.enhancer_prompt | self.llm | StrOutputParser() )
            enhanced = chain.invoke(question)
            if not enhanced:
                return question
            # Strip whitespace/newlines
            enhanced_str = enhanced.strip()
            # Log the enhanced query so it appears in runtime logs for debugging/telemetry
            try:
                logger.info("enhance_query: enhanced_query=%s", enhanced_str[:200])
            except Exception:
                # ensure logging never breaks the response path
                logger.info("enhance_query: failed to log enhanced query")
            return enhanced_str
        except Exception:
            # Best-effort: if enhancer fails, fall back to original question
            logger.exception("enhance_query failed, returning original question")
            return question

    def build_chain(self, retriever):
        def format_context(docs):
            parts = []
            use_author_year = self.config.prompt_version == "v2"
            for i, d in enumerate(docs, 1):
                src = d.metadata.get('source_name') or d.metadata.get('source') or 'UnbekannteQuelle'
                if use_author_year:
                    # Expect filenames like 'Chopra, 2022, A Review Paper on Virtualization'
                    author = src.split(',')[0].strip()
                    year = None
                    # naive year find
                    for token in src.replace('-', ' ').split():
                        if token.isdigit() and len(token) == 4:
                            year = token
                            break
                    label = f"{author} {year}" if year else author
                    parts.append(f"[{label}]\n{d.page_content}")
                else:
                    parts.append(f"[Quelle {i}: {src}]\n{d.page_content}")
            return "\n\n".join(parts)
        logger.info("build_chain using prompt_version=%s prompt_template_preview=%s", self.config.prompt_version, getattr(self.prompt, "template", str(self.prompt))[:200])
        chain = (
            {"context": lambda q: format_context(retriever.invoke(q)), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain
