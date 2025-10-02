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
        logger.debug("AnswerGenerator init prompt_version=%s answer_mode=%s prompt_preview=%s", prompt_version, config.answer_mode, (template_preview[:120] + "…" if isinstance(template_preview, str) else str(type(template_preview))))
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
            logger.debug("enhance_query: invoking enhancer for question=%s", question[:120])
            chain = ( {"question": RunnablePassthrough()} | self.enhancer_prompt | self.llm | StrOutputParser() )
            enhanced = chain.invoke(question)
            if not enhanced:
                return question
            # Strip whitespace/newlines
            enhanced_str = enhanced.strip()
            # Log the enhanced query so it appears in runtime logs for debugging/telemetry
            try:
                logger.debug("enhance_query: enhanced_query=%s", enhanced_str[:160])
            except Exception:
                logger.debug("enhance_query: failed to log enhanced query")
            return enhanced_str
        except Exception:
            # Best-effort: if enhancer fails, fall back to original question
            logger.exception("enhance_query failed, returning original question")
            return question

    def build_chain(self, retriever, fixed_docs: list | None = None):
        """Build the LLM chain that formats context and answers.

        If fixed_docs is provided, those documents are used as the context verbatim
        (no further retrieval). Otherwise, the provided retriever is invoked with
        the enhanced question at generation time.
        """
        def format_context(docs):
            parts = []
            use_author_year = self.config.prompt_version in ("v2", "v3_json")
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
        logger.debug("build_chain using prompt_version=%s", self.config.prompt_version)
        # Expect input dict with both 'question' (enhanced) and 'original_question'
        # Retrieval should use the enhanced question for best recall, unless
        # fixed_docs is provided, in which case we format those directly.
        chain = (
            {
                "context": (lambda inputs: format_context(fixed_docs)) if fixed_docs is not None else (lambda inputs: format_context(retriever.invoke(inputs["question"]))),
                "question": lambda inputs: inputs["question"],
                "original_question": lambda inputs: inputs.get("original_question", inputs.get("question")),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain
