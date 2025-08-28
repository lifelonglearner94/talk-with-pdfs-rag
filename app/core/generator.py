from __future__ import annotations
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from .prompting import get_prompt
from .config import RAGConfig

class AnswerGenerator:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.llm = ChatGoogleGenerativeAI(model=config.llm_model, temperature=0.1)
        # Use JSON prompt version if answer_mode requests structured output
        prompt_version = config.prompt_version
        if config.answer_mode == "json" and prompt_version != "v3_json":
            prompt_version = "v3_json"
        self.prompt = get_prompt(prompt_version)

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
        chain = (
            {"context": lambda q: format_context(retriever.invoke(q)), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain
