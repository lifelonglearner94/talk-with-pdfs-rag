import os
import hashlib
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

load_dotenv()

class ScientificPDFSearcher:
    def __init__(self, data_dir: str = "data", persist_dir: str = "vectorstore"):
        self.data_dir = Path(data_dir)
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True)

        # Check for API key
        if os.getenv("GOOGLE_API_KEY") is None:
            raise ValueError("Fehler: GOOGLE_API_KEY nicht gefunden. Bitte in der .env Datei oder als Umgebungsvariable setzen.")

        # Initialize models
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

        # Initialize vector store
        self.vectorstore = None
        self.retriever = None

        # Create enhanced prompt for scientific articles
        self.prompt_template = PromptTemplate.from_template("""
            Du bist ein wissenschaftlicher Assistent, der dabei hilft, Informationen aus wissenschaftlichen Artikeln zu extrahieren.

            Kontext aus den wissenschaftlichen Artikeln:
            {context}

            Frage: {question}

            Anweisungen:
            1. Beantworte die Frage basierend auf den bereitgestellten wissenschaftlichen Artikeln
            2. Gib präzise und fundierte Antworten
            3. WICHTIG: Zitiere immer die Quelle(n) deiner Antworten mit jeweils dem Dokumentnamen bzw. Name des Autor, Jahr
            4. Falls die Antwort nicht in den Artikeln gefunden wird, sage das deutlich
            5. Verwende wissenschaftliche Terminologie angemessen
            6. Strukturiere deine Antwort klar und verständlich

            Antwort:
            """)

    def _get_data_hash(self) -> str:
        """Generate hash of PDF files to detect changes"""
        pdf_files = list(self.data_dir.glob("*.pdf"))
        if not pdf_files:
            return ""

        hasher = hashlib.md5()
        for pdf_file in sorted(pdf_files):
            hasher.update(f"{pdf_file.name}-{pdf_file.stat().st_mtime}".encode())
        return hasher.hexdigest()

    def _needs_reindexing(self) -> bool:
        """Check if we need to reindex the documents"""
        hash_file = self.persist_dir / "data_hash.txt"
        current_hash = self._get_data_hash()

        if not hash_file.exists():
            return True

        stored_hash = hash_file.read_text().strip()
        return stored_hash != current_hash

    def _save_data_hash(self):
        """Save current data hash"""
        hash_file = self.persist_dir / "data_hash.txt"
        hash_file.write_text(self._get_data_hash())

    def _enhance_documents_with_metadata(self, docs: List[Document]) -> List[Document]:
        """Enhance documents with better metadata for scientific articles"""
        enhanced_docs = []

        for doc in docs:
            # Extract filename without extension
            source_file = Path(doc.metadata.get('source', '')).stem

            # Clean up source name (remove Zone.Identifier files)
            if ':Zone.Identifier' in source_file:
                source_file = source_file.replace(':Zone.Identifier', '')

            # Enhanced metadata
            doc.metadata.update({
                'source_name': source_file,
                'document_type': 'scientific_article',
                'chunk_id': f"{source_file}_chunk_{len(enhanced_docs)}"
            })

            enhanced_docs.append(doc)

        return enhanced_docs

    def load_or_create_vectorstore(self):
        """Load existing vectorstore or create new one"""
        print("Überprüfe Vektordatenbank...")

        if self.persist_dir.exists() and not self._needs_reindexing():
            print("Lade bestehende Vektordatenbank...")
            self.vectorstore = Chroma(
                persist_directory=str(self.persist_dir),
                embedding_function=self.embeddings
            )
        else:
            print("Erstelle neue Vektordatenbank...")
            self._create_vectorstore()

        # Setup retriever with more relevant chunks
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}  # Get more relevant chunks
        )

    def _create_vectorstore(self):
        """Create new vectorstore from PDFs"""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Datenordner '{self.data_dir}' existiert nicht.")

        pdf_files = list(self.data_dir.glob("*.pdf"))
        # Filter out Zone.Identifier files
        pdf_files = [f for f in pdf_files if ':Zone.Identifier' not in f.name]

        if not pdf_files:
            raise FileNotFoundError("Keine PDF-Dateien im Datenordner gefunden.")

        print(f"Lade {len(pdf_files)} PDF-Dateien...")

        # Load documents
        loader = PyPDFDirectoryLoader(str(self.data_dir))
        docs = loader.load()

        if not docs:
            raise ValueError("Keine Dokumente konnten geladen werden.")

        print(f"{len(docs)} Seiten geladen. Erstelle Textchunks...")

        # Use better chunking strategy for scientific articles
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Larger chunks for better context
            chunk_overlap=200,  # More overlap to preserve context
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )

        chunks = text_splitter.split_documents(docs)
        chunks = self._enhance_documents_with_metadata(chunks)

        print(f"Erstelle Embeddings für {len(chunks)} Textchunks...")

        # Create vectorstore
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(self.persist_dir)
        )

        # Save hash
        self._save_data_hash()
        print("Vektordatenbank erstellt und gespeichert!")

    def format_sources(self, docs: List[Document]) -> str:
        """Format source information from retrieved documents"""
        sources = set()
        for doc in docs:
            source_name = doc.metadata.get('source_name', 'Unbekannte Quelle')
            sources.add(source_name)

        if sources:
            return f"\n\n**Quellen:**\n" + "\n".join([f"- {source}" for source in sorted(sources)])
        return ""

    def search(self, question: str) -> str:
        """Search for answer in the scientific articles"""
        if not self.retriever:
            raise ValueError("Vektordatenbank nicht initialisiert. Führe zuerst load_or_create_vectorstore() aus.")

        # Retrieve relevant documents
        relevant_docs = self.retriever.invoke(question)

        # Create chain with source formatting
        def format_context_with_sources(docs):
            context_parts = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source_name', 'Unbekannte Quelle')
                context_parts.append(f"[Quelle {i}: {source}]\n{doc.page_content}")
            return "\n\n".join(context_parts)

        chain = (
            {"context": lambda x: format_context_with_sources(self.retriever.invoke(x)),
             "question": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

        # Get answer
        answer = chain.invoke(question)

        # Add source information
        sources = self.format_sources(relevant_docs)

        return answer + sources

    def interactive_search(self):
        """Start interactive search session"""
        print("\n" + "="*60)
        print("🔬 Wissenschaftlicher PDF-Assistent")
        print("="*60)
        print("Stelle Fragen zu deinen wissenschaftlichen Artikeln!")
        print("Tippe 'quit' oder 'exit' zum Beenden.\n")

        while True:
            try:
                question = input("❓ Deine Frage: ").strip()

                if question.lower() in ['quit', 'exit', 'q']:
                    print("Auf Wiedersehen! 👋")
                    break

                if not question:
                    continue

                print("\n🔍 Suche nach relevanten Informationen...")
                answer = self.search(question)

                print(f"\n💡 **Antwort:**\n{answer}")
                print("\n" + "-"*60 + "\n")

            except KeyboardInterrupt:
                print("\n\nAuf Wiedersehen! 👋")
                break
            except Exception as e:
                print(f"❌ Fehler: {e}")

def main():
    try:
        # Initialize searcher
        searcher = ScientificPDFSearcher()

        # Load or create vectorstore
        searcher.load_or_create_vectorstore()

        # Start interactive session
        searcher.interactive_search()

    except Exception as e:
        print(f"❌ Fehler beim Initialisieren: {e}")

if __name__ == "__main__":
    main()
