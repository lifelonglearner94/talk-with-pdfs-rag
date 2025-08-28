import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from dataclasses import dataclass
import time

load_dotenv()

@dataclass
class RetrieverConfig:
    name: str
    search_type: str
    search_kwargs: Dict[str, Any]
    description: str

class RetrieverTester:
    def __init__(self, data_dir: str = "data", persist_dir: str = "vectorstore"):
        self.data_dir = Path(data_dir)
        self.persist_dir = Path(persist_dir)

        # Check for API key
        if os.getenv("GOOGLE_API_KEY") is None:
            raise ValueError("Fehler: GOOGLE_API_KEY nicht gefunden. Bitte in der .env Datei setzen.")

        # Initialize models
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

        # Load existing vectorstore
        self.vectorstore = Chroma(
            persist_directory=str(self.persist_dir),
            embedding_function=self.embeddings
        )

        # Create enhanced prompt for scientific articles
        self.prompt_template = PromptTemplate.from_template("""
            Du bist ein wissenschaftlicher Assistent, der dabei hilft, Informationen aus wissenschaftlichen Artikeln zu extrahieren.

            Kontext aus den wissenschaftlichen Artikeln:
            {context}

            Frage: {question}

            Anweisungen:
            1. Beantworte die Frage basierend auf den bereitgestellten wissenschaftlichen Artikeln
            2. Gib präzise und fundierte Antworten
            3. Zitiere immer die Quelle(n) deiner Antwort mit dem Dokumentnamen
            4. Falls die Antwort nicht in den Artikeln gefunden wird, sage das deutlich
            5. Verwende wissenschaftliche Terminologie angemessen
            6. Strukturiere deine Antwort klar und verständlich

            Antwort:
            """)

        # Define test configurations
        self.test_configs = [
            RetrieverConfig(
                name="Similarity Top-5",
                search_type="similarity",
                search_kwargs={"k": 5},
                description="Standard similarity search mit 5 relevantesten Chunks"
            ),
            RetrieverConfig(
                name="Similarity Top-10",
                search_type="similarity",
                search_kwargs={"k": 10},
                description="Similarity search mit mehr Kontext (10 Chunks)"
            ),
            RetrieverConfig(
                name="MMR Balanced",
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.5},
                description="Maximal Marginal Relevance - Balance zwischen Relevanz und Diversität"
            ),
            RetrieverConfig(
                name="MMR Diverse",
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 15, "lambda_mult": 0.3},
                description="MMR mit mehr Diversität (weniger Redundanz)"
            )
        ]

        # Define test questions
        self.test_questions = [
            "Customer Churn: Eine direkte Bedrohung für unser Wachstum und unsere Profitabilität",
            "Vom reaktiven zum proaktiven Kundenmanagement: Die Chance durch Daten",
            "Wie wir durch die Churn-Prognose nachhaltigen Unternehmenswert generieren",
        ]

    def format_context_with_sources(self, docs: List[Document]) -> str:
        """Format context with source information"""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source_name', 'Unbekannte Quelle')
            context_parts.append(f"[Quelle {i}: {source}]\n{doc.page_content}")
        return "\n\n".join(context_parts)

    def evaluate_response_quality(self, question: str, answer: str, retrieved_docs: List[Document]) -> Dict[str, Any]:
        """Evaluate the quality of a response"""
        evaluation = {}

        # 1. Source diversity (different papers referenced)
        sources = set()
        for doc in retrieved_docs:
            source_name = doc.metadata.get('source_name', 'Unknown')
            sources.add(source_name)

        evaluation['source_diversity'] = len(sources)
        evaluation['sources'] = list(sources)

        # 2. Answer length (proxy for completeness)
        evaluation['answer_length'] = len(answer.split())

        # 3. Check if answer contains citations
        evaluation['has_citations'] = any(keyword in answer.lower() for keyword in ['quelle', 'source', 'studie', 'artikel', 'paper'])

        # 4. Check for specific scientific terms related to churn
        churn_terms = ['churn', 'abwanderung', 'machine learning', 'algorithmus', 'modell', 'prediction', 'vorhersage']
        evaluation['relevant_terms_count'] = sum(1 for term in churn_terms if term in answer.lower())

        # 5. Calculate relevancy score based on retrieved document content
        total_content_length = sum(len(doc.page_content) for doc in retrieved_docs)
        evaluation['total_context_length'] = total_content_length

        return evaluation

    def test_single_config(self, config: RetrieverConfig, question: str) -> Tuple[str, Dict[str, Any], List[Document]]:
        """Test a single retriever configuration"""
        print(f"  Testing: {config.name}")

        # Create retriever with this configuration
        retriever = self.vectorstore.as_retriever(
            search_type=config.search_type,
            search_kwargs=config.search_kwargs
        )

        # Retrieve relevant documents
        start_time = time.time()
        relevant_docs = retriever.invoke(question)
        retrieval_time = time.time() - start_time

        # Create chain
        chain = (
            {"context": lambda x: self.format_context_with_sources(retriever.invoke(x)),
             "question": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

        # Get answer
        start_time = time.time()
        answer = chain.invoke(question)
        total_time = time.time() - start_time + retrieval_time

        # Evaluate response
        evaluation = self.evaluate_response_quality(question, answer, relevant_docs)
        evaluation['response_time'] = total_time
        evaluation['retrieval_time'] = retrieval_time

        return answer, evaluation, relevant_docs

    def run_comprehensive_test(self):
        """Run comprehensive test of all configurations"""
        print("🧪 Starte umfassenden Retriever-Test...")
        print("="*80)

        results = {}

        for question_idx, question in enumerate(self.test_questions, 1):
            print(f"\n📝 Frage {question_idx}: {question}")
            print("-" * 60)

            question_results = {}

            for config in self.test_configs:
                try:
                    answer, evaluation, docs = self.test_single_config(config, question)

                    question_results[config.name] = {
                        'config': config,
                        'answer': answer,
                        'evaluation': evaluation,
                        'retrieved_docs': docs
                    }

                    # Print brief summary
                    print(f"    ✓ {config.name}: {evaluation['source_diversity']} Quellen, "
                          f"{evaluation['answer_length']} Wörter, "
                          f"{evaluation['response_time']:.2f}s")

                except Exception as e:
                    print(f"    ❌ {config.name}: Fehler - {e}")
                    question_results[config.name] = {'error': str(e)}

            results[question] = question_results

        return results

    def analyze_results(self, results: Dict[str, Any]):
        """Analyze and recommend best configuration"""
        print("\n" + "="*80)
        print("📊 DETAILLIERTE ANALYSE DER ERGEBNISSE")
        print("="*80)

        # Aggregate scores for each configuration
        config_scores = {}

        for config in self.test_configs:
            config_name = config.name
            scores = {
                'avg_source_diversity': [],
                'avg_answer_length': [],
                'avg_response_time': [],
                'avg_relevant_terms': [],
                'citation_rate': []
            }

            for question, question_results in results.items():
                if config_name in question_results and 'evaluation' in question_results[config_name]:
                    eval_data = question_results[config_name]['evaluation']
                    scores['avg_source_diversity'].append(eval_data['source_diversity'])
                    scores['avg_answer_length'].append(eval_data['answer_length'])
                    scores['avg_response_time'].append(eval_data['response_time'])
                    scores['avg_relevant_terms'].append(eval_data['relevant_terms_count'])
                    scores['citation_rate'].append(1 if eval_data['has_citations'] else 0)

            # Calculate averages
            if scores['avg_source_diversity']:
                config_scores[config_name] = {
                    'config': config,
                    'avg_source_diversity': sum(scores['avg_source_diversity']) / len(scores['avg_source_diversity']),
                    'avg_answer_length': sum(scores['avg_answer_length']) / len(scores['avg_answer_length']),
                    'avg_response_time': sum(scores['avg_response_time']) / len(scores['avg_response_time']),
                    'avg_relevant_terms': sum(scores['avg_relevant_terms']) / len(scores['avg_relevant_terms']),
                    'citation_rate': sum(scores['citation_rate']) / len(scores['citation_rate'])
                }

        # Print detailed analysis
        print("\n📈 METRIKEN IM VERGLEICH:")
        print("-" * 60)

        for config_name, metrics in config_scores.items():
            config = metrics['config']
            print(f"\n🔧 {config_name}")
            print(f"   Beschreibung: {config.description}")
            print(f"   Durchschn. Quellen-Diversität: {metrics['avg_source_diversity']:.2f}")
            print(f"   Durchschn. Antwortlänge: {metrics['avg_answer_length']:.0f} Wörter")
            print(f"   Durchschn. Antwortzeit: {metrics['avg_response_time']:.2f}s")
            print(f"   Durchschn. relevante Begriffe: {metrics['avg_relevant_terms']:.1f}")
            print(f"   Zitationsrate: {metrics['citation_rate']:.0%}")

        # Calculate overall score
        print("\n🏆 BEWERTUNG UND EMPFEHLUNG:")
        print("-" * 60)

        best_config = None
        best_score = -1

        for config_name, metrics in config_scores.items():
            # Weighted scoring
            score = (
                metrics['avg_source_diversity'] * 0.25 +  # Diversität wichtig für wissenschaftliche Arbeit
                (metrics['avg_answer_length'] / 100) * 0.20 +  # Vollständigkeit
                (1 / metrics['avg_response_time']) * 0.15 +  # Geschwindigkeit (invertiert)
                metrics['avg_relevant_terms'] * 0.25 +  # Relevanz
                metrics['citation_rate'] * 0.15  # Zitationen
            )

            print(f"{config_name}: Gesamtscore {score:.3f}")

            if score > best_score:
                best_score = score
                best_config = config_name

        print(f"\n🥇 EMPFOHLENE KONFIGURATION: {best_config}")
        if best_config in config_scores:
            recommended_config = config_scores[best_config]['config']
            print(f"   Search Type: {recommended_config.search_type}")
            print(f"   Search Args: {recommended_config.search_kwargs}")
            print(f"   Begründung: {recommended_config.description}")

        return best_config, config_scores

    def save_results_to_file(self, results: Dict[str, Any], config_scores: Dict[str, Any], best_config: str):
        """Save all test results to a file"""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"retriever_test_results_{timestamp}.txt"

        with open(filename, 'w', encoding='utf-8') as f:
            f.write("🧪 RETRIEVER KONFIGURATION TEST ERGEBNISSE\n")
            f.write(f"📅 Zeitstempel: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

            # Test configurations used
            f.write("🔧 GETESTETE KONFIGURATIONEN:\n")
            f.write("-" * 40 + "\n")
            for config in self.test_configs:
                f.write(f"• {config.name}: {config.description}\n")
                f.write(f"  Search Type: {config.search_type}\n")
                f.write(f"  Search Args: {config.search_kwargs}\n\n")

            # Detailed results for each question
            f.write("\n📝 DETAILLIERTE ERGEBNISSE PRO FRAGE:\n")
            f.write("="*80 + "\n")

            for question_idx, (question, question_results) in enumerate(results.items(), 1):
                f.write(f"\n❓ Frage {question_idx}: {question}\n")
                f.write("-" * 60 + "\n")

                for config_name, result in question_results.items():
                    if 'error' in result:
                        f.write(f"\n❌ {config_name}: FEHLER - {result['error']}\n")
                    else:
                        f.write(f"\n✅ {config_name}:\n")
                        if 'evaluation' in result:
                            eval_data = result['evaluation']
                            f.write(f"   📊 Quellen-Diversität: {eval_data['source_diversity']}\n")
                            f.write(f"   📝 Antwortlänge: {eval_data['answer_length']} Wörter\n")
                            f.write(f"   ⏱️  Antwortzeit: {eval_data['response_time']:.2f}s\n")
                            f.write(f"   🔍 Relevante Begriffe: {eval_data['relevant_terms_count']}\n")
                            f.write(f"   📚 Hat Zitationen: {'Ja' if eval_data['has_citations'] else 'Nein'}\n")
                            f.write(f"   📄 Verwendete Quellen: {', '.join(eval_data['sources'])}\n")

                        if 'answer' in result:
                            f.write(f"\n💡 Antwort:\n{result['answer']}\n")

                        f.write("\n" + "-"*40 + "\n")

            # Summary analysis
            f.write("\n\n📊 ZUSAMMENFASSUNG UND METRIKEN:\n")
            f.write("="*80 + "\n")

            for config_name, metrics in config_scores.items():
                config = metrics['config']
                f.write(f"\n🔧 {config_name}:\n")
                f.write(f"   Beschreibung: {config.description}\n")
                f.write(f"   Durchschn. Quellen-Diversität: {metrics['avg_source_diversity']:.2f}\n")
                f.write(f"   Durchschn. Antwortlänge: {metrics['avg_answer_length']:.0f} Wörter\n")
                f.write(f"   Durchschn. Antwortzeit: {metrics['avg_response_time']:.2f}s\n")
                f.write(f"   Durchschn. relevante Begriffe: {metrics['avg_relevant_terms']:.1f}\n")
                f.write(f"   Zitationsrate: {metrics['citation_rate']:.0%}\n")

            # Recommendation
            f.write(f"\n🏆 EMPFOHLENE KONFIGURATION: {best_config}\n")
            f.write("-" * 40 + "\n")
            if best_config in config_scores:
                recommended_config = config_scores[best_config]['config']
                f.write(f"Search Type: {recommended_config.search_type}\n")
                f.write(f"Search Args: {recommended_config.search_kwargs}\n")
                f.write(f"Begründung: {recommended_config.description}\n")

            f.write(f"\n📁 Vollständige Ergebnisse gespeichert in: {filename}\n")

        return filename

    def show_sample_responses(self, results: Dict[str, Any], best_config: str):
        """Show sample responses from the best configuration"""
        print("\n" + "="*80)
        print(f"📋 BEISPIELANTWORTEN DER EMPFOHLENEN KONFIGURATION: {best_config}")
        print("="*80)

        for i, (question, question_results) in enumerate(results.items(), 1):
            if best_config in question_results and 'answer' in question_results[best_config]:
                print(f"\n❓ Frage {i}: {question}")
                print("-" * 40)
                answer = question_results[best_config]['answer']
                # Truncate long answers for display
                if len(answer) > 500:
                    answer = answer[:500] + "... [gekürzt]"
                print(f"💡 Antwort: {answer}")

def main():
    try:
        print("🚀 Initialisiere Retriever-Test...")
        tester = RetrieverTester()

        print(f"✅ Vectorstore geladen mit {tester.vectorstore._collection.count()} Dokumenten")

        # Run comprehensive test
        results = tester.run_comprehensive_test()

        # Analyze results and get recommendation
        best_config, config_scores = tester.analyze_results(results)

        # Save results to file
        results_file = tester.save_results_to_file(results, config_scores, best_config)
        print(f"\n💾 Alle Ergebnisse gespeichert in: {results_file}")

        # Show sample responses
        tester.show_sample_responses(results, best_config)

        print("\n" + "="*80)
        print("✅ TEST ABGESCHLOSSEN!")
        print(f"📁 Detaillierte Ergebnisse finden Sie in: {results_file}")
        print("="*80)

    except Exception as e:
        print(f"❌ Fehler beim Testen: {e}")

if __name__ == "__main__":
    main()
