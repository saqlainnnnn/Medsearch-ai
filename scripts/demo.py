from pathlib import Path

from app.chunking import RecursiveChunker
from app.config.settings import settings
from app.embeddings import SentenceTransformerEmbedder
from app.graph.builder import build_graph
from app.ingestion.cleaner import MedicalTextCleaner
from app.ingestion.loader import PDFDocumentLoader
from app.ingestion.parser import MedicalDocumentParser
from app.llm.cerebras import CerebrasLLM
from app.retrieval.dense.faiss import FAISSDenseRetriever
from app.retrieval.hybrid.hybrid import HybridRetriever
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.reranker.cross_encoder import CrossEncoderReranker
from app.retrieval.sparse.bm25 import BM25Retriever
from app.schemas.retrieval import SearchQuery
from app.services.generation import GenerationService
from app.services.ingestion import IngestionService
from app.services.search import SearchService


PDF_PATH = Path("data/raw/Guiding-Principles.pdf")  # <- change this


def main() -> None:
    print("=" * 60)
    print("MedSearch AI Demo")
    print("=" * 60)

    print("\nLoading models...")

    loader = PDFDocumentLoader()
    cleaner = MedicalTextCleaner()
    parser = MedicalDocumentParser()

    chunker = RecursiveChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    embedder = SentenceTransformerEmbedder()

    ingestion = IngestionService(
        loader=loader,
        cleaner=cleaner,
        parser=parser,
        chunker=chunker,
        embedder=embedder,
    )

    print("Ingesting PDF...")
    embedded_chunks = ingestion.ingest(PDF_PATH)

    print(f"Indexed {len(embedded_chunks)} chunks.")

    dense = FAISSDenseRetriever()

    sparse = BM25Retriever()

    hybrid = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
    )

    retrieval = RetrievalPipeline(
        embedder=embedder,
        hybrid_retriever=hybrid,
    )

    print("Building retrieval indexes...")
    retrieval.build(embedded_chunks)

    reranker = CrossEncoderReranker()

    search = SearchService(
        retrieval_pipeline=retrieval,
        reranker=reranker,
    )

    llm = CerebrasLLM()

    generation = GenerationService(
        llm=llm,
    )

    graph = build_graph(
        search_service=search,
        generation_service=generation,
    )

    print("\nSystem ready.\n")

    while True:
        question = input("> ").strip()

        if not question:
            continue

        if question.lower() in {
            "exit",
            "quit",
            "q",
        }:
            break

        result = graph.invoke(
            {
                "query": SearchQuery(
                    query=question,
                    top_k=settings.final_top_k,
                ),
                "retrieval_result": None,
                "answer": None,
            }
        )

        print()
        print("=" * 80)
        print(result["answer"].answer)
        print("=" * 80)

        print("\nCitations")

        for citation in result["answer"].citations:
            print(
                f"- {citation.source_title}"
                f" | Page {citation.page_number}"
            )

        print()


if __name__ == "__main__":
    main()