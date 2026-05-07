"""
rag/evaluator.py — Full RAG Evaluation Framework

Metrics: Recall@K, MRR, NDCG, Groundedness, Context Precision,
         Faithfulness, Citation Verification, Chunk Attribution

Backends: FAISS (dense vector search) + ChromaDB (managed)
Embeddings: sentence-transformers (local, free)
"""
from __future__ import annotations
import json, math, sqlite3, statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"


@dataclass
class RAGSample:
    question_id: str
    question: str
    relevant_doc_ids: list[str]      # ground truth
    context_chunks: list[str]        # retrieved by RAG system
    context_doc_ids: list[str]       # IDs of retrieved chunks
    answer: str                      # model answer
    ground_truth_answer: Optional[str] = None

@dataclass
class RAGEvalResult:
    question_id: str
    recall_at_k: dict[int, float]    # {1: 0.8, 3: 0.9, 5: 1.0}
    mrr: float
    ndcg: float
    groundedness: float
    context_precision: float
    faithfulness: float
    citation_accuracy: float
    hallucinated_spans: list[str]
    supporting_chunks: list[str]
    metadata: dict = field(default_factory=dict)

    def to_dict(self): return asdict(self)


# ─── Embedding Engine ─────────────────────────────────────────────────────────

class EmbeddingEngine:
    """Local embedding using sentence-transformers."""
    _instance = None
    _model = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
                print("[Embeddings] Loaded all-MiniLM-L6-v2")
            except Exception as e:
                print(f"[Embeddings] sentence-transformers unavailable ({type(e).__name__}), using TF-IDF fallback")
                self._model = "tfidf"
        if self._model == "tfidf":
            return self._tfidf_encode(texts)
        return self._model.encode(texts).tolist()

    def _tfidf_encode(self, texts: list[str]) -> list[list[float]]:
        import re as _re, math as _math
        dim = 384
        vocab: dict = {}
        tokenized = []
        for t in texts:
            tokens = _re.findall(r"\w+", t.lower())
            tokenized.append(tokens)
            for tok in tokens:
                if tok not in vocab and len(vocab) < dim:
                    vocab[tok] = len(vocab)
        df: dict = {}
        for tokens in tokenized:
            for tok in set(tokens):
                df[tok] = df.get(tok, 0) + 1
        n = max(len(texts), 1)
        result = []
        for tokens in tokenized:
            vec = [0.0] * dim
            tf: dict = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            for tok, cnt in tf.items():
                idx = vocab.get(tok)
                if idx is not None:
                    idf = _math.log(n / (df.get(tok, 1)))
                    vec[idx] = (cnt / max(len(tokens), 1)) * idf
            norm = _math.sqrt(sum(v*v for v in vec)) or 1.0
            result.append([v / norm for v in vec])
        return result

    def similarity(self, a: list[float], b: list[float]) -> float:
        dot = sum(x*y for x,y in zip(a,b))
        na = math.sqrt(sum(x**2 for x in a))
        nb = math.sqrt(sum(x**2 for x in b))
        return dot / (na * nb) if na * nb > 0 else 0.0


# ─── FAISS Vector Store ───────────────────────────────────────────────────────

class FAISSRetriever:
    """FAISS-backed dense retrieval for RAG evaluation."""
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.docs: list[dict] = []
        self._index = None
        self._embeddings: list[list[float]] = []
        self.embedder = EmbeddingEngine.get()

    def add_documents(self, docs: list[dict]):
        """docs: [{"id": "...", "text": "...", "metadata": {...}}]"""
        texts = [d["text"] for d in docs]
        embeddings = self.embedder.encode(texts)
        self.docs.extend(docs)
        self._embeddings.extend(embeddings)
        self._build_index()

    def _build_index(self):
        try:
            import faiss
            import numpy as np
            if not self._embeddings: return
            matrix = np.array(self._embeddings, dtype="float32")
            faiss.normalize_L2(matrix)
            self._index = faiss.IndexFlatIP(matrix.shape[1])
            self._index.add(matrix)
        except ImportError:
            pass  # Fall back to brute-force cosine

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """Retrieve top-k documents for query."""
        query_emb = self.embedder.encode([query])[0]
        if self._index is not None:
            try:
                import faiss, numpy as np
                qv = np.array([query_emb], dtype="float32")
                faiss.normalize_L2(qv)
                scores, indices = self._index.search(qv, min(k, len(self.docs)))
                return [{"doc": self.docs[i], "score": float(s)}
                        for s, i in zip(scores[0], indices[0]) if i < len(self.docs)]
            except Exception:
                pass
        # Brute-force fallback
        sims = [(self.embedder.similarity(query_emb, e), i)
                for i, e in enumerate(self._embeddings)]
        sims.sort(reverse=True)
        return [{"doc": self.docs[i], "score": s} for s, i in sims[:k]]


# ─── ChromaDB Vector Store ────────────────────────────────────────────────────

class ChromaRetriever:
    """ChromaDB-backed retrieval."""
    def __init__(self, collection_name: str = "llm_eval_rag"):
        self._client = None
        self._collection = None
        self.collection_name = collection_name
        self._local_docs: list[dict] = []
        self.embedder = EmbeddingEngine.get()
        self._init_chroma()

    def _init_chroma(self):
        try:
            import chromadb
            self._client = chromadb.Client()
            self._collection = self._client.get_or_create_collection(
                self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"[ChromaDB] Not available: {e}. Using FAISS fallback.")

    def add_documents(self, docs: list[dict]):
        self._local_docs.extend(docs)
        if self._collection:
            try:
                self._collection.add(
                    documents=[d["text"] for d in docs],
                    ids=[d["id"] for d in docs],
                    metadatas=[d.get("metadata", {}) for d in docs],
                )
            except Exception as e:
                print(f"[ChromaDB] Add failed: {e}")

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        if self._collection:
            try:
                results = self._collection.query(query_texts=[query], n_results=min(k, len(self._local_docs)))
                docs_out = []
                for i, doc_id in enumerate(results["ids"][0]):
                    docs_out.append({
                        "doc": {"id": doc_id, "text": results["documents"][0][i]},
                        "score": 1.0 - results["distances"][0][i] if results.get("distances") else 0.5
                    })
                return docs_out
            except Exception:
                pass
        # Fallback: embedding similarity
        if not self._local_docs: return []
        q_emb = self.embedder.encode([query])[0]
        doc_embs = self.embedder.encode([d["text"] for d in self._local_docs])
        sims = [(self.embedder.similarity(q_emb, e), i) for i, e in enumerate(doc_embs)]
        sims.sort(reverse=True)
        return [{"doc": self._local_docs[i], "score": s} for s, i in sims[:k]]


# ─── Retrieval Metrics ────────────────────────────────────────────────────────

def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    retrieved_top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    if not relevant: return 0.0
    return len(retrieved_top_k & relevant) / len(relevant)

def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    relevant = set(relevant_ids)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved_ids[:k], start=1)
        if doc_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


# ─── Groundedness & Faithfulness ──────────────────────────────────────────────

def compute_groundedness(answer: str, context_chunks: list[str]) -> tuple[float, list[str]]:
    """
    Rule-based groundedness: check what fraction of answer sentences
    can be grounded in at least one context chunk.
    """
    sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
    if not sentences: return 0.0, []
    context_text = " ".join(context_chunks).lower()
    grounded = []
    ungrounded = []
    for sent in sentences:
        words = [w.lower() for w in sent.split() if len(w) > 4]
        if not words:
            grounded.append(sent)
            continue
        hits = sum(1 for w in words if w in context_text)
        if hits / len(words) >= 0.5:
            grounded.append(sent)
        else:
            ungrounded.append(sent)
    score = len(grounded) / len(sentences)
    return round(score, 4), ungrounded

def compute_context_precision(context_chunks: list[str], relevant_chunks: list[str]) -> float:
    """Fraction of retrieved chunks that are actually relevant."""
    if not context_chunks: return 0.0
    relevant_set = set(relevant_chunks)
    hits = sum(1 for c in context_chunks if c in relevant_set)
    return round(hits / len(context_chunks), 4)

def verify_citations(answer: str, context_chunks: list[str]) -> tuple[float, list[str]]:
    """Check if claims in the answer have supporting context."""
    context_text = " ".join(context_chunks).lower()
    sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 15]
    if not sentences: return 0.0, []
    supported, unsupported = [], []
    for sent in sentences:
        key_terms = [w.lower() for w in sent.split() if len(w) > 5]
        if not key_terms: supported.append(sent); continue
        match_ratio = sum(1 for t in key_terms if t in context_text) / len(key_terms)
        (supported if match_ratio >= 0.4 else unsupported).append(sent)
    accuracy = len(supported) / len(sentences) if sentences else 0.0
    return round(accuracy, 4), unsupported


# ─── Full RAG Evaluator ───────────────────────────────────────────────────────

class RAGEvaluator:
    """
    End-to-end RAG evaluation pipeline.
    Usage:
        evaluator = RAGEvaluator(backend="faiss")
        evaluator.add_knowledge_base(documents)
        results = await evaluator.evaluate(samples, client)
    """
    def __init__(self, backend: str = "faiss"):
        if backend == "chroma":
            self.retriever = ChromaRetriever()
        else:
            self.retriever = FAISSRetriever()
        self.backend = backend

    def add_knowledge_base(self, documents: list[dict]):
        self.retriever.add_documents(documents)
        print(f"[RAG] Added {len(documents)} documents to {self.backend} index")

    def evaluate_retrieval(self, sample: RAGSample, k_values: list[int] = [1, 3, 5]) -> dict:
        """Evaluate retrieval quality for a single sample."""
        retrieved = self.retriever.retrieve(sample.question, k=max(k_values))
        retrieved_ids = [r["doc"]["id"] for r in retrieved]

        metrics = {}
        for k in k_values:
            metrics[f"recall@{k}"] = recall_at_k(retrieved_ids, sample.relevant_doc_ids, k)
        metrics["mrr"] = mean_reciprocal_rank(retrieved_ids, sample.relevant_doc_ids)
        metrics["ndcg@5"] = ndcg_at_k(retrieved_ids, sample.relevant_doc_ids, 5)
        metrics["retrieved_ids"] = retrieved_ids
        metrics["retrieved_scores"] = [r["score"] for r in retrieved]
        return metrics

    async def evaluate_generation(self, sample: RAGSample, client=None) -> dict:
        """Evaluate generation quality given retrieved context."""
        groundedness, ungrounded = compute_groundedness(sample.answer, sample.context_chunks)
        context_prec = compute_context_precision(sample.context_chunks, sample.relevant_doc_ids)
        citation_acc, unsupported = verify_citations(sample.answer, sample.context_chunks)
        faithfulness_score = (groundedness + citation_acc) / 2

        return {
            "groundedness": groundedness,
            "context_precision": context_prec,
            "citation_accuracy": citation_acc,
            "faithfulness": faithfulness_score,
            "hallucinated_spans": ungrounded[:3],
            "unsupported_claims": unsupported[:3],
        }

    async def evaluate(self, samples: list[RAGSample], client=None,
                       k_values: list[int] = [1, 3, 5]) -> dict:
        all_results = []
        retrieval_metrics = {f"recall@{k}": [] for k in k_values}
        retrieval_metrics.update({"mrr": [], "ndcg@5": [], "groundedness": [],
                                  "context_precision": [], "faithfulness": []})

        for sample in samples:
            ret = self.evaluate_retrieval(sample, k_values)
            gen = await self.evaluate_generation(sample, client)
            result = RAGEvalResult(
                question_id=sample.question_id,
                recall_at_k={k: ret.get(f"recall@{k}", 0) for k in k_values},
                mrr=ret["mrr"], ndcg=ret["ndcg@5"],
                groundedness=gen["groundedness"], context_precision=gen["context_precision"],
                faithfulness=gen["faithfulness"], citation_accuracy=gen["citation_accuracy"],
                hallucinated_spans=gen["hallucinated_spans"],
                supporting_chunks=sample.context_chunks[:2],
            )
            all_results.append(result)
            for k in k_values:
                retrieval_metrics[f"recall@{k}"].append(ret.get(f"recall@{k}", 0))
            retrieval_metrics["mrr"].append(ret["mrr"])
            retrieval_metrics["ndcg@5"].append(ret["ndcg@5"])
            retrieval_metrics["groundedness"].append(gen["groundedness"])
            retrieval_metrics["context_precision"].append(gen["context_precision"])
            retrieval_metrics["faithfulness"].append(gen["faithfulness"])

        summary = {
            "backend": self.backend,
            "n_samples": len(samples),
            "timestamp": datetime.utcnow().isoformat(),
            "aggregate": {
                metric: round(statistics.mean(vals), 4)
                for metric, vals in retrieval_metrics.items() if vals
            },
            "results": [r.to_dict() for r in all_results]
        }

        # Save
        DATA_DIR.mkdir(exist_ok=True)
        out = DATA_DIR / f"rag_eval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[RAG] Evaluation complete. Results → {out}")
        return summary


# ─── Demo knowledge base ──────────────────────────────────────────────────────

DEMO_KNOWLEDGE_BASE = [
    {"id":"doc_001","text":"Python is a high-level programming language created by Guido van Rossum in 1991. It emphasizes code readability.","metadata":{"topic":"programming"}},
    {"id":"doc_002","text":"Machine learning is a subset of artificial intelligence that enables systems to learn from data without being explicitly programmed.","metadata":{"topic":"AI"}},
    {"id":"doc_003","text":"The Eiffel Tower was built in 1889 by Gustave Eiffel for the 1889 World's Fair in Paris.","metadata":{"topic":"history"}},
    {"id":"doc_004","text":"Photosynthesis is the process by which plants use sunlight, water and carbon dioxide to produce oxygen and energy in the form of glucose.","metadata":{"topic":"biology"}},
    {"id":"doc_005","text":"The speed of light in a vacuum is approximately 299,792 kilometres per second, denoted by c.","metadata":{"topic":"physics"}},
    {"id":"doc_006","text":"Neural networks are computational models inspired by the human brain, consisting of layers of interconnected nodes.","metadata":{"topic":"AI"}},
    {"id":"doc_007","text":"The water cycle describes how water evaporates, forms clouds, and falls back as precipitation.","metadata":{"topic":"science"}},
]

DEMO_SAMPLES = [
    RAGSample(question_id="q001", question="Who created Python?",
              relevant_doc_ids=["doc_001"], context_chunks=[DEMO_KNOWLEDGE_BASE[0]["text"]],
              context_doc_ids=["doc_001"],
              answer="Python was created by Guido van Rossum in 1991.",
              ground_truth_answer="Guido van Rossum"),
    RAGSample(question_id="q002", question="What is machine learning?",
              relevant_doc_ids=["doc_002", "doc_006"], context_chunks=[DEMO_KNOWLEDGE_BASE[1]["text"], DEMO_KNOWLEDGE_BASE[5]["text"]],
              context_doc_ids=["doc_002", "doc_006"],
              answer="Machine learning enables systems to learn from data. It uses neural networks as computational models.",
              ground_truth_answer="A subset of AI that learns from data"),
    RAGSample(question_id="q003", question="What is the speed of light?",
              relevant_doc_ids=["doc_005"], context_chunks=[DEMO_KNOWLEDGE_BASE[4]["text"]],
              context_doc_ids=["doc_005"],
              answer="The speed of light is approximately 299,792 km/s.",
              ground_truth_answer="299,792 kilometres per second"),
]
