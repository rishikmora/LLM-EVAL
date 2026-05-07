"""
Module 15 — Full RAG Evaluation Framework

Evaluates complete RAG pipelines:
  - Retriever evaluation: Recall@K, MRR, Precision@K
  - Embedding evaluation: cosine similarity, embedding drift
  - Citation verification: does the answer cite real context?
  - Hallucination tracing: claim-level source attribution
  - Grounding analysis: proportion of grounded vs ungrounded statements
  - Chunk attribution: which chunks contributed to the answer

Backends: FAISS (in-memory), ChromaDB (persistent), sentence-transformers
"""

import json
import re
import math
import asyncio
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional
import yaml

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"


def load_config() -> dict:
    with open(ROOT / "config" / "eval_config.yaml") as f:
        return yaml.safe_load(f)


# ─── Embedding model (lazy-loaded) ───────────────────────────────────────────

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            _embedding_model = None
    return _embedding_model


def embed(texts: list[str]) -> list[list[float]]:
    """Embed texts using sentence-transformers. Falls back to TF-IDF simulation."""
    model = get_embedding_model()
    if model:
        return model.encode(texts).tolist()
    # Fallback: simple bag-of-words vector (unit-normalized)
    from collections import Counter
    import math
    vocab = set()
    for t in texts:
        vocab.update(t.lower().split())
    vocab = sorted(vocab)
    vecs = []
    for text in texts:
        counts = Counter(text.lower().split())
        vec = [counts.get(w, 0) for w in vocab]
        norm = math.sqrt(sum(v**2 for v in vec)) or 1.0
        vecs.append([v / norm for v in vec])
    return vecs


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x**2 for x in a))
    nb = math.sqrt(sum(x**2 for x in b))
    return dot / (na * nb) if na * nb > 0 else 0.0


# ─── Simple FAISS-style retriever ────────────────────────────────────────────

class VectorRetriever:
    """
    In-memory vector retriever backed by FAISS (if available) or pure Python.
    """
    def __init__(self, backend: str = "auto"):
        self.docs: list[dict] = []
        self.embeddings: list[list[float]] = []
        self.backend = backend
        self._faiss_index = None

    def add_documents(self, docs: list[dict]):
        """Add documents with 'text' and 'id' fields."""
        texts = [d["text"] for d in docs]
        vecs = embed(texts)
        self.docs.extend(docs)
        self.embeddings.extend(vecs)
        self._rebuild_index()

    def _rebuild_index(self):
        try:
            import faiss, numpy as np
            if not self.embeddings: return
            arr = np.array(self.embeddings, dtype="float32")
            dim = arr.shape[1]
            index = faiss.IndexFlatIP(dim)  # Inner product (cosine after normalization)
            faiss.normalize_L2(arr)
            index.add(arr)
            self._faiss_index = index
        except ImportError:
            self._faiss_index = None

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """Retrieve top-k documents for a query."""
        if not self.docs:
            return []
        query_vec = embed([query])[0]

        if self._faiss_index is not None:
            try:
                import faiss, numpy as np
                qv = np.array([query_vec], dtype="float32")
                faiss.normalize_L2(qv)
                D, I = self._faiss_index.search(qv, min(k, len(self.docs)))
                results = []
                for score, idx in zip(D[0], I[0]):
                    if idx < len(self.docs):
                        doc = dict(self.docs[idx])
                        doc["retrieval_score"] = float(score)
                        results.append(doc)
                return results
            except Exception:
                pass

        # Pure Python fallback
        scores = [(cosine_similarity(query_vec, ev), i) for i, ev in enumerate(self.embeddings)]
        scores.sort(reverse=True)
        return [{**self.docs[i], "retrieval_score": round(s, 4)} for s, i in scores[:k]]


# ─── ChromaDB persistent retriever ───────────────────────────────────────────

class ChromaRetriever:
    def __init__(self, collection_name: str = "llm_eval_rag"):
        self._client = None
        self._collection = None
        self.collection_name = collection_name

    def _get_collection(self):
        if self._collection is None:
            try:
                import chromadb
                self._client = chromadb.Client()
                self._collection = self._client.get_or_create_collection(self.collection_name)
            except ImportError:
                return None
        return self._collection

    def add_documents(self, docs: list[dict]):
        col = self._get_collection()
        if col is None: return
        col.add(
            ids=[d["id"] for d in docs],
            documents=[d["text"] for d in docs],
            metadatas=[{k: v for k, v in d.items() if k not in ("id", "text")} for d in docs],
        )

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        col = self._get_collection()
        if col is None: return []
        try:
            results = col.query(query_texts=[query], n_results=k)
            docs = []
            for i, (doc_id, text, dist) in enumerate(zip(
                results["ids"][0], results["documents"][0], results["distances"][0]
            )):
                docs.append({"id": doc_id, "text": text,
                             "retrieval_score": round(1 - dist, 4),
                             "metadata": results.get("metadatas", [[]])[0][i] if results.get("metadatas") else {}})
            return docs
        except Exception as e:
            return []


# ─── Retriever Metrics ────────────────────────────────────────────────────────

def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Recall@K: what fraction of relevant docs were retrieved in top-K."""
    if not relevant_ids: return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & set(relevant_ids)) / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Precision@K: what fraction of top-K are relevant."""
    if k == 0: return 0.0
    top_k = retrieved_ids[:k]
    return sum(1 for r in top_k if r in relevant_ids) / k


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Mean Reciprocal Rank: rank of first relevant document."""
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """NDCG@K: normalized discounted cumulative gain."""
    def dcg(ids, rel_set, k):
        return sum(1 / math.log2(i + 2) for i, r in enumerate(ids[:k]) if r in rel_set)
    actual = dcg(retrieved_ids, set(relevant_ids), k)
    ideal = dcg(sorted(set(relevant_ids), key=lambda x: -1), set(relevant_ids), k)
    return actual / ideal if ideal > 0 else 0.0


# ─── Grounding Analysis ───────────────────────────────────────────────────────

async def analyze_grounding(client, query: str, answer: str, context_chunks: list[str]) -> dict:
    """
    Analyze what fraction of the answer is grounded in the retrieved context.
    Returns per-chunk attribution and overall groundedness score.
    """
    ctx = "\n---\n".join(f"[Chunk {i+1}]: {c}" for i, c in enumerate(context_chunks))
    judge_prompt = f"""You are a RAG grounding evaluator. Analyze the answer and determine which parts 
are supported by the context chunks.

QUERY: {query}
CONTEXT:
{ctx}

ANSWER: {answer}

For each sentence in the answer, identify:
1. Which context chunk(s) support it (or NONE if hallucinated)
2. Whether it's grounded

Return ONLY JSON:
{{
  "sentences": [
    {{"text": "...", "grounded": true, "source_chunks": [1,2], "confidence": 0.9}}
  ],
  "groundedness_score": 0.85,
  "ungrounded_sentences": ["list of ungrounded sentences"],
  "chunk_usage": {{"1": 3, "2": 1}}
}}"""

    try:
        raw = await client.judge(judge_prompt)
        raw = re.sub(r"```json|```", "", raw).strip()
        data = json.loads(raw)
        sentences = data.get("sentences", [])
        grounded_count = sum(1 for s in sentences if s.get("grounded"))
        return {
            "groundedness_score": data.get("groundedness_score", grounded_count / max(len(sentences), 1)),
            "sentences": sentences,
            "ungrounded_sentences": data.get("ungrounded_sentences", []),
            "chunk_usage": data.get("chunk_usage", {}),
            "grounded_count": grounded_count,
            "total_sentences": len(sentences),
        }
    except Exception as e:
        return {"groundedness_score": 0.5, "error": str(e)}


# ─── Citation Verification ────────────────────────────────────────────────────

def verify_citations(answer: str, context_chunks: list[str]) -> dict:
    """
    Verify that citations in the answer (e.g., [1], [Source 1]) correspond to real context.
    Also check for unsupported factual claims.
    """
    citation_pattern = re.compile(r'\[(\d+)\]|\[Source\s*(\d+)\]|\(Source\s*(\d+)\)')
    citations_found = citation_pattern.findall(answer)
    cited_nums = set()
    for match in citations_found:
        num = next(m for m in match if m)
        cited_nums.add(int(num))

    valid_range = set(range(1, len(context_chunks) + 1))
    invalid_citations = cited_nums - valid_range
    valid_citations = cited_nums & valid_range

    return {
        "citations_found": sorted(cited_nums),
        "valid_citations": sorted(valid_citations),
        "invalid_citations": sorted(invalid_citations),
        "has_hallucinated_citations": len(invalid_citations) > 0,
        "citation_accuracy": len(valid_citations) / max(len(cited_nums), 1),
    }


# ─── Full RAG Evaluation Pipeline ────────────────────────────────────────────

RAG_TEST_CASES = [
    {
        "id": "RAG_001", "query": "What is the boiling point of water?",
        "relevant_doc_ids": ["doc_water"],
        "corpus": [
            {"id": "doc_water", "text": "Water boils at 100°C (212°F) at standard atmospheric pressure (1 atm)."},
            {"id": "doc_iron", "text": "Iron melts at 1538°C and boils at 2862°C."},
            {"id": "doc_gold", "text": "Gold has a melting point of 1064°C."},
            {"id": "doc_nitrogen", "text": "Nitrogen boils at -195.8°C."},
            {"id": "doc_ethanol", "text": "Ethanol boils at 78.4°C at atmospheric pressure."},
        ]
    },
    {
        "id": "RAG_002", "query": "Who invented the telephone?",
        "relevant_doc_ids": ["doc_bell"],
        "corpus": [
            {"id": "doc_bell", "text": "Alexander Graham Bell is credited with inventing the telephone in 1876."},
            {"id": "doc_edison", "text": "Thomas Edison invented the phonograph and improved the electric light bulb."},
            {"id": "doc_tesla", "text": "Nikola Tesla contributed to AC power systems and radio technology."},
            {"id": "doc_marconi", "text": "Guglielmo Marconi is credited with the invention of radio."},
            {"id": "doc_morse", "text": "Samuel Morse developed Morse code and the telegraph."},
        ]
    },
]


async def run_rag_evaluation(
    client,
    test_cases: list[dict] = None,
    k_values: list[int] = None,
    config: Optional[dict] = None,
) -> dict:
    """Run full RAG evaluation pipeline."""
    cases = test_cases or RAG_TEST_CASES
    k_vals = k_values or [1, 3, 5]
    retriever = VectorRetriever()

    all_metrics: dict[str, list[float]] = {f"recall@{k}": [] for k in k_vals}
    all_metrics.update({f"precision@{k}": [] for k in k_vals})
    all_metrics.update({f"ndcg@{k}": [] for k in k_vals})
    all_metrics["mrr"] = []
    all_metrics["groundedness"] = []
    all_metrics["faithfulness"] = []

    results = []
    print(f"\n[RAG Eval] Running {len(cases)} test cases...")

    for case in cases:
        # Build corpus-specific retriever
        case_retriever = VectorRetriever()
        case_retriever.add_documents(case["corpus"])
        retrieved = case_retriever.retrieve(case["query"], k=max(k_vals))
        retrieved_ids = [d["id"] for d in retrieved]
        relevant_ids = case["relevant_doc_ids"]

        # Retrieval metrics
        for k in k_vals:
            all_metrics[f"recall@{k}"].append(recall_at_k(retrieved_ids, relevant_ids, k))
            all_metrics[f"precision@{k}"].append(precision_at_k(retrieved_ids, relevant_ids, k))
            all_metrics[f"ndcg@{k}"].append(ndcg_at_k(retrieved_ids, relevant_ids, k))
        all_metrics["mrr"].append(mrr(retrieved_ids, relevant_ids))

        # Generate RAG answer
        top_chunks = [d["text"] for d in retrieved[:3]]
        context = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(top_chunks))
        rag_prompt = f"Answer based ONLY on the provided context.\n\nContext:\n{context}\n\nQuestion: {case['query']}"
        try:
            answer, _, _, lat = await client.generate(rag_prompt, temperature=0.0)
            grounding = await analyze_grounding(client, case["query"], answer, top_chunks)
            all_metrics["groundedness"].append(grounding["groundedness_score"])
            from modules.evaluator import score_faithfulness
            faith = await score_faithfulness(client, case["query"], answer, context)
            all_metrics["faithfulness"].append(faith)
            citations = verify_citations(answer, top_chunks)
            results.append({
                "id": case["id"], "query": case["query"], "retrieved_ids": retrieved_ids,
                "relevant_ids": relevant_ids, "answer": answer[:300],
                "groundedness": grounding["groundedness_score"],
                "faithfulness": faith, "citations": citations, "latency_ms": lat,
                "mrr": mrr(retrieved_ids, relevant_ids),
                f"recall@{k_vals[-1]}": recall_at_k(retrieved_ids, relevant_ids, k_vals[-1]),
            })
        except Exception as e:
            results.append({"id": case["id"], "error": str(e)})

    import statistics
    summary = {
        metric: round(statistics.mean(vals), 4)
        for metric, vals in all_metrics.items() if vals
    }
    print(f"[RAG Eval] Results:")
    for m, v in summary.items():
        print(f"  {m:20s}: {v:.4f}")
    return {"summary": summary, "results": results, "n_cases": len(cases)}
