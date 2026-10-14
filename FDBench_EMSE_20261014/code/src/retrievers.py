"""
retrievers.py — 从 build_index.py 产物加载并执行检索

提供 arms.py 所需的两个接口（签名与手册一致）：
    retriever_text(query, k)               -> [{doc_id,title,snippet}, ...]
    retriever_visual(query, k, image_path) -> [{doc_id,title,snippet}, ...]

文本侧：BM25 与 BGE-m3 分数各自 min-max 归一后加权融合（默认 0.5/0.5），
        按 chunk 检索后聚合到 doc 级去重。
视觉侧：ColPali MaxSim（late interaction）——query 文本编码为多向量，
        与每页图像块向量做 MaxSim 打分。注意 ColPali 用文本 query 检索图像，
        故 image_path 参数在此不必需（保留以兼容 arms.py 的统一签名）。

用法（在 arms.py 顶部）：
    from retrievers import TextRetriever, VisualRetriever
    _tr = TextRetriever("../data/kb")
    _vr = VisualRetriever("../data/kb")
    retriever_text   = lambda q, k=5, **kw: _tr.search(q, k)
    retriever_visual = lambda q, k=5, **kw: _vr.search(q, k)
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np


def _load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


class TextRetriever:
    def __init__(self, kb: str, w_bm25: float = 0.5, w_dense: float = 0.5):
        kb = Path(kb)
        self.kb = kb
        self.meta = json.loads((kb / "index_meta.json").read_text(encoding="utf-8"))
        self.chunk2doc = self.meta["chunk2doc"]
        self.docs = {d["doc_id"]: d for d in _load_jsonl(kb / "docs.jsonl")}
        self.chunks = {c["chunk_id"]: c for c in _load_jsonl(kb / "chunks.jsonl")}

        with open(kb / "text_bm25.pkl", "rb") as f:
            bm = pickle.load(f)
        self.bm25 = bm["bm25"]
        self.bm25_ids = bm["ids"]

        self.hybrid = self.meta.get("text_mode") == "hybrid"
        if self.hybrid:
            self.dense = np.load(kb / "text_dense.npy")
            self.dense_ids = json.loads((kb / "text_dense_ids.json").read_text())
            self._id2row = {cid: i for i, cid in enumerate(self.dense_ids)}
            from FlagEmbedding import BGEM3FlagModel
            self._emb = BGEM3FlagModel("/home/linapink/data-center/mragexp/bge-m3", use_fp16=True)
        self.w_bm25, self.w_dense = w_bm25, w_dense

    def _tok(self, s: str):
        import re
        return re.findall(r"[a-z0-9\-]+", s.lower())

    def search(self, query: str, k: int = 5) -> list[dict]:
        bm_scores = np.asarray(self.bm25.get_scores(self._tok(query)), dtype=np.float32)
        fused = self.w_bm25 * _minmax(bm_scores)

        if self.hybrid:
            q = self._emb.encode([query], return_dense=True,
                                 return_sparse=False, return_colbert_vecs=False)
            qv = np.asarray(q["dense_vecs"], dtype=np.float32)[0]
            # BGE-m3 向量已归一，点积即余弦
            dense_scores_full = self.dense @ qv
            # 对齐到 bm25_ids 顺序
            aligned = np.array(
                [dense_scores_full[self._id2row[cid]] if cid in self._id2row else 0.0
                 for cid in self.bm25_ids], dtype=np.float32)
            fused = fused + self.w_dense * _minmax(aligned)

        order = np.argsort(-fused)
        seen, results = set(), []
        for idx in order:
            cid = self.bm25_ids[idx]
            doc_id = self.chunk2doc.get(cid, cid)
            if doc_id in seen:
                continue
            seen.add(doc_id)
            ch = self.chunks.get(cid, {})
            results.append({
                "doc_id": doc_id,
                "title": self.docs.get(doc_id, {}).get("title", doc_id),
                "snippet": ch.get("text", "")[:300],
                "score": float(fused[idx]),
            })
            if len(results) >= k:
                break
        return results


class VisualRetriever:
    def __init__(self, kb: str, model_name: str | None = None):
        import torch
        from transformers import ColPaliForRetrieval, ColPaliProcessor

        kb = Path(kb)
        self.kb = kb
        self.meta = json.loads((kb / "index_meta.json").read_text(encoding="utf-8"))
        self.docs = {d["doc_id"]: d for d in _load_jsonl(kb / "docs.jsonl")}
        model_name = model_name or self.meta["visual_model"]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.torch = torch
        self.model = ColPaliForRetrieval.from_pretrained(
            model_name, torch_dtype=dtype, device_map=self.device).eval()
        self.processor = ColPaliProcessor.from_pretrained(model_name)

        self.doc_embs = torch.load(kb / "visual_embs.pt")   # list[Tensor(n_patch,dim)]
        self.doc_ids = json.loads((kb / "visual_ids.json").read_text())

    @staticmethod
    def _2d(t):
        """把 [1, n, d] 压成 [n, d]；已是 [n, d] 则原样返回。"""
        while t.dim() > 2:
            t = t.squeeze(0)
        return t

    def _maxsim(self, q_emb, d_emb) -> float:
        # q_emb: [nq, dim]  d_emb: [nd, dim]  -> sum_i max_j q_i·d_j
        q = self._2d(q_emb).float()
        d = self._2d(d_emb).float()
        sim = q @ d.mT                             # [nq, nd]；.mT 避免高维 .T 弃用告警
        return float(sim.max(dim=1).values.sum().item())

    def _encode_query(self, query: str):
        """transformers 版与 engine 版的输出结构不同，这里统一取多向量。"""
        torch = self.torch
        batch = self.processor.process_queries([query]).to(self.device)
        out = self.model(**batch)
        emb = out.embeddings if hasattr(out, "embeddings") else out[0]
        return self._2d(emb).float().cpu()

    def search(self, query: str, k: int = 5, image_path: str | None = None) -> list[dict]:
        torch = self.torch
        with torch.no_grad():
            q_emb = self._encode_query(query)               # [nq, dim]
            scores = []
            for did, d_emb in zip(self.doc_ids, self.doc_embs):
                scores.append((self._maxsim(q_emb, d_emb), did))
        scores.sort(reverse=True)
        results = []
        for sc, did in scores[:k]:
            d = self.docs.get(did, {})
            results.append({
                "doc_id": did,
                "title": d.get("title", did),
                "snippet": f"[视觉页面] {d.get('title', did)}（{d.get('subset','')}）",
                "score": sc,
            })
        return results


if __name__ == "__main__":
    import sys
    kb = sys.argv[1] if len(sys.argv) > 1 else "../data/kb"
    q = sys.argv[2] if len(sys.argv) > 2 else "elements not centered horizontally in a flex container"
    print("=== 文本检索 ===")
    for r in TextRetriever(kb).search(q, 5):
        print(f"  {r['score']:.3f}  {r['doc_id']:28s} {r['title']}")
    print("=== 视觉检索 ===")
    for r in VisualRetriever(kb).search(q, 5):
        print(f"  {r['score']:.3f}  {r['doc_id']:28s} {r['title']}")
