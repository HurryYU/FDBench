"""
build_index.py — 阶段二(2/2)：对知识库建双模态索引

文本侧：BM25（词法）+ BGE-m3（稠密语义）混合，分数归一后加权融合
视觉侧：ColPali v1.3 多向量图像块索引（late interaction / MaxSim 检索）

针对 GTX 3060 12G 的显存策略：
  - ColPali 以 fp16 加载，页面图像 batch=1 编码，避免 OOM
  - 文档侧图像块嵌入预计算并落盘，查询时只编码 query（在线开销小）
  - BGE-m3 亦 fp16，dense-only（关闭 sparse/colbert 输出以省显存）

产物（供 retriever_text / retriever_visual 加载）：
  text_bm25.pkl      BM25 语料与倒排
  text_dense.npy     BGE-m3 文档向量  + text_dense_ids.json
  visual_embs.pt     ColPali 文档图像块多向量 + visual_ids.json
  index_meta.json    维度、模型名、chunk/doc 对照

运行：
    python build_index.py --kb ../data/kb --text-index hybrid \
                          --visual-model vidore/colpali-v1.3-hf
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# ==========================================================================
# 文本索引
# ==========================================================================
def build_text_index(kb: Path, chunks: list[dict], mode: str) -> dict:
    ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]

    # --- BM25 ---
    from rank_bm25 import BM25Okapi
    import re

    def tok(s: str) -> list[str]:
        # 英文文档：小写 + 词/数字切分即可
        return re.findall(r"[a-z0-9\-]+", s.lower())

    corpus_tok = [tok(t) for t in texts]
    bm25 = BM25Okapi(corpus_tok)
    with open(kb / "text_bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids, "corpus_tok": corpus_tok}, f)
    print(f"  BM25：{len(ids)} 个文本块")

    dense_dim = 0
    if mode == "hybrid":
        # --- BGE-m3 稠密（fp16, dense-only）---
        from FlagEmbedding import BGEM3FlagModel
        model = BGEM3FlagModel("/home/linapink/data-center/mragexp/bge-m3", use_fp16=True)
        out = model.encode(texts, batch_size=8, max_length=512,
                           return_dense=True, return_sparse=False,
                           return_colbert_vecs=False)
        dense = np.asarray(out["dense_vecs"], dtype=np.float32)
        np.save(kb / "text_dense.npy", dense)
        (kb / "text_dense_ids.json").write_text(
            json.dumps(ids, ensure_ascii=False), encoding="utf-8")
        dense_dim = int(dense.shape[1])
        print(f"  BGE-m3：{dense.shape[0]} 向量，维度 {dense_dim}")

    return {"text_mode": mode, "n_chunks": len(ids), "dense_dim": dense_dim}


# ==========================================================================
# 视觉索引：ColPali 多向量
# ==========================================================================
def build_visual_index(kb: Path, docs: list[dict], model_name: str) -> dict:
    import torch
    from PIL import Image
    from transformers import ColPaliForRetrieval, ColPaliProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"  加载 {model_name}（{device}, {dtype}）")
    model = ColPaliForRetrieval.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device).eval()
    processor = ColPaliProcessor.from_pretrained(model_name)

    def _2d(t):
        while t.dim() > 2:
            t = t.squeeze(0)
        return t

    doc_imgs = [(d["doc_id"], kb / d["image"]) for d in docs if d.get("image")]
    embs, ids = [], []
    with torch.no_grad():
        for i, (doc_id, img_path) in enumerate(doc_imgs):
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")
            # 页面过高会撑爆 patch 数与显存：限制高度
            if img.height > 1600:
                img = img.crop((0, 0, img.width, 1600))
            batch = processor.process_images([img]).to(device)
            out = model(**batch)
            # transformers 版输出在 .embeddings，engine 版可下标取；统一成 [n_patch, dim]
            emb = out.embeddings if hasattr(out, "embeddings") else out[0]
            embs.append(_2d(emb).to(torch.float16).cpu())
            ids.append(doc_id)
            if (i + 1) % 20 == 0:
                print(f"    已编码 {i+1}/{len(doc_imgs)} 张页面")
                torch.cuda.empty_cache()

    torch.save(embs, kb / "visual_embs.pt")
    (kb / "visual_ids.json").write_text(
        json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    dim = int(embs[0].shape[-1]) if embs else 0
    print(f"  ColPali：{len(ids)} 张页面，patch 维度 {dim}")
    return {"visual_model": model_name, "n_pages": len(ids), "patch_dim": dim}


# ==========================================================================
# 主流程
# ==========================================================================
def main(kb: Path, text_mode: str, visual_model: str) -> None:
    docs = load_jsonl(kb / "docs.jsonl")
    chunks = load_jsonl(kb / "chunks.jsonl")
    print(f"载入知识库：文档 {len(docs)} 篇，文本块 {len(chunks)} 个")

    print("构建文本索引 …")
    tmeta = build_text_index(kb, chunks, text_mode)

    print("构建视觉索引 …")
    vmeta = build_visual_index(kb, docs, visual_model)

    meta = {**tmeta, **vmeta,
            "doc2subset": {d["doc_id"]: d["subset"] for d in docs},
            "chunk2doc": {c["chunk_id"]: c["doc_id"] for c in chunks}}
    (kb / "index_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n索引完成，元信息写入 {kb/'index_meta.json'}")
    print("下一步：实现 retriever_text() 与 retriever_visual() 从本目录加载索引"
          "（见下方 retrievers.py 模板）。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", type=Path, required=True)
    ap.add_argument("--text-index", choices=["bm25", "hybrid"], default="hybrid")
    ap.add_argument("--visual-model", default="vidore/colpali-v1.3-hf")
    a = ap.parse_args()
    main(a.kb, a.text_index, a.visual_model)
