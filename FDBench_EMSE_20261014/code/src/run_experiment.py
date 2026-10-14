"""
run_experiment.py — 阶段三主控：六组 × 全部任务，批量运行并落盘

职责：
  1. 读取 build_tasks 产出的 tasks.jsonl，补齐 arms.py 需要但 jsonl 未存的字段：
       - _root：任务目录根，open_text() 靠它拼 HTML 路径
       - target_png / student_png：转成绝对路径，llm 才能读图
  2. 接线：llm（Qwen3.5-Plus）、retriever_text / retriever_visual（你已跑通的 retrievers.py）
  3. 六组 A~F 对每个任务依次运行 run_arm，结果写 results/runs.jsonl
  4. 断点续跑：已完成的 (task_id, arm) 跳过，中途崩溃可直接重跑接上
  5. 全程共用一个浏览器实例与一套检索器，避免重复初始化

运行：
    export DASHSCOPE_API_KEY="sk-..."
    python run_experiment.py --tasks ../data/tasks --kb ../data/kb \
                             --out ../results [--arms ABCDEF] [--limit 0]

先小样冒烟（强烈建议正式跑前先做）：
    python run_experiment.py --tasks ../data/tasks --kb ../data/kb \
                             --out ../results --limit 3
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from arms import run_arm

# 六组的运行顺序（阶梯式，A 为下界，F 为完整 M-RAG-Ed）
ARMS = ["A", "B", "C", "D", "E", "F"]


def load_tasks(tasks_dir: Path) -> list[dict]:
    """读 tasks.jsonl，补齐 _root 与绝对图像路径。"""
    jl = tasks_dir / "tasks.jsonl"
    tasks = [json.loads(l) for l in jl.read_text(encoding="utf-8").splitlines() if l.strip()]
    root = str(tasks_dir.resolve())
    for t in tasks:
        t["_root"] = root
        # jsonl 里是相对路径（img/xxx.png）；转绝对，供 llm 读图与验证器渲染
        t["target_png"] = str((tasks_dir / t["target_png"]).resolve())
        t["student_png"] = str((tasks_dir / t["student_png"]).resolve())
    return tasks


def load_done(runs_path: Path) -> set[tuple[str, str]]:
    """已完成的 (task_id, arm)，用于断点续跑。"""
    done = set()
    if runs_path.exists():
        for line in runs_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                # 只把成功（无 error）的算作已完成，出错的允许重跑
                if not r.get("error"):
                    done.add((r["task_id"], r["arm"]))
            except Exception:
                pass
    return done


def main(tasks_dir: Path, kb: Path, out: Path, arms: list[str],
         limit: int) -> None:
    from playwright.sync_api import sync_playwright
    from llm_qwen import llm
    from retrievers import TextRetriever, VisualRetriever

    out.mkdir(parents=True, exist_ok=True)
    runs_path = out / "runs.jsonl"

    tasks = load_tasks(tasks_dir)
    if limit > 0:
        tasks = tasks[:limit]
    done = load_done(runs_path)
    print(f"任务 {len(tasks)} 个 × 组 {arms} = {len(tasks)*len(arms)} 次运行"
          f"（已完成 {len(done)} 次，将跳过）")

    # 初始化检索器（一次，全程复用）
    print("加载检索器 …")
    tr = TextRetriever(str(kb))
    vr = VisualRetriever(str(kb))
    retriever_text = lambda q, k=5, **kw: tr.search(q, k)
    retriever_visual = lambda q, k=5, **kw: vr.search(q, k)

    n_total = len(tasks) * len(arms)
    n_ok = n_fail = n_skip = 0
    t_start = time.time()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--hide-scrollbars"])
        try:
            with runs_path.open("a", encoding="utf-8") as fout:
                for ti, task in enumerate(tasks):
                    for arm in arms:
                        if (task["task_id"], arm) in done:
                            n_skip += 1
                            continue
                        run = run_arm(task, arm, llm,
                                      retriever_text, retriever_visual, browser)
                        rec = asdict(run)
                        # 精简落盘：逐轮 HTML 太大，只留最后一版供复算
                        rec["fixed_html_final"] = (
                            run.fixed_html_per_iter[-1]
                            if run.fixed_html_per_iter else "")
                        rec.pop("fixed_html_per_iter", None)
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fout.flush()

                        if run.error:
                            n_fail += 1
                            print(f"  [{task['task_id']} {arm}] 失败：{run.error[:100]}")
                        else:
                            n_ok += 1

                        done_n = n_ok + n_fail
                        if done_n % 10 == 0 and done_n > 0:
                            el = time.time() - t_start
                            rate = el / done_n
                            left = (n_total - len(done) - done_n) * rate
                            print(f"  进度 {done_n} 完成 | 成功 {n_ok} 失败 {n_fail} "
                                  f"| 已用 {el/60:.1f}min 预计剩 {left/60:.1f}min")
        finally:
            browser.close()

    print(f"\n完成。成功 {n_ok}，失败 {n_fail}，跳过 {n_skip}。")
    print(f"结果写入 {runs_path}")
    if n_fail:
        print(f"有 {n_fail} 次失败，可直接重跑本脚本，会自动跳过已成功的、只补失败的。")
    print("下一步：阶段四 evaluate.py 读取 runs.jsonl 计算指标。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=Path, default=Path("../data/tasks"))
    ap.add_argument("--kb", type=Path, default=Path("../data/kb"))
    ap.add_argument("--out", type=Path, default=Path("../results"))
    ap.add_argument("--arms", type=str, default="ABCDEF",
                    help="要跑的组，如 ABDF 只跑四组")
    ap.add_argument("--limit", type=int, default=0,
                    help="只跑前 N 个任务（冒烟测试用），0 表示全部")
    a = ap.parse_args()
    main(a.tasks, a.kb, a.out, list(a.arms), a.limit)
