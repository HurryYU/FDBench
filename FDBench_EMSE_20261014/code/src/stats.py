"""
stats.py — 统计分析

设计要点（审稿人会看这几处）：
  1) 配对设计：同一任务在各组之间配对，用配对检验而非独立样本检验；
  2) 二值指标用 McNemar 精确检验，连续指标用 Wilcoxon 符号秩检验；
  3) 多重比较用 Holm 校正（比 Bonferroni 更有效且同样严格）；
  4) 一律报告效应量与 95% 置信区间，不只报 p 值。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps


# ==========================================================================
# 比例的置信区间：Wilson 区间（小样本下优于正态近似）
# ==========================================================================

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = sps.norm.ppf(1 - alpha / 2)
    p = k / n
    d = 1 + z ** 2 / n
    c = p + z ** 2 / (2 * n)
    hw = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return (max(0.0, (c - hw) / d), min(1.0, (c + hw) / d))


# ==========================================================================
# 二值配对：McNemar 精确检验
# ==========================================================================

@dataclass
class PairedBinaryResult:
    n: int
    p_a: float
    p_b: float
    b: int          # A 对 B 错
    c: int          # A 错 B 对
    p_value: float
    cohens_h: float
    ci_a: tuple[float, float]
    ci_b: tuple[float, float]

    def line(self, name_a: str, name_b: str) -> str:
        return (f"{name_a} {self.p_a:.1%} vs {name_b} {self.p_b:.1%} | "
                f"不一致对 b={self.b}, c={self.c} | p={self.p_value:.4g} | h={self.cohens_h:.2f}")


def cohens_h(p1: float, p2: float) -> float:
    """两个比例之间的效应量。0.2 小 / 0.5 中 / 0.8 大。"""
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


def mcnemar_exact(a: list[int], b: list[int]) -> PairedBinaryResult:
    """
    a, b 为同一批任务在两组下的 0/1 结果，长度相同、顺序对应。
    小样本下使用二项精确检验，不做连续性近似。
    """
    a, b = np.asarray(a, int), np.asarray(b, int)
    assert a.shape == b.shape, "配对样本长度必须一致"
    n = len(a)
    n_b = int(np.sum((a == 1) & (b == 0)))
    n_c = int(np.sum((a == 0) & (b == 1)))
    m = n_b + n_c
    p = 1.0 if m == 0 else float(
        sps.binomtest(min(n_b, n_c), m, 0.5, alternative="two-sided").pvalue)
    pa, pb = a.mean(), b.mean()
    return PairedBinaryResult(
        n=n, p_a=float(pa), p_b=float(pb), b=n_b, c=n_c, p_value=p,
        cohens_h=float(cohens_h(pa, pb)),
        ci_a=wilson_ci(int(a.sum()), n), ci_b=wilson_ci(int(b.sum()), n))


# ==========================================================================
# 连续配对：Wilcoxon 符号秩检验
# ==========================================================================

@dataclass
class PairedContinuousResult:
    n: int
    median_a: float
    median_b: float
    statistic: float
    p_value: float
    effect_r: float     # r = Z / sqrt(N)

    def line(self, name_a: str, name_b: str) -> str:
        return (f"{name_a} 中位数 {self.median_a:.4g} vs {name_b} {self.median_b:.4g} | "
                f"p={self.p_value:.4g} | r={self.effect_r:.2f}")


def wilcoxon_paired(a: list[float], b: list[float]) -> PairedContinuousResult:
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape
    diff = a - b
    nz = int(np.sum(diff != 0))
    if nz == 0:
        return PairedContinuousResult(len(a), float(np.median(a)), float(np.median(b)),
                                      0.0, 1.0, 0.0)
    res = sps.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    # p 值极小时 norm.ppf 会溢出为 inf，需先夹紧；r 的理论上界为 1。
    p_safe = float(np.clip(res.pvalue, 1e-15, 1.0))
    z = sps.norm.ppf(1 - p_safe / 2)
    return PairedContinuousResult(
        n=len(a), median_a=float(np.median(a)), median_b=float(np.median(b)),
        statistic=float(res.statistic), p_value=float(res.pvalue),
        effect_r=float(min(1.0, abs(z) / np.sqrt(nz))))


# ==========================================================================
# 多重比较校正
# ==========================================================================

def holm_correct(pvals: list[float], alpha: float = 0.05) -> list[dict]:
    """Holm-Bonferroni 逐步校正。返回与输入同序的结果列表。"""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        running = max(running, val)          # 保证单调不减
        adjusted[idx] = running
    return [{"p_raw": pvals[i], "p_holm": adjusted[i],
             "significant": adjusted[i] < alpha} for i in range(m)]


# ==========================================================================
# 阶梯式消融的成对比较（A→B→C→D→E→F 相邻组）
# ==========================================================================

ARM_LADDER = ["A", "B", "C", "D", "E", "F"]
LADDER_LABEL = {
    ("A", "B"): "检索的贡献",
    ("B", "C"): "视觉输入的贡献",
    ("C", "D"): "视觉检索相对文本检索的贡献",
    ("D", "E"): "基础执行验证的贡献",
    ("E", "F"): "渲染比对与迭代闭环的贡献",
}


def ladder_report(binary_by_arm: dict[str, list[int]], metric_name: str,
                  alpha: float = 0.05) -> str:
    """对相邻组做配对检验并输出可直接誊入论文的结论段。"""
    pairs, results = [], []
    for i in range(len(ARM_LADDER) - 1):
        x, y = ARM_LADDER[i], ARM_LADDER[i + 1]
        if x in binary_by_arm and y in binary_by_arm:
            pairs.append((x, y))
            results.append(mcnemar_exact(binary_by_arm[y], binary_by_arm[x]))

    corr = holm_correct([r.p_value for r in results], alpha)
    lines = [f"=== {metric_name} 阶梯消融（Holm 校正，α={alpha}）==="]
    for (x, y), r, c in zip(pairs, results, corr):
        star = "显著" if c["significant"] else "不显著"
        lines.append(
            f"{x}->{y} [{LADDER_LABEL.get((x, y), '')}]: "
            f"{r.p_b:.1%} -> {r.p_a:.1%} "
            f"(Δ={r.p_a - r.p_b:+.1%}, b={r.b}, c={r.c}, "
            f"p_raw={r.p_value:.4g}, p_holm={c['p_holm']:.4g}, h={r.cohens_h:.2f}) {star}")
    return "\n".join(lines)


def power_note(n: int, p_base: float, p_new: float) -> str:
    """粗略的事前功效提示，帮助判断样本量是否够用。"""
    h = abs(cohens_h(p_new, p_base))
    z_a = sps.norm.ppf(0.975)
    z_b = h * np.sqrt(n) - z_a
    power = float(sps.norm.cdf(z_b))
    return (f"n={n}, {p_base:.0%}->{p_new:.0%}, |h|={h:.2f}, "
            f"约合功效≈{power:.2f}" + ("（充足）" if power >= 0.8 else "（偏低，建议加大任务量）"))


# ==========================================================================
# 自测
# ==========================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(7)
    n = 120
    base = rng.random(n) < 0.35                       # 弱组
    strong = base | (rng.random(n) < 0.45)            # 强组严格占优

    r = mcnemar_exact(strong.astype(int), base.astype(int))
    print("McNemar:", r.line("F", "A"))
    assert r.p_value < 0.001 and r.c == 0

    same = mcnemar_exact(base.astype(int), base.astype(int))
    assert same.p_value == 1.0 and same.b == same.c == 0

    lo, hi = wilson_ci(42, 120)
    assert lo < 42 / 120 < hi and 0 < lo < hi < 1

    a = rng.normal(0.80, 0.05, n)
    b = rng.normal(0.62, 0.05, n)
    w = wilcoxon_paired(a, b)
    print("Wilcoxon:", w.line("F", "A"))
    assert w.p_value < 0.001 and w.effect_r > 0.5

    h = holm_correct([0.001, 0.02, 0.04, 0.30])
    assert h[0]["significant"] and not h[3]["significant"]
    assert h[0]["p_holm"] <= h[1]["p_holm"] <= h[2]["p_holm"] <= h[3]["p_holm"]

    arms = {}
    prev = rng.random(n) < 0.20
    for k in ARM_LADDER:
        prev = prev | (rng.random(n) < 0.15)
        arms[k] = prev.astype(int).tolist()
    print()
    print(ladder_report(arms, "Fix@1"))
    print()
    print("功效提示:", power_note(120, 0.35, 0.55))
    print("功效提示:", power_note(60, 0.35, 0.45))
    print("\nstats.py 全部自测通过。")
