"""Generate LaTeX table fragments and number macros from results."""
import glob
import json
import os

ROOT = "results"
OUT = "report/tables"
TASKS = ["mnli", "sst2", "mrpc", "cola", "qnli", "qqp", "rte", "stsb"]
PAPER_LORA = dict(mnli=87.5, sst2=95.1, mrpc=89.7, cola=63.4, qnli=93.3,
                  qqp=90.8, rte=86.6, stsb=91.5)
PAPER_FT = dict(mnli=87.6, sst2=94.8, mrpc=90.2, cola=63.6, qnli=92.8,
                qqp=91.9, rte=78.7, stsb=91.2)
PAPER_E2E = dict(FT=dict(BLEU=68.2, METEOR=46.2, ROUGE=71.0, CIDEr=2.47),
                 LoRA=dict(BLEU=70.4, METEOR=46.8, ROUGE=71.8, CIDEr=2.53))
EPOCHS = dict(mnli=3, sst2=10, mrpc=20, cola=15, qnli=5, qqp=3, rte=20, stsb=20)


def load(p):
    with open(p) as f:
        return json.load(f)


def fmt(x, n=1):
    return "--" if x is None else f"{x:.{n}f}"


def glue_table():
    ours, ft = {}, {}
    for t in TASKS:
        p = f"{ROOT}/glue/{t}_roberta-base_lora_r8_qv_s0/result.json"
        if os.path.exists(p):
            ours[t] = load(p)
        p = f"{ROOT}/glue/ft_{t}/result.json"
        if os.path.exists(p):
            ft[t] = load(p)
    lines = []
    lines.append(r"\begin{tabular}{l r c c c c c c}")
    lines.append(r"\toprule")
    lines.append(r"任务 & LoRA 参数量 & 论文 LoRA & 复现 LoRA & $\Delta$ & 论文 FT & 复现 FT & LoRA 轮数 \\")
    lines.append(r"\midrule")
    vals_o, vals_p, vals_ft, vals_ft_paper = [], [], [], []
    for t in TASKS:
        if t not in ours:
            continue
        o = ours[t]
        po = PAPER_LORA[t]
        ftv = ft[t]["best_score"] * 100 if t in ft else None
        vals_o.append(o["best_score"] * 100)
        vals_p.append(po)
        if ftv is not None:
            vals_ft.append(ftv)
            vals_ft_paper.append(PAPER_FT[t])
        lines.append(
            f"{t.upper()} & {o['n_lora']/1e6:.2f}M & {po:.1f} & {o['best_score']*100:.1f} & "
            f"{o['best_score']*100-po:+.1f} & {PAPER_FT[t]:.1f} & {fmt(ftv)} & {EPOCHS[t]} \\\\")
    if not vals_o:
        open(f"{OUT}/glue.tex", "w").write(
            "\\begin{tabular}{l r c c c c c c}\n\\toprule\n"
            "任务 & LoRA 参数量 & 论文 LoRA & 复现 LoRA & $\\Delta$ & 论文 FT & 复现 FT & Epochs \\\\\n"
            "\\midrule\n\\multicolumn{8}{c}{（实验运行中）}\\\\\n\\bottomrule\n\\end{tabular}\n")
        return None
    lines.append(r"\midrule")
    avg_o = sum(vals_o) / len(vals_o)
    avg_p = sum(vals_p) / len(vals_p)
    avg_ft = (sum(vals_ft) / len(vals_ft)) if vals_ft else None
    avg_ft_paper = (sum(vals_ft_paper) / len(vals_ft_paper)) if vals_ft_paper else None
    lines.append(f"平均 & -- & {avg_p:.1f} & {avg_o:.1f} & {avg_o-avg_p:+.1f} & "
                 f"{fmt(avg_ft_paper)} & {fmt(avg_ft)} & -- \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    open(f"{OUT}/glue.tex", "w").write("\n".join(lines))
    return dict(avg_ours=avg_o, avg_paper=avg_p, avg_ft=avg_ft, avg_ft_paper=avg_ft_paper,
                n_tasks=len(vals_o), per_task={t: ours[t]["best_score"] * 100 for t in ours},
                per_task_ft={t: ft[t]["best_score"] * 100 for t in ft})


def rank_table():
    lines = [r"\begin{tabular}{c r c}", r"\toprule",
             r"Rank $r$ & LoRA 参数量 & SST-2 Dev Acc. \\", r"\midrule"]
    vals = {}
    for r in [1, 2, 4, 8, 64]:
        p = f"{ROOT}/glue/abl_rank{r}_sst2/result.json"
        if os.path.exists(p):
            d = load(p)
            vals[r] = d["best_score"] * 100
            lines.append(f"{r} & {d['n_lora']:,} & {vals[r]:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/rank.tex", "w").write("\n".join(lines))
    return vals


def target_table():
    cfgs = [("q", 8), ("k", 8), ("v", 8), ("o", 8), ("qv", 4), ("qkvo", 2)]
    lines = [r"\begin{tabular}{l c r c}", r"\toprule",
             r"适配权重 & Rank & 参数量 & SST-2 Dev Acc. \\", r"\midrule"]
    vals = {}
    for tgt, r in cfgs:
        p = f"{ROOT}/glue/abl_tgt_{tgt}_r{r}_sst2/result.json"
        if os.path.exists(p):
            d = load(p)
            vals[tgt] = d["best_score"] * 100
            names = {"q": "$W_q$", "k": "$W_k$", "v": "$W_v$", "o": "$W_o$",
                     "qv": "$W_q{+}W_v$", "qkvo": "$W_q{+}W_k{+}W_v{+}W_o$"}
            lines.append(f"{names[tgt]} (r={r}) & {r} & {d['n_lora']:,} & {vals[tgt]:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/targets.tex", "w").write("\n".join(lines))
    return vals


def e2e_table():
    runs = {}
    for p in glob.glob(f"{ROOT}/e2e/*/result.json"):
        d = load(p)
        # prefer the re-scored multi-reference metrics when available
        mm = os.path.join(os.path.dirname(p), "metrics_multi.json")
        if os.path.exists(mm):
            m = load(mm)
            d["metrics"] = m["metrics"]
            d["n_eval"] = m["n_examples"]
        runs[d["method"]] = d
    lines = [r"\begin{tabular}{l r c c c c c}", r"\toprule",
             r"方法 & 可训练参数 & BLEU & METEOR & ROUGE-L & CIDEr & 耗时 \\", r"\midrule"]
    for m, paper_key in [("ft", "FT"), ("lora", "LoRA")]:
        if m not in runs:
            continue
        d = runs[m]
        met = d["metrics"]
        name = "全量微调" if m == "ft" else f"LoRA (r={d['rank']})"
        lines.append(f"{name} & {d['n_trainable']/1e6:.2f}M & {met['BLEU']:.1f} & "
                     f"{met['METEOR']:.1f} & {met['ROUGE-L']:.1f} & {met['CIDEr']:.2f} & "
                     f"{d['elapsed_s']/60:.0f} min \\\\")
        pp = PAPER_E2E[paper_key]
        lines.append(r"\quad 论文参考 & -- & " +
                     f"{pp['BLEU']:.1f} & {pp['METEOR']:.1f} & {pp['ROUGE']:.1f} & {pp['CIDEr']:.2f} & -- \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/e2e.tex", "w").write("\n".join(lines))
    return {m: runs[m]["metrics"] for m in runs}


def analysis_table():
    p = f"{ROOT}/analysis/subspace.json"
    if not os.path.exists(p):
        open(f"{OUT}/analysis.tex", "w").write(
            "\\begin{tabular}{l c c c c c}\n\\toprule\n"
            "权重 & $\\|W\\|_F$ & $\\|\\Delta W\\|_F$ & $\\|U^\\top W V\\|_F$ & 随机子空间 & 放大倍数 \\\\\n"
            "\\midrule\n\\multicolumn{6}{c}{（分析运行中）}\\\\\n\\bottomrule\n\\end{tabular}\n")
        return None
    d = load(p)
    meta = d.get("meta", dict(rank_a=8, rank_b=64))
    ra, rb = meta["rank_a"], meta["rank_b"]
    lines = [r"\begin{tabular}{l c c c c c}", r"\toprule",
             r"权重 & 秩 & $\|\Delta W\|_F$ & $\|U^\top W V\|_F$ & 随机子空间 & 放大倍数 \\",
             r"\midrule"]
    out = {}
    for w, label in [("query", "$W_q$"), ("value", "$W_v$")]:
        x = d[w]
        rows = [(ra, x["fro_a"], x["proj_dw"], x["proj_rand"])]
        if "proj_dw_b" in x:
            rows.append((rb, x["fro_b"], x["proj_dw_b"], x["proj_rand_b"]))
        for i, (r, fro, pdw, prand) in enumerate(rows):
            lab = label if i == 0 else ""
            lines.append(f"{lab} & {r} & {fro:.2f} & {pdw:.2f} & {prand:.2f} & "
                         f"{fro/pdw:.1f}$\\times$ \\\\")
        out[w] = x
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/analysis.tex", "w").write("\n".join(lines))
    return out


def official_proj_table():
    p = f"{ROOT}/analysis/official_projection.json"
    if not os.path.exists(p):
        open(f"{OUT}/official_proj.tex", "w").write(
            "\\begin{tabular}{l c c c c c}\n\\toprule\n"
            "权重 & $\\|W\\|_F$ & $\\|\\Delta W\\|_F$ & $\\|U^\\top W V\\|_F$ & 随机子空间 & 放大倍数 \\\\\n"
            "\\midrule\n\\multicolumn{6}{c}{（分析运行中）}\\\\\n\\bottomrule\n\\end{tabular}\n")
        return None
    d = load(p)
    lines = [r"\begin{tabular}{l c c c c c c}", r"\toprule",
             r"权重 & $\|W\|_F$ & $\|\Delta W\|_F$ & $\|U^\top W V\|_F$ & $\|U_W^\top W V_W\|_F$ & 随机子空间 & 放大倍数 \\",
             r"\midrule"]
    out = {}
    for w, label in [("query", "$W_q$"), ("value", "$W_v$")]:
        rows = [d[str(l)][w] for l in range(12)]
        n = len(rows)
        mean = {k: sum(r[k] for r in rows) / n for k in rows[0]}
        out[w] = dict(mean=mean, rows=rows)
        lines.append(f"{label} & {mean['fro_w']:.1f} & {mean['fro_dw']:.2f} & "
                     f"{mean['proj_dw']:.2f} & {mean['proj_w']:.2f} & {mean['proj_rand']:.2f} & "
                     f"{mean['amp']:.1f}$\\times$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/official_proj.tex", "w").write("\n".join(lines))
    return out


def efficiency_table():
    p = f"{ROOT}/analysis/efficiency.json"
    if not os.path.exists(p):
        open(f"{OUT}/efficiency.tex", "w").write(
            "\\begin{tabular}{l c c c c}\n\\toprule\n"
            "设置 & Base (ms) & LoRA 未合并 (ms) & LoRA 已合并 (ms) & 开销 \\\\\n"
            "\\midrule\n\\multicolumn{5}{c}{（效率测量运行中）}\\\\\n\\bottomrule\n\\end{tabular}\n")
        return None
    d = load(p)
    lines = [r"\begin{tabular}{l c c c c}", r"\toprule",
             r"设置 & Base (ms) & LoRA 未合并 (ms) & LoRA 已合并 (ms) & 开销 \\", r"\midrule"]
    for k, v in d["inference_ms"].items():
        b, u, m = v["base"], v["unmerged"], v["merged"]
        lines.append(f"{k.replace('_', ', ')} & {b:.2f} & {u:.2f} & {m:.2f} & "
                     f"{(u/b-1)*100:+.0f}\\% / {(m/b-1)*100:+.0f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/efficiency.tex", "w").write("\n".join(lines))
    t = d["train_step"]
    return dict(inf=d["inference_ms"], **{k: t[k] for k in t})


def initmnli_table():
    vals = {}
    for t in ["mrpc", "rte", "stsb"]:
        base = f"{ROOT}/glue/{t}_roberta-base_lora_r8_qv_s0/result.json"
        init = f"{ROOT}/glue/initmnli_{t}/result.json"
        off = f"{ROOT}/glue/initoff_{t}/result.json"
        if os.path.exists(base):
            d = dict(base=load(base)["best_score"] * 100)
            if os.path.exists(init):
                d["mnli"] = load(init)["best_score"] * 100
            if os.path.exists(off):
                d["off"] = load(off)["best_score"] * 100
            vals[t] = d
    return vals


def official():
    p = f"{ROOT}/analysis/official_mnli.json"
    return load(p) if os.path.exists(p) else None


def macros(glue, rank, target, e2e, analysis, eff, initvals=None, off=None, offproj=None):
    m = []
    if glue:
        m.append(f"\\newcommand{{\\GlueOursAvg}}{{{glue['avg_ours']:.1f}}}")
        m.append(f"\\newcommand{{\\GluePaperAvg}}{{{glue['avg_paper']:.1f}}}")
        m.append(f"\\newcommand{{\\GlueGap}}{{{glue['avg_paper']-glue['avg_ours']:.1f}}}")
        m.append(f"\\newcommand{{\\GlueNTasks}}{{{glue['n_tasks']}}}")
        if glue.get("avg_ft"):
            m.append(f"\\newcommand{{\\GlueFtAvg}}{{{glue['avg_ft']:.1f}}}")
        for t, v in glue["per_task"].items():
            # control sequences are letters only, so "SST2" -> "SSTTwo"
            m.append(f"\\newcommand{{\\Glue{t.upper().replace('2', 'Two')}}}{{{v:.1f}}}")
        for t, v in glue.get("per_task_ft", {}).items():
            m.append(f"\\newcommand{{\\GlueFT{t.upper().replace('2', 'Two')}}}{{{v:.1f}}}")
    if rank:
        best_r = max(rank, key=rank.get)
        m.append(f"\\newcommand{{\\RankBest}}{{{best_r}}}")
        m.append(f"\\newcommand{{\\RankBestScore}}{{{rank[best_r]:.1f}}}")
        m.append(f"\\newcommand{{\\RankOneScore}}{{{rank.get(1, float('nan')):.1f}}}")
        words = {1: "One", 2: "Two", 4: "Four", 8: "Eight", 64: "SixtyFour"}
        for r, v in rank.items():
            if r in words:
                m.append(f"\\newcommand{{\\Rank{words[r]}}}{{{v:.1f}}}")
    if target:
        bt = max(target, key=target.get)
        m.append(f"\\newcommand{{\\TgtBest}}{{{bt}}}")
        m.append(f"\\newcommand{{\\TgtBestScore}}{{{target[bt]:.1f}}}")
        for cfg, v in target.items():
            m.append(f"\\newcommand{{\\Tgt{cfg.upper()}}}{{{v:.1f}}}")
    if e2e:
        for meth, key in [("lora", "EtwoLora"), ("ft", "EtwoFt")]:
            if meth in e2e:
                met = e2e[meth]
                m.append(f"\\newcommand{{\\{key}Bleu}}{{{met['BLEU']:.1f}}}")
                m.append(f"\\newcommand{{\\{key}Meteor}}{{{met['METEOR']:.1f}}}")
                m.append(f"\\newcommand{{\\{key}Rouge}}{{{met['ROUGE-L']:.1f}}}")
                m.append(f"\\newcommand{{\\{key}Cider}}{{{met['CIDEr']:.2f}}}")
    if analysis:
        q, v = analysis["query"], analysis["value"]
        m.append(f"\\newcommand{{\\AmpQ}}{{{q['fro_a']/q['proj_dw']:.1f}}}")
        m.append(f"\\newcommand{{\\AmpV}}{{{v['fro_a']/v['proj_dw']:.1f}}}")
        m.append(f"\\newcommand{{\\AmpQB}}{{{q['fro_b']/q['proj_dw_b']:.1f}}}")
        m.append(f"\\newcommand{{\\AmpVB}}{{{v['fro_b']/v['proj_dw_b']:.1f}}}")
        m.append(f"\\newcommand{{\\PhiQ}}{{{q['phi'][0][0]:.2f}}}")
        m.append(f"\\newcommand{{\\PhiV}}{{{v['phi'][0][0]:.2f}}}")
        m.append(f"\\newcommand{{\\PhiQFull}}{{{q['phi'][-1][-1]:.2f}}}")
        m.append(f"\\newcommand{{\\PhiVFull}}{{{v['phi'][-1][-1]:.2f}}}")
    if eff:
        t = eff
        m.append(f"\\newcommand{{\\LoraPeakMem}}{{{t['lora']['peak_mem_mb']:.0f}}}")
        m.append(f"\\newcommand{{\\FtPeakMem}}{{{t['ft']['peak_mem_mb']:.0f}}}")
        m.append(f"\\newcommand{{\\LoraStepMs}}{{{t['lora']['step_ms']:.1f}}}")
        m.append(f"\\newcommand{{\\FtStepMs}}{{{t['ft']['step_ms']:.1f}}}")
        m.append(f"\\newcommand{{\\LoraMemPct}}{{{100*t['lora']['peak_mem_mb']/t['ft']['peak_mem_mb']:.0f}}}")
    if initvals:
        for t, v in initvals.items():
            m.append(f"\\newcommand{{\\Init{t.capitalize()}Base}}{{{v['base']:.1f}}}")
            if "mnli" in v:
                m.append(f"\\newcommand{{\\Init{t.capitalize()}Mnli}}{{{v['mnli']:.1f}}}")
            if "off" in v:
                m.append(f"\\newcommand{{\\Init{t.capitalize()}Off}}{{{v['off']:.1f}}}")
    if offproj:
        q, v = offproj["query"]["mean"], offproj["value"]["mean"]
        m.append(f"\\newcommand{{\\OffProjAmpQ}}{{{q['amp']:.1f}}}")
        m.append(f"\\newcommand{{\\OffProjAmpV}}{{{v['amp']:.1f}}}")
        m.append(f"\\newcommand{{\\OffProjRatioQ}}{{{q['proj_dw']/q['proj_rand']:.1f}}}")
        m.append(f"\\newcommand{{\\OffProjRatioV}}{{{v['proj_dw']/v['proj_rand']:.1f}}}")
        m.append(f"\\newcommand{{\\OffProjWQ}}{{{q['proj_w']:.1f}}}")
        m.append(f"\\newcommand{{\\OffProjDWQ}}{{{q['proj_dw']:.1f}}}")
    if off:
        m.append(f"\\newcommand{{\\OfficialM}}{{{off['scores']['matched']:.1f}}}")
        m.append(f"\\newcommand{{\\OfficialMM}}{{{off['scores']['mismatched']:.1f}}}")
        m.append(f"\\newcommand{{\\OfficialAvg}}{{{off['avg']:.1f}}}")
    open(f"{OUT}/numbers.tex", "w").write("\n".join(m) + "\n")
    return m


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    g = glue_table()
    r = rank_table()
    t = target_table()
    e = e2e_table()
    a = analysis_table()
    oj = official_proj_table()
    ef = efficiency_table()
    iv = initmnli_table()
    off = official()
    macros(g, r, t, e, a, ef, iv, off, oj)
    print("tables written")
