"""NLG metrics used for the E2E reproduction (BLEU / METEOR / ROUGE-L / CIDEr).

The E2E test split gives several human references per meaning representation, so
each example must be scored against *all* of its references: BLEU with a
multi-reference corpus, ROUGE-L/METEOR with the best-matching reference, and
CIDEr with the full reference set.  ``compute_all_multi`` implements this;
``compute_all`` is a thin single-reference wrapper kept for convenience.
"""


def compute_all_multi(preds, refs_per_example, verbose=True):
    """preds: list of strings; refs_per_example: list of lists of strings."""
    assert len(preds) == len(refs_per_example)
    assert all(len(rs) > 0 for rs in refs_per_example)
    out = {}

    # BLEU: sacrebleu wants one stream per reference index, each of length n.
    # Reference sets have variable size, so pad shorter sets by cycling their refs.
    import sacrebleu
    k_max = max(len(rs) for rs in refs_per_example)
    streams = [[rs[k] if k < len(rs) else rs[0] for rs in refs_per_example]
               for k in range(k_max)]
    out["BLEU"] = float(sacrebleu.corpus_bleu(preds, streams).score)

    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        vals = [max(scorer.score(r, p)["rougeL"].fmeasure for r in rs)
                for p, rs in zip(preds, refs_per_example)]
        out["ROUGE-L"] = float(sum(vals) / len(vals)) * 100
    except Exception as e:  # pragma: no cover
        out["ROUGE-L"] = float("nan")
        if verbose:
            print("ROUGE-L failed:", e)

    try:
        from nltk.translate.meteor_score import meteor_score
        vals = [meteor_score([r.split() for r in rs], p.split())
                for p, rs in zip(preds, refs_per_example)]
        out["METEOR"] = float(sum(vals) / len(vals)) * 100
    except Exception as e:  # pragma: no cover
        out["METEOR"] = float("nan")
        if verbose:
            print("METEOR failed:", e)

    try:
        from pycocoevalcap.cider.cider import Cider
        gts = {i: list(rs) for i, rs in enumerate(refs_per_example)}
        res = {i: [p] for i, p in enumerate(preds)}
        score, _ = Cider().compute_score(gts, res)
        out["CIDEr"] = float(score)  # paper reports CIDEr on its native ~0-10 scale
    except Exception as e:  # pragma: no cover
        out["CIDEr"] = float("nan")
        if verbose:
            print("CIDEr failed:", e)
    return out


def compute_all(preds, refs):
    """Convenience wrapper for a single reference per example."""
    return compute_all_multi(preds, [[r] for r in refs])
