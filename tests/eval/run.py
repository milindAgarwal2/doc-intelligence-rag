"""RAG evaluation harness — retrieval recall@k + confidence calibration.

Offline, maintainer-facing, EMBEDDING-ONLY (no LLM / no API cost). It drives the
SAME retrieval + confidence functions the chatbot uses (chatBOT_engine), so the
numbers reflect real app behaviour.

Run:
    .venv/bin/python -m tests.eval.run
    .venv/bin/python -m tests.eval.run --json report.json
    .venv/bin/python -m tests.eval.run --dataset path/to/dataset.json

What it measures
----------------
1. Retrieval recall@k — for each in-corpus question, is the ground-truth page
   among the top-k retrieved chunks? Reports per-question rank + overall recall.
2. Confidence calibration — distribution of confidence_probe() scores for
   in-corpus vs negative (off-topic) questions, the separating gap, and a
   SUGGESTED threshold for the #3 low-confidence banner.

Adapts to a changing store: a question whose expect_file is NOT currently
embedded is SKIPPED (reported, not failed), so the dataset can grow with new
documents. Negative questions (expect_file=null) are always evaluated (they
need no embedded doc) and feed calibration.

Exit code: 0 if recall@k >= recall_floor (over evaluated in-corpus questions),
else 1 — suitable for CI gating.
"""
import argparse
import json
import os
import sys
from datetime import datetime

# Import the real app functions so eval == runtime behaviour.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import chatBOT_engine as C  # noqa: E402
from embeddingEngine import get_vectorstore, list_embedded_documents  # noqa: E402

DEFAULT_DATASET = os.path.join(_HERE, "dataset.json")


def _basename(p):
    return os.path.basename(str(p)) if p else ""


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def embedded_basenames():
    """Set of basenames currently embedded, so we can skip questions whose
    document isn't present."""
    vs = get_vectorstore()
    return {_basename(d) for d in list_embedded_documents(vs)}


def retrieved_pages(question, k):
    """Return the ordered list of (basename, page) for the top-k retrieved
    chunks, using the chatbot's own MMR retriever."""
    docs = C.retrieve_docs(question)[:k]
    out = []
    for d in docs:
        out.append((_basename(d.metadata.get("file_name")),
                    d.metadata.get("pagenumber")))
    return out


def rank_of(expect_file, expect_page, pages):
    """1-based rank of the first retrieved chunk matching the ground-truth
    (basename + page), or None if not present."""
    want_file = _basename(expect_file)
    for i, (f, p) in enumerate(pages, start=1):
        try:
            same_page = int(p) == int(expect_page)
        except (TypeError, ValueError):
            same_page = str(p) == str(expect_page)
        if f == want_file and same_page:
            return i
    return None


def evaluate(dataset):
    k = int(dataset.get("k", 8))
    floor = float(dataset.get("recall_floor", 0.75))
    questions = dataset.get("questions", [])
    present = embedded_basenames()

    recall_rows = []      # in-corpus evaluated
    skipped_rows = []     # in-corpus but doc not embedded
    in_scores = []        # confidence scores for in-corpus
    neg_rows = []         # negatives
    neg_scores = []

    for item in questions:
        q = item.get("q", "")
        exp_file = item.get("expect_file")
        score = C.confidence_probe(q)

        if exp_file is None:
            neg_scores.append(score)
            neg_rows.append({"q": q, "score": score})
            continue

        if _basename(exp_file) not in present:
            skipped_rows.append({"q": q, "expect_file": _basename(exp_file)})
            continue

        pages = retrieved_pages(q, k)
        r = rank_of(exp_file, item.get("expect_page"), pages)
        in_scores.append(score)
        recall_rows.append({
            "q": q,
            "expect": f"{_basename(exp_file)} p.{item.get('expect_page')}",
            "rank": r,
            "found": r is not None,
            "score": score,
        })

    found = sum(1 for r in recall_rows if r["found"])
    evaluated = len(recall_rows)
    recall = (found / evaluated) if evaluated else None

    return {
        "k": k,
        "floor": floor,
        "recall": recall,
        "found": found,
        "evaluated": evaluated,
        "recall_rows": recall_rows,
        "skipped_rows": skipped_rows,
        "neg_rows": neg_rows,
        "in_scores": [s for s in in_scores if s is not None],
        "neg_scores": [s for s in neg_scores if s is not None],
        "present": sorted(present),
    }


def _stats(scores):
    if not scores:
        return None
    return {"min": min(scores), "mean": sum(scores) / len(scores), "max": max(scores)}


def suggest_threshold(in_scores, neg_scores):
    """Suggest a #3 banner threshold from the gap between in-corpus and negative
    score distributions. Returns (threshold, explanation)."""
    si, sn = _stats(in_scores), _stats(neg_scores)
    if not si or not sn:
        return None, "need both in-corpus and negative scores to calibrate"
    lo_in = si["min"]       # worst in-corpus score (should stay ABOVE threshold)
    hi_neg = sn["max"]      # best negative score (should stay BELOW threshold)
    if lo_in > hi_neg:
        thr = round((lo_in + hi_neg) / 2.0, 2)
        return thr, (f"clean gap {hi_neg:.2f}..{lo_in:.2f}; midpoint {thr:.2f} "
                     "separates all evaluated cases")
    # Overlap: no single value separates them cleanly.
    return None, (f"NO clean gap — in-corpus min {lo_in:.2f} <= negative max "
                  f"{hi_neg:.2f}; scores alone can't separate on this corpus "
                  "(keep the banner conservative / non-blocking)")


def _fmt_score(s):
    return f"{s:.2f}" if isinstance(s, (int, float)) else "  -"


def print_report(res):
    print("=" * 64)
    print("RAG EVALUATION —", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Store: ./chroma_Name   k={res['k']}   docs embedded: {len(res['present'])}")
    for d in res["present"]:
        print("   •", d)
    print("=" * 64)

    print("\nRETRIEVAL RECALL@k")
    if res["recall_rows"]:
        for r in res["recall_rows"]:
            status = f"FOUND @rank {r['rank']}" if r["found"] else "MISS       "
            mark = "✓" if r["found"] else "✗"
            print(f"  {mark} {r['q'][:34]:34s} expect {r['expect']:34s} "
                  f"{status}  score {_fmt_score(r['score'])}")
        rc = res["recall"]
        print(f"\n  recall@{res['k']} = {rc:.2f}  ({res['found']}/{res['evaluated']} "
              "in-corpus questions)")
    else:
        print("  (no in-corpus questions evaluated — none of their docs are embedded)")

    if res["skipped_rows"]:
        print("\n  SKIPPED (document not embedded):")
        for s in res["skipped_rows"]:
            print(f"    - {s['q'][:40]:40s} needs {s['expect_file']}")

    print("\nCONFIDENCE CALIBRATION")
    si, sn = _stats(res["in_scores"]), _stats(res["neg_scores"])
    if si:
        print(f"  in-corpus scores:  min {si['min']:.2f}  mean {si['mean']:.2f}  "
              f"max {si['max']:.2f}  (n={len(res['in_scores'])})")
    else:
        print("  in-corpus scores:  (none)")
    if sn:
        print(f"  off-topic scores:  min {sn['min']:.2f}  mean {sn['mean']:.2f}  "
              f"max {sn['max']:.2f}  (n={len(res['neg_scores'])})")
    else:
        print("  off-topic scores:  (none)")
    thr, why = suggest_threshold(res["in_scores"], res["neg_scores"])
    if thr is not None:
        print(f"  → SUGGESTED low_confidence_threshold = {thr}")
    print(f"  → {why}")

    print("\nNEGATIVE CASES (should be low-confidence)")
    for n in res["neg_rows"]:
        verdict = ""
        if thr is not None and isinstance(n["score"], (int, float)):
            verdict = " ✓ would banner" if n["score"] < thr else " ✗ would NOT banner"
        print(f"  {n['q'][:40]:40s} score {_fmt_score(n['score'])}{verdict}")

    print("\n" + "=" * 64)
    rc = res["recall"]
    if rc is None:
        print("RESULT: no in-corpus questions evaluated → PASS (nothing to gate)")
        return 0
    passed = rc >= res["floor"]
    print(f"RESULT: recall@{res['k']} {rc:.2f} "
          f"{'≥' if passed else '<'} floor {res['floor']:.2f}  →  "
          f"{'PASS' if passed else 'FAIL'}")
    print("=" * 64)
    return 0 if passed else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="RAG retrieval/confidence eval harness")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--json", dest="json_out", default=None,
                    help="also write the raw results to this JSON path")
    args = ap.parse_args(argv)

    dataset = load_dataset(args.dataset)
    res = evaluate(dataset)
    code = print_report(res)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"\n(raw results written to {args.json_out})")

    return code


if __name__ == "__main__":
    sys.exit(main())
