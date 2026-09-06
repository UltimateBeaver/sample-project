"""
Phase 1 – Embedding-Space Similarity Distribution Analysis
==========================================================
Demonstrates that the same cosine threshold carries different semantic content
for different embedding models, justifying per-model threshold recalibration.

Workflow
--------
Step 1  With Model A (e.g. nomic) embedding server running:
    python calibration_phase1_distribution.py --model nomic

Step 2  Swap embedding backend to Model B (e.g. qwen), then:
    python calibration_phase1_distribution.py --model qwen

Step 3  Generate comparison figure + statistics table:
    python calibration_phase1_distribution.py --compare
"""

import ast
import sys, json, argparse
import numpy as np
import matplotlib.pyplot as plt
import requests
from pathlib import Path
import pandas as pd
from env_config import eval_input_dataset_path, column_name_quintuples_ground_truth, eval_output_results_path, llamacpp_embed_base

current_file = Path(__file__).resolve()
PREFIX_DIR = current_file.parent.parent.parent / "itext2kg_atom"
CALIB_DIR  = PREFIX_DIR / eval_output_results_path / "calibration"
EMBED_URL  = llamacpp_embed_base   # e.g. "http://localhost:8080"
BATCH_SIZE = 32

DATASET_PATH = PREFIX_DIR / eval_input_dataset_path

# ── static entity-name corpus (Italian financial-news domain) ─────────────────
# ~50 strings: deliberately mixes abbreviations, cross-lingual equivalents,
# and partial-name variants to reflect real KG entity surface-form variation.
STATIC_CORPUS = [
    # People
    "Mario Draghi", "Christine Lagarde", "Ursula von der Leyen",
    "Sergio Mattarella", "Giorgia Meloni", "Mario Monti", "Pier Carlo Padoan",
    # Institutions
    "European Central Bank", "ECB", "Banca Centrale Europea",
    "International Monetary Fund", "IMF", "Fondo Monetario Internazionale",
    "Bank of Italy", "Banca d'Italia", "Bankitalia",
    "European Commission", "Commissione Europea",
    "World Bank", "Banca Mondiale",
    # Geopolitical
    "Italy", "Italia", "Germany", "France", "Francia",
    "United States", "USA", "European Union", "EU", "Unione Europea",
    # Economic concepts
    "inflation", "inflazione", "interest rate", "tasso di interesse",
    "GDP", "gross domestic product", "PIL",
    "recession", "recessione", "economic growth", "crescita economica",
    # Finance instruments
    "BTP", "Italian government bond", "spread", "spread BTP-Bund",
    "FTSE MIB", "Borsa Italiana", "UniCredit", "Intesa Sanpaolo",
    # Events / topics
    "COVID-19", "Covid-19", "coronavirus", "pandemic", "pandemia",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def embed_texts(texts: list) -> np.ndarray:
    """Embed *texts* via the running OpenAI-compatible /embeddings endpoint."""
    vecs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        r = requests.post(
            f"{EMBED_URL}/embeddings",
            json={"model": "default", "input": batch},
            timeout=300,
        )
        r.raise_for_status()
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        vecs.extend(d["embedding"] for d in data)
    return np.array(vecs, dtype=np.float32)


def pairwise_cosine_upper(emb: np.ndarray) -> np.ndarray:
    """Return upper-triangle pairwise cosine similarities (diagonal excluded)."""
    norms  = np.linalg.norm(emb, axis=1, keepdims=True)
    normed = emb / np.clip(norms, 1e-10, None)
    mat    = normed @ normed.T
    idx    = np.triu_indices(len(emb), k=1)
    return mat[idx].astype(np.float32)


def compute_stats(sims: np.ndarray) -> dict:
    return {
        "n_pairs":     int(len(sims)),
        "mean":        float(np.mean(sims)),
        "std":         float(np.std(sims)),
        "p10":         float(np.percentile(sims, 10)),
        "p50":         float(np.percentile(sims, 50)),
        "p90":         float(np.percentile(sims, 90)),
        "frac_gt_050": float(np.mean(sims > 0.50)),
        "frac_gt_070": float(np.mean(sims > 0.70)),
    }


def load_corpus() -> list:
    """Try to extract entity names from the evaluation dataset; fall back to STATIC_CORPUS."""
    try:
        df = pd.read_pickle(DATASET_PATH)
        names = set()
        for cell in df[column_name_quintuples_ground_truth].dropna():
            if isinstance(cell, str):
                cell = ast.literal_eval(cell)
            for q in cell:
                if isinstance(q, dict):
                    names.update([str(q.get("subject", "")).strip(),
                                  str(q.get("object",  "")).strip()])
                elif len(q) >= 3:
                    names.update([str(q[0]).strip(), str(q[2]).strip()])
        corpus = [n for n in names if len(n) > 1][:600]
        if len(corpus) >= 20:
            print(f"[corpus] Loaded {len(corpus)} entity names from dataset.")
            return corpus
        else:
            print(f"[corpus] Not enough sample provided [{len(corpus)} samples]. Using static corpus")
    except Exception as e:
        print(f"[corpus] Dataset load failed ({e}). Using static corpus.")
    return STATIC_CORPUS


# ── compute mode ──────────────────────────────────────────────────────────────

def run_compute(model_name: str) -> None:
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus()
    print(f"Embedding {len(corpus)} strings with backend '{model_name}' …")

    emb  = embed_texts(corpus)
    sims = pairwise_cosine_upper(emb)
    st   = compute_stats(sims)

    print(f"  mean={st['mean']:.4f}  std={st['std']:.4f}  "
          f"p90={st['p90']:.4f}  frac>0.70={st['frac_gt_070']:.3f}")

    np.save(CALIB_DIR / f"sims_{model_name}.npy", sims)
    (CALIB_DIR / f"stats_{model_name}.json").write_text(json.dumps(st, indent=2))

    # Single-model distribution plot (quick inspection)
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=150)
    ax.hist(sims, bins=60, density=True, alpha=0.4, color="#1f77b4", edgecolor="none")
    try:
        from scipy.stats import gaussian_kde
        xs = np.linspace(0, 1, 500)
        ax.plot(xs, gaussian_kde(sims)(xs), color="#1f77b4", linewidth=2)
    except ImportError:
        pass  # scipy optional; histogram alone is sufficient
    ax.axvline(0.70, color="red",    linestyle="--", linewidth=1.3, label="τ = 0.70")
    ax.axvline(0.50, color="orange", linestyle="--", linewidth=1.3, label="τ = 0.50")
    ax.set_xlabel("Cosine similarity"); ax.set_ylabel("Density")
    ax.set_title(f"Pairwise similarity distribution – {model_name}")
    ax.set_xlim(0, 1); ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(CALIB_DIR / f"dist_{model_name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {CALIB_DIR}/dist_{model_name}.{{png,pdf,json}}")


# ── compare mode ──────────────────────────────────────────────────────────────

def run_compare(model_a: str, model_b: str) -> None:
    sims_a = np.load(CALIB_DIR / f"sims_{model_a}.npy")
    sims_b = np.load(CALIB_DIR / f"sims_{model_b}.npy")
    st_a   = json.loads((CALIB_DIR / f"stats_{model_a}.json").read_text())
    st_b   = json.loads((CALIB_DIR / f"stats_{model_b}.json").read_text())

    # Comparison KDE plot
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                         "font.size": 11, "axes.labelsize": 12, "legend.fontsize": 10})
    fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=300)

    for sims, name, col in [(sims_a, model_a, "#1f77b4"),
                             (sims_b, model_b, "#ff7f0e")]:
        ax.hist(sims, bins=60, density=True, alpha=0.15, color=col, edgecolor="none")
        try:
            from scipy.stats import gaussian_kde
            xs = np.linspace(0, 1, 500)
            ax.plot(xs, gaussian_kde(sims)(xs), color=col, linewidth=2, label=name)
        except ImportError:
            ax.hist(sims, bins=60, density=True, alpha=0.5, color=col,
                    edgecolor="none", label=name)

    ax.axvline(0.70, color="red",   linestyle="--", linewidth=1.3, label="τ = 0.70")
    ax.axvline(0.50, color="green", linestyle="--", linewidth=1.3, label="τ = 0.50")
    ax.set_xlabel("Cosine similarity"); ax.set_ylabel("Density")
    ax.set_title("Pairwise similarity distributions")
    ax.set_xlim(0, 1); ax.legend()
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(CALIB_DIR / f"dist_comparison.{ext}", bbox_inches="tight")
    plt.close(fig)

    # Statistics table
    print("\n── Embedding Space Geometry Comparison ──────────────────────────────")
    print(f"{'Statistic':<28} {model_a:>16} {model_b:>16}")
    print("─" * 62)
    for label, key, fmt in [
        ("N pairs",            "n_pairs",     "d"),
        ("Mean pairwise sim",  "mean",        ".4f"),
        ("Std",                "std",         ".4f"),
        ("10th percentile",    "p10",         ".4f"),
        ("Median",             "p50",         ".4f"),
        ("90th percentile",    "p90",         ".4f"),
        ("Fraction > 0.50",    "frac_gt_050", ".3f"),
        ("Fraction > 0.70",    "frac_gt_070", ".3f"),
    ]:
        print(f"{label:<28} {format(st_a[key], fmt):>16} {format(st_b[key], fmt):>16}")

    # Key thesis insight: find the threshold on model B that yields the same
    # merge-rate as τ=0.70 on model A → empirical recalibration evidence.
    target = st_a["frac_gt_070"]
    for tau in np.arange(0.30, 0.90, 0.01):
        if np.mean(sims_b > tau) <= target:
            print(f"\n→ τ = {tau:.2f} on {model_b} yields merge-rate ≈ "
                  f"τ = 0.70 on {model_a}  ({target:.1%} of pairs merged)")
            break

    print(f"\nSaved → {CALIB_DIR}/dist_comparison.{{png,pdf}}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Phase 1: similarity distribution analysis")
    ap.add_argument("--model",   default="qwen",
                    help="Name tag for the current embedding backend (used in file names)")
    ap.add_argument("--compare", action="store_true",
                    help="Compare two already-computed models (requires prior --model runs)")
    ap.add_argument("--model-a", default="nomic", help="First model name (for --compare)")
    ap.add_argument("--model-b", default="qwen",  help="Second model name (for --compare)")
    args = ap.parse_args()

    if args.compare:
        run_compare(args.model_a, args.model_b)
    else:
        run_compute(args.model)


if __name__ == "__main__":
    main()
