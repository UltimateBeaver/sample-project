"""
Phase 2 – Threshold Calibration via Named-Entity Pair Benchmark
================================================================
Empirically identifies the optimal similarity threshold for a given embedding
model by evaluating it on a manually-labelled set of entity-name pairs drawn
from the Italian financial-news domain.

For each threshold τ the script computes Precision, Recall, and F1 on the
binary task "should these two entity names be merged in the KG?"  The chosen
operating point is the τ that maximises F1.

Workflow
--------
With Model A (e.g. nomic) embedding server running:
    python calibration_phase2_threshold_sweep.py --model nomic --operating-point 0.70

Swap to Model B (e.g. qwen) and run:
    python calibration_phase2_threshold_sweep.py --model qwen  --operating-point 0.50

Overlay both F1 curves:
    python calibration_phase2_threshold_sweep.py --compare
"""

import sys, json, argparse
import numpy as np
import matplotlib.pyplot as plt
import requests
from pathlib import Path

from env_config import eval_output_results_path, llamacpp_embed_base

current_file = Path(__file__).resolve()
PREFIX_DIR = current_file.parent.parent.parent / "itext2kg_atom"
CALIB_DIR  = PREFIX_DIR / eval_output_results_path / "calibration"
EMBED_URL  = llamacpp_embed_base   # e.g. "http://localhost:8080"
BATCH_SIZE = 32
THRESHOLDS = np.round(np.arange(0.30, 0.86, 0.05), 2)  # [0.30, 0.35, … 0.85]

# ── labeled benchmark ─────────────────────────────────────────────────────────
# 30 entity-name pairs: 15 positive (same real-world entity, different surface
# form) + 15 negative (clearly distinct entities).  Surface-form variation
# covers abbreviations, cross-lingual equivalents, and partial names —
# all representative of what the LLM translation step produces.
LABELED_PAIRS = [
    # ── Positive pairs (label = 1) ────────────────────────────────────────────
    ("European Central Bank",       "ECB",                          1),
    ("Banca d'Italia",              "Bank of Italy",                1),
    ("International Monetary Fund", "IMF",                          1),
    ("United States",               "USA",                          1),
    ("European Union",              "EU",                           1),
    ("COVID-19",                    "Covid-19",                     1),
    ("gross domestic product",      "GDP",                          1),
    ("Mario Draghi",                "Draghi",                       1),
    ("Giorgia Meloni",              "Meloni",                       1),
    ("Christine Lagarde",           "Lagarde",                      1),
    ("inflation",                   "inflazione",                   1),
    ("Italy",                       "Italia",                       1),
    ("interest rate",               "tasso di interesse",           1),
    ("recession",                   "recessione",                   1),
    ("BTP",                         "Italian government bond",      1),
    # Italian politicians — surname-only variants
    ("Sergio Mattarella",                        "Mattarella",                           1),
    ("Mario Monti",                              "Monti",                                1),
    ("Pier Carlo Padoan",                        "Padoan",                               1),
    ("Silvio Berlusconi",                        "Berlusconi",                           1),
    ("Matteo Renzi",                             "Renzi",                                1),
    ("Giuseppe Conte",                           "Conte",                                1),
    ("Matteo Salvini",                           "Salvini",                              1),
    ("Luigi Di Maio",                            "Di Maio",                              1),
    ("Ignazio Visco",                            "Visco",                                1),
    ("Romano Prodi",                             "Prodi",                                1),
    ("Enrico Letta",                             "Letta",                                1),
    ("Roberto Gualtieri",                        "Gualtieri",                            1),
    # European and international leaders — surname-only variants
    ("Ursula von der Leyen",                     "von der Leyen",                        1),
    ("Emmanuel Macron",                          "Macron",                               1),
    ("Olaf Scholz",                              "Scholz",                               1),
    ("Jerome Powell",                            "Powell",                               1),
    ("Janet Yellen",                             "Yellen",                               1),
    ("Kristalina Georgieva",                     "Georgieva",                            1),
    ("Paolo Gentiloni",                          "Gentiloni",                            1),
    ("Angela Merkel",                            "Merkel",                               1),
    # Financial institutions — abbreviation expansions
    ("Federal Reserve",                          "Fed",                                  1),
    ("Federal Reserve System",                   "Federal Reserve",                      1),
    ("European Parliament",                      "EP",                                   1),
    ("European Commission",                      "EC",                                   1),
    ("World Trade Organization",                 "WTO",                                  1),
    ("Organisation for Economic Co-operation and Development", "OECD",                  1),
    ("Bank for International Settlements",       "BIS",                                  1),
    ("European Stability Mechanism",             "ESM",                                  1),
    ("European Investment Bank",                 "EIB",                                  1),
    ("European Banking Authority",               "EBA",                                  1),
    ("Financial Stability Board",                "FSB",                                  1),
    ("Deutsche Bundesbank",                      "Bundesbank",                           1),
    ("Bank of England",                          "BoE",                                  1),
    ("Bank of Japan",                            "BoJ",                                  1),
    ("People's Bank of China",                   "PBOC",                                 1),
    ("European Systemic Risk Board",             "ESRB",                                 1),
    # Countries and regions — alternative names / abbreviations
    ("United Kingdom",                           "UK",                                   1),
    ("Great Britain",                            "United Kingdom",                       1),
    ("China",                                    "People's Republic of China",           1),
    ("Russia",                                   "Russian Federation",                   1),
    ("Greece",                                   "Hellenic Republic",                    1),
    ("Eurozone",                                 "euro area",                            1),
    ("United States of America",                 "America",                              1),
    # Economic and financial concepts — abbreviations and synonyms
    ("quantitative easing",                      "QE",                                   1),
    ("consumer price index",                     "CPI",                                  1),
    ("producer price index",                     "PPI",                                  1),
    ("fiscal deficit",                           "budget deficit",                       1),
    ("sovereign debt",                           "government debt",                      1),
    ("non-performing loans",                     "NPL",                                  1),
    ("impaired loans",                           "bad loans",                            1),
    ("purchasing power parity",                  "PPP",                                  1),
    ("foreign direct investment",                "FDI",                                  1),
    ("balance of payments",                      "BoP",                                  1),
    ("austerity measures",                       "fiscal consolidation",                 1),
    ("quantitative tightening",                  "QT",                                   1),
    ("basis points",                             "bps",                                  1),
    ("gross national product",                   "GNP",                                  1),
    ("unemployment rate",                        "jobless rate",                         1),
    ("LIBOR",                                    "London Interbank Offered Rate",        1),
    ("EURIBOR",                                  "Euro Interbank Offered Rate",          1),
    ("Stability and Growth Pact",                "SGP",                                  1),
    ("Bund",                                     "German government bond",               1),
    ("gilt",                                     "UK government bond",                   1),
    ("initial public offering",                  "IPO",                                  1),
    ("mergers and acquisitions",                 "M&A",                                  1),
    ("return on equity",                         "ROE",                                  1),
    ("earnings per share",                       "EPS",                                  1),
    ("credit default swap",                      "CDS",                                  1),
    ("collateralized debt obligation",           "CDO",                                  1),
    ("asset-backed security",                    "ABS",                                  1),
    ("Next Generation EU",                       "European Recovery Fund",               1),
    ("Recovery and Resilience Facility",         "RRF",                                  1),
    ("Treaty of Maastricht",                     "Maastricht Treaty",                    1),
    ("asset purchase programme",                 "quantitative easing",                  1),
    # Italian companies — full name vs brand / acronym / former name
    ("Eni",                                      "Ente Nazionale Idrocarburi",           1),
    ("Enel",                                     "Ente Nazionale per l'Energia Elettrica", 1),
    ("Telecom Italia",                           "TIM",                                  1),
    ("Monte dei Paschi di Siena",                "MPS",                                  1),
    ("Cassa Depositi e Prestiti",                "CDP",                                  1),
    ("Leonardo",                                 "Leonardo-Finmeccanica",                1),
    ("Fiat Chrysler Automobiles",                "FCA",                                  1),
    ("UniCredit Group",                          "UniCredit",                            1),
    # International banks and companies — name variants
    ("Goldman Sachs",                            "Goldman",                              1),
    ("JPMorgan Chase",                           "JPMorgan",                             1),
    ("Deutsche Bank",                            "DB",                                   1),
    ("BlackRock Inc.",                           "BlackRock",                            1),
    ("Apple Inc.",                               "Apple",                                1),
    ("Microsoft Corporation",                    "Microsoft",                            1),
    ("Meta Platforms",                           "Facebook",                             1),
    ("Alphabet Inc.",                            "Google",                               1),
    # Italian political parties — abbreviations
    ("Five Star Movement",                       "M5S",                                  1),
    ("Lega Nord",                                "Lega",                                 1),
    ("Partito Democratico",                      "PD",                                   1),
    ("Forza Italia",                             "FI",                                   1),
    ("Fratelli d'Italia",                        "FdI",                                  1),
    # Events, crises, and treaties — alternative names
    ("Global Financial Crisis",                  "GFC",                                  1),
    ("COVID-19 pandemic",                        "coronavirus pandemic",                 1),
    ("European Debt Crisis",                     "Eurozone debt crisis",                 1),
    ("Brexit",                                   "British exit from the European Union", 1),
    ("2008 financial crisis",                    "Global Financial Crisis",              1),
    # Market indices — abbreviation / description
    ("Dow Jones Industrial Average",             "DJIA",                                 1),
    ("Standard and Poor's 500",                  "S&P 500",                              1),
    ("FTSE MIB",                                 "Milan Stock Exchange index",           1),

    # ── Negative pairs (label = 0) ────────────────────────────────────────────
    ("Mario Draghi",                "Christine Lagarde",            0),
    ("European Central Bank",       "International Monetary Fund",  0),
    ("Italy",                       "Germany",                      0),
    ("COVID-19",                    "inflation",                    0),
    ("GDP",                         "stock market",                 0),
    ("UniCredit",                   "Mediobanca",                   0),
    ("bond",                        "trade union",                  0),
    ("recession",                   "European Union",               0),
    ("Giorgia Meloni",              "Mario Monti",                  0),
    ("Silicon Valley Bank",         "Bank of Italy",                0),
    ("pandemic",                    "interest rate",                0),
    ("Confindustria",               "CGIL",                         0),
    ("BTP",                         "FTSE MIB",                     0),
    ("Sergio Mattarella",           "Ursula von der Leyen",         0),
    ("inflation",                   "GDP",                          0),
    # Politicians vs. politicians (different people)
    ("Mario Draghi",                             "Olaf Scholz",                          0),
    ("Christine Lagarde",                        "Janet Yellen",                         0),
    ("Giorgia Meloni",                           "Emmanuel Macron",                      0),
    ("Silvio Berlusconi",                        "Matteo Renzi",                         0),
    ("Giuseppe Conte",                           "Mario Monti",                          0),
    ("Jerome Powell",                            "Christine Lagarde",                    0),
    ("Angela Merkel",                            "Ursula von der Leyen",                 0),
    ("Ignazio Visco",                            "Mario Draghi",                         0),
    ("Romano Prodi",                             "Pier Carlo Padoan",                    0),
    ("Matteo Salvini",                           "Luigi Di Maio",                        0),
    ("Enrico Letta",                             "Giuseppe Conte",                       0),
    ("Kristalina Georgieva",                     "Christine Lagarde",                    0),
    ("Paolo Gentiloni",                          "Mario Draghi",                         0),
    # Politicians vs. institutions
    ("Mario Draghi",                             "European Central Bank",                0),
    ("Christine Lagarde",                        "Federal Reserve",                      0),
    ("Giorgia Meloni",                           "European Commission",                  0),
    ("Jerome Powell",                            "Bank of England",                      0),
    ("Janet Yellen",                             "Goldman Sachs",                        0),
    ("Angela Merkel",                            "Bundesbank",                           0),
    # Politicians vs. economic concepts
    ("Mario Draghi",                             "quantitative easing",                  0),
    ("Giorgia Meloni",                           "fiscal deficit",                       0),
    ("Jerome Powell",                            "consumer price index",                 0),
    ("Janet Yellen",                             "stock market",                         0),
    # Institutions vs. institutions (different organizations)
    ("European Central Bank",                    "Federal Reserve",                      0),
    ("Bank of Italy",                            "Deutsche Bank",                        0),
    ("International Monetary Fund",              "World Bank",                           0),
    ("European Parliament",                      "European Commission",                  0),
    ("OECD",                                     "WTO",                                  0),
    ("Goldman Sachs",                            "JPMorgan Chase",                       0),
    ("ESM",                                      "EIB",                                  0),
    ("Federal Reserve",                          "Bank of England",                      0),
    ("Monte dei Paschi di Siena",                "UniCredit",                            0),
    ("BlackRock",                                "Goldman Sachs",                        0),
    ("Intesa Sanpaolo",                          "BNP Paribas",                          0),
    ("European Banking Authority",               "Bank for International Settlements",   0),
    # Institutions vs. countries (institution ≠ the country it belongs to)
    ("European Union",                           "Germany",                              0),
    ("ECB",                                      "Switzerland",                          0),
    ("IMF",                                      "United States",                        0),
    ("World Bank",                               "China",                                0),
    ("Federal Reserve",                          "United States",                        0),
    ("Bank of Italy",                            "Italy",                                0),
    ("Bundesbank",                               "Germany",                              0),
    # Countries vs. countries
    ("Italy",                                    "France",                               0),
    ("Germany",                                  "China",                                0),
    ("United States",                            "Russia",                               0),
    ("Japan",                                    "South Korea",                          0),
    ("Spain",                                    "Portugal",                             0),
    ("Greece",                                   "Turkey",                               0),
    ("United Kingdom",                           "European Union",                       0),
    ("China",                                    "India",                                0),
    # Countries vs. economic concepts
    ("Italy",                                    "inflation",                            0),
    ("Germany",                                  "recession",                            0),
    ("China",                                    "quantitative easing",                  0),
    ("United States",                            "sovereign debt",                       0),
    # Concepts vs. concepts (related but distinct)
    ("inflation",                                "unemployment",                         0),
    ("monetary policy",                          "fiscal policy",                        0),
    ("sovereign debt",                           "equity",                               0),
    ("consumer price index",                     "producer price index",                 0),
    ("austerity",                                "stimulus",                             0),
    ("trade deficit",                            "budget surplus",                       0),
    ("bail-in",                                  "bail-out",                             0),
    ("LIBOR",                                    "EURIBOR",                              0),
    ("credit default swap",                      "government bond",                      0),
    ("IPO",                                      "M&A",                                  0),
    ("return on equity",                         "earnings per share",                   0),
    ("quantitative easing",                      "quantitative tightening",              0),
    ("Next Generation EU",                       "European Stability Mechanism",         0),
    # Financial instruments vs. instruments (different asset classes)
    ("BTP",                                      "Bund",                                 0),
    ("stock",                                    "bond",                                 0),
    ("futures",                                  "options",                              0),
    ("hedge fund",                               "pension fund",                         0),
    ("gilt",                                     "Bund",                                 0),
    ("S&P 500",                                  "FTSE MIB",                             0),
    ("DAX",                                      "CAC 40",                               0),
    ("DJIA",                                     "FTSE 100",                             0),
    # Companies vs. companies (different firms)
    ("Ferrari",                                  "Volkswagen",                           0),
    ("Eni",                                      "Shell",                                0),
    ("Enel",                                     "RWE",                                  0),
    ("Apple",                                    "Microsoft",                            0),
    ("UniCredit",                                "Deutsche Bank",                        0),
    ("Goldman Sachs",                            "BlackRock",                            0),
    ("Intesa Sanpaolo",                          "Mediobanca",                           0),
    # Concepts vs. institutions (a policy/metric ≠ the body that manages it)
    ("inflation",                                "European Central Bank",                0),
    ("quantitative easing",                      "Federal Reserve",                      0),
    ("recession",                                "IMF",                                  0),
    ("fiscal deficit",                           "European Commission",                  0),
    ("sovereign debt",                           "Bank of Italy",                        0),
    ("Recovery and Resilience Facility",         "European Central Bank",                0),
    # Events vs. events (different crises)
    ("Global Financial Crisis",                  "COVID-19 pandemic",                    0),
    ("Brexit",                                   "European Debt Crisis",                 0),
    ("COVID-19 pandemic",                        "European Debt Crisis",                 0),
    # Political parties vs. people / institutions
    ("Five Star Movement",                       "Giorgia Meloni",                       0),
    ("Partito Democratico",                      "Matteo Renzi",                         0),
    ("Lega",                                     "European Commission",                  0),
    ("Forza Italia",                             "Silvio Berlusconi",                    0),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def embed_texts(texts: list) -> np.ndarray:
    """Embed *texts* via the running OpenAI-compatible /v1/embeddings endpoint."""
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


def embed_all_unique(pairs: list) -> dict:
    """Embed every unique string in *pairs* once; return name → vector map."""
    unique = list({s for p in pairs for s in p[:2]})
    emb    = embed_texts(unique)
    return {name: emb[i] for i, name in enumerate(unique)}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def evaluate_at_threshold(pairs: list, emb_map: dict, tau: float) -> dict:
    tp = fp = fn = tn = 0
    for s1, s2, label in pairs:
        pred = int(cosine(emb_map[s1], emb_map[s2]) >= tau)
        if   label == 1 and pred == 1: tp += 1
        elif label == 0 and pred == 1: fp += 1
        elif label == 1 and pred == 0: fn += 1
        else:                          tn += 1
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"tau": float(tau), "precision": p, "recall": r, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _apply_plot_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "font.size": 11, "axes.labelsize": 12, "legend.fontsize": 10,
    })


def _add_grid(ax):
    for y in [0.2, 0.4, 0.6, 0.8, 1.0]:
        ax.axhline(y, color="gray", linewidth=0.4, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── sweep mode ────────────────────────────────────────────────────────────────

def run_sweep(model_name: str, op: float) -> None:
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    n_unique = len({s for p in LABELED_PAIRS for s in p[:2]})
    print(f"Embedding {n_unique} unique strings with backend '{model_name}' …")
    emb_map = embed_all_unique(LABELED_PAIRS)

    results = [evaluate_at_threshold(LABELED_PAIRS, emb_map, tau) for tau in THRESHOLDS]
    best    = max(results, key=lambda r: r["f1"])

    # Console table
    print(f"\n── Threshold Sweep – {model_name} {'─'*32}")
    print(f"{'τ':>6}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  "
          f"{'TP':>3}  {'FP':>3}  {'FN':>3}  {'TN':>3}")
    print("─" * 56)
    for r in results:
        notes = ""
        if abs(r["tau"] - best["tau"]) < 1e-6: notes += "  ← best F1"
        if abs(r["tau"] - op)          < 1e-6: notes += "  ← operating point"
        print(f"{r['tau']:>6.2f}  {r['precision']:>7.3f}  {r['recall']:>7.3f}  "
              f"{r['f1']:>7.3f}  {r['tp']:>3}  {r['fp']:>3}  "
              f"{r['fn']:>3}  {r['tn']:>3}{notes}")

    # Save JSON
    (CALIB_DIR / f"sweep_{model_name}.json").write_text(json.dumps(results, indent=2))

    # P / R / F1 plot
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=300)
    taus = [r["tau"] for r in results]
    ax.plot(taus, [r["precision"] for r in results], "s-", color="#1f77b4",
            label="Precision", linewidth=1.8, markersize=5)
    ax.plot(taus, [r["recall"]    for r in results], "^-", color="#ff7f0e",
            label="Recall",    linewidth=1.8, markersize=5)
    ax.plot(taus, [r["f1"]        for r in results], "o-", color="#2ca02c",
            label="F1",        linewidth=2.2, markersize=6)
    ax.axvline(op, color="red", linestyle="--", linewidth=1.3,
               label=f"Chosen τ = {op:.2f}")
    ax.set_xlabel("Similarity threshold τ")
    ax.set_ylabel("Score")
    ax.set_title(f"Threshold calibration – {model_name}")
    ax.set_xlim(THRESHOLDS[0] - 0.02, THRESHOLDS[-1] + 0.02)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower left")
    _add_grid(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(CALIB_DIR / f"sweep_{model_name}.{ext}", bbox_inches="tight")
    plt.close(fig)

    print(f"\nBest F1 = {best['f1']:.3f} at τ = {best['tau']:.2f}")
    print(f"Saved → {CALIB_DIR}/sweep_{model_name}.{{png,pdf,json}}")


# ── compare mode ──────────────────────────────────────────────────────────────

def run_compare(model_a: str, model_b: str, op_a: float, op_b: float) -> None:
    """Overlay the F1 curves of two pre-computed sweeps."""
    ra = json.loads((CALIB_DIR / f"sweep_{model_a}.json").read_text())
    rb = json.loads((CALIB_DIR / f"sweep_{model_b}.json").read_text())

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=300)

    for res, name, col, op in [
        (ra, model_a, "#1f77b4", op_a),
        (rb, model_b, "#ff7f0e", op_b),
    ]:
        taus = [r["tau"] for r in res]
        ax.plot(taus, [r["f1"] for r in res], "o-", color=col,
                label=f"F1 – {name}", linewidth=2, markersize=5)
        ax.axvline(op, color=col, linestyle=":", linewidth=1.3,
                   label=f"τ = {op:.2f}  ({name})")

    ax.set_xlabel("Similarity threshold τ")
    ax.set_ylabel("F1 score")
    ax.set_title("F1 vs. threshold – model comparison")
    ax.set_xlim(THRESHOLDS[0] - 0.02, THRESHOLDS[-1] + 0.02)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    _add_grid(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(CALIB_DIR / f"sweep_comparison.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {CALIB_DIR}/sweep_comparison.{{png,pdf}}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Phase 2: threshold calibration via entity-pair benchmark"
    )
    ap.add_argument("--model",           default="qwen",
                    help="Name tag for the current embedding backend")
    ap.add_argument("--operating-point", type=float, default=0.50,
                    help="Threshold to highlight as the chosen operating point")
    ap.add_argument("--compare",  action="store_true",
                    help="Overlay F1 curves of two pre-computed sweeps")
    ap.add_argument("--model-a",  default="nomic",
                    help="First model for --compare")
    ap.add_argument("--model-b",  default="qwen",
                    help="Second model for --compare")
    ap.add_argument("--op-a",     type=float, default=0.70,
                    help="Operating point for model-a (shown in --compare plot)")
    ap.add_argument("--op-b",     type=float, default=0.50,
                    help="Operating point for model-b (shown in --compare plot)")
    args = ap.parse_args()

    if args.compare:
        run_compare(args.model_a, args.model_b, args.op_a, args.op_b)
    else:
        run_sweep(args.model, args.operating_point)


if __name__ == "__main__":
    main()
