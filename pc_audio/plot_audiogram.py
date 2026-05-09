"""
plot_audiogram.py  –  Plotta l'audiogramma da un file JSON prodotto da audiometer.py

Uso:
    python plot_audiogram.py risultati.json
    python plot_audiogram.py risultati.json --save audiogramma.png

Il grafico segue le convenzioni audiometriche standard:
  - Asse X: frequenza in Hz (scala logaritmica, da 125 a 8000 Hz)
  - Asse Y: livello in dBFS (invertito: valori bassi = soglia peggiore)
  - Orecchio sinistro (L): linea blu con simbolo X
  - Orecchio destro  (R): linea rossa con simbolo O
  - Frecce verso il basso: frequenze non percepite (> MAX_DBFS)
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# =============================================================================
# COSTANTI
# =============================================================================

# Frequenze audiometriche standard (asse X)
FREQS = [125, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000]

# Valore sentinella "frequenza non percepita" (dal firmware)
NOT_PERCEIVED = -100.0

# Stile per i due orecchi: (colore, marcatore, etichetta)
EAR_STYLE = {
    "L": {"color": "#2563eb", "marker": "x", "label": "Sinistro (L)",
          "mew": 2.5, "ms": 10},   # mew = marker edge width
    "R": {"color": "#dc2626", "marker": "o", "label": "Destro (R)",
          "mew": 2.0, "ms": 8, "mfc": "none"},  # mfc=none → cerchio vuoto
}

# =============================================================================
# FUNZIONI
# =============================================================================

def load_json(path: str) -> dict:
    """Carica e valida il file JSON."""
    p = Path(path)
    if not p.exists():
        sys.exit(f"[errore] File non trovato: {path}")
    with p.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if "results" not in data:
        sys.exit("[errore] Il JSON non contiene la chiave 'results'.")
    return data


def extract_series(results: dict, ear: str):
    """
    Estrae (freqs_valide, dbfs_validi, freqs_np) per un orecchio.
    freqs_np: frequenze non percepite (da disegnare come frecce).
    """
    freqs_ok, vals_ok, freqs_np = [], [], []
    ear_data = results.get(ear, {})
    for f in FREQS:
        v = ear_data.get(str(f)) or ear_data.get(f)
        if v is None:
            continue
        if v <= NOT_PERCEIVED:
            freqs_np.append(f)
        else:
            freqs_ok.append(f)
            vals_ok.append(v)
    return freqs_ok, vals_ok, freqs_np


def build_summary_text(data: dict) -> str:
    """Costruisce il testo del riquadro riepilogativo dal blocco interpretation."""
    lines = []
    interp = data.get("interpretation", {})
    summary = data.get("summary", {})

    # Verdetto principale
    verdict = interp.get("verdict")
    if verdict:
        lines.append(f"Verdetto: {verdict}")

    # Simmetria
    sym = interp.get("symmetry", {})
    mean_diff = sym.get("mean_abs_lr_diff_db")
    max_diff  = sym.get("max_abs_lr_diff_db")
    sym_class = sym.get("class", "")
    if mean_diff is not None:
        lines.append(f"Simmetria L-R: {sym_class}  (media Δ {mean_diff:.1f} dB,  max {max_diff:.1f} dB)")

    flagged = sym.get("flagged_freq_abs_diff_gt_4db", [])
    if flagged:
        lines.append(f"Freq. asimmetriche (>4 dB): {', '.join(str(f) for f in flagged)} Hz")

    # Profilo spettrale per orecchio
    sp = interp.get("spectral_profile", {})
    for ear in ["L", "R"]:
        ep = sp.get(ear, {})
        cls  = ep.get("class", "")
        diff = ep.get("low_minus_high_db")
        if diff is not None:
            lines.append(f"Profilo {ear}: {cls}  (Δ basse-alte {diff:.1f} dB)")

    # Pendenza
    stab = interp.get("stability", {})
    slopes = stab.get("slope_db_per_khz", {})
    sl = slopes.get("L"); sr = slopes.get("R")
    if sl is not None and sr is not None:
        lines.append(f"Pendenza: L {sl:.2f} dB/kHz,  R {sr:.2f} dB/kHz")

    # Note disclaimer
    notes = interp.get("notes", [])
    if notes:
        lines.append("")
        for n in notes:
            lines.append(f"• {n}")

    return "\n".join(lines)


def plot(data: dict, save_path: str | None = None):
    """Disegna l'audiogramma e lo mostra (o lo salva se save_path è specificato)."""

    results   = data["results"]
    timestamp = data.get("timestamp", "")

    # ------------------------------------------------------------------
    # Layout: grafico principale (sinistra) + riquadro testo (destra)
    # ------------------------------------------------------------------
    fig, (ax, ax_txt) = plt.subplots(
        1, 2,
        figsize=(14, 6),
        gridspec_kw={"width_ratios": [2.2, 1]},
    )
    fig.patch.set_facecolor("#f8fafc")

    # ------------------------------------------------------------------
    # Grafico principale – audiogramma
    # ------------------------------------------------------------------
    ax.set_facecolor("#ffffff")
    ax.set_xscale("log")

    # Asse X: etichette alle frequenze standard
    ax.set_xticks(FREQS)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x)}"
    ))
    ax.set_xlim(80, 10000)

    # Asse Y: invertito (come audiogramma clinico: peggio = più in basso)
    # Range da -20 dBFS (quasi silenzio) a -70 dBFS (volume quasi massimo)
    all_vals = [
        v for ear in ["L", "R"]
        for v in results.get(ear, {}).values()
        if v > NOT_PERCEIVED
    ]
    y_min = min(all_vals) - 5 if all_vals else -75
    y_max = max(all_vals) + 5 if all_vals else -15
    # Arrotondiamo ai 5 dB più vicini per un aspetto pulito
    y_min = (y_min // 5) * 5
    y_max = math.ceil(y_max / 5) * 5
    ax.set_ylim(y_min, y_max)
    ax.invert_yaxis()   # valori più negativi (soglia peggiore) in basso

    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

    # Griglia
    ax.grid(which="major", color="#e2e8f0", linewidth=0.8, linestyle="-")
    ax.grid(which="minor", color="#f1f5f9", linewidth=0.4, linestyle=":")

    # Linee verticali di riferimento (frequenze standard)
    for f in FREQS:
        ax.axvline(f, color="#e2e8f0", linewidth=0.5, zorder=0)

    # Curva per ogni orecchio
    for ear, style in EAR_STYLE.items():
        freqs_ok, vals_ok, freqs_np = extract_series(results, ear)

        # Linea + marcatori per le frequenze percepite
        if freqs_ok:
            kw = dict(
                color=style["color"],
                marker=style["marker"],
                label=style["label"],
                linewidth=2,
                markersize=style["ms"],
                markeredgewidth=style["mew"],
                zorder=3,
            )
            if "mfc" in style:
                kw["markerfacecolor"] = style["mfc"]
            ax.plot(freqs_ok, vals_ok, **kw)

        # Frecce verso il basso per le frequenze non percepite
        # (posizionate al limite inferiore del grafico)
        arrow_y = y_min + 2   # leggermente sopra il bordo inferiore
        for f in freqs_np:
            ax.annotate(
                "", xy=(f, arrow_y + 6), xytext=(f, arrow_y),
                arrowprops=dict(
                    arrowstyle="->", color=style["color"],
                    lw=1.8, mutation_scale=12,
                ),
                zorder=4,
            )
            ax.text(
                f, arrow_y - 1, "NP",
                ha="center", va="bottom",
                fontsize=7, color=style["color"], style="italic",
            )

    # Etichette assi e titolo
    ax.set_xlabel("Frequenza (Hz)", fontsize=11, labelpad=8)
    ax.set_ylabel("Soglia (dBFS)", fontsize=11, labelpad=8)
    ax.set_title(
        f"Audiogramma  –  {timestamp}",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    # Nota sul verso dell'asse Y
    ax.text(
        0.01, 0.02,
        "↓  soglia peggiore (serve più volume)",
        transform=ax.transAxes,
        fontsize=8, color="#94a3b8", style="italic",
    )

    # ------------------------------------------------------------------
    # Riquadro testo – riepilogo interpretazione
    # ------------------------------------------------------------------
    ax_txt.set_axis_off()
    ax_txt.set_facecolor("#f8fafc")

    summary_text = build_summary_text(data)
    ax_txt.text(
        0.04, 0.97, summary_text,
        transform=ax_txt.transAxes,
        va="top", ha="left",
        fontsize=8.5,
        family="monospace",
        color="#1e293b",
        wrap=True,
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="#f1f5f9",
            edgecolor="#cbd5e1",
            linewidth=1,
        ),
    )

    plt.tight_layout(pad=1.5)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] Salvato in: {save_path}")
    else:
        plt.show()


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import math

    parser = argparse.ArgumentParser(
        description="Plotta l'audiogramma da un file JSON prodotto da audiometer.py"
    )
    parser.add_argument("json_file", help="Percorso del file JSON dei risultati")
    parser.add_argument(
        "--save", metavar="FILE.png",
        help="Salva il grafico su file invece di mostrarlo a schermo"
    )
    args = parser.parse_args()

    data = load_json(args.json_file)
    plot(data, save_path=args.save)
