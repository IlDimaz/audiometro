# =============================================================================
# IMPORTAZIONI
# Librerie standard: argparse, json, re, threading, datetime, pathlib
# Libreria esterna obbligatoria: serial (comunicazione con STM32 via USB/UART)
#
# numpy e sounddevice sono importate in modo lazy dentro TonePlayer:
# se il firmware è compilato con USE_PC_AUDIO=0, la STM32 non invierà mai
# comandi "AUDIO *" e TonePlayer non verrà mai istanziato, quindi queste
# librerie non servono e non è necessario averle installate.
# =============================================================================
import argparse
import json
import re
import threading
from datetime import datetime
from pathlib import Path
import serial


# =============================================================================
# COSTANTI GLOBALI
#
# RESULT_LINE_RE – riconosce le righe di risultato audiometrico nella forma:
#   "Freq  500 Hz → -54.5 dBFS"
#   Gruppo 1 = frequenza in Hz, gruppo 2 = valore dBFS.
#   "\S+" al posto di "→" rende il match robusto a qualsiasi codifica del
#   carattere Unicode U+2192 (può arrivare corrotto su UART).
# =============================================================================
RESULT_LINE_RE = re.compile(
    r"^Freq\s+(\d+)\s+Hz\s+\S+\s+(-?\d+(?:\.\d+)?)\s+dBFS$"
)
NOT_PERCEIVED = -100.0


# =============================================================================
# CLASSE ResultCollector
#
# Accumula le righe UART del blocco risultati, le interpreta e salva un JSON.
#
# Flusso atteso dal firmware (identico per USE_PC_AUDIO 0 o 1):
#   1. "=== Risultati Audiometria ===" → attiva il parsing
#   2. "Orecchio L" / "Orecchio R"    → imposta orecchio corrente
#   3. "Freq  500 Hz → -54.5 dBFS"   → riga di misura
#   4. Quando tutti i valori sono arrivati → salva JSON e si azzera
# =============================================================================
class ResultCollector:
    """Parsing e salvataggio dei risultati audiometrici ricevuti via UART."""

    def __init__(self, output_dir: str, expected_freq_count: int = 11):
        self.output_dir = Path(output_dir)
        self.expected_freq_count = expected_freq_count
        self.reset()

    def reset(self):
        self.in_results_block = False
        self.current_ear = None
        self.results = {"L": {}, "R": {}}

    def feed_line(self, line: str):
        """
        Analizza una riga decodificata dalla seriale.
        Restituisce (payload, path) quando il test è completo, None altrimenti.
        """
        text = line.strip()

        if text == "=== Risultati Audiometria ===":
            self.in_results_block = True
            return None

        if not self.in_results_block:
            return None

        if text.startswith("Orecchio "):
            self.current_ear = text.split()[-1].upper()
            return None

        match = RESULT_LINE_RE.match(text)
        if match and self.current_ear:
            freq_hz = int(match.group(1))
            dbfs    = float(match.group(2))
            self.results[self.current_ear][freq_hz] = dbfs

            if self._is_complete():
                payload = self._build_payload()
                path    = self._save(payload)
                self.reset()
                return payload, path

        return None

    def _is_complete(self):
        return all(
            len(self.results[ear]) >= self.expected_freq_count
            for ear in ["L", "R"]
        )

    def _build_payload(self):
        import math

        def perceived(ear):
            return [(f, v) for f, v in self.results[ear].items() if v > NOT_PERCEIVED]

        def mean(vals):
            return sum(vals) / len(vals) if vals else None

        def std(vals):
            if len(vals) < 2:
                return None
            m = mean(vals)
            return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))

        def linear_slope(pairs):
            if len(pairs) < 2:
                return None
            xs = [f / 1000.0 for f, _ in pairs]
            ys = [v          for _, v in pairs]
            mx, my = mean(xs), mean(ys)
            num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            den = sum((x - mx) ** 2        for x in xs)
            return num / den if den != 0 else None

        perc_L = perceived("L")
        perc_R = perceived("R")
        vals_L = [v for _, v in perc_L]
        vals_R = [v for _, v in perc_R]

        common_freqs = [
            f for f in self.results["L"]
            if f in self.results["R"]
            and self.results["L"][f] > NOT_PERCEIVED
            and self.results["R"][f] > NOT_PERCEIVED
        ]
        lr_diffs = [abs(self.results["L"][f] - self.results["R"][f]) for f in common_freqs]

        summary = {
            "mean_dbfs": {"L": mean(vals_L), "R": mean(vals_R)},
            "mean_abs_lr_diff_db": mean(lr_diffs),
            "max_abs_lr_diff_db":  max(lr_diffs) if lr_diffs else None,
            "not_perceived_count": {
                "L": sum(1 for v in self.results["L"].values() if v <= NOT_PERCEIVED),
                "R": sum(1 for v in self.results["R"].values() if v <= NOT_PERCEIVED),
            },
        }

        mean_diff = summary["mean_abs_lr_diff_db"]
        if   mean_diff is None:  sym_class = "non calcolabile"
        elif mean_diff < 5.0:    sym_class = "buona"
        elif mean_diff < 10.0:   sym_class = "moderata"
        else:                    sym_class = "asimmetria significativa"

        flagged = sorted([
            f for f in common_freqs
            if abs(self.results["L"][f] - self.results["R"][f]) > 4.0
        ])

        symmetry = {
            "class":                         sym_class,
            "mean_abs_lr_diff_db":           mean_diff,
            "max_abs_lr_diff_db":            summary["max_abs_lr_diff_db"],
            "flagged_freq_abs_diff_gt_4db":  flagged,
        }

        LOW_FREQS  = {125, 250, 500, 750}
        HIGH_FREQS = {2000, 3000, 4000, 6000, 8000}

        def spectral_stats(ear_pairs):
            low  = [v for f, v in ear_pairs if f in LOW_FREQS]
            high = [v for f, v in ear_pairs if f in HIGH_FREQS]
            lm, hm = mean(low), mean(high)
            diff = (lm - hm) if (lm is not None and hm is not None) else None
            if   diff is None:  cls = "non calcolabile"
            elif diff < 10.0:   cls = "piatto"
            elif diff < 20.0:   cls = "lieve penalizzazione basse frequenze"
            else:               cls = "forte penalizzazione basse frequenze"
            return {"low_band_mean_dbfs": lm, "high_band_mean_dbfs": hm,
                    "low_minus_high_db": diff, "class": cls}

        spectral_profile = {"L": spectral_stats(perc_L), "R": spectral_stats(perc_R)}

        stability = {
            "std_db":           {"L": std(vals_L),          "R": std(vals_R)},
            "slope_db_per_khz": {"L": linear_slope(perc_L), "R": linear_slope(perc_R)},
        }

        sp_L_cls  = spectral_profile["L"]["class"]
        sp_R_cls  = spectral_profile["R"]["class"]
        both_flat = (sp_L_cls == "piatto" and sp_R_cls == "piatto")

        if   sym_class == "buona" and both_flat:
            verdict = "Profilo complessivamente buono e simmetrico"
        elif sym_class == "buona":
            verdict = "Buona simmetria L-R, ma profilo spettrale non uniforme"
        elif sym_class == "moderata" and both_flat:
            verdict = "Profilo spettrale uniforme, ma asimmetria L-R moderata da monitorare"
        elif sym_class == "asimmetria significativa":
            verdict = "Asimmetria L-R significativa: valutare con un audiologo"
        else:
            verdict = "Profilo variabile: confrontare con misure cliniche calibrate"

        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "results": {
                e: {str(k): v for k, v in sorted(d.items())}
                for e, d in self.results.items()
            },
            "summary": summary,
            "interpretation": {
                "symmetry":         symmetry,
                "spectral_profile": spectral_profile,
                "stability":        stability,
                "verdict":          verdict,
                "notes": [
                    "Valori in dBFS: non equivalgono a dB HL clinici.",
                    "Interpretazione influenzata da cuffie, ambiente e calibrazione master-gain.",
                ],
            },
        }

    def _save(self, payload):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"audiometry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        return path


# =============================================================================
# CLASSE TonePlayer
#
# Istanziata automaticamente alla prima ricezione di un comando "AUDIO START",
# quindi solo se il firmware è compilato con USE_PC_AUDIO=1.
# Con USE_PC_AUDIO=0 la STM32 non invia mai comandi AUDIO e questa classe
# non viene mai istanziata: numpy e sounddevice rimangono non importate.
#
# Sintesi audio: DDS con Look-Up Table a 4096 campioni + interpolazione
# lineare tra campioni adiacenti per ridurre la distorsione armonica.
# Routing stereo: L / R / B (entrambi) in base all'orecchio indicato dal C.
# Thread safety: tutte le variabili di stato sono protette da threading.Lock
# perché il callback audio gira su un thread separato di sounddevice.
# =============================================================================
class TonePlayer:
    """Motore audio DDS lazy: viene creato solo se arriva un comando AUDIO START."""

    def __init__(self, samplerate: int, master_gain: float):
        global np, sd
        import numpy as np
        import sounddevice as sd

        self.samplerate  = samplerate
        self.master_gain = max(0.01, min(1.0, float(master_gain)))
        self.lock        = threading.Lock()

        self.phase   = 0.0
        self.freq    = 440.0
        self.gain    = 0.0
        self.ear     = "L"
        self.playing = False

        self.lut_size = 4096
        self.lut = np.sin(
            2.0 * np.pi * np.arange(self.lut_size) / self.lut_size
        ).astype(np.float32)

        self.stream = sd.OutputStream(
            samplerate=self.samplerate, channels=2,
            dtype="float32", callback=self._callback
        )
        self.stream.start()
        print("[audio] Stream aperto (USE_PC_AUDIO=1 rilevato dal firmware)")

    def _callback(self, outdata, frames, _time, status):
        if status:
            print(f"[audio status] {status}")

        with self.lock:
            if not self.playing or self.gain <= 0.0:
                outdata.fill(0)
                return
            curr_freq  = self.freq
            curr_gain  = self.gain
            curr_ear   = self.ear
            curr_phase = self.phase

        step    = curr_freq * self.lut_size / self.samplerate
        samples = np.empty(frames, dtype=np.float32)

        for i in range(frames):
            idx0 = int(curr_phase) % self.lut_size
            idx1 = (idx0 + 1) % self.lut_size
            frac = curr_phase - int(curr_phase)
            samples[i] = self.lut[idx0] + frac * (self.lut[idx1] - self.lut[idx0])
            curr_phase = (curr_phase + step) % self.lut_size

        samples *= (curr_gain * self.master_gain)

        with self.lock:
            self.phase = curr_phase

        stereo = np.zeros((frames, 2), dtype=np.float32)
        if curr_ear == "R":
            stereo[:, 1] = samples
        elif curr_ear == "B":
            stereo[:, 0] = stereo[:, 1] = samples
        else:
            stereo[:, 0] = samples
        outdata[:] = stereo

    def start(self, ear, freq, gain):
        with self.lock:
            self.ear     = ear.upper()
            self.freq    = float(freq)
            self.gain    = float(gain)
            self.phase   = 0.0
            self.playing = True
        print(f"[tone] START {self.ear} {self.freq}Hz Gain:{self.gain:.3f}")

    def set_gain(self, ear, gain):
        with self.lock:
            if ear:
                self.ear = ear.upper()
            self.gain = float(gain)
        print(f"[tone] GAIN {self.ear} {self.gain:.3f}")

    def stop(self):
        with self.lock:
            self.playing = False
            self.gain    = 0.0
        print("[tone] STOP")

    def close(self):
        self.stream.stop()
        self.stream.close()
        print("[audio] Stream chiuso")


# =============================================================================
# FUNZIONE PRINCIPALE main()
#
# Argomenti CLI: --port (obbligatorio), --baud, --master-gain.
# NON esiste più --no-pc-audio: è il firmware C a decidere tramite
# #define USE_PC_AUDIO se inviare o meno comandi AUDIO sulla UART.
#
# TonePlayer viene creato la prima volta che arriva "AUDIO START" e rimane
# None per tutta l'esecuzione se il firmware non lo usa mai.
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="STM32 Audiometer Interface")
    parser.add_argument("--port",        required=True,
                        help="Porta seriale (es. COM3 o /dev/ttyACM0)")
    parser.add_argument("--baud",        type=int,   default=115200,
                        help="Baud rate UART (default 115200)")
    parser.add_argument("--master-gain", type=float, default=0.2,
                        help="Volume master per la modalità PC audio (0.01-1.0, default 0.2)")
    args = parser.parse_args()

    # TonePlayer viene creato lazily alla prima ricezione di "AUDIO START".
    # Se il firmware usa USE_PC_AUDIO=0, questa variabile resterà None
    # per tutta l'esecuzione e numpy/sounddevice non verranno mai importati.
    player: TonePlayer | None = None

    collector = ResultCollector("results")

    print(f"[system] Connessione a {args.port} ({args.baud} baud)...")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.1)

        while True:
            raw  = ser.readline()
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            print(f"[mcu] {line}")

            # ---- Raccolta risultati audiometrici ----
            # Funziona uguale sia con USE_PC_AUDIO=0 che =1.
            collected = collector.feed_line(line)
            if collected:
                print(f"[system] Test completato! Risultati salvati in: {collected[1]}")

            # ---- Gestione comandi AUDIO ----
            # Questi comandi arrivano SOLO se il firmware è compilato con
            # USE_PC_AUDIO=1. Con USE_PC_AUDIO=0 la STM32 non li invia mai,
            # quindi questo blocco rimane inattivo senza alcuna configurazione
            # manuale: il comportamento segue automaticamente quanto stabilito
            # dal #define nel codice C.
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "AUDIO":
                cmd = parts[1].upper()
                try:
                    if cmd == "START" and len(parts) >= 5:
                        # Prima ricezione di AUDIO START: inizializza lo stream audio.
                        if player is None:
                            player = TonePlayer(48000, args.master_gain)
                        player.start(parts[2], parts[3], parts[4])

                    elif cmd == "GAIN" and len(parts) >= 4:
                        if player is not None:
                            player.set_gain(parts[2], parts[3])
                        else:
                            print("[warn] AUDIO GAIN ricevuto prima di AUDIO START, ignorato")

                    elif cmd in ("STOP", "DONE"):
                        if player is not None:
                            player.stop()

                except ValueError:
                    print(f"[error] Parametri comando non validi: {line}")

    except KeyboardInterrupt:
        print("\n[exit] Chiusura in corso...")
    finally:
        if player is not None:
            player.close()
        if "ser" in locals():
            ser.close()


if __name__ == "__main__":
    main()
