#!/usr/bin/env python3
"""
Pitch Compare – PyQt5 GUI with progress bar and note annotations (no preprocessing, no confidence gating, no interpolation).

Requirements:
  - numpy, pydub, matplotlib, PyQt5
  - ffmpeg required for mp3/m4a via pydub

Behavior:
  * Small PyQt window: two Browse buttons show picked filenames.
  * Press Run → progress bar updates second-by-second while analyzing.
  * Pitch method: YIN (raw) with parabolic refinement; no confidence gating.
  * No interpolation of gaps and no preprocessing of audio.
  * On completion, shows matplotlib plot with Hz and note annotations.
"""

import os
import math
import numpy as np
from pydub import AudioSegment

# Use Qt backend for matplotlib GUI plotting
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

from PyQt5.QtCore import QObject, pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QMessageBox,
    QGridLayout, QProgressBar
)

NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

# ----------------- Core audio/pitch utilities -----------------

def load_mono(path):
    audio = AudioSegment.from_file(path)
    sr = audio.frame_rate
    y = np.array(audio.get_array_of_samples()).astype(np.float32)
    if audio.channels > 1:
        y = y.reshape((-1, audio.channels)).mean(axis=1)
    y /= (1 << (8 * audio.sample_width - 1))  # normalize to [-1,1]
    return y, sr


def yin_pitch(frame, sr, fmin=60.0, fmax=1000.0):
    """YIN f0 estimator (raw): returns a frequency in Hz or np.nan. No confidence gating."""
    x = frame.astype(np.float64)
    x -= np.mean(x)
    N = x.size
    if N < 8:
        return np.nan
    tau_min = max(1, int(sr / fmax))
    tau_max = min(int(sr / fmin), N - 1)
    if tau_min >= tau_max:
        return np.nan

    # Difference function d(tau)
    d = np.zeros(tau_max + 1, dtype=np.float64)
    for tau in range(1, tau_max + 1):
        diff = x[:N - tau] - x[tau:]
        d[tau] = np.dot(diff, diff)

    # Cumulative mean normalized difference (CMND)
    cmnd = np.zeros_like(d)
    cmnd[0] = 1.0
    running_sum = 0.0
    for tau in range(1, tau_max + 1):
        running_sum += d[tau]
        cmnd[tau] = d[tau] * tau / (running_sum + 1e-12)

    # Choose global minimum within [tau_min, tau_max] (no threshold)
    tregion = cmnd[tau_min:tau_max + 1]
    if tregion.size == 0 or not np.isfinite(tregion).any():
        return np.nan
    rel_idx = int(np.argmin(tregion))
    tau = tau_min + rel_idx

    # Parabolic interpolation of the CMND minimum
    if 1 <= tau < tau_max:
        a, b, c = cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]
        denom = (a - 2.0 * b + c)
        if abs(denom) > 1e-12:
            offs = 0.5 * (a - c) / denom
            tau = float(tau) + float(offs)
    if tau <= 0:
        return np.nan

    f0 = sr / float(tau)
    return f0 if f0 > 0 else np.nan


def hz_to_note(f):
    if not np.isfinite(f) or f <= 0:
        return None
    midi = 69 + 12 * math.log2(f / 440.0)
    n = int(round(midi))
    name = NOTE_NAMES[n % 12]
    octave = n // 12 - 1
    return f"{name}{octave}"


# ----------------- Worker (runs analysis off the UI thread) -----------------
class AnalyzeWorker(QObject):
    progress_max = pyqtSignal(int)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    finished = pyqtSignal(object)

    def __init__(self, mine_path, artist_path, limit=None):
        super().__init__()
        self.mine_path = mine_path
        self.artist_path = artist_path
        self.limit = limit

    def run(self):
        try:
            # Load audio (no preprocessing)
            y_m, sr_m = load_mono(self.mine_path)
            y_a, sr_a = load_mono(self.artist_path)

            max_secs = min(len(y_m)//sr_m, len(y_a)//sr_a)
            if self.limit is not None:
                max_secs = min(max_secs, int(self.limit))
            max_secs = int(max(0, max_secs))
            self.progress_max.emit(max_secs)

            frame_len_m = sr_m
            frame_len_a = sr_a
            t = np.arange(max_secs, dtype=int)
            f_m = np.empty(max_secs, dtype=float)
            f_a = np.empty(max_secs, dtype=float)

            for i in range(max_secs):
                fm = y_m[i*frame_len_m:(i+1)*frame_len_m]
                fa = y_a[i*frame_len_a:(i+1)*frame_len_a]
                f_m[i] = yin_pitch(fm, sr_m)
                f_a[i] = yin_pitch(fa, sr_a)
                self.progress.emit(i+1)

            result = {"seconds": max_secs, "t_m": t, "f_m": f_m, "t_a": t, "f_a": f_a}
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# ----------------- GUI -----------------
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pitch Compare (PyQt5)")
        layout = QGridLayout(self)

        self.mine_path = None
        self.artist_path = None

        # Row 0: Your version
        layout.addWidget(QLabel("Your version:"), 0, 0)
        btn_mine = QPushButton("Browse")
        btn_mine.clicked.connect(self.pick_mine)
        layout.addWidget(btn_mine, 0, 1)
        self.mine_label = QLabel("No file selected")
        layout.addWidget(self.mine_label, 0, 2)

        # Row 1: Artist version
        layout.addWidget(QLabel("Artist's version:"), 1, 0)
        btn_artist = QPushButton("Browse")
        btn_artist.clicked.connect(self.pick_artist)
        layout.addWidget(btn_artist, 1, 1)
        self.artist_label = QLabel("No file selected")
        layout.addWidget(self.artist_label, 1, 2)

        # Row 2: Run button
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self.run)
        layout.addWidget(self.run_btn, 2, 0, 1, 3)

        # Row 3: Progress bar + status label
        layout.addWidget(QLabel("Progress:"), 3, 0)
        self.bar = QProgressBar()
        self.bar.setMinimum(0)
        self.bar.setMaximum(100)
        self.bar.setValue(0)
        layout.addWidget(self.bar, 3, 1)
        self.status = QLabel("Ready")
        layout.addWidget(self.status, 3, 2)

        self._thread = None
        self._worker = None

    def pick_mine(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select Your Version", "", "Audio (*.m4a *.mp3 *.wav *.flac *.aac *.ogg);;All (*.*)")
        if p:
            self.mine_path = p
            self.mine_label.setText(os.path.basename(p))

    def pick_artist(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select Artist's Version", "", "Audio (*.m4a *.mp3 *.wav *.flac *.aac *.ogg);;All (*.*)")
        if p:
            self.artist_path = p
            self.artist_label.setText(os.path.basename(p))

    def _set_running(self, running):
        self.run_btn.setEnabled(not running)
        if running:
            self.status.setText("Analyzing…")
            self.bar.setMaximum(0)  # indeterminate until max known
            self.bar.setValue(0)
        else:
            self.status.setText("Ready")
            self.bar.setMaximum(100)
            self.bar.setValue(0)

    def run(self):
        if not self.mine_path or not self.artist_path:
            QMessageBox.critical(self, "Missing file", "Please choose both files.")
            return
        self._set_running(True)
        self._thread = QThread()
        self._worker = AnalyzeWorker(self.mine_path, self.artist_path, None)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress_max.connect(self._on_progress_max)
        self._worker.progress.connect(self._on_progress)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(lambda: self._set_running(False))
        self._thread.start()

    # ---- slots ----
    def _on_progress_max(self, m):
        total = max(int(m), 1)
        self.bar.setMaximum(total)
        self.bar.setValue(0)
        self.status.setText(f"Analyzing… 0/{total} s")

    def _on_progress(self, v):
        cur = int(v)
        self.bar.setValue(cur)
        total = self.bar.maximum() if self.bar.maximum() > 0 else 0
        self.status.setText(f"Analyzing… {cur}/{total} s")

    def _on_error(self, msg):
        QMessageBox.critical(self, "Error", msg)

    def _on_finished(self, result):
        # Plot on main thread after worker completes
        t_a, f_a = result["t_a"], result["f_a"]
        t_m, f_m = result["t_m"], result["f_m"]
        self.status.setText("Plotting…")
        self.bar.setMaximum(100)
        self.bar.setValue(100)

        def annotate_points(ax, t, f, label_prefix):
            for ti, fi in zip(t, f):
                if np.isnan(fi):
                    continue
                note = hz_to_note(fi) or ""
                ax.annotate(f"{label_prefix} {fi:.1f} Hz\n{note}",
                            (ti, fi),
                            textcoords="offset points",
                            xytext=(0, 8),
                            ha="center",
                            fontsize=8,
                            alpha=0.8)

        plt.figure(figsize=(12, 4))
        ax = plt.gca()
        ax.plot(t_a, f_a, marker='o', linestyle='-', label='Artist (Hz)')
        ax.plot(t_m, f_m, marker='o', linestyle='-', label='Mine (Hz)')
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title("Second-by-Second Fundamental Frequency")
        ax.grid(True, which='both', alpha=0.3)
        ax.minorticks_on()
        annotate_points(ax, t_a, f_a, "A:")
        annotate_points(ax, t_m, f_m, "M:")
        plt.legend()
        plt.tight_layout()
        plt.show()

        self.status.setText("Ready")
        self.bar.setMaximum(100)
        self.bar.setValue(0)


def main():
    app = QApplication([])
    w = App()
    w.show()
    app.exec_()


if __name__ == "__main__":
    main()
