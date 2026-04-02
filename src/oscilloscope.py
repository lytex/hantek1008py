#!/usr/bin/env python3
"""
Real-time 8-channel oscilloscope — reads sensor CSV from stdin.

Uses the officially recommended matplotlib-in-Qt pattern:
  canvas.new_timer()  for timing (backend-native, properly integrated)
  line.set_ydata()    for updating data
  canvas.draw_idle()  for rendering

Two timers decouple data ingestion from rendering:
  data_timer   (1 ms)  — drains stdin as fast as possible
  draw_timer   (20 ms) — redraws at ~50 fps

Reference:
  https://matplotlib.org/stable/gallery/user_interfaces/embedding_in_qt_sgskip.html

Usage:
    sensor_producer | python oscilloscope.py
"""

import sys
import threading
import numpy as np
from collections import deque

from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure


# ── Configuration ────────────────────────────────────────────────────────

SAMPLE_RATE = 440                          # Hz (from header)
WINDOW_SEC = 3                             # seconds of visible history
WINDOW_N = SAMPLE_RATE * WINDOW_SEC        # samples in rolling buffer
DRAW_INTERVAL_MS = 20                      # ~50 fps drawing
DATA_INTERVAL_MS = 1                       # drain stdin ASAP
NCHAN = 8

COLORS = [
    "#faed27", "#00e5ff", "#76ff03", "#ea80fc",
    "#ff9100", "#00e676", "#448aff", "#ff5252",
]
CHAN_LABELS = [f"CH{i+1}" for i in range(NCHAN)]


# ── Stdin reader (background thread) ────────────────────────────────────

class ReaderThread(threading.Thread):
    """
    Reads CSV lines from stdin.buffer, skips headers / ^C / INFO lines,
    and appends parsed rows to a thread-safe deque.
    """

    def __init__(self):
        super().__init__(daemon=True)
        self.pending: deque[np.ndarray] = deque(maxlen=200_000)
        self.alive = True

    def run(self):
        try:
            for raw in sys.stdin.buffer:
                line = raw.rstrip(b"\n\r")
                if not line or line[0:1] in (b"#", b"^", b"I"):
                    continue
                try:
                    row = np.fromstring(line.decode(), sep=",", dtype=np.float64)
                    if row.size == NCHAN + 1:
                        self.pending.append(row)
                except (ValueError, UnicodeDecodeError):
                    continue
        except (OSError, ValueError):
            pass
        finally:
            self.alive = False

    def drain(self) -> np.ndarray | None:
        """Pop all pending rows into one (N, 9) array.  None if empty."""
        rows: list[np.ndarray] = []
        try:
            while True:
                rows.append(self.pending.popleft())
        except IndexError:
            pass
        if not rows:
            return None
        return np.vstack(rows)


# ── Oscilloscope window ─────────────────────────────────────────────────

class Oscilloscope(QtWidgets.QMainWindow):
    def __init__(self, reader: ReaderThread):
        super().__init__()
        self.reader = reader
        self.setWindowTitle("Oscilloscope — 8 ch @ 440 Hz")
        self.resize(1400, 900)

        # ── rolling data buffer ──────────────────────────────────────────
        self.buf = np.zeros((WINDOW_N, NCHAN + 1), dtype=np.float64)
        self.n_total = 0

        # ── Qt layout following the official example ─────────────────────
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        layout = QtWidgets.QVBoxLayout(self._main)

        # ── matplotlib figure + canvas ───────────────────────────────────
        self.canvas = FigureCanvas(Figure(figsize=(14, 9), facecolor="#101020"))
        layout.addWidget(self.canvas)
        self.fig = self.canvas.figure
        self.fig.subplots_adjust(
            left=0.07, right=0.98, top=0.97, bottom=0.05,
            hspace=0.12,
        )

        x = np.linspace(-WINDOW_SEC, 0, WINDOW_N)
        self.axes: list = []
        self.lines: list = []

        for i in range(NCHAN):
            ax = self.fig.add_subplot(NCHAN, 1, i + 1)
            ax.set_facecolor("#0a0a1a")
            ax.tick_params(colors="#666666", labelsize=7)
            for spine in ax.spines.values():
                spine.set_color("#282840")
            ax.set_ylabel(
                CHAN_LABELS[i], color=COLORS[i],
                fontsize=9, rotation=0, labelpad=35,
            )
            ax.set_xlim(-WINDOW_SEC, 0)
            ax.grid(True, color="#1a1a30", linewidth=0.5)
            (line,) = ax.plot(
                x, np.zeros(WINDOW_N),
                color=COLORS[i], linewidth=0.7,
            )
            self.lines.append(line)
            self.axes.append(ax)
            if i < NCHAN - 1:
                ax.tick_params(labelbottom=False)

        self.axes[-1].set_xlabel("Time (s)", color="#666666", fontsize=9)

        # ── status label ─────────────────────────────────────────────────
        self.status = QtWidgets.QLabel()
        self.status.setStyleSheet("color: #888888; background: #101020; padding: 4px;")
        layout.addWidget(self.status)

        # ── backend-native timers (the key to reliable embedded Qt) ──────
        # Data timer: drain stdin into rolling buffer as fast as possible
        self._data_timer = self.canvas.new_timer(DATA_INTERVAL_MS)
        self._data_timer.add_callback(self._update_data)
        self._data_timer.start()

        # Draw timer: push buffer into lines and repaint at ~50 fps
        self._draw_timer = self.canvas.new_timer(DRAW_INTERVAL_MS)
        self._draw_timer.add_callback(self._update_canvas)
        self._draw_timer.start()

    # ── timer callbacks ──────────────────────────────────────────────────

    def _update_data(self):
        """Drain all available stdin data into the rolling buffer."""
        new = self.reader.drain()
        if new is not None:
            n = len(new)
            self.n_total += n
            if n >= WINDOW_N:
                self.buf[:] = new[-WINDOW_N:]
            else:
                self.buf[:-n] = self.buf[n:]
                self.buf[-n:] = new

    def _update_canvas(self):
        """Push buffer into line artists, auto-scale, and redraw."""
        for i in range(NCHAN):
            y = self.buf[:, i + 1]
            self.lines[i].set_ydata(y)

            # Auto-scale: only widen y-limits when data exceeds them
            lo, hi = self.axes[i].get_ylim()
            ymin, ymax = y.min(), y.max()
            if ymin < lo or ymax > hi:
                span = max(ymax - ymin, 1e-3)
                margin = span * 0.35
                self.axes[i].set_ylim(ymin - margin, ymax + margin)

        self.canvas.draw_idle()
        self._update_status()

    def _update_status(self):
        alive = "● LIVE" if self.reader.alive else "○ ENDED"
        dt = self.buf[-1, 0] - self.buf[0, 0]
        rate = self.n_total / dt if self.n_total > WINDOW_N and dt > 0 else 0
        self.status.setText(
            f"  {alive}   |   Samples: {self.n_total:,}   |   "
            f"Effective rate: {rate:.0f} Hz   |   "
            f"Window: {WINDOW_SEC}s ({WINDOW_N} pts)"
        )


# ── Entry point ──────────────────────────────────────────────────────────

def main():
    # Check whether there is already a running QApplication
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)

    reader = ReaderThread()
    reader.start()

    win = Oscilloscope(reader)
    win.show()
    win.activateWindow()
    win.raise_()

    try:
        qapp.exec()
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == "__main__":
    main()
