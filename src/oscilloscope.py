#!/usr/bin/env python3
"""
Real-time 8-channel oscilloscope — reads sensor CSV from stdin.

Usage:
    sensor_producer | python oscilloscope.py

The upstream producer may be stopped with Ctrl+C at any time;
the display will keep showing the last received data.
"""

import sys
import threading
import numpy as np
from collections import deque

from PyQt5.QtWidgets import QApplication, QMainWindow, QStatusBar
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


# ── Configuration ────────────────────────────────────────────────────────

SAMPLE_RATE = 440                          # Hz (from header)
WINDOW_SEC = 3                             # seconds of visible history
WINDOW_N = SAMPLE_RATE * WINDOW_SEC        # samples in rolling buffer
FPS = 30                                   # display refresh rate
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
        self.alive = True                  # flag visible to the GUI

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


# ── Oscilloscope widget ─────────────────────────────────────────────────

class Oscilloscope(QMainWindow):
    def __init__(self, reader: ReaderThread):
        super().__init__()
        self.reader = reader
        self.setWindowTitle("Oscilloscope — 8 ch @ 440 Hz")
        self.resize(1400, 900)

        # ── rolling data buffer (pre-allocated) ─────────────────────────
        # columns: [timestamp, ch1 … ch8]
        self.buf = np.zeros((WINDOW_N, NCHAN + 1), dtype=np.float64)
        self.n_total = 0                   # total samples received

        # ── matplotlib figure ────────────────────────────────────────────
        self.fig = Figure(facecolor="#101020")
        self.fig.subplots_adjust(
            left=0.07, right=0.98, top=0.97, bottom=0.04,
            hspace=0.12,
        )
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.setCentralWidget(self.canvas)

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

        # ── initial draw + blit backgrounds ──────────────────────────────
        self.canvas.draw()
        self._save_backgrounds()

        # ── status bar ───────────────────────────────────────────────────
        self.status = QStatusBar()
        self.status.setStyleSheet("color: #888888; background: #101020;")
        self.setStatusBar(self.status)

        # ── refresh timer ────────────────────────────────────────────────
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer)
        self._timer.start(1000 // FPS)

    # ── helpers ──────────────────────────────────────────────────────────

    def _save_backgrounds(self):
        """Capture axes backgrounds for blitting."""
        self._bgs = [
            self.canvas.copy_from_bbox(ax.bbox) for ax in self.axes
        ]

    def _append(self, new_data: np.ndarray):
        """Shift rolling buffer left and append new rows."""
        n = len(new_data)
        self.n_total += n
        if n >= WINDOW_N:
            self.buf[:] = new_data[-WINDOW_N:]
        else:
            self.buf[:-n] = self.buf[n:]
            self.buf[-n:] = new_data

    # ── periodic update (called at FPS) ──────────────────────────────────

    def _on_timer(self):
        new = self.reader.drain()
        if new is None:
            self._update_status()
            return

        self._append(new)

        # Check if any channel's range changed enough to need a full redraw
        need_full_redraw = False
        for i in range(NCHAN):
            y = self.buf[:, i + 1]
            ymin, ymax = y.min(), y.max()
            margin = max(abs(ymax - ymin) * 0.2, 5e-4)
            new_lo, new_hi = ymin - margin, ymax + margin
            old_lo, old_hi = self.axes[i].get_ylim()
            # Redraw if limits drifted noticeably
            if abs(new_lo - old_lo) > margin * 0.3 or abs(new_hi - old_hi) > margin * 0.3:
                self.axes[i].set_ylim(new_lo, new_hi)
                need_full_redraw = True

        if need_full_redraw:
            # Axes decorations changed — full redraw, then re-save backgrounds
            for i in range(NCHAN):
                self.lines[i].set_ydata(self.buf[:, i + 1])
            self.canvas.draw_idle()
            # Re-save after the draw completes (via a single-shot callback)
            self.canvas.mpl_connect("draw_event", self._on_draw_event)
        else:
            # Fast path: blit only the line artists
            for i in range(NCHAN):
                self.canvas.restore_region(self._bgs[i])
                self.lines[i].set_ydata(self.buf[:, i + 1])
                self.axes[i].draw_artist(self.lines[i])
                self.canvas.blit(self.axes[i].bbox)

        self._update_status()

    def _on_draw_event(self, _event):
        """Re-capture backgrounds after a full redraw."""
        self._save_backgrounds()
        # Disconnect so we don't re-save on every future blit
        self.canvas.mpl_disconnect(
            self.canvas.callbacks.callbacks.get("draw_event", {})
        )

    def _update_status(self):
        alive = "● LIVE" if self.reader.alive else "○ ENDED"
        rate = self.n_total / max(1, self.buf[-1, 0] - self.buf[0, 0]) if self.n_total > WINDOW_N else 0
        self.status.showMessage(
            f"  {alive}   |   Samples: {self.n_total:,}   |   "
            f"Effective rate: {rate:.0f} Hz   |   "
            f"Window: {WINDOW_SEC}s ({WINDOW_N} pts)"
        )

    def resizeEvent(self, event):
        """Re-capture blit backgrounds after a window resize."""
        super().resizeEvent(event)
        # Schedule background re-capture after Qt finishes the resize
        QTimer.singleShot(50, self._full_redraw)

    def _full_redraw(self):
        self.canvas.draw_idle()
        QTimer.singleShot(50, self._save_backgrounds)


# ── Entry point ──────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    reader = ReaderThread()
    reader.start()

    win = Oscilloscope(reader)
    win.show()

    try:
        sys.exit(app.exec_())
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == "__main__":
    main()
