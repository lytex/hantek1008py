"""
Fast async reader for multi-channel sensor data from stdin.

Skips header lines (starting with '#'), ignores upstream Ctrl+C artifacts,
and yields numpy arrays in batches for efficient downstream computation.

Usage:
    some_producer | python your_script.py

    import asyncio
    from async_stdin_reader import read_stdin_chunks

    async def main():
        async for timestamps, channels in read_stdin_chunks(batch_size=512):
            # timestamps: np.ndarray shape (N,)  — float64 unix timestamps
            # channels:   np.ndarray shape (N,8) — float64 sensor values
            ...

    asyncio.run(main())
"""

import asyncio
import sys
import os
import numpy as np
from typing import AsyncIterator

NCOLS = 9  # 1 timestamp + 8 channels


async def read_stdin_chunks(
    batch_size: int = 1024,
) -> AsyncIterator[tuple[np.ndarray, np.ndarray]]:
    """
    Async generator that reads sensor CSV from stdin and yields
    (timestamps, channels) numpy array pairs in batches.

    Parameters
    ----------
    batch_size : int
        Number of data rows to accumulate before yielding.
        Larger = fewer yields, better numpy throughput.
        Smaller = lower latency per batch.
    """
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=2**20)
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin.buffer)

    buf = bytearray()
    rows: list[bytes] = []

    while True:
        try:
            chunk = await reader.read(65536)
        except (ConnectionError, OSError):
            break

        if not chunk:
            break

        buf.extend(chunk)

        # Split on newlines; keep the trailing incomplete line in buf
        *lines, buf = buf.split(b"\n")
        buf = bytearray(buf)

        for line in lines:
            # Skip empty, header, and ^C / INFO teardown lines
            if not line or line[0:1] in (b"#", b"^", b"I"):
                continue
            rows.append(line)

            if len(rows) >= batch_size:
                yield _parse_rows(rows)
                rows.clear()

    # Flush remaining rows
    if buf:
        stripped = bytes(buf)
        if stripped and stripped[0:1] not in (b"#", b"^", b"I"):
            rows.append(stripped)

    if rows:
        yield _parse_rows(rows)


def _parse_rows(rows: list[bytes]) -> tuple[np.ndarray, np.ndarray]:
    """Parse raw CSV byte-lines into numpy arrays, fast path."""
    # Join into one blob and let numpy do the heavy lifting
    blob = b"\n".join(rows)
    data = np.loadtxt(blob.splitlines(), delimiter=",", dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, 0], data[:, 1:]


# ── Convenience wrapper that shields against upstream SIGPIPE / ^C ───────

async def run(callback, batch_size: int = 1024):
    """
    Call `await callback(timestamps, channels)` for each batch.
    Handles upstream termination cleanly.
    """
    try:
        async for ts, ch in read_stdin_chunks(batch_size=batch_size):
            await callback(ts, ch)
    except (BrokenPipeError, ConnectionResetError, KeyboardInterrupt):
        pass


# ── Demo / self-test ─────────────────────────────────────────────────────

async def _demo():
    total_samples = 0
    async for timestamps, channels in read_stdin_chunks(batch_size=512):
        total_samples += len(timestamps)
        print(
            f"batch: {len(timestamps):>5} rows | "
            f"t0={timestamps[0]:.4f} | "
            f"ch1 mean={channels[:,0].mean():.6f} | "
            f"total={total_samples}",
            file=sys.stderr,
        )
    print(f"\nDone — {total_samples} samples ingested.", file=sys.stderr)


if __name__ == "__main__":
    try:
        asyncio.run(_demo())
    except KeyboardInterrupt:
        pass
    finally:
        # Suppress broken-pipe noise on exit
        try:
            sys.stdout.close()
        except Exception:
            pass
        try:
            sys.stderr.close()
        except Exception:
            pass
