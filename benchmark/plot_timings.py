"""Draws benchmark timings as graphs.

Reads one or more timings.csv files written by benchmark/run.jl, or any file with the
same columns (label, backend, precision, problem, item, call, seconds, and optionally a
plain-English description and a block), and draws one image per measurement with two
panels:

* left: the histograms of the timings, one per label, drawn over each other. Two sets of
  timings that genuinely differ form separate humps; two that only scatter overlap.
* right: every timing against its position in the order the calls were made. Slow
  drifts, such as a processor or GPU heating up, show here and are hidden by a single
  summary number. If the file has a "block" column, the start of each block is marked.

Usage, from the repository root:

    python3 benchmark/plot_timings.py benchmark/results/before/timings.csv \\
        benchmark/results/after/timings.csv --out benchmark/results/plots

Add --only mobility (or any other text) to draw only the measurements whose item name
contains that text.
"""

import argparse
import collections
import csv
import os

import matplotlib

matplotlib.use("Agg")  # write image files; no window is needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

UNITS = (("s", 1.0), ("ms", 1e-3), ("µs", 1e-6))


def unit_for(seconds):
    """The unit, as (name, length in seconds), that writes `seconds` below 1000."""
    for name, length in UNITS:
        if seconds >= 0.9995 * length:
            return name, length
    return UNITS[-1]


def read_timings(paths):
    """Every timing in `paths`, grouped by measurement and then by label.

    Returns {(backend, problem, precision, item): {label: [(call, seconds, block)]}} and
    the plain-English description of each measurement, where the files give one.
    """
    measurements = collections.defaultdict(lambda: collections.defaultdict(list))
    descriptions = {}
    for path in paths:
        with open(path, newline="") as file:
            for row in csv.DictReader(file):
                key = (row["backend"], row["problem"], row["precision"], row["item"])
                timing = (int(row["call"]), float(row["seconds"]), row.get("block"))
                measurements[key][row["label"]].append(timing)
                descriptions[key] = row.get("description") or row["item"]
    return measurements, descriptions


def block_starts(timings):
    """The call numbers at which a new block begins, for timings sorted by call."""
    return [
        call
        for position, (call, _, block) in enumerate(timings)
        if block and position > 0 and block != timings[position - 1][2]
    ]


def plot_measurement(key, description, timings_by_label, folder):
    """Draws one measurement and returns the image's file name."""
    backend, problem, precision, item = key
    every_timing = np.array(
        [seconds for timings in timings_by_label.values() for _, seconds, _ in timings]
    )
    unit, length = unit_for(np.median(every_timing))
    bins = np.histogram_bin_edges(every_timing / length, bins=30)

    figure, (histogram, sequence) = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, timings in timings_by_label.items():
        timings = sorted(timings)
        calls = [call for call, _, _ in timings]
        times = np.array([seconds for _, seconds, _ in timings]) / length
        median = np.median(times)
        histogram.hist(
            times, bins=bins, alpha=0.5, label=f"{label}: median {median:.3g} {unit}"
        )
        sequence.plot(calls, times, ".", markersize=3, label=label)
        for call in block_starts(timings):
            sequence.axvline(call - 0.5, color="grey", linewidth=0.5)

    histogram.set_xlabel(f"time per call ({unit})")
    histogram.set_ylabel("number of calls")
    histogram.legend()
    sequence.set_xlabel("call, in the order the calls were made")
    sequence.set_ylabel(f"time per call ({unit})")
    sequence.legend()
    figure.suptitle(f"{backend.upper()}, problem {problem}, {precision}: {description}")
    figure.tight_layout()

    name = f"{backend}_{problem}_{precision}_{item}.png"
    figure.savefig(os.path.join(folder, name), dpi=120)
    plt.close(figure)
    return name


def main():
    parser = argparse.ArgumentParser(description="Draws benchmark timings as graphs.")
    parser.add_argument("files", nargs="+", help="timings.csv files to draw")
    parser.add_argument(
        "--out", default="benchmark/results/plots", help="folder for the images"
    )
    parser.add_argument(
        "--only", default="", help="draw only items whose name contains this text"
    )
    arguments = parser.parse_args()

    os.makedirs(arguments.out, exist_ok=True)
    measurements, descriptions = read_timings(arguments.files)
    for key in sorted(measurements):
        if arguments.only in key[3]:
            name = plot_measurement(
                key, descriptions[key], measurements[key], arguments.out
            )
            print(os.path.join(arguments.out, name))


if __name__ == "__main__":
    main()
