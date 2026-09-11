#!/usr/bin/env python3
"""Create the numerical contents of Figures 3 and 4 from the model input."""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "h2_task_level_long.csv"
OUTPUT = ROOT / "results"


def main() -> None:
    with INPUT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    # Figure 3: rows are delegation choices, columns are reported trust (0--4).
    choices = [("never", "Reject AI assistance"), ("verify", "Use with verification"), ("totally", "Fully delegate")]
    with (OUTPUT / "figure_3_joint_distribution.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["delegation_choice", "trust_0", "trust_1", "trust_2", "trust_3", "trust_4"])
        for choice, label in choices:
            counts = Counter(int(row["post_trust_raw"]) for row in rows if row["delegation_label"] == choice)
            writer.writerow([label, *(counts[level] for level in range(5))])

    # Figure 4: trust 3--4 is high; only full delegation counts as high delegation.
    zones = Counter()
    for row in rows:
        trust = "high" if int(row["post_trust_raw"]) >= 3 else "low"
        delegation = "high" if row["delegation_label"] == "totally" else "low"
        zones[(trust, delegation)] += 1
    with (OUTPUT / "figure_4_quadrants.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trust", "delegation", "trials"])
        for trust in ("low", "high"):
            for delegation in ("low", "high"):
                writer.writerow([trust, delegation, zones[(trust, delegation)]])

    total = len(rows)
    high_trust = sum(count for (trust, _), count in zones.items() if trust == "high")
    high_trust_retained = zones[("high", "low")]
    low_trust_full = zones[("low", "high")]
    with (OUTPUT / "paper_descriptives.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"Trials: {total}\n")
        handle.write(f"Participants: {len({row['username'] for row in rows})}\n")
        for choice, label in choices:
            count = sum(row["delegation_label"] == choice for row in rows)
            handle.write(f"{label}: {count} ({count / total:.1%})\n")
        handle.write(f"High trust with retained control: {high_trust_retained} ({high_trust_retained / high_trust:.1%} of high-trust trials)\n")
        handle.write(f"Low trust with full delegation: {low_trust_full}\n")


if __name__ == "__main__":
    main()
