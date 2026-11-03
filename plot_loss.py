# -*- coding: utf-8 -*-
"""Plot RetinexNet training losses from a saved console log.

Run from the RetinexNet directory in Anaconda Prompt:
    conda activate retinex-tf1
    python main.py --phase=train --use_gpu=0

main.py saves each fresh run under runs/YYYY-MM-DD/<run-id>/, including
train.log and loss.csv. Each run has its own log, so no shell redirection is
needed. With no log argument, this script plots the latest completed run.

Plot with the existing environment that already contains Matplotlib:
    conda run -n pytorch python plot_loss.py
    conda run -n pytorch python plot_loss.py path/to/train.log --x-axis batch
    conda run -n pytorch python plot_loss.py path/to/train.log -o loss_curve.svg

Use one training run (including its resumes) per log. Repeated phase/epoch/
batch entries keep their last value. Epoch means include only logged batches;
an unfinished epoch therefore shows a partial mean. This script does not
import TensorFlow, load checkpoints, or start training.
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
import re


LOSS_PATTERN = re.compile(
    r"(?P<phase>Decom|Relight)\s+Epoch:\s*\[\s*(?P<epoch>\d+)\s*\]"
    r"\s*\[\s*(?P<batch>\d+)\s*/\s*(?P<total>\d+)\s*\]"
    r"[^\r\n]*?\bloss:\s*"
    r"(?P<loss>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    r"|[+-]?(?:nan|inf(?:inity)?))(?=\s|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LossRecord:
    phase: str
    epoch: int
    batch: int
    total: int
    loss: float


def read_losses(log_path):
    """Read UTF-8, Windows console, and BOM-marked UTF-16 logs."""
    with log_path.open("rb") as stream:
        header = stream.read(4)
    if header.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        encoding = "utf-32"
    elif header.startswith((b"\xff\xfe", b"\xfe\xff")):
        encoding = "utf-16"
    else:
        encoding = "utf-8-sig"

    records = {}
    skipped = 0
    duplicates = 0
    with log_path.open("r", encoding=encoding, errors="replace") as stream:
        for line in stream:
            match = LOSS_PATTERN.search(line)
            if match is None:
                continue
            record = LossRecord(
                phase=match.group("phase").capitalize(),
                epoch=int(match.group("epoch")),
                batch=int(match.group("batch")),
                total=int(match.group("total")),
                loss=float(match.group("loss")),
            )
            if (not math.isfinite(record.loss) or record.epoch < 1
                    or not 1 <= record.batch <= record.total):
                skipped += 1
                continue
            key = (record.phase, record.epoch, record.batch)
            duplicates += key in records
            records[key] = record

    if not records:
        raise ValueError(
            "No valid training losses found. Expected lines such as: "
            "Decom Epoch: [ 1] [ 1/ 30] time: 1.0000, loss: 0.123456. "
            "Use a training log; --phase=test does not produce training losses."
        )
    if skipped:
        print("Warning: skipped {} invalid/non-finite loss entries.".format(skipped))
    if duplicates:
        print("Kept the last value for {} repeated batch entries.".format(duplicates))
    return sorted(records.values(), key=lambda row: (row.phase, row.epoch, row.batch))


def loss_series(records, x_axis):
    """Return chronological x/y values for one training phase."""
    records = sorted(records, key=lambda row: (row.epoch, row.batch))
    if x_axis == "batch":
        return list(range(1, len(records) + 1)), [row.loss for row in records]
    by_epoch = defaultdict(list)
    for row in records:
        by_epoch[row.epoch].append(row.loss)
    epochs = sorted(by_epoch)
    return epochs, [math.fsum(by_epoch[epoch]) / len(by_epoch[epoch])
                    for epoch in epochs]


def plot_losses(records, output_path, x_axis):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError as error:
        raise ValueError(
            "Matplotlib is required. Run this script in the existing pytorch "
            "environment: conda run -n pytorch python plot_loss.py train.log"
        ) from error

    phases = [phase for phase in ("Decom", "Relight")
              if any(row.phase == phase for row in records)]
    colors = {"Decom": "#2563EB", "Relight": "#D97706"}
    titles = {"Decom": "DecomNet loss", "Relight": "RelightNet loss"}
    with plt.rc_context({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False}):
        fig, axes = plt.subplots(1, len(phases), squeeze=False,
                                 figsize=(6.2 * len(phases), 4.5))
        for ax, phase in zip(axes[0], phases):
            phase_records = [row for row in records if row.phase == phase]
            x, y = loss_series(phase_records, x_axis)
            ax.plot(x, y, color=colors[phase], linewidth=1.5,
                    marker="o" if x_axis == "epoch" or len(x) == 1 else None,
                    markersize=3, label="Epoch mean" if x_axis == "epoch" else "Batch loss")
            ax.set(title=titles[phase], ylabel="Training loss",
                   xlabel="Epoch" if x_axis == "epoch" else "Logged batch")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if len(x) == 1:
                ax.set_xlim(x[0] - 0.5, x[0] + 0.5)
                ax.set_xticks(x)
            ax.grid(True, color="#D1D5DB", alpha=0.65, linewidth=0.6)
            ax.legend(frameon=False)
            print("{}: {} batches, {} epochs".format(
                phase, len(phase_records), len({row.epoch for row in phase_records})))
        fig.suptitle("RetinexNet Training Loss", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", nargs="?", type=Path, default=None,
                        help="Saved training log (default: latest completed run)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output PNG, SVG, or PDF (default: loss_curve.png beside the log)")
    parser.add_argument("--x-axis", choices=("epoch", "batch"), default="epoch",
                        help="Epoch mean or loss of each logged batch (default: epoch)")
    parser.add_argument("--output_root", type=Path,
                        default=Path(__file__).resolve().parent / "runs",
                        help="Training archive root when selecting the latest run")
    args = parser.parse_args()
    if args.log is None:
        from run_storage import load_run
        try:
            archive_root = args.output_root.expanduser()
            if not archive_root.is_absolute():
                archive_root = Path(__file__).resolve().parent / archive_root
            _, configuration = load_run(archive_root)
            args.log = Path(configuration["paths"]["train_log"])
        except (OSError, ValueError) as error:
            parser.exit(1, "Error: {}\n".format(error))
    if args.output is None:
        args.output = args.log.parent / "loss_curve.png"
    if args.output.suffix.lower() not in (".png", ".svg", ".pdf"):
        parser.error("Output filename must end in .png, .svg, or .pdf.")
    try:
        records = read_losses(args.log)
        plot_losses(records, args.output, args.x_axis)
    except (OSError, ValueError) as error:
        parser.exit(1, "Error: {}\n".format(error))
    if args.x_axis == "epoch":
        print("Epoch means use logged batches only, including any partial epoch.")
    print("Saved: {}".format(args.output.resolve()))


if __name__ == "__main__":
    main()
