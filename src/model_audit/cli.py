"""Command line: ``model-audit run --config config.yaml`` and ``model-audit download``."""

from __future__ import annotations

import argparse
import sys

from . import data as data_mod
from .config import load_config, resolve
from .pipeline import run_audit
from .plots import make_all
from .report import build_report
from .status import Status


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="model-audit", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run all checks and write the report and figures")
    run.add_argument("--config", default="config.yaml")
    dl = sub.add_parser("download", help="download and cache the public dataset as CSV")
    dl.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.cmd == "download":
        p = data_mod.download_taiwan(resolve(cfg, cfg["data"]["path"]))
        print(f"cached {p}")
        return 0

    out = run_audit(cfg)
    report_path = resolve(cfg, cfg["output"]["report"])
    figs = make_all(out, resolve(cfg, cfg["output"]["figures_dir"]), cfg)
    build_report(out, cfg, figs, report_path)
    for r in out.results:
        print(f"{r.status.label:5s}  {r.check_id:22s} {r.name}")
    n = {s: sum(r.status == s for r in out.results) for s in Status}
    print(f"\n{n[Status.FAIL]} Fail, {n[Status.WATCH]} Watch, {n[Status.PASS]} Pass -> {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
