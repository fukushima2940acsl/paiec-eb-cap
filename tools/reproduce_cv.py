"""Run the recorded screen or confirmation protocol with frozen benchmark folds."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from release_utils import ROOT


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase', choices=['screen', 'confirm'], default='confirm')
    args = ap.parse_args()
    reference = json.loads((ROOT / 'configs/reference_eval.json').read_text())
    work_folds = json.loads((ROOT / 'work/folds.json').read_text())
    released_folds = json.loads((ROOT / 'configs/folds.json').read_text())
    if work_folds != released_folds:
        raise SystemExit('公開版と異なるfoldです。既存foldを置き換えず停止します')
    command = [sys.executable, '-u', str(ROOT / 'experiments/run_experiments.py'), '--root', str(ROOT),
               '--reference', str(ROOT / 'configs/reference_eval.json')]
    if args.phase == 'screen':
        command += ['--seeds', '0', '1', '2', '--n-draws', str(reference['args']['n_draws'])]
    else:
        command += ['--methods', 'eb_cap', 'beta11', '--seeds', '3', '4', '5', '--n-draws', '60']
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
