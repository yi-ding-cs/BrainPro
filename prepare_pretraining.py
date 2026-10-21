"""Build BrainPro-base training HDF5 from per-corpus merged recordings."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'script'))
from script.pretraining_data import build_dataset

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path(__file__).resolve().parent / 'configs' / 'pretraining_data.json')
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='Output training .h5 file')
    p.add_argument('--metadata', type=Path, required=True, help='Output metadata .pkl file')
    p.add_argument('--stride', type=int, default=2000, help='Samples between 2000-sample windows')
    args = p.parse_args()
    print(json.dumps(build_dataset(args.config, args.input_dir, args.output, args.metadata, args.stride), indent=2))

if __name__ == '__main__':
    main()
