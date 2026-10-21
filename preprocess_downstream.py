"""Run the evaluated preprocessing for one downstream dataset."""
import argparse
import os
import runpy
from pathlib import Path

FILES = {'FACED':'preprocessing_faced.py','SEED-V':'preprocessing_seed5.py',
    'BCI-IV-2A':'preprocessing_bciciv2a.py','SHU':'preprocessing_shu.py',
    'MentalArithmetic':'preprocessing_stress.py','ATTEN':'preprocessing_atten.py',
    'decode_attention':'decode_attention.py',
    'SEED-VII':'preprocessing_seed7.py','Speech':'pre_processing_speech.py',
    'Mumtaz2016':'preprocessing_mumtaz.py','decode_seed_vii':'pre_process_seed_vii.py'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',choices=FILES)
    p.add_argument('--montage',type=Path,help='SEED-VII channel_62_pos.locs')
    p.add_argument('--trigger-dir',type=Path,help='SEED-VII save_info directory')
    p.add_argument('--raw-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    raw,output=args.raw_dir.resolve(),args.output_dir.resolve()
    if not raw.is_dir():
        raise FileNotFoundError(raw)
    if raw==output or raw in output.parents or output in raw.parents:
        raise ValueError('Use separate, non-nested input and output directories.')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Refusing to overwrite nonempty output: {output}')
    if args.dataset=='decode_seed_vii':
        if not args.montage or not args.montage.is_file() or not args.trigger_dir or not args.trigger_dir.is_dir():
            raise ValueError('decode_seed_vii requires --montage and --trigger-dir.')
        os.environ['BRAINPRO_MONTAGE']=str(args.montage.resolve())
        os.environ['BRAINPRO_TRIGGER_DIR']=str(args.trigger_dir.resolve())
    output.mkdir(parents=True,exist_ok=True)
    os.environ['BRAINPRO_RAW_DIR']=str(raw)+os.sep
    os.environ['BRAINPRO_OUTPUT_DIR']=str(output)
    runpy.run_path(str(Path(__file__).resolve().parent/'script/preprocessing_downstream'/FILES[args.dataset]),run_name='__main__')
    print(f'Preprocessed {args.dataset}: {output}')

if __name__=='__main__':
    main()
