"""Train BrainPro-base using a prepared HDF5 dataset and its metadata."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'script'))
import h5py
import os
from script.engine_pretraining import train
import argparse
import json

def get_args():
    parser = argparse.ArgumentParser('BrainPro-base pre-training', add_help=True)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--save_ckpt_freq', default=20, type=int)
    parser.add_argument('--dataset_exist', action='store_true')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--meta_data_path', type=str, default='./dataset_metadata.pkl')

    # Pre-training data
    parser.add_argument('--pretraining_data_path', type=str, default='./data/pretraining_sources')
    parser.add_argument('--window_size', default=2000, type=int)
    parser.add_argument('--stride_size', default=2000, type=int)
    parser.add_argument('--patch_size', default=20, type=int)
    parser.add_argument('--patch_stride', default=20, type=int)
    parser.add_argument('--merged_pretraining_data_path', type=str, default='./data/pretraining.h5')

    # Model parameters
    parser.add_argument('--model', default='brainpro_base', choices=['brainpro_base'], type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_true', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=False)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=True)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float,
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--input_size', default=1600, type=int,
                        help='EEG input size for backbone')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)

    parser.add_argument(
        '--config',
        type=Path,
        default=Path(__file__).resolve().parent / 'configs' / 'pretraining.json',
        help='Pre-training hyperparameter configuration',
    )

    # First find which JSON configuration to load.
    config_args, _ = parser.parse_known_args()

    with config_args.config.open('r', encoding='utf-8') as handle:
        settings = json.load(handle)

    if not isinstance(settings, dict):
        parser.error('Pre-training configuration must be a JSON object.')

    # Reject misspelled or unsupported configuration fields.
    valid_keys = {
        action.dest
        for action in parser._actions
        if action.dest not in {'help', 'config'}
    }
    unknown = set(settings) - valid_keys
    if unknown:
        parser.error(f'Unknown configuration fields: {sorted(unknown)}')

    # JSON replaces parser defaults; explicitly supplied CLI values win.
    parser.set_defaults(**settings)
    return parser.parse_args()


def main():
    args = get_args()
    if args.num_workers < 1:
        raise ValueError('The preserved data loader requires --num_workers >= 1.')
    if args.batch_size < 1 or args.epochs < 1:
        raise ValueError('Batch size and epochs must be positive.')
    for value in [args.merged_pretraining_data_path,args.meta_data_path]:
        if not Path(value).is_file():
            raise FileNotFoundError(value)
    with h5py.File(args.merged_pretraining_data_path,'r') as data:
        if data['eeg'].shape[1:] != (60,2002) or data['labels'].shape != (len(data['eeg']),2):
            raise ValueError('Use prepare_pretraining.py to build eeg[N,60,2002] and labels[N,2].')
        if len(data['eeg']) < args.batch_size*int(os.environ.get('WORLD_SIZE','1')):
            raise ValueError('Dataset is smaller than one global batch; reduce --batch_size.')
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True,exist_ok=True)
    train(args)

if __name__ == '__main__':
    main()
