"""Fine-tune BrainPro-base with the reproduced manuscript settings."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'script'))
import torch
from script.engine_downstream import parameters,setup_seed,loaders,get_model,Trainer, configs

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',choices=configs)
    p.add_argument('--data-dir',type=Path,required=True,help='Preprocessed LMDB directory')
    p.add_argument('--checkpoint',type=Path,required=True,help='BrainPro-base pretraining checkpoint')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--gpu',type=int,default=0,help='CUDA device index in this process')
    args=p.parse_args()
    if not (args.data_dir/'data.mdb').is_file():
        raise FileNotFoundError(args.data_dir/'data.mdb')
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    run_dir=args.output_dir.resolve()/args.dataset
    config_path=run_dir/f'config_seed{args.seed}.json'
    if config_path.exists() or (run_dir.exists() and any(run_dir.glob(f'*_seed{args.seed}_*.pth'))):
        raise FileExistsError(f'Seed {args.seed} already has outputs in {run_dir}; use a fresh output directory to rerun it.')

    params=parameters(args.dataset,args.data_dir.resolve(),args.checkpoint.resolve(),args.output_dir.resolve(),args.seed,args.gpu)
    setup_seed(params.seed)
    torch.cuda.set_device(params.cuda)
    data_loader=loaders(params)
    model=get_model(params,pretrained=True)
    run_dir.mkdir(parents=True,exist_ok=True)
    with config_path.open('x') as handle:
        handle.write(json.dumps(vars(params),indent=2)+'\n')
    trainer=Trainer(params,data_loader,model)
    if params.num_of_classes>1:
        trainer.train_for_multiclass()
    else:
        trainer.train_for_binaryclass()
    print(f'Completed. Checkpoints and run configuration: {run_dir}')

if __name__=='__main__':
    main()
