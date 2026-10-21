import random
import importlib
from types import SimpleNamespace
import numpy as np
import torch
from timm.models import create_model
from channels import channels_down_stream_dataset
import models.BrainPro.BrainPro
import json
from pathlib import Path
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
import copy
import os
from timeit import default_timer as timer
from tqdm import tqdm
from einops import rearrange
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, cohen_kappa_score, roc_auc_score, \
    precision_recall_curve, auc, r2_score, mean_squared_error


def load_downstream_configs():
    config_path = (
        Path(__file__).resolve().parents[1]
        / 'configs'
        / 'downstream.json'
    )

    with config_path.open('r', encoding='utf-8') as handle:
        document = json.load(handle)

    common = document['common']

    return {
        dataset: {**common, **settings}
        for dataset, settings in document['datasets'].items()
    }

configs = load_downstream_configs()

def get_model(args, pretrained=True):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=pretrained,
        init_ckpt=args.foundation_dir,
        seq_length=args.seq_length,
        number_class=args.num_of_classes,
        number_channel=args.num_chan,
        channel_names=channels_down_stream_dataset[args.downstream_dataset],
        cognition_type=args.cognition_type,
        params=args,
        linear_prob=args.frozen,
        dropout=args.dropout,
        T=args.T,
        mode=args.mode,
    )
    return model

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

LOADERS = {'FACED':'faced','SEED-V':'seedv','BCI-IV-2A':'bciciv2a',
           'SHU':'shu','MentalArithmetic':'stress','ATTEN':'atten',
           'SEED-VII':'seed7','Speech':'speech','Mumtaz2016':'mumtaz'}

def parameters(dataset, data_dir, checkpoint, output_dir, seed, gpu):
    cfg = configs[dataset]

    return SimpleNamespace(
        model=cfg['model'],
        downstream_dataset=dataset,
        datasets_dir=str(data_dir),
        foundation_dir=str(checkpoint),
        model_dir=str(output_dir),

        seq_length=cfg['seconds'] * 10,
        num_t=cfg['seconds'],
        num_chan=cfg['channels'],
        num_of_classes=cfg['classes'],
        cognition_type=cfg['branch'],

        mode=cfg['mode'],
        T=cfg['T'],
        dropout=cfg['dropout'],
        lr=cfg['lr'],
        epochs=cfg['epochs'],
        cosine_epochs=cfg['cosine_epochs'],

        seed=seed,
        cuda=gpu,
        batch_size=cfg['batch_size'],
        weight_decay=cfg['weight_decay'],
        optimizer=cfg['optimizer'],
        clip_value=cfg['clip_value'],
        label_smoothing=cfg['label_smoothing'],
        warmup_epochs=cfg['warmup_epochs'],
        multi_lr=cfg['multi_lr'],
        frozen=cfg['frozen'],

        num_workers=cfg['num_workers'],
        patch_size=cfg['patch_size'],
        patch_stride=cfg['patch_stride'],
        classifier=cfg['classifier'],
        use_pretrained_weights=cfg['use_pretrained_weights'],
    )

def loaders(params):
    module = importlib.import_module('dataset_downstream.'+LOADERS[params.downstream_dataset]+'_dataset')
    return module.LoadDataset(params).get_data_loader()


def cosine_annealing_lr(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


class Trainer(object):
    def __init__(self, params, data_loader, model):
        self.params = params
        self.data_loader = data_loader

        self.val_eval = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])

        self.model = model.cuda()

        if self.params.downstream_dataset in ['FACED', 'FACED-F', 'SEED-VII', 'SEED-V', 'PhysioNet-MI', 'ISRUC', 'ISRUC-S3',
                                              'BCIC2020-3', 'TUEV', 'BCI-IV-2A', 'HMC', 'Speech']:
            self.criterion = CrossEntropyLoss(label_smoothing=self.params.label_smoothing).cuda()
        elif self.params.downstream_dataset in ['SHU', 'CHB-MIT', 'Mumtaz2016', 'MentalArithmetic', 'TUAB', 'ATTEN']:
            self.criterion = BCEWithLogitsLoss().cuda()
        elif self.params.downstream_dataset in ['SEED-VIG', 'MOBI']:
            self.criterion = MSELoss().cuda()

        self.best_model_states = None

        backbone_params = []
        other_params = []
        if self.params.model in ['LaBraM', 'BrainPro', 'BIOT', 'CBraMod']:
            for name, param in self.model.named_parameters():
                if not param.is_floating_point():
                    continue  # don't touch int/bool params
                if "backbone" in name or 'encoders' in name:
                    backbone_params.append(param)

                    if params.frozen:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                else:
                    other_params.append(param)

        if self.params.optimizer == 'AdamW':
            if self.params.multi_lr: # set different learning rates for different modules
                print("Using Multi LR optimizer!")
                self.optimizer = torch.optim.AdamW([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr * 5}
                ], weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr,
                                                   weight_decay=self.params.weight_decay)
        else:
            if self.params.multi_lr:
                print("Using Multi LR optimizer!")
                self.optimizer = torch.optim.SGD([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr * 5}
                ],  momentum=0.9, weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.lr, momentum=0.9,
                                                 weight_decay=self.params.weight_decay)

        self.data_length = len(self.data_loader['train'])
        if self.params.warmup_epochs > 0:
            print("Using warmup learning rate scheduler!")
            lr_list = cosine_annealing_lr(1.0, 0.2, self.params.cosine_epochs, self.data_length, warmup_epochs=self.params.warmup_epochs)
            self.optimizer_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: lr_list[step - 1])
        else:
            self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.params.cosine_epochs * self.data_length, eta_min=1e-6
            )
        print(self.model)

    def train_for_multiclass(self):
        f1_best = 0
        kappa_best = 0
        acc_best = 0
        cm_best = None
        best_f1_epoch = 0
        self.best_model_states = copy.deepcopy(self.model.state_dict())  # use the original model as default
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                if self.params.model in ['LaBraM', 'BrainPro', 'EEGNet', 'Conformer']:
                    if x.ndim == 4:  # (B, C, S, L)
                        if self.params.downstream_dataset in ['ISRUC', 'ISRUC-S3']:
                            x = rearrange(x, "B S C L -> (B S) C L")
                            y = rearrange(y, 'B S -> (B S)')
                        else:
                            x = rearrange(x, "B C S L -> B C (S L)")

                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)

                loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                acc, kappa, f1, cm = self.val_eval.get_metrics_for_multiclass(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        kappa,
                        f1,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                print(cm)
                if kappa > kappa_best:
                    print("kappa increasing....saving weights !! ")
                    print("Val Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                        acc,
                        kappa,
                        f1,
                    ))
                    best_f1_epoch = epoch + 1
                    acc_best = acc
                    kappa_best = kappa
                    f1_best = f1
                    cm_best = cm
                    self.best_model_states = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            acc, kappa, f1, cm = self.test_eval.get_metrics_for_multiclass(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                    acc,
                    kappa,
                    f1,
                )
            )
            print(cm)
            model_save_path = os.path.join(
                self.params.model_dir, self.params.downstream_dataset
            )
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)

            folder_name = os.path.basename(self.params.datasets_dir)
            model_file_path = os.path.join(
                model_save_path,
                "epoch{}_acc_{:.5f}_kappa_{:.5f}_f1_{:.5f}_seed{}_{}.pth".format(
                    best_f1_epoch, acc, kappa, f1, self.params.seed, folder_name)
            )
            torch.save(self.model.state_dict(), model_file_path)
            print("model save in " + model_file_path)

    def train_for_binaryclass(self):
        acc_best = 0
        roc_auc_best = 0
        pr_auc_best = 0
        cm_best = None
        best_f1_epoch = 0
        self.best_model_states = copy.deepcopy(self.model.state_dict())  # use the original model as default
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                if self.params.model in ['LaBraM', 'BrainPro', 'EEGNet', 'Conformer']:
                    if x.ndim == 4:  # (B, C, S, L)
                        x = rearrange(x, "B C S L -> B C (S L)")
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)
                pred = pred.squeeze(1)
                y = y.float()            # LongTensor → FloatTensor

                loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                acc, pr_auc, roc_auc, cm = self.val_eval.get_metrics_for_binaryclass(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        pr_auc,
                        roc_auc,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                print(cm)
                if roc_auc > roc_auc_best:
                    print("roc_auc increasing....saving weights !! ")
                    print("Val Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                        acc,
                        pr_auc,
                        roc_auc,
                    ))
                    best_f1_epoch = epoch + 1
                    acc_best = acc
                    pr_auc_best = pr_auc
                    roc_auc_best = roc_auc
                    cm_best = cm
                    self.best_model_states = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            acc, pr_auc, roc_auc, cm = self.test_eval.get_metrics_for_binaryclass(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                    acc,
                    pr_auc,
                    roc_auc,
                )
            )
            print(cm)

            model_save_path = os.path.join(
                self.params.model_dir, self.params.downstream_dataset
            )
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            folder_name = os.path.basename(self.params.datasets_dir)
            model_file_path = os.path.join(
                model_save_path,
                "epoch{}_acc_{:.5f}_pr_{:.5f}_roc_{:.5f}_seed{}_{}.pth".format(
                    best_f1_epoch, acc, pr_auc, roc_auc, self.params.seed, folder_name))
            torch.save(self.model.state_dict(), model_file_path)
            print("model save in " + model_file_path)

    def train_for_regression(self):
        corrcoef_best = 0
        r2_best = 0
        rmse_best = 0
        best_r2_epoch = 0
        self.best_model_states = copy.deepcopy(self.model.state_dict())  # use the original model as default
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                if self.params.model in ['LaBraM', 'BrainPro', 'EEGNet', 'Conformer']:
                    if x.ndim == 4:  # (B, C, S, L)
                        x = rearrange(x, "B C S L -> B C (S L)")

                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)

                loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                corrcoef, r2, rmse = self.val_eval.get_metrics_for_regression(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        corrcoef,
                        r2,
                        rmse,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                if r2 > r2_best:
                    print("r2 increasing....saving weights !! ")
                    print("Val Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format(
                        corrcoef,
                        r2,
                        rmse,
                    ))
                    best_r2_epoch = epoch + 1
                    corrcoef_best = corrcoef
                    r2_best = r2
                    rmse_best = rmse
                    self.best_model_states = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            corrcoef, r2, rmse = self.test_eval.get_metrics_for_regression(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format(
                    corrcoef,
                    r2,
                    rmse,
                )
            )

            model_save_path = os.path.join(
                self.params.model_dir, self.params.downstream_dataset
            )
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)

            model_file_path = os.path.join(
                model_save_path,
                "epoch{}_corrcoef_{:.5f}_r2_{:.5f}_rmse_{:.5f}.pth".format(best_r2_epoch, corrcoef, r2, rmse))

            torch.save(self.model.state_dict(), model_file_path)
            print("model save in " + model_file_path)

class Evaluator:
    def __init__(self, params, data_loader):
        self.params = params
        self.data_loader = data_loader

    def get_metrics_for_multiclass(self, model):
        model.eval()

        truths = []
        preds = []
        for x, y in tqdm(self.data_loader, mininterval=1):
            if self.params.model in ['LaBraM', 'BrainPro', 'EEGNet', 'Conformer']:
                if self.params.downstream_dataset in ['ISRUC', 'ISRUC-S3']:
                    x = rearrange(x, "B S C L -> (B S) C L")
                    y = rearrange(y, 'B S -> (B S)')
                else:
                    x = rearrange(x, "B C S L -> B C (S L)")

            x = x.cuda()
            y = y.cuda()
            pred = model(x)

            pred_y = torch.max(pred, dim=-1)[1]

            truths += y.cpu().reshape(-1).numpy().tolist()
            preds += pred_y.cpu().reshape(-1).numpy().tolist()

        truths = np.array(truths)
        preds = np.array(preds)
        print(np.unique(truths))
        acc = balanced_accuracy_score(truths, preds)
        f1 = f1_score(truths, preds, average='weighted')
        kappa = cohen_kappa_score(truths, preds)
        cm = confusion_matrix(truths, preds)
        return acc, kappa, f1, cm

    def get_metrics_for_binaryclass(self, model):
        model.eval()

        truths = []
        preds = []
        scores = []
        for x, y in tqdm(self.data_loader, mininterval=1):
            if self.params.model in ['LaBraM', 'BrainPro', 'EEGNet', 'Conformer']:
                x = rearrange(x, "B C S L -> B C (S L)")

            x = x.cuda()
            y = y.cuda()
            pred = model(x)

            score_y = torch.sigmoid(pred)
            pred_y = torch.gt(score_y, 0.5).long()
            truths += y.long().cpu().reshape(-1).numpy().tolist()
            preds += pred_y.cpu().reshape(-1).numpy().tolist()
            scores += score_y.cpu().numpy().tolist()

        truths = np.array(truths)
        preds = np.array(preds)
        scores = np.array(scores)
        acc = balanced_accuracy_score(truths, preds)
        roc_auc = roc_auc_score(truths, scores)
        precision, recall, thresholds = precision_recall_curve(truths, scores, pos_label=1)
        pr_auc = auc(recall, precision)
        cm = confusion_matrix(truths, preds)
        return acc, pr_auc, roc_auc, cm

    def get_metrics_for_regression(self, model):
        model.eval()

        truths = []
        preds = []
        for x, y in tqdm(self.data_loader, mininterval=1):
            if self.params.model in ['LaBraM', 'BrainPro', 'EEGNet', 'Conformer']:
                x = rearrange(x, "B C S L -> B C (S L)")

            x = x.cuda()
            y = y.cuda()
            pred = model(x)

            truths += y.cpu().reshape(-1).numpy().tolist()
            preds += pred.cpu().reshape(-1).numpy().tolist()

        truths = np.array(truths)
        preds = np.array(preds)
        corrcoef = np.corrcoef(truths, preds)[0, 1]
        r2 = r2_score(truths, preds)
        rmse = mean_squared_error(truths, preds) ** 0.5
        return corrcoef, r2, rmse