import os
import glob
import inspect
import logging
import warnings
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.model_LN_prompt import Model
from src.dataset_retrieval import Sketchy
from experiments.options import opts

if __name__ == '__main__':
    logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
    warnings.filterwarnings('ignore', module='pytorch_lightning')

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')

    dataset_transforms = Sketchy.data_transform(opts)

    train_dataset = Sketchy(opts, dataset_transforms, mode='train', return_orig=False)
    val_dataset = Sketchy(opts, dataset_transforms, mode='val', used_cat=train_dataset.all_categories, return_orig=False)

    loader_kwargs = {
        'batch_size': opts.batch_size,
        'num_workers': opts.workers,
        'pin_memory': True,
    }
    if opts.workers > 0:
        loader_kwargs['persistent_workers'] = True

    train_loader = DataLoader(dataset=train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(dataset=val_dataset, shuffle=False, **loader_kwargs)

    if opts.logger == 'tensorboard':
        logger = TensorBoardLogger('tb_logs', name=opts.exp_name)
    elif opts.logger == 'csv':
        logger = CSVLogger('logs', name=opts.exp_name)
    else:
        logger = False

    checkpoint_metric = 'acc1' if opts.retrieval_level == 'fine_grain' else 'mAP'
    checkpoint_callback = ModelCheckpoint(
        monitor=checkpoint_metric,
        dirpath='saved_models/%s'%opts.exp_name,
        filename="{epoch:02d}-{%s:.4f}" % checkpoint_metric,
        mode='max',
        save_last=True)

    ckpt_path = os.path.join('saved_models', opts.exp_name, 'last.ckpt')
    if not os.path.exists(ckpt_path):
        ckpt_path = None
    else:
        pass

    trainer_kwargs = {
        'min_epochs': 1,
        'max_epochs': opts.max_epochs,
        'benchmark': True,
        'logger': logger,
        'enable_progress_bar': opts.progress_bar,
        'enable_model_summary': False,
        'num_sanity_val_steps': 0,
        # 'val_check_interval': 10,
        # 'accumulate_grad_batches': 1,
        'check_val_every_n_epoch': opts.check_val_every_n_epoch,
        'callbacks': [checkpoint_callback],
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if 'accelerator' in trainer_params:
        trainer_kwargs.update({'accelerator': 'auto', 'devices': 'auto'})
    else:
        trainer_kwargs['gpus'] = -1
    if 'resume_from_checkpoint' in trainer_params:
        trainer_kwargs['resume_from_checkpoint'] = ckpt_path

    trainer = Trainer(**trainer_kwargs)

    model = Model()

    fit_kwargs = {}
    if ckpt_path is not None and 'ckpt_path' in inspect.signature(trainer.fit).parameters:
        fit_kwargs['ckpt_path'] = ckpt_path
    trainer.fit(model, train_loader, val_loader, **fit_kwargs)
