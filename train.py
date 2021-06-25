from model import WaveGlowMelHF
from model import WaveGlowLoss
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from dataset_utils import SRDataset
import torch
import utils
from scipy.io import wavfile
from utils import data_loader
from hparams import set_hparams, hparams
import numpy as np
import librosa
import random
import torch.nn as nn
from training_utils import BaseTrainer, LatestModelCheckpoint
import subprocess
from datetime import datetime
from pytorch_lightning.loggers import TensorBoardLogger

import os


def save_wav(wav, path, sr, norm=False):
    if norm:
        wav = wav / np.abs(wav).max()
    wav *= 32767
    # proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))


class WaveGlowTask4(nn.Module):
    def __init__(self):
        super(WaveGlowTask4, self).__init__()
        self.current_epoch = 0
        self.global_step = 0
        self.loaded_optimizer_states_dict = {}
        self.trainer = None
        self.logger = None
        self.on_gpu = False
        self.use_dp = False
        self.use_ddp = False
        self.example_input_array = None

        self.max_tokens = hparams['max_tokens']
        self.max_sentences = hparams['max_sentences']
        self.max_eval_tokens = hparams['max_eval_tokens']
        if self.max_eval_tokens == -1:
            hparams['max_eval_tokens'] = self.max_eval_tokens = self.max_tokens
        self.max_eval_sentences = hparams['max_eval_sentences']
        if self.max_eval_sentences == -1:
            hparams['max_eval_sentences'] = self.max_eval_sentences = self.max_sentences

        self.model = None
        self.training_losses_meter = None
        self.dataset_cls = SRDataset
        self.criterion = WaveGlowLoss()

    def on_sanity_check_start(self):
        pass

    def on_train_start(self):
        pass

    @data_loader
    def train_dataloader(self):
        train_dataset = self.dataset_cls('train', shuffle=True)
        return self.build_dataloader(train_dataset, True, self.max_sentences)

    @data_loader
    def val_dataloader(self):
        valid_dataset = self.dataset_cls('valid', shuffle=False)
        return self.build_dataloader(valid_dataset, False, self.max_eval_sentences)

    @data_loader
    def test_dataloader(self):
        test_dataset = self.dataset_cls('test', shuffle=False)
        return self.build_dataloader(test_dataset, False, self.max_eval_sentences)

    def build_dataloader(self, dataset, shuffle, max_sentences):
        world_size = 1
        rank = 0
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        sampler_cls = DistributedSampler
        train_sampler = sampler_cls(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle
        )
        return torch.utils.data.DataLoader(
            dataset=dataset,
            shuffle=False,
            collate_fn=dataset.collater,
            batch_size=max_sentences,
            num_workers=dataset.num_workers,
            sampler=train_sampler,
            pin_memory=True
        )
    
    def configure_optimizers(self):
            #print("ENTERING CONFIGURE OPTIMIZER", flush=True)
        optm = self.build_optimizer(self.model)
        self.scheduler = self.build_scheduler(optm)
        #print("LEAVING CONFIGURE OPTIMIZER", flush=True)
        return [optm]
    
    def load_ckpt(self, ckpt_base_dir, current_model_name=None, model_name='model', force=True, strict=True):
        if current_model_name is None:
            current_model_name = model_name
        utils.load_ckpt(self.__getattr__(current_model_name),
                        ckpt_base_dir, current_model_name, force, strict)

    def build_model(self):
        model = WaveGlowMelHF(**hparams['waveglow_config']).cuda()
        utils.print_arch(model)
        return model

    def run_model(self, sample):
        lr = sample['resampled_wavs']
        hr = sample['wavs']
        #print(f'lr.shape = {lr.shape}, hr.shape = {hr.shape}')
        lr = lr.cuda()
        hr = hr.cuda()
        output = self.model(lr, hr)
        loss = self.criterion(output)
        loss_log = {'loss': loss.item()}
        return loss, loss_log

    def on_epoch_start(self):
        self.training_losses_meter = {'total_loss': utils.AvgrageMeter()}

    def _training_step(self, sample, batch_idx, _):
        return self.run_model(sample)

    def training_step(self, sample, batch_idx, optimizer_idx=-1):
        loss_ret = self._training_step(sample, batch_idx, optimizer_idx)
        self.opt_idx = optimizer_idx
        if loss_ret is None:
            return {'loss': None}
        total_loss, log_outputs = loss_ret
        log_outputs = utils.tensors_to_scalars(log_outputs)
        for k, v in log_outputs.items():
            if k not in self.training_losses_meter:
                self.training_losses_meter[k] = utils.AvgrageMeter()
            if not np.isnan(v):
                self.training_losses_meter[k].update(v)
        self.training_losses_meter['total_loss'].update(total_loss.item())

        try:
            log_outputs['lr'] = self.scheduler.get_lr()
            if isinstance(log_outputs['lr'], list):
                log_outputs['lr'] = log_outputs['lr'][0]
        except:
            pass

        # log_outputs['all_loss'] = total_loss.item()
        progress_bar_log = log_outputs
        tb_log = {f'tr/{k}': v for k, v in log_outputs.items()}
        return {
            'loss': total_loss,
            'progress_bar': progress_bar_log,
            'log': tb_log
        }

    def backward(self, loss, optimizer):
        loss.backward()

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.step()
        optimizer.zero_grad()
        if self.scheduler is not None:
            self.scheduler.step(self.global_step //
                                hparams['accumulate_grad_batches'])


    def validation_step(self, sample, batch_idx):
        loss, loss_log = self.run_model(sample)
        return {'loss': loss.item(), 'nsamples': sample['nsamples']}

    def _validation_end(self, outputs):
        total_loss_meter = utils.AvgrageMeter()
        for output in outputs:
            total_loss_meter.update(output['loss'], output['nsamples'])
        return {'total_loss': round(total_loss_meter.avg, 4)}

    def build_optimizer(self, model):
        self.optimizer = optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(hparams['lr'])
        )
        return optimizer

    def build_scheduler(self, optimizer):
        self.scheduler = None  # According to NVIDIA waveglow repo

#    def clip_norm(self, wav):
#        wav =

    def test_step(self, sample, batch_idx):
        lr = sample['resampled_wavs']
        if hparams['is_freqwg']:
            lr_mag = sample['lr_mags']
            lr_pha = sample['lr_phas']
            hr_ = self.model.infer(lr_mag, lr_pha)
        else:
            hr_ = self.model.infer(lr)
        hr = sample['wavs']
        gen_dir = os.path.join(hparams['work_dir'], f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
        os.makedirs(gen_dir, exist_ok=True)
        for idx, (wav_pred, wav_lr, wav_hr, item_name) in enumerate(zip(hr_, lr, hr, sample["item_name"])):
            wav_hr = wav_hr  # / wav_hr.abs().max()
            wav_lr = wav_lr  # / wav_lr.abs().max()
            wav_pred = wav_pred.view(-1).cpu().float().numpy()  # / wav_pred.abs().max()
            sr = hparams['sampling_rate']

            save_wav(wav_lr.view(-1).cpu().float().numpy(), f'{gen_dir}/{item_name}_lr.wav', sr // 4)
            save_wav(wav_hr.view(-1).cpu().float().numpy(), f'{gen_dir}/{item_name}_hr.wav', sr)
            save_wav(wav_pred, f'{gen_dir}/{item_name}_pred.wav', sr)
        return {}

    def test_end(self, self_outputs):
        return {}
    
    def validation_end(self, outputs):
        loss_output = self._validation_end(outputs)
        print(f"\n==============\n "
              f"valid results: {loss_output}"
              f"\n==============\n")
        return {
            'log': {f'val/{k}': v for k, v in loss_output.items()},
            'val_loss': loss_output['total_loss']
        }
    
    def on_batch_end(self):
        pass

    def on_epoch_end(self):
        loss_outputs = {k: round(v.avg, 4)
                        for k, v in self.training_losses_meter.items()}
        print(f"\n==============\n "
              f"Epoch {self.current_epoch} ended. Steps: {self.global_step}. {loss_outputs}"
              f"\n==============\n")

    def on_train_end(self):
        pass

    def test_start(self):
        pass

    def test_end(self, outputs):
        return self.validation_end(outputs)

    ###########
    # Running configuration
    ###########
    @classmethod
    def start(cls):
        set_hparams()
        os.environ['MASTER_PORT'] = str(random.randint(15000, 30000))
        random.seed(hparams['seed'])
        np.random.seed(hparams['seed'])
        task = cls()
        work_dir = hparams['work_dir']
        trainer = BaseTrainer(
            checkpoint_callback=LatestModelCheckpoint(
                filepath=work_dir,
                verbose=True,
                monitor='val_loss',
                mode='min',
                num_ckpt_keep=hparams['num_ckpt_keep'],
                save_best=hparams['save_best'],
                period=1 if hparams['save_ckpt'] else 100000
            ),
            logger=TensorBoardLogger(
                save_dir=work_dir,
                name='lightning_logs',
                version='lastest'
            ),
            gradient_clip_val=hparams['clip_grad_norm'],
            val_check_interval=hparams['val_check_interval'],
            row_log_interval=hparams['log_interval'],
            max_updates=hparams['max_updates'],
            num_sanity_val_steps=hparams['num_sanity_val_steps'] if not hparams['validate'] else 10000,
            accumulate_grad_batches=hparams['accumulate_grad_batches'],
            print_nan_grads=hparams['print_nan_grads'])
        if not hparams['infer']:  # train
            t = datetime.now().strftime('%Y%m%d%H%M%S')
            code_dir = f'{work_dir}/codes/{t}'
            subprocess.check_call(f'mkdir -p "{code_dir}"', shell=True)
            for c in hparams['save_codes']:
                if os.path.exists(c):
                    subprocess.check_call(
                        f'cp -r "{c}" "{code_dir}/"', shell=True)
            print(f"| Copied codes to {code_dir}.")
            trainer.checkpoint_callback.task = task
            trainer.fit(task)
        else:
            trainer.test(task)

    def init_ddp_connection(self, proc_rank, world_size):
        set_hparams(print_hparams=False)
        # guarantees unique ports across jobs from same grid search
        default_port = 12910
        # if user gave a port number, use that one instead
        try:
            default_port = os.environ['MASTER_PORT']
        except Exception:
            os.environ['MASTER_PORT'] = str(default_port)

        # figure out the root node addr
        root_node = '127.0.0.2'
        root_node = self.trainer.resolve_root_node_address(root_node)
        os.environ['MASTER_ADDR'] = root_node
        dist.init_process_group('nccl', rank=proc_rank, world_size=world_size)

    ###########
    # utils
    ###########
    def grad_norm(self, norm_type):
        results = {}
        total_norm = 0
        for name, p in self.named_parameters():
            if p.requires_grad:
                try:
                    param_norm = p.grad.data.norm(norm_type)
                    total_norm += param_norm ** norm_type
                    norm = param_norm ** (1 / norm_type)

                    grad = round(norm.data.cpu().numpy().flatten()[0], 3)
                    results['grad_{}_norm_{}'.format(norm_type, name)] = grad
                except Exception:
                    # this param had no grad
                    pass

        total_norm = total_norm ** (1. / norm_type)
        grad = round(total_norm.data.cpu().numpy().flatten()[0], 3)
        results['grad_{}_norm_total'.format(norm_type)] = grad
        return results


if __name__ == '__main__':
    set_hparams()
    WaveGlowTask4.start()