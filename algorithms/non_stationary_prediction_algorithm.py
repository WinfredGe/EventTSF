import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from einops import rearrange
from pathlib import Path
from myutils.log_plots import log_timeseries_plots
from models.flow_matching_transition import FlowMatchingModel
from torchmetrics import MetricCollection
from metrics import myCRPS, ProbMAE, ProbMSE, ProbRMSE, WeightedQuantileLoss

class NonStationPrediction(pl.LightningModule):
    def __init__(self, root_cfg):
        super().__init__()
        self.root_cfg = root_cfg
        self.x_shape = 1
        self.z_shape = self.root_cfg.model.z_shape
        self.frame_stack = self.root_cfg.data.slice_size
        self.context_frames = self.root_cfg.data.context_frames
        self.external_cond_dim = 128
        self.repeat_valid_num_samples = 3
        self.validation_outputs = []
        self.model = FlowMatchingModel(
            x_shape=self.x_shape,
            z_shape=self.z_shape,
            external_cond_dim=self.external_cond_dim,
            root_cfg=self.root_cfg
        )
        metadata_path_value = self.root_cfg.algorithm.get('metadata_path') if hasattr(self.root_cfg.algorithm, 'get') else self.root_cfg.algorithm['metadata_path']
        metadata_path = Path(metadata_path_value)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        data_mean = torch.tensor(metadata['train_data_mean'])
        data_std = torch.tensor(metadata['train_data_std'])
        self.register_buffer('data_mean', data_mean)
        self.register_buffer('data_std', data_std)
        generator = torch.Generator()
        generator.manual_seed(42)
        init_value = torch.randn([self.z_shape], generator=generator) * self.data_std.data + self.data_mean.data
        self.init_z = nn.Parameter(init_value, requires_grad=True)
        self.save_hyperparameters(root_cfg)
        self._init_metrics()

    def _init_metrics(self):
        self.metrics = MetricCollection(
            metrics={
                "crps": myCRPS(),
                "mse": ProbMSE(),
                "mae": ProbMAE(),
                "rmse": ProbRMSE(),
                "wql": WeightedQuantileLoss(),
            }
        )
        self.metrics.to("cpu")
        print("Metrics initialized.")

    def forward(self, z, x, conditions=None, deterministic_t=None):
        return self.model(z, x, conditions, deterministic_t)

    def _get_init_z_for_batch(self, batch_size):
        return self.init_z.unsqueeze(0).expand(batch_size, -1)

    def _preprocess_batch(self, batch):
        xs, conditions = batch[0], batch[1]
        batch_size, all_steps = xs.shape[:2]
        xs = self._normalize_x(xs)
        xs = rearrange(xs, "b (t fs) c ... -> t b (fs c) ...", fs=self.frame_stack)
        conditions = rearrange(conditions, "b (t fs) d -> t b fs d", fs=self.frame_stack)
        conditions = conditions[:, :, 0, :]
        return xs, conditions

    def training_step(self, batch, batch_idx):
        xs, conditions = self._preprocess_batch(batch)
        n_frames, batch_size = xs.shape[:2]
        predictions = []
        losses = []
        z = self._get_init_z_for_batch(batch_size)
        base_ratio = getattr(self.root_cfg.algorithm, 'teacher_forcing_ratio', 0.5)
        current_epoch = self.current_epoch
        max_epochs = self.trainer.max_epochs
        teacher_forcing_ratio = base_ratio * (1 - current_epoch / max_epochs)
        for t in range(n_frames):
            if t == 0 or (torch.rand(1).item() < teacher_forcing_ratio):
                input_x = xs[t]
            else:
                input_x = predictions[-1].detach()
            z, pred, loss = self.forward(z, input_x, conditions=conditions[t], deterministic_t=None)
            predictions.append(pred)
            losses.append(loss)
        predictions = torch.stack(predictions)
        transition_loss = torch.stack(losses).mean()
        mse_loss = F.mse_loss(predictions, xs, reduction="mean")
        self.log('train_loss', mse_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_average_transition_loss', transition_loss, on_step=False, on_epoch=True)
        return mse_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        xs, conditions = self._preprocess_batch(batch)
        n_frames, batch_size = xs.shape[:2]
        all_predictions = []
        for sample_idx in range(self.repeat_valid_num_samples):
            predictions = []
            z = self._get_init_z_for_batch(batch_size).clone()
            context_steps = self.context_frames // self.frame_stack
            for t in range(context_steps):
                z, pred, *_ = self.forward(z, xs[t], conditions=conditions[t], deterministic_t=1)
                predictions.append(pred)
            if context_steps < n_frames:
                for t in range(context_steps, n_frames):
                    if len(predictions) > 0:
                        prev_pred = predictions[-1]
                    else:
                        prev_pred = xs[context_steps - 1] if context_steps > 0 else torch.zeros_like(xs[t])
                    if self.training:
                        input_x = prev_pred
                    else:
                        noise_scale = 0.1
                        noise = torch.randn_like(prev_pred) * noise_scale
                        input_x = prev_pred + noise
                    z, pred, *_ = self.model.sample(z, input_x, conditions=conditions[t])
                    predictions.append(pred)
            all_predictions.append(torch.stack(predictions))
        mean_pred = torch.stack(all_predictions).mean(dim=0)
        loss = F.mse_loss(mean_pred, xs, reduction="mean")
        gt_original = xs
        gt_for_metrics = rearrange(gt_original, 't b (fs c) -> b (t fs) c', fs=self.frame_stack)
        preds_for_metrics = rearrange(all_predictions,
                                      'num_s t b (fs c) -> b (t fs) c num_s',
                                      fs=self.frame_stack, c=self.x_shape)
        self.validation_outputs.append({
            'preds_for_metrics': preds_for_metrics.cpu(),
            'gt_for_metrics': gt_for_metrics.cpu(),
            'mean_pred': mean_pred,
            'gt': xs,
            'all_samples': [p for p in all_predictions]
        })
        if context_steps < n_frames:
            future_loss = F.mse_loss(
                mean_pred[context_steps:], 
                xs[context_steps:], 
                reduction="mean"
            )
            self.log('val_future_loss', future_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.validation_step(batch, batch_idx)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self, namespace="validation"):
        if not self.validation_outputs:
            return
        all_preds = torch.cat([out['mean_pred'] for out in self.validation_outputs], dim=1)
        all_gt = torch.cat([out['gt'] for out in self.validation_outputs], dim=1)
        all_samples = []
        for out in self.validation_outputs:
            for sample_list in out['all_samples']:
                all_samples.extend(sample_list)
        mse = F.mse_loss(all_preds, all_gt, reduction="mean")
        future_mse = F.mse_loss(
            all_preds[self.context_frames // self.frame_stack:],
            all_gt[self.context_frames // self.frame_stack:],
            reduction="mean"
        )
        self.log('val_mse', mse, on_epoch=True, prog_bar=True)
        self.log('val_future_mse', future_mse, on_epoch=True, prog_bar=True)
        self.metrics.reset()
        all_preds_for_metrics = torch.cat([out['preds_for_metrics'] for out in self.validation_outputs], dim=0)
        all_gt_for_metrics = torch.cat([out['gt_for_metrics'] for out in self.validation_outputs], dim=0)
        self.metrics.update(all_preds_for_metrics, all_gt_for_metrics)
        calculated_metrics = self.metrics.compute()
        for metric_name, metric_value in calculated_metrics.items():
            self.log(f'val_{metric_name}', metric_value, on_epoch=True, prog_bar=True)
        pred_unnorm = rearrange(all_preds, "t b (c x_shape) -> b (t c) x_shape", x_shape=self.x_shape)
        gt_unnorm = rearrange(all_gt, "t b (c x_shape) -> b (t c) x_shape", x_shape=self.x_shape)
        num_per_sample = all_gt.shape[0]
        reshaped_samples = []
        for sample_idx in range(len(all_samples) //(num_per_sample * self.repeat_valid_num_samples)):
            start_idx = sample_idx * (num_per_sample * self.repeat_valid_num_samples)
            end_idx = start_idx + (num_per_sample * self.repeat_valid_num_samples)
            concatenated = torch.cat(all_samples[start_idx:end_idx], dim=1)
            reshaped_samples.append(concatenated.unsqueeze(-1))
        unnorm_samples = [sample for sample in reshaped_samples]
        sample_unnorm = []
        total_len = unnorm_samples[0].shape[1]
        len_per_sample = total_len // self.repeat_valid_num_samples
        for sample_idx in range(self.repeat_valid_num_samples):
            start_idx = sample_idx * len_per_sample
            end_idx = start_idx + len_per_sample
            current_sample_parts = [sample[:, start_idx:end_idx] for sample in unnorm_samples]
            current_sample = torch.cat(current_sample_parts, dim=0)
            sample_unnorm.append(current_sample)
        log_timeseries_plots(
            pred_unnorm, gt_unnorm,
            self.context_frames, namespace,
            getattr(self.root_cfg.data, 'frequency', 1),sample_predictions=sample_unnorm
        )
        self.validation_outputs.clear()

    def on_test_epoch_end(self):
        self.on_validation_epoch_end(namespace='test')

    def configure_optimizers(self):
        params = list(self.model.parameters()) + [self.init_z]
        optimizer = torch.optim.AdamW(
            params,
            lr=self.root_cfg.algorithm.lr,
            weight_decay=getattr(self.root_cfg.algorithm, 'weight_decay', 1e-4),
            betas=getattr(self.root_cfg.algorithm, 'optimizer_beta', (0.9, 0.999))
        )
        if hasattr(self.root_cfg.algorithm, 'max_lr'):
            total_steps = self.trainer.estimated_stepping_batches
            if total_steps is None or total_steps <= 0:
                print("Warning: Could not estimate total steps, using optimizer without scheduler")
                return optimizer
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=self.root_cfg.algorithm.max_lr,
                total_steps=total_steps,
                pct_start=getattr(self.root_cfg.algorithm, 'pct_start', 0.3),
                div_factor=getattr(self.root_cfg.algorithm, 'div_factor', 25),
                final_div_factor=getattr(self.root_cfg.algorithm, 'final_div_factor', 1e4),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
            }
        return optimizer

    def _normalize_x(self, x):
        return (x - self.data_mean) / self.data_std

    def _unnormalize_x(self, x):
        return x * self.data_std + self.data_mean