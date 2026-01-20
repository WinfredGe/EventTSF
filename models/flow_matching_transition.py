from typing import Optional, Tuple
from random import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utimedit import UDiT
from .timedit import DiT
import numpy as np
import pandas as pd
class FlowMatchingModel(nn.Module):
    def __init__(self, x_shape, z_shape, external_cond_dim, root_cfg):
        super().__init__()
        self.root_cfg = root_cfg
        self.x_shape = x_shape
        self.z_shape = z_shape
        self.timesteps = 100
        self.sampling_timesteps = 100
        self.network_size = 256
        self.cfg_scale = 0.0
        self.num_mlp_layers = 2
        self.external_cond_dim = external_cond_dim
        self.time_schedule = getattr(root_cfg.model, "time_schedule", "fixed")
        self._build_model()
    def _build_model(self):
        model_type = getattr(self.root_cfg.model, 'model_type', 'DiT')
        common_params = {
            "x_shape": self.x_shape,
            "z_shape": self.z_shape,
            "frame_stack": self.root_cfg.data.slice_size,
            "patch_size": getattr(self.root_cfg.model, 'patch_size', 6),
            "network_size": self.network_size,
            "external_condition_dim": self.external_cond_dim,
            "cfg_scale": self.cfg_scale,

            "mlp_ratio": getattr(self.root_cfg.model, 'mlp_ratio', 4.0),
            "num_heads": getattr(self.root_cfg.model, 'num_heads', 8),
        }

        if model_type == 'DiT':
            print("Building DiT model.")
            dit_params = {
                "num_transformer_layers": getattr(self.root_cfg.model, 'num_transformer_layers', 2),
            }
            params = {**common_params, **dit_params}
            self.model = DiT(**params)
        elif model_type == 'UDiT':
            print("Building UDiT model.")
            udit_params = {
                "encoder_depths": getattr(self.root_cfg.model, 'encoder_depths', [1, 1, 1]),
                "bottleneck_depth": getattr(self.root_cfg.model, 'bottleneck_depth', 1),
                "decoder_depths": getattr(self.root_cfg.model, 'decoder_depths', [1, 1, 1]),
                "downsample_factors": getattr(self.root_cfg.model, 'downsample_factors', [2, 2]),
            }
            params = {**common_params, **udit_params}
            self.model = UDiT(**params)

        self.control_linear_alpha = nn.Linear(self.external_cond_dim, 1)
        self.control_linear_beta = nn.Linear(self.external_cond_dim, 1)


        init_bias = torch.log(torch.exp(torch.tensor(1.0)) - 1.0)

        nn.init.constant_(self.control_linear_alpha.bias, init_bias)
        nn.init.constant_(self.control_linear_beta.bias, init_bias)
        nn.init.constant_(self.control_linear_alpha.weight, 0.0)
        nn.init.constant_(self.control_linear_beta.weight, 0.0)

    

    def forward(
            self,
            z: torch.Tensor,
            x_next: torch.Tensor,
            conditions: torch.Tensor,
            deterministic_t: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x_next.shape[0]
        device = x_next.device
        if hasattr(self, 'root_cfg') and hasattr(self.root_cfg, 'experiment'):
            current_epoch = getattr(self.root_cfg.experiment, 'current_epoch', 0)
            max_epochs = getattr(self.root_cfg.experiment, 'epochs', 100)
        else:
            current_epoch = 0
            max_epochs = 100

        base_drop_prob = getattr(self.root_cfg.model, 'cond_drop_prob', 0.3) if hasattr(self.root_cfg, 'model') else 0.3
        min_drop_prob = getattr(self.root_cfg.model, 'min_drop_prob', 0.1) if hasattr(self.root_cfg, 'model') else 0.1
        progress = current_epoch / max_epochs
        current_drop_prob = base_drop_prob - progress * (base_drop_prob - min_drop_prob)

        if deterministic_t is None:
            t = torch.rand(batch_size, device=device)
            # t_modulation = torch.sigmoid(self.control_linear(conditions)) * 0.2 if self.time_schedule == 'adaptive' else torch.zeros_like(self.control_linear(conditions))
            alpha = F.softplus(self.control_linear_alpha(conditions))
            beta = F.softplus(self.control_linear_beta(conditions))
            t_modulation = t.pow(beta.squeeze(-1) / (alpha.squeeze(-1) + 1e-6)) if self.time_schedule == 'adaptive' else t
            # t = torch.clamp(t + t_modulation.squeeze(-1), 0.0, 1.0)
            t = t_modulation

        else:
            t = torch.full((batch_size,), deterministic_t, device=device)

        x0 = torch.randn_like(x_next)

        t_expanded = t.view(-1, *([1] * (len(x_next.shape) - 1)))
        x_t = (1 - t_expanded) * x0 + t_expanded * x_next

        target_velocity = x_next - x0

        if conditions is not None:
            orig_conditions = conditions.clone()
            cond_mask = torch.rand(batch_size, device=device) >= current_drop_prob
            masked_conditions = conditions.clone()
            masked_conditions[~cond_mask] = 0.0
            x_self_cond = masked_conditions
        else:
            orig_conditions = None
            cond_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            x_self_cond = torch.zeros(batch_size, self.external_cond_dim, device=device)
        t_discrete = (t * (self.timesteps - 1)).long()
        pred_velocity, pred_z = self.model(x_t, t_discrete, z, x_self_cond)
        loss = F.mse_loss(pred_velocity, target_velocity, reduction='none')
        if orig_conditions is not None and torch.any(cond_mask) and torch.any(~cond_mask):
            cond_loss = loss[cond_mask].mean()
            uncond_loss = loss[~cond_mask].mean()
            cfg_contrast_weight = getattr(self.root_cfg.model, 'cfg_contrast_weight', 0.1) if hasattr(self.root_cfg, 'model') else 0.1
            cfg_contrast_loss = -F.mse_loss(cond_loss, uncond_loss) * cfg_contrast_weight
            total_loss = loss.mean() + cfg_contrast_loss
        else:
            total_loss = loss.mean()
        x_next_pred = x_t + pred_velocity * (1 - t_expanded)
        z_next_pred = pred_z
        if hasattr(self, 'log'):
            with torch.no_grad():
                self.log('drop_prob', current_drop_prob, on_step=False, on_epoch=True)
                self.log('cond_ratio', cond_mask.float().mean(), on_step=False, on_epoch=True)
                
                if 'cfg_contrast_loss' in locals():
                    self.log('cfg_contrast_loss', cfg_contrast_loss, on_step=False, on_epoch=True)

        return z_next_pred, x_next_pred, total_loss
    
    
    
    @torch.no_grad()
    def sample(
            self,
            z: torch.Tensor,
            x: torch.Tensor,
            conditions: torch.Tensor,
            num_steps: Optional[int] = None,
            guidance_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_steps is None:
            num_steps = self.sampling_timesteps
        if guidance_scale is None:
            guidance_scale = getattr(self.root_cfg.model, 'guidance_scale', self.cfg_scale)
        batch_size = z.shape[0]
        device = z.device
        x_current = torch.randn_like(x)
        z_current = z.clone()
        dt = 1.0 / num_steps
        t_history = []
        null_conditions = torch.zeros_like(conditions)
        for i in range(num_steps):
            t_base = 1.0 - (i + 1) / num_steps
            t_tensor = torch.full((batch_size,), t_base, device=device)


            # t_modulation = torch.sigmoid(self.control_linear(conditions)) * 0.2 if self.time_schedule == 'adaptive' else torch.zeros_like(self.control_linear(conditions))

            alpha = F.softplus(self.control_linear_alpha(conditions))
            beta = F.softplus(self.control_linear_beta(conditions))
            t_modulation = t_tensor.pow(beta.squeeze(-1) / (alpha.squeeze(-1) + 1e-6)) if self.time_schedule == 'adaptive' else t_tensor
            # t_effective = torch.clamp(t_tensor + t_modulation.squeeze(-1), 0.0, 1.0)
            t_effective = t_modulation



            t_discrete = (t_effective * (self.timesteps - 1)).long()
            t_history.append(t_discrete.cpu())
            if guidance_scale > 1.0 and conditions is not None:
                velocity_cond, z_cond = self.model(x_current, t_discrete, z_current, conditions)
                velocity_uncond, z_uncond = self.model(x_current, t_discrete, z_current, null_conditions)
                velocity = velocity_uncond + guidance_scale * (velocity_cond - velocity_uncond)
                pred_z = z_uncond + guidance_scale * (z_cond - z_uncond)
            else:
                velocity, pred_z = self.model(x_current, t_discrete, z_current, conditions)

            x_current = x_current + dt * velocity
            z_current = pred_z


        return z_current, x_current
