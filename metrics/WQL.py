import torch
from torchmetrics import Metric

class WeightedQuantileLoss(Metric):
    def __init__(self, quantiles: list = None, weights: list = None, dist_sync_on_step=False):

        super().__init__(dist_sync_on_step=dist_sync_on_step)

        if quantiles is None:
            self.quantiles = [i / 10.0 for i in range(1, 10)]
        else:
            if not all(0 < q < 1 for q in quantiles):
                raise ValueError("All quantiles must be between 0 and 1.")
            self.quantiles = quantiles

        if weights is None:
            self.weights = torch.ones(len(self.quantiles)) / len(self.quantiles)
        else:
            if len(weights) != len(self.quantiles):
                raise ValueError("Number of weights must match the number of quantiles.")
            self.weights = torch.tensor(weights).float()
            self.weights = self.weights / self.weights.sum()

        self.quantiles_tensor = torch.tensor(self.quantiles).float()

        self.add_state("total_weighted_quantile_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, all_gen_y: torch.Tensor, y_true: torch.Tensor):

        all_gen_y = all_gen_y.view(-1, all_gen_y.shape[-1])
        y_true = y_true.view(-1)
        quantiles_tensor_on_device = self.quantiles_tensor.to(all_gen_y.device)

        predicted_quantiles = torch.quantile(all_gen_y, quantiles_tensor_on_device, dim=1)

        current_batch_weighted_loss = torch.tensor(0.0).to(self.device)


        for i, q in enumerate(self.quantiles):
            q_pred = predicted_quantiles[i]
            errors = y_true - q_pred
            loss = torch.where(errors >= 0, q * errors, (q - 1) * errors)
            current_batch_weighted_loss += self.weights[i].to(self.device) * loss.sum()

        self.total_weighted_quantile_loss += current_batch_weighted_loss
        self.total_samples += y_true.numel()

    def compute(self):

        if self.total_samples == 0:
            return torch.tensor(float('nan')) # Avoid division by zero
        return self.total_weighted_quantile_loss / self.total_samples

