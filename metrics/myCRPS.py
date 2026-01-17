import torch
from torchmetrics import Metric
from .CRPS import CRPS as CRPSCalculator


class myCRPS(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("total_crps", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, true: torch.Tensor):
        def compute_crps(i):
            return pscore(pred_np[i], true_np[i]).compute()[0]
        pred = pred.view(-1, pred.shape[3])
        true = true.view(-1)
        pred_np = pred.cpu().numpy()
        true_np = true.cpu().numpy()
        crps_sum = 0.0
        for i in range(len(true_np)):
            calculator = CRPSCalculator(ensemble_members=pred_np[i], observation=float(true_np[i]))
            res = calculator.compute()
            crps_sum += res[0]
        self.total_crps += torch.tensor(crps_sum).to(self.device)
        self.total_samples += pred.size(0)

    def compute(self):
        return self.total_crps / self.total_samples

