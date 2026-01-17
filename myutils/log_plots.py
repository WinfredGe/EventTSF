from typing import Optional
import wandb
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.functional import mean_squared_error

plt.set_loglevel("warning")
from typing import List


def log_timeseries_plots(all_preds: object, truth: object, context_frames: object, namespace: object,
                         frequency: Optional[str] = None, sample_predictions: Optional[List[object]] = None) -> object:

    def safe_wandb_log(log_dict):
        if wandb.run is not None:
            wandb.log(log_dict)
        else:
            print(f"WandB not initialized, skipping logging: {list(log_dict.keys())}")

    if namespace == "validation":
        batch_idx = 1
        feature_idx = 0

        preds_np = all_preds[batch_idx, :, feature_idx].detach().cpu().numpy()
        truth_np = truth[batch_idx, :, feature_idx].detach().cpu().numpy()
        seq_length = len(truth_np)

        fig, ax = plt.subplots(figsize=(20, 6))
        freq_map = {"D": "d", "B": "d", "30min": "h", "H": "h"}
        freq_label = freq_map.get(frequency, "")

        ax.plot(truth_np, label="Ground Truth", color='#008000', linewidth=2)

        ax.plot(range(context_frames), preds_np[:context_frames], color='#FF8C00',
                linewidth=2, label="Context (Input)")

        ax.plot(range(context_frames, seq_length), preds_np[context_frames:],
                label="Mean Prediction", color='#0000FF', linewidth=2)

        if sample_predictions and len(sample_predictions) > 0:
            future_samples = []
            for sample in sample_predictions:
                sample_np = sample[batch_idx, context_frames:, feature_idx].detach().cpu().numpy()
                future_samples.append(sample_np)

                ax.plot(range(context_frames, seq_length), sample_np,
                        color='#0000FF', alpha=0.15, linewidth=1)

            if future_samples:
                samples_array = np.array(future_samples)
                lower_bound = np.min(samples_array, axis=0)
                upper_bound = np.max(samples_array, axis=0)

                ax.fill_between(range(context_frames, seq_length), lower_bound, upper_bound,
                                alpha=0.2, color='blue', label="Prediction Range")

        ax.axvline(x=context_frames - 1, color='black', linestyle='--', alpha=0.7)

        tick_positions = np.linspace(0, seq_length - 1, num=5).astype(int)
        tick_positions[2] = context_frames
        if frequency == "30min":
            tick_labels = [f"{(i - context_frames) / 2:+} {freq_label}" for i in tick_positions]
        else:
            tick_labels = [f"{i - context_frames:+} {freq_label}" for i in tick_positions]


        tick_labels = [r'$t_0$' if x.startswith('+0') else x for x in tick_labels]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)


        ax.set_title(f"Time Series Prediction with Uncertainty", fontsize=16, weight="bold")
        ax.set_xlim([0, seq_length - 1])


        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", fontsize=12)
        plt.tight_layout()


        safe_wandb_log({f"{namespace}/timeseries_with_uncertainty": wandb.Image(plt)})
        plt.close(fig)

    else:
        max_plots = 20
        batch_size = all_preds.shape[0]
        feature_count = all_preds.shape[2]


        plots_to_create = min(batch_size, max_plots)

        for batch_idx in range(plots_to_create):
            for feature_idx in range(feature_count):

                preds_np = all_preds[batch_idx, :, feature_idx].detach().cpu().numpy()
                truth_np = truth[batch_idx, :, feature_idx].detach().cpu().numpy()
                seq_length = len(truth_np)

                fig, ax = plt.subplots(figsize=(20, 6))
                freq_map = {"D": "d", "B": "d", "30min": "h", "H": "h"}
                freq_label = freq_map.get(frequency, "")


                ax.plot(truth_np, label="Ground Truth", color='#008000', linewidth=2)


                ax.plot(range(context_frames), preds_np[:context_frames], color='#FF8C00',
                        linewidth=2, label="Context (Input)")


                ax.plot(range(context_frames, seq_length), preds_np[context_frames:],
                        label="Mean Prediction", color='#0000FF', linewidth=2)


                if sample_predictions and len(sample_predictions) > 0:

                    future_samples = []
                    for sample in sample_predictions:
                        sample_np = sample[batch_idx, context_frames:, feature_idx].detach().cpu().numpy()
                        future_samples.append(sample_np)


                        ax.plot(range(context_frames, seq_length), sample_np,
                                color='#0000FF', alpha=0.15, linewidth=1)


                    if future_samples:
                        samples_array = np.array(future_samples)
                        lower_bound = np.min(samples_array, axis=0)
                        upper_bound = np.max(samples_array, axis=0)

                        ax.fill_between(range(context_frames, seq_length), lower_bound, upper_bound,
                                        alpha=0.2, color='blue', label="Prediction Range")


                ax.axvline(x=context_frames - 1, color='black', linestyle='--', alpha=0.7)


                tick_positions = np.linspace(0, seq_length - 1, num=5).astype(int)
                tick_positions[2] = context_frames

                if frequency == "30min":
                    tick_labels = [f"{(i - context_frames) / 2:+} {freq_label}" for i in tick_positions]
                else:
                    tick_labels = [f"{i - context_frames:+} {freq_label}" for i in tick_positions]


                tick_labels = [r'$t_0$' if x.startswith('+0') else x for x in tick_labels]
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels)


                ax.set_title(f"Time Series Prediction - Batch {batch_idx}, Feature {feature_idx}", fontsize=16,
                             weight="bold")
                ax.set_xlim([0, seq_length - 1])


                for spine in ax.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(1.5)

                handles, labels = ax.get_legend_handles_labels()
                fig.legend(handles, labels, loc="upper right", fontsize=12)
                plt.tight_layout()


                safe_wandb_log({f"{namespace}/timeseries_b{batch_idx}_f{feature_idx}": wandb.Image(plt)})
                plt.close(fig)