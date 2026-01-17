# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, network_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, network_size, bias=True),
            nn.SiLU(),
            nn.Linear(network_size, network_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Applies dropout to class labels for classifier-free guidance.
    Dropped labels are replaced with -1.
    """

    def __init__(self, dropout_prob):
        super().__init__()
        self.dropout_prob = dropout_prob

    def token_drop(self, labels):
        # Create a drop mask per batch element and unsqueeze to match (N, 1)
        drop_ids = (torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob).unsqueeze(1)
        # Replace dropped labels with -1 broadcasting over the second dimension
        return torch.where(drop_ids, torch.full_like(labels, 0.01), labels)

    def forward(self, labels, train):
        if train and self.dropout_prob > 0:
            labels = self.token_drop(labels)
        return labels

class MemoryEmbedder(nn.Module):
    """
    Embeds memory state z from (N, 48) to network dimension and processes it.
    """

    def __init__(self, z_shape, network_size):
        super().__init__()
        self.z_input_proj = nn.Linear(z_shape, network_size)
        self.z_state_processor = nn.Sequential(
            nn.Linear(network_size, network_size),
            nn.ReLU(),
            nn.Linear(network_size, network_size)
        )
        self.z_shape_memory_head = nn.Sequential(
            nn.Linear(network_size, network_size // 2),
            nn.ReLU(),
            nn.Linear(network_size // 2, z_shape)
        )

    def forward(self, z):
        """
        z: (N, z_shape) -> (N, network_size)
        """
        z_embedded = self.z_input_proj(z)  # (N, network_size)
        z_processed = self.z_state_processor(z_embedded)  # (N, network_size)
        return z_processed

    def get_memory_output(self, z_feature):
        """
        z_feature: (N, network_size) -> (N, z_shape)
        """
        return self.z_shape_memory_head(z_feature)


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, network_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(network_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(network_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(network_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(network_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=network_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(network_size, 6 * network_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, network_size, patch_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(network_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(network_size, patch_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(network_size, 2 * network_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class PatchEmbed1D(nn.Module):
    """
    1D version of PatchEmbed for processing 1D signals.
    Splits a 1D input into patches and then projects them into an embedding space.
    """

    def __init__(self, signal_length, patch_size, in_chans=1, embed_dim=768, bias=True):
        super().__init__()
        self.signal_length = signal_length
        self.patch_size = (patch_size,)
        self.num_patches = signal_length // patch_size
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        B, C, L = x.shape  # batch, channels, length
        assert L == self.signal_length, f"Input signal length ({L}) doesn't match model ({self.signal_length})"

        # Project patches
        x = self.proj(x)  # B, embed_dim, num_patches
        x = x.transpose(1, 2)  # B, num_patches, embed_dim

        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
            self,
            x_shape=1,
            z_shape=48,
            frame_stack=24,
            patch_size=6,
            num_transformer_layers=2,
            network_size=256,
            external_condition_dim=128,
            cfg_scale=7.0,
            mlp_ratio=4.0,   # mlp_dim: network_size * mlp_ratio
            dropout_prob=0.1,
            num_heads=8,
    ):
        super().__init__()
        self.x_shape = x_shape
        self.z_shape = z_shape
        self.frame_stack = frame_stack
        self.patch_size = patch_size
        self.network_size = network_size
        self.cfg_scale = cfg_scale
        self.num_heads=num_heads


        self.x_embedder = PatchEmbed1D(frame_stack, patch_size, x_shape, network_size, bias=True)
        self.t_embedder = TimestepEmbedder(network_size)
        self.y_embedder = LabelEmbedder(dropout_prob)

        self.y_embedder_affine = nn.Linear(external_condition_dim, network_size, bias=True)

        self.z_embedder = MemoryEmbedder(z_shape, network_size)

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, network_size), requires_grad=False)


        self.z_pos_embed = nn.Parameter(torch.zeros(1, 1, network_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(network_size, num_heads=self.num_heads, mlp_ratio=mlp_ratio) for _ in range(num_transformer_layers)
        ])
        self.final_layer = FinalLayer(network_size, patch_size)

        self.memory_fusion = nn.MultiheadAttention(network_size, num_heads=self.num_heads, batch_first=True)
        self.memory_norm = nn.LayerNorm(network_size)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed(self.network_size,
                                            np.arange(self.frame_stack // self.patch_size, dtype=np.float32))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        nn.init.normal_(self.z_pos_embed, std=0.02)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size)
        signals: (N, L) where L is the original signal length
        """
        p = self.patch_size
        t = x.shape[1]  # number of patches

        # x shape [batch, num_patches, patch_size]
        # Combine patches to form the original signal
        signals = x.reshape(shape=(x.shape[0], self.x_shape * t * p))
        return signals

    def forward(self, x, t, z, y):
        

        x = x.view(x.shape[0], -1, self.frame_stack)  # Reshape x to (N, C, L)
        x_embedded = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = L / patch_size

        z_embedded = self.z_embedder(z)  # (N, z_shape) -> (N, network_size)
        z_embedded = z_embedded.unsqueeze(1) + self.z_pos_embed  # (N, 1, network_size)

        z_attended, _ = self.memory_fusion(
            query=z_embedded,  # (N, 1, network_size)
            key=x_embedded,  # (N, T, network_size)
            value=x_embedded  # (N, T, network_size)
        )
        z_embedded = self.memory_norm(z_embedded + z_attended)

        x_with_memory = torch.cat([z_embedded, x_embedded], dim=1)  # (N, T+1, network_size)

        t = self.t_embedder(t)  # (N, D)
        y = self.y_embedder(y, self.training)
        y = self.y_embedder_affine(y)
        c = t + y  # (N, D)


        # for block in self.blocks:
        #     x_with_memory = block(x_with_memory, c)  # (N, T+1, D)

        z_updated = x_with_memory[:, 0:1, :]  # (N, 1, network_size)
        x_processed = x_with_memory[:, 1:, :]  # (N, T, network_size)

        x_output = self.final_layer(x_processed, c)  # (N, T, patch_size * x_shape)
        x_output = self.unpatchify(x_output)  # (N, L * C)


        z_output = self.z_embedder.get_memory_output(z_updated.squeeze(1))  # (N, z_shape)

        return x_output, z_output

    def forward_with_cfg(self, x, t, z, y):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        x_combined = torch.cat([half, half], dim=0)

        half_z = z[: len(z) // 2]  
        z_combined = torch.cat([half_z, half_z], dim=0)

        model_out, z_out = self.forward(x_combined, t, z_combined, y)  

        eps, rest = model_out, model_out
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + self.cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)

        cond_z, uncond_z = torch.split(z_out, len(z_out) // 2, dim=0)
        half_z = uncond_z + self.cfg_scale * (cond_z - uncond_z)
        z_output = torch.cat([half_z, half_z], dim=0)

        return torch.cat([eps, rest], dim=1), z_output


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_1d_sincos_pos_embed(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def test_dit_model():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Model parameters
    batch_size = 64
    frame_stack = 24
    x_shape = 1
    z_shape = 48  
    # Test classifier-free guidance
    cfg_scale = 3.0
    external_condition_dim = 128
    model = DiT(num_transformer_layers=2, network_size=256, external_condition_dim=external_condition_dim, patch_size=6,
                frame_stack=frame_stack,
                x_shape=1,
                cfg_scale=cfg_scale,
                z_shape=z_shape,dropout_prob=0.1,num_heads=8)  

    # Create dummy inputs
    x = torch.randn(batch_size, x_shape * frame_stack)
    z = torch.randn(batch_size, z_shape)  
    t = torch.randint(0, 1000, (batch_size,))
    y = torch.randint(0, 100, (batch_size, external_condition_dim)).float()

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    x = x.to(device)
    z = z.to(device)
    t = t.to(device)
    y = y.to(device)

    # Run forward pass
    with torch.no_grad():
        x_output, z_output = model(x, t, z, y)  

    # Print results
    print(f"Input shapes: x={x.shape}, z={z.shape}")
    print(f"Output shapes: x_output={x_output.shape}, z_output={z_output.shape}")

    # Need even batch size for cfg test
    x_cfg = torch.randn(batch_size * 2, x_shape * frame_stack).to(device)
    z_cfg = torch.randn(batch_size * 2, z_shape).to(device)  
    t_cfg = torch.randint(0, 1000, (batch_size * 2,)).to(device)
    y_cfg = torch.randint(0, 100, (batch_size * 2, external_condition_dim)).float().to(device)
    with torch.no_grad():
        x_output_cfg, z_output_cfg = model.forward_with_cfg(x_cfg, t_cfg, z_cfg, y_cfg)  

    print(f"\nCFG Input shapes: x_cfg={x_cfg.shape}, z_cfg={z_cfg.shape}")
    print(f"CFG Output shapes: x_output_cfg={x_output_cfg.shape}, z_output_cfg={z_output_cfg.shape}")
    print('Tests completed successfully!')


if __name__ == "__main__":

    result = test_dit_model()
