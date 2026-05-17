import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from efficient_kan import KAN

# =========================================================
# 0. Utilities you may replace with your own versions
# =========================================================

class MyDataset(Dataset):
    def __init__(self, data):
        """
        data: torch.Tensor, shape [N, 1, patch_dim]
        """
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


def cg_snr(clean, noisy):
    clean = np.asarray(clean, dtype=np.float32)
    noisy = np.asarray(noisy, dtype=np.float32)
    noise = noisy - clean
    return 10.0 * np.log10(np.sum(clean ** 2) / (np.sum(noise ** 2) + 1e-12))


def logcosh_loss_softplus_approx(y_pred, y_true):
    import math
    diff = y_pred - y_true
    return torch.mean(F.softplus(2.0 * diff) - diff - math.log(2.0))


# =========================================================
# 1. Soft attention weighting module
# =========================================================

class SoftAttentionWeightingModule(nn.Module):
    def __init__(self, channels, reduction=4, negative_slope=0.5):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // reduction, 1),
            nn.LeakyReLU(negative_slope),
            nn.Conv1d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: [B, 1, C]
        """
        x_t = x.transpose(1, 2)           # [B, C, 1]
        w = self.attention(x_t)           # [B, C, 1]
        out = (x_t * w).transpose(1, 2)   # [B, 1, C]
        return out


# =========================================================
# 2. Feature learning block
# =========================================================

class FeatureLearningBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_residual=True,
        kan_base_activation=nn.SiLU,
        kan_grid_range=[-2, 2],
        attention_reduction=4,
        negative_slope=0.5,
    ):
        super().__init__()
        self.use_residual = use_residual

        self.fc1 = nn.Linear(in_dim, out_dim)
        self.act1 = KAN(
            [out_dim, out_dim],
            base_activation=kan_base_activation,
            grid_range=kan_grid_range
        )

        self.fc2 = nn.Linear(in_dim + out_dim, out_dim)
        self.act2 = KAN(
            [out_dim, out_dim],
            base_activation=kan_base_activation,
            grid_range=kan_grid_range
        )

        self.attn = SoftAttentionWeightingModule(
            channels=out_dim,
            reduction=attention_reduction,
            negative_slope=negative_slope
        )

    def forward(self, x):
        """
        x: [B, 1, in_dim]
        """
        x1 = self.fc1(x)
        x1 = self.act1(x1)

        x2_input = torch.cat([x, x1], dim=-1)
        x2 = self.fc2(x2_input)
        x2 = self.act2(x2 + x1)

        x2 = self.attn(x2)

        if self.use_residual and x.shape[-1] == x2.shape[-1]:
            out = x + x2
        else:
            out = x2 + x1

        return out


# =========================================================
# 3. LADNet
# =========================================================

class LADNet(nn.Module):
    def __init__(
        self,
        patch_dim=64,
        hidden_dim=512,
        use_residual=True,
        attention_reduction=4,
        negative_slope=0.5,
    ):
        super().__init__()

        self.block1 = FeatureLearningBlock(
            patch_dim, hidden_dim // 4,
            use_residual=use_residual,
            attention_reduction=attention_reduction,
            negative_slope=negative_slope
        )
        self.block2 = FeatureLearningBlock(
            hidden_dim // 4, hidden_dim // 2,
            use_residual=use_residual,
            attention_reduction=attention_reduction,
            negative_slope=negative_slope
        )
        self.block3 = FeatureLearningBlock(
            hidden_dim // 2, hidden_dim // 4,
            use_residual=use_residual,
            attention_reduction=attention_reduction,
            negative_slope=negative_slope
        )
        self.block4 = FeatureLearningBlock(
            hidden_dim // 4, hidden_dim // 2,
            use_residual=use_residual,
            attention_reduction=attention_reduction,
            negative_slope=negative_slope
        )
        self.block5 = FeatureLearningBlock(
            hidden_dim // 2, patch_dim,
            use_residual=use_residual,
            attention_reduction=attention_reduction,
            negative_slope=negative_slope
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x


def build_ladnet_model(
    patch_dim,
    hidden_dim=512,
    use_residual=True,
    attention_reduction=4,
    negative_slope=0.5,
):
    return LADNet(
        patch_dim=patch_dim,
        hidden_dim=hidden_dim,
        use_residual=use_residual,
        attention_reduction=attention_reduction,
        negative_slope=negative_slope,
    )


# =========================================================
# 4. GPU patch extraction
# =========================================================

def cg_patch_torch(A, l1, l2, o1, o2):
    """
    GPU patch extraction using torch unfold.

    Parameters
    ----------
    A : torch.Tensor
        2D tensor [n1, n2]
    l1, l2 : int
        patch size
    o1, o2 : int
        stride

    Returns
    -------
    X : torch.Tensor
        shape [l1*l2, num_patches]
    meta : dict
    """
    if A.ndim != 2:
        raise ValueError(f"A must be 2D, but got shape {A.shape}")

    n1, n2 = A.shape

    tmp1 = (n1 - l1) % o1
    pad1 = 0 if tmp1 == 0 else (o1 - tmp1)

    tmp2 = (n2 - l2) % o2
    pad2 = 0 if tmp2 == 0 else (o2 - tmp2)

    A4 = A.unsqueeze(0).unsqueeze(0)  # [1, 1, n1, n2]

    if pad1 > 0 or pad2 > 0:
        A4 = F.pad(A4, (0, pad2, 0, pad1), mode='reflect')

    _, _, N1, N2 = A4.shape

    patches = F.unfold(
        A4,
        kernel_size=(l1, l2),
        stride=(o1, o2)
    )  # [1, l1*l2, num_patches]

    X = patches.squeeze(0)  # [l1*l2, num_patches]

    meta = {
        'orig_shape': (n1, n2),
        'padded_shape': (N1, N2),
        'l1': l1,
        'l2': l2,
        'o1': o1,
        'o2': o2,
        'pad1': pad1,
        'pad2': pad2,
    }

    return X, meta


# =========================================================
# 5. Patch merge weighting windows
# =========================================================

def build_patch_weight(
    l1,
    l2,
    mode='mean',
    device='cpu',
    dtype=torch.float32,
    sigma=0.25,
    custom_weight=None,
):
    """
    Build 2D patch weight window.
    """
    mode = mode.lower()

    if mode == 'mean':
        w = torch.ones((l1, l2), device=device, dtype=dtype)

    elif mode == 'hann':
        w1 = torch.hann_window(l1, periodic=False, device=device, dtype=dtype)
        w2 = torch.hann_window(l2, periodic=False, device=device, dtype=dtype)
        w = torch.outer(w1, w2)

    elif mode == 'hamming':
        w1 = torch.hamming_window(l1, periodic=False, device=device, dtype=dtype)
        w2 = torch.hamming_window(l2, periodic=False, device=device, dtype=dtype)
        w = torch.outer(w1, w2)

    elif mode == 'gaussian':
        y = torch.linspace(-1, 1, l1, device=device, dtype=dtype)
        x = torch.linspace(-1, 1, l2, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        w = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    elif mode == 'custom':
        if custom_weight is None:
            raise ValueError("custom_weight must be provided when mode='custom'")
        if not isinstance(custom_weight, torch.Tensor):
            custom_weight = torch.tensor(custom_weight, device=device, dtype=dtype)
        w = custom_weight.to(device=device, dtype=dtype)
        if w.shape != (l1, l2):
            raise ValueError(f"custom_weight shape must be ({l1}, {l2}), got {w.shape}")

    else:
        raise ValueError(f"Unsupported merge mode: {mode}")

    return w


# =========================================================
# 6. GPU patch inverse reconstruction with selectable merge
# =========================================================

def cg_patch_inv_torch(
    X,
    n1,
    n2,
    l1,
    l2,
    o1,
    o2,
    merge_mode='mean',
    sigma=0.25,
    custom_weight=None,
):
    """
    GPU inverse patch reconstruction with selectable merge mode.

    Parameters
    ----------
    X : torch.Tensor
        shape [l1*l2, num_patches]
    n1, n2 : int
        target output shape
    merge_mode : str
        'mean', 'hann', 'hamming', 'gaussian', 'custom'
    sigma : float
        for gaussian
    custom_weight : torch.Tensor or np.ndarray or None
        used when merge_mode='custom'
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, but got shape {X.shape}")

    device = X.device
    dtype = X.dtype

    tmp1 = (n1 - l1) % o1
    pad1 = 0 if tmp1 == 0 else (o1 - tmp1)

    tmp2 = (n2 - l2) % o2
    pad2 = 0 if tmp2 == 0 else (o2 - tmp2)

    N1 = n1 + pad1
    N2 = n2 + pad2

    num_patches = X.shape[1]

    patch_weight_2d = build_patch_weight(
        l1=l1,
        l2=l2,
        mode=merge_mode,
        device=device,
        dtype=dtype,
        sigma=sigma,
        custom_weight=custom_weight,
    )

    patch_weight = patch_weight_2d.reshape(-1, 1)              # [l1*l2, 1]
    X_weighted = X * patch_weight                              # [l1*l2, num_patches]

    A_sum = F.fold(
        X_weighted.unsqueeze(0),                               # [1, l1*l2, num_patches]
        output_size=(N1, N2),
        kernel_size=(l1, l2),
        stride=(o1, o2)
    )  # [1, 1, N1, N2]

    W = patch_weight.repeat(1, num_patches).unsqueeze(0)      # [1, l1*l2, num_patches]
    mask = F.fold(
        W,
        output_size=(N1, N2),
        kernel_size=(l1, l2),
        stride=(o1, o2)
    )  # [1, 1, N1, N2]

    A = A_sum / mask.clamp_min(1e-12)
    A = A.squeeze(0).squeeze(0)

    return A[:n1, :n2]


# =========================================================
# 7. Single gather denoising function
# =========================================================
def denoise_single_gather(
    noisy,
    device,
    time_size=8,
    trace_size=8,
    time_shift=1,
    trace_shift=1,
    batch_size=256,
    epochs=10,
    log_interval=1,
    lr=5e-4,
    hidden_dim=512,
    use_residual=True,
    attention_reduction=4,
    negative_slope=0.5,
    merge_mode='mean',
    merge_sigma=0.25,
    custom_weight=None,
):
    """
    Denoise a single 2D gather using LADNet.

    Parameters
    ----------
    noisy : np.ndarray
        2D array [n1, n2]
    merge_mode : str
        'mean', 'hann', 'hamming', 'gaussian', 'custom'

    Returns
    -------
    result : dict
        {
            'final_denoised': np.ndarray,
            'loss_history': list[float],
            'saved_outputs': list[(epoch, denoised_array)],
            'merge_mode': str,
        }
    """
    # print(f'Noisy shape: {noisy.shape}')

    patch_dim = time_size * trace_size
    n1, n2 = noisy.shape

    noisy_torch = torch.tensor(noisy, dtype=torch.float32, device=device)

    patch_torch, meta = cg_patch_torch(
        noisy_torch,
        l1=time_size,
        l2=trace_size,
        o1=time_shift,
        o2=trace_shift
    )
    print(f'Patch shape: {patch_torch.shape}')

    dataset = MyDataset(patch_torch.T.unsqueeze(1))   # [N, 1, patch_dim]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = build_ladnet_model(
        patch_dim=patch_dim,
        hidden_dim=hidden_dim,
        use_residual=use_residual,
        attention_reduction=attention_reduction,
        negative_slope=negative_slope,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    saved_outputs = []
    final_denoised = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            output = model(batch)
            loss = logcosh_loss_softplus_approx(output, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)

        if ((epoch + 1) % log_interval == 0) or (epoch == epochs - 1):
            model.eval()
            outputs = []

            with torch.no_grad():
                for batch in testloader:
                    output = model(batch)
                    outputs.append(output)

            predict = torch.cat(outputs, dim=0).squeeze(1).T.contiguous()

            denoised_torch = cg_patch_inv_torch(
                predict,
                n1=n1,
                n2=n2,
                l1=time_size,
                l2=trace_size,
                o1=time_shift,
                o2=trace_shift,
                merge_mode=merge_mode,
                sigma=merge_sigma,
                custom_weight=custom_weight,
            )

            denoised = denoised_torch.detach().cpu().numpy()

            print(
                f'Epoch [{epoch + 1}/{epochs}], '
                f'Loss: {avg_loss:.6f}, Merge: {merge_mode}'
            )

            final_denoised = denoised
            saved_outputs.append((epoch + 1, denoised.astype(np.float32)))

    return {
        'final_denoised': final_denoised,
        'loss_history': loss_history,
        'saved_outputs': saved_outputs,
        'merge_mode': merge_mode,
    }

