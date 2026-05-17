import os
import numpy as np
import torch
from ProposedMethod_MainFunction import denoise_single_gather, cg_snr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

trace_indices = [99, 299, 499, 699, 899]

clean_3d = np.load('angleGathers0To40.npy')
print('3D clean shape:', clean_3d.shape)

save_dir = 'pro'
os.makedirs(save_dir, exist_ok=True)

results = []

for trace_index in trace_indices:
    print('\n' + '=' * 80)
    print(f'Processing trace_index = {trace_index}')
    print('=' * 80)

    clean = clean_3d[:, :, trace_index].T

    noisy_path = f'angleGathers0To40_AG_trace_index-{trace_index}-noisy.npy'
    if not os.path.exists(noisy_path):
        print(f'[Skip] No noisy file found: {noisy_path}')
        continue

    noisy = np.load(noisy_path)

    noisy_snr = cg_snr(clean, noisy)
    print(f'[Index {trace_index}] Noisy SNR = {noisy_snr:.4f}')

    result = denoise_single_gather(
        noisy=noisy,
        device=device,
        time_size=8,
        trace_size=8,
        time_shift=1,
        trace_shift=1,
        batch_size=512,
        epochs=10,
        log_interval=2,
        lr=5e-4,
        hidden_dim=512,
        use_residual=True,
        attention_reduction=4,
        negative_slope=0.1,
        merge_mode='mean',
        merge_sigma=0.25,
        custom_weight=None,
    )

    # 外部计算每个 epoch 的 SNR 并保存
    snr_history = []
    for epoch_num, denoised in result['saved_outputs']:
        snr_val = cg_snr(clean, denoised)
        snr_history.append((epoch_num, snr_val))
        print(f'[Index {trace_index}] Epoch {epoch_num}, SNR = {snr_val:.4f}')

        save_path = os.path.join(
            save_dir,
            f'AG_trace_index-{trace_index}-pro-{epoch_num}.npy'
        )
        np.save(save_path, denoised)

    final_snr = cg_snr(clean, result['final_denoised'])
    final_path = os.path.join(
        save_dir,
        f'AG_trace_index-{trace_index}-pro-final.npy'
    )
    np.save(final_path, result['final_denoised'].astype(np.float32))

    results.append({
        'trace_index': trace_index,
        'noisy_snr': noisy_snr,
        'final_snr': final_snr,
        'snr_history': snr_history,
        **result
    })

print('\nAll done.')
