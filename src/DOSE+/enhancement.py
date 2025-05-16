import glob
import torch
from tqdm import tqdm
from os import makedirs
from soundfile import write
from os.path import join, dirname
from argparse import ArgumentParser
from librosa import resample
import librosa

from dose_plus.util.other import set_torch_cuda_arch_list
set_torch_cuda_arch_list()

from dose_plus.model import ScoreModel
from dose_plus.util.other import pad_spec


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--test_dir",
        type=str, help='Directory containing the test data')
    parser.add_argument("--enhanced_dir",
        type=str, help='Directory containing the enhanced data')
    parser.add_argument("--ckpt",
        type=str,  help='Path to model checkpoint')
    
    parser.add_argument("--corrector", type=str, choices=("ald", "langevin", "none"), default="ald", help="Corrector class for the sampler.")
    parser.add_argument("--corrector_steps", type=int, default=0, help="Number of corrector steps")
    parser.add_argument("--snr", type=float, default=0.5, help="SNR value for (annealed) Langevin dynmaics")
    parser.add_argument("--N", type=int, default=50, help="Number of reverse steps")
    parser.add_argument("--device",default='cuda:0', type=str, help="Device to use for inference")
    args = parser.parse_args()

    model = ScoreModel.load_from_checkpoint(args.ckpt, map_location=args.device)
    model.eval()

    noisy_files = []
    noisy_files += sorted(glob.glob(join(args.test_dir, '*.wav')))
    noisy_files += sorted(glob.glob(join(args.test_dir, '**', '*.wav')))
    target_sr = 16000
    pad_mode = "zero_pad"

    for noisy_file in tqdm(noisy_files):
        filename = noisy_file.split('/')[-1]
        filename = noisy_file.replace(args.test_dir, "")[1:] # Remove the first character which is a slash
        
        y,sr=librosa.load(noisy_file, sr=16000)
        y=torch.tensor(y)
        y = y.view(1, -1)

        if sr != target_sr:
            y = resample(y, orig_sr=sr, target_sr=target_sr)

        T_orig = y.size(1)   

        norm_factor = y.abs().max()
        y = y / norm_factor
        
        Y = torch.unsqueeze(model._forward_transform(model._stft(y.to(args.device))), 0)
        Y = pad_spec(Y, mode=pad_mode)
        
        sampler = model.get_sampler(
                    'reverse_dose_plus', 'ald', Y.to(args.device), N=50, 
                    corrector_steps=0, snr=args.snr)
        sample, _ = sampler()

        x_hat = model.to_audio(sample.squeeze(), T_orig)

        x_hat = x_hat * norm_factor
        
        makedirs(dirname(join(args.enhanced_dir, filename)), exist_ok=True)
        write(join(args.enhanced_dir, filename), x_hat.cpu().numpy(), target_sr)
