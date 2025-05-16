from time import sleep
import librosa
import numpy as np
import torch
from torchaudio import load

from pesq import pesq
from pystoi import stoi
from tqdm import tqdm

from .other import pad_spec
from .semp import composite

from pathlib import Path
import os

import matplotlib.pyplot as plt

sr = 16000
snr = 0.5
N = 50
corrector_steps = 0

def evaluate_model(model, num_eval_files):
    clean_files = model.data_module.valid_set.clean_files
    noisy_files = model.data_module.valid_set.noisy_files
    
    total_num_files = len(clean_files)
    
    indices = torch.linspace(0, total_num_files-1, num_eval_files, dtype=torch.int)
    clean_files = list(clean_files[i] for i in indices)
    print(f"Total number of valid files: {total_num_files}, evaluating for SE ...")
    noisy_files = list(noisy_files[i] for i in indices)
    _pesq = 0
    _estoi = 0

    for j,(clean_file, noisy_file) in enumerate(zip(clean_files, noisy_files)): 
        print(j,end='\r')   
        x,_=librosa.load(clean_file, sr=16000)
        y,_=librosa.load(noisy_file, sr=16000)
        x=torch.tensor(x)
        y=torch.tensor(y)
        x = x.view(1, -1)
        y = y.view(1, -1)
        T_orig = x.size(1) 

        norm_factor = y.abs().max()
        y = y / norm_factor

        Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        y = y * norm_factor

        sampler = model.get_sampler(
                'reverse_dose_plus', 'ald', Y.cuda(), N=N, 
                corrector_steps=0, snr=snr)
        sample, _ = sampler()
        
        x = x.squeeze().cpu().numpy()
        y = y.squeeze().cpu().numpy()
    
        x_hat = model.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu().numpy() 

        _pesq += pesq(sr, x, x_hat, 'wb') 
        _estoi += stoi(x, x_hat, sr, extended=True)

        return _pesq/num_eval_files, _estoi/num_eval_files



