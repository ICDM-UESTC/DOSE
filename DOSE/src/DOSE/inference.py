# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import os
import torch
import torchaudio
import librosa
from argparse import ArgumentParser
from tqdm import tqdm
from params import AttrDict, params as base_params
from model import DOSE
from dataset import from_path
from glob import glob


models = {}
device=torch.device('cuda',0)

def predict(condition=None, model_dir=None, params=None, device=device, fast_sampling=None):
  # Lazy load model.
  if not model_dir in models:
    if os.path.exists(f'{model_dir}/weights.pt'):
      checkpoint = torch.load(f'{model_dir}/weights.pt')
    else:
      checkpoint = torch.load(model_dir)
    model = DOSE(AttrDict(base_params)).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    models[model_dir] = model

  model = models[model_dir]
  model.params.override(params)
  training_noise_schedule = np.array(model.params.noise_schedule)
  inference_noise_schedule = np.array(model.params.inference_noise_schedule) if fast_sampling else training_noise_schedule

  talpha = 1 - training_noise_schedule
  talpha_cum = np.cumprod(talpha)
  
  beta = inference_noise_schedule
  alpha = 1 - beta

  alpha_cum = np.cumprod(alpha)
  


  with torch.no_grad():
    # Change in notation from the DiffWave paper for fast sampling.
    # DiffWave paper -> Implementation below
    # --------------------------------------
    # alpha -> talpha
    # beta -> training_noise_schedule
    # gamma -> alpha
    # eta -> beta
    
    T = []
    for s in range(len(inference_noise_schedule)):
      for t in range(len(training_noise_schedule) - 1):
        if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
          twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
          T.append(t + twiddle)
          break
    T = np.array(T, dtype=np.float32)
    # print(T)
    
    condition = condition.to(device)
    noise_level=torch.tensor(talpha_cum.astype(np.float32))
    noise_scale = torch.from_numpy(talpha_cum**0.5).float().unsqueeze(1).to(device)
    
    model.eval()
    time_step = model.params.step1
    _step = torch.full([1],time_step)
    noise_scale = noise_level[_step].unsqueeze(1).to(device)
    noise_scale_sqrt = noise_scale**0.5
    noise = torch.randn_like(condition).to(device)
    
    
    audio = noise_scale_sqrt * condition + (1.0 - noise_scale)**0.5 * noise
    audio = model(audio, torch.tensor([T[time_step]], device=audio.device), condition)
    audio = torch.clamp(audio, -1.0, 1.0).squeeze(1)
    audio = audio.type(torch.float32)
    
    
    time_step = model.params.step2
    audio = 0.5*(audio + condition)
    _step = torch.full([1],time_step)
    noise_scale = noise_level[_step].unsqueeze(1).to(device)
    noise_scale_sqrt = noise_scale**0.5
    noise = torch.randn_like(condition).to(device)
    audio = noise_scale_sqrt * audio + (1.0 - noise_scale)**0.5 * noise
    audio = model(audio, torch.tensor([T[time_step]], device=audio.device), condition)
    audio = torch.clamp(audio, -1.0, 1.0).squeeze(1)
    audio = audio.type(torch.float32)
      

  return audio, model.params.sample_rate




def _nested_map(struct, map_fn):
  if isinstance(struct, tuple):
    return tuple(_nested_map(x, map_fn) for x in struct)
  if isinstance(struct, list):
    return [_nested_map(x, map_fn) for x in struct]
  if isinstance(struct, dict):
    return { k: _nested_map(v, map_fn) for k, v in struct.items() }
  return map_fn(struct)



def main(args):
  if args.condition_path:

    print("condition_path:",args.condition_path)
    outputpath = args.output
    if not os.path.exists(outputpath):
      os.makedirs(outputpath)

    specnames = []
    for path in args.condition_path:
      specnames += glob(f'{path}/*.wav', recursive=True)
    
    output_path = os.path.join(outputpath, specnames[0].split("/")[-2])
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    
    
    for spec in tqdm(specnames):
        condition, _ = librosa.load(spec,sr=16000)
        condition = torch.tensor(condition)
        condition = condition.unsqueeze(0)
        condition = condition.to(device)
        
        
        audio, sr = predict(condition, model_dir=args.model_dir, fast_sampling=args.fast, params=base_params)
        output_name = os.path.join(output_path, spec.split("/")[-1])
        torchaudio.save(output_name, audio.cpu(), sample_rate=sr)


if __name__ == '__main__':
  parser = ArgumentParser(description='runs inference on a spectrogram file generated by diffwave.preprocess')
  parser.add_argument('model_dir',
      help='directory containing a trained model (or full path to weights.pt file)')
  parser.add_argument('condition_path',nargs = '+',
      help='path to a wave file to be enhance')
  parser.add_argument('output', 
      help='output file name')
  parser.add_argument('--fast', '-f', action='store_true',
      help='fast sampling procedure')
  main(parser.parse_args())