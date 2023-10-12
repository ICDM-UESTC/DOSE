# DOSE
## Brief
 DOSE employs two efficient condition-augmentation techniques to address the challenge that incorporating condition information into DDPMs for SE, based on two key insights: (1) We force the model to prioritize the condition factor when generating samples by training it with dropout operation; (2) We incorporate the condition information into the sampling process by providing an informative adaptive prior. Experiments demonstrate that our approach yields substantial improvements in high-quality and stable speech generation, consistency with the condition factor, and efficiency.

## Environment Requirements
We run the code on a computer with RTX-3090, i7 13700KF, and 128G memory. Install the dependencies via anaconda:


### create virtual environment
```
conda create --name DOSE python=3.8.13
```
### activate environment
```
conda activate DOSE
```
### install pytorch & cudatoolkit
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```
### install speech metrics repo:
### Note: be careful with the repo version, chiefly pesq
```
pip install https://github.com/ludlows/python-pesq/archive/master.zip
pip install pystoi
pip install librosa
```
### install utils (we use ``tensorboard`` for logging)
```
pip install tensorboard
```

## How to train
Before you start training, you'll need to prepare a training dataset. The default dataset is VOICEBANK-DEMAND dataset. You can **download them from [VOICEBANK-DEMAND](https://doi.org/10.7488/ds/2117) and resample it to 16k Hz**. By default, this implementation assumes a sample rate of 16 kHz. If you need to change this value, edit [params.py](https://github.com/lmnt-com/diffwave/blob/master/src/diffwave/params.py).

We train the model via running:

```
python src/DOSE/__main__.py /path/to/model
```
## How to sampling
We inference the audio via running:
```
python src/DOSE/inference.py --fast /path/to/model /path/to/condition /path/to/outputdir
```

The code of CDiffuSE is developed based on the code of [Diffwave](https://github.com/lmnt-com/diffwave) 
