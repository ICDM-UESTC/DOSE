# DOSE: Diffusion Dropout with Adaptive Prior for Speech Enhancement
![](https://img.shields.io/badge/python-3.8.13-green)![](https://img.shields.io/badge/pytorch-1.13.1-green)![](https://img.shields.io/badge/cudatoolkit-11.7.0-green)

## Newest

We are currently working on an extended version of this work, and the project page will be released upon its completion.

## Brief
 DOSE employs two efficient condition-augmentation techniques to address the challenge of incorporating condition information into DDPMs for SE, based on two key insights: 
 *  We force the model to prioritize the condition factor when generating samples by training it with dropout operation;
 *  We incorporate the condition information into the sampling process by providing an informative adaptive prior.

Experiments demonstrate that our approach yields substantial improvements in high-quality and stable speech generation, consistency with the condition factor, and efficiency.

We fixed a bug with the loss function in learner.py:
```python
audio_orig = features['clean_speech'].clone()
...
loss = self.loss_fn(audio_orig, predicted.squeeze(1))
```
And retest on the VB and CHIME-4.

We upload the [pre-trained model](https://github.com/ICDM-UESTC/DOSE/releases/tag/v1)(with bug in loss), trained on VB with 0.5 as the dropout ratio:

csig:3.8357 cbak:3.2350 covl:3.1840 pesq:2.5430 ssnr:8.9398 stoi:0.9335 on VB (step 1=40, step 2=15)

csig:2.8673 cbak:2.1805 covl:2.1647 pesq:1.5709 ssnr:1.6121 stoi:0.8673 on CHIME-4 (step 1=35, step 2=0)

We also release the retrained [pre-trained model](https://github.com/ICDM-UESTC/DOSE/releases/tag/v2) after fixing the bug, also trained on VB with 0.5 as the dropout ratio:

csig:3.8264 cbak:3.2791 covl:3.1965 pesq:2.5878 ssnr:9.3684 stoi:0.9336 on VB (step 1=40, step 2=10)

csig:2.7520 cbak:2.1276 covl:2.0501 pesq:1.4693 ssnr:1.8087 stoi:0.8304 on CHIME-4 (step 1=10, step 2=0)


## Environment Requirements
**Note: be careful with the repo version, especially PESQ**

 We run the code on a computer with `RTX-3090`, `i7 13700KF`, and `128G` memory. The code was tested with `python 3.8.13`, `pytorch 1.13.1`, `cudatoolkit 11.7.0`. Install the dependencies via [Anaconda](https://www.anaconda.com/):

```
# create a virtual environment
conda create --name DOSE python=3.8.13

# activate environment
conda activate DOSE

# install pytorch & cudatoolkit
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# install speech metrics repo:
pip install https://github.com/ludlows/python-pesq/archive/master.zip
pip install pystoi
pip install librosa

# install utils (we use ``tensorboard`` for logging)
pip install tensorboard
```

## How to train
Before you start training, you'll need to prepare a training dataset. The default dataset is VOICEBANK-DEMAND dataset. You can **download them from [VOICEBANK-DEMAND](https://doi.org/10.7488/ds/2117) and resample them to 16 kHz**. By default, this implementation assumes the sampling steps are `35&15` steps and the sample rate of 16 kHz. If you need to change these values, edit [params.py](https://github.com/ICDM-UESTC/DOSE/blob/main/src/DOSE/params.py).

We train the model via running:

```
python src/DOSE/__main__.py /path/to/model
```
## How to inference
We generate the audio via running:
```
python src/DOSE/inference.py /path/to/model /path/to/condition /path/to/output_dir
```

## How to evaluate
We evaluate the generated samples via running:

```
python src/DOSE/metric.py /path/to/clean_speech /path/to/output_dir
```

## Folder Structure

```tex
└── DOSE──
	├── src
	│	├── init.py 
	│	├── main.py # run the model for training
	│	├── dataset.py # Preprocess the dataset and fill/crop the speech for the model running
	│	├── inference.py # Run model for inferencing speech and adjust inference-steps
	│	├── learner.py # Load the model params for training/inferencing and saving checkpoints
	│	├── model.py # The neural network code of the proposed DOSE
	│	├── params.py # The diffusions, model, and speech params
	└── README.md
```

The code of DOSE is developed based on the code of [Diffwave](https://github.com/lmnt-com/diffwave) 

### Correction of Typographical Error in Experimental Results
We have identified a typographical error in the experimental results presented in our paper. We sincerely apologize for any inconvenience this may have caused. Specifically, the error is in Table 1, under the "CHiME4" dataset for the "Unprocessed" method. The correct values are:

- Original incorrect value:

  STOI: **72.3** & PESQ: **1.22** & CSIG: **2.21** & CBAK: **1.95** & COVL: **1.63**
  
- Corrected value:

  STOI: **87.0** & PESQ: **1.27** & CSIG: **2.61** & CBAK: **1.92** & COVL: **1.88**
