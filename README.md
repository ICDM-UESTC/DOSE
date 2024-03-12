# DOSE: Diffusion Dropout with Adaptive Prior for Speech Enhancement
![](https://img.shields.io/badge/python-3.8.13-green)![](https://img.shields.io/badge/pytorch-1.13.1-green)![](https://img.shields.io/badge/cudatoolkit-11.7.0-green)
## Brief
 DOSE employs two efficient condition-augmentation techniques to address the challenge that incorporating condition information into DDPMs for SE, based on two key insights: 
 *  We force the model to prioritize the condition factor when generating samples by training it with dropout operation;
 *  We incorporate the condition information into the sampling process by providing an informative adaptive prior.

Experiments demonstrate that our approach yields substantial improvements in high-quality and stable speech generation, consistency with the condition factor, and efficiency.

## Environment Requirements
**Note: be careful with the repo version, chiefly pesq**

 We run the code on a computer with `RTX-3090`, `i7 13700KF`, and `128G` memory. The code was tested with `python 3.8.13`, `pytorch 1.13.1`, `cudatoolkit 11.7.0`. Install the dependencies via [Anaconda](https://www.anaconda.com/):

```
# create virtual environment
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
Before you start training, you'll need to prepare a training dataset. The default dataset is VOICEBANK-DEMAND dataset. You can **download them from [VOICEBANK-DEMAND](https://doi.org/10.7488/ds/2117) and resample it to 16 kHz**. By default, this implementation assumes the sampling steps are `35&15` steps and the sample rate of 16 kHz. If you need to change these value, edit [params.py](https://github.com/ICDM-UESTC/DOSE/blob/main/src/DOSE/params.py).

We train the model via running:

```
python src/DOSE/__main__.py /path/to/model
```
## How to sampling
We inference the audio via running:
```
python src/DOSE/inference.py /path/to/model /path/to/condition /path/to/outputdir
```

## How to evaluate
We evaluate the generated samples via running:

```
python src/DOSE/metric.py /path/to/clean_speech /path/to/outputdir
```

## Folder Structure

```tex
└── DOSE──
	├── src
	│	├── init.py 
	│	├── main.py # run the model for training
	│	├── dataset.py # Preprocess dataset and fill/crop the speech for the model running
	│	├── inference.py # Run model for inferencing speech and adjust inference-steps
	│	├── learner.py # Load the model params for training/inferencing and saving checkpoints
	│	├── model.py # The neural network code of the proposed DOSE
	│	├── params.py # The diffusions, model and speech params
	└── README.md
```

The code of DOSE is developed based on the code of [Diffwave](https://github.com/lmnt-com/diffwave) 
