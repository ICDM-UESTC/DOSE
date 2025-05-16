# DOSE+: A Timestep-Aware Dropout Strategy for Diffusion Models in Speech Enhancement
The implementation of DOSE+: A Timestep-Aware Dropout Strategy for Diffusion Models in Speech Enhancement.
## Environment Requirements
```
# create virtual environment
conda create --name DOSE_plus python=3.9.0

# activate environment
conda activate DOSE_plus

# install required packages
pip install -r requirements.txt
```
## How to train
python train.py --log_dir <path_to_model> --base_dir <path_to_dataset>
## How to evaluate
python enhancement.py --test_dir <path_to_noisy> --enhanced_dir <path_to_enhanced> --ckpt <path_to_model_checkpoint>

python calc_metrics.py --clean_dir <path_to_clean> --noisy_dir <path_to_noisy> --enhanced_dir <path_to_enhanced>
## Folder Structure
```
.
├── calc_metrics.py
├── dose_plus
│   ├── backbones
│   │   ├── __init__.py
│   │   ├── ncsnpp.py
│   │   ├── ncsnpp_utils
│   │   │   ├── layerspp.py
│   │   │   ├── layers.py
│   │   │   ├── normalization.py
│   │   │   ├── op
│   │   │   │   ├── fused_act.py
│   │   │   │   ├── fused_bias_act.cpp
│   │   │   │   ├── fused_bias_act_kernel.cu
│   │   │   │   ├── __init__.py
│   │   │   │   ├── upfirdn2d.cpp
│   │   │   │   ├── upfirdn2d_kernel.cu
│   │   │   │   └── upfirdn2d.py
│   │   │   ├── up_or_down_sampling.py
│   │   │   └── utils.py
│   │   └── shared.py
│   ├── data_module.py
│   ├── model.py
│   ├── sampling
│   │   ├── correctors.py
│   │   ├── __init__.py
│   │   └── predictors.py
│   ├── sdes.py
│   └── util
│       ├── inference.py
│       ├── other.py
│       ├── registry.py
│       ├── semp.py
│       └── tensors.py
├── enhancement.py
├── README.md
├── requirements.txt
└── train.py
```

