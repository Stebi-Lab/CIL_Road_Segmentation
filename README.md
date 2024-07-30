# CIL Road Segmentation Project

This project aims to create a road segmentation method for satellite images. It is compared against 2 baseline models and competes at the ETHZ CIL Road Segmentation Kaggle Competition [https://www.kaggle.com/competitions/ethz-cil-road-segmentation-2024]


Requirements
----
- python3.9

Installation
---------------
### For general installation

If you want CUDA, first run this line otherwise skip
```
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118
```

Run following to setup the project
```
pip install -r requirements.txt
python setup.py install
```

## User Guide


### Data Download

Download the zipped datafolder from [here TODO](https://drive.google.com/file/d/1pQX9PVbFvTrSGp8py4prKIGSd-mY5GBT/view), 
unpack it and place it in the root folder of the project.

### Running 

🔴IMPORTANT❗🔴

Although the train.py/fine_tune.py/test.py file are setup, a few manual changes are needed. The training procedure can be run straight out of the box. The fine-tuning and testing code relies on checkpoint paths generated from the training procedure. These paths are unique and need to be set before each run❗ They are marked as `TODO` in the code. We provide specific instructions below.

Run in order:
- `train.py`: no changes needed, trains UNet++ with encoder on the extended dataset available from the above link.
- `fine_tune.py`: set checkpoint path & number, fine_tune on Kaggle dataset
- `test.py`: set checkpoint path & number, produces submission.csv

These python files must be run from the root directory for the relative paths to be correct!

#### Different Models

The above configuration runs our best model staight out-of-the-box. We have developed configurable code, such that all other model configurations can be run by adjusting the provided config yaml in `train.py` and by providing the corresponding checkpoint path in `fine_tune.py` and `test.py`. Caution is needed as running the other configurations requires precise instructions in the config files. 

