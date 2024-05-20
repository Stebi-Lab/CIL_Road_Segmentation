# CIL Road Segmentation Project

This project aims to create a road segmentation method for satellite images. It is compared against 2 baseline models and competes at the ETHZ CIL Road Segmentation Kaggle Competition [https://www.kaggle.com/competitions/ethz-cil-road-segmentation-2024]


Requirements
----
- python3.9

Installation
---------------
### For general installation
Depending on your setup, you might have to run the setup.py script after every change in code (because of hydra!).

If you want CUDA, first run this line otherwise skip
```
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118
```

Run following to setup the project
```
pip install -r requirements.txt
python setup.py install
```

## Baselines