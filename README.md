# Introduction

This repository refer to [3DDFA](https://github.com/cleardusk/3DDFA) and [DECA](https://github.com/YadiraF/DECA). 

**3DDFA** use 40 shape parameter and 10 expression parameter. In order to improve the reconstruction accuracy, this repository try to use 199 shape parameter and 29 expression parameter. but the result don't seem to improve much. (dataset:300W_LP, net: mobilenet_v1) 

**DECA** has better result in close eye and the outline of the mouth. this repository just extract the 3d facial landmark detection code.

## Requirements
* Python 3.7 (numpy, skimage, scipy, opencv)  
* PyTorch >= 1.6 (pytorch3d)  
  You can run 
  ```bash
  pip install -r requirements.txt
  ```
  Then follow the instruction to install [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md).
  
## Usage
1. Download config(3DDFA) and model
  - config file [Google Cloud](https://drive.google.com/file/d/1RsGUwHqfQRCsMNXGi8SkaOcjhY8e9t7y/view?usp=sharing)
  - related model [Google Cloud](https://drive.google.com/file/d/1mBQr4L6BIZ534wBSz_mzx5HAmJka9-G9/view?usp=sharing)
  
2. Run demos

3DDFA
```
python detect_landmark_bfm.py video/demo.mp4
```

DECA
```
python detect_landmark_flame.py video/demo.mp4
```

facial landmark [face-alignment](https://github.com/1adrianb/face-alignment)


## License
The code is available for non-commercial scientific research purposes. 

## Notes
We use the FLAME.py from [FLAME_PyTorch](https://github.com/soubhiksanyal/FLAME_PyTorch)
