# Connectivity-Aware-Airway-Segmentaion

## Introduction
Detailed pulmonary airway segmentation is a clinically important task for endobronchial intervention and treatment of peripheral pulmonary lesions. 
Convolutional Neural Networks (CNNs) are promising for automated analysis of medical imaging, which however performs poorly on airway segmentation. 
Specifically, breakage of small bronchi distals cannot be effectively eliminated in the prediction results of CNNs, which is detrimental to use as a reference for bronchoscopic-assisted surgery. 
We proposed a connectivity-aware segmentation framework to improve the performance of airway segmentation. 
A Connectivity-Aware Surrogate (CAS) module is first proposed to balance the training progress within-class distribution. 
Furthermore, a Local-Sensitive Distance (LSD) module is designed to identify the breakage and minimize the variation of the distance map between the prediction and ground-truth.

## Usage
To quick start, we provided the pretained networks, and can try the script in ```tests/_test_airway_model```

```
python _test_airway_model.py
```

You can download our pretrained checkpoint from [here](https://drive.google.com/file/d/1_Uz2DzVHAa0S1fRNqyfYRYQZQixbO_dT/view?usp=sharing). The configs and models are specified in 
```configs/airway_config``` and ```networks/airway_network```.

The implementation of two modules CAS and LSD is modularized in the ```networks/CAS```,```networks/LSD```.

