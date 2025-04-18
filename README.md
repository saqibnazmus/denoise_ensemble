# Title: Denoising single images by feature ensemble revisited - _PyTorch implementation_

**MDPI Sensors 2022**

## Abstract:
Image denoising is still a challenging issue in many computer vision subdomains. Recent studies have shown that significant improvements are possible in a supervised setting. However, a few challenges, such as spatial fidelity and cartoon-like smoothing, remain unresolved or decisively overlooked. Our study proposes a simple yet efficient architecture for the denoising problem that addresses the aforementioned issues. The proposed architecture revisits the concept of modular concatenation instead of long and deeper cascaded connections, to recover a cleaner approximation of the given image. We find that different modules can capture versatile representations, and a concatenated representation creates a richer subspace for low-level image restoration. The proposed architecture’s number of parameters remains smaller than in most of the previous networks and still achieves significant improvements over the current state-of-the-art networks.

## Implementations

The below sections details what python requirements are required to set up for the project. 
dataset.

### Dependencies
- PyTorch, NumPy, OpenCV

### Dataset
We have worked on DIV2K dataset for the training procedure. For the testing procedure on synthetic noisy images, we have used BSD68, KODAK24 and Set14 datasets. For real noisy images, we have used the SIDD benchmark, CC and PolyU datasets.
```
python dataset.py 
```

##Dataloader
For loading the data, run the following: 
```
python dataloader.py 
```

### Train
We have provided the training code.  
```
python train_synthetic.py 
```

### Test synthetic
To get the results of the testing procedure, write the following on your command prompt and run. 

```
python test_AWGN.py"
```

### Test SIDD
To get the results of the testing procedure, write the following on your command prompt and run. 

```
python test_sidd.py"
```

### Test DnD
To get the results of the testing procedure, write the following on your command prompt and run. 

```
python test_dnd.py"
```
## Synthetic results 
We have uploaded here our synthetic results on BSD68, Kodak24, and Urban100 datasets. Our SIDD benchmark and validation datasets will be published later. 
