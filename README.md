# COMP9517 Project 
## Contributer
- Jingyu Sun     (z5525875)
- Mingqi Xu       (z5532071)
- Yanbo Wang（z5603812）
- Yucheng Liu   (z5621848)
- Linyang Yu     (z5619437)

_**This project is a computer vision project for tree segmentation. We include four models: **U-Net**, **DeepLabv3**, **Mask R-CNN**, and **PSPNet**. Each model follows a similar setup and training workflow.**_

---
## Content
- [Set up](#Setup)
- [Model insturction](#Model-insturction)

## 🔧 Setup

Before accessing any model, install the dependencies:

```bash
pip install -r requirements.txt
```

# Model insturction 
- [Mask Rcnn](#MASK-RCNN)
- [DeeplabV3](#Deeplabv3)
- [PSP Net](#PSP-Net)
- [U Net](#U-Net)
## MASKRCNN
### Features

- Mask R-CNN with a ResNet-FPN backbone for instance segmentation.

- Region Proposal Network (RPN) for detecting objects of varying sizes.

- Multi-task loss combining classification, bounding box regression, and mask prediction.

- Supports training, validation, and per-class evaluation metrics.

- Handles small-object segmentation with customizable anchor sizes.


### Project Structure

- train.py: Main script to run training or inference also including dataset
- predict.py: predict performance with your trained model
- datamatch: data preprocess of paired dataset

1. Prepare Data
   
   Run the datamatch file and get the paired_dataset.pkl file change 

2. training
   
   change the pkl directory inside train file and run the following command to customize your train
   
   ```bash
   python train.py --epochs 100 --batch_size 8 --lr 0.0005 --img_size 256 --loss focal+dice --optimizer adam 
   ```
   

3. predict

   Change the model path inisde predict and test your customized model
   

### Training Process

| Argument        | Description                                                               | Default    | Options                                                       |
| --------------- | ------------------------------------------------------------------------- | ---------- | ------------------------------------------------------------- |
| `--epochs`      | Number of training epochs                                                 | `100`      | *any integer*                                                 |
| `--batch_size`  | Training batch size                                                       | `8`        | *any integer*                                                 |
| `--lr`          | Learning rate                                                             | `0.0005`   | *any float*                                                   |
| `--img_size`    | Image size (height and width will be square)                              | `256`      | *any integer (e.g., 256, 512)*                                |
| `--loss`        | Loss function to handle imbalance                                         | `bce+dice` | `bce+dice`, `focal`, `tversky`, `focal+dice`, `focal+tversky` |
| `--optimizer`   | Optimizer for training                                                    | `adam`     | `adam`, `adamw`, `sgd`                                        |
| `--num_workers` | Number of data loading workers (set 0 if system has low RAM)              | `0`        | *any integer*                                                 |               |


## Deeplabv3
### Features

- DeepLabv3 with ResNet backbone for semantic segmentation.

- Atrous Spatial Pyramid Pooling (ASPP) for multi-scale feature extraction.

- Optional CBAM attention module for improved feature selection.

- Handles imbalanced datasets with Dice, Focal, or Tversky loss.

- Supports both RGB-only and RGB + NIR inputs.

- CBAM attention model implementation


### Project Structure

- train.py: Main script to run training or inference also including dataset
- predict.py: predict performance with your trained model
- datamatch: data preprocess of paired dataset

1. Prepare Data
   
   Run the datamatch file and get the paired_dataset.pkl file

2. training
   
   change the pkl directory inside train file and run the following command to customize your train
   
   ```bash
   python train.py --epochs 100 --batch_size 8 --lr 0.0005 --img_size 256 --loss focal+dice --optimizer adam --use_cbam
   ```



3. predict

   Change the model path inisde predict and test your customized model
   

### Training Process

| Argument        | Description                                                               | Default    | Options                                                       |
| --------------- | ------------------------------------------------------------------------- | ---------- | ------------------------------------------------------------- |
| `--epochs`      | Number of training epochs                                                 | `100`      | *any integer*                                                 |
| `--batch_size`  | Training batch size                                                       | `8`        | *any integer*                                                 |
| `--lr`          | Learning rate                                                             | `0.0005`   | *any float*                                                   |
| `--img_size`    | Image size (height and width will be square)                              | `256`      | *any integer (e.g., 256, 512)*                                |
| `--loss`        | Loss function to handle imbalance                                         | `bce+dice` | `bce+dice`, `focal`, `tversky`, `focal+dice`, `focal+tversky` |
| `--optimizer`   | Optimizer for training                                                    | `adam`     | `adam`, `adamw`, `sgd`                                        |
| `--num_workers` | Number of data loading workers (set 0 if system has low RAM)              | `0`        | *any integer*                                                 |
| `--use_cbam`    | Enable CBAM (Convolutional Block Attention Module) in the ResNet backbone | `False`    | flag only (add `--use_cbam` to activate)                      |

## PSP Net
### Features

- PSPNet with a ResNet backbone for deep feature extraction.
- Pyramid pooling module to capture global context.
- Skip connections to blend fine details with coarse features.
- Supports training, validation, and visualization of predictions.
- Handles unbalanced datasets with optional loss functions.


### Project Structure

- main.py: Main script to run training or inference.
- config.py: Configuration settings (e.g., USE_NRG, INPUT_CHANNELS).
- utils.py: Helper functions for visualization and metrics.
- model.py: PSPNet model definition.
- dataset.py: Custom dataset loader for NRG_images.
- trainer.py: Training loop and validation logic.
- usa_segmentation\NRG_images: Directory with RGB + NIR image data.
- results/: Folder for saving model outputs and visualizations.

1. Prepare Data

   - Place your images in usa_segmentation\NRG_images (e.g., NRG_wa019_2023_n_33_15_0.png).
   - Ensure masks are in a corresponding directory (e.g., usa_segmentation\masks).
   - Images should be 4-channel (RGB + NIR) if USE_NRG = True, or 3-channel (RGB) if USE_NRG = False.

2. Check Config

   - Edit config.py to set USE_NRG = True (for 4 channels) or False (for 3 channels).
   - Adjust INPUT_CHANNELS and other hyperparameters as needed.

### Usage

Run the project with:

bash

```
python main.py --mode train
```

- --mode train: Starts training with the dataset.
- Output will be saved in results/ (e.g., epoch_001_predictions).

#### Training Process

- **Data Loading**: Loads images and masks from usa_segmentation\NRG_images using dataset.py.
- **Model**: PSPNet processes the input with a ResNet backbone and pyramid pooling.
- **Optimization**: Uses Adam optimizer (default lr 0.001) with a scheduler (e.g., ReduceLROnPlateau).
- **Loss**: Default is Cross Entropy Loss; can switch to Dice Loss or Focal Loss for unbalanced data.
- **Visualization**: Generates prediction masks every 10 epochs or at best model, saved as four-panel images (NIR | RGB | Ground Truth | Prediction).

## U Net

### Features

- U-Net encoder-decoder architecture for semantic segmentation.

- Skip connections for preserving spatial detail and improving boundary accuracy.

- Flexible input channels (supports RGB, NIR, or combined channels).

- Supports a variety of loss functions: Dice, BCE, Focal, Tversky, and their combinations.

- Includes data augmentation (flip, rotate, brightness, etc.) via Albumentations.

- Easily customizable hyperparameters (epochs, batch size, learning rate, etc.).

- Suitable for both small and large segmentation datasets.


### Project Structure

- train.py
  
Main script for training the U-Net model on your paired dataset. Supports hyperparameter customization, logging, and model saving.

- predict.py
  
Script for running inference and evaluating the trained U-Net model on test data. Outputs segmentation results and performance metrics.

- dataset.py
  
Implements the custom SegmentationDataset class for loading images and masks, data augmentation, and preprocessing.

- U_Net.py
  
Directory containing the implementation of the U-Net model architecture and related modules (e.g., DoubleConv, UNet class).

- make_paired_samples.py
  
Utility script to generate paired datasets (e.g., combining RGB and NIR images with their corresponding masks) and save them as a .pkl file for training and evaluation.

```bash
python train.py --epochs 100 --batch_size 8 --lr 0.0005 --img_size 256 --loss focal+dice --optimizer adam 
```
   

### Training Process

| Argument        | Description                                                               | Default    | Options                                                       |
| --------------- | ------------------------------------------------------------------------- | ---------- | ------------------------------------------------------------- |
| `--epochs`      | Number of training epochs                                                 | `100`      | *any integer*                                                 |
| `--batch_size`  | Training batch size                                                       | `8`        | *any integer*                                                 |
| `--lr`          | Learning rate                                                             | `0.0005`   | *any float*                                                   |
| `--img_size`    | Image size (height and width will be square)                              | `256`      | *any integer (e.g., 256, 512)*                                |
| `--loss`        | Loss function to handle imbalance                                         | `bce+dice` | `bce+dice`, `focal`, `tversky`, `focal+dice`, `focal+tversky` |
| `--optimizer`   | Optimizer for training                                                    | `adam`     | `adam`, `adamw`, `sgd`                                        |
| `--num_workers` | Number of data loading workers (set 0 if system has low RAM)              | `0`        | *any integer*                                                 |

