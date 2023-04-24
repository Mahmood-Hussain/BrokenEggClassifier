# Broken Egg Classifier Implemented in PyTorch

## Introduction
Image classification model to classify eggs into three categories `crack,` `empty, and` `good`. The script reads in command line arguments, such as the path to the dataset, the type of model to use, the learning rate, and the batch size. It then defines transforms, datasets, and dataloaders based on these arguments.

I used following details to train the model:
- Model: EggClassifierV1, EggClassifierV2
- Dataset: https://www.kaggle.com/datasets/frankpereny/broken-eggs
- Batch Size: 16
- Learning Rate: 0.001
- Epochs: 25, 10 respectively
- Optimizer: Adam
- Loss Function: Cross Entropy Loss
- LR Scheduler: ReduceLROnPlateau
- Transformations: 
```python
transforms.Resize((args.img_size + 64, args.img_size + 64)),
transforms.CenterCrop((args.img_size, args.img_size)),
transforms.RandomRotation(30),
transforms.RandomHorizontalFlip(),

# color transforms
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
transforms.RandomGrayscale(p=0.2),
transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
```



## Requirements
`pip install torch, torchvision`

## Running and Training 
```bash
git clone https://github.com/Mahmood-Hussain/BrokenEggClassifier.git
cd BrokenEggClassifier
```

```bash
python train.py --data_dir /path/to/broken/egg/dataset --num_epochs 20 --lr 0.001 --save_dir work_dirs/resnet_based --model EggClassifierV2 --batch_size 32
```


## See the more options
```bash
python train.py --help
```
## Results
| Model | Epochs | Val Accuracy | Val Loss |
| --- | --- | --- | --- |
| EggClassifierV1 | 25 | 0.85 | 0.39 |
| EggClassifierV2 | 10 | 0.96 | 0.10 |


## Dataset and DataLoader
Dataset was downloaded from Kaggle here: https://www.kaggle.com/datasets/frankpereny/broken-eggs 
The `datasets/EggDataset` constructor initializes the directory containing the data, as well as a list of classes (subdirectories in the data directory) and a mapping from class names to their corresponding index in the list of classes. It then iterates through all image files in the data directory and creates a list of tuples, where each tuple contains the path to an image file and the index of the class to which it belongs.

The `__len__` method returns the total number of samples in the dataset, which is the length of the samples list.

The `__getitem__` method loads an image from its path using the PIL library, applies any specified transformations, and returns the image tensor and its corresponding label as a tuple. The label is represented as a torch.tensor with dtype torch.long.

## EggClassifierV1
This is a custom model architecture which consists of a series of convolutional layers with batch normalization and ReLU activation functions, followed by max pooling layers. The output of the convolutional layers is then flattened and passed through three fully connected layers, each with ReLU activation and a dropout layer. The final fully connected layer has num_classes output units corresponding to the number of classes in the dataset.

The `__init__` method initializes the layers of the network and the forward method defines the forward pass of the network.
The code at the bottom of the file tests the model by creating an instance of EggClassifierV1, passing a random input tensor through the model, and printing the output shape and values.

## EggClassifierV2
The EggClassifierV2 class uses a pre-trained ResNet18 model from the torchvision package. The resnet18 function is used to create the pre-trained model, and the weights argument is set to ResNet18_Weights.DEFAULT, which means that the weights will be initialized to the default pre-trained values for the ResNet18 model. The final fully connected layer of the pre-trained model is replaced with an identity function, and a new fully connected layer is added to output the desired number of classes.

The benefits of using a pre-trained model in this case are that it can save a lot of training time, since the ResNet18 model has already been trained on a large dataset (ImageNet) and has learned to extract useful features from images. This can help improve the accuracy of the final classification model, even with a relatively small dataset of egg images. Additionally, using a pre-trained model can help avoid overfitting, since the weights of the pre-trained model have already been regularized on a large dataset.


Author: Mahmood Hussain (
[LinkedIn](https://www.linkedin.com/in/mhbhat/) |
[GitHub](
    https://github.com/Mahmood-Hussain
    ) |
    [Website](https://mhbhat.com/)
    )
