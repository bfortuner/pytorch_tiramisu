# One Hundred Layers Tiramisu
PyTorch implementation of [The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/pdf/1611.09326).

## Setup
Things to install and do before running

## Resources

* [Project Thread](http://forums.fast.ai/t/one-hundred-layers-tiramisu/2266)
* [PyTorch Docs](http://pytorch.org/docs)
* [PyTorch Tutorials](http://pytorch.org/tutorials)
* [Fast.ai Wiki](http://wiki.fast.ai/index.php/Main_Page)
* [Fast.ai Forums](http://forums.fast.ai)

## References

* https://github.com/bamos/densenet.pytorch
* https://github.com/liuzhuang13/DenseNet

## Architecture

**FirstConvLayer**

* 3x3 Conv2D (pad=, stride=, in_chans=3, out_chans=48)

**DenseLayer**

* BatchNorm
* ReLU
* 3x3 Conv2d (pad=, stride=, in_chans=, out_chans=) - "no resolution loss" - padding included
* Dropout (.2)

**DenseBlock**

* Input = FirstConvLayer, TransitionDown, or TransitionUp
* Loop to create L DenseLayers (L=n_layers)
* On TransitionDown we Concat(Input, FinalDenseLayerActivation)
* On TransitionUp we do not Concat with input, instead pass FinalDenseLayerActivation to TransitionUp block

**TransitionDown**

* BatchNorm
* ReLU
* 1x1 Conv2D (pad=, stride=, in_chans=, out_chans=)
* Dropout (0.2)
* 2x2 MaxPooling

**Bottleneck**

* DenseBlock (15 layers)

**TransitionUp**

* 3x3 Transposed Convolution (pad=, stride=2, in_chans=, out_chans=)
* Concat(PreviousDenseBlock, SkipConnection) - from cooresponding DenseBlock on transition down

**FinalBlock**

* 1x1 Conv2d (pad=, stride=, in_chans=256, out_chans=n_classes)
* Softmax

**FCDenseNet103 Architecture**

* input (in_chans=3 for RGB)
* 3x3 ConvLayer (out_chans=48)
* DB (4 layers) + TD
* DB (5 layers) + TD
* DB (7 layers) + TD
* DB (10 layers) + TD
* DB (12 layers) + TD
* Bottleneck (15 layers)
* TU + DB (12 layers)
* TU + DB (10 layers)
* TU + DB (7 layers)
* TU + DB (5 layers)
* TU + DB (4 layers)
* 1x1 ConvLayer (out_chans=n_classes) n_classes=11 for CamVid
* Softmax

## Training

**Hyperparameters**

* WeightInitialization = HeUniform
* Optimizer = RMSProp
* LR = .001 with exponential decay of 0.995 after each epoch
* Data Augmentation = Random Crops, Vertical Flips
* ValidationSet with early stopping based on IoU or MeanAccuracy with patience of 100 (50 during finetuning)
* WeightDecay = .0001
* Finetune with full-size images, LR = .0001
* Dropout = 0.2
* BatchNorm "we use current batch stats at training, validation, and test time"

**CamVid**

* TrainingSet = 367 frames
* ValidationSet = 101 frames
* TestSet = 233 frames
* Images of resolution 360x480
* Images "Cropped" to 224x224 for training --- center crop?
* FullRes images used for finetuning
* NumberOfClasses = 11 (output)
* BatchSize = 3

**FCDenseNet103**

* GrowthRate = 16 (k, number of filters to each denselayer adds to the ever-growing concatenated output)
* No pretraining
