# 1. Packages

```python=
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.linear_model import \
     (LinearRegression, LogisticRegression, Lasso)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from ISLP import load_data
from ISLP.models import ModelSpec as MS
```

**Torch-Specific Imports**
- Main library and essential tools to specify sequentially-structured networks

```python=
import torch
from torch import nn
from torch.optim import RMSprop
from torch.utils.data import TensorDataset
```

- Tools from `torchmetrics` to compute metrics to evaluate performance.
- Tools from `torchinfo` to summarize info of the layers of a model.

```python=
from torchmetrics import (MeanAbsoluteError, R2Score)
from torchinfo import summary
```

- `pytorch_lightning` package simplifies the specification and fitting and evaluate models by reducing amount of boilerplate code needed.

```python=
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
```

- `seed_everything()` set seed.
- `use_deterministic_algorithms` fix algorithms.

```python=
from pytorch_lightning import seed_everything
seed_everything(0, workers=True)
torch.use_deterministic_algorithms(True, warn_only=True)
# Seed set to 0
```

- We use datasets from `torchvision`.
- We use transforms from `torchvision` for preprocessing.

```python=
from torchvision.io import read_image
from torchvision.datasets import MNIST
from torchvision.transforms import (Resize, Normalize,
                                    CenterCrop, ToTensor)
```

- `SimpleDataModule` and `SimpleModule` from `ISLP.torch` are simple versions of objects used in `pytorch_lightning`.
- `ErrorTracker` collects targets and predictions over each mini-batch during validation or testing, enabling metric computation over the entire validation or test data set.

```python=
from ISLP.torch import (SimpleDataModule, SimpleModule,
                        ErrorTracker, rec_num_workers)
```

# 2. MNIST
- Modified National Institute of Standards and Technology.
- A collection of handwritten digits.
- It contains 60,000 training images and 10,000 testing images.
- Each image is 28x28 = 784 pixels.
- Each pixel is an integer between 0 and 255.
- Each image is labeled with the digit it represents, from 0 to 9.
- The images are stored in a flatten format, with 784 pixels per image.

```python=
mnist_train = MNIST(root='data', train=True, download=True, transform=ToTensor())
mnist_test = MNIST(root='data', train=False, download=True, transform=ToTensor())
# mnist_train, mnist_test = [MNIST(root='data', train=train, download=True, transform=ToTensor()) for train in [True, False]]
```

- Neural networks are sensitive to input data scale (like ridge, lasso), so we scale pixel values to [0, 1] by dividing by 255.
- `ToTensor()` from `torchvision.transforms` automatically rescales and converts images to PyTorch tensors with values [0, 1].
- `ToTensor()` also reorders image dimensions from H x W x C to C x H x W.

```python=
# Get a sample image and its label from the training set
image, label = mnist_train[0]
# Plot the image
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f'Label: {label}');
```

![](Figures\mnist_22_0.png)

```python=
# Describe the training dataset
print("Training Dataset:")
print(f"Number of samples: {len(mnist_train)}")
print(f"Image shape: {mnist_train[0][0].shape}")
print(f"Labels: {set(mnist_train.targets.numpy())}")

# Describe the test dataset
print("\nTest Dataset:")
print(f"Number of samples: {len(mnist_test)}")
print(f"Image shape: {mnist_test[0][0].shape}")
print(f"Labels: {set(mnist_test.targets.numpy())}")
```

    Training Dataset:
    Number of samples: 60000
    Image shape: torch.Size([1, 28, 28])
    Labels: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    Test Dataset:
    Number of samples: 10000
    Image shape: torch.Size([1, 28, 28])
    Labels: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

- We create a data module from the training and test datasets, reserving 20% of the training images for validation.

```python=
max_num_workers = rec_num_workers()
mnist_dataModule = SimpleDataModule(mnist_train, mnist_test,
                            validation=0.2,
                            num_workers=max_num_workers,
                            batch_size=256)
```

- Examine data fed into the network by looping through the first two batches of the training dataset:
  - Each batch contains 265 images.
  - Each image has dimensions 1 x 28 x 28 (1 channel, 28 rows, 28 columns).

```python=
for idx, (X_ ,Y_) in enumerate(mnist_dataModule.train_dataloader()):
    print('X: ', X_.shape)
    print('Y: ', Y_.shape)
    if idx >= 1:
        break
```

    X:  torch.Size([256, 1, 28, 28])
    Y:  torch.Size([256])
    X:  torch.Size([256, 1, 28, 28])
    Y:  torch.Size([256])

- We define class MINISTModel that inherits from `nn.Module`.
- Layer 1: `nn.Sequential` is a container for a sequence of layers.
    - `nn.Flatten()` flattens each 1x28x28 image into a 1x784 tensor.
    - `nn.Linear(28*28, 256)` is a fully connected layer with 784 input features and 256 output features.
    - `nn.ReLU()` is a rectified linear unit activation function.
    - `nn.Dropout(0.4)` is a dropout layer that randomly sets 40% of the input units to 0 at each update during training.
- Layer 2: `nn.Sequential` is a container for a sequence of layers.
    - `nn.Linear(256, 128)` is a layer with 256 input features and 128 output features.
    - `nn.ReLU()` and `nn.Dropout(0.3)` serve the same purpose as in the first layer.
- Forward method: combines the two layers with a final fully connected layer with 128 input features and 10 output features.
- We define a forward method of the network. It takes an input tensor `x` and passes it through the sequence of layers defined in `self_forward`.

```python=
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Dropout(0.4))
        self.layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3))
        self._forward = nn.Sequential(
            self.layer1,
            self.layer2,
            nn.Linear(128, 10))
    def forward(self, x):
        return self._forward(x)
```

```python=
mnist_model = MNISTModel()
```

- We verify that the model outputs the expected size using the existing batch `X_` above.

```python=
mnist_model(X_).size()
```

    torch.Size([256, 10])

- We pass a tensor of correct shape. For this case, `X_` matches the expected input shape [batch_size, channel, height, width] = [256, 1, 28, 28].
- We specify the list of column names to include in the summary output.

```python=
summary(mnist_model, input_data=X_,
        col_names=['input_size', 'output_size', 'num_params'])
```

    ======================================================================
    Layer (type:depth-idx)    Input Shape       Output Shape     Param #
    ======================================================================
    MNISTModel               [256, 1, 28, 28]    [256, 10]        --
    ├─Sequential: 1-1        [256, 1, 28, 28]    [256, 10]        --
    │    └─Sequential: 2-1   [256, 1, 28, 28]    [256, 256]       --
    │    │    └─Flatten: 3-1 [256, 1, 28, 28]    [256, 784]       --
    │    │    └─Linear: 3-2  [256, 784]          [256, 256]      200,960
    │    │    └─ReLU: 3-3    [256, 256]          [256, 256]       --
    │    │    └─Dropout: 3-4 [256, 256]          [256, 256]       --
    │    └─Sequential: 2-2   [256, 256]          [256, 128]       --
    │    │    └─Linear: 3-5  [256, 256]          [256, 128]      32,896
    │    │    └─ReLU: 3-6    [256, 128]          [256, 128]       --
    │    │    └─Dropout: 3-7 [256, 128]          [256, 128]       --
    │    └─Linear: 2-3       [256, 128]          [256, 10]       1,290
    ======================================================================
    Total params: 235,146
    Trainable params: 235,146
    Non-trainable params: 0
    Total mult-adds (Units.MEGABYTES): 60.20
    ======================================================================
    Input size (MB): 0.80
    Forward/backward pass size (MB): 0.81
    Params size (MB): 0.94
    Estimated Total Size (MB): 2.55
    ======================================================================

- We use `SimpleModule.classification()` which uses the cross-entropy loss function instead of mean squared error.
    - We must supply the number of classes.
- `SimpleModule.classification()` includes an accuracy metric by default. Other classification metrics can be added from `torchmetrics`.

```python=
mnist_module = SimpleModule.classification(mnist_model, num_classes=10)
mnist_logger = CSVLogger('logs', name='MNIST')
```

- We define a `Trainer` object to fit the model:
    - `deterministic=True` ensures reproducibility.
    - `enable_process_bar=False` suppresses the progress bar.
    - `callbacks=[ErrorTracker()]` collects targets and predictions over each mini-batch during validation or testing, enabling metric computation over the entire validation or test data set.
- To remind, we specified a validation split of 20%, so training is performed on 48,000 of 60,000 images.
- SGD uses batches of 256 observations for gradient computation, resulting in 188 gradient steps per epoch. (256*188 = 48,128)

```python=
mnist_trainer = Trainer(deterministic=True,
                        max_epochs=30,
                        logger=mnist_logger,
                        enable_progress_bar=False,
                        callbacks=[ErrorTracker()])
mnist_trainer.fit(mnist_module,
                  datamodule=mnist_dataModule)
```

    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs
      | Name  | Type             | Params | Mode 
    ---------------------------------------------------
    0 | model | MNISTModel       | 235 K  | train
    1 | loss  | CrossEntropyLoss | 0      | train
    ---------------------------------------------------
    235 K     Trainable params
    0         Non-trainable params
    235 K     Total params
    0.941     Total estimated model params size (MB)
    13        Modules in train mode
    0         Modules in eval mode
    `Trainer.fit` stopped: `max_epochs=30` reached.

- We plot the accuracy of the model on the training and validation sets across epochs.

```python=
mnist_results = pd.read_csv(mnist_logger.experiment.metrics_file_path)
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
mnist_results.plot(x='epoch', y='train_accuracy_epoch', ax=ax, label='Training', marker='o', color='k')
mnist_results.plot(x='epoch', y='valid_accuracy', ax=ax, label='Validation', marker='+', color='r')
ax.set_ylabel('Accuracy');
```

![](Figures\mnist_40_0.png)

- We use the `predict()` method of the trainer to evaluate the model on the test data.
- The model achieves 97% accuracy on the test data.

```python=
mnist_trainer.test(mnist_module, datamodule=mnist_dataModule)
```

<table>
    <tr>
        <th>Test metric</th>
        <th>DataLoader 0</th>
    </tr>
    <tr>
        <td class="teal-text">test_accuracy</td>
        <td class="purple-text">0.966</td>
    </tr>
    <tr>
        <td class="teal-text">test_loss</td>
        <td class="purple-text">0.146</td>
    </tr>
</table>

    [{'test_loss': 0.146, 'test_accuracy': 0.966}]

- We make an experiment with by remove all hidden layers and use only input and output layer and `SimpleModule.classification()` to fit a logistic regression model. For this we define a new class

```python=
class MNIST_MLR(nn.Module):
    def __init__(self):
        super(MNIST_MLR, self).__init__()
        self.linear = nn.Sequential(nn.Flatten(),
                                    nn.Linear(784, 10))
    def forward(self, x):
        return self.linear(x)

mlr_model = MNIST_MLR()
mlr_module = SimpleModule.classification(mlr_model, num_classes=10)
mlr_logger = CSVLogger('logs', name='MNIST_MLR')
```

```python=
mlr_trainer = Trainer(deterministic=True,
                      max_epochs=30,
                      enable_progress_bar=False,
                      callbacks=[ErrorTracker()])
mlr_trainer.fit(mlr_module, datamodule=mnist_dataModule)
```

    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs
      | Name  | Type             | Params | Mode 
    ---------------------------------------------------
    0 | model | MNIST_MLR        | 7.9 K  | train
    1 | loss  | CrossEntropyLoss | 0      | train
    ---------------------------------------------------
    7.9 K     Trainable params
    0         Non-trainable params
    7.9 K     Total params
    0.031     Total estimated model params size (MB)
    5         Modules in train mode
    0         Modules in eval mode
    `Trainer.fit` stopped: `max_epochs=30` reached.

```python=
mlr_trainer.test(mlr_module, datamodule=mnist_dataModule)
```

<table>
    <tr>
        <th>Test metric</th>
        <th>DataLoader 0</th>
    </tr>
    <tr>
        <td class="teal-text">test_accuracy</td>
        <td class="purple-text">0.922</td>
    </tr>
    <tr>
        <td class="teal-text">test_loss</td>
        <td class="purple-text">0.318</td>
    </tr>
</table>

    [{'test_loss': 0.318, 'test_accuracy': 0.922}]
