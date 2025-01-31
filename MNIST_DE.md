# 1. Packages

```python=
import numpy as np, pandas as pd, matplotlib.pyplot as plt
```

**Torch-Specific Imports**
- Hauptbibliothek und wesentliche Werkzeuge zur Spezifikation sequentiell strukturierter Netzwerke.

```python=
import torch
from torch import nn
from torch.optim import RMSprop
from torch.utils.data import TensorDataset
```

- Werkzeuge von torchmetrics zur Berechnung von Metriken zur Leistungsevaluierung.
- Werkzeuge von torchinfo zur Zusammenfassung der Schichtinformationen eines Modells.

```python=
from torchmetrics import (MeanAbsoluteError, R2Score)
from torchinfo import summary
```

- `pytorch_lightning` vereinfacht die Spezifikation, Anpassung und Bewertung von Modellen, indem es Boilerplate-Code reduziert.


```python=
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
```

- `seed_everything()` setzt den Seed.
- `use_deterministic_algorithms` fixiert Algorithmen.

```python=
from pytorch_lightning import seed_everything
seed_everything(0, workers=True)
torch.use_deterministic_algorithms(True, warn_only=True)
# Seed set to 0
```

- Datensatz aus `torchvision`.
- `torchvision.transforms` zur Vorverarbeitung.

```python=
from torchvision.io import read_image
from torchvision.datasets import MNIST
from torchvision.transforms import (Resize, Normalize,
                                    CenterCrop, ToTensor)
```

- `SimpleDataModule` und `SimpleModule` sind einfache Versionen von Objekten, die in `pytorch_lightning` verwendet werden.
- `ErrorTracker` sammelt Ziele und Vorhersagen während der Validierung oder des Testens über jede Mini-Batch, um die Berechnung von Metriken über den gesamten Validierungs- oder Testdatensatz zu ermöglichen.

```python=
from ISLP.torch import (SimpleDataModule, SimpleModule,
                        ErrorTracker, rec_num_workers)
```

# 2. MNIST
- Geändertes Nationales Institut für Standards und Technologie.
- Sammlung handgeschriebener Ziffern.
- Enthält 60.000 Trainingsbilder und 10.000 Testbilder.
- Jedes Bild ist 28x28 = 784 Pixel.
- Jeder Pixel ist eine Ganzzahl zwischen 0 und 255.
- Jedes Bild ist mit der Ziffer von 0 bis 9 beschriftet, die es darstellt.
- Bilder sind im flachen Format gespeichert, mit 784 Pixeln pro Bild.

```python=
mnist_train = MNIST(root='data', train=True, download=True, transform=ToTensor())
mnist_test = MNIST(root='data', train=False, download=True, transform=ToTensor())
# mnist_train, mnist_test = [MNIST(root='data', train=train, download=True, transform=ToTensor()) for train in [True, False]]
```

- Neuronale Netze sind empfindlich gegenüber der Skalierung der Eingabedaten (wie Ridge, Lasso), daher skalieren wir Pixelwerte auf [0, 1], indem wir durch 255 teilen.
- `ToTensor()` aus `torchvision.transforms` skaliert automatisch und konvertiert Bilder in PyTorch-Tensoren mit Werten [0, 1].
- `ToTensor()` ordnet auch die Bilddimensionen von H x W x C zu C x H x W um.

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

- Wir erstellen ein Datenmodul aus den Trainings- und Testdatensätzen und reservieren 20% der Trainingsbilder für die Validierung.

```python=
max_num_workers = rec_num_workers()
mnist_dataModule = SimpleDataModule(mnist_train, mnist_test,
                            validation=0.2,
                            num_workers=max_num_workers,
                            batch_size=256)
```

- Untersuche die Daten im Netzwerk durch Schleifen durch die ersten zwei Batches des Trainingsdatensatzes:
  - Jeder Batch enthält 265 Bilder.
  - Jedes Bild hat die Dimensionen 1 x 28 x 28 (1 Kanal, 28 Reihen, 28 Spalten).

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

- Definiere die Klasse MINISTModel, die von `nn.Module` erbt.
- Schicht 1: `nn.Sequential` ist ein Container für eine Sequenz von Schichten.
    - `nn.Flatten()` flacht jedes 1x28x28 Bild zu einem 1x784 Tensor ab.
    - `nn.Linear(28*28, 256)` ist eine vollverbundene Schicht mit 784 Eingangs- und 256 Ausgangsmerkmalen.
    - `nn.ReLU()` ist eine Aktivierungsfunktion.
    - `nn.Dropout(0.4)` setzt zufällig 40% der Eingabeeinheiten bei jedem Update während des Trainings auf 0.
- Schicht 2: `nn.Sequential` ist ein Container für eine Sequenz von Schichten.
    - `nn.Linear(256, 128)` ist eine Schicht mit 256 Eingangs- und 128 Ausgangsmerkmalen.
    - `nn.ReLU()` und `nn.Dropout(0.3)` haben denselben Zweck wie in der ersten Schicht.
- Forward-Methode: kombiniert die beiden Schichten mit einer abschließenden vollverbundenen Schicht mit 128 Eingangs- und 10 Ausgangsmerkmalen.
- Die Forward-Methode des Netzwerks nimmt einen Eingabetensor `x` und leitet ihn durch die definierte Schichtsequenz in `self_forward`.

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

- Wir überprüfen, ob das Modell die erwartete Größe ausgibt, indem wir das vorhandene Batch `X_` verwenden.

```python=
mnist_model(X_).size()
```

    torch.Size([256, 10])

- Wir übergeben einen Tensor der richtigen Form. In diesem Fall entspricht `X_` der erwarteten Eingabeform [batch_size, channel, height, width] = [256, 1, 28, 28].
- Wir geben die Liste der Spaltennamen an, die in der Zusammenfassungsausgabe enthalten sein sollen.

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

- Wir verwenden `SimpleModule.classification()`, das die Cross-Entropy-Verlustfunktion anstelle von Mean Squared Error (MSE) verwendet.
    - Wir müssen die Anzahl der Klassen angeben.
- `SimpleModule.classification()` enthält standardmäßig eine Genauigkeitsmetrik. Weitere Klassifikationsmetriken können aus `torchmetrics` hinzugefügt werden.

```python=
mnist_module = SimpleModule.classification(mnist_model, num_classes=10)
mnist_logger = CSVLogger('logs', name='MNIST')
```

- Wir definieren ein `Trainer`-Objekt, um das Modell zu trainieren:
    - `deterministic=True` sorgt für Reproduzierbarkeit.
    - `enable_process_bar=False` unterdrückt die Fortschrittsanzeige.
    - `callbacks=[ErrorTracker()]` sammelt Ziele und Vorhersagen während der Validierung oder des Tests, um Metriken über den gesamten Validierungs- oder Testdatensatz zu berechnen.
- Zur Erinnerung: Wir haben eine Validierungsaufteilung von 20% festgelegt, daher wird das Training auf 48.000 von 60.000 Bildern durchgeführt.
- SGD verwendet Batches von 256 Beobachtungen für die Gradientenberechnung, was zu 188 Gradienten-Schritten pro Epoche führt. (256*188 = 48.128)

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

- Wir plotten die Genauigkeit des Modells auf den Trainings- und Validierungsdatensätzen über die Epochen.

```python=
mnist_results = pd.read_csv(mnist_logger.experiment.metrics_file_path)
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
mnist_results.plot(x='epoch', y='train_accuracy_epoch', ax=ax, label='Training', marker='o', color='k')
mnist_results.plot(x='epoch', y='valid_accuracy', ax=ax, label='Validation', marker='+', color='r')
ax.set_ylabel('Accuracy');
```

![](Figures\mnist_40_0.png)

- Wir verwenden die `predict()`-Methode des Trainers, um das Modell mit den Testdaten zu evaluieren.
- Das Modell erreicht eine Genauigkeit von 97% auf den Testdaten.

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

- Wir führen ein Experiment durch, indem wir alle versteckten Schichten entfernen und nur die Eingabe- und Ausgabeschicht sowie `SimpleModule.classification()` verwenden, um ein logistisches Regressionsmodell anzupassen. Dafür definieren wir eine neue Klasse.

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
