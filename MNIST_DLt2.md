---
title: mnist
---

```python=
import numpy as np, pandas as pd, matplotlib.pyplot as plt
```

- MNIST:
  - Modified National Institute of Standards and Technology.
  - 60.000 Trainingsbilder und 10.000 Testbilder.
  - Jedes Bild ist 28x28 = 784 Pixel.
  - Jeder Pixel ist eine Ganzzahl zwischen 0 und 255.
  - Jedes Bild ist mit der Ziffer von 0 bis 9 beschriftet, die es darstellt.
  - Bilder sind im flachen Format gespeichert, mit 784 Pixeln pro Bild.

```python=
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
mnist_train = MNIST(root='data', train=True, download=True, transform=ToTensor())
mnist_test = MNIST(root='data', train=False, download=True, transform=ToTensor())
```

- Neuronale Netze benötigen skalierte Eingabedaten, daher teilen wir Pixelwerte durch 255, um sie auf [0, 1] zu bringen.
- ToTensor() übernimmt diese Skalierung und wandelt Bilder in PyTorch-Tensoren [0, 1] um.
- Dabei werden die Dimensionen von H x W x C zu C x H x W umgewandelt.
  - Alle Bilder haben Kanal C = 1.

```python=
len(mnist_train[0]) # 2
mnist_train[0][1] # 5
mnist_train[0][0].shape # torch.Size([1, 28, 28])
mnist_train.targets # 60000 response values
```

- Wir erstellen ein Datenmodul aus den Trainings- und Testdatensätzen und reservieren 20% der Trainingsbilder für die Validierung.
- Mit 20% Validierungsdaten werden 48.000 von 60.000 Bildern für das Training verwendet.

```python=
from ISLP.torch import SimpleDataModule
mnist_dataModule = SimpleDataModule(mnist_train, mnist_test,validation=0.2,
                                    num_workers=4,batch_size=256)
```

- Untersuche die ersten zwei Batches des Trainingsdatensatzes:
  - Jeder Batch enthält 265 Bilder mit den Maßen 1 x 28 x 28 (Kanal x Höhe x Breite).


```python=
for idx, (X_ ,Y_) in enumerate(mnist_dataModule.train_dataloader()):
    print(f'X_{idx}: ', X_.shape)
    print(f'Y_{idx}: ', Y_.shape)
    if idx == 1:
        break
```

    X_0:  torch.Size([256, 1, 28, 28])
    Y_0:  torch.Size([256])
    X_1:  torch.Size([256, 1, 28, 28])
    Y_1:  torch.Size([256])

- Definiere die Klasse MINISTModel, die von `nn.Module` erbt.

```python=
from torch import nn
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
###
mnist_model = MNISTModel()
```

- $X\_$ enthält 256 Beobachtungen der Form [256, 1, 28, 28]. Mit der Methode `size()` lässt sich die Form des Outputs anzeigen.

```python=
mnist_model(X_).size()
# torch.Size([256, 10])
```

```python=
summary(mnist_model,input_data=X_,col_names=['input_size','output_size','num_params'])
```
    ======================================================================
    Layer (type:depth-idx)    Input Shape       Output Shape     Param #
    ======================================================================
    MNISTModel               [256, 1, 28, 28]    [256, 10]        --
    +-Sequential: 1-1        [256, 1, 28, 28]    [256, 10]        --
    ¦    +-Sequential: 2-1   [256, 1, 28, 28]    [256, 256]       --
    ¦    ¦    +-Flatten: 3-1 [256, 1, 28, 28]    [256, 784]       --
    ¦    ¦    +-Linear: 3-2  [256, 784]          [256, 256]      200,960
    ¦    ¦    +-ReLU: 3-3    [256, 256]          [256, 256]       --
    ¦    ¦    +-Dropout: 3-4 [256, 256]          [256, 256]       --
    ¦    +-Sequential: 2-2   [256, 256]          [256, 128]       --
    ¦    ¦    +-Linear: 3-5  [256, 256]          [256, 128]      32,896
    ¦    ¦    +-ReLU: 3-6    [256, 128]          [256, 128]       --
    ¦    ¦    +-Dropout: 3-7 [256, 128]          [256, 128]       --
    ¦    +-Linear: 2-3       [256, 128]          [256, 10]       1,290
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

- Wir verwenden SimpleModule.classification(), das die Cross-Entropy-Verlustfunktion anstelle von MSE verwendet.
  - Wir müssen die Anzahl der Klassen angeben.

```python=
from ISLP.torch import SimpleModule
mnist_module = SimpleModule.classification(mnist_model, num_classes=10)
###
from pytorch_lightning.loggers import CSVLogger
mnist_logger = CSVLogger('logs', name='MNIST')
```

- Das Trainer-Objekt trainiert das Modell:
  - deterministic=True: Reproduzierbarkeit.
  - enable_process_bar=False: unterdrückt die Fortschrittsanzeige.
  - callbacks=[ErrorTracker()]: sammelt Werte für die Metrik-Berechnung.
- Zur Erinnerung: 48.000 von 60.000 Bildern werden mit 20% Validierung fürs Training genutzt.
- SGD nutzt Batches von 256 für Gradientenberechnung, ergibt 188 Schritte pro Epoche (256×188=48.128).

```python=
from pytorch_lightning import Trainer
from ISLP.torch import ErrorTracker
mnist_trainer = Trainer(deterministic=True,
                        max_epochs=30,
                        logger=mnist_logger,
                        enable_progress_bar=False,
                        callbacks=[ErrorTracker()])
mnist_trainer.fit(mnist_module,datamodule=mnist_dataModule)
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

- Wir plotten die Modellgenauigkeit auf Trainings- und Validierungsdaten über die Epochen.

```python=
mnist_results = pd.read_csv(mnist_logger.experiment.metrics_file_path)
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
mnist_results.plot(x='epoch', y='train_accuracy_epoch', ax=ax, label='Training', marker='o', color='k')
mnist_results.plot(x='epoch', y='valid_accuracy', ax=ax, label='Validation', marker='+', color='r')
ax.set_ylabel('Accuracy');
```

<img title="" src="Figures\mnist_40_0.png" alt="" width="495">

- Mit `predict()` evaluieren wir das Modell auf den Testdaten.
- Die Testgenauigkeit beträgt 97%.
- test_loss: Durchschnittlicher Verlust der Verlustfunktion über alle Testdaten, misst die Abweichung der Vorhersagen von den echten Labels.
- test_accuracy: Anteil korrekt klassifizierter Testbilder (Wert zwischen 0 und 1).

```python=
mnist_trainer.test(mnist_module, datamodule=mnist_dataModule)
```

<table>
    <tr>
        <th>Test metric</th>
        <th>DataLoader</th>
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

    [{'test_loss': 0.153, 'test_accuracy': 0.966}]

- Experiment: Wir entfernen alle versteckten Schichten und verwenden nur Eingabe-, Ausgabeschicht und SimpleModule.classification().

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
        <th>DataLoader</th>
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

- Die Testgenauigkeit liegt bei 92% und ist damit geringer als beim neuronalen Netz.