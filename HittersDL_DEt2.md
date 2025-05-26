---
title: Hitters DL
---

## 1. Packages und Data

```python=
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from ISLP import load_data
from ISLP.models import ModelSpec as MS
Hitters = load_data('Hitters').dropna()
```

- Major League Baseball Daten aus den Saisons 1986 und 1987.
- 322 Zeilen, 59 mit fehlenden Werten. 263 bleiben übrig.
- 19 Kovariate (Prädiktor) und Zielvariable *Salary*

```python=
# Design matrix without intercept
Design_matrix = MS(Hitters.columns.drop('Salary'), intercept=False)
# Convert to numpy arrays
## sklearn benötigt numpy Arrays, um Lasso zu fitten
X = Design_matrix.fit_transform(Hitters).to_numpy()
Y = Hitters['Salary'].to_numpy()
# Split training and test data
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=1/3, random_state=1)
```

## 2. Linear Regression

- Wir verwenden den mittleren absoluten Fehler (MAE), um die Modelle zu vergleichen ${MAE}(y,\hat{y}) = \frac{1}{n} \sum_{i=1}^n |y_i-\hat{y}_i|$

```python=
from sklearn.linear_model import LinearRegression
hit_lm = LinearRegression().fit(X_train, Y_train)
Yhat_test = hit_lm.predict(X_test)
np.abs(Y_test - Yhat_test).mean()
# 259.715 (MAE)
```

## 3. Lasso Regression

- Maximale $\lambda$ (`lam_max`) ermitteln:
  - Skalarprodukt der standardisierten Prädiktoren mit der Antwort berechnen, um ihre Beziehung zu erfassen.
  - Höchsten absoluten Wert des Skalarprodukts bestimmen, um das Merkmal mit der stärksten Beziehung zur Antwort zu identifizieren.
  - Diesen Wert durch die Anzahl der Beobachtungen teilen, um ihn zu normalisieren.
- Gitter von $\lambda$-Werten erstellen:
  - Gitter von 100 $\lambda$-Werten von 0.01 bis 1 wählen.
  - Gitter mit `lam_max` skalieren.

```python=
# Standardize the designed matrix
X_s = scaler.fit_transform(X_train)
# Number of observations
n = X_s.shape[0]
# Calculate the Mean-Centered Y
Y_centered = Y_train - Y_train.mean()
# Calculate the Dot Product
dot_product = X_s.T.dot(Y_centered)
# Calculate the maximum absolute value of the dot product
max_abs_value = np.fabs(dot_product).max()
# np.fabs() is a special function handling floating points.
# np.abs() is a general function.
# Compute lam_max
lam_max = max_abs_value / n # 255.658
# Generate a range of values
log_alpha_values = np.linspace(0, np.log(0.01), 100) # 0 to -4.605
# Exponentiate the values
alpha_values = np.exp(log_alpha_values) # 1 to 0.01
# Scale by lam_max
scaled_alpha_values = alpha_values * lam_max
# Create a param_grid
param_grid = {'lasso__alpha': scaled_alpha_values}
```

- `alpha_values` mit exponentieller Abstandsverteilung: Werte beginnen bei 1 und verringern sich logarithmisch bis 0,01.
- `alpha_values = np.linspace(0.01, 1, 100)` mit linearer Abstandsverteilung: Werte sind gleichmäßig zwischen 0,01 und 1 verteilt.
- Da der Solver nur MSE nutzt, erstellen wir ein Cross-Validation-Gitter und führen die Cross-Validation direkt durch für MAE. 
- `warm_start=True`: Das Modell nutzt die Lösung des vorherigen Fits als Startpunkt für den nächsten Fit. Dies kann die Konvergenz beschleunigen.
- `max_iter=10000`: Maximale Anzahl an Iterationen.

```python=
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
scaler = StandardScaler(with_mean=True, with_std=True)
lasso = Lasso(warm_start=True, max_iter=30000)
standard_lasso = Pipeline(steps=[('scaler', scaler),
                                 ('lasso', lasso)])
cv10 = KFold(10, shuffle=True, random_state=1)
grid = GridSearchCV(standard_lasso, param_grid, cv=cv10,
                    scoring='neg_mean_absolute_error')
grid.fit(X_train, Y_train)
trained_lasso = grid.best_estimator_
Yhat_test = trained_lasso.predict(X_test)
np.abs(Yhat_test - Y_test).mean()
# 235.675 (MAE of Lasso)
```

## 4. Neural Network: Klassen und Vererbung

- Neue Klassen definieren, die spezifisch für das anzupassende Modell sind.
  - Klasse HittersModel definieren, die von torch.nn.Module erbt. 
  - Diese Basisklasse torch.nn.Module ist die Elternklasse für alle neuronalen Netzwerkmodule in pytorch.
- \_\_init\_\_(): spezifizieren die Netzwerkstruktur.
  - super(HittersModel, self).\_\_init\_\_(): Ruft den Konstruktor der Elternklasse nn.Module auf, um die Basisklasse zu initialisieren.
  - Zwei Attribute hinzufügen: flatten und sequential. Diese werden in der forward()-Methode verwendet, um die Abbildung zu beschreiben, die dieses Modul implementiert.
    - self.flatten = nn.Flatten(): Definiert eine Schicht, die den Eingabetensor flach macht.
    - self.sequential = nn.Sequential(): Definiert einen Container für eine Sequenz von Schichten.
- forward(): spezifizieren den Vorwärtsdurchlauf des Netzwerks.
  - x = self.flatten(x): Flacht den Eingabetensor ab.
  - x = self.sequential(x): Wendet die Sequenz von Schichten auf den Eingabetensor an.
- Objekte von torch.nn.Module haben weitere Methoden.

```python=
import torch
from torch import nn
class HittersModel(nn.Module):

    def __init__(self, input_size):
        super(HittersModel, self).__init__()
        self.flatten = nn.Flatten()
        self.sequential = nn.Sequential(
            nn.Linear(input_size, 50),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(50, 1))

    def forward(self, x):
        x = self.flatten(x)
        return torch.flatten(self.sequential(x))
```

```python=
hit_model = HittersModel(19)
```

- Schicht 1: Linear, 19 Eingaben ? 50 Ausgaben, $(19+1) \times 50 = 1000$ Parameters
- Schicht 2: ReLU-Aktivierung
- Schicht 3: Dropout, 40% der Werte werden auf null gesetzt
- Schicht 4: Linear, 50 ? 1, $50 + 1 = 51$ Parameter
- Gesamt: 1051 Parameter
- Mit torchinfo.summary() lässt sich das Modell übersichtlich anzeigen

```python=
from torchinfo import summary
summary(hit_model,input_size=X_train.shape,
        col_names=['input_size','output_size','num_params'])
```

    =====================================================================
    Layer (type:depth-idx)     Input Shape     Output Shape    Param #
    =====================================================================
    HittersModel               [175, 19]        [175]           --
    +-Flatten: 1-1             [175, 19]        [175, 19]       --
    +-Sequential: 1-2          [175, 19]        [175, 1]        --
    ¦    +-Linear: 2-1         [175, 19]        [175, 50]      1,000
    ¦    +-ReLU: 2-2           [175, 50]        [175, 50]       --
    ¦    +-Dropout: 2-3        [175, 50]        [175, 50]       --
    ¦    +-Linear: 2-4         [175, 50]        [175, 1]        51
    =====================================================================
    Total params: 1,051
    Trainable params: 1,051
    Non-trainable params: 0
    Total mult-adds (Units.MEGABYTES): 0.18
    =====================================================================
    Input size (MB): 0.01
    Forward/backward pass size (MB): 0.07
    Params size (MB): 0.00
    Estimated Total Size (MB): 0.09
    =====================================================================

- Wir transformieren $X, Y$ in torch-Tensoren, die grundlegende Datenstruktur in pytorch.
  - torch-Tensoren ähneln numpy-Arrays, sind aber GPU-fähig und meist 32-Bit (statt 64-Bit) Gleitkommazahlen.
    - 32-Bit (4 Byte) bietet ~7 Dezimalstellen und Werte bis $\pm 3.4 \times 10^{38}$, 64-Bit (8 Byte) bietet 15–17 Stellen und bis $\pm 1.8 \times 10^{308}$.
    - 32-Bit ist schneller und braucht weniger Speicher, wird oft für ML genutzt; 64-Bit für Wissenschaft und Finanzen.
- Für die Umwandlung werden die Daten zuerst zu np.float32 konvertiert.

```python=
from torch.utils.data import TensorDataset
# Convert training data to PyTorch tensors
X_train_t = torch.tensor(X_train.astype(np.float32))
Y_train_t = torch.tensor(Y_train.astype(np.float32))
hit_trainTDF = TensorDataset(X_train_t, Y_train_t)
# Convert test data to PyTorch tensors
X_test_t = torch.tensor(X_test.astype(np.float32))
Y_test_t = torch.tensor(Y_test.astype(np.float32))
hit_testTDF = TensorDataset(X_test_t, Y_test_t)
```

- SimpleDataModule ist eine vereinfachte Version der in pytorch_lightning genutzten Module.
- Wir nutzen vorhandene Trainings- und Testdaten und setzen die Testdaten per validation=hit_test als Validierungsdaten.
  - Ist validation zwischen 0 und 1, wird dieser Anteil der Trainingsdaten zur Validierung genutzt.
  - Ist validation eine Ganzzahl, entspricht das der Anzahl Validierungsbeispiele aus den Trainingsdaten.
  - Ein übergebenes Dataset wird direkt an einen DataLoader weitergereicht.

```python=
from ISLP.torch import SimpleDataModule
hit_dataModule = SimpleDataModule(hit_trainTDF, hit_testTDF, batch_size=32,
                                  num_workers=4,validation=hit_testTDF)
```

- SimpleModule ist eine vereinfachte Version der in pytorch_lightning genutzten Module.
- SimpleModule zeichnen am Ende jeder Epoche Verlust und Metriken auf.
- SimpleModule.regression() verwendet MSE. Wir fordern MAE an.

```python=
from ISLP.torch import SimpleModule
hit_module = SimpleModule.regression(hit_model, metrics={'mae':MeanAbsoluteError()})
```

- CSVLogger() speichern Ergebnisse.

```python=
from pytorch_lightning.loggers import CSVLogger
hit_logger = CSVLogger('logs', name='hitters')
```

- deterministic=True sorgt für reproduzierbare Ergebnisse.
- max_epochs=50 trainiert das Modell für 50 Epochen.
- log_every_n_steps=5 berechnet MAE und andere Metriken alle 5 Batches pro Epoche.
- Mit callbacks lassen sich Zusatzfunktionen integrieren; z.B. berechnet der ErrorTracker()-Callback Validierungs- und Testfehler während des Trainings.

```python=
hit_trainer = Trainer(deterministic=True,
                      max_epochs=50,
                      log_every_n_steps=5,
                      logger=hit_logger,
                      enable_progress_bar=False,
                      callbacks=[ErrorTracker()])
hit_trainer.fit(hit_module, datamodule=hit_dataModule)
```

    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs
    
      | Name  | Type         | Params | Mode 
    -----------------------------------------------
    0 | model | HittersModel | 1.1 K  | train
    1 | loss  | MSELoss      | 0      | train
    -----------------------------------------------
    1.1 K     Trainable params
    0         Non-trainable params
    1.1 K     Total params
    0.004     Total estimated model params size (MB)
    8         Modules in train mode
    0         Modules in eval mode
    `Trainer.fit` stopped: `max_epochs=50` reached.

```python=
# Evaluate the model
hit_trainer.test(hit_module, datamodule=hit_dataModule);
```

<table>
    <tr>
        <th>Test metric</th>
        <th>DataLoader</th>
    </tr>
    <tr>
        <td class="teal-text">test_loss</td>
        <td class="purple-text">104098.546</td>
    </tr>
    <tr>
        <td class="teal-text">test_mae</td>
        <td class="purple-text">229.501</td>
    </tr>
</table>

- experiment.metrics_file_path gibt den Pfad zur protokollierten CSV-Datei an.
- Bei jedem Training speichert der Logger die Ergebnisse in einem neuen Unterordner unter logs/hitters.


```python=
hit_results = pd.read_csv(hit_logger.experiment.metrics_file_path)
hit_results.columns
# epoch, step, test_loss, test_mae, train_loss,train_mae_epoch, train_mae_step, valid_loss, valid_mae
```

- Wir plotten MAE als Funktion der Epoche.

```python=
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
hit_results.plot(x='epoch', y='train_mae_epoch', label='Training',
                     marker='o', color='k', ax=ax)
hit_results.plot(x='epoch', y='valid_mae', label='Validation',
                        marker='x', color='r', ax=ax)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss');
```

<img title="" src="Figures\hittersDL.png" alt="" width="384">

- Vorhersage mit dem finalen Modell und Auswertung auf den Testdaten.  
- Mit `eval()` wird das Modell in den Evaluationsmodus versetzt.  
  - Dropout ist dabei deaktiviert.

```python=
hit_model.eval()
preds = hit_model(X_test_t)
torch.abs(Y_test_t - preds).mean()
# MAE: 229.501
```
---

- In pytorch_lightning werden Trainings-, Validierungs- und Testdaten jeweils durch eigene DataLoader bereitgestellt.
- PyTorch Lightning Modul zur Steuerung der Trainingsschritte bereitstellen.
- Ein PyTorch DataLoader lädt Daten effizient in Batches, mischt sie und unterstützt paralleles Laden – ideal für große Datensätze.
  - Beispiel:
      
      ```python=
      from torch.utils.data import DataLoader
      dataset = list(range(20))
      dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
      for batch in dataloader:
              print(batch)
      ```
      
          tensor([ 0, 11, 10,  7, 12])
          tensor([ 8, 18, 14,  1, 16])
          tensor([ 4, 19, 17,  3,  5])
          tensor([ 6,  2, 13,  9, 15])

- Während jeder Epoche führen wir einen Trainingsschritt zur Modellanpassung und einen Validierungsschritt zur Fehlerüberwachung durch. 
  - Eine Epoche ist ein vollständiger Durchlauf des gesamten Trainingsdatensatzes.
  - Für hit_trainTDF mit 175 Proben und einer Batch-Größe von 32: etwa 6 Batches pro Epoche.