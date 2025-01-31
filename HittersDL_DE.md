## 1. Packages

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

- Hauptbibliothek und wesentliche Werkzeuge zur Spezifikation sequentiell strukturierter Netzwerke.

```python=
import torch
from torch import nn
from torch.optim import RMSprop
from torch.utils.data import TensorDataset
```

- Werkzeuge von `torchmetrics` zur Berechnung von Metriken zur Leistungsevaluierung.
- Werkzeuge von `torchinfo` zur Zusammenfassung der Schichtinformationen eines Modells.

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

- `SimpleDataModule` und `SimpleModule` sind einfache Versionen von Objekten, die in `pytorch_lightning` verwendet werden.
- `ErrorTracker` sammelt Ziele und Vorhersagen während der Validierung oder des Testens über jede Mini-Batch, um die Berechnung von Metriken über den gesamten Validierungs- oder Testdatensatz zu ermöglichen.

```python=
from ISLP.torch import (SimpleDataModule, SimpleModule,
                        ErrorTracker, rec_num_workers)
```

## 2. Hitters Data

- Major League Baseball Daten aus den Saisons 1986 und 1987.
- 322 Zeilen, 59 mit fehlenden Werten. 263 bleiben übrig.
- Es gibt 20 Variablen.
- Die Zielvariable ist `Salary`.

```python=
Hitters = load_data('Hitters').dropna()
# Number of observations
n = Hitters.shape[0]
```

```python=
# Design matrix without intercept
Design_matrix = MS(Hitters.columns.drop('Salary'), intercept=False)
# Convert to numpy arrays
X = Design_matrix.fit_transform(Hitters).to_numpy()
Y = Hitters['Salary'].to_numpy()
```

- `to_numpy()` konvertiert `pandas` DataFrames oder Series in `numpy` Arrays, da `sklearn` `numpy` Arrays benötigt, um Lasso zu fitten.

```python=
# Split the data into training and test sets
X_train,X_test,Y_train,Y_test = train_test_split(
    X, Y, test_size=1/3, random_state=1)
```

## 3. Linear Regression

- Wir passen zwei lineare Modelle (Least Squares und Lasso) an und vergleichen ihre Leistung mit der eines neuronalen Netzwerks.

- Wir verwenden den mittleren absoluten Fehler (MAE) auf einem Validierungsdatensatz, um die Modelle zu vergleichen.
  
  $$
  {MAE}(y,\hat{y}) = \frac{1}{n} \sum_{i=1}^n |y_i-\hat{y}_i|
  $$

```python=
hit_lm = LinearRegression().fit(X_train, Y_train)
Yhat_test = hit_lm.predict(X_test)
# Mean absolute error (MAE)
print('MAE of Linear:', np.abs(Yhat_test - Y_test).mean())
# MAE of Linear: 259.715
```

## 4. Lasso Regression

- Wir verwenden MAE zur Auswahl und Bewertung eines Modells statt MSE. Da der Solver nur MSE nutzt, erstellen wir ein Cross-Validation-Gitter und führen die Cross-Validation direkt durch.
- Pipeline: Normalisieren mit `StandardScaler`, dann das Modell anpassen.
- `warm_start=True`: Das Modell nutzt die Lösung des vorherigen Fits als Startpunkt für den nächsten Fit. Dies kann die Konvergenz beschleunigen.
- `max_iter=10000`: Maximale Anzahl an Iterationen.

```python=
scaler = StandardScaler(with_mean=True, with_std=True)
lasso = Lasso(warm_start=True, max_iter=30000)
standard_lasso = Pipeline(steps=[('scaler', scaler),
                                 ('lasso', lasso)])
```

- Maximale $\lambda$ (`lam_max`) ermitteln:
  - Skalarprodukt der standardisierten Prädiktoren mit der Antwort berechnen, um ihre Beziehung zu erfassen.
  - Höchsten absoluten Wert des Skalarprodukts bestimmen, um das Merkmal mit der stärksten Beziehung zur Antwort zu identifizieren.
  - Diesen Wert durch die Anzahl der Beobachtungen teilen, um ihn zu normalisieren.
- Gitter von $\lambda$-Werten erstellen:
  - Gitter von 100 $\lambda$-Werten von 0.01 bis 1 wählen.
  - Gitter mit `lam_max` skalieren.

```python=
# Standardize the predictors
X_s = scaler.fit_transform(X_train)
# Number of observations
n = X_s.shape[0]
# Calculate the Mean-Centered Response Vector
Y_centered = Y_train - Y_train.mean()
# Calculate the Dot Product
dot_product = X_s.T.dot(Y_centered)
# Calculate the maximum absolute value of the dot product
max_abs_value = np.fabs(dot_product).max()
# np.fabs() is a special function handling floating points. np.abs() is a general function.
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

```python=
# Set up a 10-fold cross-validation
cv10 = KFold(10, shuffle=True, random_state=1)
grid = GridSearchCV(standard_lasso, param_grid, cv=cv10,
                    scoring='neg_mean_absolute_error')
grid.fit(X_train, Y_train);
```

```python=
# Extract the best estimator
trained_lasso = grid.best_estimator_
# Predict the test set
Yhat_test = trained_lasso.predict(X_test)
# Show MAE
print('MAE of Lasso:', np.abs(Yhat_test - Y_test).mean())
# MAE of Lasso: 235.675
```

## 5. Neural Network: Klassen und Vererbung

- Modellstruktur zur Beschreibung des Netzwerks einrichten.
- Neue Klassen definieren, die spezifisch für das anzupassende Modell sind.
- In `pytorch` wird dies durch Subklassifizierung einer generischen Netzwerkdarstellung erreicht.
- Klasse `HittersModel` definieren, die von `torch.nn.Module` erbt. Diese Basisklasse ist die Elternklasse für alle neuronalen Netzwerkmodule in `pytorch`.
- Die `__init__()`-Methode definieren, um die Netzwerkstruktur zu spezifizieren.
  - `super(HittersModel, self).__init__()`: Ruft den Konstruktor der Elternklasse nn.Module auf, um die Basisklasse zu initialisieren.
  - Zwei Attribute zur Klasse hinzufügen: `flatten` und `sequential`. Diese werden in der `forward()`-Methode verwendet, um die Abbildung zu beschreiben, die dieses Modul implementiert.
  - `self.flatten = nn.Flatten()`: Definiert eine Schicht, die den Eingabetensor flach macht.
  - `self.sequential = nn.Sequential()`: Definiert einen Container für eine Sequenz von Schichten.
    - `nn.Linear(input_size, 50)`: Lineare Schicht, die den Eingabetensor auf einen 50-dimensionalen Ausgabentensor abbildet.
    - `nn.ReLU()`: ReLU-Aktivierungsfunktion.
    - `nn.Dropout(0.4)`: Dropout-Schicht, die zufällig 40% der Eingabeeinheiten auf 0 setzt, um Überanpassung zu verhindern.
    - `nn.Linear(50, 1)`: Weitere lineare Schicht, die den 50-dimensionalen Tensor auf einen 1-dimensionalen Ausgabentensor abbildet.
- Die `forward()`-Methode definieren, um den Vorwärtsdurchlauf des Netzwerks zu spezifizieren.
  - `x = self.flatten(x)`: Flacht den Eingabetensor ab.
  - `x = self.sequential(x)`: Wendet die Sequenz von Schichten auf den Eingabetensor an.
- Objekte von `torch.nn.Module` haben weitere Methoden, die verwendet werden können, da `HittersModel` von `torch.nn.Module` erbt.

```python=
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
hit_model = HittersModel(X.shape[1]) # X.shape[1] = 19
```

- Das Objekt `self.sequential` besteht aus vier Schichten.
- Erste Schicht: Lineare Abbildung von 19 Eingabefeatures auf 50 Dimensionen.
  - Enthält $19 \times 50$ Gewichte + 50 Biases = 1000 Parameter.
- Zweite Schicht: ReLU-Aktivierungsfunktion.
- Dritte Schicht: Dropout-Schicht, die zufällig 40% der ReLU-Ausgabe auf null setzt.
- Vierte Schicht: Lineare Abbildung von der 50-dimensionalen Dropout-Ausgabe auf eine einzelne Ausgabe.
  - Enthält $50 \times 1$ Gewichte + 1 Bias.
- Gesamtanzahl der Parameter: $(19+1) \times 50 + 50 + 1 = 1051$.
- Mit `summary()` von `torchinfo` können wir diese Informationen anzeigen.

```python=
summary(hit_model, 
        input_size=X_train.shape,
        col_names=['input_size',
                   'output_size',
                   'num_params'])
```

    =====================================================================
    Layer (type:depth-idx)     Input Shape     Output Shape    Param #
    =====================================================================
    HittersModel               [175, 19]        [175]           --
    ├─Flatten: 1-1             [175, 19]        [175, 19]       --
    ├─Sequential: 1-2          [175, 19]        [175, 1]        --
    │    └─Linear: 2-1         [175, 19]        [175, 50]      1,000
    │    └─ReLU: 2-2           [175, 50]        [175, 50]       --
    │    └─Dropout: 2-3        [175, 50]        [175, 50]       --
    │    └─Linear: 2-4         [175, 50]        [175, 1]        51
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

- Wir transformieren `X, Y` in `torch`-Tensoren, die grundlegende Datenstruktur in `pytorch`.
  - `torch`-Tensoren sind ähnlich wie `numpy`-Arrays, können aber auf einer GPU verwendet werden.
  - `torch`-Tensoren arbeiten typischerweise mit 32-Bit (single precision) anstelle von 64-Bit (double precision) Gleitkommazahlen.
    - 32-Bit repräsentiert reale Zahlen mit 32 Bits (4 Bytes) Speicher, 64-Bit verwendet 64 Bits (8 Bytes).
    - 32-Bit hat eine Genauigkeit von 7 Dezimalstellen, 64-Bit hat eine Genauigkeit von 15-17 Dezimalstellen.
    - 32-Bit kann Zahlen im Bereich von $\pm 3.4 \times 10^{38}$ darstellen, 64-Bit im Bereich von $\pm 1.8 \times 10^{308}$.
    - 32-Bit ist schneller als 64-Bit, da es weniger Speicher verwendet.
    - 32-Bit wird in einigen maschinellen Lernaufgaben verwendet, während 64-Bit in wissenschaftlichen Berechnungen und Finanzberechnungen verwendet wird.
- Wir konvertieren die Daten zu `np.float32`, bevor wir sie in `torch`-Tensoren umwandeln.

```python=
# Convert training data to PyTorch tensors
X_train_t = torch.tensor(X_train.astype(np.float32))
Y_train_t = torch.tensor(Y_train.astype(np.float32))
hit_trainTDF = TensorDataset(X_train_t, Y_train_t)
# Convert test data to PyTorch tensors
X_test_t = torch.tensor(X_test.astype(np.float32))
Y_test_t = torch.tensor(Y_test.astype(np.float32))
hit_testTDF = TensorDataset(X_test_t, Y_test_t)
```

- `rec_num_workers()` gibt die Anzahl der Arbeiter für das Laden von Daten zurück.
  - Die Anzahl der CPU-Kerne beeinflusst die Anzahl der Arbeiter.
  - Wir verwenden `os.cpu_count()`, um die Anzahl der CPU-Kerne zu ermitteln.
  - Die aktuelle Systemauslastung beeinflusst die Anzahl der Arbeiter.+

```python=
import os
os.cpu_count()
# 4
```

```python=
max_num_workers = rec_num_workers(); max_num_workers
# 4
```

- Allgemeines Trainingssetup in `pytorch_lightning` umfasst Trainingsdaten, Validierungsdaten und Testdaten.
  
  - Jede wird durch verschiedene `DataLoader`-Objekte dargestellt.
    
    - PyTorch `DataLoader` lädt und verarbeitet Daten effizient in Batches. Es mischt, batcht und lädt Daten parallel, was die Handhabung großer Datensätze vereinfacht.
    
    - Beispiel:
      
      ```python=
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
  - Für `hit_trainTDF` mit 175 Proben und einer Batch-Größe von 32: etwa 6 Batches pro Epoche.

- Die Testdaten werden am Ende des Trainings zur Bewertung des Modells verwendet.

- Wir haben bereits die Trainings- und Testdaten. Wir setzen die Testdaten als Validierungsdaten mit dem Argument `validation=hit_test`.
  
  - Wenn `validation` ein Wert zwischen 0 und 1 ist, wird dieser Bruchteil der Trainingsdaten als Validierungsdaten verwendet.
  - Wenn `validation` eine Ganzzahl ist, entspricht dies der Anzahl der Beobachtungen der Trainingsdaten, die als Validierungsdaten verwendet werden.
  - Wenn ein Datensatz verwendet wird, wird dieser an einen DataLoader übergeben.

```python=
hit_dataModule = SimpleDataModule(hit_trainTDF, hit_testTDF, batch_size=32,
                          num_workers=min(4, max_num_workers),
                          validation=hit_testTDF)
```

- PyTorch Lightning Modul zur Steuerung der Trainingsschritte bereitstellen.
- SimpleModule() Methoden zeichnen am Ende jeder Epoche Verlust und Metriken auf.
- Methoden: SimpleModule.[training/test/validation]_step() erledigen diese Aufgaben.
  - Keine Änderungen an diesen Methoden in unseren Beispielen.
- `SimpleModule.regression()` verwendet quadratischen Fehler (squared-error loss).
- Wir haben den mittleren absoluten Fehler (MAE) als zusätzliche Metrik angefordert.

```python=
hit_module = SimpleModule.regression(hit_model, metrics={'mae':MeanAbsoluteError()})
```

- Ergebnisse mit `CSVLogger()` speichern, was die Resultate in einer CSV-Datei im Verzeichnis `logs/hitters` speichert.
- Nach Abschluss des Trainings können wir die Ergebnisse als `pandas` DataFrame laden.
- Weitere Möglichkeiten zur Ergebnisaufzeichnung sind `TensorBoardLogger()` und `WandbLogger()`.

```python=
hit_logger = CSVLogger('logs', name='hitters')
```

- Wir verwenden das `Trainer()`-Objekt von `pytorch_lightning`, um unser Modell zu trainieren und die Ergebnisse zu protokollieren.
- `deterministic=True` sorgt für reproduzierbare Ergebnisse.
- `max_epochs=50` gibt an, dass das Modell für 50 Epochen trainiert wird.
- `log_every_n_steps=5` bedeutet, dass MAE und andere Metriken nach jeder 5. Batch in jeder Epoche berechnet werden.
- Das Argument `callbacks` ermöglicht verschiedene Aufgaben während des Modelltrainings. Der `ErrorTracker()`-Callback berechnet Validierungs- und Testfehler während des Trainings.
- `hit_module` gibt die Netzwerkarchitektur an.
- `datamodule=hit_dataModule` teilt dem Trainer mit, wie Trainings-/Validierungs-/Testprotokolle erstellt werden.

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
        <th>DataLoader 0</th>
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

- Wir verwenden die protokollierte CSV-Datei in `experiment.metrics_file_path` unseres Loggers.
- Jedes Mal, wenn das Modell trainiert wird, gibt der Logger die Ergebnisse in ein neues Unterverzeichnis in `logs/hitters` aus.
- Wir plotten MAE als Funktion der Epoche.

```python=
# Retrieve the logged summaries
hit_results = pd.read_csv(hit_logger.experiment.metrics_file_path)
# Show columns names
hit_results.columns
```

    Index(['epoch', 'step', 'test_loss', 'test_mae', 'train_loss',
           'train_mae_epoch', 'train_mae_step', 'valid_loss', 'valid_mae'],
          dtype='object')

```python=
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
hit_results.plot(x='epoch', y='train_mae_epoch', label='Training',
                     marker='o', color='k', ax=ax)
hit_results.plot(x='epoch', y='valid_mae', label='Validation',
                        marker='x', color='r', ax=ax)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss');
```

![](Figures\hittersDL.png)

```python=
#fig, ax = plt.subplots(1, 1, figsize=(6, 6))
#for column, color, label in zip(['train_mae_epoch', 'valid_mae'],
#                                ['black', 'red'],
#                                ['Training', 'Validation']):
#    hit_results.plot(x='epoch', y=column, marker='o', color=color, label=label, ax=ax)
```

- Vorhersage mit dem finalen Modell und Auswertung der Testdaten.
- Rufe `eval()` auf `hit_model` vor dem Fitten auf.
- `eval()` informiert PyTorch, das Modell als angepasst zu behandeln.
- Dropout-Schichten sind während der Vorhersage deaktiviert (keine zufälligen Gewichtsverluste).

```python=
hit_model.eval()
preds = hit_model(X_test_t)
torch.abs(Y_test_t - preds).mean()
# tensor(229.5011, grad_fn=<MeanBackward0>)
```

- SGD minimiert die Verlustfunktion durch iterative Aktualisierung der Modellparameter.
- Die Verlustfunktion misst, wie gut die Modellvorhersagen den tatsächlichen Werten entsprechen.
  - Regressionsverluste: MSE, MAE, Huber Loss, Log-Cosh Loss.
  - Klassifikationsverluste: Cross-Entropy, Hinge Loss, Squared Hinge Loss, Kullback-Leibler Divergenz.
    - Cross-Entropy: Verwendet bei Softmax-Ausgaben, misst den Unterschied zwischen vorhergesagter und tatsächlicher Wahrscheinlichkeitsverteilung.
  - Der Gradient ist ein Vektor partieller Ableitungen der Verlustfunktion bezüglich der Modellparameter.
  - Gradient Descent minimiert die Verlustfunktion durch Aktualisierung in Richtung des negativen Gradienten.
- Bei jedem Schritt berechnet SGD den Gradienten mit einem Teil der Trainingsdaten (Batch).
- Mit n = 175 und einer Batch-Größe von 32 werden nach 175/32 ≈ 5.5 Schritten alle Daten einmal verwendet (eine Epoche).
- Ein Teilbatch tritt auf, wenn die Gesamtzahl der Beobachtungen nicht perfekt durch die Batch-Größe teilbar ist.
- Eine Batch-Größe von 32 ist ein häufig verwendeter Standard, der Effizienz, Speicherverbrauch und Konvergenzverhalten ausbalanciert.