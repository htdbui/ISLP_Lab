---
title: "Caravan"
author: "db"
---

# 1. Datenbeschreibung

- **Dataset**
  - 85 Prädiktoren zu Demografie von 5.822 Personen.
  - Response-Variable: `Purchase` (Kauf von Caravan-Versicherung, 6% taten es).
- **Variablengruppen**
  - **Soziodemografische Daten (Variablen 1-43)**
    - Basierend auf Postleitzahlen, gleiche Attribute für Personen in derselben Gegend.
  - **Produktbesitz (Variablen 44-86)**
    - Variable 86 (`Purchase`) zeigt, ob Caravan-Versicherung gekauft wurde.

# 2. Packages und Daten

```python=
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from ISLP import load_data, confusion_table
from ISLP.models import (ModelSpec as MS, summarize)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
```

```python=
Caravan = load_data('Caravan')
Caravan.head()
```

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MOSTYPE</th>
      <th>MAANTHUI</th>
      <th>MGEMOMV</th>
      <th>MGEMLEEF</th>
      <th>MOSHOOFD</th>
      <th>MGODRK</th>
      <th>MGODPR</th>
      <th>MGODOV</th>
      <th>MGODGE</th>
      <th>MRELGE</th>
      <th>...</th>
      <th>APERSONG</th>
      <th>AGEZONG</th>
      <th>AWAOREG</th>
      <th>ABRAND</th>
      <th>AZEILPL</th>
      <th>APLEZIER</th>
      <th>AFIETS</th>
      <th>AINBOED</th>
      <th>ABYSTAND</th>
      <th>Purchase</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>8</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>3</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>8</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>8</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>10</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 86 columns</p>

```python=
Caravan.describe().round(1)
```

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MOSTYPE</th>
      <th>MAANTHUI</th>
      <th>MGEMOMV</th>
      <th>MGEMLEEF</th>
      <th>MOSHOOFD</th>
      <th>MGODRK</th>
      <th>MGODPR</th>
      <th>MGODOV</th>
      <th>MGODGE</th>
      <th>MRELGE</th>
      <th>...</th>
      <th>ALEVEN</th>
      <th>APERSONG</th>
      <th>AGEZONG</th>
      <th>AWAOREG</th>
      <th>ABRAND</th>
      <th>AZEILPL</th>
      <th>APLEZIER</th>
      <th>AFIETS</th>
      <th>AINBOED</th>
      <th>ABYSTAND</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5822.0</td>
      <td>5822.0</td>
      <td>5822.0</td>
      <td>5822.0</td>
      <td>5822.0</td>
      <td>5822.0</td>
      <td>5822.0</td>
      <td>5822.0</td>
      <td>5822.0</td>
      <td>5822.0</td>
      <td>...</td>
      <td>5822.0</td>
      <td>5822.0</td>
      <td>5822.0</td>
      <td>5822.0</td>
      <td>5822.0</td>
      <td>5822.0</td>
      <td>5822.0</td>
      <td>5822.0</td>
      <td>5822.0</td>
      <td>5822.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>24.3</td>
      <td>1.1</td>
      <td>2.7</td>
      <td>3.0</td>
      <td>5.8</td>
      <td>0.7</td>
      <td>4.6</td>
      <td>1.1</td>
      <td>3.3</td>
      <td>6.2</td>
      <td>...</td>
      <td>0.1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.6</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>12.8</td>
      <td>0.4</td>
      <td>0.8</td>
      <td>0.8</td>
      <td>2.9</td>
      <td>1.0</td>
      <td>1.7</td>
      <td>1.0</td>
      <td>1.6</td>
      <td>1.9</td>
      <td>...</td>
      <td>0.4</td>
      <td>0.1</td>
      <td>0.1</td>
      <td>0.1</td>
      <td>0.6</td>
      <td>0.0</td>
      <td>0.1</td>
      <td>0.2</td>
      <td>0.1</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>10.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>30.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>35.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>8.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>7.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>41.0</td>
      <td>10.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>5.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>...</td>
      <td>8.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 85 columns</p>

```python=
Purchase = Caravan.Purchase
Purchase.value_counts(normalize=True)
```

    Purchase
    No     0.940227
    Yes    0.059773
    Name: proportion, dtype: float64

# 3. K-Nearest Neighbors

- **Features**
  - Alle Spalten außer `Purchase`.

```python=
feature_df = Caravan.drop(columns=['Purchase'])
```

- **KNN-Performance**
  - Beeinflusst durch Variablenskalierungen, denn vorhersagen basieren auf den nächsten Beobachtungen.
- **Problem**
  - Großskalige Variablen dominieren Distanzberechnungen.
  - Beispiel: 1.000 USD Gehaltsunterschied > 50 Jahre Altersunterschied.
- **Lösung**
  - Daten standardisieren.
  - Mittelwert = 0, Standardabweichung = 1.
  - Verwende `StandardScaler()`.

```python=
scaler = StandardScaler(with_mean=True, with_std=True, copy=True)
```

- **Parameter Einstellungen**
  - `with_mean`: Bestimmt, ob der Mittelwert subtrahiert wird.
  - `with_std`: Bestimmt, ob Spalten eine Standardabweichung von 1 haben sollen.
  - `copy=True`: Stellt sicher, dass Daten für Berechnungen kopiert werden.

```python=
scaler.fit(feature_df)
X_std = scaler.transform(feature_df)
```

```python=
X_std[:1].round(1)
```

    array([[ 0.7, -0.3,  0.4, -1.2,  0.8, -0.7,  0.2, -0.1, -0.2,  0.4, -0.9,
            -0.2, -0.5, -0.8,  0.8, -0.3, -0.8,  1.1, -0.5, -0.5,  0.5, -0.5,
             1.6, -0.2, -0.4, -0.5, -0.1,  1.2, -0.1, -1. ,  1. ,  1.3, -1.1,
            -0.6,  0.9, -0.9, -1.2,  0.2,  1.2, -0.7, -0.4,  0.2, -0.6, -0.8,
            -0.1, -0.1,  1. , -0.1, -0.2, -0. , -0.1, -0.2, -0.1, -0.3, -0.2,
            -0.1, -0.1, -0.1,  1.7, -0. , -0.1, -0.2, -0.1, -0.1, -0.8, -0.1,
            -0.1,  0.7, -0.1, -0.2, -0. , -0.1, -0.1, -0. , -0.3, -0.2, -0.1,
            -0.1, -0.1,  0.8, -0. , -0.1, -0.2, -0.1, -0.1]])

- Now each column of `feature_std` has a mean of zero and a standard deviation of one.

```python=
feature_std = pd.DataFrame(X_std, columns=feature_df.columns);
feature_std.std()
```

    MOSTYPE     1.000086
    MAANTHUI    1.000086
    MGEMOMV     1.000086
    MGEMLEEF    1.000086
    MOSHOOFD    1.000086
                  ...   
    AZEILPL     1.000086
    APLEZIER    1.000086
    AFIETS      1.000086
    AINBOED     1.000086
    ABYSTAND    1.000086
    Length: 85, dtype: float64

- **Standardabweichungen**
  - `scaler()` nutzt $1/n$.
  - `std()` nutzt $1/(n-1)$.
  - Unterschiedliche Konventionen, aber gleiche Skalierung der Variablen.
- **Datenaufteilung**
  - Verwende `train_test_split()`.
  - Testset: 1000 Beobachtungen.
  - Trainingsset: Restliche Daten.

```python=
(X_train, X_test,  y_train, y_test) = train_test_split(
    feature_std, Purchase, test_size=1000, random_state=0)
```

- **KNN-Modell**
  - Fit auf Trainingsdaten mit K=1.
  - Bewertung auf Testdaten.

```python=
knn1 = KNeighborsClassifier(n_neighbors=1)
knn1_pred = knn1.fit(X_train, y_train).predict(X_test)
np.mean(y_test != knn1_pred), np.mean(y_test == "Yes")
```

    (0.111, 0.067)

- **KNN-Fehlerrate**
  - Fehlerrate auf 1.000 Testdaten: ca. 11%.
  - Immer "Nein" vorhersagen: Fehlerrate ca. 6% (*null rate*).
- **Verkauf von Versicherungen**
  - Erfolg von 6% durch Zufall ist zu niedrig.
  - Ziel: Kunden identifizieren, die wahrscheinlich kaufen.
  - Fokus: Korrekte Vorhersage der Käufer.

```python=
confusion_table(knn1_pred, y_test)
```

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Truth</th>
      <th>No</th>
      <th>Yes</th>
    </tr>
    <tr>
      <th>Predicted</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No</th>
      <td>880</td>
      <td>58</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>53</td>
      <td>9</td>
    </tr>
  </tbody>
</table>

```python=
(880+9)/1000
```
    0.889

- **KNN mit K=1**
  - Bessere Leistung als zufälliges Raten bei Versicherungsprognosen.
  - Von 62 vorhergesagten Käufern kaufen 9 (14,5%) tatsächlich.
  - Doppelte Rate im Vergleich zum zufälligen Raten.

## Parameteroptimierung (Tuning Parameters)
- **Anzahl der Nachbarn (KNN)**
  - Hyperparameter, dessen optimaler Wert vorher unbekannt ist.
  - Leistung wird auf Testdaten durch Variation dieses Parameters bewertet.
- **Untersuchung der Genauigkeit**
  - Verwende eine `for`-Schleife.
  - Prüfe die Klassifizierungsgenauigkeit für Nachbarn von 1 bis 5.

```python=
for K in range(1,6):
    knn = KNeighborsClassifier(n_neighbors=K)
    knn_pred = knn.fit(X_train, y_train).predict(X_test)
    C = confusion_table(knn_pred, y_test)
    templ = ('K={0:d}: # predicted to rent: {1:>2},' +
            '  # who did rent {2:d}, accuracy {3:.1%}')
    pred = C.loc['Yes'].sum()
    did_rent = C.loc['Yes','Yes']
    print(templ.format(
          K,
          pred,
          did_rent,
          did_rent / pred))
```

    K=1: # predicted to rent: 62,  # who did rent 9, accuracy 14.5%
    K=2: # predicted to rent:  6,  # who did rent 1, accuracy 16.7%
    K=3: # predicted to rent: 20,  # who did rent 3, accuracy 15.0%
    K=4: # predicted to rent:  4,  # who did rent 0, accuracy 0.0%
    K=5: # predicted to rent:  7,  # who did rent 1, accuracy 14.3%

## Vergleich zur logistischen Regression
- **Logistische Regression mit `sklearn`**
  - Standardmäßig Ridge-Regression.
  - `C` auf hohen Wert setzen für übliche logistische Regression.
- **Unterschiede zu `statsmodels`**
  - `sklearn`: Fokus auf Klassifikation.
  - Keine `summary`-Methoden für detaillierte Inferenz.

```python=
logit = LogisticRegression(C=1e10, solver='liblinear')
logit.fit(X_train, y_train)
logit_pred = logit.predict_proba(X_test)
logit_labels = np.where(logit_pred[:,1] > .5, 'Yes', 'No')
confusion_table(logit_labels, y_test)
```

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Truth</th>
      <th>No</th>
      <th>Yes</th>
    </tr>
    <tr>
      <th>Predicted</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No</th>
      <td>931</td>
      <td>67</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

- **Solver-Einstellung**
  - `solver='liblinear'` verwendet, um Konvergenzwarnungen zu vermeiden.
- **Wahrscheinlichkeitsschwellen**
  - 0.5: Nur 2 Käufe vorhergesagt.
  - 0.25: 29 Käufe vorhergesagt.
  - Genauigkeit: ca. 31%, fast fünfmal besser als zufällig.

```python=
logit_labels = np.where(logit_pred[:,1]>0.25, 'Yes', 'No')
confusion_table(logit_labels, y_test)
```

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Truth</th>
      <th>No</th>
      <th>Yes</th>
    </tr>
    <tr>
      <th>Predicted</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No</th>
      <td>913</td>
      <td>58</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>20</td>
      <td>9</td>
    </tr>
  </tbody>
</table>

```python=
9/(20+9)
```

    0.3103448275862069
