---
title: "Auto"
author: "db"
---

# 1. Datenüberblick

- Dataset von StatLib, verwendet in der ASA Exposition 1983.
- Ursprünglich 397 Beobachtungen; 5 Zeilen mit fehlender Leistung entfernt.
- Es gibt 392 Zeilen.
- Es gibt 9 Variablen:
  - mpg: Meilen pro Gallone
  - cylinders: Anzahl der Zylinder (4 bis 8)
  - displacement: Hubraum (Kubikzoll)
  - horsepower: Motorleistung
  - weight: Fahrzeuggewicht (Pfund)
  - acceleration: 0-60 mph Zeit (Sek.)
  - year: Modelljahr
  - origin: Autoherkunft (1. Amerikanisch, 2. Europäisch, 3. Japanisch)
  - name: Fahrzeugname

# 2. Packages und Daten

```python=
import numpy as np, statsmodels.api as sm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS, summarize, poly, sklearn_sm)
from sklearn.model_selection import train_test_split
from functools import partial
from sklearn.model_selection import \
     (train_test_split, cross_validate, KFold, ShuffleSplit)
from sklearn.base import clone
```

```python=
Auto = load_data('Auto')
Auto.origin = Auto.origin.astype('category')
Auto.head()
```

<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>year</th>
      <th>origin</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>chevrolet chevelle malibu</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>buick skylark 320</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>plymouth satellite</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>amc rebel sst</th>
      <td>16.0</td>
      <td>8</td>
      <td>304.0</td>
      <td>150</td>
      <td>3433</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>ford torino</th>
      <td>17.0</td>
      <td>8</td>
      <td>302.0</td>
      <td>140</td>
      <td>3449</td>
      <td>10.5</td>
      <td>70</td>
      <td>1</td>
    </tr>
  </tbody>
</table>

```python=
Auto.describe().round(1)
```

<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>392.0</td>
      <td>392.0</td>
      <td>392.0</td>
      <td>392.0</td>
      <td>392.0</td>
      <td>392.0</td>
      <td>392.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>23.4</td>
      <td>5.5</td>
      <td>194.4</td>
      <td>104.5</td>
      <td>2977.6</td>
      <td>15.5</td>
      <td>76.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.8</td>
      <td>1.7</td>
      <td>104.6</td>
      <td>38.5</td>
      <td>849.4</td>
      <td>2.8</td>
      <td>3.7</td>
    </tr>
    <tr>
      <th>min</th>
      <td>9.0</td>
      <td>3.0</td>
      <td>68.0</td>
      <td>46.0</td>
      <td>1613.0</td>
      <td>8.0</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>17.0</td>
      <td>4.0</td>
      <td>105.0</td>
      <td>75.0</td>
      <td>2225.2</td>
      <td>13.8</td>
      <td>73.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>22.8</td>
      <td>4.0</td>
      <td>151.0</td>
      <td>93.5</td>
      <td>2803.5</td>
      <td>15.5</td>
      <td>76.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>29.0</td>
      <td>8.0</td>
      <td>275.8</td>
      <td>126.0</td>
      <td>3614.8</td>
      <td>17.0</td>
      <td>79.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>46.6</td>
      <td>8.0</td>
      <td>455.0</td>
      <td>230.0</td>
      <td>5140.0</td>
      <td>24.8</td>
      <td>82.0</td>
    </tr>
  </tbody>
</table>

# 3. Validierungsansatz

- Validierungsansatz zur Schätzung der Testfehler bei linearen Modellen auf dem `Auto`-Datensatz.
- Mit `train_test_split()` teilen wir die 392 Beobachtungen in je 196 Trainings- und Validierungsdaten (`test_size=196`).
- Zur Reproduzierbarkeit setzen wir `random_state=0`.

```python=
Auto_train, Auto_valid = train_test_split(Auto, test_size=196, random_state=0)
```

```python=
hp_mm = MS(['horsepower'])
X_train = hp_mm.fit_transform(Auto_train)
y_train = Auto_train['mpg']
model = sm.OLS(y_train, X_train)
results = model.fit()
```

```python=
X_valid = hp_mm.transform(Auto_valid)
y_valid = Auto_valid['mpg']
valid_pred = results.predict(X_valid)
np.mean((y_valid - valid_pred)**2)
```

    23.61661706966988

- Geschätzte Validierungs-MSE für lineare Regression: $23.62$.
- `evalMSE()` Funktion schätzt Validierungsfehler für polynomiale Regressionen mit Modellstring und Trainings-/Testdaten.

```python=
def evalMSE(terms, response, train, test):
   mm = MS(terms)
   X_train = mm.fit_transform(train)
   y_train = train[response]
   X_test = mm.transform(test)
   y_test = test[response]
   results = sm.OLS(y_train, X_train).fit()
   test_pred = results.predict(X_test)
   return np.mean((y_test - test_pred)**2)
```

```python=
MSE = np.zeros(3)
for idx, degree in enumerate(range(1, 4)):
    MSE[idx] = evalMSE([poly('horsepower', degree)], 'mpg',
                       Auto_train, Auto_valid)
MSE
```

    array([23.61661707, 18.76303135, 18.79694163])

- Fehlerraten sind $23.62$, $18.76$ und $18.80$.
- Unterschiedliche Trainings-/Validierungsteilungen können verschiedene Validierungsfehler ergeben.

```python=
Auto_train, Auto_valid = train_test_split(Auto, test_size=196, random_state=3)
MSE = np.zeros(3)
for idx, degree in enumerate(range(1, 4)):
    MSE[idx] = evalMSE([poly('horsepower', degree)], 'mpg',
                       Auto_train, Auto_valid)
MSE
```

    array([20.75540796, 16.94510676, 16.97437833])

- Validierungsfehler für Modelle mit linearen, quadratischen und kubischen Termen: $20.76$, $16.95$ und $16.97$.
- Quadratische Funktion von `horsepower` sagt `mpg` besser voraus als eine lineare, ohne Verbesserung durch eine kubische Funktion.

# 4. Kreuzvalidierung (Cross-Validation)

- Für die Kreuzvalidierung generalisierter linearer Modelle in Python, nutze `sklearn`. Es hat eine andere API als `statsmodels`.
- Datenwissenschaftler müssen oft Funktionen für Aufgaben A und B verknüpfen, um B(A(D)) zu berechnen. Bei Inkompatibilität ist ein *Wrapper* erforderlich.
- Das `ISLP`-Paket bietet `sklearn_sm()`, einen Wrapper zur Nutzung von `sklearn` Kreuzvalidierung mit `statsmodels` Modellen.
- `sklearn_sm()` nimmt ein `statsmodels` Modell als erstes Argument. Optional sind `model_str` für Formeln und `model_args` für zusätzliche Anpassungsargumente, wie `{'family': sm.families.Binomial()}` für logistische Regression.

```python=
hp_model = sklearn_sm(sm.OLS)
X, Y = MS(['horsepower']).fit_transform(Auto), Auto['mpg']
cv_results = cross_validate(hp_model, X, Y, cv=Auto.shape[0]) # LOOCV
np.mean(cv_results['test_score'])
```

    24.231513517929226

```python=
hp_model = sklearn_sm(sm.OLS, MS(['horsepower']))
X, Y = Auto.drop(columns=['mpg']), Auto['mpg']
cv_results = cross_validate(hp_model, X, Y, cv=Auto.shape[0]) # LOOCV
np.mean(cv_results['test_score'])
```

    24.231513517929212

- Die `cross_validate()` Funktion benötigt ein Objekt mit den Methoden `fit()`, `predict()` und `score()`. Außerdem werden das Merkmalsarray `X` und die Antwort `Y` benötigt.
  - Das Argument `cv` gibt die Art der Kreuzvalidierung an: eine ganze Zahl für $K$-fache oder die Anzahl der Beobachtungen für LOOCV.
  - Die Funktion gibt ein Wörterbuch zurück. Hier ist der kreuzvalidierte Testwert (MSE) 24.23.
- Der Prozess kann mit einer for-Schleife automatisiert werden. Diese passt polynomiale Regressionen von Grad 1 bis 5 an.
  - Sie berechnet den Kreuzvalidierungsfehler für jeden Grad und speichert ihn im Vektor `cv_error`.
  - Die Variable `d` steht für den Polynomgrad.
- Der Test-MSE sinkt stark von linearen zu quadratischen Anpassungen. Es gibt keine Verbesserung bei höheren Polynomgraden.

```python=
cv_error = np.zeros(5)
H = np.array(Auto['horsepower'])
M = sklearn_sm(sm.OLS)
for i, d in enumerate(range(1,6)):
    X = np.power.outer(H, np.arange(d+1))
    M_CV = cross_validate(M, X, Y, cv=Auto.shape[0])
    cv_error[i] = np.mean(M_CV['test_score'])
cv_error
```

    array([24.23151352, 19.24821312, 19.33498406, 19.42443031, 19.03320428])

- Erklärung: die `outer()` Methode der `np.power()` Funktion wendet Operationen wie `add()`, `min()` oder `power()` an.
  - Sie nimmt zwei Arrays als Argumente. Dann bildet sie ein größeres Array, in dem die Operation auf jedes Elementpaar der beiden Arrays angewendet wird.

```python=
A = np.array([3, 5, 9])
np.power.outer(A, np.arange(2))
```

    array([[1, 3],
           [1, 5],
           [1, 9]])

- Im obigen CV-Beispiel verwendeten wir $K=n$ für LOOCV.
- Wir können auch $K<n$ verwenden, was schneller ist.
  - Mit `KFold()` und K=10` sowie `random_state` speichern wir CV-Fehler für polynomiale Anpassungen von Grad eins bis fünf.

```python=
cv_error = np.zeros(5)
cv = KFold(n_splits=10,
           shuffle=True,
           random_state=0) # use same splits for each degree
for i, d in enumerate(range(1,6)):
    X = np.power.outer(H, np.arange(d+1))
    M_CV = cross_validate(M, X, Y, cv=cv)
    cv_error[i] = np.mean(M_CV['test_score'])
cv_error
```

    array([24.20766449, 19.18533142, 19.27626666, 19.47848404, 19.13722016])

- Quadratische Anpassungen performen ähnlich wie Polynomiale höheren Grades.
- Hinweis auf Verwendung 'KFold' und 'ShuffleSplit':

```python=
# Example data
X = [i for i in range(10)]
# KFold with 3 splits
kf = KFold(n_splits=3, shuffle=True, random_state=0)
# Splitting the data
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
```

    TRAIN: [0 1 3 5 6 7] TEST: [2 4 8 9]
    TRAIN: [0 2 3 4 5 8 9] TEST: [1 6 7]
    TRAIN: [1 2 4 6 7 8 9] TEST: [0 3 5]

- `KFold(n_splits=10, shuffle=True, random_state=0)`:
  - Teilt die Daten in 10 aufeinanderfolgende, zufällig gemischte Faltungen auf.
  - Jede Faltung dient einmal als Testmenge.
  - Aufteilungen sind voneinander abhängig.
  - `KFold` sorgt dafür, dass jede Probe genau einmal getestet wird, mit Durchmischen vor den Aufteilungen.

```python=
X = [i for i in range(10)]
ss = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
for train_index, test_index in ss.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
```

    TRAIN: [9 1 6 7 3 0 5] TEST: [2 8 4]
    TRAIN: [2 9 8 0 6 7 4] TEST: [3 5 1]
    TRAIN: [4 5 1 0 6 9 7] TEST: [2 3 8]

- `ShuffleSplit(n_splits=10, test_size=196, random_state=0)`:
  
  - Generiert 10 zufällige Trainings-/Testaufteilungen.
  - Jede Testmenge hat 196 Proben.
  - Aufteilungen sind unabhängig.
  - `ShuffleSplit` erzeugt zufällige, unabhängige Aufteilungen.

- Die `cross_validate()` Funktion kann `ShuffleSplit()` für Validierungssätze oder K-fache Kreuzvalidierung verwenden. 

```python=
validation = ShuffleSplit(n_splits=1, test_size=196, random_state=0)
results = cross_validate(hp_model,
                         Auto.drop(['mpg'], axis=1),
                         Auto['mpg'],
                         cv=validation)
results['test_score']
```

    array([23.61661707])

```python=
validation = ShuffleSplit(n_splits=10, test_size=196, random_state=0)
results = cross_validate(hp_model,
                         Auto.drop(['mpg'], axis=1),
                         Auto['mpg'],
                         cv=validation)
results['test_score'].mean(), results['test_score'].std()
```

    (23.802232661034164, 1.4218450941091847)

- Die Standardabweichung hier ist keine gültige Schätzung der Variabilität des mittleren Testergebnisses aufgrund überlappender Stichproben.
- Sie zeigt jedoch die Monte-Carlo-Variation aus verschiedenen zufälligen Faltungen an.

# 5. Der Bootstrap

- Die Bootstrap-Methode bewertet die Variabilität von Koeffizientenschätzungen und Vorhersagen.
- Wir verwenden sie, um die Variabilität der Schätzungen für $\beta_0$ und $\beta_1$ in einem linearen Regressionsmodell zu bewerten.
  - Dieses Modell sagt `mpg` basierend auf `horsepower` im `Auto` Datensatz voraus.
- Wir vergleichen diese Schätzungen mit den Standardfehlerformeln.

```python=
def boot_OLS(model_matrix, response, D, idx):
    """
    Perform bootstrap OLS regression on a subset of data.

    Parameters:
    model_matrix: Model matrix to clone and fit_transform.
    response: Name of the response variable.
    D: The DataFrame containing the data.
    idx: Indices for subsetting the DataFrame.

    Returns:
    pandas.Series: OLS regression parameters.
    """
    D_ = D.loc[idx]
    Y_ = D_[response]
    X_ = clone(model_matrix).fit_transform(D_)
    return sm.OLS(Y_, X_).fit().params
```

- Die ersten beiden Argumente von `boot_SE()` sollten während des Bootstrappings unverändert bleiben.
  - Verwenden `partial()` aus `functools`, um diese Argumente in `boot_OLS()` festzulegen.

```python=
hp_func = partial(boot_OLS, model_matrix=MS(['horsepower']), response='mpg')
```

- Geben `hp_func?` ein, um zu sehen, dass es zwei Argumente hat, `D` und `idx`.
- Verwenden `hp_func()`, um Bootstrap-Schätzungen für den Achsenabschnitt und die Steigung zu erstellen.
  - Ziehen dabei Stichproben mit Zurücklegen.
  - Demonstration mit 10 Stichproben.

```python=
rng = np.random.default_rng(0)
np.array([hp_func(Auto, rng.choice(Auto.index, 392,
                     replace=True)) for _ in range(10)])
```

    array([[39.12226577, -0.1555926 ],
           [37.18648613, -0.13915813],
           [37.46989244, -0.14112749],
           [38.56723252, -0.14830116],
           [38.95495707, -0.15315141],
           [39.12563927, -0.15261044],
           [38.45763251, -0.14767251],
           [38.43372587, -0.15019447],
           [37.87581142, -0.1409544 ],
           [37.95949036, -0.1451333 ]])

- Wir erstellen die Funktion `boot_SE` für Berechnen den Standardfehler der Bootstrap-Schätzungen für Achsenabschnitt und Steigung.

```python=
def boot_SE(func, D, n=None, B=1000, seed=0):
    """
    Calculate the standard error of bootstrap estimates.

    Parameters:
    func (callable): The function to apply to the bootstrap samples.
    D (DataFrame): The data to sample from.
    n (int, optional): The sample size. Defaults to the number of rows in D.
    B (int, optional): The number of bootstrap iterations. Defaults to 1000.
    seed (int, optional): The seed for the random number generator. Defaults to 0.

    Returns:
    float: The bootstrap standard error.
    """
    rng = np.random.default_rng(seed)
    first_, second_ = 0, 0
    n = n or D.shape[0]
    for _ in range(B):
        idx = rng.choice(D.index, n, replace=True)
        value = func(D, idx)
        first_ += value
        second_ += value**2
    return np.sqrt(second_ / B - (first_ / B)**2)
```

- Wir verwenden die Funktion `boot_SE()` für Berechnen die Standardfehler von 1.000 Bootstrap-Schätzungen für den Achsenabschnitt und die Steigung.

```python=
boot_SE(hp_func, Auto, B=1000, seed=10)
```

    intercept     0.731176
    horsepower    0.006092
    dtype: float64

- Vergleichen die geschätzten Standardfehler aus dem Bootstrap mit der geschätzten Standardfehler aus der OLS-Methode.

```python=
hp_model.fit(Auto, Auto['mpg'])
summarize(hp_model.results_)['std err']
```

    intercept     0.717
    horsepower    0.006
    Name: std err, dtype: float64

- Die Standardfehler unterscheiden sich von den Bootstrap-Schätzungen.
- Es zeigt die Annahmen in den Standardformeln.
  - Diese hängen von $\sigma^2$ ab.
    - $\sigma^2$ wird mit dem RSS geschätzt, was aufgrund von Nichtlinearität aufgebläht sein kann.
  - Die Standardformeln setzen unrealistischerweise feste $x_i$ voraus.
- Der Bootstrap verlässt sich nicht auf diese Annahmen. Daher bietet der Bootstrap wahrscheinlich genauere Schätzungen.
- Als Nächstes berechnen wir Bootstrap- und Standard-Schätzungen für ein quadratisches Modell.

```python=
quad_model = MS([poly('horsepower', 2, raw=True)])
quad_func = partial(boot_OLS, quad_model, 'mpg')
boot_SE(quad_func, Auto, B=1000)
```

    intercept                                  1.538641
    poly(horsepower, degree=2, raw=True)[0]    0.024696
    poly(horsepower, degree=2, raw=True)[1]    0.000090
    dtype: float64

```python=
M = sm.OLS(Auto['mpg'],
           quad_model.fit_transform(Auto))
summarize(M.fit())['std err']
```

    intercept                                  1.800
    poly(horsepower, degree=2, raw=True)[0]    0.031
    poly(horsepower, degree=2, raw=True)[1]    0.000
    Name: std err, dtype: float64
