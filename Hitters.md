---
title: "Hitters"
author: "db"
---

# 1. Datenüberblick

- Major League Baseball Daten (1986-1987)
- Zeilen: 322 (59 fehlen, 263 übrig)
- Zielvariable: `Salary` (Jahresgehalt 1987 in Tausend Dollar)
- Variablen (20):
  - `AtBat`: Schläge 1986
  - `Hits`: Treffer 1986
  - `HmRun`: Home Runs 1986
  - `Runs`: Runs 1986
  - `RBI`: Runs Batted In 1986
  - `Walks`: Walks 1986
  - `Years`: Jahre in den Major Leagues
  - `CAtBat`: Schläge in der Karriere
  - `CHits`: Treffer in der Karriere
  - `CHmRun`: Home Runs in der Karriere
  - `CRuns`: Runs in der Karriere
  - `CRBI`: Runs Batted In in der Karriere
  - `CWalks`: Walks in der Karriere
  - `League`: Liga Ende 1986 (A, N)
  - `Division`: Division Ende 1986 (E, W)
  - `PutOuts`: Put Outs 1986
  - `Assists`: Assists 1986
  - `Errors`: Errors 1986
  - `Salary`: Jahresgehalt 1987 in Tausend Dollar
  - `NewLeague`: Liga Anfang 1987 (A, N)

# 2. Packages und Daten

```python=
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import statsmodels.api as sm
import sklearn.model_selection as skm
import sklearn.linear_model as skl
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from ISLP import load_data
from ISLP.models import \
 (ModelSpec as MS, Stepwise,
  sklearn_selected, sklearn_selection_path)
from l0bnb import fit_path
from functools import partial
```

```python=
Hitters = load_data('Hitters'); print(Hitters.shape); Hitters.head(2)
```

(322, 20)

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AtBat</th>
      <th>Hits</th>
      <th>HmRun</th>
      <th>Runs</th>
      <th>RBI</th>
      <th>Walks</th>
      <th>Years</th>
      <th>CAtBat</th>
      <th>CHits</th>
      <th>CHmRun</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>293</td>
      <td>66</td>
      <td>1</td>
      <td>30</td>
      <td>29</td>
      <td>14</td>
      <td>1</td>
      <td>293</td>
      <td>66</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>315</td>
      <td>81</td>
      <td>7</td>
      <td>24</td>
      <td>38</td>
      <td>39</td>
      <td>14</td>
      <td>3449</td>
      <td>835</td>
      <td>69</td>
    </tr>
  </tbody>
</table>

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRuns</th>
      <th>CRBI</th>
      <th>CWalks</th>
      <th>League</th>
      <th>Division</th>
      <th>PutOuts</th>
      <th>Assists</th>
      <th>Errors</th>
      <th>Salary</th>
      <th>NewLeague</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30</td>
      <td>29</td>
      <td>14</td>
      <td>A</td>
      <td>E</td>
      <td>446</td>
      <td>33</td>
      <td>20</td>
      <td>NaN</td>
      <td>A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>321</td>
      <td>414</td>
      <td>375</td>
      <td>N</td>
      <td>W</td>
      <td>632</td>
      <td>43</td>
      <td>10</td>
      <td>475.0</td>
      <td>N</td>
    </tr>
  </tbody>
</table>

```python=
Hitters.describe().round(1)
```

<table>
  <thead>
<tr style="text-align: right;">
  <th></th>
  <th>AtBat</th>
  <th>Hits</th>
  <th>HmRun</th>
  <th>Runs</th>
  <th>RBI</th>
  <th>Walks</th>
  <th>Years</th>
  <th>CAtBat</th>
  <th>CHits</th>
  <th>CHmRun</th>
  <th>CRuns</th>
  <th>CRBI</th>
  <th>CWalks</th>
  <th>PutOuts</th>
  <th>Assists</th>
  <th>Errors</th>
  <th>Salary</th>
</tr>
  </thead>
  <tbody>
<tr>
  <th>mean</th>
  <td>380.9</td>
  <td>101.0</td>
  <td>10.8</td>
  <td>50.9</td>
  <td>48.0</td>
  <td>38.7</td>
  <td>7.4</td>
  <td>2648.7</td>
  <td>717.6</td>
  <td>69.5</td>
  <td>358.8</td>
  <td>330.1</td>
  <td>260.2</td>
  <td>288.9</td>
  <td>106.9</td>
  <td>8.0</td>
  <td>535.9</td>
</tr>
<tr>
  <th>std</th>
  <td>153.4</td>
  <td>46.5</td>
  <td>8.7</td>
  <td>26.0</td>
  <td>26.2</td>
  <td>21.6</td>
  <td>4.9</td>
  <td>2324.2</td>
  <td>654.5</td>
  <td>86.3</td>
  <td>334.1</td>
  <td>333.2</td>
  <td>267.1</td>
  <td>280.7</td>
  <td>136.9</td>
  <td>6.4</td>
  <td>451.1</td>
</tr>
<tr>
  <th>min</th>
  <td>16.0</td>
  <td>1.0</td>
  <td>0.0</td>
  <td>0.0</td>
  <td>0.0</td>
  <td>0.0</td>
  <td>1.0</td>
  <td>19.0</td>
  <td>4.0</td>
  <td>0.0</td>
  <td>1.0</td>
  <td>0.0</td>
  <td>0.0</td>
  <td>0.0</td>
  <td>0.0</td>
  <td>0.0</td>
  <td>67.5</td>
</tr>
<tr>
  <th>25%</th>
  <td>255.2</td>
  <td>64.0</td>
  <td>4.0</td>
  <td>30.2</td>
  <td>28.0</td>
  <td>22.0</td>
  <td>4.0</td>
  <td>816.8</td>
  <td>209.0</td>
  <td>14.0</td>
  <td>100.2</td>
  <td>88.8</td>
  <td>67.2</td>
  <td>109.2</td>
  <td>7.0</td>
  <td>3.0</td>
  <td>190.0</td>
</tr>
<tr>
  <th>50%</th>
  <td>379.5</td>
  <td>96.0</td>
  <td>8.0</td>
  <td>48.0</td>
  <td>44.0</td>
  <td>35.0</td>
  <td>6.0</td>
  <td>1928.0</td>
  <td>508.0</td>
  <td>37.5</td>
  <td>247.0</td>
  <td>220.5</td>
  <td>170.5</td>
  <td>212.0</td>
  <td>39.5</td>
  <td>6.0</td>
  <td>425.0</td>
</tr>
<tr>
  <th>75%</th>
  <td>512.0</td>
  <td>137.0</td>
  <td>16.0</td>
  <td>69.0</td>
  <td>64.8</td>
  <td>53.0</td>
  <td>11.0</td>
  <td>3924.2</td>
  <td>1059.2</td>
  <td>90.0</td>
  <td>526.2</td>
  <td>426.2</td>
  <td>339.2</td>
  <td>325.0</td>
  <td>166.0</td>
  <td>11.0</td>
  <td>750.0</td>
</tr>
<tr>
  <th>max</th>
  <td>687.0</td>
  <td>238.0</td>
  <td>40.0</td>
  <td>130.0</td>
  <td>121.0</td>
  <td>105.0</td>
  <td>24.0</td>
  <td>14053.0</td>
  <td>4256.0</td>
  <td>548.0</td>
  <td>2165.0</td>
  <td>1659.0</td>
  <td>1566.0</td>
  <td>1378.0</td>
  <td>492.0</td>
  <td>32.0</td>
  <td>2460.0</td>
</tr>
  </tbody>
</table>

```python=
np.isnan(Hitters['Salary']).sum()
```

59

```python=
Hitters = Hitters.dropna(); Hitters.shape
```

(263, 20)

# 3. Teilmengewahlmethoden

## 3.1. Vorwärtsselektion

```python=
designMS = MS(Hitters.columns.drop('Salary')).fit(Hitters)
X = designMS.transform(Hitters)
# Shape of X: 263 x 20
```

- `Stepwise.first_peak()`: Läuft, bis keine Verbesserung mehr gefunden wird.
- `Stepwise.fixed_steps()`: Läuft eine festgelegte Anzahl von Schritten.

```python=
strategy = Stepwise.first_peak(designMS, direction='forward',
                               max_terms=len(designMS.terms))
```

- `sklearn_selected()` aus `ISLP.models`, um ein lineares Regressionsmodell anzupassen.
  - Nutzen ein `statsmodels`-Modell und eine Suchstrategie.
  - Standardbewertung ist MSE.
- Hinweis: Die Methode `fit()` kann `X` nicht als Eingabe verwenden.

```python=
# Define the response
Y = np.array(Hitters.Salary)
# Fit the model
hitters_MSE = sklearn_selected(sm.OLS, strategy).fit(Hitters, Y)
# Show the selected predictors
hitters_MSE.selected_state_
```

    ('Assists', 'AtBat', 'CAtBat', 'CHits', 'CHmRun')
    ('CRBI', 'CRuns', 'CWalks', 'Division', 'Errors')
    ('Hits', 'HmRun', 'League', 'NewLeague', 'PutOuts')
    ('RBI', 'Runs', 'Walks', 'Years')

- Definieren eine funkftion für $C_p$ als Kriterium zur Modellauswahl.
  - Da `sklearn` höhere Werte bevorzugt, müssen wir $C_p$ negieren.

```python=
def nCp(sigma2, estimator, X, Y):
    n, p = X.shape
    Yhat = estimator.predict(X)
    RSS = np.sum((Y - Yhat)**2)
    return - (RSS + 2 * p * sigma2) / n
```

- Berechnung $\sigma^2$ aus die vollige Modell für `nCp`
- Fixierung von $\sigma^2$:
  - Weil Parameter `scoring` nur `estimator, X, Y` benötigt.
  - mit `partial()` aus `functools`.

```python=
# Compute the sigma2
sigma2 = sm.OLS(Y,X).fit().scale
# Fix sigma2 in nCp
neg_Cp = partial(nCp, sigma2)
```

```python=
hitters_Cp = sklearn_selected(sm.OLS, strategy, scoring=neg_Cp).fit(Hitters, Y)
hitters_Cp.selected_state_
```

    ('Assists', 'AtBat', 'CAtBat', 'CRBI', 'CRuns')
    ('CWalks', 'Division', 'Hits', 'PutOuts', 'Walks')

### a. Auswahl von Modellen mit alle Daten

- `sklearn_selection_path()`:
  - Speichern alle Schritt
- `Stepwise.fixed_steps()`:
  - von 0 bis 19 Variablen
- Ergebnis: Data frame 263 x 20

```python=
strategy = Stepwise.fixed_steps(model_spec=designMS, n_steps=19,
                                direction='forward')
full_path = sklearn_selection_path(sm.OLS, strategy).fit(Hitters, Y)
Yhat_in = full_path.predict(Hitters)
# Shape of Yhat_in: 263 x 20
# row is fitted values, column is model
```

- Berechnung des MSE für jedes Modell und Plotten.

```python=
mse_fig, ax = plt.subplots(figsize=(8,8))
insample_mse = ((Yhat_in - Y[:,None])**2).mean(0)
# Y[:,None] is to make it a column vector.
# mean(0) is to average over the rows.
# insample_mse: 20,
ax.plot(np.arange(20), insample_mse, 'k', label='In-sample')
ax.set_ylabel('MSE'); ax.set_xlabel('Number of Variables')
ax.set_ylim([50000,250000]); ax.set_xticks(range(20)); ax.legend();
```

![](Figures/hitters_22_0.png)

### b. Auswahl von Modellen mit Kreuzvalidierung

- `cross_val_predict()` aus `sklearn`:
  - 5-facher Kreuzvalidierung für jedes Modell.
    - Erster Fold ausgeschlossen, Modell auf die Reste trainiert.
    - Angepasste Werte für ersten Fold berechnet.
    - Prozess für restliche Folds wiederholt.

```python=
K = 5
kfold = skm.KFold(K, shuffle=True, random_state=0)
Yhat_cv = skm.cross_val_predict(full_path, Hitters, Y, cv=kfold)
# Shape of Yhat_cv: 263 x 20
```

- `split()`-Methode von `kfold`:
  - identifizieren Indizes der Trainings- und Testsätze für jeden Fold

```python=
for train_idx, test_idx in kfold.split(Y):
  print(test_idx[:3])
# Shape of train_idx: 210, of test_idx: 53
```

    [ 5  7  8]
    [ 3  4 13]
    [ 0  2 10]
    [ 1  6 11]
    [ 9 21 25]

- Berechnung des MSE für jeden Fold für jedes Modell.
  - Jedes Modell has 5 MSEs.
- Berechnung von Mittelwert und Varianz des MSE für jedes Modell.
  - Varianzschätzung ist ungenau wegen überlappender Trainingssätze (Fehler sind nicht unabhängig).

```python=
cv_mse = []
for train_idx, test_idx in kfold.split(Y):
  squared_errors = (Yhat_cv[test_idx] - Y[test_idx, None])**2
  # Y[test_idx, None] is to make it a column vector.
  cv_mse.append(squared_errors.mean(0))
  # mean(0) is to average over the rows.
# Shape of cv_mse: 5 x 20
```

- Plotten: mittleren MSEs und Standardfehlers des mittleren MSEs
  - $\text{std(mean)} = \frac{Std}{\sqrt{5}}$

```python=
ax.errorbar(x = np.arange(20),
            y = np.mean(cv_mse, axis=0),
            yerr = np.std(cv_mse, axis=0) / np.sqrt(5),
            label = 'Cross-validated',
            c = 'r') # color red
ax.set_ylim([50000,250000]); ax.legend(); mse_fig
```

![](Figures/hitters_29_0.png)

### c. Auswahl von Modellen mit dem Validierungsansatz

- `skm.ShuffleSplit()`: Aufteilung der Daten in Trainingssatz (80% = 210) und Testsatz (20% = 53)

```python=
validation = skm.ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
# Get indices of the training and test set
# train_idx, test_idx = list(validation.split(Y))[0]
for train_idx, test_idx in validation.split(Y):
  train_idx = train_idx
  test_idx = test_idx
```

- Jeden Schritt:
  - Modell auf dem Trainingssatz trainieren.
  - Vorhersage auf dem Validierungssatz.
  - Ergebnis: Data frame 53 x 20.

```python=
strategy = Stepwise.fixed_steps(model_spec=designMS, n_steps=19, direction='forward')
# Fit the models on the training set
full_path = sklearn_selection_path(sm.OLS, strategy)
full_path.fit(Hitters.iloc[train_idx], Y[train_idx])
# Predict values on the validation set
Yhat_val = full_path.predict(Hitters.iloc[test_idx])
# Shape of Yhat_val: 53 x 20
```

```python=
squared_errors = (Yhat_val - Y[test_idx, None])**2
# Shape of squared_errors: 53 x 20
validation_mse = squared_errors.mean(0)
# validation_mse is a vector of 20 
```

```python=
ax.plot(np.arange(20), # n_steps=20
        validation_mse,
        'b--', # color blue, broken line
        label='Validation')
ax.set_xticks(np.arange(20)[::2]); ax.set_ylim([50000,250000]); ax.legend(); mse_fig
```

![](Figures/hitters_37_0.png)

# 4. Ridge Regression

- **Ridge Regression:**
  
  - Minimiert $\sum_{i=1}^{n} \left( y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij} \right)^2 + \lambda \sum_{j=1}^{p} \beta_j^2 = \text{RSS} + \lambda \sum_{j=1}^{p} \beta_j^2$.
  - $\lambda \geq 0$ muss separat bestimmt werden.
  - Zwei Kriterien:
    - RSS klein halten.
    - Schrumpfungsstrafe $\lambda \sum_{j} \beta_j^2$.
  - $\lambda$ kontrolliert die Balance:
    - $\lambda = 0$: Gleich wie Least Squares.
    - $\lambda \to \infty$: Schätzungen nähern sich null.
  - Verschiedene Schätzungen $\hat{\beta}^{\lambda}_R$ für jede $\lambda$.
  - Auswahl eines guten $\lambda$ ist entscheidend.

- **Schrumpfungsstrafe:**
  
  - Gilt für $\beta_1, \ldots, \beta_p$, nicht für $\beta_0$.
  - $\beta_0$ misst den Mittelwert bei $x_{i1} = x_{i2} = \ldots = x_{ip} = 0$.
  - Bei zentrierten Daten: $\hat{\beta}_0 = \bar{y} = \sum_{i=1}^{n} y_i / n$.

- **ℓ₂ Norm:**
  
  - Die ℓ₂-Norm eines Vektors: die Quadratwurzel der Summe der Quadrate seiner Einträge.
  - In Ridge Regression:
    - Die ℓ₂-Norm der Ridge-Regressionskoeffizienten: $\sum_{j=1}^{p} \beta_j^2$.
    - Diese Norm wird verwendet, um die Schrumpfungsstrafe zu berechnen.

- **Warum ist Ridge Regression besser als Least Squares?**
  
  - Bias-Variance Trade-off: $\lambda$ erhöhen -> Flexibilität und Varianz sinken, Bias steigt.
  
  - Ridge Regression vs Least Squares:
    
    - Least Squares: Niedriger Bias, hohe Varianz.
    - Ridge Regression: Geringere Varianz, etwas höherer Bias.
    - Bei $p \approx n$: Least Squares hat hohe Varianz.
    - Bei $p > n$: Least Squares nicht eindeutig, Ridge Regression funktioniert gut.
  
  - Computationale Vorteile:
    
    - Best Subset Selection: $2^p$ Modelle, oft unpraktikabel.
    - Ridge Regression: Ein Modell pro $\lambda$.

## 4.1. Ridge Regression für ein λ

- Drei Optionen für Ridge Regression:
  - `skl.ElasticNet()` aus `sklearn.linear_model` mit `l1_ratio=0` für Ridge Regression.
  - `skl.Ridge()` aus `sklearn.linear_model`.
  - `skl.ridge_regression` aus `sklearn.linear_model`.
- `l1_ratio` Werte:
  - `0`: Entspricht Ridge Regression.
  - `1`: Entspricht Lasso Regression.
  - `0 < l1_ratio < 1`: Hybridmodell.
- Parameter $\lambda$ wird in `sklearn` als `alphas` bezeichnet.

```python=
lambdas = 10**np.linspace(8, -2, 100) / Y.std()
```

- $\lambda$-Werte von $10^{8}$ bis $10^{-2}$ erstellen, skaliert durch die Standardabweichung von y.
- **Gründe für die Skalierung:**
1. Normierung des Regularisierungseffekts:
   - Verlustfunktion = RSS + Regularisierungsterm
     - Große y-Skala: Große RSS, kleinerer Effekt des Regularisierungsterms.
     - Kleine y-Skala: Kleine RSS, größerer Effekt des Regularisierungsterms.
   - Skalierung des Regularisierungsterms mit der Standardabweichung von y sorgt für einen konsistenten Effekt auf die Verlustfunktion, unabhängig von der y-Skala.
2. Vergleichbarkeit zwischen Datensätzen:
3. Stabilität bei der Modellauswahl:

```python=
Xs = X - X.mean(0)[None,:]
# X.mean(0) is the mean of each column.
# X.mean(0)[None,:] is to make it a row vector.
X_scale = X.std(0)
Xs = Xs / X_scale[None,:]
```

```python=
ridge_ElasticNet = skl.ElasticNet(alpha=lambdas[59], l1_ratio=0).fit(Xs, Y)
ridge_ElasticNet.coef_
```

    /opt/conda/lib/python=3.11/site-packages/sklearn/linear_model/_coordinate_descent.py:697: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check...
    array([-13.413,  59.911,  -5.296,  29.658,  22.029,  46.246, -11.475,
            22.718,  51.622,  43.255,  52.365,  54.56 ,  -8.126,  17.743,
           -53.458,  62.501,  10.395, -16.702,   0.558])

```python=
ridge_Ridge = skl.Ridge(alpha=lambdas[59]).fit(Xs, Y)
ridge_Ridge.coef_
```

    array([-286.928,  325.593,   30.738,  -49.645,  -20.462,  131.999,
            -24.46 , -337.102,  112.513,    8.919,  420.473,  218.463,
           -204.046,   31.265,  -59.038,   78.769,   51.901,  -22.747, -12.963])

```python=
skl.ridge_regression(Xs, Y, alpha=lambdas[59])
```

    array([-286.928,  325.593, ...

**Mit pipeline**

- `pipe.fit(X, Y)` aktualisiert das `ridge`-Objekt:
  - Fügt Attribute wie `coef_` hinzu.

```python=
ridge = skl.Ridge(alpha=lambdas[59])
scaler = StandardScaler(with_mean=True,  with_std=True)
pipe = Pipeline(steps=[('scaler', scaler), ('ridge', ridge)])
pipe.fit(X, Y)
ridge.coef_
```

    array([-286.928,  325.593, ...

## 4.2. Ridge Regression für Mehrere λ

```python=
# Ignore UserWarning
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
```

- Standardmäßig wählt`ElasticNet.path`  automatisch eine $\lambda$-Spanne:
  - Von maximalem $\lambda$ (alle Koeffizienten null) bis minimalem $\lambda$ (kaum Regularisierung).
- ℓ₂-Regularisierung: Kein endlicher $\lambda$-Wert macht alle Koeffizienten null.

```python=
# Fit 100 Ridge regressions
soln_array = skl.ElasticNet.path(Xs, Y, l1_ratio=0., alphas=lambdas)[1]
# Shape of solution soln_array: 19 x 100 (coefficient x lambda)
# Coefficients at the 2nd lambda
soln_array[:,1]
```

    array([ 1.009e-03,  1.122e-03,  8.775e-04,  1.074e-03,
            1.150e-03,  1.135e-03,  1.025e-03,  1.346e-03,
            1.404e-03,  1.343e-03,  1.439e-03,  1.450e-03,
            1.253e-03, -3.653e-05, -4.925e-04,  7.687e-04,
            6.507e-05, -1.382e-05, -7.245e-06])

```python=
path_fig, ax = plt.subplots(figsize=(12,6))
soln_path = pd.DataFrame(soln_array.T, columns=D.columns, index=-np.log(lambdas))
soln_path.plot(ax=ax, legend=False)
# Shape must be lambda x coefficient
ax.set_xlabel(r'$-\log(\lambda)$'); ax.set_ylabel('Standardized coefficients')
ax.legend(loc='upper left');
```

![](Figures/hitters_58_0.png)

- Großes $\lambda$: kleinere Koeffizienten -> kleineren ℓ₂-Norm und umgekehrt.

```python=
beta_hat = soln_path.iloc[39]
lambdas[39], np.linalg.norm(beta_hat)
```

    (25.535, 24.171)

```python=
beta_hat = soln_path.loc[soln_path.index[59]]
lambdas[59], np.linalg.norm(beta_hat)
```

    (0.244, 160.424)

## 4.3. Validierungsfehler schätzen

- Daten in Trainings- (50%) und Validierungssatz (50%) aufteilen.
- Ridge-Regression mit $\lambda = 0.01$ auf Trainingssatz anpassen.
- MSE auf Validierungssatz berechnen.

```python=
validation = skm.ShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
ridge = skl.ElasticNet(alpha=0.01, l1_ratio=0)
results = skm.cross_validate(ridge, X, Y, cv=validation,
                             scoring='neg_mean_squared_error')
-results['test_score']
```

    array([134214.004])

```python=
ridge = skl.Ridge(alpha=0.01)
results = skm.cross_validate(ridge, X, Y, cv=validation
                             scoring='neg_mean_squared_error')
-results['test_score']
```

    array([134596.762])

## 4.4. Finden λ mit Validierungssatz

- `GridSearchCV()` aus `sklearn.model_selection`.

```python=
param_grid = {'ridge_alpha': lambdas}
ridge = skl.ElasticNet(l1_ratio=0)
scaler = StandardScaler(with_mean=True,  with_std=True)
pipe = Pipeline(steps=[('scaler', scaler), ('ridge', ridge)])
grid = skm.GridSearchCV(pipe, param_grid, cv=validation,
                        scoring='neg_mean_squared_error')
grid.fit(X, Y);
```

```python=
# grid.best_params_ shows the best lambda of 0.0059
best_model = grid.best_estimator_
best_model.named_steps['ridge'].coef_
```

    array([-257.254,  278.754,   12.654,  -19.894,  -4.443,  119.746,  -44.113, -178.083,
            127.260,   48.003,  278.299,  141.610,-173.705,   30.842,  -60.784,   78.399,
             45.043,  -24.238,  -13.827])

## 4.5. Finden λ mit Kreuzvalidierung

```python=
kfold = skm.KFold(n_splits=5, shuffle=True, random_state=0)
param_grid = {'ridge__alpha': lambdas}
ridge = skl.ElasticNet(l1_ratio=0)
scaler = StandardScaler(with_mean=True,  with_std=True)
pipe = Pipeline(steps=[('scaler', scaler), ('ridge', ridge)])
grid = skm.GridSearchCV(pipe, param_grid, cv=kfold,
                        scoring='neg_mean_squared_error')
grid.fit(X, Y);
```

```python=
# grid.best_params shows the best lambda of 0.0119
best_model = grid.best_estimator_
best_model.named_steps['ridge'].coef_
```

    array([[-222.809,  238.772,    3.211,   -2.931,    3.649,  108.910],
           [ -50.819, -105.157,  122.007,   57.186,  210.352,  118.057],
           [-150.220,   30.366,  -61.625,   77.738,   40.074,  -25.022]])

- Kreuzvalidiertes MSE als Funktion von $-\log(\lambda)$ plotten.
- Schrumpfung nimmt von links nach rechts ab.

```python=
ridge_fig, ax = plt.subplots(figsize=(8,8))
ax.errorbar( x=-np.log(lambdas), y=-grid.cv_results_['mean_test_score'],
            yerr=grid.cv_results_['std_test_score'] / np.sqrt(5) )
# grid.cv_results_['mean_test_score']: mean of five negative MSEs for each lambda
# grid.cv_results_['std_test_score']: standard deviation of the mean
ax.set_xlabel(r'$-\log(\lambda)$'); ax.set_ylabel('CV MSE');
```

![](Figures/hitters_78_0.png)

- `GridSearchCV()` verwenden, um bestes $\lambda$ mit $R^2$ als Kriterium zu finden.
  - $R^2$ ist das Standard-Kriterium in `GridSearchCV()`.

```python=
# With R²
grid_r2 = skm.GridSearchCV(pipe, param_grid, cv=kfold).fit(X, Y)
r2_fig, ax = plt.subplots(figsize=(8,8))
ax.errorbar(-np.log(lambdas), grid_r2.cv_results_['mean_test_score'],
            yerr=grid_r2.cv_results_['std_test_score'] / np.sqrt(K))
ax.set_xlabel(r'$-\log(\lambda)$'); ax.set_ylabel('Cross-validated $R^2$');
```

![](Figures/hitters_80_0.png)

## 4.6. Schnelle Kreuzvalidierung für Lösungswege

- `CV`-Versionen.
- Unterschiede zwischen `GridSearchCV` und `CV` wegen Standardisierung.
  - `GridSearchCV`:
    - Mittelwert und Standardabweichung in jedem Trainingssatz berechnet.
    - Diese Standardisierungsparameter für Trainings- und Validierungssatz genutzt.
  - `CV`:
    - Mittelwert und Standardabweichung im gesamten Datensatz berechnet.
    - Diese Standardisierungsparameter für Trainings- und Validierungssatz genutzt.
  - `GridSearchCV`:
    - Keine Datenlecks, aber langsamer.
  - `CV`:
    - Schneller, aber mögliches Datenleck.
  - Bei großen und stabilen Datensätzen ist der Unterschied meist gering.

```python=
# Fit 100 Ridge regressions
ridge_CV = skl.RidgeCV(alphas=lambdas, store_cv_results = True)
ridge_CV.fit(Xs, Y)
# The coefficients of the best model
ridge_CV.coef_
```

    array([-222.195, 238.110,  3.077,  -2.687,   3.764,  108.725, -50.882, 
           -104.209, 121.879, 57.264, 209.427, 117.786, -149.829,  30.357,  
            -61.635,  77.725, 39.993, -25.031, -13.677])

```python=
ridgeCV = skl.ElasticNetCV(alphas=lambdas, l1_ratio=0, cv=kfold)
pipeCV = Pipeline(steps=[('scaler', scaler), ('ridge', ridgeCV)]).fit(X, Y)
# Plotting
ridgeCV_fig, ax = plt.subplots(figsize=(8,8))
ax.errorbar(-np.log(lambdas), ridgeCV.mse_path_.mean(1),
            yerr=ridgeCV.mse_path_.std(1) / np.sqrt(K))
ax.axvline(-np.log(ridgeCV.alpha_), c='r', ls='--') # | at the best λ
ax.set_ylim([50000,250000])
ax.set_xlabel(r'$-\log(\lambda)$'); ax.set_ylabel('Cross-validated MSE');
```

![](Figures/hitters_82_0.png)

## 4.7. Testfehler der Kreuzvalidierten Ridge-Regression bewerten

- Daten in Trainingssatz (75%) und Testsatz (25%) aufteilen.
- Ridge-Regression mit 5-facher Kreuzvalidierung auf Trainingssatz abstimmen.
- Modell auf Testsatz evaluieren.

```python=
outer_valid = skm.ShuffleSplit(n_splits=1, test_size=0.25, random_state=1)
inner_cv = skm.KFold(n_splits=5, shuffle=True, random_state=2)
ridgeCV = skl.ElasticNetCV(alphas=lambdas, l1_ratio=0, cv=inner_cv)
pipeCV = Pipeline(steps=[('scaler', scaler), ('ridge', ridgeCV)])
results = skm.cross_validate(pipeCV, X, Y, cv=outer_valid,
                             scoring='neg_mean_squared_error')
-results['test_score']
```

    array([132393.840])

# 5. The Lasso

- **Nachteil von Ridge Regression:**
  
  - Beinhaltet alle $p$ Prädiktoren im finalen Modell.
  - Schrumpft Koeffizienten, setzt aber keine auf null (außer bei $\lambda = \infty$).

- **Lasso Regression:**  - Minimiert $\sum_{i=1}^{n} \left( y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij} \right)^2 + \lambda \sum_{j=1}^{p} |\beta_j| = \text{RSS} + \lambda \sum_{j=1}^{p} |\beta_j|$.
  
  - Verwendet $\ell_1$-Strafe statt ℓ₂-Strafe.
  - ℓ₁-Norm: $\|\beta\|_1 = \sum |\beta_j|$.
  - **Vorteil:**
    - Schrumpft Koeffizienten auf null bei großem $\lambda$.
    - Führt zu variablenselektiven Modellen -> sparsame Modelle

```python=
kfold = skm.KFold(n_splits=5, shuffle=True, random_state=0)
lassoCV = skl.ElasticNetCV(alphas=lambdas, l1_ratio=1, cv=kfold)
pipeCV = Pipeline(steps=[('scaler', scaler), ('lasso', lassoCV)]).fit(X, Y)
lassoCV.alpha_
```

    3.147

```python=
lassoCV2 = skl.LassoCV(n_alphas=100, cv=kfold)
pipeCV2 = Pipeline(steps=[('scaler', scaler), ('lasso', lassoCV2)]).fit(X, Y)
lassoCV2.alpha_
```

    3.147

```python=
soln_array = skl.ElasticNet.path(Xs, Y, l1_ratio=1, n_alphas=100)[1]
# lambdas, soln_array = skl.Lasso.path(Xs, Y, l1_ratio=1, n_alphas=100)[:2]
# Shape of soln_array: 19 x 100
```

```python=
path_fig, ax = plt.subplots(figsize=(8,8))
soln_path = pd.DataFrame(soln_array.T, columns=D.columns, index=-np.log(lambdas))
soln_path.plot(ax=ax, legend=False)
ax.legend(loc='upper left')
ax.set_xlabel(r'$-\log(\lambda)$'); ax.set_ylabel('Standardized coefficiients');
```

![](Figures/hitters_96_0.png)

- Bestes $\lambda$ und zugehöriges Test-MSE anzeigen.
- Vergleich: bestes $\lambda$ und Test-MSE des Ridge-Modells: 0.012 und 117408.574.

```python=
lassoCV.alpha_, lassoCV.mse_path_.mean(1)[59]
```

    3.147, 115411.106

```python=
lassoCV.coef_
```

    array([[-210.010,  243.455,    0.000,    0.000,    0.000,   97.694],
           [ -41.523,   -0.000,    0.000,   39.623,  205.753,  124.555],
           [-126.300,   15.703,  -59.502,   75.246,   21.627,  -12.044]])

```python=
lassoCV_fig, ax = plt.subplots(figsize=(8,8))
ax.errorbar(-np.log(lassoCV.alphas_), lassoCV.mse_path_.mean(1), 
            yerr=lassoCV.mse_path_.std(1) / np.sqrt(K))
ax.axvline(-np.log(lassoCV.alpha_), c='r', ls='--')
ax.set_ylim([50000,250000])
ax.set_xlabel(r'$-\log(\lambda)$'); ax.set_ylabel('CV MSE');
```

![](Figures/hitters_100_0.png)

- Koeffizienten des Modells mit bestem $\lambda$ anzeigen.
- Beim Lasso sind 6 von 19 Koeffizienten genau null; das Modell umfasst nur 13 Variablen.

## Vergleich Ridge, Lasso

- **Alternativen Darstellung:**
  
  - Lasso:
    
    - Minimiert $\sum_{i=1}^{n} \left( y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij} \right)^2$.
    - Einschränkung: $\sum_{j=1}^{p} |\beta_j| \leq s$.
    - Kleinster RSS innerhalb eines Diamanten ($| \beta_1 | + | \beta_2 | \leq s$ bei $p = 2$).
    - Budget $s$ bestimmt die Größe der Koeffizienten.
  
  - Ridge Regression:
    
    - Minimiert $\sum_{i=1}^{n} \left( y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij} \right)^2$.
    - Einschränkung: $\sum_{j=1}^{p} \beta_j^2 \leq s$.
    - Kleinster RSS innerhalb eines Kreises ($\beta_1^2 + \beta_2^2 \leq s$ bei $p = 2$).

- **Verbindung zu Best Subset Selection:**
  
  - Problem: Minimiert $\sum_{i=1}^{n} \left( y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij} \right)^2$.
  - Einschränkung: $\sum_{j=1}^{p} I(\beta_j \neq 0) \leq s$.
  - $I(\beta_j \neq 0)$ ist eine Indikatorvariable (1 wenn $\beta_j \neq 0$, sonst 0).
  - Ziel: Kleinster RSS mit maximal $s$ nicht-null Koeffizienten.
  - Best Subset Selection ist rechnerisch aufwendig bei großen $p$.  

- **Lasso und Ridge Regression als Alternativen:**
  
  - Ersetzbar durch Lasso und Ridge Regression für einfachere Berechnungen.
  - Lasso ähnelt Best Subset Selection mehr, da es Feature Selection durchführt.

- **Die Variableauswahl-Eigenschaft des Lasso:**
  
  - Ellipsen und Einschränkungen
    - Ellipsen um $\hat{\beta}$ repräsentieren RSS-Konturen.
    - Lasso und Ridge Regression Schätzungen sind der Punkt, an dem eine Ellipse die Einschränkung zuerst berührt.
    - Ridge Regression hat keine scharfen Ecken, daher sind die Koeffizienten nicht null.
    - Lasso hat Ecken an den Achsen, daher können einige Koeffizienten null sein.
  - Höhere Dimensionen
    - Für $p = 3$ ist die Einschränkung für Ridge Regression eine Kugel, für Lasso ein Polyeder.
    - Für $p > 3$ wird Ridge Regression zu einer Hypersphäre und Lasso zu einem Polytop.
    - Lasso führt zu Feature-Auswahl aufgrund der scharfen Ecken.

- **Allgemeine Anwendung**
  
  - Weder Lasso noch Ridge Regression dominiert immer.
  - Lasso performt besser bei wenigen relevanten Prädiktoren.
  - Ridge Regression performt besser bei vielen Prädiktoren ähnlicher Größe.
  - Cross-Validation kann helfen, die beste Methode zu bestimmen.

- **Effizienz**: Effiziente Algorithmen existieren für beide Methoden.

- **Einfache Spezialfälle:**
  
  - Betrachte $n = p$ und $X$ als diagonale Matrix.
  - Schätzungen
    - Least Squares-Lösung: $\hat{\beta}_j = y_j$.
    - Ridge Regression: $\hat{\beta}^R_j = \frac{y_j}{1 + \lambda}$.
    - Lasso: $\hat{\beta}^L_j = \begin{cases} y_j - \lambda/2 & \text{if } y_j > \lambda/2 \\ y_j + \lambda/2 & \text{if } y_j < -\lambda/2 \\ 0 & \text{if } |y_j| \leq \lambda/2 \end{cases}$.
  - Schrumpfung
    - Ridge Regression schrumpft alle Koeffizienten gleichmäßig.
    - Lasso schrumpft Koeffizienten um einen konstanten Betrag $\lambda/2$; kleine Koeffizienten werden zu null.

- **Bayesianische Interpretation:**
  
  - Bayesianische Sichtweise
    - Ridge Regression und Lasso können durch bayesianische Prioren interpretiert werden.
    - Ridge Regression: Prior ist eine normale Verteilung.
    - Lasso: Prior ist eine doppelt-exponentielle Verteilung.
  - Posterior-Modi
    - Ridge Regression-Lösung ist der Posterior-Modus bei normaler Prior.
    - Lasso-Lösung ist der Posterior-Modus bei doppelt-exponentieller Prior.
  - Prior-Formen
    - Gaussian-Prior ist flacher und breiter bei null.
    - Lasso-Prior ist steil bei null, erwartet viele Koeffizienten als null.

# 6. Hauptkomponentenregression

- **Dimension Reduction Methods**
  
  - Verwendung von transformierten Prädiktoren
  - Anpassung eines Kleinste-Quadrate-Modells mit transformierten Variablen

- **Definition von Transformierten Prädiktoren**
  
  - $Z_1, Z_2, \ldots, Z_M$ sind lineare Kombinationen der originalen Prädiktoren $X_1, X_2, \ldots, X_p$
  - Berechnung: $Z_m = \sum_{j=1}^{p} \phi_{jm} X_j$
  - $Var(Z_1)$ ist am größten. $\phi'_{1}\phi_{1}=1$
  - $Var(Z_2)$ ist zweit größten. $\phi'_{2}\phi_{2}=1$ und $\phi'_{2}\phi_{1}=0$

- **Lineares Regressionsmodell**
  
  - Modell: $y_i = \theta_0 + \sum_{m=1}^{M} \theta_m z_{im} + \epsilon_i$
  - Reduzierung der Dimension von $p + 1$ auf $M + 1$

- **Nutzen der Dimension Reduktion**
  
  - Reduziert Varianz der geschätzten Koeffizienten
  - Besser bei großen $p$ im Verhältnis zu $n$
  - Kein Dimension Reduction bei $M = p$ und linearen Unabhängigkeit der $Z_m$

- **Vorteile von PCR**
  
  - Kleine Anzahl von Hauptkomponenten kann meiste Variabilität erklären
  - Vermeidung von Überanpassung durch Schätzung von nur $M \ll p$ Koeffizienten

- **Ergebnisse**
  
  - Abnahme des Bias und Zunahme der Varianz mit steigender Anzahl der Hauptkomponenten
  - Verbesserung gegenüber Kleinste-Quadrate-Anpassung vor allem bei geringem $M$

- **Einschränkungen von PCR**
  
  - Keine Merkmalsselektion, da jede Hauptkomponente eine lineare Kombination aller ursprünglichen Merkmale ist
  - Ähnlichkeit zu Ridge Regression

- **Kreuzvalidierung und Standardisierung**
  
  - Wahl der Anzahl der Hauptkomponenten $M$ durch Kreuzvalidierung
  - Standardisierung der Prädiktoren empfohlen, um gleiche Skala zu gewährleisten

```python=
pca = PCA(n_components=2)
linreg = skl.LinearRegression()
pipe = Pipeline([('pca', pca), ('linreg', linreg)])
pipe.fit(X, Y)
pipe.named_steps['linreg'].coef_
```

    array([0.098, 0.476])

- Standardisierungsschritt zur Pipeline hinzufügen und Modell neu anpassen.

```python=
pipe = Pipeline([('scaler', scaler),('pca', pca),('linreg', linreg)])
pipe.fit(X, Y)
pipe.named_steps['linreg'].coef_
```

    array([106.369, 21.604])

- `explained_variance_ratio_` verwenden, um den Prozentsatz der Varianz zu zeigen, der von jeder Komponente erklärt wird.

```python=
pipe.named_steps['pca'].explained_variance_ratio_
```

    array([0.383, 0.218])

- `GridSearchCV()` verwenden, um die Anzahl der Komponenten durch Variation des `n_components` Parameters auszuwählen.

```python=
param_grid = {'pca_n_components': range(1, 20)}
pca = PCA()
linreg = skl.LinearRegression()
pipe_grid = Pipeline([('pca', pca),('linreg', linreg)])
grid_pca = skm.GridSearchCV(pipe_grid, param_grid, cv=kfold,
                            scoring='neg_mean_squared_error')
grid_pca.fit(X, Y);
```

```python=
pcr_fig, ax = plt.subplots(figsize=(8,8))
# Set number of principal components
n_comp = range(1, 20)
# Figure
ax.errorbar(n_comp, -grid_pca.cv_results_['mean_test_score'],
            grid_pca.cv_results_['std_test_score'] / np.sqrt(K) )
ax.set_ylabel('Cross-validated MSE'); ax.set_xlabel('# principal components')
ax.set_xticks(n_comp); ax.set_ylim([50000,250000]);
```

![](Figures/hitters_111_0.png)

- 13 Komponenten haben den kleinsten Kreuzvalidierungsfehler, aber 4 Komponenten liefern einen ähnlichen Fehler. Modell mit weniger Komponenten wählen.
- Kreuzvalidiertes MSE für die beste Anzahl an Komponenten anzeigen.

```python=
print("Test MSE of 13:", -grid_pca.best_score_)
print("Test MSE of 4 :", -grid_pca.cv_results_['mean_test_score'][3])
```

Test MSE of 13: 121830.867
Test MSE of 4 : 122692.823

```python=
best_pca = grid_pca.best_estimator_
best_n_components = grid_pca.best_params_['pca_n_components']
best_pca.named_steps['linreg'].coef_
```

    array([[ 0.098,  0.476,  0.261,  0.806],
           [-0.115,  0.134,  1.450, -0.003],
           [ 4.049,  3.133,  1.290,  4.719]])

- `PCA()` erlaubt `n_components=0` nicht, daher berechnen wir das MSE für das Nullmodell selbst.

```python=
# Create a column of 263 zeros
Xn = np.zeros((X.shape[0], 1))
# Fit the null model
linreg = skl.LinearRegression()
cv_null = skm.cross_validate(linreg, Xn, Y, cv=kfold, 
                             scoring='neg_mean_squared_error')
# Show the test MSE
-cv_null['test_score'].mean()
```

    204139.307

# 7. Regression der partiellen kleinsten Quadrate

- **PCR Ansatz:**
  
  - Identifiziert lineare Kombinationen der Prädiktoren $X_1, \ldots, X_p$.
  - Unüberwachte Methode: Die Antwort $Y$ wird nicht zur Bestimmung der Hauptkomponenten verwendet.
  - Nachteil: Richtungen, die die Prädiktoren am besten erklären, sind nicht unbedingt die besten für die Vorhersage der Antwort.

- **PLS Ansatz:**
  
  - Überwachte Alternative zu PCR.
  - Reduziert Dimensionen und identifiziert neue Merkmale $Z_1, \ldots, Z_M$, die lineare Kombinationen der ursprünglichen Merkmale sind.
  - Nutzt die Antwort $Y$, um neue Merkmale zu identifizieren, die sowohl die Prädiktoren als auch die Antwort gut erklären.

- **Bestimmung $Z_1$:**
  
  - Standardisieren der *p* Prädiktoren.
  - Setzen jedes $\phi_{j1}$ auf den Koeffizienten der einfachen linearen Regression von *Y* auf $X_j$. Wir führen p Regression durch, um p $\phi_{j1}$ zu kakulieren.
  - Dieser Koeffizient ist proportional zur Korrelation zwischen *Y* und $X_j$.
  - Berechnung von $Z_1 = \sum_{j=1}^{p} \phi_{j1} X_j$ priorisiert Variablen mit der stärksten Verbindung zur Antwortvariable.

- **Berechnung $Z_2$:**
  
  - Regression jedes $X_j$ auf $Z_1$ und Berechnung der Residuen. Dies entfernt die Informationen in jedem Prädiktor, die bereits durch $Z_1$ erfasst wurden.
  - Verwendung dieser Residuen als neuen Datensatz. Diese Residuen repräsentieren die verbleibenden Informationen, die von $Z_1$ nicht erklärt wurden.
  - Berechnung von $Z_2$ aus diesen Residuen auf die gleiche Weise wie $Z_1$ aus den ursprünglichen Daten. Dies bedeutet:
    - Standardisieren der Residualwerte für jeden Prädiktor.
    - Durchführung einfacher linearer Regressionen der Antwort *Y* auf jede Menge standardisierter Residuen.
    - Verwendung der Koeffizienten aus diesen Regressionen als $\phi_{j2}$ Werte: $Z_2 = \sum_{j=1}^{p} \phi_{j2} X_j$.
    - $Z_2$ erfasst unabhängige Informationen von $Z_1$, da die Residuen orthogonal sind.
  - Wiederhole den Prozess *M* Mal, um $Z_1, …, Z_M$ zu bestimmen.

- **Beziehung der Koeffizienten in einem PLS-Modell:**
  
  - $\beta_j = \sum_{m=1}^{M} \theta_m \phi_{jm}$
    - **$\beta_j$**: Regressionskoeffizient für den _j_-ten Prädiktor ($X_j$).
    - **$\theta_m$**: Regressionskoeffizient für die _m_-te PLS-Komponente ($Z_m$), zeigt die Beziehung zu Y.
    - **$\phi_{jm}$**: Gewicht von $X_j$ auf $Z_m$. Bestimmt den Beitrag jedes Prädiktors zur PLS-Komponente.
  - Bestimme die PLS-Komponenten ($Z_m$) und passe ein lineares Modell an, um Y vorherzusagen. $\theta_m$ sind die Koeffizienten.
  - Kombiniere $\theta_m$ mit den Gewichten ($\phi_{jm}$), um die finalen Regressionskoeffizienten ($\beta_j$) zu berechnen.

- **Anzahl der PLS-Richtungen (M)**
  
  - Tuning-Parameter, typischerweise durch Kreuzvalidierung gewählt.
  - Standardisierung der Prädiktoren und der Antwort vor der Durchführung von PLS empfohlen.

- **Praktische Anwendung**
  
  - Beliebt in der Chemometrie.
  - Performt oft nicht besser als Ridge Regression oder PCR.
  - Reduktion des Bias durch überwachtes Dimension Reduction, aber potenzielle Erhöhung der Varianz.

**ANWENDUNG**

- `n_components` gibt die Anzahl der latenten Variablen an.
- Die Anzahl der Koeffizienten entspricht der Anzahl der Merkmale in den Eingabedaten X, nicht der Anzahl der Komponenten.

```python=
pls = PLSRegression(n_components=2, scale=True)
pls.fit(X, Y)
pls.coef_
```

    array([[ 0.183,  0.968,  1.434,  1.416,  1.192,  1.958],
           [ 1.869,  0.012,  0.052,  0.368,  0.105,  0.109],
           [ 0.076, 35.962, -95.900,  0.201,  0.029, -0.655],
           [30.214]])

- Wie bei PCR verwenden wir CV, um die optimale Anzahl der Komponenten zu bestimmen.

```python=
param_grid = {'n_components':range(1, 20)}
pls = PLSRegression(n_components=2, scale=True)
grid_pls = skm.GridSearchCV(pls, param_grid, cv=kfold,
                            scoring='neg_mean_squared_error').fit(X, Y)
```

```python=
pls_fig, ax = plt.subplots(figsize=(8,8))
n_comp = range(1, 20)
ax.errorbar(n_comp,
            -grid_pls.cv_results_['mean_test_score'],
            grid_pls.cv_results_['std_test_score'] / np.sqrt(K) )
ax.set_ylabel('Cross-validated MSE'); ax.set_xlabel('# principal components')
ax.set_xticks(n_comp); ax.set_ylim([50000,250000]);
```

![](Figures/hitters_124_0.png)

- Der CV-Fehler ist bei 12 Komponenten minimal, aber der Leistungsunterschied zu 2 oder 3 Komponenten ist gering.
- Das beste kreuzvalidierte MSE von PCA beträgt 121830.867.

```python=
# Retrieve the best cross-validation score (negative MSE)
print("Best test MSE:", -grid_pls.best_score_)
```

    Best test MSE: 114684.613

```python=
# Retrieve the best estimator
best_pls = grid_pls.best_estimator_
best_pls.coef_
```

    array([[-2.330,  8.263,  2.897, -1.191, -1.113,  6.244],
           [-9.474, -0.069,  0.224,  0.967,  0.782,  0.361],
           [-0.742, 47.789, -121.742,  0.283,  0.316, -2.627],
           [-15.132]])

# 8. Betrachtungen bei hoher Dimension

- **a. Hochdimensionale Daten**
  
  - Traditionelle statistische Techniken sind für niedrige Dimensionen gedacht ($n \gg p$).
  - Neue Technologien haben die Datenerhebung verändert; oft ist $p$ sehr groß, aber $n$ ist begrenzt.
  - Beispiele:
    - Vorhersage des Blutdrucks mit 500.000 SNPs und $n \approx 200$.
    - Analyse von Online-Einkaufsverhalten mit vielen Suchbegriffen und $n \approx 1.000$.

- **b. Probleme bei hoher Dimension**
  
  - Klassische Methoden wie die Kleinste-Quadrate-Regression sind ungeeignet, wenn $p \geq n$.
  - Überanpassung: Perfekte Anpassung an Trainingsdaten, aber schlechte Leistung bei Testdaten.
  - Beispiel:
    - Bei $n = 20$ und $p = 1$ passt die Regression nicht perfekt.
    - Bei $n = 2$ und $p = 1$ passt die Regression perfekt, was zu Überanpassung führt.
  - $R^2$ und Trainings-MSE sind irreführend; Test-MSE zeigt wahre Modellqualität.

- **c. Regression bei hoher Dimension**
  
  - Techniken wie Vorwärtsauswahl, Ridge-Regression, Lasso und Hauptkomponenten-Regression sind nützlich.
  - Beispiel Lasso:
    - Bei $p = 20, 50, 2.000$ und $n = 100$ zeigt die Test-MSE die Bedeutung der Regularisierung.
    - Höhere Dimensionen führen zu höherem Testfehler, wenn zusätzliche Merkmale nicht relevant sind.
  - „Fluch der Dimensionalität“: Mehr Merkmale führen nicht immer zu besseren Modellen.

- **d. Interpretation der Ergebnisse bei hoher Dimension**
  
  - Multikollinearität ist extrem; keine eindeutigen prädiktiven Variablen.
  - Beispiel SNPs und Blutdruck:
    - Vorwärtsauswahl wählt 17 SNPs, aber andere Sets könnten ebenfalls gut sein.
    - Ergebnisse sollten nicht überbewertet werden; weitere Validierung notwendig.
