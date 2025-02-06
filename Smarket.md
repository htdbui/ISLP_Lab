---
title: "Stock Market "
author: "db"
---

# 1. Datenüberblick

- Tägliche prozentuale Renditen für den S&P 500 Aktienindex zwischen 2001 und 2005.
- Es gibt 1250 Zeilen.
- Es gibt 9 Variablen:
  - Jahr: Datum der Beobachtung
  - Lag1-Lag5: Tägliche Renditen der letzten 5 Tage
  - Volumen: Tägliches Handelsvolumen in Milliarden am Vortag
  - Heute: Heutige Rendite
  - Richtung: Positive ("Up") oder negative ("Down") Rendite

# 2. Packages und Daten

```python=
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import statsmodels.api as sm
from ISLP.models import (ModelSpec as MS, summarize, contrast)
from ISLP import (load_data, confusion_table)
from sklearn.discriminant_analysis import \
 (LinearDiscriminantAnalysis as LDA,
  QuadraticDiscriminantAnalysis as QDA)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
```

```python=
Smarket = load_data('Smarket')
Smarket.head()
```

<table>
  <thead>
<tr>
  <th></th>
  <th>Year</th>
  <th>Lag1</th>
  <th>Lag2</th>
  <th>Lag3</th>
  <th>Lag4</th>
  <th>Lag5</th>
  <th>Volume</th>
  <th>Today</th>
  <th>Direction</th>
</tr>
  </thead>
  <tbody>
<tr>
  <th>0</th>
  <td>2001</td>
  <td>0.381</td>
  <td>-0.192</td>
  <td>-2.624</td>
  <td>-1.055</td>
  <td>5.010</td>
  <td>1.1913</td>
  <td>0.959</td>
  <td>Up</td>
</tr>
<tr>
  <th>1</th>
  <td>2001</td>
  <td>0.959</td>
  <td>0.381</td>
  <td>-0.192</td>
  <td>-2.624</td>
  <td>-1.055</td>
  <td>1.2965</td>
  <td>1.032</td>
  <td>Up</td>
</tr>
<tr>
  <th>2</th>
  <td>2001</td>
  <td>1.032</td>
  <td>0.959</td>
  <td>0.381</td>
  <td>-0.192</td>
  <td>-2.624</td>
  <td>1.4112</td>
  <td>-0.623</td>
  <td>Down</td>
</tr>
<tr>
  <th>3</th>
  <td>2001</td>
  <td>-0.623</td>
  <td>1.032</td>
  <td>0.959</td>
  <td>0.381</td>
  <td>-0.192</td>
  <td>1.2760</td>
  <td>0.614</td>
  <td>Up</td>
</tr>
<tr>
  <th>4</th>
  <td>2001</td>
  <td>0.614</td>
  <td>-0.623</td>
  <td>1.032</td>
  <td>0.959</td>
  <td>0.381</td>
  <td>1.2057</td>
  <td>0.213</td>
  <td>Up</td>
</tr>
  </tbody>
</table>

```python=
Smarket.describe().round(1)
```

<table>
  <thead>
<tr>
  <th></th>
  <th>Year</th>
  <th>Lag1</th>
  <th>Lag2</th>
  <th>Lag3</th>
  <th>Lag4</th>
  <th>Lag5</th>
  <th>Volume</th>
  <th>Today</th>
</tr>
  </thead>
  <tbody>
<tr>
  <th>mean</th>
  <td>2003.0</td>
  <td>0.0</td>
  <td>0.0</td>
  <td>0.0</td>
  <td>0.0</td>
  <td>0.0</td>
  <td>1.5</td>
  <td>0.0</td>
</tr>
<tr>
  <th>std</th>
  <td>1.4</td>
  <td>1.1</td>
  <td>1.1</td>
  <td>1.1</td>
  <td>1.1</td>
  <td>1.1</td>
  <td>0.4</td>
  <td>1.1</td>
</tr>
<tr>
  <th>min</th>
  <td>2001.0</td>
  <td>-4.9</td>
  <td>-4.9</td>
  <td>-4.9</td>
  <td>-4.9</td>
  <td>-4.9</td>
  <td>0.4</td>
  <td>-4.9</td>
</tr>
<tr>
  <th>25%</th>
  <td>2002.0</td>
  <td>-0.6</td>
  <td>-0.6</td>
  <td>-0.6</td>
  <td>-0.6</td>
  <td>-0.6</td>
  <td>1.3</td>
  <td>-0.6</td>
</tr>
<tr>
  <th>50%</th>
  <td>2003.0</td>
  <td>0.0</td>
  <td>0.0</td>
  <td>0.0</td>
  <td>0.0</td>
  <td>0.0</td>
  <td>1.4</td>
  <td>0.0</td>
</tr>
<tr>
  <th>75%</th>
  <td>2004.0</td>
  <td>0.6</td>
  <td>0.6</td>
  <td>0.6</td>
  <td>0.6</td>
  <td>0.6</td>
  <td>1.6</td>
  <td>0.6</td>
</tr>
<tr>
  <th>max</th>
  <td>2005.0</td>
  <td>5.7</td>
  <td>5.7</td>
  <td>5.7</td>
  <td>5.7</td>
  <td>5.7</td>
  <td>3.2</td>
  <td>5.7</td>
</tr>
  </tbody>
</table>

```python=
Smarket.Direction.value_counts(normalize=True)
```

    Direction
    Up    0.5184
    Down  0.4816
    Name: proportion, dtype: float64

```python=
Smarket.corr(numeric_only=True).round(2)
```

<table>
  <thead>
<tr>
  <th></th>
  <th>Year</th>
  <th>Lag1</th>
  <th>Lag2</th>
  <th>Lag3</th>
  <th>Lag4</th>
  <th>Lag5</th>
  <th>Volume</th>
  <th>Today</th>
</tr>
  </thead>
  <tbody>
<tr>
  <th>Year</th>
  <td>1.00</td>
  <td>0.03</td>
  <td>0.03</td>
  <td>0.03</td>
  <td>0.04</td>
  <td>0.03</td>
  <td>0.54</td>
  <td>0.03</td>
</tr>
<tr>
  <th>Lag1</th>
  <td>0.03</td>
  <td>1.00</td>
  <td>-0.03</td>
  <td>-0.01</td>
  <td>-0.00</td>
  <td>-0.01</td>
  <td>0.04</td>
  <td>-0.03</td>
</tr>
<tr>
  <th>Lag2</th>
  <td>0.03</td>
  <td>-0.03</td>
  <td>1.00</td>
  <td>-0.03</td>
  <td>-0.01</td>
  <td>-0.00</td>
  <td>-0.04</td>
  <td>-0.01</td>
</tr>
<tr>
  <th>Lag3</th>
  <td>0.03</td>
  <td>-0.01</td>
  <td>-0.03</td>
  <td>1.00</td>
  <td>-0.02</td>
  <td>-0.02</td>
  <td>-0.04</td>
  <td>-0.00</td>
</tr>
<tr>
  <th>Lag4</th>
  <td>0.04</td>
  <td>-0.00</td>
  <td>-0.01</td>
  <td>-0.02</td>
  <td>1.00</td>
  <td>-0.03</td>
  <td>-0.05</td>
  <td>-0.01</td>
</tr>
<tr>
  <th>Lag5</th>
  <td>0.03</td>
  <td>-0.01</td>
  <td>-0.00</td>
  <td>-0.02</td>
  <td>-0.03</td>
  <td>1.00</td>
  <td>-0.02</td>
  <td>-0.03</td>
</tr>
<tr>
  <th>Volume</th>
  <td>0.54</td>
  <td>0.04</td>
  <td>-0.04</td>
  <td>-0.04</td>
  <td>-0.05</td>
  <td>-0.02</td>
  <td>1.00</td>
  <td>0.01</td>
</tr>
<tr>
  <th>Today</th>
  <td>0.03</td>
  <td>-0.03</td>
  <td>-0.01</td>
  <td>-0.00</td>
  <td>-0.01</td>
  <td>-0.03</td>
  <td>0.01</td>
  <td>1.00</td>
</tr>
  </tbody>
</table>

- Verzögerte Renditen korrelieren kaum mit der heutigen Rendite.
- Signifikante Korrelation zwischen Jahr und Volumen. Volumen steigt von 2001 bis 2005.

```python=
Smarket.plot(y='Volume');
```

<img title="" src="Figures/Smarket_8_0.png" alt="" width="322">

# 3. Logistische Regression

- Verwenden Sie `sm.GLM()` oder `sm.Logit()` aus dem `statsmodels`-Paket für die logistische Regression von `Direction` mit `Lag1` bis `Lag5` und `Volume`.
- Bei `sm.GLM()` geben wir `family=sm.families.Binomial()` an. Die Syntax ähnelt `sm.OLS()`.

```python=
# Design matrix
allvars = Smarket.columns.drop(['Today', 'Direction', 'Year'])
design = MS(allvars)
X = design.fit_transform(Smarket)
y = Smarket.Direction == 'Up'
# Logistic model
glm = sm.GLM(y, X, family=sm.families.Binomial()).fit()
summarize(results)
```

<table>
  <thead>
<tr>
  <th></th>
  <th>coef</th>
  <th>std err</th>
  <th>z</th>
  <th>P>|z|</th>
</tr>
  </thead>
  <tbody>
<tr>
  <th>intercept</th>
  <td>-0.1260</td>
  <td>0.241</td>
  <td>-0.523</td>
  <td>0.601</td>
</tr>
<tr>
  <th>Lag1</th>
  <td>-0.0731</td>
  <td>0.050</td>
  <td>-1.457</td>
  <td>0.145</td>
</tr>
<tr>
  <th>Lag2</th>
  <td>-0.0423</td>
  <td>0.050</td>
  <td>-0.845</td>
  <td>0.398</td>
</tr>
<tr>
  <th>Lag3</th>
  <td>0.0111</td>
  <td>0.050</td>
  <td>0.222</td>
  <td>0.824</td>
</tr>
<tr>
  <th>Lag4</th>
  <td>0.0094</td>
  <td>0.050</td>
  <td>0.187</td>
  <td>0.851</td>
</tr>
<tr>
  <th>Lag5</th>
  <td>0.0103</td>
  <td>0.050</td>
  <td>0.208</td>
  <td>0.835</td>
</tr>
<tr>
  <th>Volume</th>
  <td>0.1354</td>
  <td>0.158</td>
  <td>0.855</td>
  <td>0.392</td>
</tr>
  </tbody>
</table>

- Kleinster p-Wert für `Lag1`, negativer Koeffizient zeigt, dass ein positiver Rückgang gestern die heutige Anstiegswahrscheinlichkeit verringert.
- p-Wert von 0,15 zeigt keine starke Verbindung zwischen `Lag1` und `Direction`.
- `predict()` Methode schätzt die Anstiegswahrscheinlichkeit basierend auf Prädiktorwerten. Ohne Datensatz nutzt es Trainingsdaten.

```python=
probs = results.predict(); probs[:5].round(3)
```

    array([0.507, 0.481, 0.481, 0.515, 0.511])

- Konvertiere Wahrscheinlichkeiten in "Up" oder "Down" Labels basierend auf über oder unter 0,5.

```python=
labels = np.array(['Down']*1250)
labels[probs>0.5] = "Up"
```

- `confusion_table()`: Zeigt richtige/falsche Klassifizierungen.
- Adaptiert von sklearn.metrics: transponiert Matrix, fügt Labels hinzu, nimmt vorhergesagte Labels zuerst.

```python=
confusion_table(labels, Smarket.Direction)
```

<table>
  <thead>
<tr>
  <th>Predicted\Truth</th>
  <th>Down</th>
  <th>Up</th>
</tr>
  </thead>
  <tbody>
<tr>
  <th>Down</th>
  <td>145</td>
  <td>141</td>
</tr>
<tr>
  <th>Up</th>
  <td>457</td>
  <td>507</td>
</tr>
  </tbody>
</table>

- Diagonale der Konfusionsmatrix: richtige Vorhersagen.
- Modell: 507 Tage aufwärts, 145 Tage abwärts, 652 korrekte Vorhersagen.
- Genauigkeit mit `np.mean()`: 52,2%.

```python=
(507+145)/1250, np.mean(labels == Smarket.Direction)
# (0.5216, 0.5216)
```

- 52,2% Genauigkeit des Modells irreführend, da Trainings- und Testdaten identisch sind. Testfehler wird unterschätzt.
- Besser: Modell mit Teil der Daten trainieren und mit zurückgehaltenen Daten testen.

```python=
train = (Smarket.Year < 2005)
Smarket_train = Smarket.loc[train]
Smarket_test = Smarket.loc[~train]
# Shape of Smarket_test: 252 x 9
```

- `train` Boolescher Vektor: 1.250 Elemente (`True` vor 2005, `False` 2005).
- Logistische Regression auf Daten vor 2005, Wahrscheinlichkeiten für 2005 (252 Beobachtungen) vorhersagen.

```python=
X_train, X_test = X.loc[train], X.loc[~train]
y_train = y.loc[train]
glm_train = sm.GLM(y_train, X_train, family=sm.families.Binomial()).fit()
probs = glm_train.predict(exog=X_test)
labels = np.array(['Down']*252)
labels[probs>0.5] = 'Up'
```

```python=
D = Smarket.Direction
L_test = D.loc[~train]
confusion_table(labels, L_test)
```

<table>
  <thead>
<tr>
  <th>Predicted\Truth</th>
  <th>Down</th>
  <th>Up</th>
</tr>
<tr>
  <th>Down</th>
  <td>77</td>
  <td>97</td>
</tr>
<tr>
  <th>Up</th>
  <td>34</td>
  <td>44</td>
</tr>
  </tbody>
</table>

```python=
np.mean(labels == L_test), np.mean(labels != L_test)
# (0.480, 0.520)
```

- Testgenauigkeit: ca. 48%
- Fehlerquote: ca. 52%
- 52% Testfehlerquote ist schlechter als zufälliges Raten.

```python=
L_test.value_counts(normalize=True)
```

    Direction
    Up   0.56
    Down 0.44
    Name: proportion, dtype: float64  

- Entfernen schwacher Prädiktoren könnte helfen.
- Modell neu anpassen mit nur `Lag1` und `Lag2`, den stärksten Prädiktoren.

```python=
model_l1l2 = MS(['Lag1', 'Lag2']).fit(Smarket)
X = model_l1l2.transform(Smarket)
X_train, X_test = X.loc[train], X.loc[~train]
glm_train = sm.GLM(y_train, X_train, family=sm.families.Binomial()).fit()
probs = glm_train.predict(exog=X_test)
labels = np.array(['Down']*252)
labels[probs>0.5] = 'Up'
confusion_table(labels, L_test)
```

<table>
  <thead>
<tr>
  <th>Predicted\Truth</th>
  <th>Down</th>
  <th>Up</th>
</tr>
<tr>
  <th>Down</th>
  <td>35</td>
  <td>35</td>
</tr>
<tr>
  <th>Up</th>
  <td>76</td>
  <td>106</td>
</tr>
  </tbody>
</table>

- Bewerten:
  - Gesamtgenauigkeit.
  - Genauigkeit für Tage mit vorhergesagtem Anstieg.

```python=
(35+106)/252, 106/(106+76)
# (0.5595, 0.5824)
```

- Gesamtgenauigkeit is 56%. Modell sagt 56% der Bewegungen korrekt voraus.
- Bei vorhergesagtem Anstieg, mögliche Handelsstrategie erreicht die Genauigkeit 58% Genauigkeit .
- Vorhersage für `Lag1`: [1.2, 1.1] und `Lag2`: [1.5, -1.8]

```python=
newdata = pd.DataFrame({'Lag1':[1.2, 1.5], 'Lag2':[1.1, -0.8]})
newX = model_l1l2.transform(newdata)
results.predict(newX)
```

    00.479
    10.496
    dtype: float64

# 4. Lineare Diskriminanzanalyse

## 4.1. Theorie

- Logistische Regression modelliert die Wahrscheinlichkeit eines Ergebnisses (Y) basierend auf Prädiktorvariablen (X) mit der logistischen Funktion.
- Eine andere Methode schätzt Wahrscheinlichkeiten, indem sie die Verteilung der Prädiktoren (X) für jedes Ergebnis modelliert und den Satz von Bayes verwendet.
- Wenn die Prädiktoren normalverteilt sind, ähnelt diese Methode der logistischen Regression.
- Andere Methoden statt logistische Regression sind nötig, weil:
  - Instabile Ergebnisse bei gut getrennten Klassen.
  - Bei kleinen Stichproben mit normalverteilten Prädiktoren können genauere Methoden existieren.
  - Einige Methoden handhaben mehr als zwei Ergebnisklassen.
- $\pi_k$ repräsentiert die priori Wahrscheinlichkeit, dass eine Beobachtung zur k-ten Klasse gehört. Zur Schätzung von $\pi_k$ berechnen wir den Anteil der Beobachtungen im Trainingssatz, die in die k-te Klasse fallen.
- $f_k(X) \equiv \Pr(X | Y = k)$ repräsentiert die Dichtefunktion von X für die k-te Klasse. $f_k(x)$ ist groß, wenn eine Beobachtung in der k-ten Klasse wahrscheinlich $X \approx x$ hat, und klein, wenn es unwahrscheinlich ist.
- $p_k(x) = \Pr(Y = k | X = x)$ ist die posteriori Wahrscheinlichkeit, dass eine Beobachtung $X = x$ zur k-ten Klasse gehört.
  $\Pr(Y = k | X = x) = \frac{\pi_k f_k(x)}{\sum_{\ell=1}^{K} \pi_\ell f_\ell(x)}$
- Zur Schätzung von $f_k(x)$ müssen wir vereinfachende Annahmen treffen.

**LDA für p = 1**

- Annahme: $f_k(x)$ ist normalverteilt.
  $f_k(x) = \sqrt{\frac{1}{2\pi\sigma_k^2}} \exp \left( -\frac{1}{2\sigma_k^2} (x - \mu_k)^2 \right)$
- Annahme: $\sigma_1^2 = \cdots = \sigma_K^2 = \sigma^2$
- Die posteriori Wahrscheinlichkeit:
  $p_k(x) = \frac{\pi_k \sqrt{\frac{1}{2\pi\sigma}} \exp \left( -\frac{1}{2\sigma^2} (x - \mu_k)^2 \right)}{\sum_{l=1}^{K} \pi_l \sqrt{\frac{1}{2\pi\sigma}} \exp \left( -\frac{1}{2\sigma^2} (x - \mu_l)^2 \right)}$
  - Bayes-Klassifikator: Beobachtung wird der Klasse mit größtem $p_k(x)$ zugeordnet.
- Logarithmieren und Umstellen von $p_k(x)$: 
  $\delta_k(x) = \frac{x \cdot \mu_k}{\sigma^2} - \frac{\mu_k^2}{2\sigma^2} + \log(\pi_k)$ 
  - Beobachtung wird der Klasse mit größtem $\delta_k(x)$ zugeordnet.
  - ${\hat{\delta}}_k(x)$ ist eine lineare Funktion von $x$ → *Lineare* Diskriminanzfunktion.
  - Beispiel: $K = 2$, $\pi_1 = \pi_2$ $\Rightarrow$ Bayes-Klassifikator: Klasse 1, wenn $2x (\mu_1 - \mu_2) > \mu_1^2 - \mu_2^2$, sonst Klasse 2.
  - Bayes-Entscheidungsgrenze: $\delta_1(x) = \delta_2(x)$ $\Rightarrow$ $x = \frac{\mu_1 + \mu_2}{2}$.
  - Abbildung 4.4: $n_1 = n_2 = 20$, $\hat{\pi}_1 = \hat{\pi}_2$ $\Rightarrow$ Entscheidungsgrenze: $(\hat{\mu}_1 + \hat{\mu}_2)/2$
    - LDA-Fehlerrate: 11,1 %, Bayes-Fehlerrate: 10,6 %

<img title="" src="Figures\Theo_4.4.png" alt="Abbildung 4.4" width="882">

- Schätzer von $\mu_1,\ldots,\mu_k,\ \pi_1,\ldots,\pi_K,\ \sigma^2$:
  ${\hat{\mu}}_k\ =\frac{1}{n_k}\sum_{i:y_i=k} x_i$
  ${\hat{\sigma}}^2=\frac{1}{n-K}\sum_{k=1}^{K}\sum_{i:y_i=k}\left(x_i-{\hat{\mu}}_k\right)^2$
  ${\hat{\pi}}_k=\frac{n_k}{n}$

**LDA für p > 1**

- Annahme: $X = (X_1, X_2, \ldots, X_p)$ ist multivariat normalverteilt, mit klassenspezifischem Mittelwert und gemeinsamer Kovarianzmatrix.
  - Die multivariate Gauß-Verteilung nimmt an, dass jeder Prädiktor einer eindimensionalen Normalverteilung folgt.
  - Die Oberfläche hat eine Glockenform, wenn $\mathrm{var}(X_1) = \mathrm{var}(X_2)$ und $\mathrm{cor}(X_1, X_2)=0$. Diese Form wird verzerrt, wenn Prädiktoren korreliert sind oder ungleiche Varianzen haben.
  - $f(\mathbf{x}) = \frac{1}{(2\pi)^\frac{p}{2} \left|\mathbf{\Sigma}\right|^\frac{1}{2}} \exp{\left(-\frac{1}{2} (\mathbf{x} - \mathbf{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \mathbf{\mu})\right)}$
- LDA-Klassifikator für $p > 1$: Annahme, dass Beobachtungen der $k$-ten Klasse multivariat normalverteilt sind, $N(\mu_k, \Sigma)$
- Bayes-Klassifikator: $\delta_k(x) = x^T \Sigma^{-1} \mu_k - \frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k + \log \pi_k$
- Bayes-Entscheidungsgrenzen: $\delta_k(x) = \delta_\ell(x)$ für $k \neq \ell$
- Beispiel mit drei Klassen.
  - Beobachtungen jeder Klasse stammen aus einer multivariaten Gauß-Verteilung mit $p = 2$, klassenspezifischem Mittelwertvektor und gemeinsamer Kovarianzmatrix.
  - Links: Ellipsen zeigen 95 % der Wahrscheinlichkeit für jede der drei Klassen. Die gestrichelten Linien sind die Bayes-Entscheidungsgrenzen.
  - Rechts: 20 Beobachtungen wurden aus jeder Klasse generiert. Die LDA-Entscheidungsgrenzen sind mit durchgezogenen schwarzen Linien markiert. Die Bayes-Entscheidungsgrenzen sind erneut als gestrichelte Linien dargestellt.

<img title="" src="Figures\Theo_4.6.png" alt="Abbildung 4.6" width="866">

## 4.2. Anwendung

- Da `LDA` automatisch einen Interzept hinzufügt, entfernen wir die Interzept-Spalte aus `X_train` und `X_test`.
- Wir verwenden die Labels direkt anstelle von Boolean-Vektoren `y_train`.

```python=
X_train, X_test = [M.drop(columns=['intercept']) for M in [X_train, X_test]]
lda = LDA(store_covariance=True).fit(X_train, L_train);
```

- In `sklearn` zeigt ein nachgestelltes `_` Werte, die durch `fit()` geschätzt wurden.
- `means_` liefert den Durchschnittswert jedes Prädiktors für jede Klasse.
  - Es zeigt, dass Renditen der letzten zwei Tage vor Marktanstiegen negativ und vor Marktabschwüngen positiv sind.
  - Dieses Ergebnis können wir selbst berechnen.

```python=
lda.means_
```

    array([[ 0.04279022,  0.03389409],
           [-0.03954635, -0.03132544]])

```python=
df = pd.concat([X_train, L_train], axis=1)
df.groupby('Direction', observed=True)[['Lag1', 'Lag2']].mean()
```

<table>
  <thead>
<tr>
  <th>Direction\</th>
  <th>Lag1</th>
  <th>Lag2</th>
</tr>
<tr>
  <th>Down</th>
  <td>0.042790</td>
  <td>0.033894</td>
</tr>
<tr>
  <th>Up</th>
  <td>-0.039546</td>
  <td>-0.031325</td>
</tr>
  </tbody>
</table>

- Das Attribut `priors_` speichert die a-priori Wahrscheinlichkeiten.
- `classes_` ordnet diese Wahrscheinlichkeiten den Labels zu.

```python=
lda.priors_
# array([0.492, 0.508])
```

```python=
lda.classes_
# array(['Down', 'Up'], dtype='<U4')
```

- Das Ergebnis zeigt $\hat\pi_{Down}=0.492$ und $\hat\pi_{Up}=0.508$.
- Wir können dies selbst berechnen.

```python=
L_train.value_counts(normalize=True)
```

    Direction
    Up    0.508
    Down  0.492
    Name: proportion, dtype: float64

- Bei LDA teilen alle Klassen (Down und Up) die gleiche Kovarianzstruktur.
- sklearn’s LDA schätzt eine gemeinsame Kovarianzmatrix für das gesamte Dataset.
- `lda.covariance_` enthält die gepoolte Kovarianzmatrix der Merkmale.
  - [ [var(lag1), cov(lag1, lag2)],
    [cov(lag1, lag2), var(lag2)] ]

```python=
lda.covariance_
```

    array([[ 1.50886781, -0.03340234],
           [-0.03340234,  1.5095363 ]])

- `lda.scalings_` liefert lineare Diskriminanzvektoren.
- Jede Spalte projiziert Merkmale auf eine neue Achse, die Klassen trennt.
- Bei k Klassen gibt es bis zu k-1 Vektoren.
- Bei binärer Klassifikation (k=2) gibt es einen Vektor.
  - Diese Werte sind die Multiplikatoren für `Lag1` und `Lag2` in der LDA-Entscheidungsregel.

```python=
lda.scalings_
```

    array([[-0.642],
           [-0.513]])

- Wenn `-0.642 * Lag1 - 0.513 * Lag2` groß ist, prognostiziert LDA einen Anstieg; wenn klein, einen Rückgang.

```python=
lda_pred = lda.predict(X_test)
confusion_table(lda_pred, L_test)
```

<table>
  <thead>
<tr>
  <th>Predicted\Truth</th>
  <th>Down</th>
  <th>Up</th>
</tr>
<tr>
  <th>Down</th>
  <td>35</td>
  <td>35</td>
</tr>
<tr>
  <th>Up</th>
  <td>76</td>
  <td>106</td>
</tr>
  </tbody>
</table>

- LDA und Logistic liefern dasselbe Ergebnis.
- Wir schätzen die Klassenwahrscheinlichkeiten für jeden Trainingspunkt.
- Ein 50%-Schwellenwert auf diesen Wahrscheinlichkeiten rekonstruiert die Vorhersagen in `lda_pred`.

```python=
lda_prob = lda.predict_proba(X_test)
lda_prob[:3]
```

    array([[0.49017925, 0.50982075],
           [0.4792185 , 0.5207815 ],
           [0.46681848, 0.53318152]])

```python=
np.all(np.where(lda_prob[:,1] >= 0.5, 'Up','Down') == lda_pred),
np.all([lda.classes_[i] for i in np.argmax(lda_prob, 1)] == lda_pred)
# (True, True)
```

```python=
max(lda_prob[:,0])
# 0.5202
```

- Höchste Abnahmewahrscheinlichkeit: 52,02%.
- `sklearn` LDA-Muster: Klassifikator erstellen, mit `fit()` anpassen, mit `predict()` vorhersagen.

# 5. Quadratische Diskriminanzanalyse

## 5.1. Theorie

- LDA: Beobachtungen aus multivariater Gaußverteilung, gleiche Kovarianzmatrix für alle Klassen.
- QDA: Beobachtungen aus Gaußverteilung, jede Klasse eigene Kovarianzmatrix $\Sigma_k$.
- Unterschied LDA vs. QDA: Bias-Varianz-Kompromiss.
  - LDA: gemeinsame Kovarianzmatrix, weniger flexibel, niedrigere Varianz, gut bei wenigen Trainingsbeobachtungen.
  - QDA: separate Kovarianzmatrizen, höhere Varianz, besser bei großen Trainingssätzen.
- $\delta_k(x) = -\frac{1}{2}(x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) - \frac{1}{2} \log |\Sigma_k| + \log \pi_k$
  - Bayes-Klassifikator ordnet Klasse mit größtem $\delta_k(x)$ zu.
- Beispiel: 
  - Links: Die Bayes- (gestrichelt), LDA- (gepunktet) und QDA- (durchgezogen) Entscheidungsgrenzen für ein Zweiklassenproblem mit $\Sigma_1 = \Sigma_2$. Die Schattierung zeigt die QDA-Entscheidungsregel. Da die Bayes-Entscheidungsgrenze linear ist, wird sie von LDA genauer approximiert als von QDA.
  - Rechts: Details wie im linken Panel, außer dass $\Sigma_1 \neq \Sigma_2$. Da die Bayes-Entscheidungsgrenze nicht linear ist, wird sie von QDA genauer approximiert als von LDA.

<img title="" src="Figures\Theo_4.9.png" alt="Abbildung 4.9" width="783">

## 5.2. Anwendung

- Wir verwenden `QuadraticDiscriminantAnalysis()` abgekürzt als `QDA()`. Syntax ist ähnlich wie bei `LDA()`.

```python=
qda = QDA(store_covariance=True).fit(X_train, L_train);
```

- `qda.means_` und `qda.priors_` liefern dieselben Ergebnisse wie `lda.means_` und `lda.priors_`, weil sie nur von den Trainingsdaten abhängen.
- Der `QDA()`-Klassifikator schätzt eine Kovarianz pro Klasse.
- Kovarianz für die erste Klasse "Down":

```python=
qda.covariance_[0]
```

    array([[ 1.50662277, -0.03924806],
           [-0.03924806,  1.53559498]])

- `qda.scalings_` enthält Skalierungsfaktoren pro Klasse. 
- Erstes Array: erste Klasse, zweites Array: zweite Klasse.
- QDA erzeugt keine linearen Diskriminanzrichtungen wie LDA, da jede Klasse ihre eigene Kovarianzmatrix hat.
- LDA projiziert Daten auf lineare Richtungen, die Klassen trennen. Diese Richtungen sind in `lda.scalings_` gespeichert.
- QDA nutzt klassenabhängige Kovarianz und quadratische Entscheidungsgrenzen.
- QDA hat keine sinnvolle `scalings_`-Attribut.

```python=
qda.scalings_
# [array([1.56294495, 1.47927279]), array([1.53455065, 1.47272326])]
```

```python=
qda_pred = qda.predict(X_test)
confusion_table(qda_pred, L_test)
```

<table>
  <thead>
<tr>
  <th>Predicted\Truth</th>
  <th>Down</th>
  <th>Up</th>
</tr>
<tr>
  <th>Down</th>
  <td>30</td>
  <td>20</td>
</tr>
<tr>
  <th>Up</th>
  <td>81</td>
  <td>121</td>
</tr>
  </tbody>
</table>

```python=
np.mean(qda_pred == L_test)
# 0.599
```

- QDA-Vorhersagen sind fast 60% genau. Es ist beeindruckend.
- Dies deutet darauf hin, dass QDA Beziehungen besser erfassen kann als LDA und logistische Regression.

# 6. Naive Bayes

## 6.1. Theorie

- Schätzt $\pi_1, \ldots, \pi_K$ als Anteil der Trainingsbeobachtungen pro Klasse.
- Schätzt $f_1(x), \ldots, f_K(x)$ unter Annahme unabhängiger Prädiktoren innerhalb jeder Klasse.
  - $f_k(x) = f_{k1}(x_1) \times f_{k2}(x_2) \times \cdots \times f_{kp}(x_p)$
  - Schätzung von $f_{kj} \equiv Pr(X_j \mid Y = k): 
    - Quantitativ: (X_j \mid Y = k) \sim N(\mu_{jk}, \sigma_{jk}^2)$ oder mit nichtparametrischen Methoden (Histogramm, Kernel Density Estimator).
    - Qualitativ: Anteil der Trainingsbeobachtungen für jede Klasse.
- Vereinfachung durch Unabhängigkeitsannahme: keine Berücksichtigung der gemeinsamen Verteilung der Prädiktoren.
- Naive Bayes führt oft zu guten Ergebnissen, besonders bei kleinen n im Vergleich zu p
- Beispiel: Klassifikation mit $p = 3$ Prädiktoren und $K = 2$ Klassen.
  - Die ersten beiden Prädiktoren sind quantitativ, der dritte Prädiktor ist qualitativ mit drei Stufen.
  - Angenommen, $\hat{\pi}_1 = \hat{\pi}_2 = 0.5$.
  - Wenn die a priori Wahrscheinlichkeiten für die beiden Klassen gleich sind, hat $x^* = (0.4, 1.5, 1)^T$ eine a posteriori Wahrscheinlichkeit von 94,4 %, zur ersten Klasse zu gehören.

<img title="" src="Figures\Theo_4.10.png" alt="Abbildung 4.10" width="458">

## 6.2. Anwendung

- - Naive-Bayes-Modell mit `GaussianNB()` auf `Smarket`-Daten angepasst, ähnlich wie `LDA()` und `QDA()`.
- Standardmäßig Gaußsche Verteilung, Kerndichtemethode auch möglich.

```python=
NB = GaussianNB().fit(X_train, L_train);
NB.classes_
# array(['Down', 'Up'], dtype='<U4')
```

- `NB.theta_` und `NB.class_prior_` liefern die gleichen Ergebnisse wie `lda.means_` und `lda.priors_` sowie `qda.means_` und `qda.priors_`.
- `NB.var_` liefert die Varianz jedes Prädiktors für jede Klasse.
- `GaussianNB` berechnet Varianzen mit der 1/n-Formel.

```python=
NB.var_
```

    array([[1.504, 1.533],
           [1.514, 1.487]])

```python=
df.groupby('Direction', observed=True)[['Lag1', 'Lag2']].var(ddof=0)
```

<table>
  <thead>
<tr>
  <th>Direction\</th>
  <th>Lag1</th>
  <th>Lag2</th>
</tr>
<tr>
  <th>Down</th>
  <td>1.504</td>
  <td>1.533</td>
</tr>
<tr>
  <th>Up</th>
  <td>1.514</td>
  <td>1.487</td>
</tr>
  </tbody>
</table>

```python=
nb_labels = NB.predict(X_test)
confusion_table(nb_labels, L_test)
```

<table>
  <thead>
<tr>
  <th>Predicted\Truth</th>
  <th>Down</th>
  <th>Up</th>
</tr>
  <th>Down</th>
  <td>29</td>
  <td>20</td>
</tr>
<tr>
  <th>Up</th>
  <td>82</td>
  <td>121</td>
</tr>
  </tbody>
</table>

- Naive Bayes sagt mit 59% Genauigkeit voraus, etwas schlechter als QDA, aber besser als LDA.
- `predict_proba()` von Naive Bayes schätzt, ähnlich wie bei LDA, die Klassenwahrscheinlichkeiten.

```python=
NB.predict_proba(X_test)[:3]
```

    array([[0.4873288 , 0.5126712 ],
           [0.47623584, 0.52376416],
           [0.46529531, 0.53470469]])

# 7. K-Nächste-Nachbarn

```python=
knn1 = KNeighborsClassifier(n_neighbors=1).fit(X_train, L_train)
knn1_pred = knn1.predict(X_test)
confusion_table(knn1_pred, L_test)
```

<table>
  <thead>
<tr>
  <th>Predicted\Truth</th>
  <th>Down</th>
  <th>Up</th>
</tr>
<tr>
  <th>Down</th>
  <td>43</td>
  <td>58</td>
</tr>
<tr>
  <th>Up</th>
  <td>68</td>
  <td>83</td>
</tr>
  </tbody>
</table>

```python=
(83+43)/252, np.mean(knn1_pred == L_test)
# (0.5, 0.5)
```

- K=1 liefert nur (43+83)/252 = 50% Genauigkeit, wahrscheinlich wegen zu flexibler Anpassung.
- Mit K=3 wiederholen wir die Analyse.

```python=
knn3 = KNeighborsClassifier(n_neighbors=3).fit(X_train, L_train)
knn3_pred = knn3.predict(X_test)
np.mean(knn3_pred == L_test)
# 0.532
```

- Die Ergebnisse verbesserten sich leicht, aber weiteres Erhöhen von K bringt keinen Nutzen. QDA schneidet am besten ab.
- KNN ist auf `Smarket`-Daten unterlegen, liefert aber oft anderswo beeindruckende Ergebnisse.

# 8. Vergleich

## Analytischer Vergleich

- Ziel: Klasse maximieren, die $\Pr(Y = k \mid X = x)$ maximiert

**LDA:**

- Annahme: Prädiktoren folgen multivariater Normalverteilung
- $\log \left( \frac{\Pr(Y = k \mid X = x)}{\Pr(Y = K \mid X = x)} \right) = a_k + \sum_{j=1}^{p} b_{kj} x_j$
- $\log \left( \frac{\Pr(Y = k \mid X = x)}{\Pr(Y = K \mid X = x)} \right)$ ist linear in $x$

**QDA:**

- Annahme: Prädiktoren folgen multivariater Normalverteilung mit klassen-spezifischer Kovarianzmatrix
- $\log \left( \frac{\Pr(Y = k \mid X = x)}{\Pr(Y = K \mid X = x)} \right) = a_k + \sum_{j=1}^{p} b_{kj} x_j + \sum_{j=1}^{p} \sum_{l=1}^{p} c_{kjl} x_j x_l$
- $\log \left( \frac{\Pr(Y = k \mid X = x)}{\Pr(Y = K \mid X = x)} \right)$ ist quadratisch in $x$

**Naive Bayes:**

- Annahme: Prädiktoren sind innerhalb jeder Klasse unabhängig
- $f_k(x)$ als Produkt von $p$ eindimensionalen Funktionen $f_{kj}(x_j)$
- $\log \left( \frac{\Pr(Y = k \mid X = x)}{\Pr(Y = K \mid X = x)} \right) = a_k + \sum_{j=1}^{p} g_{kj}(x_j)$

**Beobachtungen:**

- LDA ist Spezialfall von QDA mit $c_{kjl} = 0$
- Jeder Klassifikator mit linearer Entscheidungsgrenze ist Spezialfall von naive Bayes
- Naive Bayes mit $f_{kj}(x_j) \sim N(\mu_{kj}, \sigma_j^2)$ ist Spezialfall von LDA mit diagonalem $\Sigma$
- Weder QDA noch naive Bayes sind Spezialfälle des anderen

**Logistische Regression:**

- $\log \left( \frac{\Pr(Y = k \mid X = x)}{\Pr(Y = K \mid X = x)} \right) = \beta_{k0} + \sum_{j=1}^{p} \beta_{kj} x_j$
- Linear in $x$, wie LDA
- LDA besser bei normalverteilten Prädiktoren, logistische Regression besser bei Verstoß gegen Normalitätsannahme

**K-Nearest Neighbors (KNN):**

- Non-parametrischer Ansatz, keine Annahmen über Entscheidungsgrenze
- Dominiert LDA und logistische Regression bei hoch nicht-linearen Entscheidungsgrenzen, wenn $n$ groß und $p$ klein ist
- Erfordert viele Beobachtungen relativ zu Prädiktoren ($n$ viel größer als $p$)
- QDA bevorzugt bei nicht-linearen Entscheidungsgrenzen und moderatem $n$ oder nicht kleinem $p$
- KNN gibt keine wichtigen Prädiktoren an, im Gegensatz zur logistischen Regression

## Empirischer Vergleich

- Sechs Szenarien mit binärer Klassifikation und 2 quantiativen Prädiktore
- Drei Szenarien (1,2,3) mit linearer Entscheidungsgrenze, drei (4,5,6) mit nicht-linearer Entscheidungsgrenze

**Szenario 1:**

- 20 Trainingsbeobachtungen pro Klasse, unkorrelierte Normalverteilung
- LDA hat gut abgeschnitten, da es dieses Modell annimmt.
- Logistische Regression war auch gut, da sie eine lineare Entscheidungsgrenze annimmt.
- KNN schnitt schlecht ab wegen hoher Varianz ohne Bias-Reduktion.
- QDA schnitt schlechter als LDA ab, da es zu flexibel war.
- Naive Bayes war etwas besser als QDA, da die Annahme unabhängiger Prädiktoren korrekt ist.

**Szenario 2:**

- Wie Szenario 1, aber Korrelation zwischen zwei Prädiktore innerhalb der Klassen von -0.5
  - Die t-Verteilung neigt dazu, extremere Punkte als die Normalverteilung zu liefern.
- Ähnliche Leistung wie in Szenario 1, aber naive Bayes schneidet schlecht ab, weil die Annahme verletzt wird.

**Szenario 3:**

- Wie Szenario 2, aber t-Verteilung, 50 Beobachtungen pro Klasse
- Beobachtungen stammen nicht aus einer Normalverteilung → Annahme von LDA und QDA verletzt.
- Logistische Regression übertrifft LDA.
- QDA schneidet wegen Nicht-Normalität schlechter ab.
- Naive Bayes schneidet schlecht ab wegen verletzter Unabhängigkeitsannahme.

**Szenario 4:**

- Daten wurden aus einer Normalverteilung generiert, mit einer Korrelation von 0.5 zwischen den Prädiktoren in der ersten Klasse und einer Korrelation von -0.5 zwischen den Prädiktoren in der zweiten Klasse.
- Diese Einstellung entspricht der QDA-Annahme und führt zu quadratischen Entscheidungsgrenzen.
- QDA am besten, naive Bayes schlecht wegen verletzter Unabhängigkeitsannahme.

**Szenario 5:**

- Daten stammen aus einer Normalverteilung mit unkorrelierten Prädiktoren.
- Logistische Regression wurde auf eine komplizierte nichtlineare Funktion der Prädiktoren angewendet.
- QDA und Naive Bayes schneiden besser ab.
- KNN mit K = 1 liefert schlechte Ergebnisse wegen mangelnder Glättung.
- KNN mit K, gewählt durch Kreuzvalidierung, liefert das beste Ergebnis dank seiner Flexibilität.

**Szenario 6:**

- Daten stammen aus einer Normalverteilung mit unterschiedlicher diagonaler Kovarianzmatrix für jede Klasse.
- Kleine Stichprobengröße: n = 6 pro Klasse.
- Naive Bayes schneidet gut ab, da seine Annahmen erfüllt sind.
- LDA und logistische Regression schneiden schlecht ab wegen nichtlinearer Entscheidungsgrenze (ungleiche Kovarianzmatrizen).
- QDA und KNN leiden unter kleiner Stichprobe

**Fazit:**

- Kein Verfahren dominiert immer
- Lineare Entscheidungsgrenzen: LDA und logistische Regression gut
- Moderat nicht-lineare Grenzen: QDA oder naive Bayes besser
- Komplexe Grenzen: Nicht-parametrische Methoden wie KNN besser, aber Wahl der Glattheit wichtig
- Flexible Versionen durch Transformationen der Prädiktoren möglich (z.B. $X^2$, $X^3$, $X^4$)