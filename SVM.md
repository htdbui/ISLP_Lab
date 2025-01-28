---
title: "SVM "
author: "db"
---

# 1. Packages

```python=
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import sklearn.model_selection as skm
from sklearn.metrics import RocCurveDisplay
from sklearn.svm import SVC
from ISLP import load_data, confusion_table
from ISLP.svm import plot as plot_svm
roc_curve = RocCurveDisplay.from_estimator # shorthand
```

# 2. Support Vector Classifier

## 2.1 Theorie

### 2.1.1. Hyperplane

- Definition: Eine Hyperplane ist eine flache affine Untermannigfaltigkeit der Dimension p-1.
  
  - In 2D: Linie
  - In 3D: Ebene
  - In p>3: Schwer vorstellbar, aber gleiche Definition.

- Mathematische Definition:
  
  - 2D: $\beta_0 + \beta_1 X_1 + \beta_2 X_2 = 0$
  - p-Dimensionen: $\beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots + \beta_p X_p = 0$

- Positionierung:
  
  - $\beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots + \beta_p X_p > 0$: Punkt liegt auf einer Seite.
  - $\beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots + \beta_p X_p < 0$: Punkt liegt auf der anderen Seite.

- Klassen: $y_1, \ldots, y_n \in \{-1, 1\}$

- Klassifikation mit einer trennenden Hyperplane:
  
  - $\beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \ldots + \beta_p x_{ip} > 0 \text{ für } y_{i} = 1$
  - $\beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \ldots + \beta_p x_{ip} < 0 \text{ für } y_{i} = -1$

- Problem: Unendlich viele trennende Hyperplanes möglich.

- Lösung: Maximal Margin Hyperplane wählen.
  
  - Größte minimale Distanz zu den Trainingsbeobachtungen.
  - Abhängigkeit von den Support Vektoren.
    - Beobachtungen, die die Margin beeinflussen.
    - Nur Support Vektoren beeinflussen den Classifier.

### 2.1.2. Maximal Margin Classifier

- Optimierungsproblem:
  
  - Maximieren von $M$
  - Einschränkungen: $\sum_{j=1}^{p}\beta_j^2=1$, $y_i(\beta_0+\beta_1x_{i1}+\beta_2x_{i2}+\ldots+\beta_px_{ip})\geq M$
    - $y_i(\beta_0+\beta_1x_{i1}+\beta_2x_{i2}+\ldots+\beta_px_{ip})$: den Abstand der *i*-ten Beobachtung zur Hyperbene.
    - $\beta_0, \beta_1, ..., \beta_p$: Koeffizienten der maximalen Marginal-Hyperbene.
    - $\beta_0+\beta_1x_{i1}+\beta_2x_{i2}+\ldots+\beta_px_{ip} = 0$: Definiert die Hyperbene.
    - $\sum_{j=1}^{p}\beta_j^2=1$: Stellt sicher, dass dieser Ausdruck den senkrechten Abstand der *i*-ten Beobachtung zur Hyperbene repräsentiert.

- Wenn es keine trennende Hyperplane existiert, verwenden Soft Margin, um Klassen fast zu trennen -> Support Vector Classifier.

### 2.1.3. Support Vector Classifiers

- Überblick: Support Vector Classifier (auch Soft Margin Classifier genannt) erlaubt einige Fehler, um die meisten Beobachtungen korrekt zu klassifizieren.

- Optimierungsproblem:
  
  - Maximierung der Margin $M$.
  - Einbeziehung von Slack-Variablen $\epsilon_i$ erlaubt Fehler.
  - Die Einschränkung: $y_i(\beta_0+\beta_1x_{i1}+\beta_2x_{i2}+\ldots+\beta_px_{ip})\geq M(1-\epsilon_i)$
    - $\epsilon_i = 0$:
      - Beobachtung ist korrekt klassifiziert.
      - Liegt auf oder jenseits des Margins.
    - $\epsilon_i > 0$:
      - Beobachtung hat den Margin verletzt.
      - Kann dennoch auf der richtigen Seite der Hyperbene liegen.
    - $\epsilon_i > 1$:
      - Beobachtung liegt auf der falschen Seite der Hyperbene. Ist fehlklassifiziert.

- Parameter $C$: kontrolliert die zulässige Summe der $\epsilon_i$'s, fungiert als Budget für Margin-Verletzungen.
  
  - $C = 0$: Keine Toleranz für Fehler (maximal margin hyperplane).
  - $C > 0$: Erlaubt Fehler, größere Margin.
  - Größeres $C$: Mehr Toleranz, breitere Margin.
  - Kleineres $C$: Weniger Toleranz, schmalere Margin.

- Eigenschaften des Support Vector Classifiers
  
  - Robustheit: Weniger empfindlich gegenüber weit entfernten Beobachtungen.
  - Bias-Varianz Trade-off:
    - Großes $C$: Mehr Support Vektoren, niedrige Varianz, hoher Bias.
    - Kleines $C$: Weniger Support Vektoren, hohe Varianz, niedriger Bias.

- Vergleich mit anderen Methoden:
  
  - Linear Discriminant Analysis (LDA): Sensitiv gegenüber allen Beobachtungen.
  - Logistic Regression: Weniger sensitiv, ähnlich wie Support Vector Classifier.

## 2.2. Anwendung: Two-dimensional example

- Wir nutzen `SupportVectorClassifier()` (`SVC`) aus `sklearn.svm`.
  - Höherer `C`-Wert:
    - Modell bestraft Fehlklassifikationen stark.
    - Komplexere Entscheidungsgrenze, die Trainingsdaten genau anpasst.
    - **Kleinerer Margin** wegen Fokus auf Minimierung der Klassifikationsfehler.
    - Risiko des Overfitting.
  - Niedriger `C`-Wert:
    - Modell erlaubt mehr Fehlklassifikationen für größeren Margin.
    - Glattere Entscheidungsgrenze, die den Margin maximiert.
    - **Größerer Margin** wegen weniger Gewicht auf korrekte Klassifikation aller Trainingsbeispiele.
    - Risiko des Underfitting.
  - Der `kernel`-Parameter spezifiziert den Typ des Kernels.
    - `linear` Kernel: eine lineare Hyperebene.
    - `rbf` Kernel: eine Radial-Basis-Funktion.

a. Clear separation

```python=
# Generate data
rng = np.random.default_rng(1)
# 2D dataset with shape (50, 2)
X = rng.standard_normal((50, 2))
# Response of 50 with the first 25 being -1 and the last 25 being 1
y = np.array([-1]*25+[1]*25)
# Select the last 25 rows of X and add 2.9
X[y==1] += 2.9;
fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(X[:25,0], X[:25,1], c=y[:25], marker='o')
ax.scatter(X[25:,0], X[25:,1], c=y[25:], marker='x');
# ax.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.coolwarm);
```

![](Figures\svm_5_0.png)

```python=
SVM_clearLarge = SVC(C=1e5, kernel='linear').fit(X, y)
y_hat_clearLarge = SVM_clearLarge.predict(X)
confusion_table(y_hat_clearLarge, y)
```

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted\Truth</th>
      <th>-1</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>-1</th>
      <td>25</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>25</td>
    </tr>
  </tbody>
</table>

```python=
fig, ax = plt.subplots(figsize=(8,8))
plot_svm(X, y, SVM_clearLarge, ax=ax)
```

![](Figures\svm_8_0.png)

- Keine Trainingsfehler wurden gemacht.
  - Es wurden nur drei Support-Vektoren verwendet.
    - Der große Wert von `C` bedeutet, dass diese drei Punkte *auf der Margin* liegen und sie definieren.
    - Man könnte sich fragen, wie gut der Klassifikator auf Testdaten ist, wenn er nur von drei Punkten abhängt.
- Wir probieren jetzt einen kleineren Wert für `C`.

```python=
SVM_clearSmall = SVC(C=0.1, kernel='linear').fit(X, y)
y_hat_clearSmall = SVM_clearSmall.predict(X)
confusion_table(y_hat_clearSmall, y)
```

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted\Truth</th>
      <th>-1</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>-1</th>
      <td>25</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>25</td>
    </tr>
  </tbody>
</table>

- Mit `C=0.1` klassifizieren wir erneut keine Trainingsdaten falsch.
  - Wir erhalten jedoch eine deutlich breitere Margin.
  - Es werden zwölf Support-Vektoren verwendet.
    - Diese definieren gemeinsam die Orientierung der Entscheidungsgrenze.
    - Da es mehr Support-Vektoren gibt, ist die Grenze stabiler.
- Es ist möglich, dass dieses Modell bei Testdaten besser abschneidet als das Modell mit `C=1e5`.
  - Ein einfaches Experiment mit einem großen Testdatensatz würde dies bestätigen.

```python=
fig, ax = plt.subplots(figsize=(8,8))
plot_svm(X, y, SVM_clearSmall, ax=ax)
```

![](Figures\svm_12_0.png)

b. Unclear Separation

```python=
rng = np.random.default_rng(1)
# Create a 2D dataset with shape (50, 2)
X = rng.standard_normal((50, 2))
# Response of 50 elements with first 25 being -1 and last 25 being 1
y = np.array([-1]*25+[1]*25)
# Select the last 25 rows of X and add 1
X[y==1] += 1
# Create a scatter plot of the data
fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(X[:25,0], X[:25,1], c=y[:25], marker='o')
ax.scatter(X[25:,0], X[25:,1], c=y[25:], marker='x');
```

![](Figures\svm_14_0.png)

```python=
SVM_unclearLarge = SVC(C=10, kernel='linear').fit(X, y)
# Plot the decision boundary
fig, ax = plt.subplots(figsize=(8,8))
plot_svm(X, y, SVM_unclearLarge, ax=ax)
```

![](Figures\svm_16_0.png)

- Die Support-Vektoren sind mit `+` markiert.
  - Sie sind in `svc.support_vectors_` gespeichert.
- Die verletzenden Punkte sind mit `x` markiert.
- Die Methode `svm_linear.decision_function()` berechnet die Distanz jedes Samples in X zur Entscheidungsgrenze.
  - Positive Werte zeigen eine Seite an.
  - Negative Werte zeigen die andere Seite an.
- Das Attribut `svm_linear.support_` liefert die Indizes der Support-Vektoren im Trainingssatz.

```python=
# Show the support vectors
print('First 3 support vectors:\n', svm_linear.support_vectors_[:3])
# Show the distance of the samples to the decision boundary
print('First 3 decision function values:\n', svm_linear.decision_function(X)[:3])
```

    First 3 support vectors:
     [[ 0.345  0.821]
      [ 0.905  0.446]
      [-0.536  0.581]]
    First 3 decision function values:
     [0.221 -1.440  0.587]

```python=
SVM_unclearSmall = SVC(C=0.1, kernel='linear').fit(X, y)
SVM_unclearSmall
fig, ax = plt.subplots(figsize=(8,8))
plot_svm(X, y, SVM_unclearSmall, ax=ax)
```

![](Figures\svm_20_0.png)

- Mit einem kleineren Wert des Kostenparameters erhalten wir mehr Support-Vektoren.
  - Die Margin ist breiter.
- Wir können die Koeffizienten der linearen Entscheidungsgrenze wie folgt extrahieren:

```python=
SVM_unclearLarge.coef_
# array([[1.173, 0.773]])
```

- Wir können Standardwerkzeuge verwenden, um den Kostenparameter `C` zu optimieren, weil SVM ist ein Schätzer in `sklearn`.

```python=
kfold = skm.KFold(5, random_state=0, shuffle=True)
grid_unclearLarge = skm.GridSearchCV(SVM_unclearLarge,
                        {'C':[0.001,0.01,0.1,1,5,10,100]},
                        refit=True, cv=kfold, scoring='accuracy')
grid_unclearLarge.fit(X, y)
# Show the best parameter
grid_unclearLarge.best_params_
# {'C': 1}
```

- Wir zeigen den Kreuzvalidierungsfehler für jeden Wert von `C` im Raster an.

```python=
pd.DataFrame( {'C':[0.001,0.01,0.1,1,5,10,100],
  'Mean test score':grid_unclearLarge.cv_results_[('mean_test_score')]})
```

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>C</th>
      <th>Mean test score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.001</td>
      <td>0.46</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.010</td>
      <td>0.46</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.100</td>
      <td>0.72</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.000</td>
      <td>0.74</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.000</td>
      <td>0.74</td>
    </tr>
    <tr>
      <th>5</th>
      <td>10.000</td>
      <td>0.74</td>
    </tr>
    <tr>
      <th>6</th>
      <td>100.000</td>
      <td>0.74</td>
    </tr>
  </tbody>
</table>

- Das Ergebnis zeigt eine Kreuzvalidierungsgenauigkeit von 0,74 für `C=1` und höher.
- Wir erhalten den besten Schätzer mit `grid.best_estimator_` und verwenden ihn für den generierten Testdatensatz.

```python=
# Generate test data
X_test = rng.standard_normal((20, 2))
y_test = np.array([-1]*10+[1]*10)
X_test[y_test==1] += 1
# Best estimator
bestgrid_unclearLarge = grid_unclearLarge.best_estimator_
y_test_hat_unclearLarge = bestgrid_unclearLarge.predict(X_test)
confusion_table(y_test_hat_unclearLarge, y_test)
```

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted\Truth</th>
      <th>-1</th>
      <th>1</th>
    </tr>

</thead>
  <tbody>
    <tr>
      <th>-1</th>
      <td>8</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>6</td>
    </tr>
  </tbody>
</table>

- Mit `C=1` beträgt die Genauigkeit 14/20 = 0,7.
- Mit `C=0.001` beträgt die Genauigkeit 12/20 = 0,6.

```python=
y_test_hat_unclearSmall = SVM_unclearSmall.predict(X_test)
confusion_table(y_test_hat_unclearSmall, y_test)
```

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted\Truth</th>
      <th>-1</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>-1</th>
      <td>8</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4</td>
    </tr>
  </tbody>
</table>

# 3. Support Vector Machine

- **Klassifikation mit nicht-linearen Entscheidungsgrenzen**
  
  - Ein Support Vector Classifier (SVC) eignet sich gut für lineare Klassifikationsgrenzen.
  - Bei nicht-linearen Grenzen versagt der SVC.
  - Lösung: Erweiterung des Merkmalsraums durch quadratische, kubische oder höhere Polynome.
    - Beispiel: Statt 𝑝 Merkmale $X_1, X_2, \ldots, X_p$ zu verwenden, nutzen wir 2𝑝 Merkmale $X_1, X_1^2, X_2, X_2^2, \ldots , X_p, X_p^2$.
  - Resultat: In erweitertem Merkmalsraum ist die Grenze linear, im ursprünglichen Raum jedoch nicht-linear.
  - Achtung: Zu viele Merkmale können Berechnungen unhandlich machen.

- **Die Support Vector Machine**
  
  - SVM erweitert SVC durch Verwendung von Kernels.
  
  - Kernels definieren die Ähnlichkeit zweier Beobachtungen.
    
    - Lineares Kernel: $K(x_i,x_{i\prime}) = \sum_{j=1}^{p}{x_{ij}x_{i\prime j}}$.
    - Polynomiales Kernel: $K(x_i,x_{i\prime}) = (1+\sum_{j=1}^{p}{x_{ij}x_{i\prime j}})^d$.
    - Radial Kernel: $K(x_i,x_{i'})=exp(-\gamma\sum_{j=1}^{p}(x_{ij}-x_{i^\prime j})^2)$.
  
  - SVM mit polynomischem Kernel passt besser zu nicht-linearen Daten.
  
  - Vorteile von Kernels:
    
    - Berechnungen sind effizienter.
    - Keine explizite Arbeit im erweiterten Merkmalsraum nötig.
    - Einige Kernel, wie das radiale Kernel, arbeiten in unendlich-dimensionalen Räumen.
  
  - Kernelmatrix berechnen:
    
    - Für alle Paare von Trainingsbeobachtungen ($x_i$, $x_{i'}$) wird die Kernel-Funktion $K(x_i, x_{i'})$ berechnet.
    - Dies bildet die Kernelmatrix, die die paarweisen Ähnlichkeiten zwischen allen Trainingsdatenpunkten im transformierten hochdimensionalen Raum darstellt.
  
  - Beispiel einer Kernelmatrix für ein Datensatz mit n = 3 und p = 2:
    
    - $K = \begin{bmatrix} K(x_1, x_1) & K(x_1, x_2) & K(x_1, x_3) \\ K(x_2, x_1) & K(x_2, x_2) & K(x_2, x_3) \\ K(x_3, x_1) & K(x_3, x_2) & K(x_3, x_3) \end{bmatrix}$
  
  - SVM-Optimierungsproblem lösen: verwenden nur die Kernelmatrix und keine Berechnung der hochdimensionalen Merkmale.
  
  - Neue Daten klassifizieren:
    
    - Um eine neue Beobachtung $x^*$ zu klassifizieren, wird die Kernel-Funktion verwendet, um die Ähnlichkeit zwischen $x^*$ und jedem Support-Vektor zu berechnen.
    - Basierend auf diesen Ähnlichkeiten und den gelernten Parametern aus dem Optimierungsproblem weist die SVM $x^*$ eine Klassenmarke zu.

- Polynomial Kernel `kernel="poly"`:
  
  - Der Parameter `degree` gibt den Grad des Polynoms an.

- Radial Basis Function (RBF) Kernel `kernel="rbf"`:
  
  - Der Parameter `gamma` gibt den Koeffizienten des Kernels an.

- Zuerst generieren wir Daten mit einer nicht-linearen Klassengrenze.

```python=
# Generate non-linear data
# Create a 2D dataset with shape (200, 2)
X = rng.standard_normal((200, 2))
# Select first 100 rows of X and add 2 to each element
X[:100] += 2
# Select rows from 100 to 150 of X and subtract 2 from each element
X[100:150] -= 2
# Response of 200 elements with first 150 being 1 and the last 50 being 2
y = np.array([1]*150+[2]*50)
# Create a scatter plot of the data
fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(X[:150,0], X[:150,1], c=y[:150], marker='o')
ax.scatter(X[150:,0], X[150:,1], c=y[150:], marker='x');
```

![](Figures\svm_32_0.png)

```python=
X_train, X_test, y_train, y_test = skm.train_test_split(X, y,
                                 test_size=0.5,random_state=0)
# Radical basis function SVM
SVM_rbf1 = SVC(kernel="rbf", gamma=1, C=1).fit(X_train, y_train)
# Plot the decision boundary
fig, ax = plt.subplots(figsize=(8,8))
plot_svm(X_train, y_train, SVM_rbf1, ax=ax)
```

![](Figures\svm_33_0.png)

- Der Plot zeigt:
  
  - Die resultierende SVM hat eine eindeutig nicht-lineare Grenze.
  - Es gibt einige Trainingsfehler.

- Um die Anzahl der Trainingsfehler zu reduzieren:
  
  - Erhöhen wir den Wert von `C`.
  - Achtung: Ein zu hoher `C`-Wert kann zu Overfitting führen.
    - Je unregelmäßiger die Grenze, desto höher die Wahrscheinlichkeit von Overfitting.

```python=
SVM_rbf1e5 = SVC(kernel="rbf", gamma=1, C=1e5).fit(X_train, y_train)
fig, ax = plt.subplots(figsize=(8,8))
plot_svm(X_train, y_train, SVM_rbf1e5, ax=ax)
```

![](Figures\svm_35_0.png)

- Hyperparameter-Tuning:
  - Verwenden `skm.GridSearchCV()`, um die Hyperparameter `C` und `gamma` RBF Kernel zu optimieren.

```python=
kfold = skm.KFold(5, random_state=0, shuffle=True)
gridSVM_rbf1e5 = skm.GridSearchCV(svm_rbf, refit=True, cv=kfold,
 scoring='accuracy', {'C':[0.1,1,10,100,1000], 'gamma':[0.5,1,2,3,4]} );
gridSVM_rbf1e5.fit(X_train, y_train)
# Show the best parameters
gridSVM_rbf1e5.best_params_
# {'C': 1, 'gamma': 0.5}
```

```python=
bestgrid_rbf05 = gridSVM_rbf1e5.best_estimator_
# Plot the decision boundary of the best estimator
fig, ax = plt.subplots(figsize=(8,8))
plot_svm(X_train, y_train, bestgrid_rbf05, ax=ax)
# Test data
y_hat_test_rbf05 = bestgrid_rbf05.predict(X_test)
confusion_table(y_hat_test_rbf05, y_test)
```

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted\Truth</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>69</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>19</td>
    </tr>
  </tbody>
</table>

![](Figures\svm_40_1.png)

- Mit diesen Parametern:
  - $12/100=12\%$ der Testbeobachtungen werden von diesem SVM falsch klassifiziert.

## ROC Kurve

- Klassifikator Scores:
  
  - Klassifikatoren wie LDA und SVM berechnen Scores für jede Beobachtung.
  - Form von LDA oder SVC: $\hat{f}(X) = \hat{\beta}_0 + \hat{\beta}_1 X_1 + \hat{\beta}_2 X_2 + \ldots + \hat{\beta}_p X_p$.
  - Form von SVM ist im Buch.

- Schwellenwerte:
  
  - Beobachtungen werden in zwei Kategorien eingeteilt, basierend auf dem Score $\hat{f}(X)$ und einem Schwellenwert *t*.
  - $\hat{f}(X) < t$: eine Kategorie (z.B. "Herzkrankheit").
  - $\hat{f}(X) \geq t$: andere Kategorie (z.B. "keine Herzkrankheit").

- Wahre und falsche Positivraten:
  
  - ROC-Kurve: Plot der falschen Positivrate (x-Achse) gegen die wahre Positivrate (y-Achse) für verschiedene *t*.
  
  - Falsche Positivrate: Anteil negativer Fälle, die fälschlicherweise als positiv klassifiziert wurden.
  
  - Wahre Positivrate: Anteil positiver Fälle, die korrekt klassifiziert wurden.
  
  - Beispiel:
    
    <table>
        <tr>
            <th></th>
            <th>Actual Positive</th>
            <th>Actual Negative</th>
        </tr>
        <tr>
            <th>Predicted Positive</th>
            <td>40</td>
            <td>5</td>
        </tr>
        <tr>
            <th>Predicted Negative</th>
            <td>10</td>
            <td>45</td>
        </tr>
    </table>
    
    - Wahre Positive (TP): 40
    - Falsche Negative (FN): 10
    - Falsche Positive (FP): 5
    - Wahre Negative (TN): 45
    - Sensitivität (True Positive Rate, TPR):
      - Formel: TP / (TP + FN) = TP / P
      - Berechnung: 40 / (40 + 10) = 0.8 oder 80%
    - Falsche Positive Rate (FPR) oder 1 - Specificity:
      - Formel: FP / (FP + TN) = FP / N
      - Berechnung: 5 / (5 + 45) = 0.1 oder 10%
    - TP / P : Sensitivity, Power, Recall, 1 - Type II Error
    - FP / N : Type I Error, 1 - Specificity
    - TP / (TP+FP) : Precision, 1 - False Discovery Proportion

- Optimaler Klassifikator:
  
  - Idealer Klassifikator: Oberer linker Eckpunkt der ROC-Kurve.
  - Hohe wahre Positivrate bei niedriger falscher Positivrate.

- Modellvergleich:
  
  - ROC-Kurven vergleichen verschiedene Klassifikatoren.
  - Klassifikator mit höherer ROC-Kurve ist überlegen.

- `ROCCurveDisplay.from_estimator() roc_curve()`:
  
  - Erstes Argument: ein angepasster Schätzer.
  - Zweites und drittes Argument: Modellmatrix `X` und Labels `y`.
  - Argument `name`: Für die Legende.
  - Argument `color`: Für die Farbe der Linie.
  - Argument 'linestyle': Für das Stil der Linie.

```python=
fig, ax = plt.subplots(figsize=(8,8))
roc_curve(bestgrid_rbf05, X_train, y_train, name='Training', color='r', ax=ax);
roc_curve(bestgrid_rbf05, X_test, y_test, name='Test', color='b', linestyle='--', ax=ax);
```

![](Figures\svm_46_0.png)

- Um die Grenze unregelmäßiger zu machen und die Genauigkeit zu verbessern:
  - Erhöhen wir den Wert von $\gamma$.

```python=
# Set and fit a radial basis function SVM with large gamma
SVM_rbfG50 = SVC(kernel="rbf", gamma=50, C=1).fit(X_train, y_train)
fig, ax = plt.subplots(figsize=(8,8))
roc_curve(SVM_rbfG50, X_train, y_train, name=r'Training $\gamma=50$', color='r', ax=ax);
roc_curve(SVM_rbfG50, X_test, y_test, name=r'Test $\gamma=50$', color='b', linestyle='--', ax=ax);
```

![](Figures\svm_51_0.png)

- Der Plot zeigt:
  - Die SVM mit $\gamma=0.5$ bietet eine bessere Vorhersage als die SVM mit $\gamma=50$.

# 4. SVM with Multiple Classes

- Einleitung
  
  - Bisher: SVMs für binäre Klassifikation.
  - Herausforderung: Erweiterung auf mehr als zwei Klassen.
  - Beliebte Ansätze: One-Versus-One (OvO) und One-Versus-All (OvA).

- One-Versus-One Klassifikation
  
  - Konstruktion von $\binom{K}{2}$ SVMs, jede vergleicht ein Klassenpaar.
  - Beispiel: SVM vergleicht Klasse $k$ ($+1$) mit Klasse $k'$ ($-1$).
  - Testklassifikation: Verwendung jeder der $\binom{K}{2}$ SVMs.
  - Endergebnis: Zuweisung zur Klasse, die am häufigsten gewählt wurde.

- One-Versus-All Klassifikation
  
  - Konstruktion von $K$ SVMs, jede vergleicht eine Klasse mit den anderen $K-1$ Klassen.
  - Beispiel: SVM für Klasse $k$ ($+1$) gegen Rest ($-1$).
  - Testklassifikation: Zuweisung zur Klasse mit größtem Wert $\beta_{0k} + \beta_{1k} x_1^\ast + \ldots + \beta_{pk} x_p^\ast$.

- `SVC` Klassifikation:
  
  - Führt One-vs-One Klassifikation durch, wenn `decision_function_shape='ovo'`.
  - Führt One-vs-Rest Klassifikation durch, wenn `decision_function_shape='ovr'`.

```python=
np.hstack([y, [0]*50])
```

    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0])

```python=
# Generate data with 3 classes
rng = np.random.default_rng(123)
# Add the third class data (shape 50x2) to the previous data
X = np.vstack([X, rng.standard_normal((50, 2))])
# Add the third class y values to the previous y values
y = np.hstack([y, [0]*50])
# Select the rows of first and third class data and add 2 to each element
X[y==0,1] += 2
# Create a scatter plot of the data
fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(X[:150,0], X[:150,1], c=y[:150], marker='o')
ax.scatter(X[150:200,0], X[150:200,1], c=y[150:200], marker='x');
ax.scatter(X[200:,0], X[200:,1], c=y[200:], marker='+');
```

![](Figures\svm_58_0.png)

```python=
# Set and fit a radial basis function SVM
svm_rbf_3 = SVC(kernel="rbf", C=10, gamma=1,
                decision_function_shape='ovo');
svm_rbf_3.fit(X, y)
# Plot the decision boundary
fig, ax = plt.subplots(figsize=(8,8))
plot_svm(X, y, svm_rbf_3, scatter_cmap=plt.cm.tab10, ax=ax);
```

![](Figures\svm_59_0.png)

- `sklearn.svm` Paket:
  - Unterstützt auch Support Vector Regression.
  - Verwenden den Schätzer `SVR()`.

# 5. Application to Gene Expression Data

- Datensatz `Khan` Überblick:
  
  - Enthält eine Anzahl von Gewebeproben, die vier verschiedenen Arten von kleinen, runden, blauen Zell-Tumoren entsprechen.
    - Gewebeprobe: Ein kleines Stück Gewebe, das aus dem Körper entnommen wird, um unter dem Mikroskop untersucht zu werden. Dies wird oft zur Diagnose von Krankheiten, einschließlich Krebs, durchgeführt.
    - Kleine, runde, blaue Zell-Tumoren: Eine Krebsart, die aus kleinen, runden Zellen besteht, die unter dem Mikroskop blau aussehen.

- Genexpressionsmessungen:
  
  - Für jede Gewebeprobe sind Genexpressionsmessungen verfügbar.

- Datensatzstruktur:
  
  - Trainingdaten: `xtrain` und `ytrain`.
  - Testdaten: `xtest` und `ytest`.

```python=
Khan = load_data('Khan')
Khan['xtrain'].head(3)
```

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>G0001</th>
      <th>G0002</th>
      <th>...</th>
      <th>G2307</th>
      <th>V2308</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.773344</td>
      <td>-2.438405</td>
      <td>...</td>
      <td>-0.647600</td>
      <td>-1.763172</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.078178</td>
      <td>-2.415754</td>
      <td>...</td>
      <td>-1.209320</td>
      <td>-0.824395</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.084469</td>
      <td>-1.649739</td>
      <td>...</td>
      <td>-0.805868</td>
      <td>-1.139434</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 2308 columns</p>

```python=
Khan['ytrain'].head(3)
```

    0    2
    1    2
    2    2
    Name: Y, dtype: int64

```python=
Khan['xtrain'].describe()
```

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>G0001</th>
      <th>G0002</th>
      <th>...</th>
      <th>G2307</th>
      <th>V2308</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mean</th>
      <td>0.146931</td>
      <td>-1.739001</td>
        <th>...</th>
      <td>-0.507258</td>
      <td>-1.566933</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.853339</td>
      <td>0.905571</td>
        <th>...</th>
      <td>0.577504</td>
      <td>0.632065</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.683846</td>
      <td>-3.007805</td>
        <th>...</th>
      <td>-2.691193</td>
      <td>-3.110021</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.081323</td>
      <td>-2.427080</td>
        <th>...</th>
      <td>-0.812063</td>
      <td>-1.992975</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.244200</td>
      <td>-1.949818</td>
        <th>...</th>
      <td>-0.428939</td>
      <td>-1.500584</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.735390</td>
      <td>-1.318729</td>
        <th>...</th>
      <td>-0.178323</td>
      <td>-1.209826</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.285507</td>
      <td>0.654770</td>
        <th>...</th>
      <td>0.570471</td>
      <td>0.041142</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 2308 columns</p>

```python=
print('Shape of xtrain:', Khan['xtrain'].shape)
print('Shape of xtest:', Khan['xtest'].shape)
```

    Shape of xtrain: (63, 2308)
    Shape of xtest: (20, 2308)

- Wahl des Kernels:
  - Da die Anzahl der Merkmale (Gene) viel größer ist als die Anzahl der Beobachtungen, sollten wir einen linearen Kernel verwenden.

```python=
# Set and fit a linear SVM on the training data
khan_linear = SVC(kernel='linear', C=10)
khan_linear.fit(Khan['xtrain'], Khan['ytrain'])
# Create a confusion table
confusion_table(khan_linear.predict(Khan['xtrain']),
                Khan['ytrain'])
```

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted\Truth</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>23</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>20</td>
    </tr>
  </tbody>
</table>

- Ergebnis:
  - Keine Trainingsfehler.
  - Das ist nicht überraschend.
    - Viele Variablen im Vergleich zu Beobachtungen machen es einfach, Hyperbenen zu finden, die Klassen vollständig trennen.

```python=
confusion_table(khan_linear.predict(Khan['xtest']), Khan['ytest'])
```

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted\Truth</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>6</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
  </tbody>
</table>

# 6. Beziehung zur Logistischen Regression

- Einleitung
  
  - SVMs wurden in den 1990ern populär.
  - Neuartige Methode: Trennende Hyperebene und Kernel für nicht-lineare Grenzen.

- Verbindung zu klassischen Methoden
  
  - SVMs ähneln klassischen Methoden wie der logistischen Regression.
  - Kriterium für SVMs: Minimierung der Verlustfunktion plus Strafe.
    - Beispiel: $\text{minimiere} \left\{ \sum_{i=1}^{n} \max[0, 1 - y_i f(x_i)] + \lambda \sum_{j=1}^{p} \beta_j^2 \right\}$.
  - $\lambda$: Tuning-Parameter zur Steuerung von Bias und Varianz.
  - Form: $\text{minimiere} \{L(X, y, \beta) + \lambda P(\beta)\}$.
    - $L(X, y, \beta)$: Verlustfunktion.
    - $P(\beta)$: Strafterm.
  - Beispiel: Ridge-Regression und Lasso.

- Verlustfunktionen
  
  - SVM-Verlust: Hinge Loss.
    - Beispiel: $\sum_{i=1}^{n} \max[0, 1 - y_i(\beta_0 + \beta_1 x_{i1} + \ldots + \beta_p x_{ip})]$.
  - Logistische Regression und SVMs haben ähnliche Verlustfunktionen.
  - Unterschiede:
    - SVM: Nur Support-Vektoren beeinflussen den Klassifikator.
    - Logistische Regression: Verlustfunktion ist nie genau null.

- Tuning-Parameter
  
  - $\lambda$ und $C$ sind entscheidend für Modellanpassung.
  - Falsche Einstellung kann zu Underfitting oder Overfitting führen.

- Erweiterungen und Anwendungen
  
  - Logistische Regression kann auch mit nicht-linearen Kernen erweitert werden.
  - SVMs nutzen häufiger nicht-lineare Kerne.
  - Support Vector Regression (SVR): Verwendet Margin-ähnliche Methoden für Regression.
    - Minimiert Verlust basierend auf Residuen, die einen Schwellenwert überschreiten.