---
title: "Boston"
author: "db"
---

# 1. Datenbeschreibung
- Datensatz enthält Wohnungswerte in 506 Vororten von Boston.
- 13 Variablen:
  - **crim**: Kriminalitätsrate.
  - **zn**: Wohngebietsprozentsatz für große Grundstücke.
  - **indus**: Prozentsatz der nicht-gewerblichen Flächen.
  - **chas**: Grenzt an den Charles River? (1 = ja, 0 = nein).
  - **nox**: Stickstoffoxidwerte.
  - **rm**: Durchschnittliche Zimmeranzahl.
  - **age**: Prozentsatz der vor 1940 gebauten Häuser.
  - **dis**: Entfernung zu Beschäftigungszentren.
  - **rad**: Zugang zu Autobahnen.
  - **tax**: Grundsteuersatz.
  - **ptratio**: Schüler-Lehrer-Verhältnis.
  - **lstat**: Prozentsatz einkommensschwacher Bewohner.
  - **medv**: Medianer Hauswert (in Tausend Dollar).
- Zielvariable: **medv**.

# 2. Packages und Daten

```python=
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import sklearn.model_selection as skm
from sklearn.tree import (DecisionTreeRegressor as DTR,
                          plot_tree,
                          export_text)
from sklearn.metrics import (accuracy_score, log_loss)
from sklearn.ensemble import \
     (RandomForestRegressor as RF,
      GradientBoostingRegressor as GBR)
from ISLP import load_data, confusion_table
from ISLP.models import ModelSpec as MS
from ISLP.bart import BART
```

```python=
Boston = load_data("Boston")
Boston.head(2)
```

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>crim</th>
      <th>zn</th>
      <th>indus</th>
      <th>chas</th>
      <th>nox</th>
      <th>rm</th>
      <th>age</th>
      <th>dis</th>
      <th>rad</th>
      <th>tax</th>
      <th>ptratio</th>
      <th>lstat</th>
      <th>medv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1</td>
      <td>296</td>
      <td>15.3</td>
      <td>4.98</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242</td>
      <td>17.8</td>
      <td>9.14</td>
      <td>21.6</td>
    </tr>
  </tbody>
</table>

```python=
Boston.describe().round(1)
```

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>crim</th>
      <th>zn</th>
      <th>indus</th>
      <th>chas</th>
      <th>nox</th>
      <th>rm</th>
      <th>age</th>
      <th>dis</th>
      <th>rad</th>
      <th>tax</th>
      <th>ptratio</th>
      <th>lstat</th>
      <th>medv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mean</th>
      <td>3.6</td>
      <td>11.4</td>
      <td>11.1</td>
      <td>0.1</td>
      <td>0.6</td>
      <td>6.3</td>
      <td>68.6</td>
      <td>3.8</td>
      <td>9.5</td>
      <td>408.2</td>
      <td>18.5</td>
      <td>12.7</td>
      <td>22.5</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.6</td>
      <td>23.3</td>
      <td>6.9</td>
      <td>0.3</td>
      <td>0.1</td>
      <td>0.7</td>
      <td>28.1</td>
      <td>2.1</td>
      <td>8.7</td>
      <td>168.5</td>
      <td>2.2</td>
      <td>7.1</td>
      <td>9.2</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.4</td>
      <td>3.6</td>
      <td>2.9</td>
      <td>1.1</td>
      <td>1.0</td>
      <td>187.0</td>
      <td>12.6</td>
      <td>1.7</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.1</td>
      <td>0.0</td>
      <td>5.2</td>
      <td>0.0</td>
      <td>0.4</td>
      <td>5.9</td>
      <td>45.0</td>
      <td>2.1</td>
      <td>4.0</td>
      <td>279.0</td>
      <td>17.4</td>
      <td>7.0</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.3</td>
      <td>0.0</td>
      <td>9.7</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>6.2</td>
      <td>77.5</td>
      <td>3.2</td>
      <td>5.0</td>
      <td>330.0</td>
      <td>19.0</td>
      <td>11.4</td>
      <td>21.2</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.7</td>
      <td>12.5</td>
      <td>18.1</td>
      <td>0.0</td>
      <td>0.6</td>
      <td>6.6</td>
      <td>94.1</td>
      <td>5.2</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>17.0</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>89.0</td>
      <td>100.0</td>
      <td>27.7</td>
      <td>1.0</td>
      <td>0.9</td>
      <td>8.8</td>
      <td>100.0</td>
      <td>12.1</td>
      <td>24.0</td>
      <td>711.0</td>
      <td>22.0</td>
      <td>38.0</td>
      <td>50.0</td>
    </tr>
  </tbody>
</table>

# 3. Regressionsbäume

## 3.1. Theorie
- **Schritt 1: Prädiktorraum aufteilen**
  - Prädiktorraum ($X_1, X_2, \ldots , X_p$) in J Regionen ($R_1, R_2, \ldots , R_J$) aufteilen.
  - Gleiche Vorhersage in jeder Region $R_j$, basierend auf dem Mittelwert der Antwortwerte der Trainingsdaten.
- **Beispiel**
  - Zwei Regionen $R_1$ und $R_2$:
    - Mittelwert in $R_1$: 10
    - Mittelwert in $R_2$: 20
  - Vorhersage für $x$:
    - $x \in R_1$: Vorhersage 10
    - $x \in R_2$: Vorhersage 20
- **Regionen konstruieren**
  - Regionen als hochdimensionale Rechtecke (Boxen) definieren.
  - Ziel: RSS minimieren: $\sum_{j=1}^{J} \sum_{i \in R_j} (y_i - \hat{y}_{R_j})^2$.
  - Ansatz: Rekursive binäre Teilung (top-down, gierig).
- **Rekursive binäre Teilung**
  - Prädiktor $X_j$ und Schnittpunkt $s$ wählen, um den Prädiktorraum in $\{X \mid X_j < s\}$ und $\{X \mid X_j \geq s\}$ aufzuteilen.
  - Suche $j$ und $s$, die den RSS minimieren: $\sum_{i: x_i \in R_1(j,s)} (y_i - \hat{y}_{R_1})^2 + \sum_{i: x_i \in R_2(j,s)} (y_i - \hat{y}_{R_2})^2$.
- **Prozess wiederholen**
  - Besten Prädiktor und Schnittpunkt finden, um Daten weiter zu teilen und RSS zu minimieren.
  - Teile eine der identifizierten Regionen auf.
  - Fortsetzen, bis ein Abbruchkriterium erreicht ist (z.B. keine Region enthält mehr als fünf Beobachtungen).
- **Vorhersage**
  - Antwort für Testbeobachtung basierend auf dem Mittelwert der Trainingsbeobachtungen in der entsprechenden Region vorhersagen.

- **Problem: Überanpassung**
  - Große Bäume passen sich gut an Trainingsdaten an.
  - Überanpassung führt zu schlechter Performance bei Testdaten.

- **Lösung: Kleinere Bäume**
  - Kleinere Bäume reduzieren Varianz und verbessern die Interpretation.
  - Risiko: kleinere Bäume könnten wichtige Strukturen übersehen.

- **Strategie: Pruning**
  - Großen Baum $T_0$ wachsen lassen.
  - Baum zurückschneiden (Pruning), um einen optimalen Subbaum zu erhalten.

- **Ziel: Fehler minimieren**
  - Subbaum wählen, der den geringsten Testfehler hat.
  - Testfehler schätzen durch Kreuzvalidierung oder Validierungsansatz.

- **Kostenkomplexitäts-Pruning**
  - Sequenz von Bäumen durch Tuning-Parameter $\alpha$ indexieren.
  - Für jedes $\alpha$ Subbaum $T \subset T_0$ finden, der $\sum_{m=1}^{|T|} \sum_{i: x_i \in R_m} (y_i - \hat{y}_{R_m})^2 + \alpha |T|$ minimiert.
  - $\alpha$ steuert die Balance zwischen Baumkomplexität und Anpassung an Trainingsdaten.

- **Vorgehen**
  - $\alpha = 0$: Baum $T = T_0$.
  - $\alpha$ steigt: Komplexität wird bestraft, kleinere Subbäume werden bevorzugt.

- **Algorithmus 8.1: Regressionsbäume Bauen**
  1. **Großen Baum wachsen lassen**
     - Rekursive binäre Teilung anwenden.
     - Stopp, wenn Knoten weniger als eine Mindestanzahl an Beobachtungen haben.
  2. **Kostenkomplexitäts-Pruning anwenden**
     - Sequenz der besten Subbäume als Funktion von $\alpha$ erhalten.
  3. **Kreuzvalidierung nutzen, um $\alpha$ zu wählen**
     - Trainingsdaten in K-Folds aufteilen.
     - Schritte 1 und 2 auf allen außer einem Fold wiederholen.
     - Mittlere quadratische Vorhersagefehler im ausgelassenen Fold für jedes $\alpha$ bewerten.
     - $\alpha$ wählen, das den durchschnittlichen Fehler minimiert.
  4. **Subbaum zurückgeben**
     - Subbaum für das gewählte $\alpha$ aus Schritt 2 zurückgeben.


## 3.2. Anwendung

- Designmatrix in `ndarray` umwandeln:
  - Für die Funktion `DecisionTreeRegressor()`.

```python=
featureDF = MS(Boston.columns.drop('medv'), intercept=False).fit_transform(Boston)
feature_names = list(featureDF.columns)
featureARR = np.asarray(featureDF)
```

- Datenaufteilung:
  - Training Set: 70% der Daten.
  - Test Set: 30% der Daten.

```python=
X_train, X_test, y_train, y_test = skm.train_test_split(
    featureARR,Boston.medv, test_size=0.3, random_state=0)
```

```python=
TRE_Reg = DTR(max_depth=3)
TRE_Reg.fit(X_train, y_train)
# Plot tree
ax = plt.subplots(figsize=(12,12))[1]
plot_tree(TRE_Reg, feature_names=feature_names, ax=ax);
```

![](Figures\boston_10_0.png)

```python=
print(export_text(TRE_Reg, feature_names=feature_names))
```

    |--- rm <= 6.80
    |   |--- lstat <= 14.40
    |   |   |--- dis <= 1.47
    |   |   |   |--- value: [50.00]
    |   |   |--- dis >  1.47
    |   |   |   |--- value: [22.65]
    |   |--- lstat >  14.40
    |   |   |--- crim <= 5.78
    |   |   |   |--- value: [16.87]
    |   |   |--- crim >  5.78
    |   |   |   |--- value: [12.04]
    |--- rm >  6.80
    |   |--- rm <= 7.43
    |   |   |--- crim <= 7.39
    |   |   |   |--- value: [32.47]
    |   |   |--- crim >  7.39
    |   |   |   |--- value: [14.32]
    |   |--- rm >  7.43
    |   |   |--- ptratio <= 18.30
    |   |   |   |--- value: [46.25]
    |   |   |--- ptratio >  18.30
    |   |   |   |--- value: [28.55]
    

- Variablen:
  - `rm`: Durchschnittliche Anzahl der Zimmer.
  - `lstat`: Prozentsatz der einkommensschwachen Individuen.
- Ergebnisse der Baum-Analyse:
  - Weniger Armut (`lstat` niedrig) führt zu höheren Hauspreisen.
  - Beispiel:
    - Medianer Hauspreis: $12,042.
    - Kleine Häuser (`rm < 6.8`).
    - Hohe Armut (`lstat > 14.4`).
    - Moderate Kriminalitätsrate (`crim > 5.8`).
- Nächste Schritte:
  - Verwendung von Kreuzvalidierung.
  - Prüfen, ob Baumschnitt die Leistung verbessert.

```python=
# Find the optimal depth
ccp_path = TRE_Reg.cost_complexity_pruning_path(X_train, y_train)
# Set 5-fold cross-validation
kfold = skm.KFold(5, shuffle=True, random_state=10)
# Set grid search
grid = skm.GridSearchCV(TRE_Reg, {'ccp_alpha': ccp_path.ccp_alphas},
                        refit=True, cv=kfold,
                        scoring='neg_mean_squared_error').fit(X_train, y_train)
best_ = grid.best_estimator_
# Show training MSE
print('Training MSE:', np.mean((best_.predict(X_train) - y_train) ** 2))
# Show testing MSE
print('Testing MSE:', np.mean((best_.predict(X_test) - y_test) ** 2))
# Plot tree
ax = plt.subplots(figsize=(12,12))[1]
plot_tree(best_, feature_names=feature_names, ax=ax);
```

    Training MSE: 12.619
    Testing MSE: 28.069

    
![](Figures\boston_14_1.png)
    

- **Test Set MSE**:
  - MSE für den Regression Tree: $28.07 = 5.30^2$.
  - Durchschnittliche Abweichung zwischen vorhergesagtem und tatsächlichem Hauswert: ca. $5300.
- **Erneuter Lauf**:
  - Baum ohne maximale Tiefe wachsen lassen.

```python=
# Rerun with no restriction on depth
TRE_RegNo = DTR(random_state=2)
TRE_RegNo.fit(X_train, y_train)
# Find the optimal depth
ccp_path = TRE_RegNo.cost_complexity_pruning_path(X_train, y_train)
# Set 5-fold cross-validation
kfold = skm.KFold(5, shuffle=True, random_state=10)
# Set grid search
grid = skm.GridSearchCV(TRE_RegNo, {'ccp_alpha': ccp_path.ccp_alphas},
                        refit=True, cv=kfold,
                        scoring='neg_mean_squared_error').fit(X_train, y_train)
best_ = grid.best_estimator_
# Show training MSE
print('Training MSE:', np.mean((best_.predict(X_train) - y_train) ** 2))
# Show testing MSE
print('Testing MSE:', np.mean((best_.predict(X_test) - y_test) ** 2))
# Plot tree
ax = plt.subplots(figsize=(12,12))[1]
plot_tree(best_, feature_names=feature_names, ax=ax);
```

    Training MSE: 2.999
    Testing MSE: 24.791

![](Figures\boston_16_1.png)

# 4. Bagging and Random Forests

- Definition:
  - Kombinieren viele einfache Modelle (sogenannte schwache Lerner)
  - Ziel: Ein starkes und leistungsfähiges Modell

- Beispiele für Ensemble-Methoden:
  - Bagging
  - Random Forests
  - Boosting
  - Bayesian Additive Regression Trees

- Gemeinsames Merkmal:
  - Verwenden Regression- oder Klassifikationsbäume als Bausteine

- Nutzung von `RandomForestRegressor()` aus dem Paket `sklearn.ensemble` zum Fitten von Bagging- und Random-Forest-Modellen

## 4.1. Bagging

- **Bootstrap**
  - Einführung:
    - Bootstrap ist eine mächtige Methode.
    - Es hilft, Standardabweichungen zu berechnen, wenn dies direkt schwierig ist.
    - Kann verwendet werden, um statistische Lernmethoden wie Entscheidungsbäume zu verbessern.

  - Entscheidungsbäume und Varianz:
    - Entscheidungsbäume haben hohe Varianz.
    - Unterschiedliche Trainingsdaten führen zu unterschiedlichen Ergebnissen.
    - Verfahren mit geringer Varianz liefern ähnliche Ergebnisse bei verschiedenen Datensätzen.
    - Bootstrap Aggregation (Bagging) reduziert Varianz.

  - Bagging:
    - Viele Trainingssätze erstellen, Modell trainieren und Vorhersagen mitteln.
    - Praktisch nicht möglich, daher Bootstrapping:
      - B bootstrapped Trainingssätze erstellen.
      - Modelle auf jedem Satz trainieren und Vorhersagen mitteln.
    - Besonders nützlich für Entscheidungsbäume.

  - Anwendung bei Regressionsbäumen:
    - B Bäume mit tiefen, unbeschnittenen Bäumen erstellen.
    - Vorhersagen mitteln, um Varianz zu reduzieren.
    - Verbessert Genauigkeit durch Kombination vieler Bäume.

  - Anwendung bei Klassifikationsbäumen:
    - Vorhersage jeder Klasse durch Mehrheitswahl der B Bäume.

- **Out-of-Bag Fehlerabschätzung**

  - Einfache Schätzung des Testfehlers ohne Kreuzvalidierung.
  - Jeder Baum nutzt durchschnittlich zwei Drittel der Daten.
  - Die verbleibenden Daten sind Out-of-Bag (OOB) Beobachtungen.
  - Vorhersage für jede OOB Beobachtung durch Mittelung oder Mehrheitswahl.
  - OOB Fehler entspricht Leave-One-Out Kreuzvalidierungsfehler.

- **Variable Importance Measures**

  - Bagging verbessert Vorhersagegenauigkeit, reduziert aber Interpretierbarkeit.
  - Wichtigkeit eines Prädiktors durch RSS (Regressionsbäume) oder Gini-Index (Klassifikationsbäume) messen.
  - Grafische Darstellung der Variablenwichtigkeit zeigt wichtigste Variablen.

- Bagging:
  - Spezialfall von Random Forests
  - Mit `m = p` (alle Variablen werden bei jedem Split berücksichtigt)

```python=
BAGG_boston = RF(max_features=X_train.shape[1], random_state=0).fit(X_train, y_train)
```

- Einstellung `max_features`:
  - Wert auf 12 gesetzt
  - Alle 12 Prädiktoren werden bei jedem Split berücksichtigt
  - Bedeutet, dass wir Bagging verwenden

- Evaluierung:
  - Test-MSE für das Bagging-Modell berechnen
  - Streudiagramme der vorhergesagten vs. tatsächlichen Werte plotten

```python=
print('Training MSE:', np.mean((BAGG_boston.predict(X_train) - y_train) ** 2))
print('Testing MSE:', np.mean((BAGG_boston.predict(X_test) - y_test) ** 2))
ax = plt.subplots(figsize=(8,8))[1]
ax.scatter(BAGG_boston.predict(X_test), y_test);
```

    Training MSE: 1.371
    Testing MSE: 14.635

    
![](Figures\boston_21_1.png)
    

- Test-MSE für den Bagged Regressionsbaum:
  - Wert: 14.63
  - Ungefähr die Hälfte des Wertes eines optimal beschnittenen einzelnen Baums

- Anpassen der Anzahl der Bäume:
  - Standardwert: 100
  - Nutzung des Arguments `n_estimators`

```python=
BAGG_boston = RF(max_features=X_train.shape[1],
                n_estimators=500,
                random_state=0).fit(X_train, y_train)
print('Training MSE:', np.mean((BAGG_boston.predict(X_train) - y_train) ** 2))
print('Testing MSE:', np.mean((BAGG_boston.predict(X_test) - y_test) ** 2))
```

    Training MSE: 1.377
    Testing MSE: 14.605

- Ergebnis bleibt unverändert.
- Keine Überanpassungsgefahr bei Erhöhung der Baumanzahl in Bagging und Random Forests.
- Zu wenige Bäume können jedoch zu Unteranpassung führen.

## 4.2. Random Forest

- Verbesserung von Bagging:
  - Dekorrelieren der Bäume durch zufällige Auswahl von m Prädiktoren bei jedem Split
  - Typischerweise $m \approx \sqrt{p}$

- Verfahren:
  - Entscheidungbäume auf bootstrapped Trainingsproben aufbauen
  - Bei jedem Split wird eine zufällige Auswahl von m Prädiktoren betrachtet
  - Der Split nutzt nur einen dieser m Prädiktoren

- Vorteil:
  - Verhindert, dass starke Prädiktoren alle Splits dominieren
  - Reduziert Korrelation zwischen Bäumen
  - Führt zu geringerer Varianz und zuverlässigeren Vorhersagen

- Hauptunterschied zu Bagging:
  - Größe des Prädiktor-Subsets m
  - Bei m = p entspricht Random Forests dem Bagging

- Beispiel:
  - Bei Herz-Daten: $m = \sqrt{p}$ reduziert Test- und OOB-Fehler im Vergleich zu Bagging
  - Bei biologischen Daten: Vorhersage von Krebsarten basierend auf 500 Genen
    - Fehlerrate eines einzelnen Baums: 45,7%
    - Nullrate: 75,4%
    - Nutzung von 400 Bäumen ausreichend für gute Leistung
    - $m = \sqrt{p}$ verbessert Testfehler geringfügig im Vergleich zu Bagging

- Überanpassung:
  - Random Forests überpassen nicht, wenn B erhöht wird
  - In der Praxis wird ein ausreichend großer Wert von B verwendet, damit sich die Fehlerquote stabilisiert hat

- Wachstum eines Random Forests:
  - Ähnlich wie Bagging, jedoch mit kleinerem `max_features` Wert.

- Standardwerte:
  - `RandomForestRegressor()` nutzt alle $p$ Variablen (entspricht Bagging).
  - `RandomForestClassifier()` nutzt $\sqrt{p}$ Variablen.

- Anpassung:
  - Setzen von `max_features=6`.

```python=
RF_boston = RF(max_features=6,
               random_state=0).fit(X_train, y_train)
print('Training MSE:', np.mean((RF_boston.predict(X_train) - y_train) ** 2))
print('Testing MSE:', np.mean((RF_boston.predict(X_test) - y_test) ** 2))
```

    Training MSE: 1.132
    Testing MSE: 20.043

- Test-MSE für den Random Forest:
  - Wert: 20.04
  - Schlechtere Leistung als Bagging in diesem Fall

- Bestimmen der Variablenwichtigkeit:
  - Nutzung des Attributs `feature_importances_`

```python=
feature_imp = pd.DataFrame(
    {'importance':RF_boston.feature_importances_},
    index=feature_names)
feature_imp.sort_values(by='importance', ascending=False)
```

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>lstat</th>
      <td>0.356203</td>
    </tr>
    <tr>
      <th>rm</th>
      <td>0.332163</td>
    </tr>
    <tr>
      <th>ptratio</th>
      <td>0.067270</td>
    </tr>
    <tr>
      <th>crim</th>
      <td>0.055404</td>
    </tr>
    <tr>
      <th>indus</th>
      <td>0.053851</td>
    </tr>
    <tr>
      <th>dis</th>
      <td>0.041582</td>
    </tr>
    <tr>
      <th>nox</th>
      <td>0.035225</td>
    </tr>
    <tr>
      <th>tax</th>
      <td>0.025355</td>
    </tr>
    <tr>
      <th>age</th>
      <td>0.021506</td>
    </tr>
    <tr>
      <th>rad</th>
      <td>0.004784</td>
    </tr>
    <tr>
      <th>chas</th>
      <td>0.004203</td>
    </tr>
    <tr>
      <th>zn</th>
      <td>0.002454</td>
    </tr>
  </tbody>
</table>

- Ergebnisse:
  - Wichtigste Variablen im Random Forest: Gemeinschaftswohlstand (`lstat`) und Hausgröße (`rm`)

- Grundlage der Messung:
  - Durchschnittliche Verringerung der Knotenverunreinigung bei Splits auf eine bestimmte Variable über alle Bäume im Wald

# 5. Boosting

- Verbesserung von Vorhersagen durch Entscheidungsbäume:
  - Allgemeiner Ansatz für Regression und Klassifikation
  - Hier auf Entscheidungsbäume beschränkt

- Unterschied zu Bagging:
  - Bäume werden sequentiell anstatt unabhängig aufgebaut
  - Keine Bootstrap-Stichproben, sondern modifizierte Datensätze

- Algorithmus für Regression:
  1. Setze ${\hat{f}}{(x)} = 0$ und $r_i = y_i$ für alle i im Trainingssatz.
  2. Für b = 1, 2, ..., B:
     - Passe einen Baum ${\hat{f^b}}$ mit d Splits an die Daten (X, r) an.
     - Aktualisiere ${\hat{f}}$ um eine geschrumpfte Version des neuen Baums: ${\hat{f}}(x) \leftarrow {\hat{f}}(x) + λ {\hat{f^{b}}}{(x)}$.
     - Aktualisiere die Residuen: $r_i \leftarrow r_i - λ {\hat{f^{b}}}{(x_i)}$.
  3. Ausgabe des Modells: ${\hat{f}}{(x)} = \sum_{b=1}^B {λ {\hat{f^{b}}}{(x)}}$.

- Idee:
  - Langsames Lernen durch Anpassen von Bäumen an die Residuen
  - Kleine Bäume verbessern das Modell schrittweise
  - Shrinkage-Parameter λ verlangsamt den Prozess weiter

- Unterschiede zu Bagging:
  - Konstruktion jedes Baums hängt stark von vorherigen Bäumen ab

- Boosting für Klassifikationsbäume:
  - Ähnlich, aber komplexer als bei Regressionsbäumen

- Abstimmungsparameter:
  1. Anzahl der Bäume B:
     - Kann bei zu großer Anzahl überanpassen
     - Kreuzvalidierung zur Auswahl von B
  2. Shrinkage-Parameter λ:
     - Kontrolliert die Lernrate
     - Typische Werte: 0.01 oder 0.001
  3. Anzahl d der Splits pro Baum:
     - Kontrolliert die Komplexität
     - Oft funktioniert d = 1 gut

- Beispiel:
  - Boosting auf 15-Klassen Krebs-Gendaten angewendet
  - Testfehler als Funktion der Baumanzahl und Interaktionstiefe d
  - Einfache Stümpfe (d = 1) performen gut
  - Übertreffen Modell mit Tiefe zwei und Random Forest

- Vorteil kleiner Bäume:
  - Bessere Interpretierbarkeit
  - Additives Modell bei Verwendung von Stümpfen

- Verwendung von `GradientBoostingRegressor()`:
  - Fitten von Boosted Regressionsbäumen

- Verwendung von `GradientBoostingClassifier()`:
  - Durchführung von Klassifikationen

- Argumente:
  - `n_estimators=5000`: Erstellen von 5.000 Bäumen
  - `max_depth=3`: Begrenzung der Tiefe jedes Baums
  - `learning_rate`: Entspricht dem zuvor beschriebenen $\lambda$ beim Boosting

```python=
boost_boston = GBR(n_estimators=5000,
                   learning_rate=0.001,
                   max_depth=3,
                   random_state=0)
boost_boston.fit(X_train, y_train)
print('Training MSE:', np.mean((boost_boston.predict(X_train) - y_train) ** 2))
print('Testing MSE:', np.mean((boost_boston.predict(X_test) - y_test) ** 2))
```

    Training MSE: 2.581
    Testing MSE: 14.481

- Test-MSE is ähnlich wie der MSE für Bagging

- Verfolgen des Trainingsfehlers: `train_score_`

```python=
# Track the decrease in training error
boost_boston.train_score_
```

    array([84.620, 84.476, 84.332, ..., 2.581, 2.581, 2.580])

- Berechnung des Test-MSE:
  - Nutzung der Methode `staged_predict()` um den Test-MSE in jeder Phase des Boostings zu berechnen.

```python=
test_error = np.zeros_like(boost_boston.train_score_)
for idx, y_ in enumerate(boost_boston.staged_predict(X_test)):
   test_error[idx] = np.mean((y_test - y_)**2)
test_error
```

    array([83.633, 83.499, 83.366, ..., 14.482, 14.482, 14.481])

```python=
# Set plot_idx from 1 to 
plot_idx = np.arange(boost_boston.train_score_.shape[0])
# Set the plot up
ax = plt.subplots(figsize=(8,8))[1]
# Plot the training error in blue
ax.plot(plot_idx,
        boost_boston.train_score_,
        'b', label='Training')
# Plot the testing error in red
ax.plot(plot_idx,
        test_error,
        'r', label='Test')
ax.legend();
```

    
![](Figures\boston_37_0.png)
    

- Durchführung von Boosting mit unterschiedlichem Shrinkage-Parameter $\lambda$:
  - Standardwert: 0.001
  - Änderung zu $\lambda=0.2$
  - Ähnliches Ergebnis

```python=
boost_boston = GBR(n_estimators=5000,
                   learning_rate=0.2,
                   max_depth=3,
                   random_state=0)
boost_boston.fit(X_train,
                 y_train)
y_hat_boost = boost_boston.predict(X_test);
np.mean((y_test - y_hat_boost)**2)
```

    14.502


# 6. Bayesian Additive Regression Trees

- Einführung in BART:
  - Ensemble-Methode mit Entscheidungsbäumen
  - Präsentiert für Regression

- Vergleich mit anderen Methoden:
  - Bagging und Random Forests: Vorhersagen aus Durchschnitt von Bäumen, unabhängig gebaut
  - Boosting: Gewichtete Summe von Bäumen, jeder Baum passt auf Residuen des aktuellen Fits
  - BART: Kombination aus beiden, zufälliges Bauen wie Bagging und Random Forests, Signalaufnahme wie Boosting

- Algorithm Notation:
  - K: Number of trees
  - B: Number of iterations
  - ${\hat{f}}^b_k(x)$: Prediction of the k-th tree in the b-th iteration

- **BART Algorithm**:
  1. **Initialization**:
     - All trees start with a single root node.
     - ${\hat{f}}^1_k(x) = \frac{1}{nK} \sum_{i=1}^{n} y_i$
  2. **Iterations**:
     - Update each of the K trees sequentially.
     - Compute partial residuals $r_i = y_i - \sum_{k^\prime<k}{{\hat{f}}_{k'}^b(x_i)} - \sum_{k^\prime>k}{{\hat{f}}_{k'}^{b-1}(x_i)}$
     - Perturb the previous tree ${\hat{f}}_{k}^{b-1}$ for a better fit.
  3. **Perturbations**:
     - Change tree structure by adding or pruning branches.
     - Change predictions in terminal nodes.

- Output:
  - Collection of prediction models ${\hat{f}}^b(x) = \sum_{k=1}^{K}{{\hat{f}}_k^b(x)}$
  - Average predictions after burn-in period: $\hat{f}(x) = \frac{1}{B-L} \sum_{b=L+1}^{B}{{\hat{f}}^b(x)}$

- Key Points:
  - Avoid overfitting by slight modifications of trees.
  - Trees are small to prevent overfitting.
  - Burn-in period helps stabilize error rates.

- Example:
  - Applied to Heart data with K = 200 trees and B = 10,000 iterations.
  - Test and training errors stabilize after burn-in.
  - BART shows small difference between training and test errors, indicating minimal overfitting.

- Boosting Comparison:
  - Boosting test error increases with more iterations, indicating overfitting.
  - BART maintains stable error rates.

- Bayesian Perspective:
  - BART can be viewed as drawing trees from a posterior distribution.
  - Algorithm 8.3 resembles a Markov chain Monte Carlo algorithm.

- Parameter Selection:
  - Choose large B and K, and moderate L (burn-in iterations).
  - Example: K = 200, B = 1,000, L = 100
    - The final model bases on 900 iterations.
  - BART performs well with minimal tuning.

- Verwendung von `BART()` aus dem Paket `ISLP.bart`:
  - Fitten eines Bayesian Additive Regression Tree Modells

- Zweck:
  - `BART()` für quantitative Ergebnisvariablen
  - Andere Implementierungen existieren für logistische und Probit-Modelle bei kategorialen Ergebnissen

- Argumente:
  - `burnin`: Anzahl der Iterationen, die als Burn-in verworfen werden
  - `ndraw`: Anzahl der Iterationen, die nach dem Burn-in beibehalten werden

```python=
# Set and fit a BART model
bart_boston = BART(random_state=0, burnin=5, ndraw=15).fit(X_train, y_train)
```

```python=
print('Training MSE:', np.mean((bart_boston.predict(X_train.astype(np.float32)) - y_train) ** 2))
print('Testing MSE:', np.mean((bart_boston.predict(X_test.astype(np.float32)) - y_test) ** 2))
```

    Training MSE: 6.613
    Testing MSE: 22.145

- **Test MSE**:
  - The test MSE for BART is comparable to that of the random forest.

- **Variable Importance**:
  - Check the frequency of each variable's appearance across all trees.
  - This is similar to the variable importance measure in random forests.
  - Provides a summary similar to the variable importance plot for boosting and random forests.

```python=
variable_inclusion = pd.Series(bart_boston.variable_inclusion_.mean(0), index=Design_matrix.columns)
variable_inclusion.sort_values(ascending=False)
```

    lstat      31.000000
    rm         29.800000
    zn         27.866667
    crim       26.933333
    nox        26.600000
    indus      26.466667
    dis        26.466667
    ptratio    24.266667
    tax        24.133333
    rad        23.666667
    age        22.733333
    chas       22.466667
    dtype: float64

# 7. Zusammenfassung

- **Vorteile von Bäumen als schwache Lernende**:
  - Flexibilität
  - Umgang mit qualitativen und quantitativen Prädiktoren

- **Methoden zur Anpassung eines Ensemble von Bäumen**:
  - Bagging
  - Random Forests
  - Boosting
  - BART

- **Bagging**:
  - Bäume wachsen unabhängig auf zufälligen Stichproben der Beobachtungen.
  - Bäume sind ähnlich und können in lokalen Optima stecken bleiben.

- **Random Forests**:
  - Bäume wachsen unabhängig auf zufälligen Stichproben der Beobachtungen.
  - Jede Teilung erfolgt mit einer zufälligen Teilmenge der Merkmale.
  - Dekorreliert die Bäume und erkundet den Modellraum gründlicher als Bagging.

- **Boosting**:
  - Verwendet nur die Originaldaten, keine zufälligen Stichproben.
  - Bäume wachsen nacheinander mit „langsamem“ Lernansatz.
  - Jeder neue Baum passt sich dem verbleibenden Signal der vorherigen Bäume an und wird geschrumpft.

- **BART**:
  - Verwendet nur die Originaldaten.
  - Bäume wachsen nacheinander.
  - Jeder Baum wird zufällig verändert, um lokale Minima zu vermeiden und den Modellraum gründlicher zu erkunden.