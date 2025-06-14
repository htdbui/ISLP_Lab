## Im Allgemeinen

**Klassiﬁkations- und Regressionsmethoden**

- Thema:
  - Vorhersage einer Zielgröße (y) durch Prädiktoren (Merkmale)
  - Lernen des Zusammenhangs durch Beobachtungen von Zielgröße und Prädiktoren

- **Regressionsprobleme**:
  - Zielvariable y nimmt kontinuierliche Werte an
  - Beispiel: Vorhersage von Einkommen

- **Klassifikationsprobleme**:
  - Zielvariable y nimmt endliche Werte an ($y \in \{C_1, ..., C_n\}$)
  - Beispiele: Storno / kein Storno, Spam / kein Spam
  - Ausprägungen von y werden als Klassen bezeichnet

**Arten von Variablen**

- **Numerische Variablen**:
  - Unterteilung in diskrete und stetige Variablen
  - Diskrete Variablen: Abzählbare Werte (z.B. Anzahl der Kinder)
  - Stetige Variablen:
    - **Intervallskalierte Variablen**: Differenzen haben Bedeutung (z.B. Temperatur in °C)
    - **Verhältnisskalierte Variablen**: Bedeutsame Quotienten und sinnvoller Nullpunkt (z.B. Herzfrequenz)

- **Kategorielle Variablen**:
  - Unterteilung in ordinalskalierte und nominalskalierte Variablen
  - **Nominalskalierte Variablen**:
    - Keine lineare Ordnung (z.B. Geschlecht: 0/1 oder M/F)
    - Vorsicht bei mehr als zwei Gruppen, um keine falsche Ordinalordnung zu implizieren
  - **Ordinalskalierte Variablen**:
    - Ausprägungen können geordnet werden (z.B. Schulnoten)

## Multiple Lineare Regression

- Grundlagen
  - Variablen:
    - $p$ erklärende Variablen $X_1, ..., X_p$
    - Stetige Zielvariable $Y$
  - Ziel:
    - Zusammenhang zwischen $X$ und $Y$ bestimmen: $Y = f(X) + \epsilon$
    - $\epsilon$ repräsentiert Störgrößen (meist additive Annahme)
  - Schätzaufgabe:
    - Beste Schätzung der unbekannten Funktion $f$
    - Trennung zwischen systematischer Komponente $f$ und Fehler $\epsilon$
  - Designmatrix $X$:
    - $X \in R^{n \times (p+1)}$
    - Zeilenvektoren: $x_i$
    - Spalten: $x_1, ..., x_{p+1}$
      $$
      \left(\begin{matrix}
      1&x_{11}&\cdots&x_{1p}\\
      \vdots&\vdots&\ddots&\vdots\\
      1&x_{n1}&\cdots&x_{np}\\
      \end{matrix}\right)
      $$
  - Bedingung:
    - Die Designmatrix $X$ hat vollen Spaltenrang (linear unabhängige Spalten)
  - Generalized Linear Models (GLM):
    - Anwendung in Risikobewertung und überwachten Lernen
    - Typische Vertreter: lineare, logistische, Poisson-Regression
    - Fokus hier: lineares Regressionsmodell

- Lineares Regressionsmodell
  - Modellformulierung:
    - $Y = X\beta + \epsilon$
    - $\epsilon$ repräsentiert die Fehlerterme
  - Annahmen:
    - $\mathbb{E}(\epsilon) = 0$ (Störungen im Mittel 0)
    - $Var(\epsilon) = \sigma^2$ (konstante Varianz, Homoskedastizität)
    - $Cov(\epsilon_i, \epsilon_j) = 0$ für $i \neq j$ (unkorrelierte Fehler)
    - Normalverteilte Fehler: $\epsilon \sim N(0, \sigma^2I)$
  - Eigenschaften:
    - $\mathbb{E}(Y_i) = \beta_0 + \beta_1 x_{i1} + \ldots + \beta_p x_{ip}$
    - $Var(Y_i) = \sigma^2$
    - $Cov(Y_i, Y_j) = 0$ für $i \neq j$
  - Matrixnotation:
    - $\mathbb{E}(Y) = X\beta$
    - $Cov(Y) = \sigma^2 I$
    - $Y \sim N(X\beta, \sigma^2 I)$

- Schätzung im linearen Modell
  - Schätzer der Modellkoeffizienten:
    - $\hat{\beta} = {(X^TX)}^{-1} X^TY$
    - Verteilung: $\hat{\beta} \sim N(\beta, \sigma^2 (X^TX)^{-1})$
  - Schätzer für die Varianz:
    - $\hat{\sigma}^2 = \frac{1}{n - (p + 1)} \sum_{i=1}^n (y_i - (X\hat{\beta})_i)^2$
    - $(n - (p+1)) \hat{\sigma}^2 \sim \sigma^2 \chi^2_{n-(p+1)}$
    - $\frac{1}{\sigma^2} \sum_{i=1}^n (y_i - (X \hat{\beta})_i)^2 \sim \chi^2_{n-(p+1)}$

- Statistische Tests & Konfidenzintervalle
  - Standardfehler:
    - $se(\hat{\beta}_j) = \hat{\sigma} \sqrt{(X^TX)^{-1}_{jj}}$
  - Hypothesentest für einzelnen Koeffizienten:
    - Nullhypothese $H_0: \beta_j = 0$, Alternative $H_a: \beta_j \neq 0$
    - Teststatistik: $z_j = \frac{\hat{\beta}_j}{se(\hat{\beta}_j)}$ mit $z_j \sim t(n-(p+1))$
  - Globaler F-Test:
    - Nullhypothese $H_0: \beta_1 = \beta_2 = ... = \beta_p = 0$
    - Alternative: Mindestens ein $\beta_j \neq 0$
    - Keine Einfluss der Kovariablen unter H0: $\hat{\beta}_0 = \bar{y}$
    - Teststatistik: $F = \frac{(TSS - RSS) / p}{RSS / (n - p - 1)}$, $F \sim F_{p, n-p-1}$
  - Konfidenzintervall:
    - $\hat{\beta}_j \pm t_{n-p-1,1-\alpha/2} \cdot se(\hat{\beta}_j)$

- Modellgüte & Residuenanalyse
  - Homoskedastizität:
    - Residuen: $\hat{\epsilon} = y - \hat{y} = y - X\hat{\beta}$
    - Residuen sind oft nicht homoskedastisch.
    - Varianz der Residuen: $Var(\hat{\epsilon}_i) = \sigma^2 (1 - h_{ii})$
    - Standardisierte Residuen: $r_i = \frac{\hat{\epsilon}_i}{\hat{\sigma}\sqrt{1-h_{ii}}}$

- Modellerweiterungen
  - Polynomiale Regression:
    - Modell: $Y = \beta_0 + \beta_1 X + \beta_2 X^2 + \epsilon$
    - Dient zur Reduktion von Residuen-Trends und Streuung
    - Q-Q-Plot zur Prüfung der Normalverteilung
  - Transformationen:
    - Bei nichtlinearem Zusammenhang, z.B. $\ln(X)$ oder $\sqrt{X}$

## Lineare Diskriminanzanalyse

- $\pi_k$ repräsentiert die priori Wahrscheinlichkeit, dass eine Beobachtung zur k-ten Klasse gehört. Zur Schätzung von $\pi_k$ berechnen wir den Anteil der Beobachtungen im Trainingssatz, die in die k-te Klasse fallen
- $f_k(X) \equiv \Pr(X | Y = k)$ repräsentiert die Dichtefunktion von X für die k-te Klasse. $f_k(x)$ ist groß, wenn eine Beobachtung in der k-ten Klasse wahrscheinlich $X \approx x$ hat, und klein, wenn es unwahrscheinlich ist
- $p_k(x) = \Pr(Y = k | X = x)$ ist die posteriori Wahrscheinlichkeit, dass eine Beobachtung $X = x$ zur k-ten Klasse gehört
  $\Pr(Y = k | X = x) = \frac{\pi_k f_k(x)}{\sum_{\ell=1}^{K} \pi_\ell f_\ell(x)}$
- Zur Schätzung von $f_k(x)$ müssen wir vereinfachende Annahmen treffen
- Annahme: $X = (X_1, X_2, \ldots, X_p)$ ist multivariat normalverteilt, mit klassenspezifischem Mittelwert und gemeinsamer Kovarianzmatrix
  - $f(\mathbf{x}) = \frac{1}{(2\pi)^\frac{p}{2} \left|\mathbf{\Sigma}\right|^\frac{1}{2}} \exp{\left(-\frac{1}{2} (\mathbf{x} - \mathbf{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \mathbf{\mu})\right)}$
  - Voraussetzung: jeder Prädiktor folgt einer eindimensionalen Normalverteilung
  - Die Oberfläche hat eine Glockenform bei $\mathrm{var}(X_1) = \mathrm{var}(X_2)$ und $\mathrm{cor}(X_1, X_2)=0$
  
```python=
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
LDA(store_covariance=True).fit(X, y)
# X does not have an intercept column
# y is response with two labels (not 0 and 1)
```

## Quadratische Diskriminanzanalyse

- LDA: Beobachtungen aus multivariater Gaußverteilung, gleiche Kovarianzmatrix für alle Klassen
- QDA: Beobachtungen aus Gaußverteilung, jede Klasse eigene Kovarianzmatrix $\Sigma_k$
- Unterschied LDA vs. QDA: Bias-Varianz-Kompromiss.
  - LDA: gemeinsame Kovarianzmatrix, weniger flexibel, niedrigere Varianz, gut bei wenigen Trainingsbeobachtungen
  - QDA: separate Kovarianzmatrizen, höhere Varianz, besser bei großen Trainingssätzen

```python=
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
QDA(store_covariance=True).fit(X, y)
# X does not have an intercept column
# y is response with two labels (not 0 and 1)
```

## Naive Bayes

- Schätzt $\pi_1, \ldots, \pi_K$ als Anteil der Beobachtungen pro Klasse
- Schätzt $f_1(x), \ldots, f_K(x)$ unter Annahme unabhängiger Prädiktoren innerhalb jeder Klasse
  - Quantitativ: $(X_j \mid Y = k) \sim N(\mu_{jk}, \sigma_{jk}^2)$ oder mit nichtparametrischen Methoden (Histogramm, Kernel Density Estimator)
  - Qualitativ: Anteil der Trainingsbeobachtungen für jede Klasse
- $f_k(x) = f_{k1}(x_1) \times f_{k2}(x_2) \times \cdots \times f_{kp}(x_p)$
- Vereinfachung durch Unabhängigkeitsannahme: keine Berücksichtigung der gemeinsamen Verteilung der Prädiktoren
- gute Ergebnisse, besonders bei kleinen n im Vergleich zu p

```python=
from sklearn.naive_bayes import GaussianNB
GaussianNB().fit(X, y)
# X does not have an intercept column
# y is response with two labels (not 0 and 1)
```

## Ridge Regression

- Minimiert $\sum_{i=1}^{n} \left( y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij} \right)^2 + \lambda \sum_{j=1}^{p} \beta_j^2 = \text{RSS} + \lambda \sum_{j=1}^{p} \beta_j^2$
  - Zwei Kriterien:
    - RSS klein halten
    - Schrumpfungsstrafe $\lambda \sum_{j} \beta_j^2$
  - $\lambda$ kontrolliert die Balance:
    - $\lambda = 0$: Gleich wie Least Squares
    - $\lambda \to \infty$: Schätzungen nähern sich null
- Schrumpfungsstrafe:
  - Gilt für $\beta_1, \ldots, \beta_p$, nicht für $\beta_0$
  - ${\hat{\beta}}_0 = \bar{y} = \sum_{i=1}^{n} y_i/n$
  - $\beta_0$ misst den Mittelwert bei $x_{i1} = x_{i2} = \ldots = x_{ip} = 0$
- ℓ₂ Norm:
  - Die Quadratwurzel der Summe der Quadrate seiner Einträge von einem Vektor
  - Die ℓ₂-Norm der Ridge-Regressionskoeffizienten: $\sum_{j=1}^{p} \beta_j^2$
- Bias-Variance Trade-off: $\lambda$ erhöhen -> Flexibilität und Varianz sinken, Bias steigt
  - Least Squares: Niedriger Bias, hohe Varianz
  - Ridge Regression: Geringere Varianz, etwas höherer Bias
- Gründe für die Skalierung:
  - Verlustfunktion = RSS + Regularisierungsterm 
    - Große y-Skala: Große RSS, kleinerer Effekt des Regularisierungsterms
    - Kleine y-Skala: Kleine RSS, größerer Effekt des Regularisierungsterms
  - Skalierung des Regularisierungsterms mit der Standardabweichung von y sorgt für einen konsistenten Effekt auf die Verlustfunktion, unabhängig von der y-Skala

```python=
ridge = skl.Ridge(alpha=0.01)
# skl.ridge_regression(alpha=0.01)
# skl.ElasticNet(alpha=0.01, l1_ratio=0)
scaler = StandardScaler(with_mean=True,  with_std=True)
pipe = Pipeline(steps=[('scaler', scaler), ('ridge', ridge)])
pipe.fit(X, Y)
ridge.coef_
```

```python=
skl.ElasticNet.path(X, Y, l1_ratio=0., alphas=lambdas)
# X is standardized design matrix
```

```python=
validation = skm.ShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
ridge = skl.Ridge(alpha=0.01)
results = skm.cross_validate(ridge, X, Y, cv=validation, scoring='neg_mean_squared_error')
-results['test_score']
```

```python=
param_grid = {'ridge__alpha': lambdas}
ridge = skl.ElasticNet(l1_ratio=0)
scaler = StandardScaler(with_mean=True,  with_std=True)
pipe = Pipeline(steps=[('scaler', scaler), ('ridge', ridge)])
grid = skm.GridSearchCV(pipe, param_grid, cv=validation, scoring='neg_mean_squared_error')
grid.fit(X, Y)
best_model = grid.best_estimator_
best_model.named_steps['ridge'].coef_
```

```python=
kfold = skm.KFold(n_splits=5, shuffle=True, random_state=0)
grid = skm.GridSearchCV(pipe, param_grid, cv=kfold, scoring='neg_mean_squared_error')
```

```python=
ridgeCV = skl.ElasticNetCV(alphas=lambdas, l1_ratio=0, cv=kfold)
ridgeCV.fit(Xs, Y)
ridgeCV.coef_
ridgeCV.alpha_
```

```python=
ridgeCV = skl.RidgeCV(alphas=lambdas, store_cv_results = True)
ridgeCV.fit(Xs, Y)
```

- GridSearchCV:
  - Mittelwert und Standardabweichung in jedem Trainingssatz berechnet
- CV:
  - Mittelwert und Standardabweichung im gesamten Datensatz berechnet
- Unterschiede zwischen GridSearchCV und CV wegen Standardisierung

## Lasso Regression

- Minimiert $\sum_{i=1}^{n} \left( y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij} \right)^2 + \lambda \sum_{j=1}^{p} |\beta_j| = \text{RSS} + \lambda \sum_{j=1}^{p} |\beta_j|$
- Verwendet ℓ₁-Strafe statt ℓ₂-Strafe
  - ℓ₁-Norm: $\|\beta\|_1 = \sum |\beta_j|$
  - **Vorteil:**
    - Schrumpft Koeffizienten auf null bei großem $\lambda$
    - Führt zu variablenselektiven Modellen -> sparsame Modelle

```python=
param_grid = {'lasso__alpha': lambdas}
lasso = skl.ElasticNet(l1_ratio=1)
scaler = StandardScaler(with_mean=True,  with_std=True)
pipe = Pipeline(steps=[('scaler', scaler), ('lasso', lasso)])
grid = skm.GridSearchCV(pipe, param_grid, cv=validation, scoring='neg_mean_squared_error')
grid.fit(X, Y)
```

```python=
lassoCV = skl.ElasticNetCV(alphas=lambdas, l1_ratio=1, cv=kfold)
pipeCV = Pipeline(steps=[('scaler', scaler), ('lasso', lassoCV)]).fit(X, Y)
```

```python=
lassoCV = skl.LassoCV(n_alphas=100, cv=kfold)
pipeCV = Pipeline(steps=[('scaler', scaler), ('lasso', lassoCV2)]).fit(X, Y)
```

**Vergleich Ridge, Lasso**

- Lasso:
  - Minimiert $\sum_{i=1}^{n} \left( y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij} \right)^2$
  - Einschränkung: $\sum_{j=1}^{p} |\beta_j| \leq s$.
  - Kleinster RSS innerhalb eines Diamanten ($| \beta_1 | + | \beta_2 | \leq s$ bei $p = 2$)
- Ridge:
  - Minimiert $\sum_{i=1}^{n} \left( y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij} \right)^2$
  - Einschränkung: $\sum_{j=1}^{p} \beta_j^2 \leq s$.
  - Kleinster RSS innerhalb eines Kreises ($\beta_1^2 + \beta_2^2 \leq s$ bei $p = 2$)
- Best Subset Selection:
  - Minimiert $\sum_{i=1}^{n} \left( y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij} \right)^2$
  - Einschränkung: $\sum_{j=1}^{p} I(\beta_j \neq 0) \leq s$
  - Kleinster RSS mit maximal s nicht-null Koeffizienten
- Ellipsen und Einschränkungen:
  - Ellipsen um $\hat{\beta}$ repräsentieren RSS-Konturen.
  - Lasso und Ridge Schätzungen sind der Punkt, an dem eine Ellipse die Einschränkung zuerst berührt
  - Ridge hat keine scharfen Ecken, daher sind die Koeffizienten nicht null
  - Lasso hat Ecken an den Achsen, daher können einige Koeffizienten null sein
  - p=3: Einschränkung für Ridge eine Kugel, für Lasso ein Polyeder
  - p>3: Einschränkung für Ridge eine Hypersphäre, für Lasso ein Polytop
- Einfache Spezialfall:
  - Diagonale Matrix: n = p
  - Least Squares-Lösung: $\hat{\beta}_j = y_j$
  - Ridge: $\hat{\beta}^R_j = \frac{y_j}{1 + \lambda}$
    - Ridge schrumpft alle Koeffizienten gleichmäßig.
  - Lasso: $\hat{\beta}^L_j = \begin{cases} y_j - \lambda/2 & \text{if } y_j > \lambda/2 \\ y_j + \lambda/2 & \text{if } y_j < -\lambda/2 \\ 0 & \text{if } |y_j| \leq \lambda/2 \end{cases}$
    - Lasso schrumpft Koeffizienten um einen konstanten Betrag $\lambda/2$; kleine Koeffizienten werden zu null
- Bayesianische Interpretation:
  - Ridge: Prior ist eine normale Verteilung
    - Ridge Lösung ist der Posterior-Modus bei normaler Prior
  - Lasso: Prior ist eine doppelt-exponentielle Verteilung
    - Lasso Lösung ist der Posterior-Modus bei doppelt-exponentieller Prior
    - Gaussian-Prior ist flacher und breiter bei null. Lasso-Prior ist steil bei null, erwartet viele Koeffizienten als null

## Hauptkomponenten

- Hauptkomponenten sind transformierten Prädiktoren.
- $Z_1, Z_2, \ldots, Z_M$ sind lineare Kombinationen der originalen Prädiktoren $X_1, X_2, \ldots, X_p$
  - $Z_m = \sum_{j=1}^{p} \phi_{jm} X_j$
  - $Var(Z_1)$ ist am größten. $\phi^{'}_{1}\phi_{1}=1$
  - $Var(Z_2)$ ist zweit größten. $\phi^{'}_{2}\phi_{2}=1$ und $\phi^{'}_{2}\phi_{1}=0$
  - $\phi_{j1}, ..., \phi_{jp}$ ist loading Vektor
    - $\Phi=\left(\begin{matrix}\phi_{11}&\ldots&\phi_{m1}\\\vdots&\ddots&\vdots\\\phi_{1p}&\cdots&\phi_{mp}\\\end{matrix}\right)$ 
  - $z_{1m}, ..., z_{nm}$ ist score Vektor
    - $Z=\left(\begin{matrix}z_{11}&\ldots&z_{1m}\\\vdots&\ddots&\vdots\\z_{n1}&\cdots&z_{nm}\\\end{matrix}\right)$
  - $X \approx Z \times \Phi^{'}$
    - $X = Z \times \Phi^{'}$ wenn m = p
- Standardisierung der Prädiktoren empfohlen, um gleiche Skala zu gewährleisten.

```python=
scaler = StandardScaler(with_mean=True,  with_std=True)
pca = PCA(n_components=2)
linreg = skl.LinearRegression()
pipe = Pipeline([('scaler', scaler),('pca', pca),('linreg', linreg)])
pipe.fit(X, Y)
```

```python=
scaler = StandardScaler(with_mean=True,  with_std=True)
pca = PCA()
linreg = skl.LinearRegression()
param_grid = {'pca__n_components': range(1, 20)}
pipe_grid = Pipeline([('scaler', scaler),('pca', pca),('linreg', linreg)])
grid = skm.GridSearchCV(pipe_grid, param_grid, cv=kfold, scoring='neg_mean_squared_error')
grid.fit(X, Y)
```

- Der Anteil der erklärten Varianz - Proportion of variance explained (PVE):
  - Nach der Zentrierung der Variablen ist die Gesamtvarianz in einem Datensatz:
    - $\sum_{j=1}^{p} \text{Var}(X_j) = \sum_{j=1}^{p} \frac{1}{n} \sum_{i=1}^{n} x_{ij}^2$
  - Die durch die m-te Hauptkomponente erklärte Varianz ist:
    - $\frac{1}{n} \sum_{i=1}^{n} z_{im}^2 = \frac{1}{n} \sum_{i=1}^{n} (\sum_{j=1}^{p} \phi_{jm} x_{ij})^2$
  - Der PVE (Anteil der erklärten Varianz) der m-ten Hauptkomponente ist:
    - $\frac{\sum_{i=1}^{n} z_{im}^2}{\sum_{j=1}^{p} \sum_{i=1}^{n} x_{ij}^2} = \frac{\sum_{i=1}^{n} (\sum_{j=1}^{p} \phi_{jm} x_{ij})^2}{\sum_{j=1}^{p} \sum_{i=1}^{n} x_{ij}^2}$
  - Um den kumulativen PVE der ersten M Hauptkomponenten zu berechnen:
    - Summiere über jeden der ersten M PVEs
    - Die PVEs aller möglichen Hauptkomponenten summieren sich zu eins
  - Die Varianz der Daten kann zerlegt werden:
    - $\underbrace{\sum_{j=1}^{p} \frac{1}{n} \sum_{i=1} x_{ij}^2}_{\text{Var. der Daten}} = \underbrace{\sum_{m=1}^{M} \frac{1}{n} \sum_{i=1}^{n} z_{im}^2}_{\text{Var. der ersten M HKs}} + \underbrace{\frac{1}{n} \sum_{j=1}^{p} \sum_{i=1}^{n} (x_{ij} - \sum_{m=1}^{M} z_{im} \phi_{jm})^2}_{\text{MSE der M-dimensionalen Approximation}}$
    - Da der erste Term fest ist: Durch Maximierung der Varianz der ersten M Hauptkomponenten minimieren wir den mittleren quadratischen Fehler der M-dimensionalen Approximation und umgekehrt
    - Hauptkomponenten können äquivalent als Minimierung des Approximationsfehlers oder Maximierung der Varianz betrachtet werden
  - Der PVE entspricht:
    - $1 - \frac{\sum_{j=1}^{p} \sum_{i=1}^{n} ( x_{ij} - \sum_{m=1}^{M} z_{im} \phi_{jm} )^2}{\sum_{j=1}^{p} \sum_{i=1}^{n} x_{ij}^2} = 1 - \frac{\text{RSS}}{\text{TSS}}$
    - Wir können den PVE als das $R^2$ der Approximation für X durch die ersten M Hauptkomponenten interpretieren

- Sobald wir die Hauptkomponenten berechnet haben, können wir sie gegeneinander darstellen, um niedrigdimensionale Ansichten der Daten zu erzeugen
- Zum Beispiel können wir darstellen:
  - Den Scorevektor $Z_1$ gegen $Z_2$
  - $Z_1$ gegen $Z_3$
  - $Z_2$ gegen $Z_3$
  - Und so weiter
- Biplot: Darstellung von zwei Hauptkomponentenwerten und den Hauptkomponentenladungen
  - Die Variablen liegen nahe beieinander, wenn sie miteinander korreliert sind

- Algorithmus: Iterativer Algorithmus für Matrix-Vervollständigung
  1. Erstelle eine vollständige Datenmatrix $\tilde{X}$ der Dimension $n \times p$, wobei das $(i, j)$-Element gleich $\tilde{x}_{ij} = \begin{cases} x_{ij} & \text{wenn } (i, j) \in O \\ \bar{x}_j & \text{wenn } (i, j) \notin O, \end{cases}$ ist, wobei $\bar{x}_j$ der Durchschnitt der beobachteten Werte für die $j$-te Variable in der unvollständigen Datenmatrix $X$ ist. Hier indiziert $O$ die Beobachtungen, die in $X$ vorhanden sind
  2. Wiederhole die Schritte (a)-(c), bis sich das Ziel (12.14) nicht mehr verringert: 
		- a. Löse $\text{minimize}_{A \in \mathbb{R}^{n \times M}, B \in \mathbb{R}^{p \times M}} \{ \sum_{j=1}^{p} \sum_{i=1}^{n} ( \tilde{x}_{ij} - \sum_{m=1}^{M} a_{im} b_{jm} )^2 \}$ (12.13) durch Berechnung der Hauptkomponenten von $\tilde{X}$
		- b. Für jedes Element $(i, j) \notin O$, setze $\tilde{x}_{ij} \leftarrow \sum_{m=1}^{M} \hat{a}_{im} \hat{b}_{jm}$
		- c. Berechne das Ziel $\sum_{(i,j) \in O} ( x_{ij} - \sum_{m=1}^{M} \hat{a}_{im} \hat{b}_{jm} )^2$ (12.14)
  3. Gib die geschätzten fehlenden Einträge $\tilde{x}_{ij}$, $(i, j) \notin O$ zurück

## Regression der partiellen kleinsten Quadrate

- PCR ist eine unüberwachte Methode
- PLS ist eine überwachte Methode. Nutzt Y, um neue Merkmale zu identifizieren, die sowohl die Prädiktoren als auch die Antwort gut erklären
- Bestimmung $Z_1$:
  - Standardisieren der p Prädiktoren
  - Setzen jedes $\phi_{j1}$ auf den Koeffizienten der einfachen linearen Regression von Y auf $X_j$. Wir führen p Regression durch, um p $\phi_{j1}$ zu kakulieren
  - Berechnung von $Z_1 = \sum_{j=1}^{p} \phi_{j1} X_j$ priorisiert Variablen mit der stärksten Verbindung zur Antwortvariable
- Berechnung $Z_2$:
  - Regression jedes $X_j$ auf $Z_1$ und Berechnung der Residuen
  - Verwendung dieser Residuen als neuen Datensatz
  - Standardisieren der p Residualwerte
  - Durchführung einfacher linearer Regressionen Y auf jede standardisierter Residuen
  - Verwendung der Koeffizienten aus diesen Regressionen als $\phi_{j2}$ Werte: $Z_2 = \sum_{j=1}^{p} \phi_{j2} X_j$
- Wiederhole den Prozess *M* Mal, um $Z_1, …, Z_M$ zu bestimmen
- Durchführung linearer Regressionen Y auf $Z_1, …, Z_M$. Die Koeffizienten sind $\theta_1, ..., \theta_m$
- $\beta_j = \sum_{m=1}^{M} \theta_m \phi_{jm}$
  - $\beta_j$: Regressionskoeffizient für den  j-ten Prädiktor ($X_j$)
  - $\theta_m$: Regressionskoeffizient für die m-te PLS-Komponente ($Z_m$)
  - $\phi_{jm}$: Gewicht von $X_j$ auf $Z_m$

```python=
pls = PLSRegression(n_components=2, scale=True)
pls.fit(X, Y)
```

```python=
param_grid = {'n_components':range(1, 20)}
pls = PLSRegression(scale=True)
grid = skm.GridSearchCV(pls, param_grid, cv=kfold, scoring='neg_mean_squared_error')
grid.fit(X, Y)
```

## Spline

**Stückweise (piecewise) Polynome**
- Ansatz: separate Polynome niedrigen Grades über verschiedene Regionen von X anpassen 
- Beispiel:
  - Kubisches Polynom: $y_i = \beta_0 + \beta_1 x_i + \beta_2 x_i^2 + \beta_3 x_i^3 + \epsilon_i$
  - $\beta_0, \ldots, \beta_3$ variieren in verschiedenen X-Bereichen
  - Beispiel mit Knoten bei c:
    - Für $x_i < c$: $y_i = \beta_{01} + \beta_{11} x_i + \beta_{21} x_i^2 + \beta_{31} x_i^3 + \epsilon_i$
    - Für $x_i \geq c$: $y_i = \beta_{02} + \beta_{12} x_i + \beta_{22} x_i^2 + \beta_{32} x_i^3 + \epsilon_i$
- Koeffizienten:
  - Zwei Polynome anpassen: eins für $x_i < c$, eins für $x_i \geq c$
  - Schätzung mit Methode der kleinsten Quadrate
- Flexibilität:
  - Mehr Knoten = flexiblere Polynome
  - (K + 1) Polynome für K Knoten
- Einschränkungen für kontinuierliche Kurve: 
  - Ein d-Grad Polynom: Kontinuität der Ableitungen bis zum Grad d-1 an jedem Knoten
    - Ein kubisches Polynom: Kontinuität, Kontinuität der ersten und zweiten Ableitung
    - Ein quadratisches Polynom: Kontinuität, Kontinuität der ersten Ableitung
	- Ein lineares Polynom: Kontinuität
	- Eine konstante Linie: 0
- Freiheitsgrade:
  - Ein kubisches Polynom benötigt 4 Freiheitsgrade
    - Berechnung:
      - 4(K + 1) Freiheitsgrade für K + 1 kubische Segmente
      - Abzug von 3K Freiheitsgraden
      - Ergebnis: 4(K + 1) - 3K = K + 4
  - Ein quadratisches Polynom benötigt 3 Freiheitsgrade
    - Ergebnis: 3(K + 1) - 2K = K + 3
  - Eine lineares Polynom benötigt 2 Freiheitsgrade
    - Ergebnis: 2(K + 1) - K = K + 2
  - Eine konstante Linie benötigt 1 Freiheitsgrade
    - Ergebnis: K + 1

**Basis-Spline**

- Kubischer Spline mit K Knoten:
  - $y_i = \beta_0 + \beta_1 x_i + \beta_2 x_i^2 + \beta_3 x_i^3 + \beta_4 h(x, \xi_1) + \ldots + \beta_{K+3} h(x, \xi_K) + \epsilon_i$
  - Truncated Power Basisfunktion pro Knoten:
    - $h(x, \xi) = (x - \xi)_+^3 = \begin{cases} (x - \xi)^3 & \text{wenn } x > \xi \\ 0 & \text{sonst} \end{cases}$
  - Es gibt 4 + K Koeffizienten -> 4 + K Freiheitsgrade

```python=
from ISLP.transform import BSpline
ageBSDF = BSpline(internal_knots=[25,40,60], intercept=True).fit_transform(WageDF.age)
sm.OLS(WageDF.wage, ageBSDF).fit()
```

**Natürlicher Spline**

- Basis-Splines haben hohe Varianz an den Rändern der Prädiktoren, wenn X sehr klein oder sehr groß ist.
- Ein natürlicher Spline hat zusätzliche Randbedingungen: Funktion ist linear am Rand.
  - Ergebnis: 4 + K - 2 = 2 + K

```python=
from ISLP.transform import NaturalSpline
ageNSDF = NaturalSpline(internal_knots=[25,40,60], intercept=True).fit_transform(WageDF.age)
sm.OLS(WageDF.wage, ageNSDF).fit()
```

**Glättungsspline (Smoothing Spline)**

- Minimierung von $\sum_{i=1}^{n} (y_i - g(x_i))^2 + \lambda \int (g''(t))^2 \, dt$
  - $\lambda \geq 0$: Tuning-Parameter zur Kontrolle der Glätte
    - $\lambda = 0$: Die Funktion interpoliert die Daten exakt
    - $\lambda \to \infty$: Die Funktion ist eine gerade Linie
    - $\lambda$ steuert den Bias-Varianz-Kompromiss. Je größer der Wert von $\lambda$, desto glatter wird $g$
  - $\sum_{i=1}^{n} (y_i - g(x_i))^2$ ist eine Verlustfunktion. Sie sorgt dafür, dass $g$ gut an die Daten angepasst wird
  - $\lambda \int (g''(t))^2 dt$ bestraft die Variabilität von $g$
    - $g''(t)$ misst, wie stark sich die Steigung von $g$ an der Stelle $t$ verändert
    - Großer absoluter Wert: $g(t)$ ist in der Nähe von $t$ sehr "wellig"
    - Nahe Null: $g(t)$ ist an dieser Stelle glatt (zum Beispiel eine Gerade, deren zweite Ableitung null ist)
    - $\int (g''(t))^2 dt$ misst die gesamte Änderung von $g'(t)$ (der ersten Ableitung bzw. Steigung) über den gesamten Bereich
- $g(x)$ ist nicht identisch mit dem natürlichen kubischen Spline aus dem Basisfunktionsansatz mit Knoten bei $x_1, \ldots, x_n$.
  - $g(x)$ ist eine geschrumpfte Version eines solchen natürlichen kubischen Splines.
  - Der Grad der Schrumpfung wird durch $\lambda$ bestimmt.
- Nominale Freiheitsgrade zählen die Anzahl der freien Parameter (z.B. die Anzahl der Koeffizienten in einem Polynom oder Spline)
  - Für Glättungssplines: $n$ Parameter $\Rightarrow n$ nominale Freiheitsgrade
  - Diese $n$ Parameter werden durch $\lambda$ stark eingeschränkt (geschrumpft). Daher ist die tatsächliche Flexibilität kleiner als $n$.
- $df_\lambda$ sind die effektiven Freiheitsgrade und messen die Flexibilität des Glättungssplines
  - Höhere $df_\lambda$: flexibler (niedrigere Verzerrung, höhere Varianz)
  - Niedrigere $df_\lambda$: glatter (höhere Verzerrung, niedrigere Varianz)
  - $\hat{g}_\lambda = S_\lambda y$
    - $\hat{g}_\lambda$: Der Vektor der geschätzten Werte des Glättungssplines für ein gegebenes $\lambda$
    - $S_\lambda$: Eine $n\times n$-Matrix ("Smoother-Matrix"), die auf den Antwortvektor $y$ angewendet wird
    - $df_\lambda$ ist die Summe der Diagonaleinträge von $S_\lambda$
- Leave-One-Out-Cross-Validation kann für Glättungssplines effizient berechnet werden
  - $RSS_{cv}(\lambda) = \sum_{i=1}^{n} (y_i - \hat{g}_{\lambda}^{(-i)}(x_i))^2 = \sum_{i=1}^{n} \left[ \frac{y_i - \hat{g}_{\lambda}(x_i)}{1 - \{S_{\lambda}\}_{ii}} \right]^2$

```python=
from pygam import (s as s_gam, l as l_gam, f as f_gam, LinearGAM)
smooth_spline = LinearGAM(s_gam(0, lam=0.6))
ageCol = np.asarray(WageDF.age).reshape((-1,1))
smooth_spline.fit(ageCol, WageDF.wage)
# s_gam() take only one integer, not list. 
# If multiple columns are needed, use sperate s_gam() for each column.
```

**GAM**

```python=
XSmS_GamARR = np.column_stack( [WageDF.age, WageDF.year, WageDF.education.cat.codes] )
GAM = LinearGAM(s_gam(0) + l_gam(1, lam=0) + f_gam(2, lam=0))
GAM.fit(XSmS_GamARR, WageDF.wage)
```

## Lokales Regressionsmodell

- Lokale Regression ist ein alternativer Ansatz zur Anpassung flexibler, nichtlinearer Funktionen.
- Die Anpassung an einem Zielpunkt $x_0$ wird nur mit nahegelegenen Trainingsdaten berechnet.
- Algorithmus: Lokale Regression bei $X=x_0$
  1. Auswahl der Nachbarschaft
      - Bestimme den Anteil $s=k/n$ der Trainingspunkte, deren $x_i$ am nächsten zu $x_0$ liegen.
  2. Gewichtszuteilung
      - Weisen Sie jedem Punkt in dieser Nachbarschaft ein Gewicht $K_{i0}=K(x_i,x_0)$ zu,
          - Der am weitesten entfernte Punkt von $x_0$: Gewicht null
          - Der nächstgelegene Punkt: höchstes Gewicht
          - Alle anderen Punkte (nicht unter den $k$-nächsten Nachbarn): Gewicht null
  3. Gewichtete Regression
      - Schätze eine gewichtete kleinste-Quadrate-Regression von $y_i$ auf $x_i$ unter Verwendung dieser Gewichte.
      - Finde $\hat{\beta}_0$, $\hat{\beta}_1$, die $\sum_{i=1}^n K_{i0}(y_i-\beta_0-\beta_1 x_i)^2$ minimieren.
  4. Vorhersage
      - Der geschätzte Wert bei $x_0$ ist $\hat{f}(x_0)=\hat{\beta}_0+\hat{\beta}_1 x_0$
- Die Gewichte $K_{i0}$ sind für jedes $x_0$ einzigartig.
      - Für jeden Zielpunkt muss eine neue gewichtete Regression berechnet werden.
- Speicherbasierte Methode
    - Ähnlich wie das k-nächste-Nachbarn-Verfahren
    - Benötigt die gesamten Trainingsdaten für jede Vorhersage
- Der Span $s$ ist der Anteil der Punkte, die für die lokale Anpassung bei $x_0$ verwendet werden.
    - Kleineres $s$: stärker lokalisiert und "wackeliger" Fit
    - Größeres $s$: globalerer Fit unter Verwendung mehr Trainingsdaten
- Verallgemeinerungen der lokalen Regression
    - Global in manchen Variablen
    - Lokal in anderen
    - Lokale Regression in zwei Variablen
        - Schätze bivariate Regressionsmodelle nahe jedem Zielpunkt im 2D-Raum
        - Erweiterbar auf höhere Dimensionen
        - Die Leistung verschlechtert sich, wenn $p>3$ oder $4$, da dann zu wenige Trainingsdaten in der Nähe sind

```python=
import statsmodels.api as sm
lowess = sm.nonparametric.lowess
lowess(WageDF.wage, WageDF.age, frac=0.2)
```

## Entscheidungsbäume

**Regressionsbäume**

- Aufteilung des Prädiktorraums
  - Grundprinzip
    - Prädiktorraum ($X_1, X_2, \ldots, X_p$) wird in $J$ Regionen ($R_1, R_2, \ldots, R_J$) unterteilt.
    - In jeder Region $R_j$ erfolgt die gleiche Vorhersage: Mittelwert der Antwortwerte der Trainingsdaten in $R_j$.

  - Beispiel
    - Zwei Regionen $R_1$ und $R_2$:
      - Mittelwert in $R_1$: 10
      - Mittelwert in $R_2$: 20
    - Vorhersage für $Y$:
      - $Y \in R_1$: Vorhersage 10
      - $Y \in R_2$: Vorhersage 20

  - Konstruktion der Regionen
    - Regionen als hochdimensionale Rechtecke (Boxen) definiert
    - Minimierung der RSS: $\sum_{j=1}^{J} \sum_{i \in R_j} (y_i - \hat{y}_{R_j})^2$

  - Ansatz: Rekursive binäre Teilung (top-down, gierig)
    - Wahl $X_j$ und Schnittpunkt $s$ zur Aufteilung in $\{ X| X_j < s \}$ und $\{ X| X_j \geq s \}$
    - Suche nach $(j, s)$, die den RSS minimieren: $\sum_{i: x_i \in R_1(j,s)} (y_i - \hat{y}_{R_1})^2 + \sum_{i: x_i \in R_2(j,s)} (y_i - \hat{y}_{R_2})^2$
      
  - Rekursive Anwendung und Stoppkriterium
    - Wähle besten Prädiktor und Schnittpunkt jeweils erneut für entstehende Regionen
    - Wiederhole den Prozess, bis ein Abbruchkriterium erreicht ist, z.B. maximal 5 Beobachtungen pro Region

- Überanpassung (Overfitting)
  - Große Bäume passen sich zu gut an Trainingsdaten an, aber sie generalisieren schlecht auf Testdaten
  - Lösung: Kleinere Bäume
    - Kleinere Bäume reduzieren die Varianz und sind besser interpretierbar
    - Nachteil: Risiko, wichtige Strukturen zu übersehen

- Pruning (Beschneiden) von Bäumen
  - Zuerst einen großen Baum $T_0$ wachsen lassen
  - Baum dann beschneiden (Pruning), um einen optimalen Subbaum zu erhalten

  - Kostenkomplexitäts-Pruning (Weakest Link Pruning)
    - Statt alle Teilbäume einzeln zu prüfen, betrachtet man eine durch den nicht-negativen Parameter $\alpha$ indizierte Folge von Teilbäumen
    - Für jeden Wert von $\alpha$ wird ein Teilbaum $T \subset T_0$ ausgewählt, der folgendes minimiert: $\sum_{m=1}^{|T|} \sum_{i: x_i \in R_m } (y_i - \hat{y}_{R_m})^2 + \alpha |T|$ minimiert
      - $|T|$: Anzahl der terminalen Knoten (Blätter)
    - $\alpha$ steuert Gleichgewicht zwischen Baumkomplexität und Anpassung
      - $\alpha = 0$: keine Bestrafung, $T = T_0$
      - Mit steigendem $\alpha$ werden kleinere, einfachere Subbäume bevorzugt

  - Ziel: Testfehler minimieren
    - Subbaum mit geringstem Testfehler auswählen
    - Testfehler wird mittels Kreuzvalidierung oder Validierungsdaten geschätzt

- Algorithmus: Vorgehen beim Regressionsbaum
  1. Großen Baum wachsen lassen
      - Rekursive binäre Teilung durchführen
      - Stoppen, wenn ein Knoten weniger als die Mindestanzahl Beobachtungen enthält
  2. Kostenkomplexitäts-Pruning anwenden
      - Sequenz der besten Subbäume für verschiedene $\alpha$ berechnen
  3. Kreuzvalidierung zur Auswahl von $\alpha$
      - Daten in $K$ Folds teilen
      - Für jedes $\alpha$: Baum auf $K-1$ Folds bauen, Fehler auf dem Auslass-Fold messen
      - Das $\alpha$ wählen, das den mittleren Vorhersagefehler minimiert

  4. Der Teilbaum wird ausgewählt (aus der Folge von Teilbäumen), der zum optimalen $\alpha$ gehört.

  - Erläuterung Schritt 2
      - Nach dem vollständigen Baum wird das Fehlermaß MSE berechnet
      - Ein Knoten wird geschnitten, das MSE des beschnittenen Baums und die Anzahl der entfernten Blätter werden berechnet
      - Das zugehörige $\alpha$ für diesen Schnitt ist die Differenz der Fehlermaße geteilt durch die Zahl der entfernten Blätter

```python=
from sklearn.tree import (DecisionTreeRegressor as DTR, plot_tree, export_text)
import sklearn.model_selection as skm
Regression_TRE = DTR()
# max_depth
DTR(XARR_train, y_train)
ccp_path = Regression_TRE.cost_complexity_pruning_path(XARR_train, y_train)
kfold = skm.KFold(5, shuffle=True, random_state=10)
grid = skm.GridSearchCV(Regression_TRE, {'ccp_alpha': ccp_path.ccp_alphas}, refit=True, cv=kfold, scoring='neg_mean_squared_error').fit(XARR_train, y_train)
best_ = grid.best_estimator_
ax = plt.subplots(figsize=(12,12))[1]
plot_tree(best_, feature_names=feature_names, ax=ax)
print(export_text(best_, feature_names=feature_names))
```

**Klassifikationsbäume**

- Klassifikationsbaum vs. Regressionsbaum:
  
  - Regressionsbaum: Vorhersage durch Mittelwert der Trainingsbeobachtungen.
  - Klassifikationsbaum: Vorhersage durch häufigste Klasse der Trainingsbeobachtungen.

- Klassifikationsbaum erstellen:
  
  - Nutzen rekursives binäres Teilen.
- Klassifikationsfehlerrate als Kriterium:
  - Fehlerquote: $E = 1 - \max_k {\hat{p}}_{mk}$.
    - $k$ ist eine Klasse, $m$ ist ein Bereich im Klassifikationsbaum.
    - ${\hat{p}}_{mk}$ ist der Anteil der Trainingsbeobachtungen im Bereich $m$ aus der Klasse $k$.
    - Beispiele:
      - $k = 1$ könnte "Herzkrankheit" sein, $k = 2$ "Keine Herzkrankheit".
      - $m = 3$ bezieht sich auf den dritten Endknoten.
      - ${\hat{p}}_{3,1}$ ist der Anteil der Beobachtungen im dritten Bereich, die als "Herzkrankheit" klassifiziert sind.

- Bessere Kriterien als Klassifikationsfehlerrate:
  
  - Gini-Index: $G = \sum_{k=1}^{K} {\hat{p}}_{mk} (1 - {\hat{p}}_{mk})$
    - Klein, wenn Knoten rein ist.
  - Entropie: $D = -\sum_{k=1}^{K} {\hat{p}}_{mk} \log {\hat{p}}_{mk}$
    - Klein bei reinen Knoten.

- Node Purity:
  
  - Split erhöht Knotenreinheit.
  - Wichtig für genaue Vorhersagen.
  - Beispiel: Split "RestECG<1" am Baumende erhöht Reinheit, obwohl Vorhersage gleich bleibt.

```python=
from sklearn.tree import (DecisionTreeClassifier as DTC, plot_tree, export_text)
from sklearn.metrics import accuracy_score
import sklearn.model_selection as skm
Clas_TRE = DTC(criterion='entropy')
Clas_TRE.fit(XARR_train, label_train)
ccp_path = TRE_clas.cost_complexity_pruning_path(XARR_train, label_train)
kfold = skm.KFold(10, random_state=1, shuffle=True)
grid = skm.GridSearchCV(Clas_TRE, {'ccp_alpha': ccp_path.ccp_alphas}, refit=True, cv=kfold, scoring='accuracy').fit(XARR_train, label_train)
best_ = grid.best_estimator_
grid.best_score_ # Training sample
accuracy_score(label_test, best_.predict(X_test)) # Test sample
grid.best_estimator_.tree_.n_leaves
ax = plt.subplots(figsize=(12,12))[1]
plot_tree(best_, feature_names=feature_names, ax=ax)
print(export_text(best_, feature_names=feature_names, show_weights=True))
```

**Bagging (Bootstrap Aggregation)**

- Bootstrap: Grundlagen und Nutzen

  - Einführung
    - Bootstrap ist eine mächtige Methode in der Statistik.
    - Wird verwendet, um Standardabweichungen zu berechnen, wenn dies auf direktem Wege schwierig ist
    - Kann zur Verbesserung statistischer Lernmethoden wie Entscheidungsbäume verwendet werden

  - Entscheidungsbäume und Varianz
    - Entscheidungsbäume sind durch hohe Varianz gekennzeichnet
      - Unterschiedliche Trainingsdaten führen zu stark abweichenden Ergebnissen
    - Methoden mit geringer Varianz (z. B. lineare Regression) liefern stabilere Ergebnisse bei neuen Datensätzen
    - Bagging reduziert die Varianz eines Modells
      - Viele Trainingsdatensätze erzeugen, Modelle darauf trainieren, Vorhersagen mitteln
      - In der Praxis durch Bootstrapping umgesetzt

- Bagging: Vorgehensweise und Anwendung

  - Allgemeine Vorgehensweise
    - Erstellen von B bootstrapped Trainingssätzen durch Ziehen mit Zurücklegen aus dem ursprünglichen Datensatz
    - Trainieren eines Modells (z. B. Entscheidungsbaum) auf jedem bootstrapped Datensatz
    - Mitteln (bei Regression) oder Mehrheitswahl (bei Klassifikation) der Vorhersagen über alle Modelle

- Out-of-Bag (OOB) Fehlerabschätzung

  - Prinzip
    - Ermöglicht eine Schätzung des Testfehlers ohne separate Kreuzvalidierung
    - Jeder Baum nutzt im Mittel etwa zwei Drittel der Daten zum Training
    - Das verbleibende Drittel sind Out-of-Bag (OOB) Beobachtungen

  - Vorgehen
    - Für jede OOB-Beobachtung erfolgt die Vorhersage nur durch die Modelle, die diese Beobachtung nicht zum Training genutzt haben
    - OOB-Vorhersagen werden gemittelt (Regression) oder per Mehrheitswahl (Klassifikation) aggregiert
    - Der OOB-Fehler entspricht typischerweise dem Fehler einer Leave-One-Out Kreuzvalidierung

- Variable Importance Measures (Variablenwichtigkeit)

  - Auswirkungen des Bagging auf Interpretierbarkeit
    - Bagging erhöht die Vorhersagegenauigkeit durch Kombination vieler Modelle
    - Die Interpretierbarkeit ist im Vergleich zu einem einzelnen Baum reduziert

  - Messung der Variablenwichtigkeit
    - Bei Regressionsbäumen: Bedeutung eines Prädiktors durch durchschnittliche Reduktion des RSS über alle Bäume
    - Bei Klassifikationsbäumen: Bedeutung eines Prädiktors durch die durchschnittliche Reduktion des Gini-Index

  - Darstellung
    - Grafische Darstellungen zeigen die wichtigsten Prädiktoren und ihre relative Bedeutung

- Überanpassung
  - Keine Überanpassungsgefahr bei Erhöhung der Baumanzahl in Bagging
  - Zu wenige Bäume können jedoch zu Unteranpassung führen

```python=
from sklearn.ensemble import RandomForestRegressor as RF
BAGG = RF(max_features=len(X_train.columns), n_estimators=500, random_state=0).fit(X_train, y_train)
# n_estimators is B
np.mean( (BAGG.predict(X_test) - y_test)**2)
BAGG.feature_importances_
```

**Random Forest**

- Dekorrelierung der Bäume
  - Bei jedem Split wird eine zufällige Auswahl von m Prädiktoren betrachtet (statt alle)
  - Dadurch werden die Entscheidungsbäume stärker voneinander unterschieden ("dekorreliert")
  - Typischerweise wird $m \approx \sqrt{p}$ gewählt
  - Starke Prädiktoren sollen nicht jeden Split dominieren
- Vorgehen/Verfahren (Ablauf des Random Forest)
  - Entscheidungsbäume werden auf bootstrapped Trainingsproben trainiert
  - An jedem Knoten wird aus den m zufällig ausgewählten Prädiktoren der beste Split gewählt
- Vorteile
  - Geringere Korrelation zwischen den Bäumen
  - Reduzierte Varianz und robustere Vorhersagen
- Überanpassung
  - Random Forests überpassen nicht, wenn die Anzahl der Bäume B erhöht wird
  - B sollte so groß gewählt werden, dass sich der Fehler stabilisiert

**Boosting**

- Allgemeiner Ansatz für Regression und Klassifikation. Hier auf Entscheidungsbäume beschränkt
- Unterschied zu Bagging:
  - Bäume werden sequentiell anstatt unabhängig aufgebaut
  - Keine Bootstrap-Stichproben, sondern modifizierte Datensätze
- Algorithmus für Regression:
  1. Setze ${\hat{f}}{(x)} = 0$ und Residuen $r_i = y_i$ für alle i im Trainingssatz
  2. Für b = 1, 2, ..., B:
     - Passe einen Baum ${\hat{f^b}}$ mit $d$ Schnitt ($d+1$ Blätter) an die Daten $(X, r)$ an
     - Aktualisiere Modell: ${\hat{f}}(x) \leftarrow {\hat{f}}(x) + λ {\hat{f^{b}}}{(x)}$
     - Aktualisiere Residuen: $r_i \leftarrow r_i - λ {\hat{f^{b}}}{(x_i)}$
  3. Ausgabe des Modells: ${\hat{f}}{(x)} = \sum_{b=1}^B {λ {\hat{f^{b}}}{(x)}}$
  - Erklärung Schritt 2
    - Einen Regressionsbaum ${\hat{f^b}}$ mit maximal d Teilungen auf die Residuen r anpassen (begrenzte Tiefe verhindert Überanpassung)
    - Gesamtvorhersage ${\hat{f}}$ mit $\lambda$-facher Vorhersage des neuen Baums aktualisieren (λ steuert den Einfluss).
    - Residuen aktualisieren, indem die Beiträge des neuen Baums (mit λ gewichtet) abgezogen werden – folgende Bäume konzentrieren sich so auf verbleibende Fehler
- Idee:
  - Jeder Baum wird auf die Residuen (Fehler) des aktuellen Modells angepasst, nicht auf die Originalwerte
  - Die Bäume sind meist klein (wenige Endknoten, gesteuert durch $d$)
  - Der Schrumpfungsparameter $\lambda$ reguliert das Lerntempo und verhindert Überanpassung
  - Das Modell wird gezielt dort verbessert, wo es noch Fehler macht
  - Der Aufbau jedes Baums hängt von den zuvor gebauten Bäumen ab
- Abstimmungsparameter:
  1. Anzahl der Bäume $B$:
     - Große Anzahl $B$ kann zu überanpassen führen
     - Kreuzvalidierung zur Auswahl von $B$
  2. Shrinkage-Parameter $\lambda$:
     - Kontrolliert die Lernrate
     - Typische Werte: 0.01 oder 0.001
  3. Anzahl $d$ der Splits pro Baum:
     - Kontrolliert die Komplexität
     - Oft funktioniert d = 1 gut

```python=
from sklearn.ensemble import GradientBoostingRegressor as GBR
# GradientBoostingClassifier as GBC
Boost = GBR(n_estimators=5000, learning_rate=0.001, max_depth=3, random_state=0).fit(X_train, y_train)
np.mean( (y_test - Boost.predict(X_test))**2)
```

**BART (Bayesian Additive Regression Trees)**

- Überblick
    - BART ist ein Ensemble-Verfahren mit Entscheidungsbäumen als Basis, meist für Regression.
    - Verwandt mit Bagging/Random Forests (zufällige Bäume) und Boosting (Anpassung an Residuen).
        - Kombiniert beide Ansätze: zufällige Bäume, Anpassung an Residuen, Bäume werden leicht verändert statt neu gebaut.
- Notation
    - $K$: Anzahl der Bäume
    - $B$: Anzahl der Iterationen
    - ${\hat{f}}^b_k(x)$: Vorhersage des $k$-ten Baums in Iteration $b$
    - ${\hat{f}}^b(x) = \sum_{k=1}^K {\hat{f}}^b_k(x)$: Modellvorhersage in Iteration $b$
- Algorithmus
    1. Initialisierung
        - Jeder Baum sagt Mittelwert der Antwort geteilt durch $K$ vorher: ${\hat{f}}^1_k(x) = \frac{1}{nK}\sum_{i=1}^n y_i$
        - Anfangsmodell ist der Mittelwert der Antworten ${\hat{f}}^1(x) = \sum_{k=1}^K {{\hat{f}}^1_k(x)} = \frac{1}{n} \sum_{i=1}^n y_i$
    2. Iterative Updates ($b = 2, ..., B$)
        - Für jeden Baum $k$:
            - Berechne partielle Residuen: ziehe aktuelle Vorhersagen der anderen Bäume ab, außer vom Baum $k$: $r_i = y_i - \sum_{k^\prime < k}{{\hat{f}}_{k'}^b(x_i)} - \sum_{k^\prime > k}{{\hat{f}}_{k'}^{b-1}(x_i)}$
            	- $\sum_{k^\prime < k}{{\hat{f}}_{k'}^b(x_i)}$: Summe der Vorhersagen der Bäume in der *aktuellen Iteration ($b$)*, die bereits aktualisiert wurden ($k'$ kleiner als $k$)
            	- $\sum_{k^\prime > k}{{\hat{f}}_{k'}^{b-1}(x_i)}$: Summe der Vorhersagen der Bäume aus der *vorherigen Iteration ($b-1$)*, die in der aktuellen Iteration noch nicht aktualisiert wurden ($k'$ größer als $k$)
            - Aktualisiere Baum $k$ zufällig (wachsen, stutzen, Endknoten anpassen); Ziel ist bessere Anpassung
        - Modell ist die Summe der $K$ aktualisierten Bäume: ${\hat{f}}^b(x) = \sum_{k=1}^K {\hat{f}}^b_k(x)$
    - Finale Vorhersage: Mittelwert der Vorhersagen aus den Iterationen $L+1$ bis $B$: $\hat{f}(x) = \frac{1}{B-L} \sum_{b=L+1}^B {\hat{f}}^b(x)$
    - Baumveränderungen:
        - Grow: Teilung hinzufügen
        - Prune: Teilung entfernen
        - Endknoten anpassen
        - Änderungen sind zufällig, bevorzugen Fehlerreduktion, balancieren Passung und Regularisierung, sind als Posterior-Sampling interpretierbar
- Performance und Überanpassung
    - Nach dem Burn-in stabilisieren sich Fehlerraten; wenige Überanpassung
    - Bessere Kontrolle der Modellkomplexität als Boosting
- Bayes-Perspektive
    - Updates sind als Ziehen aus der Posterior-Verteilung interpretierbar (MCMC für Baum-Ensembles)
- BART-Tuning
    - Zu wählen sind: Anzahl der Bäume $K$ (200), Iterationen $B$ (1000), Burn-in $L$(100)
    - Meist ist wenig Feinjustierung nötig; BART funktioniert oft direkt gut

![Tree Perturbations](https://i.postimg.cc/D0Bk2dsP/BART.jpg)


```python=
from ISLP.bart import BART
Bart = BART(random_state=0, burnin=5, ndraw=15).fit(X_train, y_train)
np.mean((Bart.predict(X_test.astype(np.float32)) - y_test) ** 2)
Bart.variable_inclusion_.mean(0)
```

## SVM

**Hyperplane**
- Definition: Eine Hyperplane ist eine flache affine Untermannigfaltigkeit der Dimension p-1
  - In 2D: Linie
  - In 3D: Ebene
  - In p>3: Schwer vorstellbar, aber gleiche Definition
- Mathematische Definition:
  - 2D: $\beta_0 + \beta_1 X_1 + \beta_2 X_2 = 0$
  - p-Dimensionen: $\beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots + \beta_p X_p = 0$
- Positionierung:
  - $\beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots + \beta_p X_p > 0$: Punkt liegt auf einer Seite
  - $\beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots + \beta_p X_p < 0$: Punkt liegt auf der anderen Seite
- Klassen: $y_1, \ldots, y_n \in \{-1, 1\}$
- Klassifikation mit einer trennenden Hyperplane:
  - $\beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \ldots + \beta_p x_{ip} > 0 \text{ für } y_{i} = 1$
  - $\beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \ldots + \beta_p x_{ip} < 0 \text{ für } y_{i} = -1$
- Problem: Unendlich viele trennende Hyperplanes möglich
- Lösung: Maximal Margin Hyperplane wählen
  - Größte minimale Distanz zu den Trainingsbeobachtungen
  - Abhängigkeit von den Support Vektoren
    - Beobachtungen, die die Margin beeinflussen
    - Nur Support Vektoren beeinflussen den Classifier

**Maximal Margin Classifier**
- Optimierungsproblem:
  - Maximieren von $M$
  - Einschränkungen: $\sum_{j=1}^{p}\beta_j^2=1$, $y_i(\beta_0+\beta_1x_{i1}+\beta_2x_{i2}+\ldots+\beta_px_{ip})\geq M$
    - $y_i(\beta_0+\beta_1x_{i1}+\beta_2x_{i2}+\ldots+\beta_px_{ip})$: den Abstand der *i*-ten Beobachtung zur Hyperbene.
    - $\beta_0, \beta_1, ..., \beta_p$: Koeffizienten der maximalen Marginal-Hyperbene
    - $\beta_0+\beta_1x_{i1}+\beta_2x_{i2}+\ldots+\beta_px_{ip} = 0$: Definiert die Hyperbene
    - $\sum_{j=1}^{p}\beta_j^2=1$: Stellt sicher, dass dieser Ausdruck den senkrechten Abstand der *i*-ten Beobachtung zur Hyperbene repräsentiert
- Wenn es keine trennende Hyperplane existiert, verwenden Soft Margin (Support Vector Classifier), um Klassen fast zu trennen

**Support Vector Classifiers**
- Überblick: Support Vector Classifier (auch Soft Margin Classifier genannt) erlaubt einige Fehler, um die meisten Beobachtungen korrekt zu klassifizieren
- Optimierungsproblem:
  - Maximierung der Margin $M$
  - Einbeziehung von Slack-Variablen $\epsilon_i$ erlaubt Fehler
  - Die Einschränkung: $y_i(\beta_0+\beta_1x_{i1}+\beta_2x_{i2}+\ldots+\beta_px_{ip})\geq M(1-\epsilon_i)$
    - $\epsilon_i = 0$:
      - Beobachtung ist korrekt klassifiziert
    - $\epsilon_i > 0$:
      - Beobachtung hat den Margin verletzt
      - Kann dennoch auf der richtigen Seite der Hyperbene liegen
    - $\epsilon_i > 1$:
      - Beobachtung liegt auf der falschen Seite der Hyperbene (fehlklassifiziert)
- Parameter $C$: kontrolliert die zulässige Summe der $\epsilon_i$'s, fungiert als Budget für Margin-Verletzungen
  - $C = 0$: Keine Toleranz für Fehler (maximal margin hyperplane)
  - $C > 0$: Erlaubt Fehler, größere Margin
  - Größeres $C$:
      - Mehr Toleranz, breitere Margin
      - Mehr Support Vektoren, niedrige Varianz, hoher Bias
  - Kleineres $C$:
      - Weniger Toleranz, schmalere Margin
      - Weniger Support Vektoren, hohe Varianz, niedriger Bias
- Ein Support Vector Classifier eignet sich gut für lineare Klassifikationsgrenzen
    - Bei nicht-linearen Grenzen versagt der SVC. Lösung: Erweiterung des Merkmalsraums durch quadratische, kubische oder höhere Polynome
        - Beispiel: Statt 𝑝 Merkmale $X_1, X_2, \ldots, X_p$ nutzen wir 2𝑝 Merkmale $X_1, X_1^2, X_2, X_2^2, \ldots , X_p, X_p^2$
    - Resultat: In erweitertem Merkmalsraum ist die Grenze linear, im ursprünglichen Raum jedoch nicht-linear
    - Zu viele Merkmale können Berechnungen unhandlich machen

```python=
from sklearn.svm import SVC
import sklearn.model_selection as skm
SVC_Model = SVC(C=1e5, kernel='linear').fit(X, y)
# Höher C-Wert: Modell bestraft Fehlklassifikationen stark (kleiner Margin)
# Niedriger C-Wert: Modell erlaubt mehr Fehlklassifikationen (größer Margin)
# linear Kernel: eine lineare Hyperebene
# rbf Kernel: eine Radial-Basis-Funktion
kfold = skm.KFold(5, random_state=0, shuffle=True)
Grid_SVC = skm.GridSearchCV(SVC_Model,{'C':[5,10,100]}, refit=True, cv=kfold, scoring='accuracy')
Grid_SVC.fit(X,y)
Grid_SVC.best_params_
```

**Support Vector Machine**
- SVM erweitert SVC durch Verwendung von Kernels
- Kernels definieren die Ähnlichkeit zweier Beobachtungen
    - Lineares Kernel: $K(x_i,x_{i\prime}) = \sum_{j=1}^{p}{x_{ij}x_{i\prime j}}$
    - Polynomiales Kernel: $K(x_i,x_{i\prime}) = (1+\sum_{j=1}^{p}{x_{ij}x_{i\prime j}})^d$ -> passt besser zu nicht-linearen Daten
    - Radial Kernel: $K(x_i,x_{i'})=exp(-\gamma\sum_{j=1}^{p}(x_{ij}-x_{i^\prime j})^2)$
- Kernels ermöglichen effiziente Berechnungen ohne expliziten Merkmalsraum 
- Die Kernelmatrix enthält die paarweisen Kernel-Ähnlichkeiten aller Trainingsdatenpunkte
    - Beispiel: n = 3, p = 2
        - $K = \begin{bmatrix} K(x_1, x_1) & K(x_1, x_2) & K(x_1, x_3) \\ K(x_2, x_1) & K(x_2, x_2) & K(x_2, x_3) \\ K(x_3, x_1) & K(x_3, x_2) & K(x_3, x_3) \end{bmatrix}$
- Zur Klassifikation neuer Daten berechnet die SVM die Kernel-Ähnlichkeit zu den Support-Vektoren und weist anhand dieser Werte eine Klasse zu

```python=
from sklearn.svm import SVC
import sklearn.model_selection as skm
SVM_Model = SVC(kernel="rbf", gamma=1, C=1).fit(X, y)
# kernel="poly": degree gibt den Grad des Polynoms an
# kernel="rbf": gamma gibt den Koeffizienten des Kernels an
# Ein zu hoher C-Wert kann zu Overfitting führen
kfold = skm.KFold(5, random_state=0, shuffle=True)
Grid_SVM = skm.GridSearchCV(svm_rbf, refit=True, cv=kfold, scoring='accuracy', {'C':[0.1,10,100], 'gamma':[0.5,2,3]})
Grid_SVM.fit(X, y)
Grid_SVM.best_params_
```

**ROC Kurve**

- Klassifikatoren wie LDA und SVM berechnen für jede Beobachtung Scores
    - Form von LDA oder SVC: $\hat{f}(X) = \hat{\beta}_0 + \hat{\beta}_1 X_1 + \hat{\beta}_2 X_2 + \ldots + \hat{\beta}_p X_p$
- Schwellenwert $t$ teilt Beobachtungen anhand des Scores $\hat{f}(X)$ in zwei Kategorien:
  - $\hat{f}(X) < t$: Kategorie 1 (z.B. "Herzkrankheit")
  - $\hat{f}(X) \geq t$: Kategorie 2 (z.B. "keine Herzkrankheit")
- ROC-Kurve: Zeigt die wahre Positivrate (y-Achse) gegen die falsche Positivrate (x-Achse) für verschiedene Schwellenwerte *t*
  - Falsche Positivrate: Anteil fälschlich positiver Klassifikationen
  - Wahre Positivrate: Anteil korrekt positiver Klassifikationen
    <table>
        <tr>
            <th></th>
            <th>Actual Positive</th>
            <th>Actual Negative</th>
        </tr>
        <tr>
            <th>Predicted Positive</th>
            <td>Wahre Positive (TP)</td>
            <td>Falsche Positive (FP)</td>
        </tr>
        <tr>
            <th>Predicted Negative</th>
            <td>Falsche Negative (FN)</td>
            <td>Wahre Negative (TN)</td>
        </tr>
        <tr>
            <th></th>
            <td>P</td>
            <td>N</td>
        </tr>    
    </table>
    
    - Sensitivität (True Positive Rate, TPR) = TP/(TP + FN) = TP/P
    - Falsche Positive Rate (FPR) oder 1 - Specificity = FP/(FP + TN) = FP/N
    - TP/P : Sensitivity, Power, Recall, 1 - Type II Error
    - FP/N : Type I Error, 1 - Specificity
    - TP/(TP+FP) : Precision, 1 - False Discovery Proportion
- Optimaler Klassifikator liegt im oberen linken Eckpunkt der ROC-Kurve (hohe wahre, niedrige falsche Positivrate)
- Höhere Kurve = besserer Klassifikator

```python=
from sklearn.metrics import RocCurveDisplay
roc_curve = RocCurveDisplay.from_estimator
fig, ax = plt.subplots(figsize=(8,8))
roc_curve(SVM_Model, X, y, name='Training', color='r', ax=ax)
```

**SVM with Multiple Classes**

- Ansätze: One-Versus-One (OvO) und One-Versus-All (OvA) or One-Versus-Rest (OvR)
- OvO:
    - Konstruktion von $\binom{K}{2}$ SVMs, jede vergleicht ein Klassenpaar
    - Zuweisung zur Klasse, die am häufigsten gewählt wurde
- OvA:
    - Konstruktion von $K$ SVMs, jede vergleicht eine Klasse mit den anderen $K-1$ Klassen
    - Zuweisung zur Klasse mit größtem Wert $\beta_{0k} + \beta_{1k} x_1^\ast + \ldots + \beta_{pk} x_p^\ast$

```python=
from sklearn.svm import SVC
OvO = SVC(kernel="rbf", C=10, gamma=1, decision_function_shape='ovo')
OvR = SVC(kernel="rbf", C=10, gamma=1, decision_function_shape='ovr')
```

## Clustering

**K-Means Clustering**

- K-Means-Clustering ist eine einfache und elegante Methode, um einen Datensatz in vorgegebene K verschiedene, nicht überlappende Cluster zu unterteilen.
    - Die Auswahl von K ist ein nicht-triviales Problem.
    - Jede Beobachtung gehört zu mindestens einem Cluster.
- Die Variation innerhalb eines Clusters $W(C_k)$ misst, wie sehr sich die Beobachtungen innerhalb eines Clusters voneinander unterscheiden.
    - $W(C_k) = \frac{1}{|C_k|} \sum_{i, i' \in C_k} \sum_{j=1}^{p} (x_{ij} - x_{i'j})^2$
        - $|C_k|$ = Anzahl der Beobachtungen im $k$-ten Cluster
- Eine gute Clusterbildung minimiert die Variation innerhalb der Cluster $\text{minimize}_{C_1, \ldots, C_K} \left\{ \sum_{k=1}^{K} W(C_k) \right\}$
- Algorithmus:
  - Problem: Die Anzahl möglicher Partitionen ist nahezu $K^n$, was eine exakte Lösung außer bei kleinen K und n schwierig macht.
  - Ein einfacher Algorithmus findet ein lokales Optimum (eine ziemlich gute Lösung):
    - Schritt 1: Weise jeder Beobachtung zufällig eine Zahl von 1 bis K zu (initiale Clusterzuweisung).
    - Schritt 2: Iteriere, bis sich die Clusterzuweisungen nicht mehr ändern:
      - (a) Für jedes der K Cluster berechne das Cluster-Zentrum:
        - Das Zentrum des $k$-ten Clusters ist der Vektor der Mittelwerte der p Merkmale der Beobachtungen in Cluster $k$.
      - (b) Weise jede Beobachtung dem Cluster zu, dessen Zentrum am nächsten liegt (unter Verwendung der euklidischen Distanz).
- Der K-Means-Algorithmus findet ein lokales, aber kein globales Optimum.
- Die Ergebnisse hängen von der initialen (zufälligen) Clusterzuweisung in Schritt 1 ab.
    - Führe den Algorithmus mehrmals mit unterschiedlichen zufälligen Anfangskonfigurationen aus.
    - Wähle die beste Lösung (kleinster Wert der Variation innerhalb der Cluster).

```python=
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=2, n_init=20).fit(X)
kmeans.labels_
```

**Hierarchical Clustering**

- Die häufigste Art des hierarchischen Clusterings ist das Bottom-up- oder agglomerative Clustering.
- Dendrogramm-Struktur:
    - Jedes Blatt stellt eine der Beobachtungen dar.
    - Beim Aufsteigen im Baum:
        - Blätter beginnen, sich zu Ästen zu verbinden, was auf ähnliche Beobachtungen hinweist.
        - Äste selbst verbinden sich auf höheren Ebenen mit anderen Blättern oder Ästen.
    - Frühere (niedrigere) Fusionen deuten auf eine höhere Ähnlichkeit zwischen Beobachtungen/Gruppen hin.
    - Spätere (höhere) Fusionen deuten auf größere Unähnlichkeit hin.
    - Die Höhe der ersten Fusion (auf der vertikalen Achse) zwischen zwei Beobachtungen misst deren Unähnlichkeit.
    - Die Ähnlichkeit wird durch die vertikale Fusionshöhe bestimmt, nicht durch die horizontale Nähe.
    - Um Cluster mit einem Dendrogramm zu identifizieren:
        - Ziehe einen waagerechten Schnitt durch das Dendrogramm.
        - Die Beobachtungsgruppen unterhalb des Schnitts sind die Cluster.
        - Ein Dendrogramm kann beliebig viele Cluster liefern.
        - Praktiker wählen die Anzahl der Cluster oft durch visuelle Inspektion des Dendrogramms.
- Für manche Datensätze spiegelt diese hierarchische Struktur möglicherweise nicht die tatsächliche Gruppierung wider.
    - Beispiel: Wenn Beobachtungen Männer und Frauen sind, aufgeteilt in Amerikaner, Japaner und Franzosen:
        - Beste Aufteilung in zwei: nach Geschlecht.
        - Beste Aufteilung in drei: nach Nationalität.
        - Die wahren Cluster (nach Nationalität) sind nicht innerhalb der geschlechtsbasierten Cluster verschachtelt.
    - Hierarchisches Clustering kann nicht immer nicht-verschachtelte Cluster darstellen und liefert möglicherweise weniger genaue Ergebnisse als K-Means für eine gegebene Clusteranzahl.
- Algorithmus:
  - Initialisierung
    - Beginne mit der Definition eines Unähnlichkeitsmaßes zwischen jedem Beobachtungspaar.
      - Am häufigsten verwendet: euklidische Distanz.
      - Die Wahl des Unähnlichkeitsmaßes kann variieren (später im Kapitel diskutiert).
  - Iterativer Clustering-Prozess
    - Beginne am unteren Rand des Dendrogramms:
      - Jede der $n$ Beobachtungen ist zu Beginn ihr eigenes Cluster.
    - In jedem Schritt:
      - Identifiziere die beiden ähnlichsten (am wenigsten unähnlichen) Cluster.
      - Verbinde diese beiden Cluster miteinander, wodurch die Clusteranzahl um eins reduziert wird.
      - Wiederhole dies, bis alle Beobachtungen zu einem einzigen Cluster gehören.
    - Endergebnis: Ein vollständiges Dendrogramm, das das hierarchische Clustering darstellt.
- Unähnlichkeit zwischen Clustern
    - Wie definiert man die Unähnlichkeit zwischen zwei Clustern, wenn diese mehrere Beobachtungen beinhalten können?
    - Lösung: Nutze das Konzept der Linkage, um die Unähnlichkeit zwischen Clustern zu definieren.
    - Complete Linkage: Berechne alle paarweisen Unähnlichkeiten zwischen Beobachtungen in den Clustern A und B; notiere den größten Wert. Führt zu balancierten Dendrogrammen.
    - Single Linkage: Notiere den kleinsten Wert. Kann zu verlängerten, "schlängelnden" Clustern mit Einzel-Fusionen führen.
    - Average Linkage: Notiere den Durchschnittswert. Wird von Statistikern bevorzugt und liefert balancierte Dendrogramme.
    - Centroid Linkage: Unähnlichkeit zwischen den Schwerpunkten (Mittelwert-Vektoren) der Cluster A und B. Häufig in der Genomik. Kann zu Inversionen führen (Cluster werden auf einer Höhe verschmolzen, die unter der einzelner Cluster liegt), was Visualisierungs- und Interpretationsprobleme verursachen kann.
    - Average, Complete und Single Linkage sind unter Statistikern am beliebtesten; Average und Complete werden im Allgemeinen gegenüber Single bevorzugt.
- Wahl des Unähnlichkeitsmaßes
    - Euklidische Distanz: Wird als Standard-Unähnlichkeitsmaß verwendet.
    - Korrelationsbasierte Distanz:
        - Betrachtet zwei Beobachtungen als ähnlich, wenn ihre Merkmale hoch korreliert sind.
        - Konzentriert sich auf die Form der Beobachtungsprofile, nicht auf deren Größe.
    - Hat starken Einfluss auf das entstehende Dendrogramm und das Clustering-Ergebnis.
    - Die Wahl sollte sich richten nach:
        - Der Art der zu clusternden Daten.
        - Der wissenschaftlichen Fragestellung oder dem Geschäftsziel.
    - Beispiel: Ein Online-Händler möchte Käufer nach Kaufhistorie clustern.
        - Daten: Zeilen = Käufer, Spalten = Artikel, Einträge = Kaufanzahlen.
        - Euklidische Distanz: Gruppiert seltene Käufer zusammen, ignoriert Präferenzen.
        - Korrelationsbasierte Distanz: Gruppiert Käufer mit ähnlichen Präferenzen (z.B. kaufen A & B, nie C & D), unabhängig vom Kaufvolumen – besser geeignet, um präferenzbasierte Subgruppen zu finden.
- Variablen skalieren
    - Einige Artikel werden häufiger gekauft (z.B. Socken vs. Computer), wodurch häufig gekaufte Artikel die Distanzmaße dominieren können.
        - Beispiel: Socken-Käufe überwiegen Computer-Käufe beim Clustering, was möglicherweise nicht den Geschäftsprioritäten entspricht.
    - Skalierung (Standardabweichung = 1) gibt jeder Variablen das gleiche Gewicht, verhindert Verzerrungen durch unterschiedliche Maßeinheiten und ist hilfreich, wenn Variablen unterschiedliche Skalen haben.
    - Das Skalieren hängt von den Zielen ab und beeinflusst sowohl hierarchisches als auch K-Means-Clustering.
- Probleme
    - Kleine Entscheidungen mit großen Konsequenzen: K, Linkage-Typ, Unähnlichkeitsmaß.
        - Oft werden mehrere Kombinationen ausprobiert.
        - Es gibt selten eine einzig „richtige“ Lösung – jede Variante, die interessante Aspekte der Daten sichtbar macht, ist wertvoll.
    - K-Means und hierarchisches Clustering ordnen alle Beobachtungen Clustern zu, was problematisch ist, wenn es viele Ausreißer gibt oder die Mehrheit der Daten nur wenigen Untergruppen entspricht. Das Erzwingen von Clustern für alle Daten kann zu Verzerrungen führen.
        - Mischungsmodelle („mixture models“) bieten „weiches“ Clustering und gehen besser mit Ausreißern um.
    - Clustering-Verfahren sind oft instabil; das Entfernen einiger Datenpunkte und erneutes Clustern kann sehr unterschiedliche Ergebnisse liefern. Idealerweise sollten Cluster stabil sein, sind es aber häufig nicht.
        - Überprüfe die Robustheit der Cluster, indem du auf Teilmengen der Daten clustert.
    - Clustering bildet immer Gruppen, aber es ist unklar, ob diese echt sind oder nur Rauschen.
        - Würden neue Daten die gleichen Cluster ergeben?
        - Dies zu beurteilen ist schwierig; einige Methoden weisen Clustern p-Werte zu, aber es gibt keinen allgemein anerkannten Ansatz.
    - Lösungen: Führe das Clustering mit verschiedenen Parametereinstellungen durch (Standardisierung, Linkage, Clusteranzahl etc.).

```python=
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scale = scaler.fit_transform(X)
from sklearn.cluster import AgglomerativeClustering
HClust = AgglomerativeClustering
hc_comp = HClust(distance_threshold=0, n_clusters=None, linkage='complete').fit(X_scale )
# linkage='average', linkage='single'
from ISLP.cluster import compute_linkage
linkage_comp = compute_linkage(hc_comp)
from scipy.cluster.hierarchy import dendrogram, cut_tree
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
dendrogram(linkage_comp, ax=ax, color_threshold=-np.inf, above_threshold_color='k')
cut_tree(linkage_comp, n_clusters=4).T
```

```python=
corD = 1 - np.corrcoef(X_scale)
from sklearn.cluster import AgglomerativeClustering
HClust = AgglomerativeClustering
hc_cor = HClust(linkage='complete', distance_threshold=0, n_clusters=None, metric='precomputed').fit(corD)
# metric='precomputed' indicates that the input data is a precomputed distance matrix.
# metric='precomputed' for Correlation-based distance
```
