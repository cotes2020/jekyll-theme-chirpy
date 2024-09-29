```python
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from IPython.display import display, Markdown
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import patsy
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
```


```python
def render_plotly_html(fig: go.Figure) -> None:
    fig.show()
    display(
        Markdown(
            fig.to_html(
                include_plotlyjs="cdn",
            )
        )
    )


def render_table(df: pd.DataFrame | pd.Series) -> Markdown:
    return Markdown(df.to_markdown())
```

## NFL Game Data.
This data was collected through [ESPN's public API](https://github.com/pseudo-r/Public-ESPN-API). Our goal will be to model the probability of a home team beating an away team,

$$p_{ij} = \text{Probability that home team $i$ beats away team $j$}$$

We will use the [Bradley Terry model](https://web.stanford.edu/class/archive/stats/stats200/stats200.1172/Lecture24.pdf) using the NFL game data.


```python
df = pd.concat([pd.read_csv("games_2023.csv"), pd.read_csv("games_2024.csv")])
print(df.shape)
ALL_TEAMS = list(sorted(set(df.awayTeamAbbr)))
render_table(df.describe())
```

    (332, 60)





|       |   season_type |      week |           event |   firstDowns_away |   firstDownsPassing_away |   firstDownsRushing_away |   firstDownsPenalty_away |   totalOffensivePlays_away |   totalYards_away |   yardsPerPlay_away |   totalDrives_away |   netPassingYards_away |   yardsPerPass_away |   interceptions_away |   rushingYards_away |   rushingAttempts_away |   yardsPerRushAttempt_away |   turnovers_away |   fumblesLost_away |   interceptions_away.1 |   defensiveTouchdowns_away |   firstDowns_home |   firstDownsPassing_home |   firstDownsRushing_home |   firstDownsPenalty_home |   totalOffensivePlays_home |   totalYards_home |   yardsPerPlay_home |   totalDrives_home |   netPassingYards_home |   yardsPerPass_home |   interceptions_home |   rushingYards_home |   rushingAttempts_home |   yardsPerRushAttempt_home |   turnovers_home |   fumblesLost_home |   interceptions_home.1 |   defensiveTouchdowns_home |   awayScore |   homeScore |   awayTeamId |   homeTeamId |        year |
|:------|--------------:|----------:|----------------:|------------------:|-------------------------:|-------------------------:|-------------------------:|---------------------------:|------------------:|--------------------:|-------------------:|-----------------------:|--------------------:|---------------------:|--------------------:|-----------------------:|---------------------------:|-----------------:|-------------------:|-----------------------:|---------------------------:|------------------:|-------------------------:|-------------------------:|-------------------------:|---------------------------:|------------------:|--------------------:|-------------------:|-----------------------:|--------------------:|---------------------:|--------------------:|-----------------------:|---------------------------:|-----------------:|-------------------:|-----------------------:|---------------------------:|------------:|------------:|-------------:|-------------:|------------:|
| count |    332        | 332       |   332           |         332       |                332       |                332       |                332       |                  332       |           332     |           332       |          332       |               332      |           332       |           332        |            332      |              332       |                  332       |        332       |         332        |             332        |                 332        |         332       |                 332      |                332       |                332       |                  332       |          332      |           332       |          332       |               332      |           332       |           332        |             332     |              332       |                  332       |        332       |         332        |             332        |                 332        |   332       |    332      |    332       |    332       |  332        |
| mean  |      2.03614  |   8.13855 |     4.01565e+08 |          18.7018  |                 10.7831  |                  6.26205 |                  1.65663 |                   62.3886  |           320.193 |             5.11566 |           10.9759  |               210.798  |             5.95542 |             0.792169 |            109.395  |               26.4127  |                    4.08253 |          1.33434 |           0.542169 |               0.792169 |                   0.162651 |          19.8072  |                  11.2229 |                  6.78614 |                  1.79819 |                   63.1416  |          342.117  |             5.43283 |           11.0452  |               224.039  |             6.35361 |             0.746988 |             118.078 |               27.2289  |                    4.27861 |          1.26506 |           0.518072 |               0.746988 |                   0.105422 |    20.4398  |     23.1355 |     16.7982  |     16.4669  | 2023.14     |
| std   |      0.186932 |   5.62623 | 43744.9         |           4.85857 |                  3.60864 |                  2.96617 |                  1.36071 |                    8.44189 |            83.831 |             1.14303 |            1.69218 |                72.2784 |             1.97181 |             0.900922 |             46.2201 |                7.38179 |                    1.15109 |          1.19653 |           0.762429 |               0.900922 |                   0.393362 |           4.78153 |                   3.7463 |                  3.26096 |                  1.40723 |                    8.34606 |           84.0121 |             1.22029 |            1.72971 |                74.2508 |             2.07376 |             0.880722 |              50.733 |                7.78809 |                    1.21486 |          1.1327  |           0.735158 |               0.880722 |                   0.353277 |     9.40878 |     10.3198 |      9.42348 |      9.57442 |    0.352206 |
| min   |      2        |   1       |     4.01547e+08 |           6       |                  3       |                  0       |                  0       |                   38       |            58     |             1.2     |            6       |                17      |             0.6     |             0        |             23      |                9       |                    1.2     |          0       |           0        |               0        |                   0        |           6       |                   0      |                  0       |                  0       |                   41       |          119      |             2.1     |            6       |                -9      |            -0.5     |             0        |              17     |               10       |                    1.5     |          0       |           0        |               0        |                   0        |     0       |      0      |      1       |      1       | 2023        |
| 25%   |      2        |   3       |     4.01547e+08 |          15       |                  8       |                  4       |                  1       |                   56       |           265     |             4.4     |           10       |               160      |             4.7     |             0        |             73      |               21       |                    3.3     |          0       |           0        |               0        |                   0        |          17       |                   9      |                  5       |                  1       |                   57.75    |          283      |             4.6     |           10       |               169.75   |             4.9     |             0        |              82     |               22       |                    3.5     |          0       |           0        |               0        |                   0        |    14       |     17      |      9       |      8       | 2023        |
| 50%   |      2        |   7.5     |     4.01548e+08 |          19       |                 11       |                  6       |                  1       |                   62       |           323     |             5.1     |           11       |               211      |             5.8     |             1        |            106      |               25       |                    4       |          1       |           0        |               1        |                   0        |          20       |                  11      |                  6.5     |                  2       |                   63       |          351      |             5.5     |           11       |               219      |             6.3     |             1        |             112     |               27       |                    4.2     |          1       |           0        |               1        |                   0        |    20       |     23      |     17       |     16       | 2023        |
| 75%   |      2        |  13       |     4.01548e+08 |          22       |                 13       |                  8       |                  2       |                   68       |           378     |             5.9     |           12       |               259.25   |             7.1     |             1        |            139      |               31.25    |                    4.8     |          2       |           1        |               1        |                   0        |          23       |                  13      |                  9       |                  3       |                   68       |          396.25   |             6.1     |           12       |               277      |             7.5     |             1        |             147.25  |               32       |                    5       |          2       |           1        |               1        |                   0        |    27       |     29.25   |     25       |     24.25    | 2023        |
| max   |      3        |  18       |     4.01672e+08 |          32       |                 23       |                 16       |                  6       |                   92       |           544     |             8.3     |           17       |               466      |            14.2     |             4        |            274      |               46       |                    7.8     |          5       |           4        |               4        |                   2        |          37       |                  25      |                 20       |                  8       |                   89       |          726      |            10.2     |           17       |               472      |            14.4     |             5        |             350     |               53       |                    9.7     |          6       |           3        |               5        |                   2        |    48       |     70      |     34       |     34       | 2024        |



### Treatment Contrasts
For a good review of different contrasts, see [this documentation](https://www.statsmodels.org/stable/examples/notebooks/generated/contrasts.html). We will use the Treatment (dummy) coding.


```python
# Dummy encode the home team as an indicator variable.
# We `drop_first=True` to avoid collinearity amongst the variables.
# In this case, `ARI` is dropped from the dataset and represents
# the reference level (equivalent with setting \beta_ARI = 0).
X_home = pd.get_dummies(df.homeTeamAbbr, drop_first=True).astype(int)
# Ensure we have linear independence.
assert not (X_home.sum(axis=1) == 1).all()
render_table(X_home.head())
```




|    |   ATL |   BAL |   BUF |   CAR |   CHI |   CIN |   CLE |   DAL |   DEN |   DET |   GB |   HOU |   IND |   JAX |   KC |   LAC |   LAR |   LV |   MIA |   MIN |   NE |   NO |   NYG |   NYJ |   PHI |   PIT |   SEA |   SF |   TB |   TEN |   WSH |
|---:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|-----:|------:|------:|------:|-----:|------:|------:|-----:|------:|------:|-----:|-----:|------:|------:|------:|------:|------:|-----:|-----:|------:|------:|
|  0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |    0 |     0 |     0 |     0 |    0 |     0 |     0 |    0 |     0 |     0 |    0 |    0 |     0 |     1 |     0 |     0 |     0 |    0 |    0 |     0 |     0 |
|  1 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |    0 |     0 |     0 |     0 |    1 |     0 |     0 |    0 |     0 |     0 |    0 |    0 |     0 |     0 |     0 |     0 |     0 |    0 |    0 |     0 |     0 |
|  2 |     0 |     1 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |    0 |     0 |     0 |     0 |    0 |     0 |     0 |    0 |     0 |     0 |    0 |    0 |     0 |     0 |     0 |     0 |     0 |    0 |    0 |     0 |     0 |
|  3 |     0 |     0 |     0 |     0 |     0 |     0 |     1 |     0 |     0 |     0 |    0 |     0 |     0 |     0 |    0 |     0 |     0 |    0 |     0 |     0 |    0 |    0 |     0 |     0 |     0 |     0 |     0 |    0 |    0 |     0 |     0 |
|  4 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |    0 |     0 |     0 |     0 |    0 |     0 |     0 |    0 |     0 |     1 |    0 |    0 |     0 |     0 |     0 |     0 |     0 |    0 |    0 |     0 |     0 |




```python
# Same for the away team.
X_away = pd.get_dummies(df.awayTeamAbbr, drop_first=True).astype(int)
# Ensure we have linear independence.
assert not (X_away.sum(axis=1) == 1).all()
render_table(X_away.head())
```




|    |   ATL |   BAL |   BUF |   CAR |   CHI |   CIN |   CLE |   DAL |   DEN |   DET |   GB |   HOU |   IND |   JAX |   KC |   LAC |   LAR |   LV |   MIA |   MIN |   NE |   NO |   NYG |   NYJ |   PHI |   PIT |   SEA |   SF |   TB |   TEN |   WSH |
|---:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|-----:|------:|------:|------:|-----:|------:|------:|-----:|------:|------:|-----:|-----:|------:|------:|------:|------:|------:|-----:|-----:|------:|------:|
|  0 |     0 |     0 |     1 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |    0 |     0 |     0 |     0 |    0 |     0 |     0 |    0 |     0 |     0 |    0 |    0 |     0 |     0 |     0 |     0 |     0 |    0 |    0 |     0 |     0 |
|  1 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     1 |    0 |     0 |     0 |     0 |    0 |     0 |     0 |    0 |     0 |     0 |    0 |    0 |     0 |     0 |     0 |     0 |     0 |    0 |    0 |     0 |     0 |
|  2 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |    0 |     1 |     0 |     0 |    0 |     0 |     0 |    0 |     0 |     0 |    0 |    0 |     0 |     0 |     0 |     0 |     0 |    0 |    0 |     0 |     0 |
|  3 |     0 |     0 |     0 |     0 |     0 |     1 |     0 |     0 |     0 |     0 |    0 |     0 |     0 |     0 |    0 |     0 |     0 |    0 |     0 |     0 |    0 |    0 |     0 |     0 |     0 |     0 |     0 |    0 |    0 |     0 |     0 |
|  4 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |    0 |     0 |     0 |     0 |    0 |     0 |     0 |    0 |     0 |     0 |    0 |    0 |     0 |     0 |     0 |     0 |     0 |    0 |    1 |     0 |     0 |



## Bradley Terry Model Contrasts

To specify the Bradley Terry model, we need to model the (logit) of the probability of the home team winning as the difference between the coefficient's for each team:

$$\log{\frac{p_{ij}}{1 - p_{ij}}} = \beta_i - \beta_j$$

$$\iff p_{ij} = \frac{\exp{\beta_i - \beta_j}}{1 + \exp{\beta_i - \beta_j}} = \frac{\exp{\beta_i}}{\exp{\beta_i} + \exp{\beta_j}}$$

We can represent this numerically by taking the difference of the contrasts we created before, `X_diff = X_home - X_away`. Each team pairing will be uniquely encoded and the parameter estimate will focus only on the difference of those two teams.


```python
# We should have the same teams represented in the home and away teams.
assert (X_home.columns == X_away.columns).all()
# X_diff will represent which two teams played that week.
# Specifically, +1 will represent home and -1 will be away (except reference level).
X_diff = X_home - X_away
# Ensure we have linear independence.
assert not (X_away.sum(axis=1) == 0).all()
render_table(X_diff.head())
```




|    |   ATL |   BAL |   BUF |   CAR |   CHI |   CIN |   CLE |   DAL |   DEN |   DET |   GB |   HOU |   IND |   JAX |   KC |   LAC |   LAR |   LV |   MIA |   MIN |   NE |   NO |   NYG |   NYJ |   PHI |   PIT |   SEA |   SF |   TB |   TEN |   WSH |
|---:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|-----:|------:|------:|------:|-----:|------:|------:|-----:|------:|------:|-----:|-----:|------:|------:|------:|------:|------:|-----:|-----:|------:|------:|
|  0 |     0 |     0 |    -1 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |    0 |     0 |     0 |     0 |    0 |     0 |     0 |    0 |     0 |     0 |    0 |    0 |     0 |     1 |     0 |     0 |     0 |    0 |    0 |     0 |     0 |
|  1 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |    -1 |    0 |     0 |     0 |     0 |    1 |     0 |     0 |    0 |     0 |     0 |    0 |    0 |     0 |     0 |     0 |     0 |     0 |    0 |    0 |     0 |     0 |
|  2 |     0 |     1 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |    0 |    -1 |     0 |     0 |    0 |     0 |     0 |    0 |     0 |     0 |    0 |    0 |     0 |     0 |     0 |     0 |     0 |    0 |    0 |     0 |     0 |
|  3 |     0 |     0 |     0 |     0 |     0 |    -1 |     1 |     0 |     0 |     0 |    0 |     0 |     0 |     0 |    0 |     0 |     0 |    0 |     0 |     0 |    0 |    0 |     0 |     0 |     0 |     0 |     0 |    0 |    0 |     0 |     0 |
|  4 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |    0 |     0 |     0 |     0 |    0 |     0 |     0 |    0 |     0 |     1 |    0 |    0 |     0 |     0 |     0 |     0 |     0 |    0 |   -1 |     0 |     0 |




```python
# We can visualize the "encoding" and see how often one team played another.
def encoding(x: pd.Series) -> str:
    # Just concatenate all of the contrasts into one long string encoding.
    return "".join(str(i) for i in x)


# We can see how the encoding maps back to the original teams
# and the frequency of this matchup.
df["encoding"] = X_diff.apply(encoding, axis=1)
render_table(
    df.groupby("encoding")[["homeTeamAbbr", "awayTeamAbbr"]]
    .agg(["unique", "count"])
    .sort_values(
        [("homeTeamAbbr", "count"), ("awayTeamAbbr", "count")],  # type: ignore
        ascending=False,
    )
    .head()
)
```




| encoding                         | ('homeTeamAbbr', 'unique')   |   ('homeTeamAbbr', 'count') | ('awayTeamAbbr', 'unique')   |   ('awayTeamAbbr', 'count') |
|:---------------------------------|:-----------------------------|----------------------------:|:-----------------------------|----------------------------:|
| 00-10000000000000001000000000000 | ['MIA']                      |                           2 | ['BUF']                      |                           2 |
| 000-1000000000000000001000000000 | ['NO']                       |                           2 | ['CAR']                      |                           2 |
| 00000-10000000010000000000000000 | ['KC']                       |                           2 | ['CIN']                      |                           2 |
| 000000-1000010000000000000000000 | ['HOU']                      |                           2 | ['CLE']                      |                           2 |
| 00000000000-11000000000000000000 | ['IND']                      |                           2 | ['HOU']                      |                           2 |




```python
# Tie goes to the home team, since we want to keep the outcomes binary.
y = pd.Series((df.homeScore >= df.awayScore), name="homeTeamWin").astype(int)
render_table(y.head())
```




|    |   homeTeamWin |
|---:|--------------:|
|  0 |             1 |
|  1 |             0 |
|  2 |             1 |
|  3 |             1 |
|  4 |             0 |



We also include an intercept $\alpha$, which is a coefficient for the home team advantage. Our final model is then:

$$\log{\frac{p_{ij}}{1 - p_{ij}}} = \alpha + \beta_i - \beta_j = \alpha + \Delta_{ij}$$

Where $\Delta_{ij}$ is the difference of the contrasts we computed before.

### Design Matrix


```python
formula = "homeTeamWin ~ 1 + " + " + ".join(X_diff.columns)
y, X = patsy.dmatrices(
    formula,
    pd.concat([y, X_diff], axis=1),
    return_type="dataframe",
)
Markdown(formula)
```




homeTeamWin ~ 1 + ATL + BAL + BUF + CAR + CHI + CIN + CLE + DAL + DEN + DET + GB + HOU + IND + JAX + KC + LAC + LAR + LV + MIA + MIN + NE + NO + NYG + NYJ + PHI + PIT + SEA + SF + TB + TEN + WSH



### VIF
[Variance inflation factor](https://www.statsmodels.org/dev/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html) measures the amount of collinearity in the dataset. It is recommended to keep this under < 5 when performing statistical inference.


```python
# Print the VIF to see how much collinearity there are in the features.
# We want to avoid VIF > 5.
render_plotly_html(
    px.histogram(
        pd.Series(
            [variance_inflation_factor(X, i) for i, _ in enumerate(X.columns)],
            name="VIF",
        ),
        nbins=15,
        title="Distribution of VIF",
    )
)
```




<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>                        <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>                <div id="73fd6e32-fe49-4e52-8365-4b59c68b6b65" class="plotly-graph-div" style="height:100%; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("73fd6e32-fe49-4e52-8365-4b59c68b6b65")) {                    Plotly.newPlot(                        "73fd6e32-fe49-4e52-8365-4b59c68b6b65",                        [{"alignmentgroup":"True","bingroup":"x","hovertemplate":"variable=VIF\u003cbr\u003evalue=%{x}\u003cbr\u003ecount=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"VIF","marker":{"color":"#636efa","pattern":{"shape":""}},"name":"VIF","nbinsx":15,"offsetgroup":"VIF","orientation":"v","showlegend":true,"x":[1.0134009895869243,2.0373059007662526,2.0830561156313934,2.164888339592824,2.150138413144364,2.038577703812638,1.9713446735097981,2.0228447560475575,2.0137827110033144,2.130076217173779,2.1637444455123194,2.2040256229236155,2.156434661859151,2.1263988128694407,2.1224991710947227,2.312628538907933,2.1672970463345838,1.8025768042666699,2.166058579318688,2.1993130917298673,2.1081560998285966,2.1339159889650183,2.1310163337320276,1.976617317931521,2.1306788849020437,2.031876104130207,2.025844009899823,1.838263288785051,1.9395894935237872,2.235846107384558,2.1225605353413894,1.9686563374012414],"xaxis":"x","yaxis":"y","type":"histogram"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"value"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"count"}},"legend":{"title":{"text":"variable"},"tracegroupgap":0},"title":{"text":"Distribution of VIF"},"barmode":"relative"},                        {"responsive": true}                    )                };                            </script>        </div>
</body>
</html>


## Bradley Terry GLM
Now we fit the BT model. Since we are modeling logits of a probability, this reduces to logistic regression. This is widely available, we will use [statsmodel's GLM](https://www.statsmodels.org/stable/glm.html).


```python
# Fit a GLM with Binomial family and logit link function.
results = sm.GLM(
    endog=y,
    exog=X,
    family=sm.families.Binomial(
        link=sm.families.links.Logit(),
    ),
).fit()
```


```python
results.summary()
```




<table class="simpletable">
<caption>Generalized Linear Model Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>      <td>homeTeamWin</td>   <th>  No. Observations:  </th>  <td>   332</td> 
</tr>
<tr>
  <th>Model:</th>                  <td>GLM</td>       <th>  Df Residuals:      </th>  <td>   300</td> 
</tr>
<tr>
  <th>Model Family:</th>        <td>Binomial</td>     <th>  Df Model:          </th>  <td>    31</td> 
</tr>
<tr>
  <th>Link Function:</th>         <td>Logit</td>      <th>  Scale:             </th> <td>  1.0000</td>
</tr>
<tr>
  <th>Method:</th>                <td>IRLS</td>       <th>  Log-Likelihood:    </th> <td> -198.10</td>
</tr>
<tr>
  <th>Date:</th>            <td>Sat, 28 Sep 2024</td> <th>  Deviance:          </th> <td>  396.19</td>
</tr>
<tr>
  <th>Time:</th>                <td>23:13:58</td>     <th>  Pearson chi2:      </th>  <td>  332.</td> 
</tr>
<tr>
  <th>No. Iterations:</th>          <td>4</td>        <th>  Pseudo R-squ. (CS):</th>  <td>0.1657</td> 
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>    0.2235</td> <td>    0.122</td> <td>    1.829</td> <td> 0.067</td> <td>   -0.016</td> <td>    0.463</td>
</tr>
<tr>
  <th>ATL</th>       <td>    0.2448</td> <td>    0.724</td> <td>    0.338</td> <td> 0.735</td> <td>   -1.174</td> <td>    1.664</td>
</tr>
<tr>
  <th>BAL</th>       <td>    1.9120</td> <td>    0.717</td> <td>    2.666</td> <td> 0.008</td> <td>    0.506</td> <td>    3.318</td>
</tr>
<tr>
  <th>BUF</th>       <td>    1.5916</td> <td>    0.734</td> <td>    2.167</td> <td> 0.030</td> <td>    0.152</td> <td>    3.031</td>
</tr>
<tr>
  <th>CAR</th>       <td>   -0.9843</td> <td>    0.855</td> <td>   -1.152</td> <td> 0.249</td> <td>   -2.659</td> <td>    0.691</td>
</tr>
<tr>
  <th>CHI</th>       <td>    0.3354</td> <td>    0.727</td> <td>    0.461</td> <td> 0.645</td> <td>   -1.089</td> <td>    1.760</td>
</tr>
<tr>
  <th>CIN</th>       <td>    0.8876</td> <td>    0.713</td> <td>    1.245</td> <td> 0.213</td> <td>   -0.510</td> <td>    2.285</td>
</tr>
<tr>
  <th>CLE</th>       <td>    1.2966</td> <td>    0.708</td> <td>    1.832</td> <td> 0.067</td> <td>   -0.091</td> <td>    2.684</td>
</tr>
<tr>
  <th>DAL</th>       <td>    1.3452</td> <td>    0.711</td> <td>    1.892</td> <td> 0.058</td> <td>   -0.048</td> <td>    2.739</td>
</tr>
<tr>
  <th>DEN</th>       <td>    0.7358</td> <td>    0.740</td> <td>    0.994</td> <td> 0.320</td> <td>   -0.714</td> <td>    2.186</td>
</tr>
<tr>
  <th>DET</th>       <td>    1.7979</td> <td>    0.719</td> <td>    2.499</td> <td> 0.012</td> <td>    0.388</td> <td>    3.208</td>
</tr>
<tr>
  <th>GB</th>        <td>    1.1053</td> <td>    0.718</td> <td>    1.540</td> <td> 0.124</td> <td>   -0.302</td> <td>    2.513</td>
</tr>
<tr>
  <th>HOU</th>       <td>    1.1260</td> <td>    0.714</td> <td>    1.577</td> <td> 0.115</td> <td>   -0.273</td> <td>    2.525</td>
</tr>
<tr>
  <th>IND</th>       <td>    0.7037</td> <td>    0.733</td> <td>    0.960</td> <td> 0.337</td> <td>   -0.733</td> <td>    2.141</td>
</tr>
<tr>
  <th>JAX</th>       <td>    0.7857</td> <td>    0.741</td> <td>    1.060</td> <td> 0.289</td> <td>   -0.667</td> <td>    2.239</td>
</tr>
<tr>
  <th>KC</th>        <td>    2.0205</td> <td>    0.749</td> <td>    2.698</td> <td> 0.007</td> <td>    0.553</td> <td>    3.488</td>
</tr>
<tr>
  <th>LAC</th>       <td>    0.2262</td> <td>    0.760</td> <td>    0.298</td> <td> 0.766</td> <td>   -1.262</td> <td>    1.715</td>
</tr>
<tr>
  <th>LAR</th>       <td>    1.2363</td> <td>    0.670</td> <td>    1.845</td> <td> 0.065</td> <td>   -0.077</td> <td>    2.549</td>
</tr>
<tr>
  <th>LV</th>        <td>    0.6112</td> <td>    0.753</td> <td>    0.812</td> <td> 0.417</td> <td>   -0.864</td> <td>    2.087</td>
</tr>
<tr>
  <th>MIA</th>       <td>    1.1409</td> <td>    0.746</td> <td>    1.530</td> <td> 0.126</td> <td>   -0.321</td> <td>    2.602</td>
</tr>
<tr>
  <th>MIN</th>       <td>    0.9293</td> <td>    0.730</td> <td>    1.272</td> <td> 0.203</td> <td>   -0.502</td> <td>    2.361</td>
</tr>
<tr>
  <th>NE</th>        <td>   -0.2335</td> <td>    0.778</td> <td>   -0.300</td> <td> 0.764</td> <td>   -1.757</td> <td>    1.290</td>
</tr>
<tr>
  <th>NO</th>        <td>    0.7731</td> <td>    0.743</td> <td>    1.041</td> <td> 0.298</td> <td>   -0.682</td> <td>    2.229</td>
</tr>
<tr>
  <th>NYG</th>       <td>    0.2182</td> <td>    0.718</td> <td>    0.304</td> <td> 0.761</td> <td>   -1.190</td> <td>    1.626</td>
</tr>
<tr>
  <th>NYJ</th>       <td>    0.5562</td> <td>    0.744</td> <td>    0.748</td> <td> 0.455</td> <td>   -0.902</td> <td>    2.014</td>
</tr>
<tr>
  <th>PHI</th>       <td>    1.3884</td> <td>    0.714</td> <td>    1.944</td> <td> 0.052</td> <td>   -0.012</td> <td>    2.788</td>
</tr>
<tr>
  <th>PIT</th>       <td>    1.4859</td> <td>    0.716</td> <td>    2.076</td> <td> 0.038</td> <td>    0.083</td> <td>    2.889</td>
</tr>
<tr>
  <th>SEA</th>       <td>    1.3372</td> <td>    0.701</td> <td>    1.907</td> <td> 0.057</td> <td>   -0.037</td> <td>    2.712</td>
</tr>
<tr>
  <th>SF</th>        <td>    1.8241</td> <td>    0.699</td> <td>    2.611</td> <td> 0.009</td> <td>    0.455</td> <td>    3.193</td>
</tr>
<tr>
  <th>TB</th>        <td>    1.0179</td> <td>    0.729</td> <td>    1.396</td> <td> 0.163</td> <td>   -0.411</td> <td>    2.447</td>
</tr>
<tr>
  <th>TEN</th>       <td>   -0.1036</td> <td>    0.763</td> <td>   -0.136</td> <td> 0.892</td> <td>   -1.599</td> <td>    1.392</td>
</tr>
<tr>
  <th>WSH</th>       <td>   -0.0545</td> <td>    0.726</td> <td>   -0.075</td> <td> 0.940</td> <td>   -1.478</td> <td>    1.369</td>
</tr>
</table>




```python
# Verify that the Binomial family with a Logit link is equivalent with the Logit binary model.
logit_params = sm.Logit(endog=y, exog=X).fit().params
pd.testing.assert_series_equal(logit_params, results.params)
```

    Optimization terminated successfully.
             Current function value: 0.596675
             Iterations 6



```python
# Compute the confusion matrix on the training predictions.
render_table(
    pd.DataFrame(
        confusion_matrix(y_pred=results.predict(X) > 0.5, y_true=y),
        columns=["Home Loss", "Home Win"],
        index=["Home Loss", "Home Win"],
    )
)
```




|           |   Home Loss |   Home Win |
|:----------|------------:|-----------:|
| Home Loss |          89 |         59 |
| Home Win  |          43 |        141 |



## Analysis of GLM Results


```python
results_df = pd.DataFrame(
    {
        "p-value": results.pvalues,
        "parameter": results.params,
        "team": results.params.index,
        "Parameter Rank": results.params.rank(ascending=False),
    }
)
```

### P-values


```python
# Visualize the parameter estimate vs. p-value.
# As expected, this should roughly trace out a normal distribution.
render_plotly_html(
    px.scatter(
        results_df,
        x="parameter",
        y="p-value",
        color="team",
        title="P-Value vs. Parameter Estimate",
    )
)
```




<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>                        <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>                <div id="000c0ac4-f22a-44dd-95c2-b45db6fcebe1" class="plotly-graph-div" style="height:100%; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("000c0ac4-f22a-44dd-95c2-b45db6fcebe1")) {                    Plotly.newPlot(                        "000c0ac4-f22a-44dd-95c2-b45db6fcebe1",                        [{"hovertemplate":"team=Intercept\u003cbr\u003eparameter=%{x}\u003cbr\u003ep-value=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"Intercept","marker":{"color":"#636efa","symbol":"circle"},"mode":"markers","name":"Intercept","orientation":"v","showlegend":true,"x":[0.2235338927155498],"xaxis":"x","y":[0.0674191413492592],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=ATL\u003cbr\u003eparameter=%{x}\u003cbr\u003ep-value=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"ATL","marker":{"color":"#EF553B","symbol":"circle"},"mode":"markers","name":"ATL","orientation":"v","showlegend":true,"x":[0.24483750911080399],"xaxis":"x","y":[0.7352316317767453],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=BAL\u003cbr\u003eparameter=%{x}\u003cbr\u003ep-value=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"BAL","marker":{"color":"#00cc96","symbol":"circle"},"mode":"markers","name":"BAL","orientation":"v","showlegend":true,"x":[1.9120176552670398],"xaxis":"x","y":[0.007674766580901157],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=BUF\u003cbr\u003eparameter=%{x}\u003cbr\u003ep-value=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"BUF","marker":{"color":"#ab63fa","symbol":"circle"},"mode":"markers","name":"BUF","orientation":"v","showlegend":true,"x":[1.59163488824148],"xaxis":"x","y":[0.03022979014051185],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=CAR\u003cbr\u003eparameter=%{x}\u003cbr\u003ep-value=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"CAR","marker":{"color":"#FFA15A","symbol":"circle"},"mode":"markers","name":"CAR","orientation":"v","showlegend":true,"x":[-0.984282622292548],"xaxis":"x","y":[0.24938202760892614],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=CHI\u003cbr\u003eparameter=%{x}\u003cbr\u003ep-value=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"CHI","marker":{"color":"#19d3f3","symbol":"circle"},"mode":"markers","name":"CHI","orientation":"v","showlegend":true,"x":[0.3353504896317428],"xaxis":"x","y":[0.6445378219614288],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=CIN\u003cbr\u003eparameter=%{x}\u003cbr\u003ep-value=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"CIN","marker":{"color":"#FF6692","symbol":"circle"},"mode":"markers","name":"CIN","orientation":"v","showlegend":true,"x":[0.8876446702457397],"xaxis":"x","y":[0.2130918931836906],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=CLE\u003cbr\u003eparameter=%{x}\u003cbr\u003ep-value=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"CLE","marker":{"color":"#B6E880","symbol":"circle"},"mode":"markers","name":"CLE","orientation":"v","showlegend":true,"x":[1.2965631142534657],"xaxis":"x","y":[0.06700037638066571],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=DAL\u003cbr\u003eparameter=%{x}\u003cbr\u003ep-value=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"DAL","marker":{"color":"#FF97FF","symbol":"circle"},"mode":"markers","name":"DAL","orientation":"v","showlegend":true,"x":[1.34519480901394],"xaxis":"x","y":[0.05848328482134264],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=DEN\u003cbr\u003eparameter=%{x}\u003cbr\u003ep-value=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"DEN","marker":{"color":"#FECB52","symbol":"circle"},"mode":"markers","name":"DEN","orientation":"v","showlegend":true,"x":[0.7358257205563545],"xaxis":"x","y":[0.3199923237425235],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=DET\u003cbr\u003eparameter=%{x}\u003cbr\u003ep-value=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"DET","marker":{"color":"#636efa","symbol":"circle"},"mode":"markers","name":"DET","orientation":"v","showlegend":true,"x":[1.7979012514091246],"xaxis":"x","y":[0.01244461727772893],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=GB\u003cbr\u003eparameter=%{x}\u003cbr\u003ep-value=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"GB","marker":{"color":"#EF553B","symbol":"circle"},"mode":"markers","name":"GB","orientation":"v","showlegend":true,"x":[1.105346441745444],"xaxis":"x","y":[0.12366758399313202],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=HOU\u003cbr\u003eparameter=%{x}\u003cbr\u003ep-value=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"HOU","marker":{"color":"#00cc96","symbol":"circle"},"mode":"markers","name":"HOU","orientation":"v","showlegend":true,"x":[1.1260282919078892],"xaxis":"x","y":[0.11468932147277472],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=IND\u003cbr\u003eparameter=%{x}\u003cbr\u003ep-value=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"IND","marker":{"color":"#ab63fa","symbol":"circle"},"mode":"markers","name":"IND","orientation":"v","showlegend":true,"x":[0.7036619275349717],"xaxis":"x","y":[0.3371512844272667],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=JAX\u003cbr\u003eparameter=%{x}\u003cbr\u003ep-value=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"JAX","marker":{"color":"#FFA15A","symbol":"circle"},"mode":"markers","name":"JAX","orientation":"v","showlegend":true,"x":[0.7856645888086481],"xaxis":"x","y":[0.28924766268324276],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=KC\u003cbr\u003eparameter=%{x}\u003cbr\u003ep-value=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"KC","marker":{"color":"#19d3f3","symbol":"circle"},"mode":"markers","name":"KC","orientation":"v","showlegend":true,"x":[2.0204713195722523],"xaxis":"x","y":[0.006966942593857271],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=LAC\u003cbr\u003eparameter=%{x}\u003cbr\u003ep-value=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"LAC","marker":{"color":"#FF6692","symbol":"circle"},"mode":"markers","name":"LAC","orientation":"v","showlegend":true,"x":[0.22621176423415762],"xaxis":"x","y":[0.7658391324330528],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=LAR\u003cbr\u003eparameter=%{x}\u003cbr\u003ep-value=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"LAR","marker":{"color":"#B6E880","symbol":"circle"},"mode":"markers","name":"LAR","orientation":"v","showlegend":true,"x":[1.2362668989102168],"xaxis":"x","y":[0.06501569636891201],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=LV\u003cbr\u003eparameter=%{x}\u003cbr\u003ep-value=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"LV","marker":{"color":"#FF97FF","symbol":"circle"},"mode":"markers","name":"LV","orientation":"v","showlegend":true,"x":[0.6112058762743744],"xaxis":"x","y":[0.4168000992294323],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=MIA\u003cbr\u003eparameter=%{x}\u003cbr\u003ep-value=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"MIA","marker":{"color":"#FECB52","symbol":"circle"},"mode":"markers","name":"MIA","orientation":"v","showlegend":true,"x":[1.1409244251651698],"xaxis":"x","y":[0.1260196358150992],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=MIN\u003cbr\u003eparameter=%{x}\u003cbr\u003ep-value=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"MIN","marker":{"color":"#636efa","symbol":"circle"},"mode":"markers","name":"MIN","orientation":"v","showlegend":true,"x":[0.9292923758876177],"xaxis":"x","y":[0.2032328752036715],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=NE\u003cbr\u003eparameter=%{x}\u003cbr\u003ep-value=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"NE","marker":{"color":"#EF553B","symbol":"circle"},"mode":"markers","name":"NE","orientation":"v","showlegend":true,"x":[-0.23352126312880434],"xaxis":"x","y":[0.7639140670641416],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=NO\u003cbr\u003eparameter=%{x}\u003cbr\u003ep-value=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"NO","marker":{"color":"#00cc96","symbol":"circle"},"mode":"markers","name":"NO","orientation":"v","showlegend":true,"x":[0.7731330097582161],"xaxis":"x","y":[0.2978327782004435],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=NYG\u003cbr\u003eparameter=%{x}\u003cbr\u003ep-value=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"NYG","marker":{"color":"#ab63fa","symbol":"circle"},"mode":"markers","name":"NYG","orientation":"v","showlegend":true,"x":[0.21822514996357978],"xaxis":"x","y":[0.761259371260067],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=NYJ\u003cbr\u003eparameter=%{x}\u003cbr\u003ep-value=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"NYJ","marker":{"color":"#FFA15A","symbol":"circle"},"mode":"markers","name":"NYJ","orientation":"v","showlegend":true,"x":[0.5561722234894783],"xaxis":"x","y":[0.45468739613129894],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=PHI\u003cbr\u003eparameter=%{x}\u003cbr\u003ep-value=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"PHI","marker":{"color":"#19d3f3","symbol":"circle"},"mode":"markers","name":"PHI","orientation":"v","showlegend":true,"x":[1.3884335589788224],"xaxis":"x","y":[0.05193245067931672],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=PIT\u003cbr\u003eparameter=%{x}\u003cbr\u003ep-value=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"PIT","marker":{"color":"#FF6692","symbol":"circle"},"mode":"markers","name":"PIT","orientation":"v","showlegend":true,"x":[1.4859236121746227],"xaxis":"x","y":[0.03788707071737749],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=SEA\u003cbr\u003eparameter=%{x}\u003cbr\u003ep-value=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"SEA","marker":{"color":"#B6E880","symbol":"circle"},"mode":"markers","name":"SEA","orientation":"v","showlegend":true,"x":[1.3371746932112787],"xaxis":"x","y":[0.056583768503733575],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=SF\u003cbr\u003eparameter=%{x}\u003cbr\u003ep-value=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"SF","marker":{"color":"#FF97FF","symbol":"circle"},"mode":"markers","name":"SF","orientation":"v","showlegend":true,"x":[1.8241075729315166],"xaxis":"x","y":[0.009026472664465934],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=TB\u003cbr\u003eparameter=%{x}\u003cbr\u003ep-value=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"TB","marker":{"color":"#FECB52","symbol":"circle"},"mode":"markers","name":"TB","orientation":"v","showlegend":true,"x":[1.0178714679860812],"xaxis":"x","y":[0.16277309836581155],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=TEN\u003cbr\u003eparameter=%{x}\u003cbr\u003ep-value=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"TEN","marker":{"color":"#636efa","symbol":"circle"},"mode":"markers","name":"TEN","orientation":"v","showlegend":true,"x":[-0.10355304810370343],"xaxis":"x","y":[0.8920156885934684],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=WSH\u003cbr\u003eparameter=%{x}\u003cbr\u003ep-value=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"WSH","marker":{"color":"#EF553B","symbol":"circle"},"mode":"markers","name":"WSH","orientation":"v","showlegend":true,"x":[-0.05445482265687207],"xaxis":"x","y":[0.9402409112703815],"yaxis":"y","type":"scatter"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"parameter"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"p-value"}},"legend":{"title":{"text":"team"},"tracegroupgap":0},"title":{"text":"P-Value vs. Parameter Estimate"}},                        {"responsive": true}                    )                };                            </script>        </div>
</body>
</html>


### Parameter Ranks


```python
# Tally all of the team's wins (if y == 1, then home won, otherwise away won).
team_wins = pd.Series(
    np.where(y.squeeze(), df.homeTeamAbbr, df.awayTeamAbbr)
).value_counts()
# Tally games played in total by each team.
team_n = pd.concat([df.homeTeamAbbr, df.awayTeamAbbr]).value_counts()
win_df = pd.DataFrame({"wins": team_wins, "n": team_n})
win_df["win"] = win_df.wins / win_df.n
win_df.sort_values(by="win", ascending=False, inplace=True)
win_df["Win Rank"] = win_df.win.rank(ascending=False)
render_table(win_df.head())
```




|     |   wins |   n |      win |   Win Rank |
|:----|-------:|----:|---------:|-----------:|
| KC  |     17 |  23 | 0.73913  |          1 |
| DET |     16 |  23 | 0.695652 |          2 |
| BAL |     15 |  22 | 0.681818 |          4 |
| BUF |     15 |  22 | 0.681818 |          4 |
| SF  |     15 |  22 | 0.681818 |          4 |




```python
# Join the parameter results with the raw win percentages.
results_win_df = results_df.merge(win_df, left_index=True, right_index=True)
# Plot the parameter rank vs. the win percentage rank
fig_rank = px.scatter(
    results_win_df,
    x="Parameter Rank",
    y="Win Rank",
    color="team",
    title="Win Percentage Rank vs. Parameter Estimate Rank",
)
# Add a reference line. If points fall on the line, then parameter_rank == win_rank.
# Otherwise, the Bradley Terry model and raw win percentage disagree.
fig_rank.add_scatter(
    x=results_win_df["Parameter Rank"],
    y=results_win_df["Parameter Rank"],
    mode="lines",
    name="Reference",
    line=dict(color="red", width=2, dash="dash"),
    hoverinfo="skip",
)
render_plotly_html(fig_rank)
```




<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>                        <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>                <div id="6e291206-bd33-4b27-8471-9c15c525e393" class="plotly-graph-div" style="height:100%; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("6e291206-bd33-4b27-8471-9c15c525e393")) {                    Plotly.newPlot(                        "6e291206-bd33-4b27-8471-9c15c525e393",                        [{"hovertemplate":"team=ATL\u003cbr\u003eParameter Rank=%{x}\u003cbr\u003eWin Rank=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"ATL","marker":{"color":"#636efa","symbol":"circle"},"mode":"markers","name":"ATL","orientation":"v","showlegend":true,"x":[25.0],"xaxis":"x","y":[24.5],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=BAL\u003cbr\u003eParameter Rank=%{x}\u003cbr\u003eWin Rank=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"BAL","marker":{"color":"#EF553B","symbol":"circle"},"mode":"markers","name":"BAL","orientation":"v","showlegend":true,"x":[2.0],"xaxis":"x","y":[4.0],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=BUF\u003cbr\u003eParameter Rank=%{x}\u003cbr\u003eWin Rank=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"BUF","marker":{"color":"#00cc96","symbol":"circle"},"mode":"markers","name":"BUF","orientation":"v","showlegend":true,"x":[5.0],"xaxis":"x","y":[4.0],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=CAR\u003cbr\u003eParameter Rank=%{x}\u003cbr\u003eWin Rank=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"CAR","marker":{"color":"#ab63fa","symbol":"circle"},"mode":"markers","name":"CAR","orientation":"v","showlegend":true,"x":[32.0],"xaxis":"x","y":[32.0],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=CHI\u003cbr\u003eParameter Rank=%{x}\u003cbr\u003eWin Rank=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"CHI","marker":{"color":"#FFA15A","symbol":"circle"},"mode":"markers","name":"CHI","orientation":"v","showlegend":true,"x":[24.0],"xaxis":"x","y":[24.5],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=CIN\u003cbr\u003eParameter Rank=%{x}\u003cbr\u003eWin Rank=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"CIN","marker":{"color":"#19d3f3","symbol":"circle"},"mode":"markers","name":"CIN","orientation":"v","showlegend":true,"x":[17.0],"xaxis":"x","y":[21.0],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=CLE\u003cbr\u003eParameter Rank=%{x}\u003cbr\u003eWin Rank=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"CLE","marker":{"color":"#FF6692","symbol":"circle"},"mode":"markers","name":"CLE","orientation":"v","showlegend":true,"x":[10.0],"xaxis":"x","y":[11.5],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=DAL\u003cbr\u003eParameter Rank=%{x}\u003cbr\u003eWin Rank=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"DAL","marker":{"color":"#B6E880","symbol":"circle"},"mode":"markers","name":"DAL","orientation":"v","showlegend":true,"x":[8.0],"xaxis":"x","y":[7.0],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=DEN\u003cbr\u003eParameter Rank=%{x}\u003cbr\u003eWin Rank=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"DEN","marker":{"color":"#FF97FF","symbol":"circle"},"mode":"markers","name":"DEN","orientation":"v","showlegend":true,"x":[20.0],"xaxis":"x","y":[21.0],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=DET\u003cbr\u003eParameter Rank=%{x}\u003cbr\u003eWin Rank=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"DET","marker":{"color":"#FECB52","symbol":"circle"},"mode":"markers","name":"DET","orientation":"v","showlegend":true,"x":[4.0],"xaxis":"x","y":[2.0],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=GB\u003cbr\u003eParameter Rank=%{x}\u003cbr\u003eWin Rank=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"GB","marker":{"color":"#636efa","symbol":"circle"},"mode":"markers","name":"GB","orientation":"v","showlegend":true,"x":[14.0],"xaxis":"x","y":[14.5],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=HOU\u003cbr\u003eParameter Rank=%{x}\u003cbr\u003eWin Rank=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"HOU","marker":{"color":"#EF553B","symbol":"circle"},"mode":"markers","name":"HOU","orientation":"v","showlegend":true,"x":[13.0],"xaxis":"x","y":[10.0],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=IND\u003cbr\u003eParameter Rank=%{x}\u003cbr\u003eWin Rank=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"IND","marker":{"color":"#00cc96","symbol":"circle"},"mode":"markers","name":"IND","orientation":"v","showlegend":true,"x":[21.0],"xaxis":"x","y":[17.5],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=JAX\u003cbr\u003eParameter Rank=%{x}\u003cbr\u003eWin Rank=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"JAX","marker":{"color":"#ab63fa","symbol":"circle"},"mode":"markers","name":"JAX","orientation":"v","showlegend":true,"x":[18.0],"xaxis":"x","y":[21.0],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=KC\u003cbr\u003eParameter Rank=%{x}\u003cbr\u003eWin Rank=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"KC","marker":{"color":"#FFA15A","symbol":"circle"},"mode":"markers","name":"KC","orientation":"v","showlegend":true,"x":[1.0],"xaxis":"x","y":[1.0],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=LAC\u003cbr\u003eParameter Rank=%{x}\u003cbr\u003eWin Rank=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"LAC","marker":{"color":"#19d3f3","symbol":"circle"},"mode":"markers","name":"LAC","orientation":"v","showlegend":true,"x":[26.0],"xaxis":"x","y":[26.5],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=LAR\u003cbr\u003eParameter Rank=%{x}\u003cbr\u003eWin Rank=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"LAR","marker":{"color":"#FF6692","symbol":"circle"},"mode":"markers","name":"LAR","orientation":"v","showlegend":true,"x":[11.0],"xaxis":"x","y":[16.0],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=LV\u003cbr\u003eParameter Rank=%{x}\u003cbr\u003eWin Rank=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"LV","marker":{"color":"#B6E880","symbol":"circle"},"mode":"markers","name":"LV","orientation":"v","showlegend":true,"x":[22.0],"xaxis":"x","y":[21.0],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=MIA\u003cbr\u003eParameter Rank=%{x}\u003cbr\u003eWin Rank=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"MIA","marker":{"color":"#FF97FF","symbol":"circle"},"mode":"markers","name":"MIA","orientation":"v","showlegend":true,"x":[12.0],"xaxis":"x","y":[11.5],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=MIN\u003cbr\u003eParameter Rank=%{x}\u003cbr\u003eWin Rank=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"MIN","marker":{"color":"#FECB52","symbol":"circle"},"mode":"markers","name":"MIN","orientation":"v","showlegend":true,"x":[16.0],"xaxis":"x","y":[17.5],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=NE\u003cbr\u003eParameter Rank=%{x}\u003cbr\u003eWin Rank=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"NE","marker":{"color":"#636efa","symbol":"circle"},"mode":"markers","name":"NE","orientation":"v","showlegend":true,"x":[31.0],"xaxis":"x","y":[30.5],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=NO\u003cbr\u003eParameter Rank=%{x}\u003cbr\u003eWin Rank=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"NO","marker":{"color":"#EF553B","symbol":"circle"},"mode":"markers","name":"NO","orientation":"v","showlegend":true,"x":[19.0],"xaxis":"x","y":[13.0],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=NYG\u003cbr\u003eParameter Rank=%{x}\u003cbr\u003eWin Rank=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"NYG","marker":{"color":"#00cc96","symbol":"circle"},"mode":"markers","name":"NYG","orientation":"v","showlegend":true,"x":[28.0],"xaxis":"x","y":[26.5],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=NYJ\u003cbr\u003eParameter Rank=%{x}\u003cbr\u003eWin Rank=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"NYJ","marker":{"color":"#ab63fa","symbol":"circle"},"mode":"markers","name":"NYJ","orientation":"v","showlegend":true,"x":[23.0],"xaxis":"x","y":[21.0],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=PHI\u003cbr\u003eParameter Rank=%{x}\u003cbr\u003eWin Rank=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"PHI","marker":{"color":"#FFA15A","symbol":"circle"},"mode":"markers","name":"PHI","orientation":"v","showlegend":true,"x":[7.0],"xaxis":"x","y":[7.0],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=PIT\u003cbr\u003eParameter Rank=%{x}\u003cbr\u003eWin Rank=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"PIT","marker":{"color":"#19d3f3","symbol":"circle"},"mode":"markers","name":"PIT","orientation":"v","showlegend":true,"x":[6.0],"xaxis":"x","y":[7.0],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=SEA\u003cbr\u003eParameter Rank=%{x}\u003cbr\u003eWin Rank=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"SEA","marker":{"color":"#FF6692","symbol":"circle"},"mode":"markers","name":"SEA","orientation":"v","showlegend":true,"x":[9.0],"xaxis":"x","y":[9.0],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=SF\u003cbr\u003eParameter Rank=%{x}\u003cbr\u003eWin Rank=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"SF","marker":{"color":"#B6E880","symbol":"circle"},"mode":"markers","name":"SF","orientation":"v","showlegend":true,"x":[3.0],"xaxis":"x","y":[4.0],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=TB\u003cbr\u003eParameter Rank=%{x}\u003cbr\u003eWin Rank=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"TB","marker":{"color":"#FF97FF","symbol":"circle"},"mode":"markers","name":"TB","orientation":"v","showlegend":true,"x":[15.0],"xaxis":"x","y":[14.5],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=TEN\u003cbr\u003eParameter Rank=%{x}\u003cbr\u003eWin Rank=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"TEN","marker":{"color":"#FECB52","symbol":"circle"},"mode":"markers","name":"TEN","orientation":"v","showlegend":true,"x":[30.0],"xaxis":"x","y":[28.5],"yaxis":"y","type":"scatter"},{"hovertemplate":"team=WSH\u003cbr\u003eParameter Rank=%{x}\u003cbr\u003eWin Rank=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"WSH","marker":{"color":"#636efa","symbol":"circle"},"mode":"markers","name":"WSH","orientation":"v","showlegend":true,"x":[29.0],"xaxis":"x","y":[28.5],"yaxis":"y","type":"scatter"},{"hoverinfo":"skip","line":{"color":"red","dash":"dash","width":2},"mode":"lines","name":"Reference","x":[25.0,2.0,5.0,32.0,24.0,17.0,10.0,8.0,20.0,4.0,14.0,13.0,21.0,18.0,1.0,26.0,11.0,22.0,12.0,16.0,31.0,19.0,28.0,23.0,7.0,6.0,9.0,3.0,15.0,30.0,29.0],"y":[25.0,2.0,5.0,32.0,24.0,17.0,10.0,8.0,20.0,4.0,14.0,13.0,21.0,18.0,1.0,26.0,11.0,22.0,12.0,16.0,31.0,19.0,28.0,23.0,7.0,6.0,9.0,3.0,15.0,30.0,29.0],"type":"scatter"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"Parameter Rank"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"Win Rank"}},"legend":{"title":{"text":"team"},"tracegroupgap":0},"title":{"text":"Win Percentage Rank vs. Parameter Estimate Rank"}},                        {"responsive": true}                    )                };                            </script>        </div>
</body>
</html>


It seems the BT model is especially useful when there are ties in the standings and in some cases (i.e. `LAR`) where the parameter rank < win rank (indicating the team may be undervalued).


```python
# How often do the rankings agree?
(results_win_df["Parameter Rank"] == results_win_df["Win Rank"]).mean()
```




    0.12903225806451613



### Probability of team i beating team j
We can compute the probabilities using the equation:

$$p_{ij} = \frac{\exp{\alpha + \beta_i - \beta_j}}{1 + \exp{\alpha + \beta_i - \beta_j}}$$


```python
# Compute the probability of home team i defeating away team j
log_odds = (
    results.params.filter(items=["Intercept"]).sum()
    + results.params.filter(items=ALL_TEAMS).values
    - results.params.filter(items=ALL_TEAMS).values.reshape(-1, 1)
)
probability_matrix = pd.DataFrame(np.exp(log_odds) / (1 + np.exp(log_odds)))
probability_matrix.columns = results.params.filter(items=ALL_TEAMS).index
probability_matrix.index = results.params.filter(items=ALL_TEAMS).index
```


```python
render_plotly_html(
    px.imshow(
        probability_matrix,
        labels={"x": "Home Team", "y": "Away Team"},
        title="Probability of a Home Team Winning",
        aspect="auto",
        color_continuous_scale="RdBu_r",
    )
)
```




<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>                        <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>                <div id="7d87dbb0-b4ed-47fa-b79c-893643d15b77" class="plotly-graph-div" style="height:100%; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("7d87dbb0-b4ed-47fa-b79c-893643d15b77")) {                    Plotly.newPlot(                        "7d87dbb0-b4ed-47fa-b79c-893643d15b77",                        [{"coloraxis":"coloraxis","name":"0","x":["ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU","IND","JAX","KC","LAC","LAR","LV","MIA","MIN","NE","NO","NYG","NYJ","PHI","PIT","SEA","SF","TB","TEN","WSH"],"y":["ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU","IND","JAX","KC","LAC","LAR","LV","MIA","MIN","NE","NO","NYG","NYJ","PHI","PIT","SEA","SF","TB","TEN","WSH"],"z":[[0.5556519340572265,0.8688369233810522,0.8278308285590172,0.26784451885966565,0.5778727483845391,0.7039837774681854,0.7816417576941158,0.7898283710890741,0.6713996084439291,0.8552762341197682,0.7472582878073488,0.751144320338433,0.6642648446023038,0.682299798378201,0.8807096644508505,0.5510485457361414,0.7711759663209109,0.6433427192578762,0.7539183735954367,0.7125884244877702,0.4366362902770369,0.6795771841009645,0.5490719095403286,0.6306179196496867,0.7969160568282245,0.812238274284143,0.7884939410805692,0.8584899162305993,0.7303832423989504,0.4688263210487596,0.48106944344776553],[0.19098134029755454,0.5556519340572263,0.47580668897584083,0.0645996041847329,0.20535859209307777,0.309846057509682,0.4032550293541025,0.41501077280825793,0.27835058256502787,0.5273271138768175,0.3582110161320108,0.36297950177428423,0.2719360746235507,0.2884715010189098,0.5822429013493129,0.18812007492904806,0.38883381631903596,0.25402149697384924,0.36643084652908775,0.3188218506570868,0.12763814669686388,0.28590615681310544,0.186903309311202,0.24373476687538767,0.4255452105535947,0.44953240422277596,0.4130650112136952,0.5338540763532004,0.33835975136114477,0.14282318604604832,0.14894006412441738],[0.24540644221377592,0.6327230613169765,0.5556519340572263,0.08687649619631195,0.2625512088192301,0.38214437645220367,0.48212315301300757,0.4942737037489578,0.34699480527452725,0.6058259703724952,0.43468669883058453,0.4397756289030707,0.33974314260788063,0.35837216940521316,0.6575444108130591,0.2419736647377221,0.46708912858442614,0.3193207486732262,0.4434488616888067,0.39202488793052614,0.1677550039517941,0.355495781313637,0.24051176109661504,0.30747963843332027,0.5050829657506951,0.5294216257028659,0.4922690405922854,0.6120663988137766,0.4133267623589728,0.18669133765655038,0.1942612245867233],[0.810406554855168,0.9577035113209243,0.9426461717666358,0.5556519340572265,0.8239246429647213,0.8904612467113492,0.9244482785047977,0.9277755577692794,0.8747517359278084,0.9528317690043034,0.9099613379775016,0.9116415313045125,0.8711851520850196,0.8801109348589543,0.9618850307640203,0.8075281924843137,0.9201277042970678,0.8604487803921298,0.9128340987371927,0.8944579925057048,0.7259748042497902,0.8787823456068052,0.8062838119738396,0.8537085180360204,0.9306198505181431,0.936656251081823,0.9272362980639258,0.953995690968753,0.9025326998144739,0.7510580982279497,0.7601244080013503],[0.5332062782671214,0.8581734080375698,0.8145451331164897,0.25047149650135064,0.5556519340572265,0.6847802724718212,0.7658001666985311,0.7744092934103178,0.6511298111303563,0.8437085596095978,0.7297845687324835,0.7338435753940855,0.6437884381215683,0.6623642527627372,0.8708679375006174,0.5285676449998299,0.7548132683989882,0.6223157988538097,0.7367428931742576,0.6937002554902808,0.4145134325571511,0.6595560403091261,0.52657706580351,0.6092963985729194,0.7818733582806612,0.7980428933055708,0.7730051000573225,0.8471331982882478,0.7121921894450958,0.4463647471565914,0.4585276471391566],[0.39669066369106504,0.7769373213972928,0.7165727113785896,0.1613262044192675,0.4185422947475863,0.5556519340572263,0.6530453139568559,0.6639805990391238,0.5179210556996791,0.7565377390849256,0.608553425374646,0.6134689651539413,0.5098864987526776,0.530351091405586,0.7951675537063992,0.3922417060368197,0.6392605399055988,0.4867768587611713,0.6169951822949398,0.5659096141413809,0.28953735671575487,0.5272285938228101,0.3903394390833856,0.47304153023312107,0.6735582118060602,0.6946210190284712,0.6621888813281999,0.7613321325501411,0.5875292385161869,0.31698468422304293,0.3277089257196025],[0.30402755877917603,0.6982521256921786,0.626821666794119,0.11331564249760093,0.32351194958829266,0.45378614073258255,0.5556519340572263,0.5676244760909483,0.41648893963967754,0.6736789676876928,0.5080786019546314,0.5132466670224486,0.40869391448462516,0.42864917067372815,0.7206004724425333,0.30010089433957343,0.5407190410126694,0.3865533649164358,0.5169672834840059,0.4641275286087793,0.21306464533853536,0.4255828552774575,0.29842606589709064,0.3735874658748338,0.578203847497652,0.6017816959396601,0.5656550661242146,0.6794136211317533,0.4862140566054784,0.2356672696128334,0.2446257104513037],[0.2938364791880224,0.6879079242239209,0.615377595880679,0.10852048145129231,0.3129614408624051,0.4417614755070912,0.5436144228701675,0.5556519340572263,0.4047202932892935,0.662899063263098,0.495921471823824,0.5010918421668948,0.39699567356072063,0.4167823436104698,0.7107049787990137,0.28998659506184404,0.528620176458189,0.3750868595871251,0.5048157282981313,0.45205562487392253,0.20502444017467597,0.4137394453713042,0.28834495838702984,0.36227843572728763,0.5663004225380962,0.5900719194624394,0.5536708709112157,0.6687300040425844,0.4740759054795398,0.2270201915038622,0.23575131531211369],[0.43353216249717014,0.8021403779072096,0.7463696434873136,0.18293698832068456,0.45587971611989336,0.592751777876809,0.6865999430440994,0.6969683998099525,0.5556519340572265,0.7834031114710288,0.644065708626257,0.6487926681661276,0.5476970482068666,0.5679207247552245,0.8187912508899253,0.42896380208326773,0.6734817534874353,0.524708369857827,0.6521793431715971,0.602765285043519,0.3217342909375924,0.5648430616958717,0.42700856977266366,0.5109683390171896,0.7060220554719605,0.7258427972897505,0.6952718527297034,0.7878168371138248,0.6237696686232077,0.35072705838674384,0.36198785665751876],[0.20923714423864184,0.58361963920927,0.504316775127311,0.0718475168705599,0.22460716026063846,0.33476252363506387,0.4309921767907268,0.4429563058240339,0.3018420207318984,0.5556519340572263,0.3848480056001817,0.3897557402956906,0.29510753765521763,0.3124472777245389,0.6097125174466097,0.2061720731203801,0.41627097175881017,0.2762456601388503,0.39330449121834477,0.34410023533022316,0.1408934985497299,0.3097615416010386,0.20486800986704765,0.26537911538084913,0.45365000512715,0.47790346529623284,0.4409782897332312,0.5621125577230588,0.3643586300210017,0.1573710377118232,0.16399177446190288],[0.345930655482211,0.736955658023358,0.6703619020806965,0.13399419062922022,0.3666856271138568,0.5014580261712326,0.6022264263722403,0.6138162348746148,0.46356797365693025,0.7142444819916564,0.5556519340572265,0.5607522898254478,0.4555797653287112,0.47598151019895396,0.757433302562121,0.34172852548000265,0.5876973287800371,0.43275816382411186,0.5644179587662658,0.5118677273131356,0.24687783891643603,0.4728568261412015,0.33993421312285443,0.41930177567288185,0.6240140268846691,0.6465962884974376,0.6119133728196665,0.719563008355698,0.5339623532233122,0.27182843022895276,0.28165493234995875],[0.3412661549361971,0.7329268393209762,0.6657756969938765,0.13161238833172376,0.3618961136560413,0.49628763598225833,0.5972617747157047,0.6089022959054541,0.45842901726780677,0.7100046955603566,0.5505398243137485,0.5556519340572263,0.45045500208334344,0.4708257236632047,0.7536132631579362,0.3370914843133599,0.5826770000184781,0.42768844462925243,0.559326717653117,0.5066990932760952,0.24305264104035998,0.4677046769067655,0.3353091174645855,0.4142745670942633,0.6191493366719937,0.641856085900259,0.6069907238649076,0.7153706869134215,0.5288123121461052,0.2677540837686511,0.27748942707630714],[0.4414472509620062,0.807195569785418,0.752409928754333,0.18779365346924357,0.46386870261816693,0.600492265394114,0.6934790828681182,0.7037181605359998,0.5635783951077618,0.7888109892230145,0.6514044614275409,0.6560860011945516,0.5556519340572265,0.5757954123938472,0.8235146231804177,0.43685980733658714,0.6805148530246651,0.5327226220682315,0.6594392637987355,0.6104405301908324,0.3287929807592899,0.5727316408713478,0.43489599461868,0.5190018911674148,0.7126532954744244,0.732196531260168,0.7020432475571002,0.7931436133067156,0.631287326816528,0.35808601709983,0.36944864505287695],[0.4213355601953825,0.7941111467929444,0.7368197618492882,0.17560490025430425,0.4435466872709866,0.5806674589924601,0.6757771816017486,0.6863401244005068,0.543314910071187,0.7748269602796763,0.6325601645373333,0.637353888234971,0.5353238629149615,0.5556519340572265,0.8112787462639897,0.416801156660784,0.6624287046176293,0.5122663333183155,0.6407897962041317,0.5907729625831937,0.3109563743914563,0.5525557325289008,0.4148610862614761,0.4985103862562879,0.695572622596601,0.7158143077342771,0.6846110002046102,0.7793662283972844,0.6120032837748711,0.33946399537172933,0.3505584196449055],[0.17478318170854534,0.5287383478400515,0.44885388629961637,0.0583474861812146,0.18822472729028922,0.287144582919508,0.3774527079798422,0.3889465230166338,0.2570970635041134,0.5002409561194523,0.3336792441918276,0.33829330017208986,0.25100198246663685,0.26673082819875044,0.5556519340572263,0.1721129673949191,0.3633923266897024,0.23402321357885705,0.34163580328214815,0.29574455406645606,0.11604186312954705,0.26428700895579066,0.17097793233578099,0.22430281909562608,0.3992709197662674,0.42286729751274815,0.3870421115440834,0.5067921186858129,0.31452122821497586,0.13005297235024593,0.1357095138253924],[0.5602457891709793,0.8709449578259904,0.830469312023488,0.27151285490137567,0.582409523546984,0.7078504055080996,0.7848040807972387,0.7929035461852315,0.6754956659196388,0.8575664849706669,0.7507597812788139,0.754609656047577,0.6684059139336198,0.6863234626332979,0.8826526541143626,0.5556519340572265,0.7744461206145369,0.6476049469875182,0.7573575587004984,0.7163879373447868,0.44122322575828227,0.6836193410526843,0.5536791497760968,0.6349459267335061,0.7999137982861584,0.8150623301810869,0.7915834890208417,0.8607375914481179,0.7340353135829086,0.4734672172920325,0.485720710383852],[0.3169345282312539,0.7108024756089824,0.6408146902856997,0.11951661815140781,0.3368457479288303,0.4687686287410938,0.570484974755125,0.5823582505490723,0.4312121587034167,0.6867929187410423,0.5231368237275418,0.5282935636391083,0.4233414942086574,0.4434755542191299,0.7325773003246036,0.3129161138668942,0.5556519340572265,0.4009454858735071,0.532004039769569,0.4791519369596964,0.22334921616782638,0.4403849276762864,0.31120156577180486,0.3878020182336197,0.5928357114489035,0.6161386514860804,0.5804063422758278,0.6924023387671535,0.5012846126213014,0.24670083200226128,0.25593824912462676],[0.46435196745714113,0.8211775116911721,0.7692290118929588,0.20230423400390998,0.4869226097039875,0.6224529124504181,0.7127732002114601,0.722625559483596,0.5861697934665754,0.8038021026744253,0.6720947114946539,0.6766363216131889,0.5783466611100858,0.5982052665501455,0.8365527589581254,0.45972240823643984,0.7002723108638712,0.5556519340572265,0.679886976532989,0.632189281713938,0.34951011476513005,0.5951895579065883,0.4577393596796173,0.5420256731022822,0.7312082870944249,0.7499323701430777,0.7210151579208428,0.807902075085905,0.6525346933049267,0.37960502399373375,0.3912343107531986],[0.33792541208313814,0.7300008963454813,0.6624528890274134,0.12991922344434245,0.35846331505299245,0.4925640827260233,0.5936735242358313,0.6053492282443617,0.4547330836931235,0.7069280434296555,0.5468511309859432,0.5519710527379837,0.4467703219433222,0.46711604978066856,0.750836907138083,0.33377090170998774,0.5790503896638394,0.42404629315222925,0.5556519340572263,0.5029754257361997,0.24032257927454104,0.46399803179519195,0.3319972992471414,0.41066467972241927,0.6156305970531294,0.6384246220976242,0.6034316002772763,0.7123278981994973,0.5250991198355063,0.26484364323573806,0.2745128366362657],[0.38676736596716693,0.7696363834749581,0.7080384749007631,0.15577043443177496,0.40844243962387783,0.5453466004142133,0.6435497441230131,0.6546260301802038,0.5075162431066761,0.7487850040466624,0.5985886587076815,0.6035478109224969,0.4994758612827156,0.5199659040384772,0.7883007153394186,0.382359166664156,0.6296016840582618,0.4763794433534832,0.6071065706171523,0.5556519340572263,0.2810458534309332,0.5168372629750452,0.3804748232266358,0.462673011679582,0.6643354848510847,0.6857157019277301,0.6528105141323007,0.7536823611619672,0.5774009327902273,0.30803722361073466,0.3185999569984435],[0.6686072666935312,0.9144383443808201,0.8858151880295708,0.3711637692796098,0.6883476357706555,0.7932617637138654,0.8524085906462178,0.8584226011956295,0.7672559106926834,0.905077207988302,0.8266976965468604,0.8296407661956411,0.7614629882120162,0.7760370703616258,0.9225512480549298,0.6644674496440267,0.8446605429734723,0.7442660715810995,0.8317358142575095,0.8000085071622317,0.5556519340572265,0.773851505953949,0.6626844978366483,0.7336512773865106,0.8635965558239316,0.8746790275943348,0.8574450858283712,0.9073048818730388,0.8138050539285525,0.5874665725166072,0.5993122574666349],[0.4243938832076225,0.7961524923860872,0.739242618469382,0.1774264507229989,0.44664179331092985,0.583715686460122,0.6785168098157554,0.6890315695180902,0.5464225658003221,0.777005814133882,0.6354679834333082,0.6402453462978134,0.5384397010382361,0.5587438200288967,0.813189920322337,0.4198504465614172,0.6652252487465512,0.5153968203675339,0.6436691751073877,0.5937991185675693,0.3136477626614392,0.5556519340572265,0.41790635340549864,0.5016432706951484,0.6982196792322906,0.7183566250315803,0.6873105228618765,0.7815135442984261,0.6149747636014303,0.3422795560885421,0.3534167648633784],[0.5622124981264881,0.871839995586944,0.8315907818255804,0.2730954358283443,0.5843506494700923,0.7094992771085946,0.7861498449549988,0.7942119405572385,0.6772438822996634,0.8585392365235566,0.7522512379038323,0.7560855596942888,0.6701736756893059,0.6880402855249313,0.8834773584697891,0.5576229654378333,0.7758381584188734,0.6494254386415182,0.7588222172724424,0.7180078196644895,0.44319320218576685,0.6853441787976629,0.5556519340572265,0.6367951379981703,0.8011890083024116,0.8162631702637168,0.7928980449279969,0.8616921787344093,0.7355916028865118,0.47545865978991547,0.4877159524339715],[0.47806388487118817,0.8291165972285127,0.7788533223727331,0.21133102278057683,0.5006780392988266,0.6352963181473941,0.7239069236243478,0.733520158013464,0.5994532232978812,0.8123363053096493,0.6841064211605348,0.6885587570950489,0.5917062923230159,0.6113585125321161,0.843938964196494,0.4734184435492018,0.7116946033413404,0.5691950037053145,0.6917441612031017,0.6448904319369954,0.362123449914816,0.6083768965315908,0.4714278663303815,0.5556519340572265,0.7418861883220019,0.7601104753114212,0.7319495462093527,0.8162986953821202,0.6649056767999962,0.39264886066914484,0.4044172602477573],[0.28494522952004614,0.6785503993913795,0.6050938007143168,0.10440765988800788,0.30374040284415527,0.4311268784337616,0.5328683938546052,0.5449520826398252,0.3943474864498239,0.6531697506258688,0.4851160925327157,0.49028337985160225,0.38669223893487054,0.40631140032214164,0.701734616247102,0.28116545643646845,0.5178342392369047,0.3650078468529578,0.4940064768185706,0.44136914724824394,0.19806680567057275,0.4032920865544022,0.2795540951224329,0.352349540047988,0.5556519340572263,0.5795737753248567,0.5429625532521234,0.6590824332322092,0.4633090232561686,0.21952223692641448,0.22805006848258583],[0.2655044744415998,0.6569266139608311,0.5815757034523545,0.09563688862135708,0.28352577645629257,0.4073949499088243,0.5085425173705767,0.5206894519885668,0.3713186366500794,0.6307676693155992,0.4608196715271036,0.4659623693278543,0.363841875188634,0.3830257355585972,0.680937083581725,0.2618881329786359,0.49346966621783106,0.3427228105795611,0.4696709563589212,0.4174871831559657,0.1830361783711899,0.38006868348175576,0.2603472370906246,0.3304351733670312,0.5314693080362759,0.5556519340572263,0.5186875346248172,0.6368499251809014,0.43917320090674955,0.20327614431907312,0.21134375849154527],[0.2955033731700783,0.6896271676043191,0.6172740960566924,0.10929881740064254,0.3146884801367305,0.443740216047739,0.5456034893903831,0.5576312295662067,0.4066539811150509,0.6646889215912033,0.4979264222003036,0.5030968332524007,0.3989171870301322,0.4187331233525312,0.7123511513168981,0.29164066889338774,0.5306181666931695,0.37696862342185666,0.506820483077483,0.4540429718012714,0.20633472696257626,0.4156861338990967,0.2899934928872089,0.3641333842301115,0.5682691397039984,0.5920104708334512,0.5556519340572263,0.6705042912493887,0.47607594955883575,0.22843065807482066,0.23719938182313435],[0.20493416014822424,0.5772376798755523,0.4977653168859963,0.07011942510584221,0.22007605959647344,0.32895195938345834,0.4245773609399557,0.4365000213987386,0.29634831626435343,0.5491724391822593,0.3786629392771649,0.3835409769455152,0.2896855463551797,0.3068453858008283,0.6034587556022823,0.20191602735535422,0.409917416885442,0.2710369662926357,0.3870690336094299,0.338209994719913,0.13775117117722777,0.30418649874206194,0.20063208034333746,0.2603016231227968,0.447162975173252,0.47136883830930837,0.4345283462359317,0.5556519340572263,0.3583110519727797,0.15392705885593344,0.16043049065359052],[0.3659804052374163,0.7535581415934154,0.689396053528761,0.1444735953699905,0.38722614279174866,0.5233098644749286,0.6229821984074884,0.6343344529128773,0.48537620829644235,0.73175866268238,0.5771314948741291,0.5821706972884055,0.4773466074270065,0.4978317669758449,0.7731411728077562,0.36166939654515273,0.60871865380663,0.45434460024980294,0.5857896549452587,0.5336875868989803,0.26349942352641,0.49469905724393,0.3598276142621043,0.44073852035581573,0.6443044016897953,0.6663196556357202,0.6324721565141006,0.7368713026528043,0.5556519340572265,0.2894841677302348,0.299686240959893],[0.6392071132394531,0.9037065674999455,0.871995833213206,0.3413698438653663,0.6598077106579432,0.7711350821756116,0.8352979020551692,0.8418798005877985,0.7432467651044433,0.8933082806774358,0.8072801818274538,0.8104774083848518,0.737061053823466,0.7526417256704176,0.9127399493065972,0.6349005766162037,0.8268340805207027,0.7187546854739426,0.8127549352664487,0.778402197680791,0.523374369263713,0.7503013154397684,0.6330472778866525,0.7074971444708845,0.8475509482194966,0.8597253333277161,0.8408092475650522,0.8957803469297347,0.7933041674267072,0.5556519340572265,0.5677389716708752],[0.6278083942867658,0.899348483112671,0.8664147192615798,0.3304182678337021,0.6487021430716455,0.7623548229335236,0.8284315397244891,0.8352337089077989,0.7337659766656658,0.8885376312828783,0.7995261070016778,0.8028205532596892,0.7274356171652462,0.743388077254886,0.9087494194148286,0.6234459505466907,0.8196910325272215,0.7087239365963628,0.8051679745875044,0.769817515954592,0.5111150315916269,0.7409902430129196,0.6215691638403965,0.6972340189635278,0.8410982343215028,0.8536989747927648,0.8341270242056936,0.8911067600289418,0.785137434264541,0.5434986752885218,0.5556519340572265]],"type":"heatmap","xaxis":"x","yaxis":"y","hovertemplate":"Home Team: %{x}\u003cbr\u003eAway Team: %{y}\u003cbr\u003ecolor: %{z}\u003cextra\u003e\u003c\u002fextra\u003e"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"Home Team"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"autorange":"reversed","title":{"text":"Away Team"}},"coloraxis":{"colorscale":[[0.0,"rgb(5,48,97)"],[0.1,"rgb(33,102,172)"],[0.2,"rgb(67,147,195)"],[0.3,"rgb(146,197,222)"],[0.4,"rgb(209,229,240)"],[0.5,"rgb(247,247,247)"],[0.6,"rgb(253,219,199)"],[0.7,"rgb(244,165,130)"],[0.8,"rgb(214,96,77)"],[0.9,"rgb(178,24,43)"],[1.0,"rgb(103,0,31)"]]},"title":{"text":"Probability of a Home Team Winning"}},                        {"responsive": true}                    )                };                            </script>        </div>
</body>
</html>


To interpret, we can look at one column (i.e. `KC`) and see their chances of beating away teams (red scores are better). Conversely, following a row tells us how likely a team is to win on the road.

## Adding in Numeric Features
Since the BT model amounts to fitting a logisitic regression model, we can extend our previous attempt with numeric features.


```python
# Focus on x_per_y style metrics and turnovers.
numeric_features = [
    i + j
    for i in [
        "thirdDownEff",
        "fourthDownEff",
        "completionAttempts",
        "yardsPerRushAttempt",
        "yardsPerPass",
        "redZoneAttempts",
        "interceptions",
        "fumblesLost",
    ]
    for j in ["_home", "_away"]
]
```


```python
def compute_percentage(x: str, delim: str = "-") -> float:
    first, second = x.split(delim)
    return float(first) / float(second) if float(second) > 0 else 0
```


```python
# Construct a numeric features dataframe.
X_numeric = df[numeric_features].copy()
X_numeric["redZoneAttempts_home"] = X_numeric.redZoneAttempts_home.map(
    compute_percentage
)
X_numeric["thirdDownEff_home"] = X_numeric.thirdDownEff_home.map(compute_percentage)
X_numeric["redZoneAttempts_away"] = X_numeric.redZoneAttempts_away.map(
    compute_percentage
)
X_numeric["fourthDownEff_home"] = X_numeric.fourthDownEff_home.map(compute_percentage)
X_numeric["completionAttempts_away"] = X_numeric.completionAttempts_away.map(
    lambda x: compute_percentage(x, "/")
)
X_numeric["fourthDownEff_away"] = X_numeric.fourthDownEff_away.map(compute_percentage)
X_numeric["thirdDownEff_away"] = X_numeric.thirdDownEff_away.map(compute_percentage)
X_numeric["completionAttempts_home"] = X_numeric.completionAttempts_home.map(
    lambda x: compute_percentage(x, "/")
)
# After these conversions, we should only have numeric features
assert (X_numeric.dtypes != np.object_).all()
render_table(X_numeric.head())
```




|    |   thirdDownEff_home |   thirdDownEff_away |   fourthDownEff_home |   fourthDownEff_away |   completionAttempts_home |   completionAttempts_away |   yardsPerRushAttempt_home |   yardsPerRushAttempt_away |   yardsPerPass_home |   yardsPerPass_away |   redZoneAttempts_home |   redZoneAttempts_away |   interceptions_home |   interceptions_away |   fumblesLost_home |   fumblesLost_away |
|---:|--------------------:|--------------------:|---------------------:|---------------------:|--------------------------:|--------------------------:|---------------------------:|---------------------------:|--------------------:|--------------------:|-----------------------:|-----------------------:|---------------------:|---------------------:|-------------------:|-------------------:|
|  0 |            0.384615 |            0.384615 |                    1 |             0        |                  0.636364 |                  0.707317 |                        6.1 |                        4.4 |                 4.7 |                 4.7 |               0.333333 |               0.5      |                    1 |                    3 |                  0 |                  1 |
|  1 |            0.357143 |            0.333333 |                    0 |             0.333333 |                  0.538462 |                  0.628571 |                        3.9 |                        3.5 |                 5.8 |                 6.9 |               0.666667 |               0.666667 |                    1 |                    0 |                  0 |                  1 |
|  2 |            0.533333 |            0.388889 |                    0 |             0.25     |                  0.772727 |                  0.636364 |                        3.4 |                        3.1 |                 6   |                 4   |               0.6      |               0        |                    1 |                    0 |                  1 |                  1 |
|  3 |            0.285714 |            0.133333 |                    0 |             0        |                  0.551724 |                  0.4375   |                        5.2 |                        3.8 |                 4.5 |                 2   |               0.666667 |               0        |                    1 |                    0 |                  1 |                  0 |
|  4 |            0.428571 |            0.352941 |                    0 |             1        |                  0.75     |                  0.617647 |                        2.4 |                        2.2 |                 7.1 |                 4.8 |               0.333333 |               0.5      |                    1 |                    0 |                  2 |                  0 |



### Design Matrix
We expand the design matrix from before with the numeric features.


```python
formula_full = formula
formula_full += " + " + " + ".join(X_numeric.columns)
y, X_full = patsy.dmatrices(
    formula_full,
    pd.concat([y, X_diff, X_numeric], axis=1),
    return_type="dataframe",
)
Markdown(formula_full)
```




homeTeamWin ~ 1 + ATL + BAL + BUF + CAR + CHI + CIN + CLE + DAL + DEN + DET + GB + HOU + IND + JAX + KC + LAC + LAR + LV + MIA + MIN + NE + NO + NYG + NYJ + PHI + PIT + SEA + SF + TB + TEN + WSH + thirdDownEff_home + thirdDownEff_away + fourthDownEff_home + fourthDownEff_away + completionAttempts_home + completionAttempts_away + yardsPerRushAttempt_home + yardsPerRushAttempt_away + yardsPerPass_home + yardsPerPass_away + redZoneAttempts_home + redZoneAttempts_away + interceptions_home + interceptions_away + fumblesLost_home + fumblesLost_away



### Full Bradley Terry GLM


```python
# Fit a GLM with Binomial family and logit link function.
model = sm.GLM(
    endog=y,
    exog=X_full,
    family=sm.families.Binomial(
        link=sm.families.links.Logit(),
    ),
)
results_full = model.fit()
y_hat = results_full.predict(X_full)
```


```python
fpr, tpr, _ = roc_curve(y, y_hat)
fig_roc = px.line(
    pd.DataFrame({"True Positive Rate": tpr, "False Positive Rate": fpr}),
    x="False Positive Rate",
    y="True Positive Rate",
    title=f"Probability of Home Team Winning ROC Curve, AUC: {round(roc_auc_score(y, y_hat), 4)}",
)
fig_roc.add_scatter(
    x=fpr,
    y=fpr,
    mode="lines",
    name="Reference",
    line=dict(color="red", width=2, dash="dash"),
    hoverinfo="skip",
)
render_plotly_html(fig_roc)
```




<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>                        <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>                <div id="f99c3241-fd94-4fd0-a059-be447fb897bb" class="plotly-graph-div" style="height:100%; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("f99c3241-fd94-4fd0-a059-be447fb897bb")) {                    Plotly.newPlot(                        "f99c3241-fd94-4fd0-a059-be447fb897bb",                        [{"hovertemplate":"False Positive Rate=%{x}\u003cbr\u003eTrue Positive Rate=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"","line":{"color":"#636efa","dash":"solid"},"marker":{"symbol":"circle"},"mode":"lines","name":"","orientation":"v","showlegend":false,"x":[0.0,0.0,0.0,0.006756756756756757,0.006756756756756757,0.013513513513513514,0.013513513513513514,0.02702702702702703,0.02702702702702703,0.033783783783783786,0.033783783783783786,0.0472972972972973,0.0472972972972973,0.06756756756756757,0.06756756756756757,0.11486486486486487,0.11486486486486487,0.12837837837837837,0.12837837837837837,0.16216216216216217,0.16216216216216217,0.17567567567567569,0.17567567567567569,0.1891891891891892,0.1891891891891892,0.19594594594594594,0.19594594594594594,0.21621621621621623,0.21621621621621623,0.22297297297297297,0.22297297297297297,0.32432432432432434,0.32432432432432434,1.0],"xaxis":"x","y":[0.0,0.005434782608695652,0.6141304347826086,0.6141304347826086,0.7663043478260869,0.7663043478260869,0.9130434782608695,0.9130434782608695,0.9184782608695652,0.9184782608695652,0.9347826086956522,0.9347826086956522,0.9402173913043478,0.9402173913043478,0.9456521739130435,0.9456521739130435,0.9510869565217391,0.9510869565217391,0.9565217391304348,0.9565217391304348,0.9619565217391305,0.9619565217391305,0.967391304347826,0.967391304347826,0.9728260869565217,0.9728260869565217,0.9782608695652174,0.9782608695652174,0.9836956521739131,0.9836956521739131,0.9945652173913043,0.9945652173913043,1.0,1.0],"yaxis":"y","type":"scatter"},{"hoverinfo":"skip","line":{"color":"red","dash":"dash","width":2},"mode":"lines","name":"Reference","x":[0.0,0.0,0.0,0.006756756756756757,0.006756756756756757,0.013513513513513514,0.013513513513513514,0.02702702702702703,0.02702702702702703,0.033783783783783786,0.033783783783783786,0.0472972972972973,0.0472972972972973,0.06756756756756757,0.06756756756756757,0.11486486486486487,0.11486486486486487,0.12837837837837837,0.12837837837837837,0.16216216216216217,0.16216216216216217,0.17567567567567569,0.17567567567567569,0.1891891891891892,0.1891891891891892,0.19594594594594594,0.19594594594594594,0.21621621621621623,0.21621621621621623,0.22297297297297297,0.22297297297297297,0.32432432432432434,0.32432432432432434,1.0],"y":[0.0,0.0,0.0,0.006756756756756757,0.006756756756756757,0.013513513513513514,0.013513513513513514,0.02702702702702703,0.02702702702702703,0.033783783783783786,0.033783783783783786,0.0472972972972973,0.0472972972972973,0.06756756756756757,0.06756756756756757,0.11486486486486487,0.11486486486486487,0.12837837837837837,0.12837837837837837,0.16216216216216217,0.16216216216216217,0.17567567567567569,0.17567567567567569,0.1891891891891892,0.1891891891891892,0.19594594594594594,0.19594594594594594,0.21621621621621623,0.21621621621621623,0.22297297297297297,0.22297297297297297,0.32432432432432434,0.32432432432432434,1.0],"type":"scatter"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"False Positive Rate"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"True Positive Rate"}},"legend":{"tracegroupgap":0},"title":{"text":"Probability of Home Team Winning ROC Curve, AUC: 0.9851"}},                        {"responsive": true}                    )                };                            </script>        </div>
</body>
</html>



```python
render_table(
    pd.DataFrame(
        confusion_matrix(y_pred=y_hat > 0.5, y_true=y),
        columns=["Home Loss", "Home Win"],
        index=["Home Loss", "Home Win"],
    )
)
```




|           |   Home Loss |   Home Win |
|:----------|------------:|-----------:|
| Home Loss |         137 |         11 |
| Home Win  |          10 |        174 |



We see a large improvement in the confusion matrix and ROC AUC from adding the numerical features.

## Summary
The Bradley Terry model is a classic way to rank preferences applicable in spots statistics. More recently, it has been used in LLM training (check out the [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) paper for one example) to learn human preferences.
