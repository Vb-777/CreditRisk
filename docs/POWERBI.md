# One-page Power BI dashboard: approval rate vs expected loss

The pipeline writes three Power BI–ready CSVs to `reports/powerbi/`. Building the page takes about 30 minutes in Power BI Desktop (free, Windows).

| File | Grain | Used for |
|---|---|---|
| `cutoff_strategy.csv` | one row per model × approval rate (5% steps) | the cut-off trade-off visuals |
| `scored_test_applications.csv` | one row per test-set application (61,503) | score-band and segment visuals |
| `portfolio_segments.csv` | one row per segment × view (full portfolio) | segment default-rate bars |

## 1. Load

Home → Get data → Text/CSV → load all three files. In Power Query, set `approval_rate`, `bad_rate`, `loss_rate`, `pd_*` and `default_rate` to Decimal Number, and `TARGET` / `has_bureau_history` to Whole Number.

## 2. Parameters (what-if)

Modeling → New parameter → Numeric range:

- **LGD**: 0.30 to 0.90, increment 0.05, default 0.60
- **Net margin**: 0.02 to 0.20, increment 0.01, default 0.08

## 3. Measures (DAX)

```DAX
Expected Loss = SUMX ( cutoff_strategy, cutoff_strategy[exposure_bad] * [LGD Value] )
Revenue       = SUMX ( cutoff_strategy, ( cutoff_strategy[exposure_approved] - cutoff_strategy[exposure_bad] ) * [Net margin Value] )
Profit        = [Revenue] - [Expected Loss]
Bad Rate      = DIVIDE ( SUM ( cutoff_strategy[approved_bads] ), SUM ( cutoff_strategy[approved] ) )

Default Rate (scored) = AVERAGE ( scored_test_applications[TARGET] )
Applications          = COUNTROWS ( scored_test_applications )

Best Approval Rate =
VAR t = ADDCOLUMNS ( VALUES ( cutoff_strategy[approval_rate] ), "p", [Profit] )
RETURN MAXX ( TOPN ( 1, t, [p], DESC ), cutoff_strategy[approval_rate] )
```

## 4. Page layout (16:9)

| Position | Visual | Fields |
|---|---|---|
| Top row | 4 cards | Best Approval Rate · Profit at best · Bad Rate at best · Applications |
| Top right | Slicers | `model` (Scorecard / Gradient boosting), LGD, Net margin |
| Middle left | Line chart | X `approval_rate`, Y `Profit`, legend `model` |
| Middle right | Line chart | X `approval_rate`, Y `Bad Rate`, legend `model` |
| Bottom left | Clustered bar | Y `score_band`, X `Default Rate (scored)` |
| Bottom right | Clustered bar | Y `segment`, X `default_rate` from `portfolio_segments` (filter `view`) |

Keep profit and bad rate on **separate** charts. They have different scales, so don't use a dual-axis chart. Use one colour per model (blue for the scorecard, orange for gradient boosting) on every visual so a model keeps its colour across the page.
