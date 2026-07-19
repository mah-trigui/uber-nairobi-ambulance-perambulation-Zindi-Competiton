# Nairobi Ambulance Deployment Optimization

This competition is hosted on Zindi, a machine learning platform for data science challenges.  
Here is the link to the competition: [Uber Nairobi Ambulance Perambulation Challenge 🌾 - $6 000](https://zindi.africa/competitions/uber-nairobi-ambulance-perambulation-challenge/data)

Ranked in the TOP 30%
---

**Competition**: Zindi – Ambulance Deployment Optimization
**Date**: 2019-2020
**Language**: R

## Task

Position 6 ambulances across Nairobi every 3 hours to minimize distance to reported traffic crashes during the test period (Jul-Dec 2019). Training data: 6,318 crashes from Jan 2018 to Jun 2019.

## Approach

A **two-stage system**: predict where crashes will happen, then place ambulances optimally.

1. **Stage 1 (Prediction)**: Binary classification at the (road segment × 3-hour window) level — will a crash occur here, now?
2. **Stage 2 (Deployment)**: Cluster predicted hotspots into 6 positions per time window.

## Engineering Decisions

### 1. Segment-Level ABT (Analytical Base Table)

Instead of predicting at individual lat/long points, the city is decomposed into ~800 road segments. Each segment becomes a row in the ABT, crossed with every 3-hour time window. This converts a spatial problem into a structured tabular classification.

### 2. Multi-Source Feature Fusion

Five data sources merged per segment per time window:
- **Crash history**: segment frequency, monthly elasticity, seasonal patterns
- **Road survey** (228 features): reduced to 2 dimensions via t-SNE
- **Uber Movement**: hourly speeds per OSM way, matched to segments
- **Weather**: temperature, humidity, wind, precipitation (simulated forward for test period)
- **Temporal**: hour bin, weekend, holidays

### 3. t-SNE for Obfuscated Road Survey Features

The 228 obfuscated road survey columns are reduced to 2 features via t-SNE (after NZV + correlation filtering). This captures the latent structure of road characteristics (crosswalks, obstacles, traffic behavior) without needing column semantics.

```r
tsne_out <- Rtsne(dr_final, dims = 2, perplexity = 25, max_iter = 7000)
```

### 4. Crash Elasticity as a Temporal Feature

Instead of using raw crash counts (which are always zero for the test period), compute month-over-month evolution per segment — the *trend*. Segments with increasing crash rates get higher predicted probabilities even without future crash data.

```r
elastic[, diff := (nb_crash_monthly - shift(nb_crash_monthly)) /
                  pmax(shift(nb_crash_monthly), 1), by = segment_id]
```

### 5. Weather Simulation for Test Period

Weather data only covers the training period. Future weather is simulated using year-over-year monthly adjustment: `weather_2019_m = weather_2018_m × (recent_trend_ratio)`. This preserves seasonal patterns without using future data.

### 6. Prediction → Deployment Conversion

The model outputs P(crash) per segment per window. Converting this to 6 ambulance positions:
- Take top-50 segments by predicted probability
- Cluster their midpoints into 6 groups (K-means)
- Place ambulances at cluster centroids

This ensures ambulances are distributed across predicted hotspots rather than concentrated at a single high-risk point.

### 7. Extreme Class Imbalance Handling

With ~0.1% positive rate (crash in a specific segment-window), stratified downsampling of negatives creates manageable training sets while preserving signal. Multiple subsets with different negative ratios feed into the ensemble.

## Pipeline

| Step | File | Purpose |
|------|------|---------|
| 0 | `00_config.R` | Configuration, libraries |
| 1 | `01_data_loading.R` | Load crashes, segments, weather |
| 2 | `02_segment_features.R` | Crash-to-segment matching, elasticity, t-SNE |
| 3 | `03_uber_speeds.R` | Uber Movement speed integration |
| 4 | `04_clustering.R` | Segment zones + crash clusters |
| 5 | `05_abt_construction.R` | Full segment × time grid |
| 6 | `06_feature_selection.R` | IV, RF, Elastic Net, random search |
| 7 | `07_model_lightgbm.R` | Primary LightGBM model |
| 8 | `08_model_ensemble.R` | XGBoost + CatBoost ensemble |
| 9 | `09_ambulance_deployment.R` | Predictions → ambulance positions |

## Run

```r
source("MAIN.R")
```

## Key Libraries

`data.table`, `rgdal`, `geosphere`, `lightgbm`, `xgboost`, `catboost`, `Rtsne`, `caret`, `text2vec`
