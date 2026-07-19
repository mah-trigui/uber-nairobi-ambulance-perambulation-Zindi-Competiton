# =============================================================================
# 09_ambulance_deployment.R - Convert Predictions to Ambulance Positions
# Nairobi Ambulance Deployment Optimization
# =============================================================================
# Key design decision:
# The model predicts P(crash) per segment per time window. But the submission
# requires 6 (latitude, longitude) pairs per 3-hour slot.
#
# Strategy: For each time window:
# 1. Score all segments with the ensemble
# 2. Select top-K segments by predicted crash probability
# 3. Cluster the predicted hotspots into 6 groups (K-means)
# 4. Place ambulances at cluster centroids
#
# This converts a "where will crashes happen" prediction into a
# "where should ambulances wait" deployment.
# =============================================================================

source("00_config.R")
print_section("Step 9: Ambulance Deployment")

set.seed(GLOBAL_SEED)

submit_abt <- readRDS(file.path(OUTPUT_DIR, "submit_abt.rds"))
seg_tab <- readRDS(file.path(OUTPUT_DIR, "seg_tab_clustered.rds"))
features <- readRDS(file.path(OUTPUT_DIR, "final_features.rds"))
model_lgb <- readRDS(file.path(OUTPUT_DIR, "model_lightgbm.rds"))
model_xgb <- readRDS(file.path(OUTPUT_DIR, "model_xgboost.rds"))
sample_sub <- readRDS(file.path(OUTPUT_DIR, "sample_submission.rds"))
crash_km <- readRDS(file.path(OUTPUT_DIR, "crash_kmeans_model.rds"))

# Segment coordinates lookup
seg_coords <- seg_tab[, .(segment_id, lat_mid, long_mid)]

# -----------------------------------------------------------------------------
# 1. PREDICT CRASH PROBABILITIES
# -----------------------------------------------------------------------------

print_step("Scoring all segments for test period...")

x_submit <- as.matrix(submit_abt[, features, with = FALSE])

pred_lgb <- predict(model_lgb, x_submit)
pred_xgb <- predict(model_xgb, xgb.DMatrix(data = x_submit))

# Ensemble average
submit_abt$pred <- (pred_lgb + pred_xgb) / 2

# -----------------------------------------------------------------------------
# 2. PLACE AMBULANCES PER TIME WINDOW
# -----------------------------------------------------------------------------

print_step("Placing ambulances per 3-hour window...")

# Time windows in submission
time_windows <- unique(submit_abt[, .(year, month, day, hour)])
setorder(time_windows, year, month, day, hour)

# Build submission frame
submission <- data.table()

for (i in seq_len(nrow(time_windows))) {
    tw <- time_windows[i]

    # Get predictions for this time window
    window_preds <- submit_abt[
        year == tw$year & month == tw$month &
            day == tw$day & hour == tw$hour,
        .(segment_id, pred)
    ]

    # Join coordinates
    window_preds <- merge(window_preds, seg_coords, by = "segment_id")

    # Select top predicted hotspots
    top_k <- min(50, nrow(window_preds))
    hotspots <- window_preds[order(-pred)][1:top_k]

    # Cluster into 6 ambulance positions (weighted by prediction probability)
    if (nrow(hotspots) >= NUM_AMBULANCES) {
        km <- kmeans(hotspots[, .(lat_mid, long_mid)],
            centers = NUM_AMBULANCES, nstart = 10, iter.max = 100
        )
        ambulance_pos <- as.data.table(km$centers)
        setnames(ambulance_pos, c("latitude", "longitude"))
    } else {
        # Fallback to static crash centroids
        ambulance_pos <- as.data.table(crash_km$centers)
        setnames(ambulance_pos, c("latitude", "longitude"))
    }

    # Format submission row
    date_str <- sprintf(
        "%d/%d/%d %d:00",
        tw$month, tw$day, tw$year, tw$hour
    )

    row <- data.table(Date = date_str)
    for (a in 1:NUM_AMBULANCES) {
        row[[paste0("A", a, "_Latitude")]] <- ambulance_pos$latitude[a]
        row[[paste0("A", a, "_Longitude")]] <- ambulance_pos$longitude[a]
    }

    submission <- rbind(submission, row)
}

cat(sprintf("  Submission rows: %d\n", nrow(submission)))

# -----------------------------------------------------------------------------
# 3. WRITE SUBMISSION
# -----------------------------------------------------------------------------

write.csv(submission, file.path(OUTPUT_DIR, "submission.csv"),
    row.names = FALSE, quote = FALSE
)

print_step("Ambulance deployment complete.")
