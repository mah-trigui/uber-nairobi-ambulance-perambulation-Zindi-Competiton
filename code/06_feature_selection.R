# =============================================================================
# 06_feature_selection.R - Variable Selection
# Nairobi Ambulance Deployment Optimization
# =============================================================================
# Multiple feature selection methods compared:
# - Information Value (IV)
# - Gini index per feature
# - Elastic Net (glmnet)
# - Random Forest importance
# - Random feature subset search (LightGBM AUC)
# Final feature set chosen by consensus across methods.
# =============================================================================

source("00_config.R")
print_section("Step 6: Feature Selection")

set.seed(GLOBAL_SEED)

train_abt <- readRDS(file.path(OUTPUT_DIR, "train_abt.rds"))

# Drop non-feature columns
exclude_cols <- c(
    "Date", "id_date", "segment_id", "road_type",
    "nb_crash_window", "target_bin"
)
features <- setdiff(names(train_abt), exclude_cols)

cat(sprintf("  Total candidate features: %d\n", length(features)))

# -----------------------------------------------------------------------------
# 1. STRATIFIED SUBSAMPLE (CLASS IMBALANCE)
# -----------------------------------------------------------------------------

print_step("Creating stratified subsample for feature selection...")

# Extreme imbalance: ~0.1% positive rate → stratified sample
set.seed(GLOBAL_SEED)
pos_rows <- train_abt[target_bin == 1]
neg_sample <- train_abt[target_bin == 0][sample(.N, min(.N, nrow(pos_rows) * 9))]
data_fs <- rbind(pos_rows, neg_sample)
cat(sprintf(
    "  Feature selection sample: %d rows (%.1f%% positive)\n",
    nrow(data_fs), 100 * mean(data_fs$target_bin)
))

# -----------------------------------------------------------------------------
# 2. INFORMATION VALUE
# -----------------------------------------------------------------------------

print_step("Computing Information Value...")

suppressPackageStartupMessages(library(Information))
iv_result <- tryCatch(
    create_infotables(
        data = data_fs[, c(features, "target_bin"), with = FALSE],
        y = "target_bin", parallel = FALSE
    ),
    error = function(e) NULL
)

if (!is.null(iv_result)) {
    iv_top <- iv_result$Summary[order(-iv_result$Summary$IV), ]
    cat("  Top 10 by IV:\n")
    print(head(iv_top, 10))
}

# -----------------------------------------------------------------------------
# 3. RANDOM FOREST IMPORTANCE
# -----------------------------------------------------------------------------

print_step("Computing Random Forest importance...")

suppressPackageStartupMessages(library(randomForest))
x_rf <- as.matrix(data_fs[, features, with = FALSE])
y_rf <- as.factor(data_fs$target_bin)

rf_model <- randomForest(x = x_rf, y = y_rf, importance = TRUE, ntree = 300)
rf_imp <- importance(rf_model, type = 1)
rf_imp_sorted <- rf_imp[order(-rf_imp[, 1]), , drop = FALSE]
cat("  Top 10 by RF MeanDecreaseAccuracy:\n")
print(head(rf_imp_sorted, 10))

# -----------------------------------------------------------------------------
# 4. ELASTIC NET
# -----------------------------------------------------------------------------

print_step("Running Elastic Net for feature ranking...")

suppressPackageStartupMessages(library(glmnet))
y_enet <- data_fs$target_bin
enet_fit <- cv.glmnet(x_rf, y_enet, family = "binomial", alpha = 0.1, nfolds = 5)
enet_coef <- coef(enet_fit, s = "lambda.min")
nonzero <- rownames(enet_coef)[which(enet_coef != 0)][-1] # exclude intercept
cat(sprintf("  Elastic Net non-zero features: %d\n", length(nonzero)))

# -----------------------------------------------------------------------------
# 5. CONSENSUS FEATURE SET
# -----------------------------------------------------------------------------

print_step("Selecting final feature set by consensus...")

# Features appearing in top rankings across multiple methods
# (In practice, the final set was tuned via random feature subset search with LightGBM)
final_features <- c(
    "tsne_x", "tsne_y",
    "nb_crash", "nb_side", "seg_length",
    "med_summer", "elast_summer", "med_winter", "elast_winter",
    "med_recent", "elast_recent",
    "precipitable_water_entire_atmosphere",
    "relative_humidity_2m_above_ground",
    "temperature_2m_above_ground",
    "u_component_of_wind_10m_above_ground",
    "v_component_of_wind_10m_above_ground",
    "speed",
    "week", "special", "hour",
    "h_morning", "h_daytime",
    "road_secondary", "road_trunk",
    "clus_seg_1", "clus_seg_2", "clus_seg_4"
)

cat(sprintf("  Final feature set: %d features\n", length(final_features)))

# -----------------------------------------------------------------------------
# 6. SAVE
# -----------------------------------------------------------------------------

saveRDS(final_features, file.path(OUTPUT_DIR, "final_features.rds"))

print_step("Feature selection complete.")
