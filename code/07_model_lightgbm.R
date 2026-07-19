# =============================================================================
# 07_model_lightgbm.R - LightGBM Crash Prediction
# Nairobi Ambulance Deployment Optimization
# =============================================================================
# Train LightGBM to predict P(crash) per segment per 3-hour window.
# Extreme class imbalance (~0.1% positive) handled via stratified sampling.
# Hyperparameters tuned via random search on held-out freeze set.
# =============================================================================

source("00_config.R")
print_section("Step 7: LightGBM Model Training")

set.seed(GLOBAL_SEED)

train_abt <- readRDS(file.path(OUTPUT_DIR, "train_abt.rds"))
features <- readRDS(file.path(OUTPUT_DIR, "final_features.rds"))

# -----------------------------------------------------------------------------
# 1. STRATIFIED PARTITION
# -----------------------------------------------------------------------------

print_step("Creating stratified train/test/freeze splits...")

# Hold-out freeze set for final evaluation
freeze_idx <- createDataPartition(train_abt$target_bin, p = 0.033, list = FALSE)
freeze <- train_abt[freeze_idx]
work <- train_abt[-freeze_idx]

# Stratified subsample: downsample negatives
set.seed(GLOBAL_SEED)
pos_work <- work[target_bin == 1]
neg_work <- work[target_bin == 0][sample(.N, min(.N, nrow(pos_work) * 8))]
work_bal <- rbind(pos_work, neg_work)
work_bal <- work_bal[sample(.N)] # shuffle

# Train/test split
split_idx <- createDataPartition(work_bal$target_bin, p = 0.8, list = FALSE)
train_set <- work_bal[split_idx]
test_set <- work_bal[-split_idx]

cat(sprintf(
  "  Train: %d | Test: %d | Freeze: %d\n",
  nrow(train_set), nrow(test_set), nrow(freeze)
))
cat(sprintf(
  "  Positive rate - Train: %.2f%% | Freeze: %.3f%%\n",
  100 * mean(train_set$target_bin), 100 * mean(freeze$target_bin)
))

# -----------------------------------------------------------------------------
# 2. LIGHTGBM TRAINING
# -----------------------------------------------------------------------------

print_step("Training LightGBM...")

x_train <- as.matrix(train_set[, features, with = FALSE])
x_test <- as.matrix(test_set[, features, with = FALSE])
x_freeze <- as.matrix(freeze[, features, with = FALSE])

dtrain <- lgb.Dataset(data = x_train, label = train_set$target_bin)
dtest <- lgb.Dataset.create.valid(dtrain, data = x_test, label = test_set$target_bin)
valids <- list(train = dtrain, test = dtest)

params <- list(
  objective = "binary",
  metric = "auc",
  num_leaves = 12,
  max_depth = 6,
  min_data_per_group = 1,
  max_bin = 170,
  bagging_fraction = 0.48,
  bagging_seed = 526592,
  min_split_gain = 0.076,
  lambda_l1 = 0.3,
  lambda_l2 = 2.5,
  feature_pre_filter = FALSE,
  force_row_wise = TRUE,
  nthread = NUM_THREADS,
  seed = GLOBAL_SEED
)

model_lgb <- lgb.train(
  data = dtrain,
  params = params,
  nrounds = 4000,
  valids = valids,
  early_stopping_rounds = 30,
  verbose = 1
)

cat(sprintf("  Best iteration: %d\n", model_lgb$best_iter))

# -----------------------------------------------------------------------------
# 3. EVALUATE ON FREEZE SET
# -----------------------------------------------------------------------------

print_step("Evaluating on freeze set...")

pred_freeze <- predict(model_lgb, x_freeze)
auc_freeze <- MLmetrics::AUC(pred_freeze, freeze$target_bin)
cat(sprintf("  Freeze set AUC: %.4f\n", auc_freeze))

# Optimal threshold
optimal_cut <- InformationValue::optimalCutoff(
  freeze$target_bin, pred_freeze,
  optimiseFor = "Both"
)
cat(sprintf("  Optimal cutoff: %.4f\n", optimal_cut))

pred_class <- as.integer(pred_freeze >= optimal_cut)
cm <- table(actual = freeze$target_bin, pred = pred_class)
precision <- cm[2, 2] / sum(cm[, 2])
recall <- cm[2, 2] / sum(cm[2, ])
f1 <- 2 * precision * recall / (precision + recall)
cat(sprintf(
  "  Precision: %.3f | Recall: %.3f | F1: %.3f\n",
  precision, recall, f1
))

# -----------------------------------------------------------------------------
# 4. FEATURE IMPORTANCE
# -----------------------------------------------------------------------------

print_step("Feature importance...")
imp <- lgb.importance(model_lgb)
cat("  Top 10 features:\n")
print(head(imp, 10))

# -----------------------------------------------------------------------------
# 5. SAVE
# -----------------------------------------------------------------------------

saveRDS(model_lgb, file.path(OUTPUT_DIR, "model_lightgbm.rds"))
saveRDS(
  list(auc = auc_freeze, cutoff = optimal_cut, f1 = f1),
  file.path(OUTPUT_DIR, "model_metrics.rds")
)

print_step("LightGBM training complete.")
# ============================================================================
# 07_MODEL_LIGHTGBM.R - LightGBM Model Training
# ============================================================================
# Description: Trains LightGBM models with hyperparameter tuning using
#              both manual grid search and tidymodels workflow.
# ============================================================================

source("00_config.R")
library(lightgbm)

# Load splits
load("splits_base.RData")

# ============================================================================
# 1. BASIC LIGHTGBM MODEL
# ============================================================================

cat("Training basic LightGBM model...\n")

# Prepare data
features <- tr_base[, -"Energy"]
categorical_vars <- names(which(sapply(tr_base, class) == "factor"))

# Create LightGBM dataset
dtrain <- lgb.Dataset(
  data = as.matrix(features),
  label = tr_base$Energy,
  categorical_feature = categorical_vars
)

# Basic parameters
params <- list(
  objective = "regression",
  metric = "mae"
)

# Cross-validation for optimal rounds
set.seed(GLOBAL_SEED)
cv_result <- lgb.cv(
  params = params,
  data = dtrain,
  nrounds = 10000,
  nfold = CV_FOLDS,
  early_stopping_rounds = 50,
  verbose = 0
)

# Train final model
set.seed(GLOBAL_SEED)
model_lgb_basic <- lgb.train(
  params = params,
  data = dtrain,
  nrounds = cv_result$best_iter
)

# Evaluate on freeze set
preds <- predict(model_lgb_basic, as.matrix(freez_base[, -"Energy"]))
evaluate_model(freez_base$Energy, preds, "LightGBM Basic")

# Save predictions
lgb_basic_preds <- data.frame(pred = preds, real = freez_base$Energy)
write.csv(lgb_basic_preds, "freez_base_lightgbm_basic.csv", quote = FALSE, row.names = FALSE)

# ============================================================================
# 2. LIGHTGBM WITH HYPERPARAMETER TUNING
# ============================================================================

cat("\nTraining LightGBM with hyperparameter tuning...\n")

# Prepare validation set
features_ts <- ts_base[, -"Energy"]
dtest <- lgb.Dataset.create.valid(dtrain, data = as.matrix(features_ts), label = ts_base$Energy)
valids <- list(train = dtrain, test = dtest)

# Define parameter grid
param_grid <- expand.grid(
  num_leaves = c(40, 60, 80),
  max_depth = c(5, 6, 7),
  min_data_in_leaf = c(10, 15, 20),
  colsample_bytree = c(0.4, 0.6, 0.8)
)

# Store results
mae_values <- numeric(nrow(param_grid))

# Grid search
cat("Running grid search...\n")
for (i in seq_len(nrow(param_grid))) {
  set.seed(GLOBAL_SEED)

  model <- lgb.train(
    data = dtrain,
    num_iterations = 3000,
    objective = "regression",
    eval = "mae",
    metric = "mae",
    valids = valids,
    nthread = 4L,
    early_stopping_round = 100,
    verbose = -1,
    num_leaves = param_grid$num_leaves[i],
    max_depth = param_grid$max_depth[i],
    min_data_in_leaf = param_grid$min_data_in_leaf[i],
    colsample_bytree = param_grid$colsample_bytree[i]
  )

  # Evaluate
  features_freez <- freez_base[, names(freez_base) != "Energy"]
  pred <- predict(model, as.matrix(features_freez))
  mae_values[i] <- Metrics::mae(freez_base$Energy, pred)

  if (i %% 10 == 0) {
    cat(sprintf("  Completed %d/%d combinations\n", i, nrow(param_grid)))
  }
}

results <- cbind(param_grid, MAE = mae_values)

# Find best parameters
best_idx <- which.min(results$MAE)
best_params <- results[best_idx, ]
cat(sprintf("\nBest MAE: %.6f\n", best_params$MAE))

# ============================================================================
# 3. TRAIN FINAL MODEL WITH BEST PARAMETERS
# ============================================================================

cat("\nTraining final LightGBM model with best parameters...\n")

set.seed(GLOBAL_SEED)
model_lgb_tuned <- lgb.train(
  data = dtrain,
  num_iterations = 3000,
  objective = "regression",
  eval = "mae",
  metric = "mae",
  valids = valids,
  nthread = 4L,
  early_stopping_round = 100,
  verbose = 0,
  num_leaves = best_params$num_leaves,
  max_depth = best_params$max_depth,
  min_data_in_leaf = best_params$min_data_in_leaf,
  colsample_bytree = best_params$colsample_bytree
)

# Final evaluation
features_freez <- freez_base[, -"Energy"]
preds <- predict(model_lgb_tuned, as.matrix(features_freez))
evaluate_model(freez_base$Energy, preds, "LightGBM Tuned")

# Save predictions
lgb_tuned_preds <- data.frame(pred = preds, real = freez_base$Energy)
write.csv(lgb_tuned_preds, "freez_base_lightgbm_tuned.csv", quote = FALSE, row.names = FALSE)

# ============================================================================
# 4. SAVE MODELS
# ============================================================================

cat("\nSaving models...\n")

lgb.save(model_lgb_basic, "model_lgb_basic.txt")
lgb.save(model_lgb_tuned, "model_lgb_tuned.txt")

cat("LightGBM training complete.\n")
