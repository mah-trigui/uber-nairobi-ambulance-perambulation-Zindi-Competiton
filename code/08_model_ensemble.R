# =============================================================================
# 08_model_ensemble.R - Multi-Model Ensemble
# Nairobi Ambulance Deployment Optimization
# =============================================================================
# Multiple models trained on different stratified subsets and feature sets:
# - LightGBM (primary)
# - XGBoost
# - CatBoost
# - Random Forest
# Predictions averaged to produce final crash probability per segment × window.
# =============================================================================

source("00_config.R")
print_section("Step 8: Model Ensemble")

set.seed(GLOBAL_SEED)

train_abt <- readRDS(file.path(OUTPUT_DIR, "train_abt.rds"))
features <- readRDS(file.path(OUTPUT_DIR, "final_features.rds"))

# Load primary model
model_lgb <- readRDS(file.path(OUTPUT_DIR, "model_lightgbm.rds"))

# -----------------------------------------------------------------------------
# 1. PREPARE STRATIFIED SUBSETS
# -----------------------------------------------------------------------------

print_step("Creating alternative stratified subsets...")

work <- train_abt[!(year == 2019 & month > 6)]
pos_rows <- work[target_bin == 1]

# Subset 2: different negative sampling ratio
set.seed(721)
neg_sample_2 <- work[target_bin == 0][sample(.N, nrow(pos_rows) * 3)]
work_sub2 <- rbind(pos_rows, neg_sample_2)[sample(.N)]

idx2 <- createDataPartition(work_sub2$target_bin, p = 0.7, list = FALSE)
train_2 <- work_sub2[idx2]
test_2 <- work_sub2[-idx2]

# -----------------------------------------------------------------------------
# 2. XGBOOST MODEL
# -----------------------------------------------------------------------------

print_step("Training XGBoost...")

x_tr2 <- as.matrix(train_2[, features, with = FALSE])
x_te2 <- as.matrix(test_2[, features, with = FALSE])

dtrain_xgb <- xgb.DMatrix(data = x_tr2, label = train_2$target_bin)
dtest_xgb <- xgb.DMatrix(data = x_te2, label = test_2$target_bin)

xgb_params <- list(
    objective = "binary:logistic",
    eval_metric = "auc",
    booster = "gbtree",
    eta = 0.01,
    max_depth = 6,
    subsample = 0.8,
    colsample_bytree = 0.7,
    nthread = NUM_THREADS
)

model_xgb <- xgb.train(
    params = xgb_params,
    data = dtrain_xgb,
    nrounds = 3000,
    watchlist = list(train = dtrain_xgb, test = dtest_xgb),
    early_stopping_rounds = 50,
    print_every_n = 500,
    verbose = 1
)

cat(sprintf("  XGBoost best iteration: %d\n", model_xgb$best_iteration))

# -----------------------------------------------------------------------------
# 3. CATBOOST MODEL (if available)
# -----------------------------------------------------------------------------

print_step("Training CatBoost (if available)...")

catboost_available <- requireNamespace("catboost", quietly = TRUE)
model_cat <- NULL

if (catboost_available) {
    library(catboost)

    pool_train <- catboost.load_pool(data = x_tr2, label = train_2$target_bin)
    pool_test <- catboost.load_pool(data = x_te2, label = test_2$target_bin)

    cat_params <- list(
        loss_function   = "Logloss",
        eval_metric     = "AUC",
        iterations      = 2000,
        depth           = 6,
        learning_rate   = 0.03,
        l2_leaf_reg     = 3,
        random_seed     = GLOBAL_SEED
    )

    model_cat <- catboost.train(pool_train, pool_test, params = cat_params)
    cat("  CatBoost trained.\n")
} else {
    cat("  CatBoost not available, skipping.\n")
}

# -----------------------------------------------------------------------------
# 4. SAVE MODELS
# -----------------------------------------------------------------------------

saveRDS(model_xgb, file.path(OUTPUT_DIR, "model_xgboost.rds"))
if (!is.null(model_cat)) saveRDS(model_cat, file.path(OUTPUT_DIR, "model_catboost.rds"))

print_step("Ensemble models trained.")
