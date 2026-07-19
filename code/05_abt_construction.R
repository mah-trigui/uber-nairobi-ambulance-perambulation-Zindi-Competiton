# =============================================================================
# 05_abt_construction.R - Analytical Base Table (Segment × Time Window)
# Nairobi Ambulance Deployment Optimization
# =============================================================================
# Build the ABT: one row per (segment × 3-hour window) over the full period.
# Target: binary (1 = at least one crash in this segment during this window)
# Features: t-SNE embedding, segment characteristics, elasticity, weather,
#           Uber speeds, temporal indicators, geographic cluster.
# =============================================================================

source("00_config.R")
print_section("Step 5: ABT Construction")

seg_tab <- readRDS(file.path(OUTPUT_DIR, "seg_tab_clustered.rds"))
df_tsne <- readRDS(file.path(OUTPUT_DIR, "segment_tsne.rds"))
evol <- readRDS(file.path(OUTPUT_DIR, "segment_elasticity.rds"))
weather <- readRDS(file.path(OUTPUT_DIR, "weather.rds"))
uber_by <- readRDS(file.path(OUTPUT_DIR, "uber_speeds.rds"))
train_org <- readRDS(file.path(OUTPUT_DIR, "train_with_segments.rds"))

# -----------------------------------------------------------------------------
# 1. CREATE TIME GRID × SEGMENTS
# -----------------------------------------------------------------------------

print_step("Building full time × segment grid...")

# 3-hour intervals covering train + test period
dates_seq <- seq(
    from = as.POSIXct("2018-01-01 00:00:00", tz = "UTC"),
    to   = as.POSIXct("2019-12-31 21:00:00", tz = "UTC"),
    by   = "3 hours"
)

time_grid <- data.table(Date = dates_seq)
time_grid[, `:=`(
    year  = year(Date),
    month = month(Date),
    day   = mday(Date),
    hour  = hour(Date)
)]
time_grid[, week := as.integer(weekdays(as.Date(Date), abbr = TRUE) %in% c("Sat", "Sun"))]
time_grid[, special := as.integer(
    paste0(month, day) %chin% SP_DAYS_COMMON |
        paste0(year, month, day) %chin% SP_DAYS_YEAR
)]
time_grid[, id_date := paste0(year, month, day)]

# Cross join with segments
seg_ids <- seg_tab[, .(segment_id)]
abt <- CJ.dt <- time_grid[, .(Date, year, month, day, hour, week, special, id_date)]

# Use sqldf for cross join
abt <- as.data.table(sqldf(
    "SELECT a.*, b.segment_id
     FROM abt a CROSS JOIN seg_ids b"
))

cat(sprintf(
    "  ABT rows: %s (segments × time windows)\n",
    format(nrow(abt), big.mark = ",")
))

# -----------------------------------------------------------------------------
# 2. JOIN FEATURES
# -----------------------------------------------------------------------------

print_step("Joining features to ABT...")

# t-SNE (segment characteristics)
abt <- merge(abt, df_tsne, by = "segment_id", all.x = TRUE)

# Segment static features
abt <- merge(abt, seg_tab[, .(
    segment_id, nb_crash, nb_side, seg_length,
    road_type, clus_seg
)],
by = "segment_id", all.x = TRUE
)

# Crash elasticity
abt <- merge(abt, evol, by = "segment_id", all.x = TRUE)

# Weather (by date)
abt <- merge(abt, weather[, .(
    id_date, precipitable_water_entire_atmosphere,
    relative_humidity_2m_above_ground,
    temperature_2m_above_ground,
    u_component_of_wind_10m_above_ground,
    v_component_of_wind_10m_above_ground
)],
by = "id_date", all.x = TRUE
)

# Uber speeds (by segment × time)
abt <- merge(abt, uber_by,
    by = c("year", "month", "day", "hour", "segment_id"),
    all.x = TRUE
)

# -----------------------------------------------------------------------------
# 3. TARGET VARIABLE
# -----------------------------------------------------------------------------

print_step("Creating target variable...")

# Count crashes per segment × time window
crash_counts <- train_org[, .(nb_crash_window = .N),
    by = .(segment_id, year, month, day, hour)
]
abt <- merge(abt, crash_counts,
    by = c("segment_id", "year", "month", "day", "hour"),
    all.x = TRUE
)
abt$nb_crash_window[is.na(abt$nb_crash_window)] <- 0L
abt$target_bin <- as.integer(abt$nb_crash_window > 0)

# -----------------------------------------------------------------------------
# 4. IMPUTE MISSING VALUES
# -----------------------------------------------------------------------------

print_step("Imputing missing values...")

na_cols <- c(
    "med_summer", "elast_summer", "med_winter", "elast_winter",
    "med_recent", "elast_recent", "speed"
)
for (col in na_cols) {
    if (col %in% names(abt)) {
        abt[[col]][is.na(abt[[col]])] <- -9999
    }
}

# -----------------------------------------------------------------------------
# 5. DERIVED FEATURES
# -----------------------------------------------------------------------------

print_step("Creating derived features...")

# Hour groups
abt$h_morning <- as.integer(abt$hour == 6)
abt$h_daytime <- as.integer(abt$hour %in% c(9, 12, 15, 18))

# Road type encoding
abt$road_secondary <- as.integer(abt$road_type %in%
    c("secondary", "primary_link", "residential", "tertiary"))
abt$road_trunk <- as.integer(abt$road_type == "trunk")

# Segment cluster dummies
abt$clus_seg_1 <- as.integer(abt$clus_seg == 1)
abt$clus_seg_2 <- as.integer(abt$clus_seg == 2)
abt$clus_seg_4 <- as.integer(abt$clus_seg == 4)

# -----------------------------------------------------------------------------
# 6. SPLIT TRAIN / TEST
# -----------------------------------------------------------------------------

print_step("Splitting train / test periods...")

train_abt <- abt[!(year == 2019 & month > 6)]
submit_abt <- abt[year == 2019 & month > 6]

cat(sprintf(
    "  Train ABT: %s rows | Test ABT: %s rows\n",
    format(nrow(train_abt), big.mark = ","),
    format(nrow(submit_abt), big.mark = ",")
))
cat(sprintf(
    "  Positive rate (train): %.3f%%\n",
    100 * mean(train_abt$target_bin)
))

# -----------------------------------------------------------------------------
# 7. SAVE
# -----------------------------------------------------------------------------

saveRDS(train_abt, file.path(OUTPUT_DIR, "train_abt.rds"))
saveRDS(submit_abt, file.path(OUTPUT_DIR, "submit_abt.rds"))

print_step("ABT construction complete.")
