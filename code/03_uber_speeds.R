# =============================================================================
# 03_uber_speeds.R - Uber Movement Speed Data Integration
# Nairobi Ambulance Deployment Optimization
# =============================================================================
# Uber Movement provides hourly average speeds per OSM way.
# Process: load 18 monthly files → match OSM nodes to road segments →
# aggregate to segment-level speeds per 3-hour window.
# For the test period (Jul-Dec 2019): simulate speeds using year-over-year
# monthly trends (since actual Uber data for future is unavailable).
# =============================================================================

source("00_config.R")
print_section("Step 3: Uber Movement Speed Data")

seg_tab <- readRDS(file.path(OUTPUT_DIR, "seg_tab_enriched.rds"))

# -----------------------------------------------------------------------------
# 1. LOAD AND COMBINE UBER MONTHLY FILES
# -----------------------------------------------------------------------------

print_step("Loading Uber Movement speed files...")

uber_files <- list.files(DATA_DIR, pattern = "movement-speeds-hourly-nairobi", full.names = TRUE)
cat(sprintf("  Found %d Uber Movement files\n", length(uber_files)))

uber_list <- lapply(uber_files, function(f) {
    dt <- fread(f, select = c(
        "year", "month", "day", "hour",
        "osm_way_id", "osm_start_node_id", "osm_end_node_id",
        "speed_kph_mean"
    ))
    dt$osm_start_node_id <- as.numeric(dt$osm_start_node_id)
    dt$osm_end_node_id <- as.numeric(dt$osm_end_node_id)
    dt
})

uber <- rbindlist(uber_list)
rm(uber_list)
invisible(gc())

cat(sprintf("  Total Uber rows: %s\n", format(nrow(uber), big.mark = ",")))

# -----------------------------------------------------------------------------
# 2. MATCH OSM NODES TO ROAD SEGMENTS
# -----------------------------------------------------------------------------

print_step("Matching OSM nodes to road segments (pre-computed)...")

# This matching is computationally expensive (dist2Line for each node to each segment)
# If pre-computed seg_id_node.rds exists, load it. Otherwise compute.
node_match_file <- file.path(OUTPUT_DIR, "seg_id_node.rds")
if (file.exists(node_match_file)) {
    seg_id_node <- readRDS(node_match_file)
} else {
    cat("  [NOTE] OSM-to-segment matching must be pre-computed. Skipping.\n")
    seg_id_node <- NULL
}

# Join segment IDs to Uber data
if (!is.null(seg_id_node)) {
    uber <- merge(uber, seg_id_node,
        by = c("osm_start_node_id", "osm_end_node_id"), all.x = TRUE
    )
    uber <- uber[!is.na(segment_id)]
}

# -----------------------------------------------------------------------------
# 3. AGGREGATE TO SEGMENT-LEVEL PER 3-HOUR WINDOW
# -----------------------------------------------------------------------------

print_step("Aggregating speeds to segment × 3h windows...")

# Bin hours to 3-hour windows
uber[, hour_bin := fcase(
    hour %between% c(0, 2),   0L,
    hour %between% c(3, 5),   3L,
    hour %between% c(6, 8),   6L,
    hour %between% c(9, 11),  9L,
    hour %between% c(12, 14), 12L,
    hour %between% c(15, 17), 15L,
    hour %between% c(18, 20), 18L,
    hour %between% c(21, 23), 21L
)]

uber_by <- uber[, .(speed = mean(speed_kph_mean, na.rm = TRUE)),
    by = .(year, month, day, hour_bin, segment_id)
]
setnames(uber_by, "hour_bin", "hour")

# -----------------------------------------------------------------------------
# 4. SIMULATE FUTURE SPEEDS (JUL-DEC 2019)
# -----------------------------------------------------------------------------

print_step("Simulating speeds for test period using YoY trends...")

# Use monthly ratio: speed_2019_m = speed_2018_m * (trend from recent months)
uber_2018_h2 <- uber_by[year == 2018 & month %in% 7:12]
uber_2019_h1 <- uber_by[year == 2019 & month %in% 1:6]
uber_2018_h1 <- uber_by[year == 2018 & month %in% 1:6]

if (nrow(uber_2018_h2) > 0) {
    # Overall speed trend ratio
    trend_ratio <- mean(uber_2019_h1$speed, na.rm = TRUE) /
        mean(uber_2018_h1$speed, na.rm = TRUE)

    uber_future <- copy(uber_2018_h2)
    uber_future[, year := 2019L]
    uber_future[, speed := speed * trend_ratio]

    uber_by <- rbind(uber_by, uber_future)
}

cat(sprintf("  Uber segment-speed rows: %s\n", format(nrow(uber_by), big.mark = ",")))

# -----------------------------------------------------------------------------
# 5. SAVE
# -----------------------------------------------------------------------------

saveRDS(uber_by, file.path(OUTPUT_DIR, "uber_speeds.rds"))

print_step("Uber speed processing complete.")
