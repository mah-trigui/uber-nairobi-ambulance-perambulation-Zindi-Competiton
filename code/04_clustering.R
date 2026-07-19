# =============================================================================
# 04_clustering.R - Geographic Clustering of Crash Locations
# Nairobi Ambulance Deployment Optimization
# =============================================================================
# Two clustering tasks:
# 1. Segment clusters (4 zones): group road segments by midpoint location
#    to create spatial neighborhoods for feature aggregation
# 2. Crash clusters (6 clusters): group historical crash locations into
#    6 zones matching the 6 ambulances — these become the initial
#    static deployment strategy (baseline)
# =============================================================================

source("00_config.R")
print_section("Step 4: Geographic Clustering")

set.seed(GLOBAL_SEED)

seg_tab <- readRDS(file.path(OUTPUT_DIR, "seg_tab_enriched.rds"))
train_org <- readRDS(file.path(OUTPUT_DIR, "train_with_segments.rds"))

# -----------------------------------------------------------------------------
# 1. SEGMENT CLUSTERING (4 ZONES)
# -----------------------------------------------------------------------------

print_step("Clustering road segments into 4 geographic zones...")

clus_seg <- seg_tab[, .(segment_id, long_mid, lat_mid, seg_length)]

library(kmodR)
seg_km <- kmod(clus_seg[, .(long_mid, lat_mid)],
    k = 4, l = 10,
    i_max = 1000, conv_method = "delta_C", conv_error = 0,
    allow_empty_c = FALSE
)

seg_tab$clus_seg <- seg_km$XC_dist_sqr_assign[, 2]

cat(sprintf(
    "  Segment zone sizes: %s\n",
    paste(table(seg_tab$clus_seg), collapse = " | ")
))

# -----------------------------------------------------------------------------
# 2. CRASH CLUSTERING (6 ZONES = 6 AMBULANCES)
# -----------------------------------------------------------------------------

print_step("Clustering crash locations into 6 zones (= 6 ambulances)...")

crash_locs <- unique(train_org[, .(latitude, longitude)])

# K-means++ initialization for better convergence
library(Rfast)
crash_km <- kmeans(crash_locs, kpp_init(crash_locs, NUM_AMBULANCES),
    iter.max = 1000, nstart = 20, algorithm = "Lloyd"
)

crash_locs$cluster <- crash_km$cluster

cat(sprintf(
    "  Crash cluster sizes: %s\n",
    paste(table(crash_locs$cluster), collapse = " | ")
))
cat("  Cluster centroids (initial ambulance positions):\n")
print(round(crash_km$centers, 5))

# -----------------------------------------------------------------------------
# 3. SAVE
# -----------------------------------------------------------------------------

saveRDS(seg_tab, file.path(OUTPUT_DIR, "seg_tab_clustered.rds"))
saveRDS(crash_locs, file.path(OUTPUT_DIR, "crash_clusters.rds"))
saveRDS(crash_km, file.path(OUTPUT_DIR, "crash_kmeans_model.rds"))

print_step("Clustering complete.")
