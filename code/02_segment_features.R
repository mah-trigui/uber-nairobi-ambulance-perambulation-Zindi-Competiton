# =============================================================================
# 02_segment_features.R - Road Segment Processing
# Nairobi Ambulance Deployment Optimization
# =============================================================================
# Key steps:
# 1. Compute segment midpoints and lengths (Haversine)
# 2. Match each crash to nearest segment
# 3. Compute crash frequency and monthly elasticity per segment
# 4. Reduce 228 obfuscated segment survey features via t-SNE
# =============================================================================

source("00_config.R")
print_section("Step 2: Segment Feature Engineering")

set.seed(GLOBAL_SEED)

seg <- readRDS(file.path(OUTPUT_DIR, "segments_geo.rds"))
seg_tab <- readRDS(file.path(OUTPUT_DIR, "seg_tab.rds"))
seg_info <- readRDS(file.path(OUTPUT_DIR, "seg_info.rds"))
train_org <- readRDS(file.path(OUTPUT_DIR, "train_crashes.rds"))

# -----------------------------------------------------------------------------
# 1. SEGMENT GEOMETRY (MIDPOINTS, LENGTHS)
# -----------------------------------------------------------------------------

print_step("Computing segment lengths and midpoints...")

for (j in seq_len(length(seg))) {
    coords <- seg@lines[[j]]@Lines[[1]]@coords
    seg_tab$seg_length[j] <- distHaversine(coords[1, ], coords[nrow(coords), ])
    seg_tab$long_start[j] <- coords[1, 1]
    seg_tab$lat_start[j] <- coords[1, 2]
    seg_tab$long_end[j] <- coords[nrow(coords), 1]
    seg_tab$lat_end[j] <- coords[nrow(coords), 2]
}

# Midpoints
seg_tab$long_mid <- (seg_tab$long_start + seg_tab$long_end) / 2
seg_tab$lat_mid <- (seg_tab$lat_start + seg_tab$lat_end) / 2

# -----------------------------------------------------------------------------
# 2. MATCH CRASHES TO NEAREST SEGMENT
# -----------------------------------------------------------------------------

print_step("Matching crashes to nearest road segments...")

crashs <- unique(train_org[, .(latitude, longitude)])
crashs$segment_id <- NA_character_
crashs$dist_to_seg <- NA_real_

for (i in seq_len(nrow(crashs))) {
    point <- cbind(crashs$longitude[i], crashs$latitude[i])
    min_dist <- Inf
    nearest_seg <- NA_character_
    for (j in seq_len(length(seg))) {
        d <- dist2Line(point, seg@lines[[j]]@Lines[[1]]@coords, distfun = distGeo)[1]
        if (d < min_dist) {
            min_dist <- d
            nearest_seg <- seg_tab$segment_id[j]
        }
    }
    crashs$segment_id[i] <- nearest_seg
    crashs$dist_to_seg[i] <- min_dist
}

crashs <- merge(crashs, seg_tab[, .(segment_id, road_type)], by = "segment_id", all.x = TRUE)

# Crash count per segment
seg_crash <- crashs[, .(nb_crash = .N), by = segment_id]
seg_tab <- merge(seg_tab, seg_crash, by = "segment_id", all.x = TRUE)
seg_tab$nb_crash[is.na(seg_tab$nb_crash)] <- 0

# Join crash segments back to train
train_org <- merge(train_org, crashs[, .(latitude, longitude, segment_id, dist_to_seg)],
    by = c("latitude", "longitude"), all.x = TRUE
)

# -----------------------------------------------------------------------------
# 3. CRASH ELASTICITY (MONTHLY EVOLUTION PER SEGMENT)
# -----------------------------------------------------------------------------

print_step("Computing crash elasticity per segment...")

# Monthly crash counts per segment
elastic <- train_org[, .(nb_crash_monthly = .N), by = .(segment_id, year, month)]
setorder(elastic, segment_id, year, month)

# Month-over-month change
elastic[, diff := (nb_crash_monthly - shift(nb_crash_monthly)) /
    pmax(shift(nb_crash_monthly), 1),
by = segment_id
]
elastic$diff[is.na(elastic$diff)] <- 0

# Seasonal aggregates
evol_summer <- elastic[month %in% 7:9,
    .(med_summer = median(nb_crash_monthly), elast_summer = mean(diff)),
    by = segment_id
]
evol_winter <- elastic[month %in% 10:12,
    .(med_winter = median(nb_crash_monthly), elast_winter = mean(diff)),
    by = segment_id
]
evol_recent <- elastic[year == 2019 & month %in% 4:6,
    .(med_recent = median(nb_crash_monthly), elast_recent = mean(diff)),
    by = segment_id
]

evol <- merge(evol_summer, evol_winter, by = "segment_id", all = TRUE)
evol <- merge(evol, evol_recent, by = "segment_id", all = TRUE)

# -----------------------------------------------------------------------------
# 4. SEGMENT SURVEY DIMENSIONALITY REDUCTION (t-SNE)
# -----------------------------------------------------------------------------

print_step("Reducing segment survey features (228 → 2 via t-SNE)...")

# Take row with highest sum per segment (deduplicate sides)
seg_info$row_sum <- rowSums(seg_info[, 3:ncol(seg_info)], na.rm = TRUE)
seg_info_dedup <- seg_info[, .SD[which.max(row_sum)], by = segment_id]
seg_info_dedup[is.na(seg_info_dedup)] <- 0

# Remove near-zero-variance columns
dr <- seg_info_dedup[, -c("segment_id", "side", "row_sum")]
nzv_cols <- nearZeroVar(dr, names = TRUE)
dr_clean <- dr[, !nzv_cols, with = FALSE]

# Remove highly correlated columns
cor_mat <- cor(dr_clean)
high_cor <- findCorrelation(cor_mat, cutoff = 0.7, names = TRUE)
dr_final <- dr_clean[, !high_cor, with = FALSE]

cat(sprintf("  Survey features: 228 → %d (after NZV + correlation filter)\n", ncol(dr_final)))

# t-SNE embedding
tsne_out <- Rtsne(dr_final,
    dims = 2, perplexity = 25, max_iter = 7000,
    check_duplicates = FALSE, verbose = FALSE
)
df_tsne <- data.table(
    segment_id = seg_info_dedup$segment_id,
    tsne_x = tsne_out$Y[, 1],
    tsne_y = tsne_out$Y[, 2]
)

# -----------------------------------------------------------------------------
# 5. SAVE
# -----------------------------------------------------------------------------

saveRDS(seg_tab, file.path(OUTPUT_DIR, "seg_tab_enriched.rds"))
saveRDS(crashs, file.path(OUTPUT_DIR, "crashes_matched.rds"))
saveRDS(evol, file.path(OUTPUT_DIR, "segment_elasticity.rds"))
saveRDS(df_tsne, file.path(OUTPUT_DIR, "segment_tsne.rds"))
saveRDS(train_org, file.path(OUTPUT_DIR, "train_with_segments.rds"))

print_step("Segment features complete.")
