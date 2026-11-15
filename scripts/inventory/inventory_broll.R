args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("usage: Rscript --vanilla inventory_broll.R SOURCE_RDA OUTPUT_DIR")
source_rda <- args[[1L]]
output_dir <- args[[2L]]
loaded_environment <- new.env(parent = emptyenv())
loaded_names <- load(source_rda, envir = loaded_environment)
if (!identical(loaded_names, "example_data_5_subject")) stop("unexpected RDA object names")
source_data <- loaded_environment[["example_data_5_subject"]]
if (!is.data.frame(source_data)) stop("RDA object is not a data.frame")
if (!identical(names(source_data), c("id", "time", "gl"))) stop("unexpected source columns")
if (!inherits(source_data$time, "POSIXct")) stop("time is not POSIXct")

source_tzone <- attr(source_data$time, "tzone")
if (length(source_tzone) != 1L || !identical(source_tzone, "EST")) stop("source timezone is not fixed EST")
timestamp_text <- function(x) format(x, "%Y-%m-%d %H:%M:%S", tz = "EST", usetz = FALSE)
source_clock_dates <- format(source_data$time, "%Y-%m-%d", tz = "EST")
participant_ids <- unique(as.character(source_data$id))
participant_rows <- lapply(participant_ids, function(participant_id) {
  rows <- source_data[as.character(source_data$id) == participant_id, , drop = FALSE]
  ordered_time <- sort(rows$time)
  gaps_minutes <- as.numeric(diff(ordered_time), units = "mins")
  data.frame(
    participant_id = participant_id,
    row_count = nrow(rows),
    source_start = timestamp_text(min(rows$time)),
    source_end = timestamp_text(max(rows$time)),
    median_gap_minutes = if (length(gaps_minutes)) median(gaps_minutes) else NA_real_,
    maximum_gap_minutes = if (length(gaps_minutes)) max(gaps_minutes) else NA_real_,
    gaps_over_120_minutes = sum(gaps_minutes > 120),
    nonpositive_gaps = sum(gaps_minutes <= 0),
    stringsAsFactors = FALSE
  )
})
participant_inventory <- do.call(rbind, participant_rows)
all_gaps <- unlist(lapply(participant_ids, function(participant_id) {
  times <- sort(source_data$time[as.character(source_data$id) == participant_id])
  as.numeric(diff(times), units = "mins")
}))
inventory <- data.frame(
  object_name = "example_data_5_subject",
  row_count = nrow(source_data),
  column_count = ncol(source_data),
  column_names = paste(names(source_data), collapse = ";"),
  object_class = paste(class(source_data), collapse = ";"),
  id_class = paste(class(source_data$id), collapse = ";"),
  time_class = paste(class(source_data$time), collapse = ";"),
  gl_class = paste(class(source_data$gl), collapse = ";"),
  time_tzone_attribute = source_tzone,
  missing_id = sum(is.na(source_data$id)),
  missing_time = sum(is.na(source_data$time)),
  missing_gl = sum(is.na(source_data$gl)),
  duplicated_id_time = sum(duplicated(source_data[c("id", "time")])),
  participant_count = length(participant_ids),
  source_time_min = timestamp_text(min(source_data$time)),
  source_time_max = timestamp_text(max(source_data$time)),
  glucose_min = min(source_data$gl),
  glucose_max = max(source_data$gl),
  median_gap_minutes = median(all_gaps),
  source_clock_date_2015_03_08_readings = sum(source_clock_dates == "2015-03-08"),
  stringsAsFactors = FALSE
)
write.csv(inventory, file.path(output_dir, "source_inventory.csv"), row.names = FALSE, na = "")
write.csv(participant_inventory, file.path(output_dir, "participant_inventory.csv"), row.names = FALSE, na = "")
