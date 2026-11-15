args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) stop("usage: Rscript --vanilla export_broll.R SOURCE_RDA OUTPUT_CSV DATASET")
source_rda <- args[[1L]]
output_csv <- args[[2L]]
dataset <- args[[3L]]
if (!nzchar(dataset)) stop("dataset must be nonempty")

loaded_environment <- new.env(parent = emptyenv())
loaded_names <- load(source_rda, envir = loaded_environment)
if (!identical(loaded_names, "example_data_5_subject")) stop("unexpected RDA object names")
source_data <- loaded_environment[["example_data_5_subject"]]
if (!identical(class(source_data), "data.frame")) stop("unexpected source object class")
if (!identical(names(source_data), c("id", "time", "gl"))) stop("unexpected source columns")
if (!identical(class(source_data$id), "factor")) stop("unexpected id class")
if (!identical(class(source_data$time), c("POSIXct", "POSIXt"))) stop("unexpected time class")
if (!identical(class(source_data$gl), "integer")) stop("unexpected gl class")
if (!identical(attr(source_data$time, "tzone"), "EST")) stop("unexpected time tzone attribute")
if (anyNA(source_data)) stop("unexpected missing source value")

source_id <- as.character(source_data$id)
if (any(!nzchar(source_id))) stop("empty source participant id")
time_seconds <- as.numeric(source_data$time)
if (any(!is.finite(time_seconds)) || any(time_seconds != floor(time_seconds))) {
  stop("unexpected nonintegral or nonfinite source time")
}
analysis_id <- paste0(nchar(dataset, type = "bytes"), ":", dataset,
                      nchar(source_id, type = "bytes"), ":", source_id)
output <- data.frame(
  source_row_number = seq_len(nrow(source_data)),
  source_participant_id = source_id,
  analysis_id = analysis_id,
  source_time_unix_seconds = sprintf("%.0f", time_seconds),
  source_time_fixed_EST = format(source_data$time, "%Y-%m-%d %H:%M:%S", tz = "EST", usetz = FALSE),
  source_gl = source_data$gl,
  stringsAsFactors = FALSE,
  check.names = FALSE
)
write.csv(output, output_csv, row.names = FALSE, na = "")
