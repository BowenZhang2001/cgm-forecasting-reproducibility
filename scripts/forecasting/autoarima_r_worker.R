# Persistent diagnostic worker. Installation stays immutable; observer wrappers
# exist only in this R process and do not change candidate acceptance/search.
# Protocol: JSONL metadata + base64 little-endian doubles; no target input.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) == 2L)
.libPaths(c(normalizePath(args[1]), .libPaths()))
cfg <- jsonlite::fromJSON(args[2], simplifyVector=FALSE)
stopifnot(identical(cfg$use_test_partitions, FALSE) ||
            (identical(cfg$use_test_partitions, TRUE) &&
             identical(cfg$status, "configured_for_test_evaluation")),
          identical(as.character(packageVersion("forecast")), "8.24.0"),
          identical(as.character(getRversion()), "4.1.2"))
load_start <- proc.time()[3]
suppressPackageStartupMessages(library(forecast))
load_seconds <- unname(proc.time()[3] - load_start)
input <- file("stdin", open="r")
output <- stdout()
emit <- function(x) {
  cat(jsonlite::toJSON(x, auto_unbox=TRUE, digits=17, null="null", na="null"),
      "\n", sep="", file=output)
  flush(output)
}
pack <- function(x) gsub("[[:space:]]", "", jsonlite::base64_enc(
  writeBin(as.double(x), raw(), size=8, endian="little")))
unpack <- function(x, n) {
  bytes <- jsonlite::base64_dec(x)
  stopifnot(length(bytes) == n * 8L)
  readBin(bytes, what="double", n=n, size=8, endian="little")
}
audit <- new.env(parent=emptyenv())
audit$active <- NA_integer_
audit$calls <- list()
audit$warnings <- list()
audit$branches <- list()
audit$unit_root_checks <- list()
audit$unit_root_test_failed <- FALSE
audit$enabled <- TRUE
wrap <- function(original, kind) {
  if (kind == "ndiffs") {
    # Shadow only this function's imported ur.kpss; return the original S4
    # object unchanged. Native errors still follow ndiffs' original catch path.
    kpss_original <- get("ur.kpss", envir=environment(original), inherits=TRUE)
    observed_kpss <- local({
      original_kpss <- kpss_original
      function(...) {
        if (!audit$enabled) return(original_kpss(...))
        ans <- tryCatch(original_kpss(...), error=function(e) {
          audit$unit_root_test_failed <- TRUE
          stop(e)
        })
        finite <- all(is.finite(ans@teststat)) && all(is.finite(ans@cval))
        audit$unit_root_checks[[length(audit$unit_root_checks)+1L]] <- list(
          statistic=as.numeric(ans@teststat), critical_values=as.numeric(ans@cval), finite=finite)
        if (!finite) audit$unit_root_test_failed <- TRUE
        ans
      }
    })
    environment(original) <- list2env(list(ur.kpss=observed_kpss), parent=environment(original))
  }
  f <- original
  environment(f) <- new.env(parent=environment(original))
  environment(f)$.original <- original
  environment(f)$.kind <- kind
  environment(f)$audit <- audit
  body(f) <- quote({
    cl <- match.call(expand.dots=TRUE)
    cl[[1]] <- .original
    pf <- parent.frame()
    ev <- function(n, default=NULL) if (!is.null(cl[[n]])) eval(cl[[n]], pf) else default
    if (!audit$enabled) return(eval(cl, pf))
    if (.kind == "arima") {
      idx <- length(audit$calls) + 1L
      previous <- audit$active
      audit$active <- idx
      audit$calls[[idx]] <- list(index=idx, order=ev("order", c(0,0,0)),
                                method=ev("method", "CSS-ML"),
                                optimizations=list(), warnings=list())
      on.exit({audit$active <- previous}, add=TRUE)
      ans <- tryCatch(withCallingHandlers(eval(cl, pf), warning=function(w) {
        a <- audit$calls[[idx]]
        a$warnings[[length(a$warnings)+1L]] <- conditionMessage(w)
        audit$calls[[idx]] <- a
      }), error=function(e) {
        a <- audit$calls[[idx]]; a$error <- conditionMessage(e); audit$calls[[idx]] <- a
        stop(e)
      })
      a <- audit$calls[[idx]]
      a$code <- ans$code; a$coef <- as.list(ans$coef); a$loglik <- ans$loglik
      a$mask <- ans$mask; a$free_parameters <- sum(ans$mask)
      audit$calls[[idx]] <- a
      return(ans)
    }
    if (.kind == "optim") {
      ctrl <- ev("control", list()); pars <- ev("par", numeric())
      t0 <- proc.time()[3]
      ans <- eval(cl, pf)
      if (!is.na(audit$active)) {
        a <- audit$calls[[audit$active]]
        a$optimizations[[length(a$optimizations)+1L]] <- list(
          method=ev("method", "Nelder-Mead"), control=ctrl,
          parameters=as.numeric(ans$par), code=ans$convergence,
          value=ans$value, counts=as.list(ans$counts), message=ans$message,
          zero_iteration_rescore=isTRUE(ctrl$maxit == 0),
          seconds=unname(proc.time()[3]-t0))
        audit$calls[[audit$active]] <- a
      }
      return(ans)
    }
    before <- length(audit$calls)
    ans <- eval(cl, pf)
    if (.kind == "myarima" && length(audit$calls) > before) {
      a <- audit$calls[[length(audit$calls)]]
      a$native_ic <- ans$ic; a$aicc <- ans$aicc
      a$native_ic_rejected <- !is.finite(ans$ic)
      a$candidate_coef <- as.list(ans$coef)
      audit$calls[[length(audit$calls)]] <- a
    }
    if (.kind == "Arima") audit$branches[[length(audit$branches)+1L]] <- list(
      function_name="Arima", order=ev("order", c(0,0,0)), fixed=ev("fixed"),
      requested_method=ev("method"))
    ans
  })
  f
}
original_functions <- list()
for (spec in list(c("stats","arima"), c("stats","optim"),
                  c("forecast","myarima"), c("forecast","Arima"), c("forecast","ndiffs"))) {
  ns <- asNamespace(spec[1]); name <- spec[2]; orig <- get(name, ns)
  original_functions[[paste(spec, collapse="::")]] <- digest::digest(orig, algo="sha256")
  if (Sys.getenv("AUTOARIMA_R_OBSERVER", "1")=="1") {
    unlockBinding(name, ns); assign(name, wrap(orig, name), ns); lockBinding(name, ns)
  }
}
settings <- cfg$auto_arima
settings$d <- NA_integer_
settings$seasonal.test.args <- list()
roots <- function(co, sign, threshold=0) {
  ix <- which(abs(co) > threshold)
  if (!length(ix)) return(numeric())
  Mod(polyroot(c(1, sign * co[seq_len(max(ix))])))
}
emit(list(type="ready", pid=Sys.getpid(), forecast_version=as.character(packageVersion("forecast")),
          R_version=as.character(getRversion()), load_seconds=load_seconds,
          library=normalizePath(args[1]), observer_original_hashes=original_functions,
          worker_source_sha256=digest::digest(file=sub("^--file=", "",
            grep("^--file=",commandArgs(),value=TRUE)[1]),algo="sha256"),
          config_sha256=digest::digest(file=args[2],algo="sha256"),
          OS_threads=as.integer(sub("^Threads:[[:space:]]*", "",
            grep("^Threads:",readLines("/proc/self/status"),value=TRUE))),
          thread_environment=as.list(Sys.getenv(c("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "CLI_NO_THREAD"),names=TRUE))))
repeat {
  line <- readLines(input, n=1L, warn=FALSE)
  if (!length(line)) break
  request <- jsonlite::fromJSON(line, simplifyVector=FALSE)
  if (identical(request$action, "stop")) break
  t0 <- proc.time()[3]
  audit$active <- NA_integer_; audit$calls <- list(); audit$warnings <- list(); audit$branches <- list()
  audit$unit_root_checks <- list(); audit$unit_root_test_failed <- FALSE
  audit$enabled <- !identical(request$observe, FALSE)
  error_stage <- "input"
  ans <- tryCatch({
    allowed_partitions <- if (isTRUE(cfg$use_test_partitions)) c("internal_test", "external_test") else c("train", "validation", "synthetic")
    stopifnot(request$partition %in% allowed_partitions,
              request$width %in% c(48L,144L,288L), request$horizon %in% c(6L,12L,24L))
    x <- unpack(request$context_b64, request$width)
    stopifnot(length(x)==request$width, all(is.finite(x)))
    raw <- writeBin(x, raw(), size=8, endian="little")
    stopifnot(identical(digest::digest(raw, algo="sha256", serialize=FALSE), request$context_sha256))
    x <- ts(x, frequency=1)
    tfit <- proc.time()[3]
    error_stage <- "fit"
    fit <- withCallingHandlers(do.call(forecast::auto.arima, c(list(y=x), settings)), warning=function(w) {
      audit$warnings[[length(audit$warnings)+1L]] <- list(message=conditionMessage(w),
                                                      active_candidate=audit$active)
    })
    fit_seconds <- unname(proc.time()[3]-tfit)
    error_stage <- "forecast"
    tfc <- proc.time()[3]
    terminal_hash <- digest::digest(fit$model, algo="sha256")
    fc <- forecast::forecast(fit, h=request$horizon, level=95, fan=FALSE, bootstrap=FALSE)
    forecast_seconds <- unname(proc.time()[3]-tfc)
    pred <- as.numeric(fc$mean)
    stopifnot(length(pred)==request$horizon)
    # Nonfinite selected-model forecasts are emitted as diagnostic evidence;
    # the coordinator rejects them and publishes null, never a valid forecast.
    prefixes <- NULL
    if (isTRUE(request$prefix_check) && request$horizon==24L) {
      prefixes <- vapply(c(6L,12L), function(h) identical(
        pack(as.numeric(forecast::forecast(fit,h=h,level=95,fan=FALSE,bootstrap=FALSE)$mean)),
        pack(pred[seq_len(h)])), logical(1))
    }
    match_selected <- which(vapply(audit$calls, function(a) {
      is.null(a$error) &&
        (identical(unlist(a$coef), fit$coef) ||
         (length(a$coef)==0L && length(fit$coef)==0L)) &&
        isTRUE(all.equal(a$loglik, fit$loglik, tolerance=0))
    }, logical(1)))
    rp <- roots(fit$model$phi, -1); rq <- roots(fit$model$theta, 1)
    margin <- c(roots(fit$model$phi,-1,1e-8), roots(fit$model$theta,1,1e-8))
    diffx <- if (fit$arma[6] > 0) diff(as.numeric(x), differences=fit$arma[6]) else as.numeric(x)
    cov <- fit$var.coef
    list(type="result", id=request$id, partition=request$partition, width=request$width,
         horizon=request$horizon, context_sha256=request$context_sha256,
         input_roundtrip_b64=if (isTRUE(request$roundtrip)) pack(x) else NULL,
         input_summary=list(n=length(x), first=unname(x[1]), last=unname(tail(x,1)),
                            minimum=min(x), maximum=max(x), mean=mean(x), sd=sd(x)),
         native_constant=isTRUE(fit$constant), exact_constant=all(x==x[1]),
         native_tolerant_constant=forecast::is.constant(x),
         differenced_constant=forecast::is.constant(diffx),
         exact_differenced_constant=all(diffx==diffx[1]),
         order=c(fit$arma[1],fit$arma[6],fit$arma[2]), arma=fit$arma,
         coef=as.list(fit$coef), coefficient_b64=pack(fit$coef),
         mean=("intercept" %in% names(fit$coef)), drift=("drift" %in% names(fit$coef)),
         code=fit$code, free_parameters=sum(fit$mask), mask=fit$mask,
         loglik=fit$loglik, aic=fit$aic, aicc=fit$aicc, bic=fit$bic,
         sigma2=fit$sigma2, degenerate_likelihood=!is.finite(fit$loglik),
         degenerate_aicc=!is.finite(fit$aicc),
         covariance_finite=all(is.finite(cov)),
         covariance_diagonal_nonnegative=all(diag(as.matrix(cov))>=0),
         covariance_b64=pack(cov), ar_roots=rp, ma_roots=rq,
         mathematical_roots_valid=all(c(rp,rq)>1), native_margin_valid=all(margin>=1.01),
         forecast_b64=pack(pred), forecast_finite=all(is.finite(pred)),
         forecast_start=unname(start(fc$mean)), forecast_frequency=frequency(fc$mean),
         same_fit_prefix_bitwise=prefixes,
         forecast_terminal_state_unchanged=identical(terminal_hash, digest::digest(fit$model,algo="sha256")),
         selected_candidate_matches=match_selected, candidates=audit$calls,
         unit_root_checks=audit$unit_root_checks, unit_root_test_failed=audit$unit_root_test_failed,
         search_warnings=audit$warnings, native_branches=audit$branches,
         fit_seconds=fit_seconds, forecast_seconds=forecast_seconds,
         elapsed_seconds=unname(proc.time()[3]-t0))
  }, error=function(e) list(type="error", id=request$id, partition=request$partition,
                          width=request$width, horizon=request$horizon,
                          context_sha256=request$context_sha256,
                          error_stage=error_stage, error=conditionMessage(e),
                          unit_root_checks=audit$unit_root_checks, unit_root_test_failed=audit$unit_root_test_failed,
                          candidates=audit$calls, search_warnings=audit$warnings,
                          native_branches=audit$branches,
                          elapsed_seconds=unname(proc.time()[3]-t0)))
  emit(ans)
}
close(input)
