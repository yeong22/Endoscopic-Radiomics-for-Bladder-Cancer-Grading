# Table 1: patient characteristics by cohort (base R, no packages).
# Usage: Rscript stats/table1.R outputs/paper/regions_split.csv
# Age is taken per patient (maximum over the patient's regions, as in the original script) and
# summarised as median (IQR), as in the paper. Grade is counted at the tumour-region level.

args <- commandArgs(trailingOnly = TRUE)
path <- if (length(args) > 0) args[1] else "outputs/paper/regions_split.csv"
d <- read.csv(path, stringsAsFactors = FALSE)

summarise_cohort <- function(x) {
  age <- tapply(x$age, x$patient_id, max)
  male <- tapply(x$sex == "M", x$patient_id, function(v) v[1])
  q <- quantile(age, c(0.25, 0.5, 0.75))
  hg <- x$grade == "HG"
  c("Patients, n" = length(age),
    "Male, n (%)" = sprintf("%d (%.0f)", sum(male), 100 * mean(male)),
    "Age (yr), median (IQR)" = sprintf("%.0f (%.0f-%.0f)", q[2], q[1], q[3]),
    "Tumour regions, n" = nrow(x),
    "HG regions, n (%)" = sprintf("%d (%.0f)", sum(hg), 100 * mean(hg)))
}

cohorts <- intersect(c("train", "internal", "external"), unique(d$cohort))
tab <- sapply(cohorts, function(k) summarise_cohort(d[d$cohort == k, ]))
print(noquote(tab))
out <- file.path(dirname(path), "table1.csv")
write.csv(tab, out)
cat("saved", out, "\n")
