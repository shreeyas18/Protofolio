# Calling the need libraries
library(tidyverse)
library(rio)

# Importing needed documents
# Land
sec8a1 <- import("raw_data/sec8a1.dta")

# Livestock
sec8a2 <- import("raw_data/sec8a2.dta")

# Equipment
sec8a3 <- import("raw_data/sec8a3.dta")

# Farm Land Details
sec8b <- import("raw_data/sec8b.dta")
