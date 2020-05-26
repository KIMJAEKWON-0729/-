install.packages("jsonlite")
library(jsonlite)
library(tidyverse)

train <-  as.data.frame(fromJSON(file = "train.json"))

data <- jsonlite::fromJSON("train.json")
song_meta <-jsonlite::fromJSON("song_meta.json")

train <-  data.frame(data)
