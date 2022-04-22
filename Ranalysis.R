# clear the variable environment 
rm(list = ls())

# Set the working directory here 
setwd("/Users/gwk/Desktop/UniTutoring")

# Load the desired packages here 
# Load function
#source("http://pcwww.liv.ac.uk/~william/R/crosstab.r")
library(table1)
library(crosstable)
library(dplyr)

# Check the dowrking directory and its contents here 
getwd()
dir()

# Load that data to be analysed here 
df <- read.csv("final_Data.csv")

# Explore the dataset here
head(df)

## Remove the extra spaces in the spesis 
df$Diagnosis <- gsub(" ","",df$Diagnosis)

# Create a demographic table here
#### Create demographics table for publication here 
table1::label(df$Gender) <- "Gender"
table1::label(df$AgeGroup) <- "Age Group"
table1::label(df$Concern) <- "Presentation"

# Generate the table here
table1::table1(~Gender + diagnosis + Concern | AgeGroup, data = df)
table1::table1(~Gender + AgeGroup + diagnosis + Concern | Cause, data = df)
table1::table1(~Gender + AgeGroup + Concern , data = df)

##### Diagnostic Yeild of ilr for the underlying causes of syncope
table1(~Diagnosis | AgeGroup, data = df)
table1(~ Concern + diagnosis + Cause | AgeGroup, data = df)
table1(~ Age + AgeGroup | Concern, data = df, overall=F, extra.col=list("P-value"=pvalue(df$AgeGroup)))

############### Statistical testing starts from here ###################
x <- df %>%
  group_by("Gender") %>%
  select(Concern)
head(x)
kruskal.test(x)
wilcox.test(x)
