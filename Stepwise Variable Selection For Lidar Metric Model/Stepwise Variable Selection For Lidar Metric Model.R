### Stepwise Variable Selection For Lidar Metric Model

# DESCRIPTION -----
#This script is designed to explore lidar metrics and 
#develop statistical models to predict forest attributes

#The script is broken into 3 PARTS (A-C)

#INSTRUCTIONS
# INSTRUCTIONS -----
#Step 1: Set your working directory in R studio 
#(Session > Set Working Directory > Choose Directory)

#Step 2: Fill in the required inputs in Part A. Ensure that all 
#paths are relative to the working directory you set in Step 1

#Step 3: Run PART A to create a master dataframe of relevant lab data

#Step 4: Explore the lidar data in Part C. Run portions of code by highlighting 
#and pressing Ctrl+Enter. This section is useful for determining a subset of  
#variables to use in model development.

#Step 5: Develop your model using foward stepwise selection. 
#Only use the variables you selected in Step 4 to develop your model

# SETUP    -----

#PART A) Lab Set-up
#load 'lidR', tidyverse' 
library(lidR)
library(tidyverse)

#set working directory to Lab 2 Data Folder 
setwd("path to workspace")

#Read in relevant .csvs
plot_table <- read_csv("Plots/Plot_Table.csv")
mkrf_plot_metrics <- read_csv("Plots/MKRF_Plot_Metrics.csv")

#Add column to "mkrf_plot_metrics' called Plot_ID (join key)
mkrf_plot_metrics$Plot_ID = 1:20

#Join 'Plot_Table' and 'MKRF_Plot_Metrics' into 'data_table'
data_table <- plot_table %>% 
  full_join(mkrf_plot_metrics)

# Data Exploration    -----

#PART B) DATA EXPLORATION
#list all the column names in our data_table (This is a combination of our two tables)
colnames(data_table)

#B.1
#Explore the relationship between lidar metrics and volume

#plot the relationship between lidar metrics and volume (Change zq50 to other variables to explore)
plot(Net_Volume ~ zq50, data = data_table)

#Fit a linear model between a single lidar metric and volume
model = lm(Net_Volume ~ zq50, data = data_table)
#Display a summary of the model
summary(model)

#Fit a linear model using multiple predictors (separating the predictors by '+' sign)
model = lm(Net_Volume ~ zmean + zsd, data = data_table)
#Display a summary of the model
summary(model)  

#Notice how the model barely improved when we add a second variable which is highly correlated with the first
#Look at the coefficients for this model using zq50 and zq90
model$coefficients
#There is a negative relationship to zq90, which does not make physical sense. 
#The model is likely fitting noise!

#B.2
#Explore the correlation between our lidar metrics

#Create a matrix of scatterplots between different height percentiles
pairs(~ zq10 + zq25 + zq50 + zq75 + zq90, data = data_table)
#See how height percentiles are all highly correlated? We won't need
#more than one or two to develop our model, as they basically provide the same information

#Create a matrix of scatterplots between less correlated variables
pairs(~ zq90 + pzabove2 + zentropy + zskew, data = data_table)
#The goal is to find a set of metrics where each metric provides unique information
#Notice how we have at least one cover metric, one height metric, and one complexity metric (i.e., variability metric)

# Initial Modelling    ---- 

#PART C) MODEL DEVELOPMENT
#Forward Variable Selection using the F-test

#We will work through two examples, where different initial variables were selected 

#Example 1

#Selected Variables
# zq90, pzabove2, zentropy, zskew

#We start with no variables in our model
model1 = lm(Net_Volume ~ 1, data = data_table)
#Now, we add each selected variable to our model one by one, to see which variable is the most significant
#predictor of volume
add1(model1,~  zq90 + pzabove2 + zentropy + zskew, test = 'F')

#Elev.P90 was the most significant (lowest Pr(>F)), so we add it to our model
model1 = lm(Net_Volume ~ zq90, data = data_table)
#Now, We add each remaining variable to the new model one by one, to see 
#if any variable is a significant addition
add1(model1,~  zq90 + pzabove2 + zentropy + zskew, test = 'F')

#pzabove2 was the most significant addition (lowest Pr(>F)), so we add it to our model
model1 = lm(Net_Volume ~ zq90 + pzabove2, data = data_table)

#Again, we test all the remaining variables one by one
add1(model1,~ zq90 + pzabove2 + zentropy + zskew, test = 'F')

#No additional variables were significant, so we can stop building our model

#Get the summary of the final model
summary(model1)

#Plot our predicted volume against our measured volume
plot(Net_Volume ~ model1$fitted,data = data_table,xlab = 'Predicted',ylab = 'Measured')
abline(0,1) #Adds a one to one line

#Get the coefficients to our model
model1$coefficients

#Example 2

#Here is a slightly different selection of metrics (zq25 instead of zq90)
# zq25, pzabove2, zentropy, zskew

#Again, we start with no variables in our model
model2 = lm(Net_Volume ~ 1, data = data_table)
#Now, add in one variable at time, and select the most significant
add1(model2,~ zq25 + pzabove2 + zentropy + zskew, test = 'F')

#Elev.P25 was the most significant, so we add it to our model
model2 = lm(Net_Volume ~ zq25, data = data_table)
#Add each remaining variable one by one
add1(model2,~ zq25 + pzabove2 + zentropy + zskew, test = 'F')

#No additional variable comes out as significant, so our final model only has one variable

#Get the summary of our model
summary(model2)

#Plot our predicted volume against our measured volume
plot(Net_Volume ~ model2$fitted,data = data_table,xlab = 'Predicted', ylab = 'Measured')
abline(0,1, col = "red")

#Get the output coefficients to our model
model2$coefficients

#We can compare our two models
summary(model1)
summary(model2)

# Calculating Grid Metrics    -----
#PART D) GRID METRICS
#Area-based approach to calculate metrics for the entire study area
#Explore grid_metrics (lidR) and calc (raster)
?grid_metrics
?calc

# Calculate grid_metrics for all MKRF 
#Create LAScatalog of filtered, normalized tiles with points 2 m - 65 m 
norm_cat_mkrf <- readLAScatalog("Normalized")
opt_filter(norm_cat_mkrf) <- '-keep_z_above 2 -drop_z_above 65'
plot(norm_cat_mkrf)

#Calculate grid metrics of mean Z at 10 m resolution for entire study area
grid_metrics_mkrf <- grid_metrics(norm_cat_mkrf, .stdmetrics_z, 10) 
plot(grid_metrics_mkrf)

#subset "zq25" from grid_metrics_mkrf RasterBrick
zq25_mkrf_r <- subset(grid_metrics_mkrf, "zq25")
plot(zq25_mkrf_r)

#Create function from model2 coefficients
f <- function(x){
  24.96*x -139.45
}

#Apply function to raster
net_volume_mkrf <- calc(zq25_mkrf_r, f)
plot(net_volume_mkrf)

#Visualize scatterplot between canopy cover and volume
plot(Net_Volume ~ pzabove2, data = data_table)


