# Load necessary libraries
library(tidyverse)
library(tensorflow)
library(keras)
library(reticulate)
library(caret)
library(lime)
library(imager)
library(abind)

# Set Python path
use_python("C:/Users/Hp/AppData/Local/Programs/Python/Python311/python.exe")

# Define the paths
train_image_files_path <- "D:/PFE_References/Brain_Tumor/Training"

# Load the model
model <- load_model_hdf5("www/my_model.h5")

# Preprocess function to match the preprocessing used in training
preprocess <- function(path) {
  img <- image_load(path, target_size = c(128, 128))
  img <- image_to_array(img)
  img <- array_reshape(img, c(1, 128, 128, 3))
  img <- img / 255
  return(img)
}

# Function to predict class probabilities
model_predict <- function(paths) {
  images <- lapply(paths, preprocess)
  images_array <- abind::abind(images, along = 1)
  preds <- predict(model, images_array)
  return(as.data.frame(preds))
}

# Define a custom model class
CustomModel <- R6::R6Class(
  "CustomModel",
  public = list(
    predict = model_predict
  )
)

# Define the model_type method for the custom model
model_type.CustomModel <- function(x, ...) {
  return("classification")
}

# Define the predict_model method for the custom model
predict_model.CustomModel <- function(x, newdata, ...) {
  return(x$predict(newdata))
}

# Create an instance of the custom model
custom_model <- CustomModel$new()

# Prepare a set of images for the explainer
image_paths <- list.files(train_image_files_path, pattern = "*.jpg", full.names = TRUE, recursive = TRUE)
sampled_image_paths <- sample(image_paths, 50) # sample 50 images for explanation

# Create an explainer
explainer <- lime::lime(
  x = sampled_image_paths,
  model = custom_model
)

# Choose an image for explanation
image_path <- "D:/PFE_References/Brain_Tumor/Training/meningioma/Tr-me_0022.jpg"

# Generate explanations
explanation <- lime::explain(
  x = image_path,
  explainer = explainer,
  n_labels = 1, 
  n_features = 5
)

# Visualize the explanation
plot_features(explanation)
