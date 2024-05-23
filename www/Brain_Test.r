# Load libraries
library(tidyverse)
library(tensorflow)
library(keras)
library(reticulate)
library(caret)

# Set Python path
use_python("C:/Users/Hp/AppData/Local/Programs/Python/Python311/python.exe")

# Set seed
tf$random$set_seed(42)

# Paths
train_image_files_path <- "D:/PFE_References/Brain_Tumor/Training"
valid_image_files_path <- "D:/PFE_References/Brain_Tumor/Validation"

# List of classes (types of brain tumors)
data_list <- c("glioma", "meningioma", "pituitary","notumor")
output_n <- length(data_list)

# Image dimensions
img_width <- 128
img_height <- 128
target_size <- c(img_width, img_height)
channels <- 3
batch_size <- 32

# Data generators with augmentation
train_data_gen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.3,
  rotation_range = 20,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)

# Define K for K-Fold Cross Validation
k <- 1
folds <- createFolds(1:length(list.files(train_image_files_path, recursive = TRUE)), k = k)

# Initialize lists to store validation accuracies
validation_accuracies <- vector("list", length = k)

# Initialize lists to store labels
all_train_labels <- list()
all_valid_labels <- list()

# K-fold Cross-validation loop
for (i in 1:k) {
  cat("\nFold", i, "...\n")

  # Data generators for this fold
  train_image_array_gen <- flow_images_from_directory(
    train_image_files_path,
    train_data_gen,
    subset = 'training',
    target_size = target_size,
    class_mode = "categorical",
    classes = data_list,
    batch_size = batch_size,
    seed = 42
  )
  
  valid_image_array_gen <- flow_images_from_directory(
    train_image_files_path,
    train_data_gen,
    subset = 'validation',
    target_size = target_size,
    class_mode = "categorical",
    classes = data_list,
    batch_size = batch_size,
    seed = 42
  )

  # Define and compile the model using transfer learning
  base_model <- application_vgg16(weights = 'imagenet', include_top = FALSE, input_shape = c(img_width, img_height, channels))

  model <- keras_model_sequential() %>%
    base_model %>%
    layer_flatten() %>%
    layer_dense(units = 512, activation = "relu") %>%
    layer_dropout(rate = 0.5) %>%
    layer_dense(units = output_n, activation = "softmax")

  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_adam(learning_rate = 0.0001),
    metrics = "accuracy"
  )

  # Training the model
  history <- model %>% fit(
    x = train_image_array_gen,
    epochs = 10,  # Keeping epochs at 10 to save time
    validation_data = valid_image_array_gen,
    callbacks = list(
      callback_early_stopping(monitor = "val_loss", patience = 3),
      callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.2, patience = 2)
    )
  )

  # Save validation accuracy for this fold
  validation_accuracies[[i]] <- max(history$metrics$val_accuracy)

  # Save labels once
  all_train_labels <- c(all_train_labels, train_image_array_gen$classes)
  all_valid_labels <- c(all_valid_labels, valid_image_array_gen$classes)
}

# Save labels in RData format once at the end
save(all_train_labels, file = "train_labels.RData")
save(all_valid_labels, file = "valid_labels.RData")

# Print validation accuracies for each fold
cat("\nValidation accuracies for each fold:\n")
print(validation_accuracies)

# Save the final model
model %>% save_model_hdf5("www/my_model.h5")
plot(history)
