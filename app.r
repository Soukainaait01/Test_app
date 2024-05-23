library(shiny)
library(shinydashboard)
library(shinyjs)
library(keras)
# Utiliser le chemin vers Python
use_python("C:/Users/Hp/AppData/Local/Programs/Python/Python311/python.exe")

# Define the user interface (UI)
ui <- dashboardPage(
  dashboardHeader(title = "App Image Recognition"),
  dashboardSidebar(
    useShinyjs(),
    sidebarMenu(
      menuItem("Brain Tumor", tabName = "classify")
    ),
    fileInput("imageFile", "Choose an image file", accept = c("image/jpeg", "image/png")),
    actionButton("classify", "Classify", icon = icon("magic"))
  ),
  dashboardBody(
    tabItems(
      tabItem(tabName = "classify",
              fluidRow(
                column(6,
                       # Image display area
                       imageOutput("image")
                ),
                column(6,
                       fluidRow(
                         column(12,
                                # Classification Result
                                textOutput("classificationResult")
                         )
                       )
                )
              )
      )
    )
  )
)

# Define the server logic
server <- function(input, output) {
  # List of images to model
  data_list <-  c("glioma", "meningioma", "pituitary","notumor")

  # Load your pre-trained model here
  loaded_model <- load_model_hdf5("www/my_model.h5")

  # Render the uploaded image
  output$image <- renderImage({
    req(input$imageFile)
    list(src = input$imageFile$datapath,
         height = "auto", width = "90%")
  }, deleteFile = FALSE)

  # Reactive expression for classification
  classify_image <- eventReactive(input$classify, {
    req(input$imageFile)
    # Preprocess the image
    img <- keras::image_load(input$imageFile$datapath, target_size = c(128, 128))
    img_array <- keras::image_to_array(img)
    img_array <- array_reshape(img_array, c(1, dim(img_array)))
    img_array <- img_array / 255

    # Classify the image
    prediction <- loaded_model %>% predict(img_array)
    predicted_class <- data_list[which.max(prediction)]
    max_probability <- max(prediction) * 100 # Convert probability to percentage

    paste("The image is classified as:", predicted_class,
          sprintf("with a probability of %.2f%%.", max_probability))
  })
  

  # Display the classification result
  output$classificationResult <- renderText({
    classify_image()
  })
}

# Run the Shiny app
shinyApp(ui = ui, server = server)
