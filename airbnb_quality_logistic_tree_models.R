# -----------------------------
# 1. Libraries
# -----------------------------
library(dplyr)
library(fastDummies)
library(corrplot)
library(caret)
library(rpart)
library(rpart.plot)

# -----------------------------
# 2. Load Data
# -----------------------------

# Read and view the Airbnb listings data from CSV
df_listings <- read.csv("C:/Users/USER/Desktop/MDX/Applied Data Science Lifecycle/Coursework/airbnb data/listings.csv")
View(df_listings)

# Convert relevant columns to numeric for calculations
cols_to_convert = c("reviews_per_month", "number_of_reviews", "availability_365", "review_scores_rating")
df_listings[cols_to_convert] <- lapply(df_listings[cols_to_convert], as.numeric)

# Calculate medians for thresholds, ignoring NAs
median_reviews_pm <- median(df_listings$reviews_per_month, na.rm = TRUE)#, )
median_reviews_total <- median(df_listings$number_of_reviews, na.rm = TRUE)#)
median_availability <- median(df_listings$availability_365, na.rm = TRUE)#, na.rm = TRUE)

summary(df_listings[, c("reviews_per_month", "number_of_reviews", "review_scores_rating", "availability_365")])

# Calculate estimated occupancy rate
df_listings$est_occupancy_rate <- (365 - df_listings$availability_365) / 365

# Define quality label based on review score, occupancy, and number of reviews
df_listings$quality <- ifelse(
  df_listings$review_scores_rating >= 4.3 &
    df_listings$est_occupancy_rate >= 0.5 &
    df_listings$number_of_reviews >= 5,
  "Good", "Bad"
)
df_listings$quality <- as.factor(df_listings$quality)

# Check distribution of quality labels
table(df_listings$quality)

# Check for zero or NA values in key columns
sum(df_listings$review_scores_rating == 0, na.rm = TRUE)
sum(df_listings$number_of_reviews == 0, na.rm = TRUE)
sum(df_listings$availability_365 == 0, na.rm = TRUE)

sum(is.na(df_listings$review_scores_rating) & !is.na(df_listings$number_of_reviews) & df_listings$number_of_reviews == 0)

# -----------------------------
# 3. Preprocessing
# -----------------------------

# Cap reviews rating to a maximum of 5
summary(df_listings$review_scores_rating)
df_listings$review_scores_rating[df_listings$review_scores_rating > 5] <- 5

# Ensure no ratings > 0 when number of reviews is 0
sum(df_listings$number_of_reviews == 0 & df_listings$review_scores_rating > 0)

# Replace NA in availability_365 with 365 (assuming full availability if missing)
df_listings$availability_365[is.na(df_listings$availability_365)] <- 365

# Replace NA in review_scores_rating and number_of_reviews with 0
df_listings$review_scores_rating[is.na(df_listings$review_scores_rating)] <- 0
df_listings$number_of_reviews[is.na(df_listings$number_of_reviews)] <- 0

# Ensure no ratings > 0 when number of reviews is 0
sum(df_listings$number_of_reviews == 0 & df_listings$review_scores_rating > 0)

# Remove $ and commas from price, convert to numeric
df_listings$price <- gsub("\\$", "", df_listings$price)
df_listings$price <- gsub(",", "", df_listings$price)
df_listings$price <- as.numeric(df_listings$price)

# Check for zero or extreme prices
sum(df_listings$price == 0, na.rm = TRUE)
sum(df_listings$price > 1000, na.rm = TRUE)
boxplot(df_listings$price, main="Price Distribution")

# Create amenities_count by counting items in amenities string
df_listings$amenities_count <- sapply(df_listings$amenities, function(x) {
  x <- gsub("[{}]", "", x)                 # Remove curly braces
  items <- unlist(strsplit(x, ","))        # Split by comma
  length(trimws(items))                    # Count trimmed items
})

# Relocate amenities_count next to amenities for better organization
df_listings <- df_listings %>%
  relocate(amenities_count, .after = amenities)

# Convert latitude and longitude to numeric
df_listings$latitude <- as.numeric(df_listings$latitude)
df_listings$longitude <- as.numeric(df_listings$longitude)

# Check for NA in latitude and longitude 
sum(is.na(df_listings$latitude == 0 & df_listings$longitude == 0))

# Convert categorical variables to factors
factor_vars <- c(
  "host_is_superhost", 
  "host_identity_verified",
  "property_type", 
  "room_type",
  "instant_bookable",
  "neighbourhood_cleansed"
)

df_listings[factor_vars] <- lapply(df_listings[factor_vars], as.factor)

# Convert host response and acceptance rates to numeric (remove %)
df_listings[, c("host_response_rate", "host_acceptance_rate")] <-
  lapply(df_listings[, c("host_response_rate", "host_acceptance_rate")], function(x) {
    x <- gsub("%", "", x)
    as.numeric(x)
  }
)

# Set all the bedrooms to 1 if it is a private or shared room
df_listings$bedrooms <- ifelse(
  df_listings$room_type == 'Private room' | df_listings$room_type == 'Shared room' | df_listings$room_type == 'Hotel room',
  1,
  df_listings$bedrooms
)

# Private room → 1 bedroom if NA
df_listings$bedrooms[df_listings$room_type == "Private room" & is.na(df_listings$bedrooms)] <- 1

# Shared room → 1 bedroom if NA
df_listings$bedrooms[df_listings$room_type == "Shared room" & is.na(df_listings$bedrooms)] <- 1

# Hotel room → 1 bedroom if NA
df_listings$bedrooms[df_listings$room_type == "Hotel room" & is.na(df_listings$bedrooms)] <- 1

# Impute median on Entire home/apt if NA
df_listings$bedrooms[df_listings$room_type == "Entire home/apt" & is.na(df_listings$bedrooms)] <- median(
  df_listings$bedrooms[df_listings$room_type == "Entire home/apt"], na.rm = TRUE
)

# Convert superhost and identity verified to logical
df_listings$host_is_superhost <- trimws(df_listings$host_is_superhost) == 't'
df_listings$host_identity_verified <- trimws(df_listings$host_identity_verified) == 't'

# Replace NA in superhost and identity verified with FALSE
df_listings$host_is_superhost[is.na(df_listings$host_is_superhost)] <- FALSE
df_listings$host_identity_verified[is.na(df_listings$host_identity_verified)] <- FALSE


# Replace NA in host_listings_count with 1
df_listings$host_listings_count[is.na(df_listings$host_listings_count)] <- 1

# Check room types with NA bedrooms
table(df_listings$room_type[is.na(df_listings$bedrooms)])

# Impute beds with bedrooms if NA
df_listings$beds[is.na(df_listings$beds)] <- df_listings$bedrooms[is.na(df_listings$beds)]

# Handle bathrooms_text: replace empty with NA, extract numeric bathrooms
df_listings$bathrooms_text[df_listings$bathrooms_text == ""] <- NA
df_listings$bathrooms <- as.numeric(sub("^([0-9\\.]+).*", "\\1", df_listings$bathrooms_text))
df_listings$bathrooms[is.na(df_listings$bathrooms)] <- median(df_listings$bathrooms, na.rm = TRUE)

# Cap outliers in various columns
df_listings$price <- pmin(df_listings$price, 1000)
df_listings$maximum_nights <- pmin(df_listings$maximum_nights, 365)
df_listings$minimum_nights <- pmin(df_listings$minimum_nights, 30)
df_listings$beds <- pmin(df_listings$beds, 10)
df_listings$bathrooms <- pmin(df_listings$bathrooms, 5)
df_listings$accommodates <- pmin(df_listings$accommodates, 15)
df_listings$amenities_count <- pmin(df_listings$amenities_count, 50)

# Feature Engineering ----------------------------------------------

# Create indicator for hosts with multiple listings (>5)
df_listings$multi_host <- ifelse(df_listings$host_listings_count > 5, 1, 0)

# Create a column bathroom shared
df_listings$bathrooms_shared <- ifelse(grepl("shared", df_listings$bathrooms_text), 1, 0)

# Create a column which groups property types into 4 groups
df_listings$property_type_group <- case_when(
  grepl("Entire", df_listings$property_type) ~ "Entire",
  grepl("Private", df_listings$property_type) ~ "Private",
  grepl("Shared", df_listings$property_type) ~ "Shared",
  TRUE ~ "Other"
)

# Convert property type as factor
df_listings$property_type_group <- as.factor(df_listings$property_type_group)

# df_listings$instant_bookable <- ifelse(df_listings$instant_bookable == 't', 1, 0)

# --------------------------------
# 4. Feature Selection & Encoding
# --------------------------------

features = df_listings[, c(
  "host_is_superhost",
  "host_identity_verified",
  "multi_host",
  "price", 
  "property_type_group", 
  "room_type", 
  "accommodates", 
  # "bathrooms", 
  # "bathrooms_shared",
  "neighbourhood_cleansed", 
  "amenities_count",
  "instant_bookable", 
  "minimum_nights", 
  "maximum_nights"
)]

summary(features)

df_listings$instant_bookable <- ifelse(features$instant_bookable == 't', 1, 0)

# One-hot encode categorical variables
features <- fastDummies::dummy_cols(
  features, 
  select_columns = c("room_type", "property_type_group", "neighbourhood_cleansed"),
  remove_selected_columns = TRUE,
  remove_first_dummy = TRUE  # avoid dummy trap
)

colnames(features) <- make.names(colnames(features), unique = TRUE)

num_features <- c("price", "accommodates", 
              "amenities_count",
              "minimum_nights", "maximum_nights")

# Convert logical TRUE/FALSE to numeric only for plotting corr plot
features_numeric_bool <- features
features_numeric_bool[sapply(features_numeric_bool, is.logical)] <- 
  lapply(features_numeric_bool[sapply(features_numeric_bool, is.logical)], as.numeric)

features_numeric_bool <- data.frame(lapply(features_numeric_bool, as.numeric))

# Plot numeric features
numeric_only <- features_numeric_bool[, num_features]  # just numeric columns
cor_matrix <- cor(numeric_only, use = "complete.obs")
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.8)

# Scale numeric features
features_scaled <- features
features_scaled[num_features] <- scale(features[num_features])

summary(features_scaled[num_features])

# -----------------------------
# 5. Train-Test Split
# -----------------------------

set.seed(123)

# 80% training + 20% test
train_index <- createDataPartition(df_listings$quality, p = 0.8, list = FALSE)

train_features <- features_scaled[train_index, ]
train_target <- df_listings$quality[train_index]

test_features <- features_scaled[-train_index, ]
test_target <- df_listings$quality[-train_index]


# Combine for training
train_df <- data.frame(train_features, quality = train_target)

cat("Train samples:", nrow(train_features), "\n")
cat("Test samples:", nrow(test_features), "\n")

# -----------------------------
# 6. Logistic Regression
# -----------------------------

row_weights_log <- ifelse(train_target == "Bad", 1, 1.5)

# Train Logistic Regeression Model
log_model <- glm(
  quality ~ .,
  data = train_df,
  family = binomial(link = "logit"),
  weights = row_weights_log
)

summary(log_model)

# Predict
test_probs <- predict(log_model, newdata = test_features, type = "response")
test_pred <- ifelse(test_probs > 0.45, "Good", "Bad")

# Confusion Matrix for Logistic
conf_mat_log <- confusionMatrix(
  factor(test_pred, levels = c("Bad", "Good")),
  factor(test_target, levels = c("Bad", "Good")),
  positive = "Good"
)
print(conf_mat_log)

# F1 Scores for Logistic
precision_good_log <- conf_mat_log$byClass["Pos Pred Value"]
recall_good_log    <- conf_mat_log$byClass["Sensitivity"]
f1_good_log        <- 2 * (precision_good_log * recall_good_log) / (precision_good_log + recall_good_log)

precision_bad_log <- conf_mat_log$table["Bad","Bad"] / sum(conf_mat_log$table[,"Bad"])
recall_bad_log    <- conf_mat_log$table["Bad","Bad"] / sum(conf_mat_log$table["Bad",])
f1_bad_log        <- 2 * (precision_bad_log * recall_bad_log) / (precision_bad_log + recall_bad_log)

print(paste("F1 Score for Good (Logistic):", round(f1_good_log, 4)))
print(paste("F1 Score for Bad (Logistic):", round(f1_bad_log, 4)))

# -----------------------------
# 7. Decision Tree
# -----------------------------

row_weights_dt <- ifelse(train_target == "Bad", 1, 2)

# Train the tree
dt_model <- rpart(
  quality ~ .,
  data = cbind(train_features, quality=train_target),  
  method = "class",
  weights = row_weights_dt,
  control = rpart.control(
    minsplit = 20, 
    cp = 0.005    
  )
)

# Plot the tree
rpart.plot(dt_model, type=3, extra=101)

# Predict
dt_pred <- predict(dt_model, test_features, type="class")

# Confusion matrix for Tree
conf_mat_dt <- confusionMatrix(dt_pred, test_target, positive = "Good")
print(conf_mat_dt)

# F1 Scores for Tree
precision_good_dt <- conf_mat_dt$byClass["Pos Pred Value"]
recall_good_dt    <- conf_mat_dt$byClass["Sensitivity"]
f1_good_dt        <- 2 * (precision_good_dt * recall_good_dt) / (precision_good_dt + recall_good_dt)

precision_bad_dt <- conf_mat_dt$table["Bad","Bad"] / sum(conf_mat_dt$table[,"Bad"])
recall_bad_dt    <- conf_mat_dt$table["Bad","Bad"] / sum(conf_mat_dt$table["Bad",])
f1_bad_dt        <- 2 * (precision_bad_dt * recall_bad_dt) / (precision_bad_dt + recall_bad_dt)

print(paste("F1 Score for Good (Decision Tree):", round(f1_good_dt, 4)))
print(paste("F1 Score for Bad (Decision Tree):", round(f1_bad_dt, 4)))

# Feature importance
importance <- dt_model$variable.importance
print(importance)


