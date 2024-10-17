# Calling the need libraries
library(tidyverse)
library(rio)

# Importing needed documents
# Education:
# General Education
sec2a <- import("raw_data/sec2a.dta")

# Created HouseholdEducation Dataframe
HouseholdEducation <- sec2a %>%
  select(nh, clust:s2aq4, s2aq17) %>%
  filter(s2aq2 != "96")
summary(HouseholdEducation)

# Mapping numbers to words based on the key
EducationLevel <- HouseholdEducation %>%
  mutate(education_level = case_when(
    s2aq2 == 1 ~ "None",
    s2aq2 == 2 ~ "Kindergarten",
    s2aq2 == 3 ~ "Primary",
    s2aq2 == 4 ~ "Middle",
    s2aq2 == 5 ~ "JSS",
    s2aq2 == 6 ~ "SSS",
    s2aq2 == 7 ~ "Voc/Commercial",
    s2aq2 == 8 ~ "Secondary (o Level)",
    s2aq2 == 9 ~ "Sixth Form",
    s2aq2 == 10 ~ "Teacher Training",
    s2aq2 == 11 ~ "Technical",
    s2aq2 == 12 ~ "Post Secondary T/T",
    s2aq2 == 13 ~ "Nursing",
    s2aq2 == 14 ~ "P/Sec Nursing",
    s2aq2 == 15 ~ "Polytechnic",
    s2aq2 == 16 ~ "University",
    s2aq2 == 17 ~ "Koranic stage",
    s2aq2 == 96 ~ "Other",
    TRUE ~ "Unknown"  # Add a default for unexpected values
  ))

# Dummy Data
sec8b <- import("raw_data/sec8b.dta")
dummy_data <- merge(final_aggregated_data, sec8b, by = c("nh", "clust"))
# Select only the desired columns
selected_data <- dummy_data %>%
  select(profit_per_acre, s8bq11)

filtered_dummy_data <- dummy_data %>%
  select(nh, clust, profit_per_acre, s8bq11)

# Mapping numbers to words based on the key
clean_dummy_data <- filtered_dummy_data %>%
  filter(!is.na(s8bq11)) %>%
  select(nh, clust, profit_per_acre, s8bq11)

# Merge clean_dummy_data with EducationLevel
dummy_data_with_education <- merge(clean_dummy_data, EducationLevel, by = c("nh", "clust"), all = TRUE)

# Recode s8bq11 such that 1 means Yes (farm was cultivated) and 0 means No (farm was not cultivated)
clean_dummy_data$s8bq11 <- ifelse(clean_dummy_data$s8bq11 == 1, 1, 0)

# regression model with the recoded dummy variable
regression_model <- lm(profit_per_acre ~ s8bq11, data = clean_dummy_data)

# Display the summary of the regression model
summary(regression_model)

# Bar Graph based on dummy data
ggplot(clean_dummy_data, aes(x = factor(s8bq11), y = profit_per_acre)) +
  geom_bar(stat = "summary", fun = "mean", fill = "darkgreen", color = "black") +
  labs(title = "Profit Per Acre Based On Education",
       x = "Attended School (1: Yes, 0: No)",
       y = "Mean Profit per Acre") +
  theme_minimal()

# Box Plot based on attended school
ggplot(clean_dummy_data, aes(x = factor(s8bq11), y = profit_per_acre)) +
  geom_boxplot(fill = "skyblue", color = "black") +
  labs(title = "Profit Per Acre Based On Education",
       x = "Attended School",
       y = "Profit per Acre") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels for better readability

# Scatter plot with jitter to separate overlapping points
ggplot(clean_dummy_data, aes(x = factor(s8bq11), y = profit_per_acre)) +
  geom_jitter(aes(color = factor(s8bq11)), width = 0.2, height = 0) +
  labs(title = "Profit Per Acre Based On Education",
       x = "Attended School",
       y = "Profit per Acre") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),  # Rotate x-axis labels for better readability
        legend.title = element_blank())  # Remove the legend title if not needed

# Calculate the IQR
Q1 <- quantile(clean_dummy_data$profit_per_acre, 0.25, na.rm = TRUE)
Q3 <- quantile(clean_dummy_data$profit_per_acre, 0.75, na.rm = TRUE)
IQR <- Q3 - Q1

# Define the bounds for what you consider outliers
# Typically, 1.5 is used, but this can be adjusted depending on how stringent you want to be
lower_bound <- Q1 - 1.5 * IQR
upper_bound <- Q3 + 1.5 * IQR

# Filter out the outliers
clean_data_without_outliers <- clean_dummy_data %>%
  filter(profit_per_acre >= lower_bound, profit_per_acre <= upper_bound)

# Create the box plot with the data without outliers
ggplot(clean_data_without_outliers, aes(x = factor(s8bq11), y = profit_per_acre)) +
  geom_boxplot(fill = "skyblue", color = "black") +
  labs(title = "Profit Per Acre Based On Education (Without Outliers)",
       x = "Attended School",
       y = "Profit per Acre") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels for better readability

# Print names of clean_dummy_data
print(names(clean_dummy_data))

# Scatter plot with education level
ggplot(dummy_data_with_education, aes(x = education_level, y = profit_per_acre)) +
  geom_point(size = 3) +
  labs(title = "Profit Per Acre Based On Education Level",
       x = "Education Level",
       y = "Profit per Acre") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels for better readability

# Boxplot without removing missing values
ggplot(dummy_data_with_education, aes(x = education_level, y = profit_per_acre)) +
  geom_boxplot() +
  labs(
    title = "Profit per Acre by Education Level",
    x = "Education Level",
    y = "Profit per Acre"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Exclude rows with missing values
clean_dummy_data <- clean_dummy_data[complete.cases(clean_dummy_data)]
summary(clean_dummy_data)

# Plot without removing missing values
ggplot(dummy_data_with_education, aes(x = education_level, y = profit_per_acre)) +
  geom_boxplot() +
  labs(
    title = "Profit per Acre by Education Level",
    x = "Education Level",
    y = "Profit per Acre"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
