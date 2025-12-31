# Airbnb Listing Success Classification (Coursework)

This repository contains the R code developed for a Master’s-level coursework in Applied Data Science.  
The objective of the coursework was to build a predictive model that classifies Airbnb listings as **Good** or **Bad** based on listing characteristics, host attributes, availability, and review-related indicators.

The project follows key stages of the data science lifecycle, including feature engineering, model development, evaluation, and interpretation.

---

## Coursework Objective

The goal of the coursework was to estimate **Airbnb listing success** using a categorical target variable derived from domain-informed logic.  
Listing success was defined internally (Good / Bad) using a combination of:

- Review scores
- Estimated occupancy
- Engagement (number of reviews)

This approach aligns with the coursework requirement to construct a reproducible success metric grounded in business understanding rather than relying on a predefined label :contentReference[oaicite:1]{index=1}.

---

## Dataset

The analysis uses Airbnb listing-level data containing information on:

- Property and room characteristics
- Pricing and availability
- Host attributes
- Review metrics
- Neighbourhood information

The raw dataset is not included in this repository.

---

## Success Metric (Dependent Variable)

A binary **listing quality** label was constructed using the following logic:

A listing is classified as **Good** if it satisfies all of the following:
- Review score ≥ 4.3  
- Estimated occupancy rate ≥ 50%  
- Number of reviews ≥ 5  

Listings that do not meet these conditions are classified as **Bad**.

This formulation reflects listing performance, guest satisfaction, and demand, and is fully reproducible from the raw data.

---

## Methodology

### 1. Data Preprocessing
Key preprocessing steps include:
- Conversion of numeric and categorical variables to appropriate formats
- Handling missing values using domain-informed assumptions (e.g. availability, reviews, bedrooms)
- Outlier capping for price, nights, beds, bathrooms, and amenities
- Cleaning and standardisation of price and percentage variables

### 2. Feature Engineering
Additional features were created to improve predictive performance, including:
- Estimated occupancy rate
- Amenities count
- Multi-host indicator
- Grouped property type categories

Categorical variables were one-hot encoded, and numeric features were scaled where appropriate.

---

## Models

Two supervised classification models were implemented:

### Logistic Regression
- Weighted logistic regression to address class imbalance
- Probability threshold adjustment for classification
- Used as a baseline, interpretable model

### Decision Tree
- CART model using `rpart`
- Class weighting to improve minority class performance
- Visualisation used to support interpretability and feature importance analysis

---

## Evaluation

Models were evaluated on a test set (80/20 split) using:
- Confusion matrices
- Precision, recall, and F1 scores for both Good and Bad listings

---

## Key Insights

- Logistic regression provided more stable and balanced predictive performance.
- Decision trees offered greater interpretability, highlighting the influence of price, amenities, and neighbourhood-related variables.
- Host behaviour, listing capacity, and availability play a significant role in determining listing success.
