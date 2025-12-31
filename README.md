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

This approach aligns with the coursework requirement to construct a reproducible success metric grounded in business understanding rather than relying on a predefined label.

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

## Results Summary

Both models achieved comparable overall accuracy (~68%) on the held-out test set, with differences in class-level performance.

### Logistic Regression
- Accuracy: 68.2%
- Balanced accuracy: 63.6%
- F1 score (Good listings): 0.50
- F1 score (Bad listings): 0.77

The logistic regression model showed stronger performance in identifying **Bad** listings, reflecting higher precision and recall for the majority class.

### Decision Tree
- Accuracy: 67.9%
- Balanced accuracy: 64.9%
- F1 score (Good listings): 0.52
- F1 score (Bad listings): 0.76

The decision tree slightly improved recall for **Good** listings and offered interpretability through feature importance.

### Feature Importance (Decision Tree)
The most influential predictors included:
- Amenities count
- Multi-host indicator
- Superhost status
- Maximum and minimum nights
- Price

## Key Insights

- **Amenities are the strongest driver of listing quality.**  
  Both models highlight amenities_count as a key predictor, and the decision tree ranks it as the most important feature. Listings offering more amenities are substantially more likely to be classified as “Good”.

- **Host reputation matters.**  
  Superhost status and host identity verification are strongly associated with higher-quality listings, indicating that trust and host commitment play a critical role in guest satisfaction.

- **Pricing and stay-length policies influence performance.**  
  Higher prices, very short minimum stays, and very long maximum stays are negatively associated with listing quality, highlighting the importance of balanced pricing and booking flexibility.

- **Property and room characteristics affect outcomes.**  
  Entire homes and private properties generally perform better than shared spaces, while listings that accommodate more guests show slightly higher chances of being classified as “Good”.

- **Location effects are significant.**  
  Certain neighbourhoods (e.g. Westminster, Camden, Southwark, City of London) consistently increase the likelihood of higher-quality listings, confirming the importance of geographic desirability.

