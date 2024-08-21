# Instacart Market Basket Analysis

**Dataset Reference:** [Kaggle: Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis)

## Project Overview

Instacart, a grocery ordering and delivery app, is designed to simplify the process of filling your fridge and pantry with your favorite items. After selecting products through the Instacart app, personal shoppers take care of the in-store shopping and delivery. This project explores the shopping patterns within Instacart's dataset to predict which products customers are likely to purchase again.

## Business Objective

### Predicting Repeat Purchases

**Objective:** Use historical customer order data to predict which previously purchased products will be included in a user’s next order.

## Dataset Description

The dataset is a relational set of files that details customers' orders over time. It is anonymized and includes over 3 million grocery orders from more than 200,000 Instacart users. Each user has between 4 and 100 orders, with details such as the sequence of purchased products, the week and hour of the order, and the time elapsed between orders.

### Dataset Components

- **Departments (21 rows):**
  - `department_id`: Unique identifier for each department
  - `department`: Name of the department

- **Aisles (134 rows):**
  - `aisle_id`: Unique identifier for each aisle
  - `aisle`: Name of the aisle

- **Products (50k rows):**
  - `product_id`: Unique identifier for each product
  - `product_name`: Name of the product
  - `aisle_id`: Foreign key referencing `aisle_id`
  - `department_id`: Foreign key referencing `department_id`

- **Orders (3.4m rows, 206k users):**
  - `order_id`: Unique identifier for each order
  - `user_id`: Unique identifier for each customer
  - `eval_set`: Specifies the dataset (e.g., "prior", "train", "test")
  - `order_number`: The order sequence number for each user
  - `order_dow`: Day of the week when the order was placed
  - `order_hour_of_day`: Hour of the day when the order was placed
  - `days_since_prior`: Days since the last order

- **Order Products (30m+ rows):**
  - `order_id`: Foreign key referencing `order_id`
  - `product_id`: Foreign key referencing `product_id`
  - `add_to_cart_order`: The order in which each product was added to the cart
  - `reordered`: Indicates if the product was reordered (1 = Yes, 0 = No)

## Prerequisites

### Tools and Environment

- **Jupyter Notebook:** Recommended via Anaconda-Navigator or any IDE supporting Jupyter Notebooks.
- **Python Version:** 3.11.5

### Essential Libraries

```text
matplotlib 3.7.2
seaborn 0.12.2
pandas 2.0.3
numpy 1.21.5
sklearn 1.0.2
```

## Exploratory Data Analysis

### Order Frequency

A histogram analysis shows that most users placed at least 5 orders, with a maximum of 100 orders. Instacart appears to have sampled users who made at least 5 orders, capping the maximum at 100.

![Total orders by each user](/images/orders_by_users_histo.png)

### Most Ordered and Reordered Products

Analysis reveals that bananas are the most ordered item, followed by organic bananas and organic strawberries. Notably, 11 out of the top 15 products are organic, indicating a preference for organic products among Instacart users. The reordered products largely mirror the most ordered products, suggesting customer satisfaction.

![Top 15 popular orders](/images/top_most_popular_products.png)
![Top 15 popular reorders](/images/top_most_reordered_products.png)

### Order Timing

The majority of orders are placed between 9 AM and 5 PM, as shown in the pie chart.

![Orders by hour](/images/orders_by_hr.png)

### Order Composition

- **Orders with Reordered Products:** 88.13%
- **Orders with No Reordered Products:** 11.87%

### Product Distribution by Department and Aisle

Treemap analysis shows that the top three departments by product share are Personal Care, Snacks, and Pantry.

![Product Treemap](/images/product_treemap.png)

## Data Preparation

### Feature Engineering

To predict which products will be reordered in the next order, we engineered the `in_cart` feature to identify products reordered in the previous order.

**Product Features:**

- `product_num_unique_orders`: Number of unique orders in which each product appears.
- `product_avg_add_to_cart_position`: Average position at which each product is added to the cart.

**User Features:**

- `user_num_unique_orders`: Number of distinct orders placed by each user.
- `user_avg_products_per_order`: Average number of products a user adds to their cart per order.
- `user_num_unique_products`: Variety of products ordered by each user.
- `user_avg_days_between_orders`: Average days between orders for each user.

**User-Product Features:**

- `user_product_avg_add_to_cart_order`: Average position at which a user adds a specific product to their cart.

**Department Features:**

- For simplicity, only department-level features were used to avoid feature explosion.

## Final Dataset

The final dataset includes 29 features and 1 target variable, with irrelevant columns like `user_id`, `product_id`, and `latest_cart` removed.

### Reducing Dataset Size

To expedite training, the dataset size was reduced to 2.5%, resulting in 211,867 rows and 29 features.

### Train-Test Split

- **Test Size:** 25%
- **Random State:** 25 for reproducibility
- **Stratification:** Applied to maintain class distribution.

## Model Building

Given the class imbalance (9.7% 'yes' class), accuracy alone is not a reliable metric. The focus is on achieving a higher F1 score by balancing recall and precision. Initial models with default hyperparameters were tested, followed by hyperparameter tuning using GridSearchCV with 5-fold cross-validation.

### Model Performance

Final model performance and analysis to be detailed after testing and validation.

| Model                  | Train Time (s) | Train Accuracy (%) | Test Accuracy (%) | Train Recall (%) | Test Recall (%) | Train Precision (%) | Test Precision (%) | Train F1 (%) | Test F1 (%) | Train ROC AUC (%) | Test ROC AUC (%) | Train Log Loss   | Test Log Loss   |
|------------------------|----------------|--------------------|-------------------|------------------|-----------------|---------------------|--------------------|--------------|-------------|-------------------|------------------|------------------|-----------------|
| Logistic               | 2.61           | 90.26               | 90.42             | 7.14             | 8.08            | 55.48               | 61.57              | 12.65        | 14.28       | 53.26             | 53.76            | 3.509457         | 3.451766        |
| KNN                    | 0.21           | 91.32               | 89.59             | 21.99            | 12.16           | 69.07               | 40.85              | 33.36        | 18.74       | 60.46             | 55.12            | 3.127315         | 3.753732        |
| DecisionTree           | 2.56           | 100.00              | 83.74             | 100.00           | 22.72           | 100.00              | 20.65              | 100.00       | 21.64       | 100.00            | 56.58            | 0.000000         | 5.859837        |
| RandomForest           | 6.47           | 99.99               | 90.41             | 99.95            | 9.41            | 100.00              | 59.25              | 99.97        | 16.25       | 99.97             | 54.35            | 0.001914         | 3.455169        |
| GradientBoosting       | 29.33          | 90.70               | 90.49             | 12.96            | 11.85           | 64.37               | 59.40              | 21.58        | 19.76       | 56.09             | 55.48            | 3.353580         | 3.426248        |
| AdaBoost               | 2.64           | 100.00              | 83.82             | 100.00           | 23.23           | 100.00              | 21.07              | 100.00       | 22.09       | 100.00            | 56.84            | 0.000000         | 5.830916        |
| GaussianNB             | 0.26           | 69.31               | 69.30             | 59.95            | 58.73           | 18.13               | 17.89              | 27.84        | 27.43       | 65.14             | 64.60            | 11.062990        | 11.063855       |
| Logistic CV F1         | 1.59           | 90.26               | 90.42             | 7.14             | 8.08            | 55.48               | 61.57              | 12.65        | 14.28       | 53.26             | 53.76            | 3.509457         | 3.451766        |
| KNN CV F1              | 23.35          | 100.00              | 88.63             | 100.00           | 15.60           | 100.00              | 33.71              | 100.00       | 21.33       | 100.00            | 56.12            | 0.000000         | 4.096527        |
| DecisionTree CV F1     | 2.74           | 98.31               | 85.25             | 83.66            | 21.22           | 99.11               | 23.12              | 90.73        | 22.13       | 91.79             | 56.74            | 0.608408         | 5.316299        |
| RandomForest CV F1     | 21.77          | 99.99               | 90.38             | 99.97            | 11.73           | 100.00              | 56.24              | 99.99        | 19.41       | 99.99             | 55.37            | 0.001063         | 3.467077        |
| GradientBoosting CV F1 | 76.70          | 93.97               | 89.91             | 40.24            | 14.03           | 96.95               | 46.40              | 56.88        | 21.54       | 70.05             | 56.13            | 2.172278         | 3.637199        |
| AdaBoost CV F1         | 73.12          | 95.28               | 89.21             | 56.93            | 16.08           | 92.34               | 38.81              | 70.44        | 22.74       | 78.21             | 56.65            | 1.701033         | 3.889829        |

#### ROC Curve

![ROC Curve](/images/roc_curve.png)

#### Model ROC AUC Scores

# Model ROC AUC Scores

| Model                  | ROC AUC Score |
|------------------------|---------------|
| Logistic               | 0.770101      |
| KNN                    | 0.664311      |
| DecisionTree           | 0.565766      |
| RandomForest           | 0.769018      |
| GradientBoosting       | 0.791920      |
| AdaBoost               | 0.568445      |
| GaussianNB             | 0.696612      |
| Logistic CV F1         | 0.770101      |
| KNN CV F1              | 0.634780      |
| DecisionTree CV F1     | 0.548056      |
| RandomForest CV F1     | 0.765705      |
| GradientBoosting CV F1 | 0.770662      |
| AdaBoost CV F1         | 0.724107      |

![ROC AUC](/images/ROC_AUC.png)

Below are the performance metrics presented in bar plots for all the models:

### Models Analysis

![Training time](/images/models_comp_by_train_time.png)
![Accuracy](/images/models_comp_by_test_accuracy.png)
![F1 Score](/images/models_comp_by_test_f1.png)

**Overfitting Models:**
Several models, including DecisionTree, AdaBoost, and KNN CV F1, show signs of significant overfitting. These models achieve 100% accuracy on the training data, with perfect scores in recall, precision, and F1 metrics. However, their test performance drops considerably, indicating that they may not generalize well to unseen data. For example, DecisionTree and AdaBoost both have test accuracies around 83-85%, with Test Recall and Test Precision dropping drastically.

**Balanced Performance:**
`RandomForest`, `GradientBoosting`, and `Logistic` models demonstrate more balanced performance. The `RandomForest` and `GradientBoosting` models, in particular, maintain high training accuracy (~99-100%) while showing reasonable test accuracy (~90%) and more balanced recall and precision scores. `GradientBoosting`, both with and without cross-validation, shows the highest ROC AUC scores (0.7919 and 0.7707), suggesting these models are better at distinguishing between classes in unseen data.

**Underperforming Model:**
GaussianNB has the lowest Train Accuracy (69.31%) and a significant drop in Test Precision (17.89%) and F1 scores. This indicates the model struggles to accurately predict the positive class and performs poorly on the training set, resulting in lower overall generalization.

### Best Models

Considering all metrics, `GradientBoosting CV F1` and `RandomForest CV F1` emerge as the best models. They offer a strong balance between high accuracy, precision, and recall, with robust test performance, suggesting they generalize well to new data.

## Conclusion

Although these  models are very close in Test accuracy and ROC AUC score, `GradientBoosting CV F1` has a higher F1 score enough though this takes more time to train.

**GradientBoosting CV F1 confusion matrix**

![GradientBoosting CV F1](/images/confusion_matrix_GradientBoosting_CV_F1.png)

We need to find a way to reduce the training time of the model without sacrificing the performance.

### Feature Importance

The feature importance analysis shows that `ordered_count` is the most important feature, followed by user-related features like `user_num_unique_orders` and `user_avg_days_between_orders`. Product-related features like `product_num_unique_orders` and `product_avg_add_to_cart_position` also play a significant role in predicting repeat purchases.
While department features have lower importance, they still contribute to the model's predictive power like `dairy eggs`, `snacks`, and `frozen`.

| Feature                              | Importance  |
|--------------------------------------|-------------|
| ordered_count                        | 0.268590    |
| user_num_unique_orders               | 0.136885    |
| user_avg_days_between_orders         | 0.110880    |
| user_avg_products_per_order          | 0.098324    |
| product_num_unique_orders            | 0.096149    |
| product_avg_add_to_cart_position     | 0.094577    |
| user_num_unique_products             | 0.077493    |
| user_product_avg_add_to_cart_order   | 0.070627    |
| dept_dairy eggs                      | 0.005503    |
| dept_snacks                          | 0.003856    |
| dept_frozen                          | 0.003561    |
| dept_produce                         | 0.003558    |
| dept_bakery                          | 0.003542    |
| dept_beverages                       | 0.003361    |
| dept_deli                            | 0.003204    |
| dept_pantry                          | 0.002540    |
| dept_breakfast                       | 0.002388    |
| dept_meat seafood                    | 0.002308    |
| dept_canned goods                    | 0.001939    |
| dept_dry goods pasta                 | 0.001919    |
| dept_household                       | 0.001753    |
| dept_missing                         | 0.001518    |
| dept_personal care                   | 0.001095    |
| dept_pets                            | 0.001003    |
| dept_alcohol                         | 0.001001    |
| dept_international                   | 0.000856    |
| dept_babies                          | 0.000758    |
| dept_bulk                            | 0.000497    |
| dept_other                           | 0.000313    |

![Feature Importance](/images/feature_importances.png)

## Next Steps

1. **Feature Engineering:** Continue to explore and engineer new features to improve model performance and predictive power.
2. **Model Optimization:** Experiment with different models, hyperparameters, and ensemble methods to enhance model performance and reduce training time.
3. **Speeding Up Training:** Use techniques like feature selection to reduce the number of features or reduce the size of the training set further for faster training times, especially when using GridSearchCV.
4. **Model Evaluation:** Evaluate the model on the full dataset to ensure it generalizes well to unseen data.
5. **Neural Network:** Try using neural networks to see if they can improve the model performance.
6. **Deployment:** Deploy the model for real-time predictions.
7. **Feedback Loop:** Implement a feedback loop to continuously improve the model based on new data and user feedback.