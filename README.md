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
