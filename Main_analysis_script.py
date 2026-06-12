import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import numpy as np


# Load your dataset
df = pd.read_csv('customerOrders.csv')

# Convert ord_datetime to datetime format
df['ord_datetime'] = pd.to_datetime(df['ord_datetime'])

# Group by day and count the number of orders
df['ord_date'] = df['ord_datetime'].dt.date  # Extract the date part
orders_per_day = df.groupby('ord_date').size().reset_index(name='order_count')

# Merge the orders_per_day back into the original DataFrame
df = df.merge(orders_per_day, on='ord_date', how='left')

# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = orders_per_day['order_count'].quantile(0.25)
Q3 = orders_per_day['order_count'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
orders_per_day_no_outliers = orders_per_day[(orders_per_day['order_count'] >= lower_bound) & (orders_per_day['order_count'] <= upper_bound)]

# Plotting
plt.figure(figsize=(12, 6))

plt.plot(
    orders_per_day_no_outliers['ord_date'], 
    orders_per_day_no_outliers['order_count'], 
    label='Total Sales', 
    color='green', 
    linestyle='-',       # Use a solid line
    linewidth=1.5,       # Increase the line width
    alpha=0.8            # Add slight transparency to the line
)

# Add titles and labels
plt.title('Number of Orders Per Day (Without Outliers)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.xticks(rotation=45)
plt.legend(fontsize=10)


plt.xlabel('Date')
plt.ylabel('Number of Orders')
plt.title('Number of Orders Per Day (Without Outliers)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig("chart.png")
print("Chart saved as 'chart.png'")

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import numpy as np


# Prepare data for regression
orders_per_day_no_outliers['ordinal_date'] = pd.to_datetime(orders_per_day_no_outliers['ord_date']).map(pd.Timestamp.toordinal)
X = orders_per_day_no_outliers[['ordinal_date']]  # Independent variable (date as ordinal)
y = orders_per_day_no_outliers['order_count']  # Dependent variable (order count)


# Fit model for Simple Linear Regression
model = LinearRegression()
model.fit(X, y)
print("Coefficient (β1):", model.coef_[0])
print("Intercept (β0):", model.intercept_)

# Statistical Significance
X_with_const = sm.add_constant(X)
model_sm = sm.OLS(y, X_with_const).fit()
print(model_sm.summary())

# Plot the regression line
plt.figure(figsize=(12, 6))

plt.scatter(
    orders_per_day_no_outliers['ord_date'], 
    y, 
    label='Actual Orders', 
    color='blue', 
    alpha=0.7,           # Make points slightly transparent
    edgecolors='black',  # Add a black edge to each point
    linewidths=0.5,      # Make edge lines thin
    s=50                 # Adjust marker size
)

# Generate predictions using the fitted model
predictions = model.predict(X)

plt.plot(
    orders_per_day_no_outliers['ord_date'],
    predictions,
    label='Regression Line',
    color='red',
)
plt.xlabel('Date')
plt.ylabel('Number of Orders')
plt.title('Linear Regression: Number of Orders Per Day')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("linear_regression_chart.png")
print("Chart saved as 'linear_regression_chart.png'")


