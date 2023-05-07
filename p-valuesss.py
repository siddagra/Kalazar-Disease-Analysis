import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Read in data
data = pd.read_csv("C:\\Users\\sidda\\Downloads\\socio_eco_disease.csv")
data.columns = [col.strip() for col in data.columns]

# Replace 'MORE' with 15 in 'FAMILY SIZE'
data["FAMILY SIZE"] = data["FAMILY SIZE"].replace("MORE", 15)

# Define predictor, ordinal, categorical, and discrete integer variables
predictors = ['GENDER', 'EDUCATION', 'OCCUPATION', 'MONTHLY FAMILY INCOME', 'TYPE OF HOUSE', 'FAMILY SIZE', 'NO.OF BED ROOMS']
ordinal_variables = ['Overall attitude', 'EDUCATION']
categorical_variables = ['MONTHLY FAMILY INCOME', 'TYPE OF HOUSE', 'GENDER', 'OCCUPATION']
discrete_integer_variables = ["FAMILY SIZE", "NO.OF BED ROOMS"]
response = 'Overall attitude'


# Calculate correlation matrix
corr_matrix = data.corr()

# Plot heatmap of correlation matrix
sns.heatmap(corr_matrix, cmap="YlGnBu")

# Show plot
plt.show()

# Preprocess data

le = LabelEncoder()
scaler = StandardScaler()

for var in ordinal_variables:
    data[var] = le.fit_transform(data[var])

for var in categorical_variables:
    data[var] = le.fit_transform(data[var])

for var in discrete_integer_variables:
    data[var] = scaler.fit_transform(data[var].values.reshape(-1, 1))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[predictors], data[response], test_size=0.02, random_state=42)


# Get p-values for each predictor variable
import numpy as np
from scipy.stats import chi2

# Add intercept term to predictors
X_train = np.hstack((np.ones((len(X_train), 1)), X_train))
X_test = np.hstack((np.ones((len(X_test), 1)), X_test))


# Fit multinomial logistic regression model
clf = LogisticRegression(multi_class='multinomial', solver='newton-cg')
clf.fit(X_train, y_train)

import shap
import matplotlib.pyplot as plt

explainer = shap.Explainer(clf, X_train)
shap_values = explainer(X_test)

# Plot beeswarm plot
shap.plots.heatmap(shap_values, max_display=7, show=False)

# Add feature names
feature_names = ['Intercept'] + predictors
plt.xlabel("SHAP value")
plt.xticks(rotation=30)
plt.yticks(range(len(feature_names)), feature_names)
plt.tight_layout()
plt.show()


# Create explainer object
explainer = shap.KernelExplainer(clf.predict_proba, X_train)

# Calculate SHAP values for the test set
shap_values = explainer.shap_values(X_train)


# Create summary plot

shap.summary_plot(shap_values, X_train, feature_names=["intercept"] + predictors)
shap.plots.heatmap(shap_values)
shap.plots.beeswarm(shap_values)
clustering = shap.utils.hclust(X_train, y_train)
shap.plots.bar(shap_values, clustering=clustering, clustering_cutoff=0.8)


for j, var in enumerate(["intercept"] + predictors):
    for i in range(0,4):
        print(f"\\num[round-mode=places,round-precision=4]<{clf.coef_[i][j]}>", end=" & ")
    print("\\\\\n")



# Define function to fit logistic regression model
def fit_model(X, y):
    model = LogisticRegression(multi_class='multinomial', solver='newton-cg')
    model.fit(X, y)
    return model.coef_

# Define function to compute coefficient estimates
def get_coef(X, y):
    model = LogisticRegression(multi_class='multinomial', solver='newton-cg')
    model.fit(X, y)
    return model.coef_[0]

# Set number of bootstrap samples
n_bootstrap = 1000

# Initialize array to store coefficient estimates
coef_boot = np.zeros((n_bootstrap, len(predictors)+1))

# Generate bootstrap samples and compute coefficient estimates
for i in range(n_bootstrap):
    indices = np.random.choice(range(len(X_train)), size=len(X_train), replace=True)
    X_boot = X_train[indices]
    y_boot = y_train.iloc[indices]
    coef_boot[i] = get_coef(X_boot, y_boot)

# Compute p-values for each predictor
p_values = []
for i, var in enumerate(['Intercept'] + predictors):
    coef_var_boot = coef_boot[:, i]
    se_var_boot = np.std(coef_var_boot)
    z_var_boot = coef_var_boot.mean() / se_var_boot
    p_var_boot = chi2.sf(z_var_boot ** 2, 1)
    p_values.append(p_var_boot)

# Print p-values
for var, p in zip(['Intercept'] + predictors, p_values):
    print('{}: {}'.format(var, p))

