import streamlit as st
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
# Get the current working directory
current_directory = os.getcwd()

# Specify the file name
file_name = "SeoulBikeData(Sujit).csv"

# Create the full file path by joining the directory and file name
file_path = os.path.join(current_directory, file_name)

# Check if the file exists at the specified path
if os.path.exists(file_path):
    # If the file exists, read it using pandas
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
else:
    st.write("CSV file not found at the specified path:", file_path)

# Assuming 'Date' column contains date strings
df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format="%m/%d/%Y")

# Drop rows with invalid dates
df.dropna(subset=['Date'], inplace=True)

df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['month_name'] = df['Date'].dt.month_name()
df['day_name'] = df['Date'].dt.day_name()
df.drop(['Date'], axis=1, inplace=True)

df['day_or_night'] = df['Hour'].apply(lambda x: 'day' if (20 > x > 5) else 'night')
df.drop(['month', 'day'], axis=1, inplace=True)

numeric_features = df.describe().columns

# Select only numeric columns for correlation calculation
numeric_df = df.select_dtypes(include=[np.number])

# Calculate correlation
correlation = numeric_df.corr()


# Plot histograms for numeric columns
df.hist(bins=50, figsize=(20, 15))

# Calculate VIF (Variance Inflation Factor)
def cal_vif(X):
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif

column_list = numeric_features[1:]
X = df[column_list]
vif_df = cal_vif(X)
# Drop 'Humidity(%)' column
df.drop(['Humidity(%)'], axis=1, inplace=True)
df_copy=df.copy()
# Calculate VIF again for the updated DataFrame
cal_vif(df[[i for i in df.describe().columns if i in column_list]])

numeric_features = ['Hour', 'Wind speed (m/s)', 'Visibility (10m)', 'Dew point temperature(°C)',
                    'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)']

# Load your DataFrame here
# df = pd.read_csv("Bike.csv", encoding='ISO-8859-1')

# Select only numeric columns
numeric_df = df.select_dtypes(include=np.number)

# Calculate the correlation matrix
correlation_matrix = numeric_df.corr()

# Create a DataFrame with correlations to 'Rented Bike Count'
corr_df = correlation_matrix['Rented Bike Count'].reset_index()

# Rename columns
corr_df.columns = ['variables', 'correlation']

# Sort by correlation in descending order
corr_df = corr_df.sort_values(by='correlation', ascending=False)

charactorical_features = df.describe(include='object').columns

# Label Encoding
df = df.replace({'Holiday': {'Holiday': 1, 'No Holiday': 0}, 'Functioning Day': {'Yes': 1, 'No': 0},
                 'day_or_night': {'day': 1, 'night': 0}})

# One Hot Encoding
df = pd.get_dummies(df, columns=['Seasons', 'month_name'], prefix='', prefix_sep='')

df.drop(['day_name'], axis=1, inplace=True)

dependend_variable = 'Rented Bike Count'

independend_variable = list(set(df.describe(include='all').columns.tolist()) - {dependend_variable})

# Checking skewness of the dependent variable
original_skewness = df[dependend_variable].skew()
log_transformed_skewness = np.log(df['Rented Bike Count']).skew()
sqrt_transformed_skewness = np.sqrt(df['Rented Bike Count']).skew()
# Splitting dataset into training and test set
training_df, testing_df = train_test_split(df, train_size=0.7, test_size=0.3, random_state=42)

# Transforming the dataset
col_to_scale = ['Wind speed (m/s)', 'Hour', 'Snowfall (cm)', 'Temperature(°C)', 'Solar Radiation (MJ/m2)',
                'Rainfall(mm)', 'Visibility (10m)']

scaler = MinMaxScaler()
training_df[col_to_scale] = scaler.fit_transform(training_df[col_to_scale])
testing_df[col_to_scale] = scaler.transform(testing_df[col_to_scale])

training_df['Rented Bike Count'] = np.sqrt(training_df[dependend_variable])
testing_df['Rented Bike Count'] = np.sqrt(testing_df[dependend_variable])

x_train = training_df.drop(['Rented Bike Count'], axis=1)
y_train = training_df['Rented Bike Count']
x_test = testing_df.drop(['Rented Bike Count'], axis=1)
y_test = testing_df['Rented Bike Count']

regressor = LinearRegression()
rfe = RFE(regressor, n_features_to_select=13)
rfe.fit(x_train, y_train)
y_predd = rfe.predict(x_train)
final = x_train.columns[rfe.support_]
feature_not_use = x_train.columns[~rfe.support_]
rfe_x_train = x_train[final]

# Exclude non-numeric columns from the feature matrix
numeric_cols = rfe_x_train.select_dtypes(include=[np.number])

# Add a constant term to the feature matrix
rfe_x_train_const = sm.add_constant(numeric_cols)

# Fit the OLS (Ordinary Least Squares) linear regression model
ols_model = sm.OLS(y_train, rfe_x_train_const).fit()

# Print the summary
print(ols_model.summary())

# Convert boolean columns to numeric (True to 1, False to 0)
rfe_x_train['Autumn'] = rfe_x_train['Autumn'].astype(int)
rfe_x_train['Winter'] = rfe_x_train['Winter'].astype(int)
rfe_x_train['February'] = rfe_x_train['February'].astype(int)
rfe_x_train['May'] = rfe_x_train['May'].astype(int)

# Add a constant term to the feature matrix
final_x_traindf = sm.add_constant(rfe_x_train)

# Fit the OLS (Ordinary Least Squares) linear regression model
ols_model = sm.OLS(y_train, final_x_traindf).fit()

# Print the summary
print(ols_model.summary())

# Defining a function to train the input model and print evaluation matrix
def analyse_model(model, X_train=None, X_test=None, y_train=None, y_test=None):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    a, p = y_test ** 2, y_pred ** 2

    # Calculating Evaluation Matrix
    mse = mean_squared_error(a, p)
    rmse = np.sqrt(mse)
    r2 = r2_score(a, p)

    # checking most imp features in predicting dependend variable
    try:
        features = X_train.columns
        importances = model.feature_importances_
        indices = np.argsort(importances)
        indices = indices[::-1]
    except:
        features = X_train.columns
        importances = np.abs(model.coef_)
        indices = np.argsort(importances)
        indices = indices[::-1]

    # Printing Evaluation Matrix
    print("MSE         :", mse)
    print("RMSE        :", rmse)
    print("MAE         :", mean_absolute_error(a, p))
    print("Train R2    :", r2_score(y_train ** 2, model.predict(X_train) ** 2))
    print("Test R2     :", r2)
    print("Adjusted R2 :", 1 - (1 - r2) * ((len(X_test) - 1) / (len(X_test) - X_test.shape[1] - 1)))

    # Plotting actual and predicted values
    plt.figure(figsize=(22, 6))
    plt.subplot(2, 1, 1)
    plt.plot((y_pred)[:100])
    plt.plot((np.array(y_test)[:100]))
    plt.legend(["Predicted", "Actual"])
    plt.title('Actual and Predicted Bike Counts')

    # ploting global explainability
    plt.subplot(2, 1, 2)
    plt.title('Feature Importance')
    plt.bar(range(len(indices)), importances[indices], color='orange', align='center')
    plt.xticks(range(len(indices)), [features[i] for i in indices], rotation=30)
    plt.tight_layout()
    plt.show()

    return model

final_x_traindf.drop(columns=['const'], inplace=True)

# Save final_x_traindf as CSV
final_x_traindf.to_csv('final_x_traindf.csv', index=False)

x_test_df = x_test[final_x_traindf.columns]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
analyse_model(regressor, X_train=final_x_traindf, X_test=x_test_df, y_train=y_train, y_test=y_test)
# Assuming 'new_df' is your DataFrame
bike_df=df_copy
new_df = df.copy()

new_df.head()

new_df.to_csv('new_df.csv', index=False)
df.head()
# ... (Continue with your code)

# Create dummy variables for categorical features
feature1 = pd.get_dummies(bike_df['Seasons'], drop_first=True)
feature2 = pd.get_dummies(bike_df['Holiday'], drop_first=True)
feature3 = pd.get_dummies(bike_df['Functioning Day'], drop_first=True)
feature4 = pd.get_dummies(bike_df['day_name'], drop_first=True)
feature5 = pd.get_dummies(bike_df['month_name'], drop_first=True)
feature6 = pd.get_dummies(bike_df['day_or_night'], drop_first=True)

# Joining two dataframes
rf_bike_df = pd.concat([bike_df, feature1, feature2, feature3, feature4, feature5, feature6], axis=1)

# Dropping columns for which we made dummy variables
rf_bike_df.drop(['Seasons', 'Holiday', 'Functioning Day', 'day_name', 'month_name', 'day_or_night'], axis=1, inplace=True)

# Dependent variable
label = 'Rented Bike Count'

# Extracting features
features = list(set(rf_bike_df.columns.tolist()) - {label})

# Create the data of dependent variables
x = rf_bike_df[features]

# Create the data of independent variables
y = np.sqrt(rf_bike_df[label])

# Splitting data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
# Splitting data into train and test (if not already split)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Define the parameter grid for GridSearchCV
parameter = {'max_depth': np.arange(8, 20, 3), 'min_samples_leaf': np.arange(5, 15, 3)}

# Create a Decision Tree Regressor with GridSearchCV
dt_regressor = GridSearchCV(DecisionTreeRegressor(), param_grid=parameter, scoring='neg_mean_squared_error', cv=5)
dt_regressor.fit(x_train, y_train)

# Get the best parameters from the grid search
best_params = dt_regressor.best_params_


# Create a Decision Tree Regressor with the best parameters
dt_regressor = DecisionTreeRegressor(max_depth=best_params['max_depth'], min_samples_leaf=best_params['min_samples_leaf'], random_state=0)

# Analyze the Decision Tree model
analyse_model(dt_regressor, X_train=x_train, X_test=x_test, y_train=np.sqrt(y_train), y_test=np.sqrt(y_test))
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

# Define your updated features list
features = ['Hour', 'Temperature(°C)', 'Wind speed (m/s)', 'Visibility (10m)', 'Dew point temperature(°C)', 'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)', 'Holiday', 'Functioning Day', 'year', 'day_or_night', 'Autumn', 'Spring', 'Summer', 'Winter', 'April', 'August', 'December', 'February', 'January', 'July', 'June', 'March', 'May', 'November', 'October', 'September']

# Define your label
label = 'Rented Bike Count'

# Create the data of dependent and independent variables
x = new_df[features]
y = np.sqrt(new_df[label])  # Assuming you want to use the square root of the label

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split  # Import train_test_split

# Define features and labels
features = ['Hour', 'Temperature(°C)', 'Wind speed (m/s)', 'Visibility (10m)', 'Dew point temperature(°C)',
            'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)', 'Holiday', 'Functioning Day', 'year',
            'day_or_night', 'Autumn', 'Spring', 'Summer', 'Winter', 'April', 'August', 'December', 'February',
            'January', 'July', 'June', 'March', 'May', 'November', 'October', 'September']
label = 'Rented Bike Count'

# Create a Streamlit sidebar with input widgets
st.sidebar.markdown('<p class="centered-subheader">Input Parameters</p>', unsafe_allow_html=True)

hour = st.sidebar.number_input("Hour", value=0)
temperature = st.sidebar.number_input("Temperature(°C)", value=-5.2)
wind_speed = st.sidebar.number_input("Wind speed (m/s)", value=2.2)
visibility = st.sidebar.number_input("Visibility (10m)", value=2000.0)
dew_point_temperature = st.sidebar.number_input("Dew point temperature(°C)", value=-17.6)
solar_radiation = st.sidebar.number_input("Solar Radiation (MJ/m2)", value=0.0)
rainfall = st.sidebar.number_input("Rainfall(mm)", value=0.0)
snowfall = st.sidebar.number_input("Snowfall (cm)", value=0)
holiday = st.sidebar.checkbox("Holiday")
functioning_day = st.sidebar.checkbox("Functioning Day")
year = st.sidebar.number_input("Year", value=2017)
day_or_night = st.sidebar.radio("Day or Night", ["Day", "Night"])
season = st.sidebar.radio("Season", ["Spring", "Summer", "Autumn", "Winter"])
month = st.sidebar.selectbox("Month",
                             ["January", "February", "March", "April", "May", "June", "July", "August", "September",
                              "October", "November", "December"])

# Map day_or_night and season to binary variables
day_or_night = 1 if day_or_night == "Night" else 0
seasons = ["Spring", "Summer", "Autumn", "Winter"]
season = [1 if season == s else 0 for s in seasons]

# Create a DataFrame with the input data
new_data = pd.DataFrame({
    'Hour': [hour],
    'Temperature(°C)': [temperature],
    'Wind speed (m/s)': [wind_speed],
    'Visibility (10m)': [visibility],
    'Dew point temperature(°C)': [dew_point_temperature],
    'Solar Radiation (MJ/m2)': [solar_radiation],
    'Rainfall(mm)': [rainfall],
    'Snowfall (cm)': [snowfall],
    'Holiday': [1 if holiday else 0],
    'Functioning Day': [1 if functioning_day else 0],
    'year': [year],
    'day_or_night': [day_or_night],
    'Autumn': [season[2]],
    'Spring': [season[0]],
    'Summer': [season[1]],
    'Winter': [season[3]],
    'April': [1 if month == "April" else 0],
    'August': [1 if month == "August" else 0],
    'December': [1 if month == "December" else 0],
    'February': [1 if month == "February" else 0],
    'January': [1 if month == "January" else 0],
    'July': [1 if month == "July" else 0],
    'June': [1 if month == "June" else 0],
    'March': [1 if month == "March" else 0],
    'May': [1 if month == "May" else 0],
    'November': [1 if month == "November" else 0],
    'October': [1 if month == "October" else 0],
    'September': [1 if month == "September" else 0]
})

# Define a function to make predictions and display the result
def predict_bike_counts(model, model_name):
    y = np.sqrt(new_df[label])
    xtrain, xtest, ytrain, ytest = train_test_split(new_df[features], y, test_size=0.2, random_state=0)

    model.fit(xtrain, ytrain)
    predicted_counts = model.predict(new_data)
    predicted_counts_original_scale = np.round(predicted_counts ** 2).astype(int)

    import streamlit as st

    # Set the color to Twitch's brand color (Twitch purple)
    twitch_color = '#6441A4'  # Purple Twitch color

    # Define the desired text size (18px)
    text_size = '18px'

    # Use HTML and CSS to style the title with a box, the Twitch color, and a specific text size
    st.markdown(
        f"""
        <div style="background-color: {twitch_color}; padding: 10px; border-radius: 5px;">
            <h2 style="color: white; text-align: center; font-size: {text_size};">{model_name} Predicted Bike Counts (Original Scale): {predicted_counts_original_scale[0]}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )


# Custom CSS
custom_css = """
    <style>
        /* Lemon yellow subheaders with font size 18px */
        .lemon-yellow-subheader {
            color: lemonchiffon;
            font size: 18px;
        }

        /* US Air Force Blue subheaders with font size 16px */
        .us-air-force-blue-subheader {
            color: #00308F;
            font size: 16px;
        }

        /* Lipstick text with font size 15px */
        .lipstick-text {
            color: #AB0808;
            font size: 15px;
        }

        /* Centered subheader with a green background */
        .centered-subheader {
            text-align: center;
            background-color: green;
            color: lemonchiffon;
            padding: 10px;
        }
    </style>
"""
image = st.image("Bike (1).jpg", caption="Bike Sharing Usage Visualization", use_column_width=True)

st.markdown(custom_css, unsafe_allow_html=True)

# Title and Subheader
st.title("Bike Rental Prediction App")
st.markdown('<p class="lemon-yellow-subheader">Enter the parameters and get predictions for bike rental counts</p>',
            unsafe_allow_html=True)

# Create placeholders for predictions
dt_prediction_placeholder = st.empty()
rf_prediction_placeholder = st.empty()
xgb_prediction_placeholder = st.empty()

# Decision Tree
st.markdown('<p class="centered-subheader">Decision Tree Prediction:</p>', unsafe_allow_html=True)
if st.checkbox("Show Decision Tree Prediction"):
    decision_tree_regressor = DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, random_state=0)
    predict_bike_counts(decision_tree_regressor, "Decision Tree")
else:
    dt_prediction_placeholder.empty()

# Random Forest
st.markdown('<p class="centered-subheader">Random Forest Prediction:</p>', unsafe_allow_html=True)
if st.checkbox("Show Random Forest Prediction"):
    random_forest_regressor = RandomForestRegressor(n_estimators=100, max_depth=19, random_state=0)
    predict_bike_counts(random_forest_regressor, "Random Forest")
else:
    rf_prediction_placeholder.empty()


# XGBoost
st.markdown('<p class="centered-subheader">XGBoost Prediction:</p>', unsafe_allow_html=True)
if st.checkbox("Show XGBoost Prediction"):
    xgboost_regressor = XGBRegressor(n_estimators=120, max_depth=9, learning_rate=0.3, random_state=0)
    predict_bike_counts(xgboost_regressor, "XGBoost")
else:
    xgb_prediction_placeholder.empty()

