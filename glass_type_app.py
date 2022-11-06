# S2.1: Open Sublime text editor, create a new Python file, copy the following code in it and save it as 'glass_type_app.py'.
# You have already created this ML model in ones of the previous classes.

# Importing the necessary Python modules.
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Creating the SVC model 
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)

# Creating the Logistic Regression model 
rf_clf = RandomForestClassifier(n_jobs = -1, n_estimators = 100)
rf_clf.fit(X_train, y_train)

# Creating the Random Forest Classifier model 
log_reg = LogisticRegression(n_jobs = -1)
log_reg.fit(X_train, y_train)

# S3.1: Create a function that accepts an ML model object say 'model' and the nine features as inputs 
# and returns the glass type.
@st.cache()
def prediction(model, RI, Na, Mg, Al, Si, K, Ca, Ba, Fe):
  glass_type = model.predict([[RI, Na, Mg, Al, Si, K, Ca, Ba, Fe]])
  glass_type = glass_type[0]
  if glass_type == 1:
    return "building windows float processed"
  elif glass_type == 2:
    return "building windows non float processed"
  elif glass_type == 3:
    return "vehicle windows float processed"
  elif glass_type == 4:
    return "vehicle windows non float processed"
  elif glass_type == 5:
    return "containers"
  elif glass_type == 6:
    return "tableware"
  else:
    return "headlamp"

# S4.1: Add title on the main page and in the sidebar.
st.title("Glass Type Predictor")
st.sidebar.title("Exploratory Data Analysis:")

# S5.1: Using the 'if' statement, display raw data on the click of the checkbox.
if st.sidebar.checkbox("Show raw data"):
  st.subheader("Full Dataset")
  st.dataframe(glass_df)

# S6.1: Scatter Plot between the features and the target variable.
# Add a subheader in the sidebar with label "Scatter Plot".
st.sidebar.subheader("Scatter Plot")
# Choosing x-axis values for the scatter plot.
# Add a multiselect in the sidebar with the 'Select the x-axis values:' label
# and pass all the 9 features as a tuple i.e. ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe') as options.
# Store the current value of this widget in the 'features_list' variable.
features_list = st.sidebar.multiselect("Select the x-axis values:", ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))

# S6.2: Create scatter plots between the features and the target variable.
# Remove deprecation warning.
st.set_option('deprecation.showPyplotGlobalUse', False)
for feature in features_list:
  st.subheader(f"Scatter plot between {feature} and Glasstype")
  plt.figure(figsize=(15,8), dpi=96)
  sns.scatterplot(x=feature, y='GlassType', data=glass_df)
  st.pyplot()

# Add a subheader in the sidebar with label "Visualisation Selector"
st.sidebar.subheader("Visualisation Selector")
# Add a multiselect in the sidebar with label 'Select the Charts/Plots:'
# and with 6 options passed as a tuple ('Histogram', 'Box Plot', 'Count Plot', 'Pie Chart', 'Correlation Heatmap', 'Pair Plot').
# Store the current value of this widget in a variable 'plot_types'.
plot_types = st.sidebar.multiselect("Select the charts or plots:", ('Histogram', 'Box Plot', 'Count Plot', 'Pie Chart', 'Correlation Heatmap', 'Pair Plot'))

# S1.2: Create histograms for the selected features using the 'selectbox' widget.
if 'Histogram' in plot_types:
  hist_features = st.sidebar.multiselect("Select features to create histograms:", ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
  # Create histograms.
  for feature in hist_features:
    st.subheader(f"Histogram for {feature}")
    plt.figure(figsize = (12, 6))
    plt.hist(glass_df[feature], bins = 'sturges', edgecolor = 'black')
    st.pyplot()

# S1.3: Create box plots for the selected column using the 'selectbox' widget.
if 'Box Plot' in plot_types:
  box_plot_cols = st.sidebar.multiselect("Select the columns to create box plots:", ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType'))

  # Create box plots.
  for col in box_plot_cols:
    st.subheader(f"Box plot for {col}")
    plt.figure(figsize = (12, 2))
    sns.boxplot(glass_df[col])
    st.pyplot()

from enum import auto
# S1.4: Create count plot, pie chart, correlation heatmap and pair plot.
# Create count plot using the 'seaborn' module and the 'st.pyplot()' function.
if 'Count Plot' in plot_types:
  st.subheader("Count plot")
  sns.countplot(x='GlassType', data=glass_df)
  st.pyplot()
# Create pie chart using the 'matplotlib.pyplot' module and the 'st.pyplot()' function.   
if 'Pie Chart' in plot_types:
  st.subheader("Pie Chart")
  pie_data = glass_df["GlassType"].value_counts()
  plt.figure(figsize=(5, 5))
  plt.pie(pie_data, labels=pie_data.index, autopct= '%1.2f%%', startangle=30, explode = np.linspace(.06, .16, 6))
  st.pyplot()
# Display correlation heatmap using the 'seaborn' module and the 'st.pyplot()' function.
if 'Correlation Heatmap' in plot_types:
  st.subheader("Correlation Heatmap")
  plt.figure(figsize=(10, 6))
  ax = sns.heatmap(glass_df.corr(), annot=True)
  bottom, top = ax.get_ylim()
  ax.set_ylim(bottom+0.5, top-0.5)
  st.pyplot()
# Display pair plots using the the 'seaborn' module and the 'st.pyplot()' function. 
if 'Pair Plot' in plot_types:
    st.subheader("Pair Plots")
    plt.figure(figsize = (15, 15))
    sns.pairplot(glass_df)
    st.pyplot()

# S2.1: Add 9 slider widgets for accepting user input for 9 features.
st.sidebar.subheader("Select your values:")
ri = st.sidebar.slider("Input Ri", float(glass_df['RI'].min()), float(glass_df['RI'].max()))
na = st.sidebar.slider("Input Na", float(glass_df['Na'].min()), float(glass_df['Na'].max()))
mg = st.sidebar.slider("Input Mg", float(glass_df['Mg'].min()), float(glass_df['Mg'].max()))
al = st.sidebar.slider("Input Al", float(glass_df['Al'].min()), float(glass_df['Al'].max()))
si = st.sidebar.slider("Input Si", float(glass_df['Si'].min()), float(glass_df['Si'].max()))
k = st.sidebar.slider("Input K", float(glass_df['K'].min()), float(glass_df['K'].max()))
ca = st.sidebar.slider("Input Ca", float(glass_df['Ca'].min()), float(glass_df['Ca'].max()))
ba = st.sidebar.slider("Input Ba", float(glass_df['Ba'].min()), float(glass_df['Ba'].max()))
fe = st.sidebar.slider("Input Fe", float(glass_df['Fe'].min()), float(glass_df['Fe'].max()))

# S3.1: Add a subheader and multiselect widget.
# Add a subheader in the sidebar with label "Choose Classifier"
st.sidebar.subheader("Choose Classifier")

# Add a selectbox in the sidebar with label 'Classifier'.
# and with 2 options passed as a tuple ('Support Vector Machine', 'Random Forest Classifier').
# Store the current value of this slider in a variable 'classifier'.

classifier = st.sidebar.selectbox("Classifier", 
                                 ('Support Vector Machine', 'Random Forest Classifier', 'Logistic Regression'))


# S4.1: Implement SVM with hyperparameter tuning
# if classifier == 'Support Vector Machine', ask user to input the values of 'C','kernel' and 'gamma'.
if classifier == 'Support Vector Machine':
    st.sidebar.subheader("Model Hyperparameters")
    c_value = st.sidebar.number_input("C (Error Rate)", 1, 100, step = 1)
    kernel_input = st.sidebar.radio("Kernel", ("linear", "rbf", "poly"))
    gamma_input = st. sidebar.number_input("Gamma", 1, 100, step = 1)

    # If the user clicks 'Classify' button, perform prediction and display accuracy score and confusion matrix.
    # This 'if' statement must be inside the above 'if' statement.
    if st.sidebar.button('Classify'):
        st.subheader("Support Vector Machine")
        svc_model = SVC(C = c_value, kernel = kernel_input, gamma = gamma_input)
        svc_model.fit(X_train,y_train)
        y_pred = svc_model.predict(X_test)
        accuracy = svc_model.score(X_test, y_test)
        glass_type = prediction(svc_model, ri, na, mg, al, si, k, ca, ba, fe)
        st.write("The Type of glass predicted is:", glass_type)
        st.write("Accuracy", accuracy.round(2))
        plot_confusion_matrix(svc_model, X_test, y_test)
        st.pyplot()

# S5.1: Implement Random Forest Classifier with hyperparameter tuning.
# if classifier == 'Random Forest Classifier', ask user to input the values of 'n_estimators' and 'max_depth'.
if classifier == 'Random Forest Classifier':
  st.sidebar.subheader("Model Hyperparameters")
  n_estimators_input = st.sidebar.number_input("Number of trees in the forest:", 100, 5000, step = 1)
  max_depth_input = st.sidebar.number_input("Maximum depth of the tree:", 1, 20, step = 1)

  # If the user clicks 'Classify' button, perform prediction and display accuracy score and confusion matrix.
  # This 'if' statement must be inside the above 'if' statement. 
  if st.sidebar.button('Classify'):
    st.subheader("Random Forest Classifier")
    rfc_model = RandomForestClassifier(n_estimators=n_estimators_input, max_depth=max_depth_input, n_jobs = -1)
    rfc_model.fit(X_train, y_train)
    y_pred = svc_model.predict(X_test)
    rfc_accuracy = rfc_model.score(X_test, y_test)
    glass_type = prediction(rfc_model, ri, na, mg, al, si, k, ca, ba, fe)
    st.write("The Type of glass predicted is:", glass_type)
    st.write("Accuracy", rfc_accuracy.round(2))
    plot_confusion_matrix(rfc_model, X_test, y_test)
    st.pyplot()

 # S1.1: Implement Logistic Regression with hyperparameter tuning
if classifier == 'Logistic Regression':
  st.sidebar.subheader("Model Hyperparameters")
  log_c_value = st.sidebar.number_input("C (Error Rate)", 1, 100, step = 1)
  max_iter_input = st.sidebar.slider("Maximum number of iterations:", 100, 500, step = 10)

  if st.sidebar.button("Classify"):
    st.subheader("Logistic Regression")
    log_reg = LogisticRegression(C=log_c_value, max_iter=max_iter_input)
    log_reg.fit(X_train, y_train)
    accuracy = log_reg.score(X_train, y_train)
    glass_type = prediction(log_reg, ri, na, mg, al, si, k, ca, ba, fe)
    st.write("The Type of glass predicted is:", glass_type)
    st.write("Accuracy", accuracy.round(2))
    plot_confusion_matrix(log_reg, X_test, y_test)
    st.pyplot()
