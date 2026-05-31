#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd


# In[9]:


df = pd.read_csv("Concrete_Data_Yeh.csv")


# Data Understanding

# In[10]:


df.head()


# In[11]:


df.sample(10)


# In[12]:


df.shape


# In[13]:


df.duplicated().sum()


# In[14]:


df = df.drop_duplicates()


# In[15]:


df.shape


# In[16]:


df.isnull().sum()


# In[17]:


df.describe()


# In[18]:


df.info()


# In[19]:


num_cols = df.select_dtypes(include=["int64","float64"]).drop(columns=["csMPa"]).columns
tar_col = df["csMPa"]


# EDA

# In[20]:


import matplotlib.pyplot as plt
import seaborn as sns


# Univariate Analysis

# num cols

# In[21]:


for col in num_cols:
    sns.histplot(x=col,data=df,kde=True)
    plt.title(col)
    plt.show()


# In[22]:


for col in num_cols:
    sns.boxplot(x=col,data=df)
    plt.title(col)
    plt.show()


# Target col

# In[23]:


sns.histplot(x=tar_col,data=df,kde=True)
plt.title("Target col histplot")
plt.show()
sns.boxplot(x=tar_col,data=df)
plt.title("Target col box plot")
plt.show()


# Bivariate Analysis

# In[24]:


for col in num_cols:
    sns.regplot(x=col,y=tar_col,data=df)
    plt.title(col)
    plt.show()


# Multivariate Analysis

# In[25]:


sns.heatmap(df.corr(),annot=True,cmap="Blues")


# Preprocessing

# In[26]:


from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error,r2_score
import optuna


# In[27]:


from outliers_handler import outliers_handling


# In[28]:


df.sample(1)


# In[29]:


x = df.drop(columns=["csMPa"])
y = df["csMPa"]


# In[65]:


x.columns


# In[30]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# In[31]:


num_pipeline = Pipeline(steps=[
    ("outliers_handling",outliers_handling()),
    ("scaling",StandardScaler())
])


# In[33]:


preprocessing = ColumnTransformer(transformers=[
    ("num_pipeline",num_pipeline,num_cols)
])


# In[34]:


def objective(trial):
    model_name = trial.suggest_categorical("regressor", ["lr", "dt", "rf", "xgb"])

    if model_name == "lr":
        model = LinearRegression()

    elif model_name == "dt":
        model = DecisionTreeRegressor(
        max_depth=trial.suggest_int("max_depth", 3, 20),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10)
    )

    elif model_name == "rf":
        model = RandomForestRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 500),
        max_depth=trial.suggest_int("max_depth", 5, 30),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        n_jobs=-1
    )

    elif model_name == "xgb":
        model = XGBRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 500),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        gamma=trial.suggest_float("gamma", 0, 5),
        reg_alpha=trial.suggest_float("reg_alpha", 0, 5),
        reg_lambda=trial.suggest_float("reg_lambda", 0, 5),
        n_jobs=-1,
        verbosity=0
    )

    pipe = Pipeline(steps=[
        ("preprocessing",preprocessing),
        ("model",model)
    ])
    score = cross_val_score(pipe,x_train,y_train,cv=5,scoring="neg_root_mean_squared_error")
    return score.mean()


# In[35]:


study = optuna.create_study(direction="maximize")
study.optimize(objective,n_trials=100)


# In[37]:


params = study.best_params
model_name = params["regressor"]

if model_name == "lr":
    final_model = LinearRegression()

elif model_name == "dt":
    final_model = DecisionTreeRegressor(
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"]
    )

elif model_name == "rf":
    final_model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        n_jobs=-1
    )

elif model_name == "xgb":
    final_model = XGBRegressor(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        gamma=params["gamma"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        n_jobs=-1,
        verbosity=0
    )


# In[38]:


final_pipeline = Pipeline(steps=[
    ("preprocessing",preprocessing),
    ("final_model",final_model)
])


# In[39]:


final_pipeline.fit(x_train,y_train)


# In[40]:


y_train_pred = final_pipeline.predict(x_train)
y_test_pred = final_pipeline.predict(x_test)


# In[41]:


train_rmse = root_mean_squared_error(y_train,y_train_pred)
print(f"Train RMSE: {train_rmse}")
test_rmse = root_mean_squared_error(y_test,y_test_pred)
print(f"Test RMSE: {test_rmse}")


# In[42]:


train_r2 = r2_score(y_train,y_train_pred)
print(f"Train r2: {train_r2}")
test_r2 = r2_score(y_test,y_test_pred)
print(f"Test r2: {test_r2}")


# In[43]:


train_nrmse_range = train_rmse/(y_train.max()-y_train.min())
print(f"Train NRMSE Range: {train_nrmse_range}")
test_nrmse_range = test_rmse/(y_test.max()-y_test.min())
print(f"Test NRMSE Range: {test_nrmse_range}")


# In[44]:


train_nrmse_mean = train_rmse/y_train.mean()
print(f"Train NRMSE mean: {train_nrmse_mean}")
test_nrmse_mean = test_rmse/y_test.mean()
print(f"Test NRMSE mean: {test_nrmse_mean}")


# In[45]:


train_nrmse_std = train_rmse/y_train.std()
print(f"Train NRMSE std: {train_nrmse_std}")
test_nrmse_std = test_rmse/y_test.std()
print(f"Test NRMSE std: {test_nrmse_std}")


# Model Explainability

# In[49]:


import shap


# In[46]:


shap_preprocessing = final_pipeline.named_steps["preprocessing"]
shap_model = final_pipeline.named_steps["final_model"]


# In[47]:


shap_feature_names = []
for col in shap_preprocessing.get_feature_names_out():
    shap_feature_names.append(col.split("__")[-1])


# In[58]:


x_train_t = pd.DataFrame(
    shap_preprocessing.transform(x_train),
    columns = shap_feature_names
)
x_test_t = pd.DataFrame(
    shap_preprocessing.transform(x_test),
    columns = shap_feature_names
)


# In[59]:


explainer = shap.TreeExplainer(shap_model)


# In[60]:


shap_values = explainer(x_test_t)


# In[61]:


shap.plots.beeswarm(shap_values)


# In[62]:


shap.plots.bar(shap_values)


# Saving Required data

# In[54]:


import joblib


# In[64]:


joblib.dump(final_pipeline,"final_pipeline.pkl")


# In[63]:


joblib.dump(shap_feature_names,"shap_feature_names.pkl")

