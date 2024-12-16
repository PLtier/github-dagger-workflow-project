import datetime
import joblib
import json
import numpy as np
import os
import pandas as pd
import warnings

from sklearn.preprocessing import MinMaxScaler

import utils


# Define min and max date
max_date = "2024-01-31"
min_date = "2024-01-01"

if not max_date:
    max_date = pd.to_datetime(datetime.datetime.now().date()).date()
else:
    max_date = pd.to_datetime(max_date).date()

min_date = pd.to_datetime(min_date).date()

# Create artifacts folder
os.makedirs("artifacts", exist_ok=True)

# Warnings and pandas settings
warnings.filterwarnings("ignore")
pd.set_option("display.float_format", lambda x: "%.3f" % x)

# Loading raw data
data = pd.read_csv("./artifacts/raw_data.csv")

# Time limit data
data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
data = data[(data["date_part"] >= min_date) & (data["date_part"] <= max_date)]

min_date = data["date_part"].min()
max_date = data["date_part"].max()

data = data.drop(
    [
        "is_active",
        "marketing_consent",
        "first_booking",
        "existing_customer",
        "last_seen",
    ],
    axis=1,
)

# Removing columns that will be added back after the EDA
data = data.drop(["domain", "country", "visited_learn_more_before_booking", "visited_faq"], axis=1)

data["lead_indicator"].replace("", np.nan, inplace=True)
data["lead_id"].replace("", np.nan, inplace=True)
data["customer_code"].replace("", np.nan, inplace=True)

data = data.dropna(axis=0, subset=["lead_indicator"])
data = data.dropna(axis=0, subset=["lead_id"])

data = data[data.source == "signup"]
result = data.lead_indicator.value_counts(normalize=True)

vars = [
    "lead_id",
    "lead_indicator",
    "customer_group",
    "onboarding",
    "source",
    "customer_code",
]

for col in vars:
    data[col] = data[col].astype("object")

cont_vars = data.loc[:, ((data.dtypes == "float64") | (data.dtypes == "int64"))]
cat_vars = data.loc[:, (data.dtypes == "object")]

cont_vars = cont_vars.apply(
    lambda x: x.clip(lower=(x.mean() - 2 * x.std()), upper=(x.mean() + 2 * x.std()))
)

# Continuous variables missing values
cont_vars = cont_vars.apply(utils.impute_missing_values)
cont_vars.apply(utils.describe_numeric_col).T

cat_vars.loc[cat_vars["customer_code"].isna(), "customer_code"] = "None"
cat_vars = cat_vars.apply(utils.impute_missing_values)
cat_vars.apply(lambda x: pd.Series([x.count(), x.isnull().sum()], index=["Count", "Missing"])).T

# Scaling continuous variables
scaler = MinMaxScaler()
scaler.fit(cont_vars)

cont_vars = pd.DataFrame(scaler.transform(cont_vars), columns=cont_vars.columns)
cont_vars = cont_vars.reset_index(drop=True)
cat_vars = cat_vars.reset_index(drop=True)

# Concatenating the categorical and continuous variables
data = pd.concat([cat_vars, cont_vars], axis=1)
data_columns = list(data.columns)

# Storing the final training data
data["bin_source"] = data["source"]
values_list = ["li", "organic", "signup", "fb"]
data.loc[~data["source"].isin(values_list), "bin_source"] = "Others"
mapping = {"li": "socials", "fb": "socials", "organic": "group1", "signup": "group1"}

data["bin_source"] = data["source"].map(mapping)

data.to_csv("./artifacts/train_data_gold.csv", index=False)
