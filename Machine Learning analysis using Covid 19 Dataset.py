#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# In[2]:


# Load cleaned dataset
file_path = "covid_19_cleaned.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip().str.lower()


# In[4]:


# Feature engineering
df["cases_per_100k"] = np.where(df["population"] > 0, (df["cases"] / df["population"]) * 100000, np.nan)
df["deaths_per_100k"] = np.where(df["population"] > 0, (df["deaths"] / df["population"]) * 100000, np.nan)
df["tests_per_100k"] = np.where(df["population"] > 0, (df["tests"] / df["population"]) * 100000, np.nan)
df["recovery_rate"] = np.where(df["cases"] > 0, (df["recovered"] / df["cases"]) * 100, np.nan)
df["death_rate"] = np.where(df["cases"] > 0, (df["deaths"] / df["cases"]) * 100, np.nan)

df.replace([np.inf, -np.inf], np.nan, inplace=True)


# In[5]:


# -----------------------------
# 1) Regression model: predict deaths
# -----------------------------
reg_features = ["cases", "recovered", "tests", "population", "cases_per_100k", "tests_per_100k"]
X_reg = df[reg_features]
y_reg = df["deaths"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.25, random_state=42
)

lin_reg = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

rf_reg = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestRegressor(n_estimators=300, random_state=42))
])

lin_reg.fit(X_train_reg, y_train_reg)
rf_reg.fit(X_train_reg, y_train_reg)

lin_pred = lin_reg.predict(X_test_reg)
rf_pred = rf_reg.predict(X_test_reg)

print("Regression Results")
print("Linear Regression MAE:", mean_absolute_error(y_test_reg, lin_pred))
print("Linear Regression R²:", r2_score(y_test_reg, lin_pred))
print("Random Forest Regressor MAE:", mean_absolute_error(y_test_reg, rf_pred))
print("Random Forest Regressor R²:", r2_score(y_test_reg, rf_pred))


# In[6]:


# Random forest regression feature importance
rf_model = rf_reg.named_steps["model"]
rf_reg_importance = pd.DataFrame({
    "feature": reg_features,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nRegression Feature Importance")
print(rf_reg_importance)


# In[7]:


# -----------------------------
# 2) Classification model: classify high-risk countries
#    High-risk = death_rate above median
# -----------------------------
median_death_rate = df["death_rate"].median(skipna=True)
df["high_risk"] = (df["death_rate"] > median_death_rate).astype(int)

clf_features = ["cases", "recovered", "tests", "population", "cases_per_100k", "tests_per_100k", "recovery_rate"]
X_clf = df[clf_features]
y_clf = df["high_risk"]

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.25, random_state=42, stratify=y_clf
)

log_clf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000))
])

rf_clf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(n_estimators=300, random_state=42))
])

log_clf.fit(X_train_clf, y_train_clf)
rf_clf.fit(X_train_clf, y_train_clf)

log_pred = log_clf.predict(X_test_clf)
rfc_pred = rf_clf.predict(X_test_clf)

print("\nClassification Results")
print("Logistic Regression Accuracy:", accuracy_score(y_test_clf, log_pred))
print("Logistic Regression Precision:", precision_score(y_test_clf, log_pred, zero_division=0))
print("Logistic Regression Recall:", recall_score(y_test_clf, log_pred, zero_division=0))
print("Logistic Regression F1:", f1_score(y_test_clf, log_pred, zero_division=0))

print("Random Forest Accuracy:", accuracy_score(y_test_clf, rfc_pred))
print("Random Forest Precision:", precision_score(y_test_clf, rfc_pred, zero_division=0))
print("Random Forest Recall:", recall_score(y_test_clf, rfc_pred, zero_division=0))
print("Random Forest F1:", f1_score(y_test_clf, rfc_pred, zero_division=0))

rf_clf_model = rf_clf.named_steps["model"]
rf_clf_importance = pd.DataFrame({
    "feature": clf_features,
    "importance": rf_clf_model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nClassification Feature Importance")
print(rf_clf_importance)

print("\nConfusion Matrix")
print(confusion_matrix(y_test_clf, rfc_pred))

# Score all countries using classifier probabilities
full_probs = rf_clf.predict_proba(X_clf)[:, 1]
risk_scored = df[["country", "continent", "cases", "deaths", "death_rate"]].copy()
risk_scored["predicted_high_risk_probability"] = full_probs
risk_scored = risk_scored.sort_values("predicted_high_risk_probability", ascending=False)

print("\nTop Predicted High-Risk Countries")
print(risk_scored.head(20))


# In[8]:


# -----------------------------
# Charts
# -----------------------------

reg_results = pd.DataFrame({
    "model": ["Linear Regression", "Random Forest Regressor"],
    "R2": [
        r2_score(y_test_reg, lin_pred),
        r2_score(y_test_reg, rf_pred),
    ],
})


# In[9]:


plt.figure(figsize=(8, 5))
plt.bar(reg_results["model"], reg_results["R2"])
plt.title("Regression Model Comparison (R²)")
plt.ylabel("R² Score")
plt.xlabel("Model")
plt.tight_layout()
plt.show()


# In[10]:


plot_imp_reg = rf_reg_importance.sort_values("importance")
plt.figure(figsize=(8, 5))
plt.barh(plot_imp_reg["feature"], plot_imp_reg["importance"])
plt.title("Random Forest Regressor Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# In[11]:


clf_results = pd.DataFrame({
    "model": ["Logistic Regression", "Random Forest Classifier"],
    "f1": [
        f1_score(y_test_clf, log_pred, zero_division=0),
        f1_score(y_test_clf, rfc_pred, zero_division=0),
    ],
})


# In[12]:


plt.figure(figsize=(8, 5))
plt.bar(clf_results["model"], clf_results["f1"])
plt.title("Classification Model Comparison (F1 Score)")
plt.ylabel("F1 Score")
plt.xlabel("Model")
plt.tight_layout()
plt.show()


# In[13]:


plot_imp_clf = rf_clf_importance.sort_values("importance")
plt.figure(figsize=(8, 5))
plt.barh(plot_imp_clf["feature"], plot_imp_clf["importance"])
plt.title("Random Forest Classifier Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# In[14]:


best_pred_df = pd.DataFrame({"actual": y_test_reg.values, "predicted": rf_pred})
plt.figure(figsize=(8, 6))
plt.scatter(best_pred_df["actual"], best_pred_df["predicted"])
plt.title("Actual vs Predicted Deaths (Random Forest Regressor)")
plt.xlabel("Actual Deaths")
plt.ylabel("Predicted Deaths")
plt.tight_layout()
plt.show()


# # unsupervised learning
# K-Means clustering
# PCA visualization
# country segments like low, medium, high, and extreme risk

# In[15]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# In[16]:


# Load data
file_path = "covid_19_cleaned.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip().str.lower()


# In[17]:


# Feature engineering
df["cases_per_100k"] = np.where(df["population"] > 0, (df["cases"] / df["population"]) * 100000, np.nan)
df["deaths_per_100k"] = np.where(df["population"] > 0, (df["deaths"] / df["population"]) * 100000, np.nan)
df["tests_per_100k"] = np.where(df["population"] > 0, (df["tests"] / df["population"]) * 100000, np.nan)
df["recovery_rate"] = np.where(df["cases"] > 0, (df["recovered"] / df["cases"]) * 100, np.nan)
df["death_rate"] = np.where(df["cases"] > 0, (df["deaths"] / df["cases"]) * 100, np.nan)

df.replace([np.inf, -np.inf], np.nan, inplace=True)

cluster_features = [
    "cases_per_100k",
    "deaths_per_100k",
    "tests_per_100k",
    "recovery_rate",
    "death_rate",
]

X = df[cluster_features].copy()


# In[18]:


# Impute + scale
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)


# In[19]:


# Try several k values
k_results = []
models = {}
for k in range(2, 7):
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    k_results.append({"k": k, "silhouette_score": sil, "inertia": model.inertia_})
    models[k] = model

k_results_df = pd.DataFrame(k_results).sort_values("silhouette_score", ascending=False)
best_k = int(k_results_df.iloc[0]["k"])
best_model = models[best_k]
df["cluster"] = best_model.predict(X_scaled)


# In[20]:


# Rank clusters by severity using a simple composite score
cluster_profile = (
    df.groupby("cluster")
      .agg(
          countries=("country", "count"),
          median_cases_per_100k=("cases_per_100k", "median"),
          median_deaths_per_100k=("deaths_per_100k", "median"),
          median_tests_per_100k=("tests_per_100k", "median"),
          median_recovery_rate=("recovery_rate", "median"),
          median_death_rate=("death_rate", "median"),
      )
      .reset_index()
)

cluster_profile["severity_score"] = (
    cluster_profile["median_cases_per_100k"].fillna(0).rank(pct=True) * 0.35 +
    cluster_profile["median_deaths_per_100k"].fillna(0).rank(pct=True) * 0.30 +
    cluster_profile["median_death_rate"].fillna(0).rank(pct=True) * 0.20 +
    cluster_profile["median_tests_per_100k"].fillna(0).rank(pct=True) * 0.05 -
    cluster_profile["median_recovery_rate"].fillna(0).rank(pct=True) * 0.10
)

cluster_profile = cluster_profile.sort_values("severity_score").reset_index(drop=True)
risk_labels = ["Low Risk", "Moderate Risk", "High Risk", "Severe Risk", "Extreme Risk"]

label_map = {}
for i, original_cluster in enumerate(cluster_profile["cluster"]):
    label_map[original_cluster] = risk_labels[i]

df["risk_cluster"] = df["cluster"].map(label_map)


# In[21]:


# PCA projection
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)


# In[22]:


# Plot cluster quality
plt.figure(figsize=(8, 5))
plt.plot(pd.DataFrame(k_results).sort_values("k")["k"],
         pd.DataFrame(k_results).sort_values("k")["silhouette_score"],
         marker="o")
plt.title("KMeans Cluster Quality by k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.tight_layout()
plt.show()


# In[23]:


# PCA cluster plot
pca_df = pd.DataFrame(X_pca, columns=["pc1", "pc2"])
pca_df["risk_cluster"] = df["risk_cluster"]

plt.figure(figsize=(9, 6))
for label in sorted(pca_df["risk_cluster"].dropna().unique()):
    subset = pca_df[pca_df["risk_cluster"] == label]
    plt.scatter(subset["pc1"], subset["pc2"], label=label)
plt.title("Country Clusters in PCA Space")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.tight_layout()
plt.show()


# In[24]:


# Severity score chart
cluster_profile["risk_cluster"] = cluster_profile["cluster"].map(label_map)
plot_profile = cluster_profile.sort_values("severity_score")

plt.figure(figsize=(9, 5))
plt.bar(plot_profile["risk_cluster"], plot_profile["severity_score"])
plt.title("Cluster Severity Score")
plt.xlabel("Risk Cluster")
plt.ylabel("Severity Score")
plt.tight_layout()
plt.show()


# In[ ]:




