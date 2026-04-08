#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


# Load file
df = pd.read_csv("country_ml_risk_report.csv")


# In[4]:


# Keeping cluster order consistent
cluster_order = ["Low Risk", "Moderate Risk", "High Risk", "Severe Risk", "Extreme Risk"]
df["risk_cluster"] = pd.Categorical(df["risk_cluster"], categories=cluster_order, ordered=True)


# In[5]:


# -----------------------------
# 1. Cluster count bar chart
# -----------------------------
cluster_counts = df["risk_cluster"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
plt.bar(cluster_counts.index, cluster_counts.values)
plt.title("Number of Countries in Each Risk Cluster")
plt.xlabel("Risk Cluster")
plt.ylabel("Country Count")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()


# In[6]:


# -----------------------------
# 2. Top 15 countries by death rate
# -----------------------------
top_death = df.sort_values("death_rate", ascending=False).head(15)

plt.figure(figsize=(10, 6))
plt.barh(top_death["country"], top_death["death_rate"])
plt.title("Top 15 Countries by Death Rate")
plt.xlabel("Death Rate (%)")
plt.ylabel("Country")
plt.tight_layout()
plt.show()


# In[7]:


# -----------------------------
# 3. Top 15 countries by cases per 100k
# -----------------------------
top_cases = df.sort_values("cases_per_100k", ascending=False).head(15)

plt.figure(figsize=(10, 6))
plt.barh(top_cases["country"], top_cases["cases_per_100k"])
plt.title("Top 15 Countries by Cases per 100k")
plt.xlabel("Cases per 100,000")
plt.ylabel("Country")
plt.tight_layout()
plt.show()


# In[8]:


# -----------------------------
# 4. Top 15 countries by deaths per 100k
# -----------------------------
top_deaths_per_100k = df.sort_values("deaths_per_100k", ascending=False).head(15)

plt.figure(figsize=(10, 6))
plt.barh(top_deaths_per_100k["country"], top_deaths_per_100k["deaths_per_100k"])
plt.title("Top 15 Countries by Deaths per 100k")
plt.xlabel("Deaths per 100,000")
plt.ylabel("Country")
plt.tight_layout()
plt.show()


# In[9]:


# -----------------------------
# 5. Recovery rate by cluster
# -----------------------------
recovery_by_cluster = df.groupby("risk_cluster", observed=False)["recovery_rate"].median()

plt.figure(figsize=(8, 5))
plt.bar(recovery_by_cluster.index, recovery_by_cluster.values)
plt.title("Median Recovery Rate by Risk Cluster")
plt.xlabel("Risk Cluster")
plt.ylabel("Median Recovery Rate (%)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()


# In[10]:


# -----------------------------
# 6. Death rate by cluster
# -----------------------------
death_by_cluster = df.groupby("risk_cluster", observed=False)["death_rate"].median()

plt.figure(figsize=(8, 5))
plt.bar(death_by_cluster.index, death_by_cluster.values)
plt.title("Median Death Rate by Risk Cluster")
plt.xlabel("Risk Cluster")
plt.ylabel("Median Death Rate (%)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()


# In[11]:


# -----------------------------
# 7. Cases vs deaths scatter
# -----------------------------
plt.figure(figsize=(8, 6))
plt.scatter(df["cases"], df["deaths"])
plt.title("Cases vs Deaths")
plt.xlabel("Cases")
plt.ylabel("Deaths")
plt.tight_layout()
plt.show()


# In[12]:


# -----------------------------
# 8. Tests per 100k vs cases per 100k
# -----------------------------
plt.figure(figsize=(8, 6))
plt.scatter(df["tests_per_100k"], df["cases_per_100k"])
plt.title("Tests per 100k vs Cases per 100k")
plt.xlabel("Tests per 100,000")
plt.ylabel("Cases per 100,000")
plt.tight_layout()
plt.show()


# In[13]:


# -----------------------------
# 9. Extreme risk countries only
# -----------------------------
extreme_df = df[df["risk_cluster"] == "Extreme Risk"].sort_values("death_rate", ascending=False).head(15)

plt.figure(figsize=(10, 6))
plt.barh(extreme_df["country"], extreme_df["death_rate"])
plt.title("Extreme Risk Countries by Death Rate")
plt.xlabel("Death Rate (%)")
plt.ylabel("Country")
plt.tight_layout()
plt.show()


# # single dashboard-style

# In[15]:


cluster_order = ["Low Risk", "Moderate Risk", "High Risk", "Severe Risk", "Extreme Risk"]
df["risk_cluster"] = pd.Categorical(df["risk_cluster"], categories=cluster_order, ordered=True)
fig = plt.figure(figsize=(14, 10))

# 1
ax1 = fig.add_subplot(2, 2, 1)
cluster_counts = df["risk_cluster"].value_counts().sort_index()
ax1.bar(cluster_counts.index, cluster_counts.values)
ax1.set_title("Countries per Risk Cluster")
ax1.set_xlabel("Risk Cluster")
ax1.set_ylabel("Count")
ax1.tick_params(axis="x", rotation=20)

# 2
ax2 = fig.add_subplot(2, 2, 2)
top_cases = df.sort_values("cases_per_100k", ascending=False).head(10)
ax2.barh(top_cases["country"], top_cases["cases_per_100k"])
ax2.set_title("Top 10 Cases per 100k")
ax2.set_xlabel("Cases per 100,000")
ax2.set_ylabel("Country")

# 3
ax3 = fig.add_subplot(2, 2, 3)
top_deaths = df.sort_values("death_rate", ascending=False).head(10)
ax3.barh(top_deaths["country"], top_deaths["death_rate"])
ax3.set_title("Top 10 Death Rate")
ax3.set_xlabel("Death Rate (%)")
ax3.set_ylabel("Country")

# 4
ax4 = fig.add_subplot(2, 2, 4)
ax4.scatter(df["tests_per_100k"], df["cases_per_100k"])
ax4.set_title("Testing vs Cases per 100k")
ax4.set_xlabel("Tests per 100,000")
ax4.set_ylabel("Cases per 100,000")

plt.tight_layout()
plt.show()

