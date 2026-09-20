# models to compare:
#FR_FWH = Year+Month+Location+Duration of Stay+Experience+CRCID
#FR_FWH = Year+Month+Location+Duration of Stay+Experience
#FR_FWH = Year+Month+Location+Duration of Stay
#FR_FWH = randomeffect(Year) + Month + Location + Duration of Stay + Experience + randomeffect(CRC-ID)

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

wdir = 'C:\\_temp workspace\\1ER-LAT-FEED\\data\\'
os.chdir(wdir)
ratedat = pd.DataFrame(pd.read_csv("model-rate-data_ERLATFEED.csv"))
ratedat = ratedat.replace([np.inf, -np.inf], np.nan).dropna()

X = ratedat[['nDaysNPSThisSeason', 'nYearsNPSTotal', 'month', 'year', 'Location', 'crcid']].to_numpy()
y = ratedat['FR_perFWH'].to_numpy()

# Standardize continuous predictors
ratedat['nDaysNPSThisSeason'] = (ratedat['nDaysNPSThisSeason'] - ratedat['nDaysNPSThisSeason'].mean()) / ratedat['nDaysNPSThisSeason'].std()
ratedat['nYearsNPSTotal'] = (ratedat['nYearsNPSTotal'] - ratedat['nYearsNPSTotal'].mean()) / ratedat['nYearsNPSTotal'].std()

# Convert categorical variables
ratedat['month'] = ratedat['month'].astype('category')
ratedat['year'] = ratedat['year'].astype('category')
ratedat['Location'] = ratedat['Location'].astype('category')
ratedat['crcid'] = ratedat['crcid'].astype('category')

# Define the Poisson regression model
poisson_model = smf.glm(
    formula="FR_perFWH ~ nDaysNPSThisSeason + nYearsNPSTotal + C(month) + C(year) + C(Location) + C(crcid)",
    data=ratedat,
    family=sm.families.Poisson()
).fit()

# Print model summary
print(poisson_model.summary())
print("Mean of FR_FWH:", ratedat['FR_perFWH'].mean())
print("Variance of FR_FWH:", ratedat['FR_perFWH'].var())

# Fit Negative Binomial regression
nb_model = smf.glm(
    formula="FR_perFWH ~ nDaysNPSThisSeason + nYearsNPSTotal + C(month) + C(year) + C(Location) + C(crcid)",
    data=ratedat,
    family=sm.families.NegativeBinomial()
).fit()

# Print model summary
print(nb_model.summary())
print("Poisson AIC:", poisson_model.aic)
print("Negative Binomial AIC:", nb_model.aic)

# check the dispersion parameter
dispersion_ratio = poisson_model.deviance / poisson_model.df_resid
print("Dispersion Ratio:", dispersion_ratio)
# If ≈ 1, Poisson is fine.
# If >> 1, Poisson underestimates standard errors, making Negative Binomial more reliable for ranking predictor importance.
# Dispersion Ratio: 2.5875760573666824

# Poisson model
poisson_model = smf.glm(
    "FR_perFWH ~ nDaysNPSThisSeason + nYearsNPSTotal + C(month) + C(year) + C(Location) + C(crcid)",
    data=ratedat,
    family=sm.families.Poisson()
).fit()

# Negative Binomial model
nb_model = smf.glm(
    "FR_perFWH ~ nDaysNPSThisSeason + nYearsNPSTotal + C(month) + C(year) + C(Location) + C(crcid)",
    data=ratedat,
    family=sm.families.NegativeBinomial()
).fit()

# Compare results
print("Poisson Model Summary:")
print(poisson_model.summary())

print("\nNegative Binomial Model Summary:")
print(nb_model.summary())

# If Poisson and Negative Binomial rank predictors differently, trust Negative Binomial more because it accounts for overdispersion.
# If coefficients and p-values are similar across models, Poisson is fine.
print(poisson_model.conf_int())
print(nb_model.conf_int())


# Extract coefficients and confidence intervals
coef_df = pd.DataFrame({
    'Predictor': poisson_model.params.index,
    'Poisson_Estimate': poisson_model.params.values,
    'Poisson_LowerCI': poisson_model.conf_int()[0].values,
    'Poisson_UpperCI': poisson_model.conf_int()[1].values,
    'NB_Estimate': nb_model.params.values,
    'NB_LowerCI': nb_model.conf_int()[0].values,
    'NB_UpperCI': nb_model.conf_int()[1].values
}).reset_index(drop=True)

# Plot coefficient estimates with confidence intervals
plt.figure(figsize=(10, 6))
for i, row in coef_df.iterrows():
    plt.plot([row['Poisson_LowerCI'], row['Poisson_UpperCI']], [i + 0.2, i + 0.2], color='blue', linewidth=2)
    plt.scatter(row['Poisson_Estimate'], i + 0.2, color='blue', label='Poisson' if i == 0 else "")

    plt.plot([row['NB_LowerCI'], row['NB_UpperCI']], [i - 0.2, i - 0.2], color='red', linewidth=2)
    plt.scatter(row['NB_Estimate'], i - 0.2, color='red', label='Negative Binomial' if i == 0 else "")

plt.axvline(0, linestyle="--", color="gray", alpha=0.7)
plt.yticks(range(len(coef_df)), coef_df['Predictor'])
plt.xlabel("Coefficient Estimate")
plt.title("Comparison of Coefficient Estimates (Poisson vs Negative Binomial)")
plt.legend()
plt.show()

# Plot Predicted vs Observed Counts
plt.figure(figsize=(10, 5))
sns.scatterplot(x=poisson_model.fittedvalues, y=ratedat['FR_perFWH'], label='Poisson', color='blue', alpha=0.6)
sns.scatterplot(x=nb_model.fittedvalues, y=ratedat['FR_perFWH'], label='Negative Binomial', color='red', alpha=0.6)
plt.plot([0, max(ratedat['FR_perFWH'])], [0, max(ratedat['FR_perFWH'])], 'k--', alpha=0.5)  # Reference line
plt.xlabel("Predicted FR_FWH")
plt.ylabel("Observed FR_FWH")
plt.title("Predicted vs Observed Counts")
plt.legend()
plt.show()

# Plot Residuals vs. Fitted Values
plt.figure(figsize=(10, 5))
sns.scatterplot(x=poisson_model.fittedvalues, y=poisson_model.resid_pearson, label="Poisson", color='blue', alpha=0.6)
sns.scatterplot(x=nb_model.fittedvalues, y=nb_model.resid_pearson, label="Negative Binomial", color='red', alpha=0.6)
plt.axhline(0, linestyle="--", color="gray", alpha=0.7)
plt.xlabel("Fitted Values")
plt.ylabel("Pearson Residuals")
plt.title("Residuals vs. Fitted Values (Poisson vs Negative Binomial)")
plt.legend()
plt.show()