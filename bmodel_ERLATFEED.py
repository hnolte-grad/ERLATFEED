import pandas as pd
import bambi as bmb
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import os

# Setup working directory
wdir = 'C:\\_temp workspace\\1ER-LAT-FEED\\data\\model\\'
os.chdir(wdir)

# Load and clean data
df = pd.DataFrame(pd.read_csv("model-rate-data_ERLATFEED.csv"))
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Convert categorical variables
df['Location'] = df['Location'].astype('category')
df['year'] = df['year'].astype('category')
df['crcid'] = df['crcid'].astype('category')
# force non‑centered
nb_model = bmb.Model("FR_perFWH ~ (1|year) + Location +  (1|crcid)", df, family="negativebinomial",
                  priors={"(1|crcid)": {"re_param": "noncentered"}})

# # Define model
# nb_model = bmb.Model(
#     "FR_perFWH ~ (1|year) + Location +  (1|crcid)",
#     df,
#     family="negativebinomial"
#     # , offset="log_effort"  # if using effort
# )

if __name__ == "__main__":
    # Fit model
    nb_trace = nb_model.fit(
        draws=2000,
        chains=4,
        cores=2,
        init="jitter+adapt_diag"
    )

    # Summarize and diagnose
    print(az.rhat(nb_trace))
    summary = az.summary(nb_trace, var_names=["~sigma"], hdi_prob=0.99)
    print(summary)
    summary.to_csv("bayesian_model_summary.csv")

    # Posterior predictive checks
    ppc = nb_model.predict(idata=nb_trace, kind='pps', inplace=False)
    az.plot_ppc(ppc)
    plt.tight_layout()
    plt.show()

    # Forest and trace plots
    az.plot_forest(nb_trace, var_names=["~sigma"], kind='forestplot', hdi_prob=0.99)
    plt.tight_layout()
    plt.show()

    az.plot_trace(nb_trace, var_names=["~sigma"])
    plt.tight_layout()
    plt.show()


    # posterior is an xarray.Dataset: posterior = nb_trace.posterior
    posterior = nb_trace.posterior

    # stack chains & draws into one “sample” axis for easy indexing
    # (you could also leave them separate and reshape later)
    posterior_stacked = posterior.stack(sample=("chain", "draw"))

    # extract Intercept draws
    intercept = posterior_stacked["Intercept"].values

    # extract Location effects for each level (levels are strings in Location_dim)
    loc2 = posterior_stacked["Location"].sel(Location_dim="2").values
    loc3 = posterior_stacked["Location"].sel(Location_dim="3").values
    loc4 = posterior_stacked["Location"].sel(Location_dim="4").values

    # now exponentiate to get rate ratios
    df = {
        "Intercept": np.exp(intercept),
        "Loc2_vs_1": np.exp(loc2),
        "Loc3_vs_1": np.exp(loc3),
        "Loc4_vs_1": np.exp(loc4),
    }

    # summarize
    import pandas as pd

    summary = pd.DataFrame({
        name: [np.median(vals), *az.hdi(vals, hdi_prob=0.95)]
        for name, vals in df.items()
    }, index=["median", "hdi_2.5", "hdi_97.5"]).T

    # plot forest of rate ratios
    fig, ax = plt.subplots(figsize=(5, 3))
    y = np.arange(len(summary))
    ax.errorbar(summary["median"], y,
                xerr=[summary["median"] - summary["hdi_2.5"],
                      summary["hdi_97.5"] - summary["median"]],
                fmt="o", capsize=4)
    ax.axvline(1, color="red", ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(summary.index)
    ax.set_xlabel("Rate ratio (exp(coefficient))")
    ax.set_title("Posterior median & 95% CI of rate ratios")
    plt.tight_layout()
    plt.show()

