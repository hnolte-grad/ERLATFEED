import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from plotnine import ggplot, aes, geom_boxplot, theme_void, annotate
from scipy import stats

os.chdir('C:\\_temp workspace\\1ER-LAT-FEED\\data\\')
ratedat = pd.DataFrame(pd.read_csv("ratedata_ERLATFEED.csv"))


def plot_scatterwithreg(dat, xvar, yvar):
    dat = ratedat.loc[dat['Feeding Observed?'] == 'Y'].dropna(subset=[xvar, yvar])              # Drop deployments that don't yet have duration stay data
    correlation, _ = stats.pearsonr(dat[xvar], dat[yvar])
    sns.regplot(x=dat[xvar], y=dat[yvar], ci=True, line_kws={'color':'red'})
    plt.title(f'Regression Line (Correlation (Pearson): {correlation:.2f})')
    plt.show()

def plot_boxplot(dat, xvar,yvar, l, w):
    sns.set_theme(rc={'figure.figsize':(l,w)})
    sns.boxplot(x = dat[xvar],
                y = dat[yvar],
                palette = 'husl')
    plt.tight_layout()
    plt.show()

# feeding rates v. deploment duration
plot_scatterwithreg(ratedat, 'Tidal Feed Rate', yvar='Dur Tidal Feed Hrs')
plot_scatterwithreg(ratedat, 'Tidal Feed Rate', yvar='TimeInNPS_days')
plot_scatterwithreg(ratedat, 'Tidal Feed Rate', yvar='sumYearsNPS')
plot_boxplot(ratedat, 'CRC-ID', yvar='Tidal Feed Rate', l=16, w=8)
plot_boxplot(ratedat, 'firstSeen', yvar='Tidal Feed Rate', l=16, w=8)
plot_boxplot(ratedat, 'yn_UMEYear', yvar='Tidal Feed Rate', l=8, w=8)
plot_boxplot(ratedat, 'Sex', yvar='Tidal Feed Rate', l=8, w=8)
plot_boxplot(ratedat, 'depMonth', yvar='Tidal Feed Rate', l=8, w=8)

