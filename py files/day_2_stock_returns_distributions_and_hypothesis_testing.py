# Day 2 — Stock Return Distributions & Hypothesis Testing
# ----------------------------------------------------
# This is a runnable Jupyter-style notebook (use in VSCode or copy into a .ipynb) with
# mixed explanatory markdown and code cells. Some cells are fully coded, others have
# TODO sections for you to implement — so you *code while learning*.

# %% [markdown]
# # Day 2 — Probability & Stats for Finance
# **Goals:**
# - Understand distributions: normal, lognormal, Student's t
# - Compute expectation, variance, covariance, correlation from data
# - Build a 2-asset portfolio and compute return & risk
# - Run hypothesis tests (one- and two-tailed), and interpret p-values
#
# Use this file as your interactive exercise. Run cells, modify parameters, and fill
# in the TODOs marked below.

# %%
# === Imports ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# make plots look consistent
plt.style.use('default')

# %% [markdown]
# ## 1) Quick sanity: simulate distributions and plot PDFs
# - Run the cells – then **implement** the TODOs to change parameters and observe.

# %%
# baseline x-range
x = np.linspace(-5, 5, 1000)

# normal(0,1)
pdf_normal = stats.norm.pdf(x, 0, 1)

# t-distribution with df=5
pdf_t5 = stats.t.pdf(x, df=5)

# lognormal: we will plot on positive x only
x_pos = np.linspace(0.001, 10, 1000)
# shape s = sigma of underlying normal; here s=0.5
pdf_lognorm = stats.lognorm.pdf(x_pos, s=0.5)

plt.figure(figsize=(9,4))
plt.plot(x, pdf_normal, label='Normal(0,1)')
plt.plot(x, pdf_t5, label='t df=5')
plt.plot(x_pos, pdf_lognorm, label='Lognormal(s=0.5)')
plt.xlim(-5,5)
plt.legend(); plt.title('Normal vs t vs Lognormal (visual)')
plt.show()

# %% [markdown]
# TODO:
# - Change df of the t-distribution to 2 and 30. Re-run and note how tails change.
# - Change the lognormal shape parameter (s) and observe skew.

# %% [markdown]
# ## 2) Coin flips (empirical probability)

# %%
np.random.seed(42)
flips = np.random.choice([0,1], size=10000)  # 0=heads, 1=tails
p_heads = np.mean(flips==0)
p_tails = np.mean(flips==1)
print(f"P(heads)={p_heads:.4f}, P(tails)={p_tails:.4f}")

# %% [markdown]
# TODO:
# - Change the coin to a biased coin (p=0.6 for heads) and re-estimate.
# - Plot the running empirical probability (cumulative mean) vs number of flips.

# %% [markdown]
# ## 3) Simulate stock returns: normal assumption vs fat tails (t)
# We'll simulate two series: `sim_normal` (normal) and `sim_t` (Student's t scaled)

# %%
np.random.seed(123)
n = 1000
# normal returns: mean 0.1% , daily volatility 2%
sim_normal = np.random.normal(loc=0.001, scale=0.02, size=n)

# t-distributed returns: generate from standard t and scale to have approx std=0.02
# pick df = 5 (heavy tails)
df = 5
raw_t = stats.t.rvs(df=df, size=n)
# scale raw_t to have standard deviation ~0.02 (approx)
scale_factor = 0.02 / np.std(raw_t)
sim_t = raw_t * scale_factor + 0.001  # add same mean

# quick histograms
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(sim_normal, bins=30, density=True, alpha=0.6)
mu_n, sd_n = np.mean(sim_normal), np.std(sim_normal, ddof=1)
xn = np.linspace(mu_n - 4*sd_n, mu_n + 4*sd_n, 200)
plt.plot(xn, stats.norm.pdf(xn, mu_n, sd_n), lw=2, label='Normal PDF fit')
plt.title('Simulated Normal returns')
plt.legend()

plt.subplot(1,2,2)
plt.hist(sim_t, bins=30, density=True, alpha=0.6)
mu_t, sd_t = np.mean(sim_t), np.std(sim_t, ddof=1)
xt = np.linspace(mu_t - 4*sd_t, mu_t + 4*sd_t, 200)
plt.plot(xt, stats.norm.pdf(xt, mu_t, sd_t), lw=2, label='Normal PDF fit')
plt.title(f'Simulated t returns (df={df})')
plt.legend()
plt.tight_layout(); plt.show()

# %% [markdown]
# TODO:
# - Compute empirical kurtosis for sim_normal and sim_t (`stats.kurtosis`) and explain.
# - Load a small CSV of real returns (if you have one) and overlay its histogram.

# %% [markdown]
# ## 4) Expectation, Variance, Covariance, Correlation — practice with data
# We'll create two correlated assets and compute all statistics.  

# %%
np.random.seed(44)
# base noise
z = np.random.normal(0, 0.02, size=n)
# asset A
returns_A = 0.001 + z
# asset B correlated with A
returns_B = 0.8 * returns_A + np.random.normal(0, 0.015, size=n)

# sample expectations (means)
mean_A = np.mean(returns_A)
mean_B = np.mean(returns_B)
print('mean_A, mean_B =', mean_A, mean_B)

# sample variances (use ddof=1 for unbiased)
var_A = np.var(returns_A, ddof=1)
var_B = np.var(returns_B, ddof=1)
print('var_A, var_B =', var_A, var_B)

# sample covariance and correlation
cov_AB = np.cov(returns_A, returns_B, ddof=1)[0,1]
corr_AB = np.corrcoef(returns_A, returns_B)[0,1]
print('cov_AB =', cov_AB)
print('corr_AB =', corr_AB)

# %% [markdown]
# TODO:
# - Compute the same covariance using the formula cov = E[XY] - E[X]E[Y] (estimate with averages).
# - Try `ddof=0` vs `ddof=1` and see the difference.

# %% [markdown]
# ## 5) Portfolio return & variance (2-asset)

# %%
w1, w2 = 0.6, 0.4
E_Rp = w1*mean_A + w2*mean_B
var_p = (w1**2)*var_A + (w2**2)*var_B + 2*w1*w2*cov_AB
print('Expected portfolio return (E_Rp)=', E_Rp)
print('Portfolio variance (var_p)=', var_p)
print('Portfolio std dev =', np.sqrt(var_p))

# %% [markdown]
# TODO:
# - Write a function `portfolio_stats(returns_A, returns_B, w1, w2)` that returns E_Rp, var_p, std_p.
# - Use it to compute stats for a grid of weights w1 in [0,1] (w2=1-w1) and plot expected return vs risk (efficient frontier).

# %% [markdown]
# ## 6) Hypothesis testing: t-test & p-values
# We'll test whether mean of returns_A is different from 0 (two-tailed) and >0 (one-tailed).

# %%
# two-tailed one-sample t-test
t_stat, p_two = stats.ttest_1samp(returns_A, 0)
# convert to one-tailed p-value for H1: mean>0
p_one = p_two/2 if t_stat > 0 else 1 - p_two/2
print('t-statistic =', t_stat)
print('two-tailed p-value =', p_two)
print('one-tailed p-value =', p_one)

# decision at alpha=0.05
alpha = 0.05
if p_two < alpha:
    print('Reject H0 in two-tailed test (mean != 0)')
else:
    print('Fail to reject H0 in two-tailed test')

if p_one < alpha:
    print('Reject H0 in one-tailed test (mean > 0)')
else:
    print('Fail to reject H0 in one-tailed test')

# %% [markdown]
# TODO:
# - Explain why p_one = p_two/2 when t_stat>0. What if t_stat<0?
# - Try a bootstrap test for the mean (resampling) and compare p-values.

# %% [markdown]
# ## 7) Challenge tasks (apply what you've learned)
# 1. Fit a t-distribution to `returns_B` (use `stats.t.fit`) and compare fit parameters with normal fit.
# 2. Compute 99% VaR using normal assumption and using fitted t-distribution; compare values.
# 3. Create an array of portfolios (w1 from 0 to 1), compute E and std, and plot the efficient frontier.
# 4. Optional: Load real stock returns CSV and repeat the analyses.

# %%
# --- starter for portfolio grid (fill in the rest) ---

def portfolio_stats(ra, rb, w1):
    """Return expected return, variance, std for portfolio with weights w1 and 1-w1."""
    w2 = 1 - w1
    E1, E2 = np.mean(ra), np.mean(rb)
    v1, v2 = np.var(ra, ddof=1), np.var(rb, ddof=1)
    cov = np.cov(ra, rb, ddof=1)[0,1]
    E_p = w1*E1 + w2*E2
    var_p = (w1**2)*v1 + (w2**2)*v2 + 2*w1*w2*cov
    return E_p, var_p, np.sqrt(var_p)

# TODO: sweep w1 values, compute E_p and std_p, and plot.

# %% [markdown]
# ## HOW TO USE THIS NOTEBOOK
# - Run each cell top-to-bottom.
# - For each TODO, implement the code and write a one-line note describing what you observed.
# - Commit this notebook as your deliverable: "Stock Return Distributions & Hypothesis Testing".

# %% [markdown]
# Good luck — ask me if you want solutions for any TODO cell or want the completed version.
