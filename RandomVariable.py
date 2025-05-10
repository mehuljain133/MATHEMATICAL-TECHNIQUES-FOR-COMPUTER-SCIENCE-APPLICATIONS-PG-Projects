# Unit-III Random variable: Introduction to random variable, Discrete random variables (Bernoulli,Binomial, Multinomial, Poisson, Geometric, Negative Binomial), Continuous random variables(Uniform, Exponential, Normal, Gamma), Expectation, variance, Conditional probability andconditional expectation, Central Limit Theorem, Markov and Chebyshev’s inequality. 

pip install numpy pandas matplotlib seaborn scipy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import bernoulli, binom, poisson, geom, norm, expon, gamma

# 1. Discrete Random Variables
# Bernoulli Distribution (Success Probability = 0.5)
bernoulli_data = bernoulli.rvs(p=0.5, size=1000)
print(f"Bernoulli Mean: {bernoulli_data.mean()}, Variance: {bernoulli_data.var()}")

# Binomial Distribution (n = 10, p = 0.5)
binomial_data = binom.rvs(n=10, p=0.5, size=1000)
print(f"\nBinomial Mean: {binomial_data.mean()}, Variance: {binomial_data.var()}")

# Poisson Distribution (lambda = 3)
poisson_data = poisson.rvs(mu=3, size=1000)
print(f"\nPoisson Mean: {poisson_data.mean()}, Variance: {poisson_data.var()}")

# Geometric Distribution (p = 0.5)
geom_data = geom.rvs(p=0.5, size=1000)
print(f"\nGeometric Mean: {geom_data.mean()}, Variance: {geom_data.var()}")

# Negative Binomial Distribution (r = 5, p = 0.5)
nbinom_data = np.random.negative_binomial(5, 0.5, size=1000)
print(f"\nNegative Binomial Mean: {nbinom_data.mean()}, Variance: {nbinom_data.var()}")

# 2. Continuous Random Variables
# Uniform Distribution (a = 0, b = 10)
uniform_data = np.random.uniform(0, 10, 1000)
print(f"\nUniform Mean: {uniform_data.mean()}, Variance: {uniform_data.var()}")

# Exponential Distribution (lambda = 1)
exp_data = expon.rvs(scale=1, size=1000)
print(f"\nExponential Mean: {exp_data.mean()}, Variance: {exp_data.var()}")

# Normal Distribution (mu = 0, sigma = 1)
normal_data = norm.rvs(loc=0, scale=1, size=1000)
print(f"\nNormal Mean: {normal_data.mean()}, Variance: {normal_data.var()}")

# Gamma Distribution (shape = 2, scale = 1)
gamma_data = gamma.rvs(a=2, scale=1, size=1000)
print(f"\nGamma Mean: {gamma_data.mean()}, Variance: {gamma_data.var()}")

# 3. Expectation and Variance for a given Distribution
# Expectation for a normal distribution: E(X) = mu
# Variance for a normal distribution: Var(X) = sigma^2
mu, sigma = 0, 1  # Normal distribution parameters
expected_value = mu
variance_value = sigma ** 2
print(f"\nNormal Distribution Expected Value: {expected_value}, Variance: {variance_value}")

# 4. Central Limit Theorem
# Sampling from any distribution (we use normal here) and calculating sample means
sample_means = [np.mean(np.random.choice(normal_data, size=30)) for _ in range(1000)]
plt.figure(figsize=(6, 4))
sns.histplot(sample_means, kde=True, color="purple")
plt.title("Central Limit Theorem: Sampling Distribution of Mean")
plt.show()

# 5. Markov's Inequality: P(X ≥ a) ≤ E[X] / a for a ≥ 0
# For normal distribution, E[X] = 0
# P(X ≥ a) ≤ E[X] / a = 0 / a = 0 for a > 0
a = 5
markov_bound = expected_value / a
print(f"\nMarkov's Inequality Bound: P(X ≥ {a}) ≤ {markov_bound}")

# 6. Chebyshev’s Inequality: P(|X - μ| ≥ kσ) ≤ 1/k²
k = 2
chebyshev_bound = 1 / (k ** 2)
print(f"\nChebyshev’s Inequality Bound: P(|X - μ| ≥ {k}σ) ≤ {chebyshev_bound}")

# 7. Conditional Probability and Conditional Expectation (Example)
# Suppose we have two independent random variables X and Y
# Let X ~ Normal(0, 1), Y ~ Uniform(0, 10)
X = np.random.normal(0, 1, 1000)
Y = np.random.uniform(0, 10, 1000)

# Conditional Expectation E[Y|X] - Assuming independence, E[Y|X] = E[Y]
conditional_expectation_Y_given_X = Y.mean()
print(f"\nConditional Expectation E[Y|X]: {conditional_expectation_Y_given_X:.4f}")

# 8. Data Visualization for Random Variables

# Bernoulli Distribution Plot
plt.figure(figsize=(6, 4))
sns.countplot(x=bernoulli_data)
plt.title("Bernoulli Distribution (p=0.5)")
plt.show()

# Binomial Distribution Plot
plt.figure(figsize=(6, 4))
sns.histplot(binomial_data, bins=10, kde=True)
plt.title("Binomial Distribution (n=10, p=0.5)")
plt.show()

# Poisson Distribution Plot
plt.figure(figsize=(6, 4))
sns.histplot(poisson_data, bins=10, kde=True)
plt.title("Poisson Distribution (λ=3)")
plt.show()

# Geometric Distribution Plot
plt.figure(figsize=(6, 4))
sns.histplot(geom_data, bins=15, kde=True)
plt.title("Geometric Distribution (p=0.5)")
plt.show()

# Normal Distribution Plot
plt.figure(figsize=(6, 4))
sns.histplot(normal_data, bins=30, kde=True)
plt.title("Normal Distribution (μ=0, σ=1)")
plt.show()

