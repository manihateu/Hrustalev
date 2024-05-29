import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Параметры для распределений
a, b = -1, 1
mu, sigma2 = 0, 7
lambda_ = 5
n_values = [50, 100, 1000]
confidence_levels = [0.9, 0.99]

# Функция для построения доверительных интервалов для равномерного распределения
def ci_uniform(sample, a, confidence_level):
    n = len(sample)
    theta_hat = max(sample) - a
    alpha = 1 - confidence_level
    lower_bound = a + theta_hat * (1 - (1 - alpha / 2) ** (1 / n))
    upper_bound = a + theta_hat * (1 - (alpha / 2) ** (1 / n))
    return lower_bound, upper_bound

# Функция для построения доверительных интервалов для нормального распределения
def ci_normal(sample, sigma2=None, mu=None, confidence_level=0.95):
    n = len(sample)
    alpha = 1 - confidence_level
    mean_sample = np.mean(sample)
    var_sample = np.var(sample, ddof=1)

    if sigma2 is not None:  # sigma2 known
        se = np.sqrt(sigma2 / n)
        z = stats.norm.ppf(1 - alpha / 2)
        mean_ci = (mean_sample - z * se, mean_sample + z * se)
    else:  # sigma2 unknown
        se = np.sqrt(var_sample / n)
        t = stats.t.ppf(1 - alpha / 2, df=n - 1)
        mean_ci = (mean_sample - t * se, mean_sample + t * se)

    if mu is not None:  # mu known
        se = np.sqrt((sample - mu) ** 2).mean() / np.sqrt(n)
        chi2_lower = stats.chi2.ppf(alpha / 2, df=n)
        chi2_upper = stats.chi2.ppf(1 - alpha / 2, df=n)
        var_ci = (var_sample * (n - 1) / chi2_upper, var_sample * (n - 1) / chi2_lower)
    else:  # mu unknown
        se = np.sqrt(var_sample / n)
        chi2_lower = stats.chi2.ppf(alpha / 2, df=n - 1)
        chi2_upper = stats.chi2.ppf(1 - alpha / 2, df=n - 1)
        var_ci = (var_sample * (n - 1) / chi2_upper, var_sample * (n - 1) / chi2_lower)

    return mean_ci, var_ci

# Функция для построения доверительных интервалов для экспоненциального распределения
def ci_exponential(sample, confidence_level):
    n = len(sample)
    alpha = 1 - confidence_level
    lambda_hat = 1 / np.mean(sample)
    chi2_lower = stats.chi2.ppf(alpha / 2, df=2 * n)
    chi2_upper = stats.chi2.ppf(1 - alpha / 2, df=2 * n)
    lower_bound = 2 * n / chi2_upper
    upper_bound = 2 * n / chi2_lower
    return lower_bound, upper_bound

# Генерация выборок и построение доверительных интервалов
for n in n_values:
    # Равномерное распределение
    uniform_sample = np.random.uniform(a, b, n)
    for confidence_level in confidence_levels:
        ci_theta = ci_uniform(uniform_sample, a, confidence_level)
        print(f'Равномерное распределение (n={n}, доверительная вероятность={confidence_level}): θ ∈ {ci_theta}')

    # Нормальное распределение
    normal_sample = np.random.normal(mu, np.sqrt(sigma2), n)
    for confidence_level in confidence_levels:
        ci_mu_known_sigma = ci_normal(normal_sample, sigma2=sigma2, confidence_level=confidence_level)[0]
        ci_mu_unknown_sigma = ci_normal(normal_sample, confidence_level=confidence_level)[0]
        ci_sigma2_known_mu = ci_normal(normal_sample, mu=mu, confidence_level=confidence_level)[1]
        ci_sigma2_unknown_mu = ci_normal(normal_sample, confidence_level=confidence_level)[1]
        print(f'Нормальное распределение (n={n}, доверительная вероятность={confidence_level}):')
        print(f'  µ (σ² известно) ∈ {ci_mu_known_sigma}')
        print(f'  µ (σ² неизвестно) ∈ {ci_mu_unknown_sigma}')
        print(f'  σ² (µ известно) ∈ {ci_sigma2_known_mu}')
        print(f'  σ² (µ неизвестно) ∈ {ci_sigma2_unknown_mu}')

    # Экспоненциальное распределение
    exponential_sample = np.random.exponential(1 / lambda_, n)
    for confidence_level in confidence_levels:
        ci_lambda = ci_exponential(exponential_sample, confidence_level)
        print(f'Экспоненциальное распределение (n={n}, доверительная вероятность={confidence_level}): λ ∈ {ci_lambda}')
