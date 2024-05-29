import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Генерация выборок
np.random.seed(42)

# 1. Равномерное распределение на отрезке [a, b]
a, b = 0, 1
uniform_sample = np.random.uniform(a, b, 1000)

# 2. Нормальное распределение с параметрами µ и σ²
mu, sigma = 0, 1
normal_sample = np.random.normal(mu, sigma, 1000)

# 3. Экспоненциальное распределение с параметром λ
lambda_exp = 1
exponential_sample = np.random.exponential(1/lambda_exp, 1000)

# Построение гистограмм выборок
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(uniform_sample, bins=30, density=True, alpha=0.6, color='g')
plt.title('Uniform Distribution')

plt.subplot(1, 3, 2)
plt.hist(normal_sample, bins=30, density=True, alpha=0.6, color='b')
plt.title('Normal Distribution')

plt.subplot(1, 3, 3)
plt.hist(exponential_sample, bins=30, density=True, alpha=0.6, color='r')
plt.title('Exponential Distribution')

plt.show()

# Критерий Колмогорова-Смирнова для равномерного распределения
ks_stat_uniform, p_value_uniform = stats.kstest(uniform_sample, 'uniform', args=(a, b-a))

# Критерий Колмогорова-Смирнова для нормального распределения
ks_stat_normal, p_value_normal = stats.kstest(normal_sample, 'norm', args=(mu, sigma))

# Критерий Колмогорова-Смирнова для экспоненциального распределения
ks_stat_exponential, p_value_exponential = stats.kstest(exponential_sample, 'expon', args=(0, 1/lambda_exp))

# Печать результатов
print(f'Uniform Distribution: KS Statistic = {ks_stat_uniform}, p-value = {p_value_uniform}')
print(f'Normal Distribution: KS Statistic = {ks_stat_normal}, p-value = {p_value_normal}')
print(f'Exponential Distribution: KS Statistic = {ks_stat_exponential}, p-value = {p_value_exponential}')


# Критерий Колмогорова-Смирнова для проверки экспоненциального распределения на принадлежность к нормальному
ks_stat_exp_to_norm, p_value_exp_to_norm = stats.kstest(exponential_sample, 'norm', args=(mu, sigma))

# Печать результатов
print(f'Exponential to Normal Distribution: KS Statistic = {ks_stat_exp_to_norm}, p-value = {p_value_exp_to_norm}')

# Проверка гипотезы о среднем при известной дисперсии
known_variance = sigma**2
sample_mean = np.mean(normal_sample)
sample_size = len(normal_sample)
population_mean = 0

z_stat = (sample_mean - population_mean) / (sigma / np.sqrt(sample_size))
p_value_z = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

# Проверка гипотезы о среднем при неизвестной дисперсии
t_stat, p_value_t = stats.ttest_1samp(normal_sample, population_mean)

# Печать результатов
print(f'Z-test (known variance): Z Statistic = {z_stat}, p-value = {p_value_z}')
print(f'T-test (unknown variance): T Statistic = {t_stat}, p-value = {p_value_t}')

# Функция для определения критического уровня значимости
def find_critical_significance_level(statistic, dist='norm'):
    critical_value = 1 - stats.norm.cdf(statistic)
    return 2 * critical_value

# Критические уровни значимости для каждой из гипотез
critical_level_uniform = find_critical_significance_level(ks_stat_uniform)
critical_level_normal = find_critical_significance_level(ks_stat_normal)
critical_level_exponential = find_critical_significance_level(ks_stat_exponential)
critical_level_exp_to_norm = find_critical_significance_level(ks_stat_exp_to_norm)
critical_level_z_test = find_critical_significance_level(np.abs(z_stat))
critical_level_t_test = find_critical_significance_level(np.abs(t_stat), dist='t')

print(f'Critical level (Uniform Distribution): {critical_level_uniform}')
print(f'Critical level (Normal Distribution): {critical_level_normal}')
print(f'Critical level (Exponential Distribution): {critical_level_exponential}')
print(f'Critical level (Exponential to Normal Distribution): {critical_level_exp_to_norm}')
print(f'Critical level (Z-test): {critical_level_z_test}')
print(f'Critical level (T-test): {critical_level_t_test}')
