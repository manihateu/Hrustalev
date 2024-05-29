import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, expon

# Параметры для распределений
a, b = -1, 1
mu, sigma2 = 0, 7
lambda_ = 5
n_values = [50, 100, 1000]

# Функция для генерации и анализа выборок
def analyze_distribution(distribution, params, n_values):
    for n in n_values:
        if distribution == 'uniform':
            sample = np.random.uniform(params[0], params[1], n)
            theoretical_mean = (params[0] + params[1]) / 2
            theoretical_variance = ((params[1] - params[0]) ** 2) / 12
        elif distribution == 'normal':
            sample = np.random.normal(params[0], np.sqrt(params[1]), n)
            theoretical_mean = params[0]
            theoretical_variance = params[1]
        elif distribution == 'exponential':
            sample = np.random.exponential(1 / params[0], n)
            theoretical_mean = 1 / params[0]
            theoretical_variance = 1 / (params[0] ** 2)
        else:
            raise ValueError("Unknown distribution type")

        # Рассчет эмпирических характеристик
        sample_mean = np.mean(sample)
        sample_variance = np.var(sample)
        unbiased_sample_variance = np.var(sample, ddof=1)
        second_moment = np.mean(sample ** 2)

        # Печать результатов
        print(f"Распеределение: {distribution}, n: {n}")
        print(f"Выборочное среднее: {sample_mean}, Теоретическое среднее: {theoretical_mean}")
        print(f"Выборочная дисперсия: {sample_variance}, Теоритическая дисперсия: {theoretical_variance}")
        print(f"Несмещенная выборочная дисперсия: {unbiased_sample_variance}")
        print(f"Второй выборочный момент: {second_moment}\n")

        # Построение гистограммы и эмпирической функции распределения
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.hist(sample, bins=30, density=True, alpha=0.6, color='g')
        plt.title(f'Гистограмма {distribution} (n={n})')

        plt.subplot(1, 2, 2)
        plt.hist(sample, bins=30, density=True, cumulative=True, alpha=0.6, color='g')
        plt.title(f'Эмпирическая ФР {distribution} (n={n})')
        plt.show()

# Генерация и анализ выборок
analyze_distribution('uniform', (a, b), n_values)
analyze_distribution('normal', (mu, sigma2), n_values)
analyze_distribution('exponential', (lambda_,), n_values)
