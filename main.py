import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import ndcg_score

# Загрузка тренировочного и тестового датасетов из csv файлов
train_df = pd.read_csv('data/train_df.csv')
test_df = pd.read_csv('data/test_df.csv')

# Получение значений признаков и целевой переменной из тренировочного датасета
X_train = train_df.drop('target', axis=1).values
y_train = train_df['target'].values

# Получение значений признаков и целевой переменной из тестового датасета
X_test = test_df.drop('target', axis=1).values
y_test = test_df['target'].values

# Обучение модели
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Предсказание релевантностей на тестовом датасете
y_pred = model.predict(X_test)

# Расчет метрики NDCG на тестовом датасете
ndcg = ndcg_score([y_test], [y_pred])
print(f"\nNDCG на тестовом датасете:{ndcg}")