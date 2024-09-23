import pandas as pd
import numpy as np
import math
import pickle
from sklearn.model_selection import train_test_split
import streamlit as st

# Загрузка датасета
data = st.file_uploader("Выберите файл датасета", type=["csv"])

# Определение функции для предсказания и оценки модели
def Prediction(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    st.write(f'MAE: {MAE(y_test, y_pred)}')
    st.write(f'MSE: {MSE(y_test, y_pred)}')
    st.write(f'RMSE: {math.sqrt(MSE(y_test, y_pred))}')
    st.write(f'MAPE: {MAPE(y_test, y_pred)}')
    st.write(f'R^2: {model.score(X_test, y_test)}')

# Функция для расчета MSE
def MSE(y_test, y_pred):
    diff = y_pred - y_test
    diff_squar = diff ** 2
    mean_diff = diff_squar.mean()
    return mean_diff

# Функция для расчета MAE
def MAE(y_test, y_pred):
    diff = y_pred - y_test
    abs_diff = np.absolute(diff)
    mean_diff = abs_diff.mean()
    return mean_diff

# Функция для расчета MAPE
def MAPE(y_test, y_pred):
    y_test = np.where(y_test == 0, np.finfo(float).eps, y_test)  # Избегаем деления на 0
    mean_diff = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    return mean_diff

if data is not None:
    st.header("Датасет")
    df = pd.read_csv(data)
    st.dataframe(df)

    # Удаление дубликатов
    df = df.drop_duplicates()

    # Преобразование категориальных признаков в дамми-переменные
    df2 = pd.get_dummies(df)

    st.write("---")

    # Выбор признака для предсказания
    st.header("Выбор признака")
    feature = st.selectbox("Выберите признак", df2.columns)

    # Кнопка для запуска обработки данных и предсказания
    button_clicked = st.button("Обработка данных и предсказание")
    if button_clicked:

        st.header("Обработка данных")

        # Удаление дубликатов и обработка пропусков данных
        df2 = df2.drop_duplicates()
        
        # Замена пропусков на случайные значения в пределах минимального и максимального значения каждого столбца
        for i in df2.columns:
            df2[i] = df2[i].map(lambda x: np.random.uniform(df2[i].min(), df2[i].max()) if pd.isna(x) else x)

        # Удаление выбросов
        outlier = df2[df2.columns[:-1]]
        Q1 = outlier.quantile(0.25)
        Q3 = outlier.quantile(0.75)
        IQR = Q3 - Q1
        data_filtered = outlier[~((outlier < (Q1 - 1.5 * IQR)) | (outlier > (Q3 + 1.5 * IQR))).any(axis=1)]
        index_list = list(data_filtered.index.values)
        data_filtered = df2[df2.index.isin(index_list)]

        st.write("Очистка от выбросов: выполнено")

        # Выделение целевого признака (Y) и признаков (X)
        Y = df2[feature]
        X = df2.drop([feature], axis=1)

        # Разделение данных на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

        st.success("Обработка завершена")

        st.header("Предсказание")

        # Загрузка модели и выполнение предсказания
        try:
            with open('model/lasso.pkl', 'rb') as file:
                g_model = pickle.load(file)
            Prediction(g_model, X_test, y_test)
        except FileNotFoundError:
            st.error("Модель не найдена. Загрузите файл модели lasso.pkl.")
