import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import NearMiss
from sklearn.metrics import accuracy_score, f1_score, precision_score

# Функция для предсказания и оценки модели
def Prediction(model, X_test, y_test):
    y_pred = model.predict(X_test)
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    st.write(f"F1: {f1_score(y_test, y_pred)}")
    st.write(f"Precision: {precision_score(y_test, y_pred)}")

# Загрузка датасета
data = st.file_uploader("Выберите файл датасета", type=["csv"])
if data is not None:
    st.header("Датасет")
    df = pd.read_csv(data)
    st.dataframe(df)

    st.write("---")

    # Выбор типа модели обучения
    st.title("Тип модели обучения")
    model_type = st.selectbox("Выберите тип", ['Knn', 'Boosting', 'Bagging', 'Stacking'])

    # Выбор признака для предсказания
    st.title("Выбор признака")
    feature = st.selectbox("Выберите признак", df.columns)

    button_clicked = st.button("Обработка данных и предсказание")
    if button_clicked:
        st.header("Обработка данных")

        # Преобразование категориальных признаков в числовые значения
        if df[feature].dtype == 'object':
            df[feature] = df[feature].astype('category').cat.codes

        df = df.drop_duplicates()

        # Обработка пропусков значений
        for col in df.columns[:-1]:
            df[col] = df[col].fillna(df[col].mean())  # Заполняем пропуски средними значениями

        # Удаление выбросов
        outlier = df[df.columns[:-1]]
        Q1 = outlier.quantile(0.25)
        Q3 = outlier.quantile(0.75)
        IQR = Q3 - Q1
        data_filtered = outlier[~((outlier < (Q1 - 1.5 * IQR)) | (outlier > (Q3 + 1.5 * IQR))).any(axis=1)]
        index_list = list(data_filtered.index.values)
        df = df[df.index.isin(index_list)]

        st.write("Очистка от выбросов: выполнено")

        # Масштабирование данных
        scaler = StandardScaler()
        X = df.drop([feature], axis=1)
        X = scaler.fit_transform(X)
        st.write("Масштабирование: выполнено")

        # Разделение на целевой признак и признаки
        y = df[feature]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Балансировка данных
        nm = NearMiss(n_neighbors=1)  # Уменьшаем количество соседей до 1, чтобы избежать ошибки
        try:
            X_resampled, y_resampled = nm.fit_resample(X_train, y_train)
        except ValueError as e:
            st.error(f"Ошибка при балансировке данных: {e}")
            X_resampled, y_resampled = X_train, y_train  # Используем оригинальные данные в случае ошибки

        st.write("Балансировка целевого признака: выполнено")

        st.success("Обработка завершена")

        st.header("Предсказание")

        # Предсказание в зависимости от выбранной модели
        if model_type is not None:
            try:
                if model_type == "Knn":
                    with open('model/knn.pkl', 'rb') as file:
                        knn_model = pickle.load(file)
                    Prediction(knn_model, X_test, y_test)

                elif model_type == "Boosting":
                    with open('model/boosting.pkl', 'rb') as file:
                        boosting_model = pickle.load(file)
                    Prediction(boosting_model, X_test, y_test)

                elif model_type == "Bagging":
                    with open('model/bagging.pkl', 'rb') as file:
                        bagging_model = pickle.load(file)
                    Prediction(bagging_model, X_test, y_test)

                elif model_type == "Stacking":
                    with open('model/stacking.pkl', 'rb') as file:
                        stacking_model = pickle.load(file)
                    Prediction(stacking_model, X_test, y_test)

            except FileNotFoundError:
                st.error(f"Модель {model_type} не найдена. Загрузите соответствующий файл.")
            except Exception as e:
                st.error(f"Ошибка при предсказании: {e}")
