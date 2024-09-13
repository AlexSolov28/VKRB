import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import randint, uniform


# Основная функция приложения
def main():

    # Сайдбар для навигации по страницам
    page = st.sidebar.selectbox(
        "Выберите страницу:",
        [
            "Загрузка данных",
            "Анализ датасета",
            "Масштабирование данных и корреляционный анализ",
            "Разделение данных",
            "Оценка моделей",
            "Сравнение моделей",
        ],
    )

    data = pd.read_csv("data_1.csv")

    if page == "Загрузка данных":
        st.title(
            "Система анализа алгоритмов машинного обучения для решения задач классификации с использованием Pandas"
        )
        st.subheader(
            "В качестве предметной области был выбран набор данных, содержащий информацию о численности экономически активного населения, безработных, уровне безработницы и сопоставляющий эти покказатели между различными возрастными группами по субъектам РФ."
        )
        st.header("Загрузка данных")
        if st.checkbox("Показать все данные"):
            st.write(data)

    elif page == "Анализ датасета":
        st.header("Анализ датасета")
        st.markdown("Первые 5 значений")
        st.write(data.head())
        st.markdown("Размер датасета:")
        st.write(data.shape)
        st.markdown("Столбцы:")
        st.write(data.columns)

    elif page == "Масштабирование данных и корреляционный анализ":
        st.header("Масштабирование данных")
        # Числовые колонки для масштабирования
        scale_cols = [
            "Численность населения",
            "Занятые в экономике",
            "Безработные",
            "Уровень экономической активности",
            "Уровень занятости",
            "Уровень безработицы",
            "Численность безработных (до 20 лет)",
            "Численность безработных (от 20 до 29 лет)",
            "Численность безработных (от 30 до 39 лет)",
            "Численность безработных (от 40 до 49 лет)",
            "Численность безработных (от 50 до 59 лет)",
            "Численность безработных (60 и более лет)",
        ]

        # Преобразование значения признаков таким образом,
        # чтобы они находились в диапазоне от 0 до 1
        sc = MinMaxScaler()
        sc_data = sc.fit_transform(data[scale_cols])

        # Добавим масштабированные данные в набор данных
        for i in range(len(scale_cols)):
            col = scale_cols[i]
            new_col_name = col + "_scaled"
            data[new_col_name] = sc_data[:, i]

        # Проверим, что масштабирование не повлияло на распределение данных
        if st.checkbox("Показать данные"):
            for col in scale_cols:
                col_scaled = col + "_scaled"
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                ax[0].hist(data[col], bins=50, alpha=0.7, color="blue")
                ax[1].hist(data[col_scaled], bins=50, alpha=0.7, color="orange")
                ax[0].set_title(f"Ориг_данные: {col}")
                ax[1].set_title(f"Масшта_данные: {col_scaled}")
                st.pyplot(fig)

        st.header("Корреляционный анализ данных")
        # Воспользуемся наличием тестовых выборок,
        # включив их в корреляционную матрицу
        corr_cols_1 = scale_cols + ["Год"]
        corr_cols_postfix = [x + "_scaled" for x in scale_cols]
        corr_cols_2 = corr_cols_postfix + ["Год"]

        if st.checkbox("Показать исходные данные (до масштабирования)"):
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(data[corr_cols_1].corr(), annot=True, fmt=".2f")
            ax.set_title("Исходные данные (до масштабирования)")
            st.pyplot(fig)

        if st.checkbox("Показать масштабированные данные"):
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(data[corr_cols_2].corr(), annot=True, fmt=".2f")
            ax.set_title("Масштабированные данные")
            st.pyplot(fig)

    elif page == "Разделение данных":
        # Формируем обучающие и тестовые выборки
        st.header("Разделение данных на обучающую и тестовую выборки")
        X = data[
            ["Уровень занятости", "Занятые в экономике", "Численность населения"]
        ]  # Наименование признаков
        y = data["Год"]  # Целевая переменная

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )

        # Размер обучающей выборки
        st.write("Размер обучающей выборки", X_train.shape, y_train.shape)
        # Размер тестовой выборки
        st.write("Размер тестовой выборки", X_train.shape, y_train.shape)

    elif page == "Оценка моделей":
        st.header("Оценка обученных моделей")
        X = data[
            ["Уровень занятости", "Занятые в экономике", "Численность населения"]
        ]  # Наименование признаков
        y = data["Год"]  # Целевая переменная

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )

        # Создание и обучение модели логистической регрессии
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Предсказание и оценка производительности
        accuracy_model = accuracy_score(y_test, y_pred)
        precision_model = precision_score(y_test, y_pred)
        recall_model = recall_score(y_test, y_pred)
        f1_model = f1_score(y_test, y_pred)

        # Создание и обучение модели К-ближайших соседей
        model2 = KNeighborsClassifier()
        model2.fit(X_train, y_train)
        y_pred2 = model2.predict(X_test)

        # Предсказание и оценка производительности
        accuracy_model2 = accuracy_score(y_test, y_pred2)
        precision_model2 = precision_score(y_test, y_pred2)
        recall_model2 = recall_score(y_test, y_pred2)
        f1_model2 = f1_score(y_test, y_pred2)

        # Создание и обучение модели случайный лес
        model3 = RandomForestClassifier()
        model3.fit(X_train, y_train)
        y_pred3 = model3.predict(X_test)

        # Предсказание и оценка производительности
        accuracy_model3 = accuracy_score(y_test, y_pred3)
        precision_model3 = precision_score(y_test, y_pred3)
        recall_model3 = recall_score(y_test, y_pred3)
        f1_model3 = f1_score(y_test, y_pred3)

        # результаты моделей
        results = {
            "Logistic Regression": {
                "Accuracy": {accuracy_model},
                "Precision": {precision_model},
                "Recall": {recall_model},
                "F1 Score": {f1_model},
            },
            "KNN": {
                "Accuracy": {accuracy_model2},
                "Precision": {precision_model2},
                "Recall": {recall_model2},
                "F1 Score": {f1_model2},
            },
            "Random Forest": {
                "Accuracy": {accuracy_model3},
                "Precision": {precision_model3},
                "Recall": {recall_model3},
                "F1 Score": {f1_model3},
            },
        }

        # Список моделей
        model_names = list(results.keys())

        # Выбор моделей в сайдбаре
        selected_model = st.sidebar.selectbox(
            "Выберите модель для оценки:", model_names
        )

        # Отображение результатов выбранной модели
        st.write(f"Модель: {selected_model}")
        st.write(f"Accuracy: {results[selected_model]['Accuracy']}")
        st.write(f"Precision: {results[selected_model]['Precision']}")
        st.write(f"Recall: {results[selected_model]['Recall']}")
        st.write(f"F1 Score: {results[selected_model]['F1 Score']}")

    elif page == "Сравнение моделей":
        st.header("Сравнение моделей")
        # Результаты моделей в виде графиков
        results = {
            "Logistic Regression": {
                "Accuracy": 0.8108,
                "Precision": 0.8261,
                "Recall": 0.8636,
                "F1 Score": 0.8444,
            },
            "KNN": {
                "Accuracy": 0.5405,
                "Precision": 0.6471,
                "Recall": 0.5,
                "F1 Score": 0.5641,
            },
            "Random Forest": {
                "Accuracy": 0.4324,
                "Precision": 0.5294,
                "Recall": 0.4090,
                "F1 Score": 0.4615,
            },
        }
        # Построение графиков для каждой метрики
        metrics = list(results["Logistic Regression"].keys())
        for metric in metrics:
            values = [results[model][metric] for model in results]
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(results.keys(), values, color=["blue", "green", "red"])
            ax.set_title(f"{metric} Сравнение")
            ax.set_xlabel("Model")
            ax.set_ylabel(metric)
            ax.set_ylim(0, 1)  # Установка границ для оси y
            st.pyplot(fig)


# Запуск основного приложения
if __name__ == "__main__":
    main()
