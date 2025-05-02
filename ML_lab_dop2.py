import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time


st.set_page_config(
    page_title="Предсказание шанса поступления",
    page_icon="🎓",
    layout="wide"
)


st.title("Предсказание шанса поступления в университет")
st.markdown("### Модель: Random Forest для регрессии")


@st.cache_data
def load_data():
    file_path = 'Admission_Predict_Ver1.1.csv'

    df = pd.read_csv(file_path)
    df = df.drop('Serial No.', axis=1)

    df.columns = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP',
                    'LOR', 'CGPA', 'Research', 'Chance of Admit']

    return df


df = load_data()
target_variable = 'Chance of Admit'


st.sidebar.header("Настройка параметров модели")


st.sidebar.subheader("Гиперпараметры Random Forest")
n_estimators = st.sidebar.slider("Количество деревьев (n_estimators)", 10, 500, 100, 10)
max_depth = st.sidebar.slider("Макс. глубина деревьев (max_depth)", 1, 30, 10, 1)
min_samples_split = st.sidebar.slider("Мин. образцов для разделения (min_samples_split)", 2, 20, 2, 1)
min_samples_leaf = st.sidebar.slider("Мин. образцов в листе (min_samples_leaf)", 1, 20, 1, 1)

max_features_options = {
    "sqrt": "sqrt",
    "log2": "log2",
    "1/3 признаков": 1/3,
    "1/2 признаков": 0.5,
    "2/3 признаков": 2/3,
    "Все признаки": 1.0,
}
max_features_selection = st.sidebar.selectbox(
    "Макс. количество признаков (max_features)",
    list(max_features_options.keys()),
    index=0 # По умолчанию 'sqrt'
)
max_features = max_features_options[max_features_selection]


test_size = st.sidebar.slider("Размер тестовой выборки", 0.1, 0.5, 0.2, 0.05)


train_button = st.sidebar.button("Обучить модель")


tab1, tab2, tab3 = st.tabs(["Данные", "Обучение модели", "Результаты"])

if df.empty:
    st.error("Данные не были загружены. Пожалуйста, проверьте путь к файлу и его содержимое.")


with tab1:
    st.header("Обзор данных")

    st.subheader("Первые 5 строк датасета")
    st.dataframe(df.head())

    st.subheader("Описательная статистика")
    st.dataframe(df.describe())

    st.subheader("Пропущенные значения")
    missing_values = df.isnull().sum()
    st.dataframe(missing_values[missing_values > 0])
    if missing_values.sum() == 0:
        st.success("Пропущенные значения не обнаружены.")
    else:
        st.warning("Обнаружены пропущенные значения. Рекомендуется их обработать.")


    st.subheader(f"Распределение целевой переменной: {target_variable}")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[target_variable], kde=True, ax=ax, bins=30)
    ax.set_title(f"Распределение '{target_variable}'")
    ax.set_xlabel("Шанс поступления")
    ax.set_ylabel("Частота")
    st.pyplot(fig)

    st.subheader("Матрица корреляций")
    numeric_cols = df.select_dtypes(include=np.number).columns
    if not numeric_cols.empty:
            correlation = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
            ax.set_title("Матрица корреляций признаков")
            st.pyplot(fig)
    else:
            st.warning("Не найдено числовых столбцов для расчета корреляции.")


with tab2:
    st.header("Обучение модели")
    if train_button:
        st.info("Начинаем подготовку данных и обучение модели...")

        try:
            if target_variable not in df.columns:
                st.error(f"Целевая переменная '{target_variable}' не найдена в столбцах DataFrame.")
                st.stop()

            X = df.drop(target_variable, axis=1)
            y = df[target_variable]

            st.write("Признаки, используемые для обучения:", X.columns.tolist())
            st.write(f"Целевая переменная: {target_variable}")


            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            st.write(f"Размер обучающей выборки: {X_train.shape[0]} записей")
            st.write(f"Размер тестовой выборки: {X_test.shape[0]} записей")

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            st.info("Стандартизация признаков выполнена.")

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=42,
                n_jobs=-1
            )

            progress_bar = st.progress(0)
            status_text = st.empty()

            start_time = time.time()
            for i in range(101):
                progress_bar.progress(i)
                status_text.text(f"Обучение: {i}%...")
                time.sleep(0.005)

            model.fit(X_train_scaled, y_train)
            end_time = time.time()
            status_text.text(f"Обучение завершено за {end_time - start_time:.2f} сек.")

            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)

            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.y_pred_train = y_pred_train
            st.session_state.y_pred_test = y_pred_test
            st.session_state.feature_names = X.columns.tolist()
            st.session_state.target_variable_name = target_variable
            st.session_state.trained = True

            st.success("Модель успешно обучена!")
            st.balloons()

        except Exception as e:
            st.error(f"Ошибка при обучении модели: {e}")
            st.exception(e)
    else:
        st.warning("Нажмите кнопку 'Обучить модель' в боковой панели для начала обучения.")

with tab3:
    st.header("Результаты обучения")

    if 'trained' in st.session_state and st.session_state.trained:
        y_train = st.session_state.y_train
        y_pred_train = st.session_state.y_pred_train
        y_test = st.session_state.y_test
        y_pred_test = st.session_state.y_pred_test
        model = st.session_state.model
        feature_names = st.session_state.feature_names
        target_var_name = st.session_state.target_variable_name


        st.subheader("Метрики качества модели")

        train_mse = mean_squared_error(y_train, y_pred_train)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_r2 = r2_score(y_train, y_pred_train)

        test_mse = mean_squared_error(y_test, y_pred_test)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)

        metrics_data = {
            "Метрика": ["MSE (Mean Squared Error)", "RMSE (Root Mean Squared Error)", "MAE (Mean Absolute Error)", "R² (Coefficient of Determination)"],
            "Обучающая выборка": [f"{train_mse:.4f}", f"{train_rmse:.4f}", f"{train_mae:.4f}", f"{train_r2:.4f}"],
            "Тестовая выборка": [f"{test_mse:.4f}", f"{test_rmse:.4f}", f"{test_mae:.4f}", f"{test_r2:.4f}"]
        }
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df.set_index("Метрика"))

        st.subheader("Сравнение фактических и предсказанных значений")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Обучающая выборка**")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_train, y_pred_train, alpha=0.5, label='Предсказания')
            ax.plot([y_train.min(), y_train.max()],
                    [y_train.min(), y_train.max()],
                    'r--', label='Идеальная линия')
            ax.set_xlabel(f'Фактический {target_var_name}')
            ax.set_ylabel(f'Предсказанный {target_var_name}')
            ax.set_title('Обучающая выборка: Факт vs Предсказание')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        with col2:
            st.markdown("**Тестовая выборка**")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test, y_pred_test, alpha=0.5, label='Предсказания')
            ax.plot([y_test.min(), y_test.max()],
                    [y_test.min(), y_test.max()],
                    'r--', label='Идеальная линия')
            ax.set_xlabel(f'Фактический {target_var_name}')
            ax.set_ylabel(f'Предсказанный {target_var_name}')
            ax.set_title('Тестовая выборка: Факт vs Предсказание')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        st.subheader("Важность признаков (Feature Importance)")
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Признак': feature_names,
                'Важность': model.feature_importances_
            }).sort_values('Важность', ascending=False)

            fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.4)))
            sns.barplot(x='Важность', y='Признак', data=feature_importance, ax=ax, palette='viridis')
            ax.set_title('Важность признаков для модели Random Forest')
            st.pyplot(fig)
        else:
            st.warning("Модель не поддерживает атрибут 'feature_importances_'.")


        st.subheader("Анализ остатков")
        residuals_train = y_train - y_pred_train
        residuals_test = y_test - y_pred_test

        col1_res, col2_res = st.columns(2)

        with col1_res:
            st.markdown("**Распределение остатков (Тестовая выборка)**")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(residuals_test, kde=True, ax=ax, bins=30)
            ax.set_xlabel('Остатки (Факт - Предсказание)')
            ax.set_ylabel('Частота')
            ax.set_title('Гистограмма остатков (Тестовая выборка)')
            ax.axvline(0, color='red', linestyle='--')
            st.pyplot(fig)

        with col2_res:
            st.markdown("**Остатки vs Предсказанные значения (Тестовая выборка)**")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_pred_test, residuals_test, alpha=0.5)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_xlabel(f'Предсказанный {target_var_name}')
            ax.set_ylabel('Остатки')
            ax.set_title('Остатки vs Предсказанные значения (Тестовая выборка)')
            ax.grid(True)
            st.pyplot(fig)

    else:
        st.warning("Сначала обучите модель во вкладке 'Обучение модели'.")


st.sidebar.markdown("---")
st.sidebar.subheader("О приложении")
st.sidebar.info(
    """
    Это веб-приложение для демонстрации обучения модели Random Forest
    с целью предсказания шанса поступления абитуриента ('Chance of Admit').

    Приложение позволяет:
    - Загружать и просматривать данные.
    - Настраивать гиперпараметры модели Random Forest.
    - Обучать модель на данных.
    - Визуализировать результаты обучения: метрики, сравнение предсказаний,
      важность признаков и анализ остатков.
    """
)