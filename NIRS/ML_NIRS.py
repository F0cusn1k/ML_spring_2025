import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score, confusion_matrix, roc_curve
import time
import warnings

warnings.filterwarnings("ignore")

# 1. Настройка страницы Streamlit
st.set_page_config(
    page_title="Предсказание дохода (>50K) с XGBoost",
    page_icon="🚀",
    layout="wide"
)
st.title("Предсказание дохода (>50K USD) с использованием XGBoost")
st.markdown("### Модель: XGBoost Classifier")


df = pd.read_csv('adult_ready.csv')
df_app, features_app_list = df, df.drop('income', axis=1).columns.tolist()
target_var_app_name = 'income'

# 2. Боковая панель с настройками
st.sidebar.header("Параметры XGBoost")
n_estimators_val = st.sidebar.slider("Количество деревьев (n_estimators)", 50, 500, 100, 10)
learning_rate_val = st.sidebar.select_slider(
    "Скорость обучения (learning_rate)",
    options=[0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
    value=0.1
)
max_depth_val = st.sidebar.slider("Макс. глубина деревьев (max_depth)", 2, 10, 5, 1)
subsample_val = st.sidebar.slider("Доля выборки для дерева (subsample)", 0.5, 1.0, 0.8, 0.05)
colsample_bytree_val = st.sidebar.slider("Доля признаков для дерева (colsample_bytree)", 0.5, 1.0, 0.8, 0.05)
gamma_val = st.sidebar.slider("Минимальное снижение потерь для разделения (gamma)", 0.0, 0.5, 0.0, 0.05)
min_child_weight_val = st.sidebar.slider("Минимальный вес потомка (min_child_weight)", 1, 10, 1, 1)

# Учет дисбаланса классов
use_scale_pos_weight = st.sidebar.checkbox("Использовать scale_pos_weight для баланса классов", value=True)
calculated_scale_pos_weight = 1.0
if use_scale_pos_weight and target_var_app_name in df_app.columns:
    counts = df_app[target_var_app_name].value_counts()
    if 1 in counts and 0 in counts and counts[1] > 0:
        calculated_scale_pos_weight = counts[0] / counts[1]
    st.sidebar.caption(f"Расчетный scale_pos_weight: {calculated_scale_pos_weight:.2f}")


st.sidebar.header("Параметры обучения")
test_size_val = st.sidebar.slider("Размер тестовой выборки", 0.1, 0.5, 0.3, 0.05)
train_button = st.sidebar.button("Обучить XGBoost и показать результаты")

# 3. Вкладки для отображения
tab_data, tab_results = st.tabs(["Обзор данных", "Результаты моделирования XGBoost"])

with tab_data:
    st.header("Обзор предоставленных данных")
    st.markdown(f"**Признаки для модели ({len(features_app_list)}):** ` {', '.join(features_app_list)} `")
    st.markdown(f"**Целевая переменная:** `{target_var_app_name}` (0: <=50K, 1: >50K)")
    st.subheader("Первые 5 строк")
    st.dataframe(df_app.head())

    st.subheader("Описательная статистика признаков")
    st.dataframe(df_app[features_app_list].describe())
    
    st.subheader(f"Распределение целевой переменной '{target_var_app_name}'")
    fig_dist_target, ax_dist_target = plt.subplots(figsize=(6,4))
    sns.countplot(x=target_var_app_name, data=df_app, ax=ax_dist_target, palette=["#66b3ff","#ff9999"])
    ax_dist_target.set_title(f"Распределение '{target_var_app_name}'")
    st.pyplot(fig_dist_target)

    st.subheader("Распределения признаков")
    num_features_to_plot = len(features_app_list)
    cols_per_row = 3
    num_rows = (num_features_to_plot - 1) // cols_per_row + 1
    fig_features_dist, axes = plt.subplots(num_rows, cols_per_row, figsize=(5 * cols_per_row, 4 * num_rows))
    axes = axes.flatten() # Для удобного итерирования
    for i, col_name in enumerate(features_app_list):
        sns.histplot(df_app[col_name], kde=True, ax=axes[i], bins=20)
        axes[i].set_title(f"Распределение '{col_name}'", fontsize=10)
        axes[i].tick_params(axis='both', which='major', labelsize=8)
    # Скрываем пустые сабплоты
    for j in range(i + 1, len(axes)):
        fig_features_dist.delaxes(axes[j])
    plt.tight_layout()
    st.pyplot(fig_features_dist)

    st.subheader("Матрица корреляций признаков")
    corr_matrix_app = df_app[features_app_list].corr()
    fig_corr_app, ax_corr_app = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix_app, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_corr_app, annot_kws={"size": 8})
    ax_corr_app.tick_params(axis='both', which='major', labelsize=8)
    st.pyplot(fig_corr_app)


if 'model_trained_app_xgb' not in st.session_state:
    st.session_state.model_trained_app_xgb = False

if train_button:
    st.session_state.model_trained_app_xgb = False
    with tab_results:
        st.info(f"Начинаем обучение XGBoost...")
        
        X = df_app[features_app_list]
        y = df_app[target_var_app_name]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size_val, random_state=52, stratify=y
        )
        
        st.write(f"Размер обучающей выборки: {X_train.shape[0]}, Тестовой: {X_test.shape[0]}")

        model_app = XGBClassifier(
            n_estimators=n_estimators_val,
            learning_rate=learning_rate_val,
            max_depth=max_depth_val,
            subsample=subsample_val,
            colsample_bytree=colsample_bytree_val,
            gamma=gamma_val,
            min_child_weight=min_child_weight_val,
            scale_pos_weight=calculated_scale_pos_weight if use_scale_pos_weight else 1.0,
            random_state=52,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        start_time_train = time.time()
        model_app.fit(X_train, y_train)
        end_time_train = time.time()
        status_text.success(f"Обучение XGBoost завершено за {end_time_train - start_time_train:.2f} сек.")

        y_pred_model = model_app.predict(X_test)
        y_pred_proba_model = model_app.predict_proba(X_test)[:, 1]

        st.session_state.y_test_model_xgb = y_test
        st.session_state.y_pred_model_xgb = y_pred_model
        st.session_state.y_pred_proba_model_xgb = y_pred_proba_model
        st.session_state.model_object_app_xgb = model_app
        st.session_state.features_used_app_xgb = features_app_list
        st.session_state.model_trained_app_xgb = True
        st.balloons()

if st.session_state.model_trained_app_xgb:
    with tab_results:
        if not train_button:
            st.info("Отображаются результаты предыдущего обучения XGBoost. Для переобучения измените параметры и нажмите кнопку.")

        y_test_res = st.session_state.y_test_model_xgb
        y_pred_res = st.session_state.y_pred_model_xgb
        y_pred_proba_res = st.session_state.y_pred_proba_model_xgb
        model_res = st.session_state.model_object_app_xgb
        features_res = st.session_state.features_used_app_xgb

        st.subheader("Метрики качества XGBoost на тестовой выборке")
        roc_auc_res = roc_auc_score(y_test_res, y_pred_proba_res)
        f1_res = f1_score(y_test_res, y_pred_res, pos_label=1)
        recall_res = recall_score(y_test_res, y_pred_res, pos_label=1)
        accuracy_res = accuracy_score(y_test_res, y_pred_res)

        col_m1, col_m2 = st.columns(2)
        col_m1.metric("ROC AUC", f"{roc_auc_res:.4f}")
        col_m2.metric("Accuracy", f"{accuracy_res:.4f}")
        col_m3, col_m4 = st.columns(2)
        col_m3.metric("F1-score (для >50K)", f"{f1_res:.4f}")
        col_m4.metric("Recall (для >50K)", f"{recall_res:.4f}")
        
        st.subheader("Матрица ошибок")
        cm_res = confusion_matrix(y_test_res, y_pred_res)
        fig_cm_res, ax_cm_res = plt.subplots(figsize=(5,4))
        sns.heatmap(cm_res, annot=True, fmt='d', cmap='viridis', ax=ax_cm_res)
        ax_cm_res.set_xlabel("Предсказанные")
        ax_cm_res.set_ylabel("Истинные")
        st.pyplot(fig_cm_res)

        st.subheader("ROC-кривая")
        fpr_res, tpr_res, _ = roc_curve(y_test_res, y_pred_proba_res, pos_label=1)
        fig_roc_res, ax_roc_res = plt.subplots(figsize=(6,4))
        ax_roc_res.plot(fpr_res, tpr_res, color='purple', lw=2, label=f'ROC AUC = {roc_auc_res:.3f}')
        ax_roc_res.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
        ax_roc_res.set_xlabel('False Positive Rate')
        ax_roc_res.set_ylabel('True Positive Rate')
        ax_roc_res.legend(loc="lower right")
        st.pyplot(fig_roc_res)

        st.subheader("Важность признаков (XGBoost)")
        importance_df = pd.DataFrame({
            'Признак': features_res,
            'Важность': model_res.feature_importances_
        }).sort_values('Важность', ascending=False)

        fig_imp_res, ax_imp_res = plt.subplots(figsize=(8, max(5, len(features_res) * 0.3)))
        sns.barplot(x='Важность', y='Признак', data=importance_df, ax=ax_imp_res, palette='mako')
        ax_imp_res.tick_params(axis='y', labelsize=8)
        st.pyplot(fig_imp_res)

elif not train_button:
     with tab_results:
        st.warning("Нажмите кнопку 'Обучить XGBoost...' в боковой панели, чтобы увидеть результаты.")

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **О приложении (XGBoost):**
    Демонстрация модели XGBoost для предсказания дохода
    на основе предобработанных данных Adult Census.
    Настройте гиперпараметры и обучите модель.
    """
)