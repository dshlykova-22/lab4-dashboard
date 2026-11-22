import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# --- 1. Загрузка и подготовка данных ---
@st.cache_data
def load_and_prepare_data():
    # Загружаем датасет (он должен лежать в той же папке, что и app.py)
    df = pd.read_csv('Econom_Cities_data.csv', sep=';', decimal=',')
    # Очистка от маркеров пропуска
    df = df.replace(-9999, np.nan).dropna()
    df = df.set_index('City')
    
    # Масштабирование
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    return df, X_scaled

df, X_scaled = load_and_prepare_data()

# --- 2. Обучение моделей и расчет метрик ---
def evaluate_clustering(X, labels, name):
    # Для DBSCAN исключаем шум (-1) из расчета метрик
    if name == 'DBSCAN':
        mask = labels != -1
        if mask.sum() < 2: # Если после удаления шума осталось < 2 точек
            return {'Алгоритм': name, 'Кластеров': 0, 'Шум': len(labels), 'Silhouette': np.nan, 'Davies-Bouldin': np.nan, 'Calinski-Harabasz': np.nan}
        X_clean = X[mask]
        labels_clean = labels[mask]
    else:
        X_clean = X
        labels_clean = labels

    n_clusters = len(set(labels_clean))
    
    if n_clusters < 2:
        return {'Алгоритм': name, 'Кластеров': n_clusters, 'Шум': 0, 'Silhouette': np.nan, 'Davies-Bouldin': np.nan, 'Calinski-Harabasz': np.nan}

    return {
        'Алгоритм': name,
        'Кластеров': n_clusters,
        'Шум': len(labels) - len(labels_clean) if name == 'DBSCAN' else 0,
        'Silhouette': round(silhouette_score(X_clean, labels_clean), 3),
        'Davies-Bouldin': round(davies_bouldin_score(X_clean, labels_clean), 3),
        'Calinski-Harabasz': round(calinski_harabasz_score(X_clean, labels_clean), 3)
    }

# Обучаем модели с параметрами из лаб 1-3
hier_labels = AgglomerativeClustering(n_clusters=4, linkage='ward').fit_predict(X_scaled)
kmeans_labels = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X_scaled)
dbscan_labels = DBSCAN(eps=0.75, min_samples=4).fit_predict(X_scaled)

# Собираем метрики в таблицу
metrics_df = pd.DataFrame([
    evaluate_clustering(X_scaled, hier_labels, 'Иерархический (Ward)'),
    evaluate_clustering(X_scaled, kmeans_labels, 'K-Means'),
    evaluate_clustering(X_scaled, dbscan_labels, 'DBSCAN')
])

# --- 3. Интерфейс дашборда ---
st.set_page_config(page_title="Анализ кластеризации городов", layout="wide")
st.title("🌍 Аналитический дашборд: Кластеризация городов")
st.markdown("Объединение результатов лабораторных работ №1-3. Сравнение трех алгоритмов машинного обучения.")

# Боковая панель
st.sidebar.header("⚙️ Настройки")
algorithm = st.sidebar.selectbox(
    "Выберите алгоритм кластеризации:",
    ('Иерархический (Ward)', 'K-Means', 'DBSCAN')
)

# Выбор меток в зависимости от алгоритма
labels_map = {
    'Иерархический (Ward)': hier_labels,
    'K-Means': kmeans_labels,
    'DBSCAN': dbscan_labels
}
selected_labels = labels_map[algorithm]

# Копируем датафрейм для фильтрации
df_plot = df.copy()
df_plot['Cluster'] = selected_labels
# Делаем текстовые метки для красивой легенды в Plotly
df_plot['Cluster_Label'] = df_plot['Cluster'].apply(lambda x: f"Кластер {x}" if x != -1 else "Шум (выброс)")

# Фильтр по кластерам
if algorithm != 'DBSCAN':
    unique_clusters = sorted(df_plot['Cluster'].unique())
    selected_clusters = st.sidebar.multiselect(
        "Кластеры для отображения:",
        unique_clusters,
        default=unique_clusters
    )
    df_filtered = df_plot[df_plot['Cluster'].isin(selected_clusters)]
else:
    df_filtered = df_plot
    st.sidebar.info("ℹ️ Для DBSCAN черным цветом выделены объекты шума (кластер -1).")

# Вкладки
tab1, tab2, tab3, tab4 = st.tabs(["📊 3D Визуализация", "📋 Сравнение алгоритмов", "📈 Характеристики групп", "🎯 Итоговые выводы"])

# --- Вкладка 1: 3D Визуализация ---
with tab1:
    st.subheader("Трехмерное распределение кластеров")
    
    # Создаем 3D-график
    fig_3d = px.scatter_3d(
        df_filtered,
        x='Price',
        y='Salary',
        z='Work',
        color='Cluster_Label',
        hover_name=df_filtered.index,
        labels={'Price': 'Индекс цен', 'Salary': 'Индекс зарплат', 'Work': 'Рабочие часы', 'Cluster_Label': 'Группа'},
        color_discrete_map={'Шум (выброс)': 'black'} # Шум всегда черный
    )
    fig_3d.update_traces(marker=dict(size=6, opacity=0.9, line=dict(width=1, color='DarkSlateGrey')))
    fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=700)
    
    st.plotly_chart(fig_3d, use_container_width=True)
    st.caption("💡 Совет: Вы можете вращать график мышкой, приближать колесиком и наводить курсор на точки, чтобы увидеть название города.")

# --- Вкладка 2: Сравнение алгоритмов ---
with tab2:
    st.subheader("Сравнительная таблица метрик качества")
    st.markdown("""
    *   **Silhouette Score (Коэффициент силуэта):** от -1 до 1. Чем ближе к 1, тем лучше объекты разделены.
    *   **Davies-Bouldin Index:** Чем **ниже** значение, тем лучше (расстояние между кластерами больше, а внутри меньше).
    *   **Calinski-Harabasz Index:** Чем **выше**, тем лучше (плотные, хорошо разделенные кластеры).
    """)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

# --- Вкладка 3: Характеристики групп ---
with tab3:
    st.subheader(f"Средние экономические показатели: {algorithm}")
    
    # Исключаем шум для средних значений
    df_mean = df_plot[df_plot['Cluster'] != -1] if algorithm == 'DBSCAN' else df_plot
    cluster_means = df_mean.groupby('Cluster')[['Work', 'Price', 'Salary']].mean().round(1)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(cluster_means, use_container_width=True)
    with col2:
        st.bar_chart(cluster_means)

# --- Вкладка 4: Выводы ---
with tab4:
    st.subheader("Обоснование выбора алгоритма")
    st.markdown("""
    В ходе исследования были применены три алгоритма кластеризации. На основе метрик и визуального анализа можно сделать следующие выводы:
    
    1. **Иерархический анализ (Ward)** отлично справляется с поиском структуры данных и не требует заданного K, однако он чувствителен к выбросам и строит кластеры сферической формы.
    2. **K-Means** показал высокую скорость работы и дал хорошие результаты (4 экономические группы), но потребовал ручного удаления аномалии (Тель-Авив), так как метод чувствителен к выбросам, которые "перетягивают" центроиды.
    3. **DBSCAN** оказался наиболее устойчивым к аномалиям. Он автоматически выделил города-одиночки (Цюрих, Женева, Гонконг) в "шум", не искажая границы основных плотных кластеров.
    
    **Итог:** Для поиска закономерностей в экономике лучше всего подходит **K-Means** (при условии очистки выбросов), так как он дает четкие, интерпретируемые центры. Однако для поиска аномалий (городов с уникальной экономикой) безусловным лидером является **DBSCAN**.
    """)