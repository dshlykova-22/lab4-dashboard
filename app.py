import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# ==========================================
# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ==========================================
st.set_page_config(page_title="Лабораторная работа №4", layout="wide")

@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv('Econom_Cities_data.csv', sep=';', decimal=',')
    df = df.replace(-9999, np.nan).dropna()
    df = df.set_index('City')
    return df

df_full = load_and_prepare_data()

# Создаем данные для K-Means (без Тель-Авива, как в твоей ЛР №2)
df_km = df_full.drop('Tel_Aviv', errors='ignore').copy()

# Масштабирование
scaler_full = StandardScaler()
X_scaled_full = scaler_full.fit_transform(df_full[['Work', 'Price', 'Salary']])

scaler_km = StandardScaler()
X_scaled_km = scaler_km.fit_transform(df_km[['Work', 'Price', 'Salary']])

# ==========================================
# 2. ИНТЕРФЕЙС: БОКОВАЯ ПАНЕЛЬ
# ==========================================
st.title("🌍 Разработка интерактивного аналитического дашборда")
st.markdown("Объединение результатов лабораторных работ №1-3. Выполнила: Шлыкова Д.В., 563316.")

st.sidebar.header("⚙️ Настройки")
algorithm = st.sidebar.selectbox(
    "Выбор алгоритма кластеризации:",
    ('Иерархический (Ward)', 'K-Means', 'DBSCAN')
)

# Интерактивные ползунки для параметров
if algorithm in ['Иерархический (Ward)', 'K-Means']:
    n_clusters = st.sidebar.slider("Количество кластеров (K):", min_value=2, max_value=8, value=4)
elif algorithm == 'DBSCAN':
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Подбор параметров DBSCAN**")
    eps_val = st.sidebar.slider("Эпсилон (eps):", min_value=0.1, max_value=2.0, value=0.75, step=0.05)
    min_samples_val = st.sidebar.slider("Мин. точек (min_samples):", min_value=2, max_value=10, value=4)

# ==========================================
# 3. ОБУЧЕНИЕ МОДЕЛЕЙ И РАСЧЕТ МЕТРИК
# ==========================================
def evaluate_clustering(X, labels, name):
    mask = labels != -1
    X_clean = X[mask] if name == 'DBSCAN' else X
    labels_clean = labels[mask] if name == 'DBSCAN' else labels
    
    n_clusters_val = len(set(labels_clean))
    
    if n_clusters_val < 2:
        return {'Алгоритм': name, 'Кластеров': n_clusters_val, 'Шум': len(labels) - len(labels_clean) if name == 'DBSCAN' else 0, 'Silhouette': np.nan, 'Davies-Bouldin': np.nan, 'Calinski-Harabasz': np.nan}

    return {
        'Алгоритм': name,
        'Кластеров': n_clusters_val,
        'Шум': len(labels) - len(labels_clean) if name == 'DBSCAN' else 0,
        'Silhouette': round(silhouette_score(X_clean, labels_clean), 3),
        'Davies-Bouldin': round(davies_bouldin_score(X_clean, labels_clean), 3),
        'Calinski-Harabasz': round(calinski_harabasz_score(X_clean, labels_clean), 3)
    }

# Обучение
hier = AgglomerativeClustering(n_clusters=n_clusters if algorithm == 'Иерархический (Ward)' else 4, linkage='ward')
hier_labels = hier.fit_predict(X_scaled_full)

kmeans = KMeans(n_clusters=n_clusters if algorithm == 'K-Means' else 4, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled_km)

dbscan = DBSCAN(eps=eps_val if algorithm == 'DBSCAN' else 0.75, min_samples=min_samples_val if algorithm == 'DBSCAN' else 4)
dbscan_labels = dbscan.fit_predict(X_scaled_full)

# Сбор метрик
metrics_df = pd.DataFrame([
    evaluate_clustering(X_scaled_full, hier_labels, 'Иерархический (Ward)'),
    evaluate_clustering(X_scaled_km, kmeans_labels, 'K-Means'),
    evaluate_clustering(X_scaled_full, dbscan_labels, 'DBSCAN')
])

# Выбор активных данных
if algorithm == 'K-Means':
    df_plot = df_km.copy()
    current_labels = kmeans_labels
    current_X = X_scaled_km
else:
    df_plot = df_full.copy()
    current_labels = hier_labels if algorithm == 'Иерархический (Ward)' else dbscan_labels
    current_X = X_scaled_full

df_plot['Cluster'] = current_labels
df_plot['Cluster_Label'] = df_plot['Cluster'].apply(lambda x: f"Кластер {x}" if x != -1 else "Шум (выброс)")

# Фильтр кластеров в сайдбаре
if algorithm != 'DBSCAN':
    unique_clusters = sorted(df_plot['Cluster'].unique())
    selected_clusters = st.sidebar.multiselect("Кластеры для отображения:", unique_clusters, default=unique_clusters)
    df_filtered = df_plot[df_plot['Cluster'].isin(selected_clusters)]
else:
    df_filtered = df_plot

# ==========================================
# 4. ВКЛАДКИ ДАШБОРДА
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Общие данные", 
    "🎯 Визуализация метода", 
    "📋 Сравнение алгоритмов", 
    "📈 Характеристики групп", 
    "📝 Выводы"
])

# --- ВКЛАДКА 1: ДАННЫЕ ---
with tab1:
    st.subheader("Общая характеристика исходных данных")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write(f"**Количество городов:** {len(df_full)}")
        st.write(f"**Признаки:** Рабочие часы, Индекс цен, Индекс зарплат")
        st.dataframe(df_full.head(), use_container_width=True)
    with col2:
        # Ящик с усами (Boxplot) из ЛР №2
        fig_box = px.box(df_full, y=['Work', 'Price', 'Salary'], title="Проверка на выбросы (Ящик с усами)")
        st.plotly_chart(fig_box, use_container_width=True)

# --- ВКЛАДКА 2: ВИЗУАЛИЗАЦИЯ (Дендрограмма, Центроиды, k-расстояния) ---
with tab2:
    st.subheader(f"Результаты: {algorithm}")
    
    # 1. Дендрограмма (ЛР №1)
    if algorithm == 'Иерархический (Ward)':
        st.markdown("#### Дендрограмма (Метод Уорда)")
        fig_dendro, ax = plt.subplots(figsize=(12, 4))
        sch.dendrogram(sch.linkage(X_scaled_full, method='ward'), labels=df_full.index.tolist(), leaf_rotation=90, ax=ax)
        ax.set_ylabel("Расстояние")
        ax.axhline(y=5, color='r', linestyle='--', label='Линия разреза (K=4)')
        ax.legend()
        st.pyplot(fig_dendro)

    # 2. Центроиды (ЛР №2)
    elif algorithm == 'K-Means':
        st.markdown("#### Кластеры K-Means с центроидами (Тель-Авив исключен)")
        fig_km = px.scatter(df_plot, x='Price', y='Salary', color='Cluster_Label', hover_name=df_plot.index, labels={'Price': 'Индекс цен', 'Salary': 'Индекс зарплат'})
        # Добавляем центроиды
        centroids_orig = scaler_km.inverse_transform(kmeans.cluster_centers_)
        for i, c in enumerate(centroids_orig):
            fig_km.add_trace(go.Scatter(x=[c[1]], y=[c[2]], mode='markers', marker=dict(symbol='x', size=15, color='black', linewidth=2), name=f'Центр {i}'))
        st.plotly_chart(fig_km, use_container_width=True)

    # 3. График k-расстояний (ЛР №3)
    elif algorithm == 'DBSCAN':
        st.markdown("#### График k-расстояний (для выбора eps)")
        neighbors = NearestNeighbors(n_neighbors=min_samples_val)
        distances, _ = neighbors.fit(X_scaled_full).kneighbors(X_scaled_full)
        distances = np.sort(distances[:, min_samples_val-1], axis=0)
        fig_k_dist = px.line(y=distances, title="k-расстояния", labels={'y': 'Расстояние', 'index': 'Точки'})
        fig_k_dist.add_hline(y=eps_val, line_dash="dash", line_color="red", annotation_text=f"Выбранный eps={eps_val}")
        st.plotly_chart(fig_k_dist, use_container_width=True)

    # 4. Общий 3D-график (Твой оригинальный)
    st.markdown("#### Пространство признаков (3D)")
    fig_3d = px.scatter_3d(df_filtered, x='Price', y='Salary', z='Work', color='Cluster_Label', hover_name=df_filtered.index, 
                           labels={'Price': 'Индекс цен', 'Salary': 'Индекс зарплат', 'Work': 'Рабочие часы'}, 
                           color_discrete_map={'Шум (выброс)': 'black'})
    fig_3d.update_traces(marker=dict(size=6, opacity=0.9, line=dict(width=1, color='DarkSlateGrey')))
    fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=600)
    st.plotly_chart(fig_3d, use_container_width=True)

# --- ВКЛАДКА 3: СРАВНЕНИЕ ---
with tab3:
    st.subheader("Сравнительная таблица метрик")
    st.markdown("""
    *   **Silhouette Score:** от -1 до 1. Чем ближе к 1, тем лучше объекты разделены.
    *   **Davies-Bouldin Index:** Чем **ниже** значение, тем лучше.
    *   **Calinski-Harabasz Index:** Чем **выше**, тем лучше (плотные кластеры).
    """)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

# --- ВКЛАДКА 4: ХАРАКТЕРИСТИКИ ГРУПП (С выбором конкретного кластера) ---
with tab4:
    st.subheader(f"Анализ групп: {algorithm}")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Средние значения по всем кластерам:**")
        df_mean = df_plot[df_plot['Cluster'] != -1]
        cluster_means = df_mean.groupby('Cluster')[['Work', 'Price', 'Salary']].mean().round(1)
        st.dataframe(cluster_means, use_container_width=True)
        
    with col2:
        st.markdown("**Детальное изучение кластера:**")
        # Требование ЛР 4: выбор отдельных кластеров для детального изучения
        cluster_to_study = st.selectbox("Выберите кластер:", sorted([c for c in df_plot['Cluster'].unique() if c != -1]))
        st.write(f"Города в кластере {cluster_to_study}:")
        st.dataframe(df_plot[df_plot['Cluster'] == cluster_to_study][['Work', 'Price', 'Salary']], use_container_width=True)

# --- ВКЛАДКА 5: ВЫВОДЫ ---
with tab5:
    st.subheader("📝 Обоснование выбора алгоритма и итоговые выводы")
    
    st.markdown("""
    В ходе выполнения лабораторных работ №1–3 и создания данного дашборда было проведено исследование экономических показателей 46 городов мира. 
    Для сегментации данных были применены три принципиально разных алгоритма кластеризации. Опираясь на визуальный анализ, метрики качества (вкладка «Сравнение») и экономическую интерпретацию, можно сделать следующие выводы:
    """)

    # Блок 1: Иерархический анализ
    with st.expander("🔹 1. Иерархический анализ (Метод Уорда) — «Исследователь»", expanded=True):
        st.markdown("""
        *   **Качество и структура:** Метод отлично справляется с поиском естественной структуры данных без необходимости заранее задавать количество кластеров. Дендрограмма наглядно показывает, как города объединяются в группы.
        *   **Интерпретируемость:** Полученные 4 кластера легко интерпретируются экономически (сверхдорогие города, крупные мегаполисы, развивающиеся города, города с высокой нагрузкой).
        *   **Недостатки:** Алгоритм чувствителен к выбросам и строит кластеры преимущественно сферической формы. Из-за этого аномальные города (например, Цюрих или Женева) могут сильно искажать границы «обычных» кластеров.
        """)

    # Блок 2: K-Means
    with st.expander("🔹 2. K-Means — «Математик»", expanded=True):
        st.markdown("""
        *   **Качество и структура:** Показал высокую скорость работы и четкие, хорошо разделенные центры. Метод минимизирует евклидово расстояние до центроидов, деля пространство прямыми линиями. Из-за этого точки на границе (близкие по цене, но разные по зарплате) могут распределяться математически, а не визуально.
        *   **Проблема выбросов:** Метод крайне чувствителен к аномалиям. В ЛР №2 было выявлено, что Тель-Авив (точка ~110 по цене, 40 по зарплате) «перетягивал» на себя центр кластера, размывая границы остальных групп. После его принудительного удаления качество кластеризации и четкость центроидов значительно выросли.
        *   **Интерпретируемость:** Центроиды совпадают со средними значениями по группам, что дает очень четкое понимание «среднестатистического» города в каждом кластере.
        """)

    # Блок 3: DBSCAN
    with st.expander("🔹 3. DBSCAN — «Детектор аномалий»", expanded=True):
        st.markdown("""
        *   **Качество и структура:** Наиболее устойчив к выбросам. Настройка параметров (eps и min_samples) с помощью графика k-расстояний позволила найти оптимальную плотность.
        *   **Работа с шумом:** Алгоритм блестяще справился с задачей поиска уникальных городов. Он автоматически выделил объекты с уникальной экономикой (Гонконг, Женеву, Цюрих) в отдельный класс «Шум» (-1). Это позволяет не искажать статистику по основным группам.
        *   **Недостатки:** Не требует задавать K, но сложен в настройке параметров. Если выбрать `eps` слишком маленьким — все станет шумом, если слишком большим — выбросы попадут в кластеры.
        """)

    st.markdown("---")
    st.subheader("🏆 Итоговое обоснование выбора")
    
    col1, col2 = st.columns(2)
    with col1:
        st.success("**Для поиска закономерностей лучше подходит K-Means (с предварительной очисткой)**\n\n"
                   "Если цель — понять общую экономическую структуру мира и получить четкие, интерпретируемые «профили» городов, K-Means выигрывает. Он дает плотные группы и понятные средние значения (при условии удаления экстремальных выбросов вроде Тель-Авива).")
    
    with col2:
        st.info("**Для поиска аномалий и уникальных объектов — DBSCAN**\n\n"
                "Если цель — найти города с уникальным укладом экономики, которые не вписываются в общие тренды, DBSCAN незаменим. Он не «притягивает за уши» уникальные точки к общим кластерам, а честно выделяет их в шум.")

    st.markdown("""
    **Общий вывод по датасету:** 
    Экономические показатели городов неоднородны. Наличие четко выраженных выбросов (городов-одиночек) говорит о том, что в реальном мире экономика не подчиняется строгим законам сферических кластеров. Поэтому комбинированный подход (K-Means для поиска базы + DBSCAN для поиска исключений) является наиболее качественным и обоснованным для данного набора данных.
    """)
