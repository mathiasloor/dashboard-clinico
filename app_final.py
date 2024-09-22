import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import pathlib
import plotly.express as px
import plotly.figure_factory as ff
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64
import matplotlib

# Usar el backend 'Agg' para evitar errores relacionados con el hilo principal
matplotlib.use('Agg')

# Cargar el dataset desde la carpeta data
BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("data").resolve()
df = pd.read_csv(DATA_PATH.joinpath("vgsales.csv"))

# Convertir Critic_Score y User_Score a numérico
df['Critic_Score'] = pd.to_numeric(df['Critic_Score'], errors='coerce')
df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce')

# Llenar valores nulos con predeterminados
df['Rating'] = df['Rating'].fillna('Unknown')
df['Genre'] = df['Genre'].fillna('Unknown')
df['Platform'] = df['Platform'].fillna('Unknown')
df['Year_of_Release'] = df['Year_of_Release'].replace(0, float('nan'))

# Definir los rangos de años
year_ranges = {
    "Antes de 2000": (df['Year_of_Release'] < 2000),
    "2000-2009": (df['Year_of_Release'] >= 2000) & (df['Year_of_Release'] <= 2009),
    "2010-2020": (df['Year_of_Release'] >= 2010) & (df['Year_of_Release'] <= 2020),
    "Año desconocido": df['Year_of_Release'].isna()
}

# Función para manejar nulos en el gráfico de barras para ventas globales por género
def genre_sales_bar_chart(df, selected_genres=None):
    if selected_genres:
        df = df[df['Genre'].isin(selected_genres)]
    if df.empty:
        return px.bar(title="No hay datos disponibles")

    # Agrupar y ordenar los datos
    genre_sales = df.groupby('Genre')['Global_Sales'].sum().reset_index()
    genre_sales = genre_sales.sort_values(by='Global_Sales', ascending=False)

    # Crear gráfico de barras con una paleta de colores mejorada
    fig = px.bar(genre_sales, 
                 x='Genre', 
                 y='Global_Sales', 
                 title='Ventas Globales por Género',
                 labels={'Global_Sales': 'Ventas Globales (millones)', 'Genre': 'Género'},
                 height=500,
                 color='Genre')  # Usar paleta de colores más moderna
    
    # Mejorar el diseño del gráfico
    fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')), selector=dict(type='bar'))
    fig.update_layout(
        xaxis_title="Género", 
        yaxis_title="Ventas Globales (millones)", 
        title_x=0.5, 
        hovermode="x unified",
        plot_bgcolor='rgba(0,0,0,0)',  # Hacer transparente el fondo del gráfico
        paper_bgcolor='rgba(0,0,0,0)',  # Hacer transparente el fondo del papel
        modebar_remove=['zoom', 'pan', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale', "select2d","lasso"]
        )
    return fig


# Gráfico de burbujas para ventas por plataforma
def platform_sales_bubble_chart(df):
    platform_sales = df.groupby('Platform').agg(total_sales=('Global_Sales', 'sum'),
                                                avg_critic_score=('Critic_Score', 'mean'),
                                                avg_user_score=('User_Score', 'mean')).reset_index()
    fig = px.scatter(platform_sales, x='avg_critic_score', y='avg_user_score', size='total_sales', color='Platform',
                     hover_name='Platform', title='Relación entre Puntuaciones y Ventas Globales por Plataforma',
                     labels={'avg_critic_score': 'Puntuación Media Crítica', 'avg_user_score': 'Puntuación Media de Usuario', 'total_sales':"Ventas totales"}, height=600)
    fig.update_layout(xaxis_title="Puntuación Media de Críticos", yaxis_title="Puntuación Media de Usuarios", legend_title="Plataformas",        modebar_remove=['zoom', 'pan', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale', "select2d","lasso"])
    return fig

# Heatmap de correlación mejorado
def correlation_heatmap(df):
    # Seleccionamos solo las columnas relevantes y eliminamos valores nulos
    df_corr = df[['Global_Sales', 'Critic_Score', 'User_Score', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].dropna()
    
    # Calculamos la correlación
    correlation = df_corr.corr()
    
    # Definimos etiquetas personalizadas en español
    labels = {
        'Global_Sales': 'Ventas Globales',
        'Critic_Score': 'Punt. Críticos',
        'User_Score': 'Punt. Usuarios',
        'NA_Sales': 'Ventas NA',
        'EU_Sales': 'Ventas EU',
        'JP_Sales': 'Ventas JP',
        'Other_Sales': 'Otras Ventas'
    }

    # Creamos el heatmap anotado con etiquetas personalizadas
    fig = ff.create_annotated_heatmap(
        z=correlation.values,
        x=[labels.get(col, col) for col in correlation.columns.tolist()],
        y=[labels.get(row, row) for row in correlation.columns.tolist()],
        annotation_text=correlation.round(2).values,
        colorscale='Viridis'
    )

    # Actualizamos el layout
    fig.update_layout(
        title="Heatmap de Correlación entre Ventas y Puntuaciones",
        xaxis_title="Variables",
        yaxis_title="Variables",
        height=800,
        modebar_remove=['zoom', 'pan', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale', "select2d","lasso"]

    )

    return fig


# WordCloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width=400, height=200, background_color='white').generate(text)
    img = io.BytesIO()
    plt.figure(figsize=(5, 2.5), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(img, format='png')
    img.seek(0)
    encoded_image = base64.b64encode(img.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded_image}"

def generate_wordclouds_by_year(df):
    old_games = df[df['Year_of_Release'] < 2000]
    new_games = df[df['Year_of_Release'] >= 2000]
    old_games_text = ' '.join(old_games['Name'].dropna())
    old_wordcloud = generate_wordcloud(old_games_text)
    new_games_text = ' '.join(new_games['Name'].dropna())
    new_wordcloud = generate_wordcloud(new_games_text)
    return old_wordcloud, new_wordcloud

import plotly.graph_objects as go

# Gráfico de líneas con puntos y dos ejes y para puntuaciones medias de usuarios y críticos
def line_chart_scores_over_time(df):
    # Agrupamos por año de lanzamiento y calculamos las puntuaciones medias
    df_grouped = df.groupby('Year_of_Release').agg(
        avg_critic_score=('Critic_Score', 'mean'), 
        avg_user_score=('User_Score', 'mean')
    ).reset_index()
    
    # Eliminamos los datos faltantes
    df_grouped = df_grouped.dropna(subset=['Year_of_Release', 'avg_critic_score', 'avg_user_score'])
    
    # Creamos un gráfico vacío y añadimos la primera línea (puntuación de críticos)
    fig = go.Figure()

    # Línea de puntuaciones de críticos (escala de 0 a 100)
    fig.add_trace(go.Scatter(
        x=df_grouped['Year_of_Release'], 
        y=df_grouped['avg_critic_score'], 
        name='Puntuación Críticos',
        line=dict(color='firebrick'),
        mode='lines+markers',  # Añadimos tanto líneas como puntos
        marker=dict(size=8, symbol='circle'),  # Personalizamos los puntos
        yaxis='y1'  # Asociamos esta línea con el eje y1 (para críticos)
    ))

    # Línea de puntuaciones de usuarios (escala de 0 a 10)
    fig.add_trace(go.Scatter(
        x=df_grouped['Year_of_Release'], 
        y=df_grouped['avg_user_score'], 
        name='Puntuación Usuarios',
        line=dict(color='royalblue'),
        mode='lines+markers',  # Añadimos tanto líneas como puntos
        marker=dict(size=8, symbol='circle'),  # Personalizamos los puntos
        yaxis='y2'  # Asociamos esta línea con el eje y2 (para usuarios)
    ))

    # Configuramos los dos ejes y
    fig.update_layout(
        title="Evolución de las Puntuaciones Medias de Usuarios y Críticos a lo largo del Tiempo",
        xaxis=dict(title='Año de Lanzamiento'),
        yaxis=dict(
            title='Puntuación Media Críticos (0-100)',
            titlefont=dict(color='firebrick'),
            tickfont=dict(color='firebrick'),
            range=[0            , 100]  # Rango de 0 a 100 para los críticos
        ),
        yaxis2=dict(
            title='Puntuación Media Usuarios (0-10)',
            titlefont=dict(color='royalblue'),
            tickfont=dict(color='royalblue'),
            overlaying='y',
            side='right',
            range=[0, 10]  # Rango de 0 a 10 para los usuarios
        ),
        legend_title='Puntuación',
        height=600,
        hovermode='x unified',
        modebar_remove=['zoom', 'pan', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale', "select2d","lasso"]
    )

    return fig


# Gráfico de Sunburst optimizado para mayor legibilidad
def sunburst_sales_chart(df):
    # Agrupar datos por ventas globales y filtrar los top 5 géneros y plataformas
    top_genres = df['Genre'].value_counts().nlargest(5).index
    top_platforms = df['Platform'].value_counts().nlargest(5).index
    df_filtered = df[df['Genre'].isin(top_genres) & df['Platform'].isin(top_platforms)]

    # Crear gráfico Sunburst
    fig = px.sunburst(
        df_filtered,
        path=['Genre', 'Platform'],  # Jerarquía: Género (interior) -> Plataforma (exterior)
        values='Global_Sales',
        title="Distribución de Ventas por Género y Plataforma (Top 5)",
        color='Global_Sales',
        color_continuous_scale=px.colors.sequential.Plasma,
        height=600
    )

    # Personalizamos el hovertemplate para diferenciar entre nodos interiores (Género) y exteriores (Plataforma)
    fig.update_traces(
        hovertemplate=(
            '<b>%{parent}<br>' +  
            '<b>%{label}<br>' +
            '<b>Ventas Globales:</b> %{value:.2f} millones<br>' +  # Ventas Globales
            '<extra></extra>'
        )
    )

    # Ajustar el layout
    fig.update_layout(
        margin=dict(t=50, l=25, r=25, b=25),
        coloraxis_colorbar=dict(title="Ventas Globales"),  # Agregar una barra de color para las ventas globales
        modebar_remove=['zoom', 'pan', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale', "select2d","lasso"]    
    )

    return fig


# Gráfico de barras apiladas para la distribución de ventas por las 10 principales plataformas
def sales_distribution_stacked_bar_chart(df):
    top_platforms = df.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False).nlargest(10).index

    df_top_platforms = df[df['Platform'].isin(top_platforms)]

    sales_by_platform = df_top_platforms.groupby(['Platform']).agg({
        'NA_Sales': 'sum',
        'EU_Sales': 'sum',
        'JP_Sales': 'sum',
        'Other_Sales': 'sum',
        'Global_Sales': 'sum'
    }).reset_index().sort_values(by='Global_Sales', ascending=False)
    # Cambiar los nombres de las columnas para que sean más descriptivos
    sales_by_platform = sales_by_platform.rename(columns={
        'NA_Sales': 'Ventas NA',
        'EU_Sales': 'Ventas EU',
        'JP_Sales': 'Ventas JP',
        'Other_Sales': 'Otras Ventas'
    })
    fig = px.bar(
        sales_by_platform,
        x='Platform',
        y=['Ventas NA', 'Ventas EU', 'Ventas JP', 'Otras Ventas'],
        title="Distribución de Ventas por las 10 Principales Plataformas (Barras Apiladas)",
        labels={'value': 'Ventas (millones)', 'Platform': 'Plataforma'},
        height=600
    )

    fig.update_layout(
        barmode='stack',
        xaxis_title="Plataforma",
        yaxis_title="Ventas (millones)",
        title_x=0.5,
        legend_title="Región",
        modebar_remove=['zoom', 'pan', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale', "select2d","lasso"]
    )

    return fig

# Consola de resumen con estadísticas clave en base a los filtros actuales
def summary_console(df_filtered):
    total_games = df_filtered['Name'].shape[0]
    avg_critic_score = df_filtered['Critic_Score'].mean()
    avg_user_score = df_filtered['User_Score'].mean()
    total_sales = df_filtered['Global_Sales'].sum()

    return html.Div([
        html.H5(f"Total de Juegos (Basado en los filtros actuales): {total_games}"),
        html.H5(f"Promedio de Calificación de Críticos: {avg_critic_score:.2f}"),
        html.H5(f"Promedio de Calificación de Usuarios: {avg_user_score:.2f}"),
        html.H5(f"Ventas Globales Totales: {total_sales:.2f} millones")
    ])

# Inicializar la app Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Barra lateral con estilo para permitir scroll y botones de "Seleccionar Todo"
nav_item_style = {
    "padding": "10px", 
    "font-weight": "500", 
    "border-radius": "5px", 
    "transition": "background-color 0.3s ease, box-shadow 0.3s ease"
}

sidebar = html.Div(
    [
        html.H5("Dashboard Análisis de Ventas y Puntaciones de Videojuegos", className="display-7"),
        html.Hr(),
        html.P("Menu"),
        dbc.Nav(
            [
                dbc.NavLink("Ventas por Género", href="/sales", active="exact", id="link-sales", className="nav-item", style=nav_item_style),
                dbc.NavLink("Puntuaciones y Ventas", href="/platforms", active="exact", id="link-platforms", className="nav-item", style=nav_item_style),
                dbc.NavLink("Palabras mas comunes", href="/wordcloud", active="exact", id="link-wordcloud", className="nav-item", style=nav_item_style),
                dbc.NavLink("Ventas Regionales y Puntuaciones", href="/heatmap", active="exact", id="link-heatmap", className="nav-item", style=nav_item_style),
                dbc.NavLink("Puntuaciones en el Tiempo", href="/line-chart-scores", active="exact", id="link-linechart", className="nav-item", style=nav_item_style),
                dbc.NavLink("Ventas Regionales (Top 10 Plataformas)", href="/regional-sales", active="exact", id="link-regional-sales", className="nav-item", style=nav_item_style),
                dbc.NavLink("Ventas Globales (Top 5 Plataformas)", href="/sunburst", active="exact", id="link-sunburst", className="nav-item", style=nav_item_style)
            ],
            vertical=True,
            pills=True,
        ),
        html.Hr(),
        html.H3("Filtros"),
        
        # Filtro por Género
        html.Label("Filtro por Género"),
        dcc.Dropdown(id='genre_filter',
                     options=[{'label': genre, 'value': genre} for genre in df['Genre'].unique()],
                     value=[genre for genre in df['Genre'].unique()],
                     multi=True, clearable=True),
        html.Div(
            dbc.Button("Seleccionar Todo", id="select_all_genres", n_clicks=0, 
                       color="primary", className="me-1 btn-custom", style={'margin-top': '10px', 'width': '100%'})
        ),

        # Filtro por Plataforma
        html.Label("Filtro por Plataforma"),
        dcc.Dropdown(id='platform_filter',
                     options=[{'label': platform, 'value': platform} for platform in df['Platform'].unique()],
                     value=[platform for platform in df['Platform'].unique()],
                     multi=True, clearable=True),
        html.Div(
            dbc.Button("Seleccionar Todo", id="select_all_platforms", n_clicks=0, 
                       color="primary", className="me-1 btn-custom", style={'margin-top': '10px', 'width': '100%'})
        ),

        # Filtro por Año de Lanzamiento
        html.Label("Filtro por Año de Lanzamiento"),
        dcc.Dropdown(id='year_filter',
                     options=[{'label': key, 'value': key} for key in year_ranges.keys()],
                     value=[key for key in year_ranges.keys()],
                     multi=True, clearable=True),
        html.Div(
            dbc.Button("Seleccionar Todo", id="select_all_years", n_clicks=0, 
                       color="primary", className="me-1 btn-custom", style={'margin-top': '10px', 'width': '100%'})
        ),

        # Filtro por Puntuación de Críticos
        html.Label("Filtro por Puntuación de Críticos"),
        dcc.RangeSlider(id='critic_score_filter', min=0, max=100, marks={i: str(i) for i in range(0, 101, 10)}, value=[0, 100], step=1),

        # Filtro por Puntuación de Usuarios
        html.Label("Filtro por Puntuación de Usuarios"),
        dcc.RangeSlider(id='user_score_filter', min=0, max=10, marks={i: str(i) for i in range(0, 11)}, value=[0, 10], step=0.1),

        # Filtro por Clasificación ESRB
        html.Label("Filtro por Clasificación ESRB"),
        dcc.Dropdown(id='rating_filter',
            options=[{'label': rating, 'value': rating} for rating in df['Rating'].unique()],
                       value=[rating for rating in df['Rating'].unique()],
                       multi=True, clearable=True),
        html.Div(
            dbc.Button("Seleccionar Todo", id="select_all_ratings", n_clicks=0, 
                       color="primary", className="me-1 btn-custom", style={'margin-top': '10px', 'width': '100%'})
        ),
    ],
    style={'position': 'fixed', 'top': 0, 'left': 0, 'bottom': 0, 'width': '18%', 'padding': '20px', 'background-color': '#f8f9fa', 'overflow-y': 'auto'}
)

# Layout del dashboard con barra lateral, pestañas y descripción separada
app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    dbc.Container(
        [
            dbc.Tabs(
                [
                    # Tab para la descripción del dataset
                    dbc.Tab(
                        label="Descripción del Dataset",
                        tab_id="tab-description",
                        children=[
                            html.H3("Descripción del Dataset de Ventas de Videojuegos"),
                            html.P("Este dataset contiene información sobre más de 16,000 videojuegos, "
                                   "incluyendo nombre, plataforma, género, ventas en distintas regiones y "
                                   "calificaciones de críticos y usuarios."),
                            html.P("El análisis incluye varias visualizaciones interactivas que permiten explorar "
                                   "la relación entre ventas, puntuaciones y otros factores."),
                            html.Hr(),
                            
                            # Deja el espacio vacío para ser actualizado dinámicamente por el callback
                            html.Div(id="summary-console", style={"padding": "10px", "background-color": "#f9f9f9"}),

                            html.Hr(),
                            html.H4("Explicación del Menú"),
                            html.P("""
                                A continuación, se presentan las opciones disponibles en el menú del dashboard:
                            """),
                            html.Ul([
                                html.Li([html.B("Ventas por Género: "), "Muestra un gráfico de barras que representa las ventas globales de videojuegos agrupadas por género."]),
                                html.Li([html.B("Puntuaciones y Ventas: "), "Muestra un gráfico de burbujas que relaciona las puntuaciones de críticos y usuarios con las ventas globales por plataforma."]),
                                html.Li([html.B("Palabras más Comunes: "), "Genera una nube de palabras (WordCloud) basada en los nombres de los videojuegos, separando los juegos antiguos y los nuevos."]),
                                html.Li([html.B("Ventas Regionales y Puntuaciones: "), "Muestra un heatmap que visualiza las correlaciones entre ventas en diferentes regiones y puntuaciones."]),
                                html.Li([html.B("Puntuaciones en el Tiempo: "), "Muestra un gráfico de líneas con la evolución de las puntuaciones medias de críticos y usuarios a lo largo de los años."]),
                                html.Li([html.B("Ventas Regionales (Top 10 Plataformas): "), "Muestra un gráfico de barras apiladas para visualizar la distribución de ventas por las 10 plataformas más populares."]),
                                html.Li([html.B("Ventas Globales (Top 5 Plataformas): "), "Muestra un gráfico de sunburst que representa la distribución de ventas globales agrupadas por género y plataforma."])
                            ]),

                            html.Hr(),
                            html.H4("Explicación de los Gráficos"),
                            html.P("A continuación se describen los gráficos que se pueden visualizar en el dashboard:"),
                            html.Ul([
                                html.Li([html.B("Gráfico de Barras por Género: "), 
                                         "Este gráfico muestra las ventas globales de videojuegos agrupadas por género. "
                                         "Permite identificar cuáles son los géneros más vendidos y cómo se distribuyen las ventas en cada uno."]),
                                html.Li([html.B("Gráfico de Burbujas por Plataforma: "), 
                                         "Este gráfico relaciona las puntuaciones medias de críticos y usuarios con las ventas globales de videojuegos, "
                                         "agrupadas por plataforma. Las plataformas con más ventas y mejores puntuaciones se muestran como burbujas más grandes."]),
                                html.Li([html.B("Nube de Palabras: "), 
                                         "La nube de palabras visualiza las palabras más comunes en los nombres de los videojuegos. "
                                         "Se muestran por separado los juegos anteriores al 2000 y los juegos más recientes, permitiendo ver las tendencias en los títulos a lo largo del tiempo."]),
                                html.Li([html.B("Heatmap de Correlación: "), 
                                         "Este gráfico de calor muestra las correlaciones entre diferentes variables, como ventas por región y puntuaciones de críticos y usuarios. "
                                         "Ayuda a identificar relaciones fuertes o débiles entre las ventas y las puntuaciones en distintas regiones."]),
                                html.Li([html.B("Gráfico de Líneas: "), 
                                         "Este gráfico visualiza la evolución de las puntuaciones medias de críticos y usuarios a lo largo del tiempo. "
                                         "Permite ver cómo han cambiado las valoraciones de los videojuegos desde los años 80 hasta la actualidad."]),
                                html.Li([html.B("Gráfico de Barras Apiladas: "), 
                                         "Este gráfico apilado muestra la distribución de las ventas en las 10 principales plataformas, agrupadas por región (NA, EU, JP y Otras). "
                                         "Permite ver en qué plataformas se han vendido más videojuegos y en qué regiones."]),
                                html.Li([html.B("Gráfico Sunburst: "), 
                                         "Este gráfico sunburst representa la distribución de ventas globales, primero por género y luego por plataforma. "
                                         "Las secciones interiores representan los géneros, mientras que las secciones exteriores representan las plataformas más vendidas para cada género."])
                            ])
                        ],
                        style={"padding": "20px"}
                    ),
                    # Tab para las visualizaciones
                    dbc.Tab(
                        label="Visualizaciones",
                        tab_id="tab-visualizations",
                        children=html.Div(id="page-content", style={"padding": "20px"}),
                        style={"padding": "20px"}
                    )
                ],
                id="tabs",
                active_tab="tab-visualizations"  # Pestaña activa por defecto
            )
        ],
        style={"margin-left": "18%"}
    )
])

# Callbacks para "Seleccionar Todo" en los filtros
@app.callback(
    Output('genre_filter', 'value'), Input('select_all_genres', 'n_clicks'),
    [Input('genre_filter', 'options')])
def select_all_genres(n_clicks, options):
    return [opt['value'] for opt in options]

@app.callback(
    Output('platform_filter', 'value'), Input('select_all_platforms', 'n_clicks'),
    [Input('platform_filter', 'options')])
def select_all_platforms(n_clicks, options):
    return [opt['value'] for opt in options]

@app.callback(
    Output('year_filter', 'value'), Input('select_all_years', 'n_clicks'),
    [Input('year_filter', 'options')])
def select_all_years(n_clicks, options):
    return [opt['value'] for opt in options]

@app.callback(
    Output('rating_filter', 'value'), Input('select_all_ratings', 'n_clicks'),
    [Input('rating_filter', 'options')])
def select_all_ratings(n_clicks, options):
    return [opt['value'] for opt in options]
@app.callback(
    Output("summary-console", "children"),
    [Input("genre_filter", "value"), 
     Input("platform_filter", "value"),
     Input("year_filter", "value"), 
     Input("critic_score_filter", "value"),
     Input("user_score_filter", "value"), 
     Input("rating_filter", "value"),
     Input("tabs", "active_tab")]
)
def update_summary_console(genre_filter, platform_filter, year_filter, critic_score_filter, user_score_filter, rating_filter, active_tab):
    # Solo actualiza el summary si estamos en la pestaña de descripción
    if active_tab == "tab-description":
        df_filtered = df.copy()

        # Aplicar los filtros al DataFrame
        if genre_filter:
            df_filtered = df_filtered[df_filtered['Genre'].isin(genre_filter)]
        if platform_filter:
            df_filtered = df_filtered[df_filtered['Platform'].isin(platform_filter)]
        if year_filter:
            year_filters = [year_ranges[year] for year in year_filter]
            year_filter_combined = year_filters[0]
            for yf in year_filters[1:]:
                year_filter_combined |= yf
            df_filtered = df_filtered.loc[year_filter_combined]
        if critic_score_filter:
            df_filtered = df_filtered[(df_filtered['Critic_Score'].isna()) |
                                      ((df_filtered['Critic_Score'] >= critic_score_filter[0]) &
                                       (df_filtered['Critic_Score'] <= critic_score_filter[1]))]
        if user_score_filter:
            df_filtered = df_filtered[(df_filtered['User_Score'].isna()) |
                                      ((df_filtered['User_Score'] >= user_score_filter[0]) &
                                       (df_filtered['User_Score'] <= user_score_filter[1]))]
        if rating_filter:
            df_filtered = df_filtered[df_filtered['Rating'].isin(rating_filter)]

        # Consola de resumen actualizada con los datos filtrados
        return summary_console(df_filtered)

    return html.Div()  # Si no estamos en la pestaña de descripción, no muestra nada
@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname"), 
     Input("genre_filter", "value"), 
     Input("platform_filter", "value"),
     Input("year_filter", "value"), 
     Input("critic_score_filter", "value"),
     Input("user_score_filter", "value"), 
     Input("rating_filter", "value")]
)
def render_page_content(pathname, genre_filter, platform_filter, year_filter, critic_score_filter, user_score_filter, rating_filter):
    df_filtered = df.copy()

    if genre_filter:
        df_filtered = df_filtered[df_filtered['Genre'].isin(genre_filter)]
    if platform_filter:
        df_filtered = df_filtered[df_filtered['Platform'].isin(platform_filter)]
    if year_filter:
        year_filters = [year_ranges[year] for year in year_filter]
        year_filter_combined = year_filters[0]
        for yf in year_filters[1:]:
            year_filter_combined |= yf
        df_filtered = df_filtered.loc[year_filter_combined]
    if critic_score_filter:
        df_filtered = df_filtered[(df_filtered['Critic_Score'].isna()) |
                                  ((df_filtered['Critic_Score'] >= critic_score_filter[0]) &
                                   (df_filtered['Critic_Score'] <= critic_score_filter[1]))]
    if user_score_filter:
        df_filtered = df_filtered[(df_filtered['User_Score'].isna()) |
                                  ((df_filtered['User_Score'] >= user_score_filter[0]) &
                                   (df_filtered['User_Score'] <= user_score_filter[1]))]
    if rating_filter:
        df_filtered = df_filtered[df_filtered['Rating'].isin(rating_filter)]

    if pathname == "/sales":
        return html.Div([html.H3("Ventas Globales por Género"), dcc.Graph(figure=genre_sales_bar_chart(df_filtered))])
    
    elif pathname == "/platforms":
        return html.Div([html.H3("Relación entre Puntuaciones y Ventas Globales por Plataforma"),
                         dcc.Graph(figure=platform_sales_bubble_chart(df_filtered))])
    
    elif pathname == "/wordcloud":
        old_wordcloud, new_wordcloud = generate_wordclouds_by_year(df_filtered)
        return dbc.Row([dbc.Col([html.H3("Juegos Antiguos (Antes del 2000)"),
                                 html.Img(src=old_wordcloud, style={'width': '100%', 'height': 'auto'})], width=6),
                        dbc.Col([html.H3("Juegos Nuevos (2000 en adelante)"),
                                 html.Img(src=new_wordcloud, style={'width': '100%', 'height': 'auto'})], width=6)])
    
    elif pathname == "/heatmap":
        return html.Div([html.H3("Heatmap de Correlación entre Ventas y Puntuaciones"),
                         dcc.Graph(figure=correlation_heatmap(df_filtered))])
    
    elif pathname == "/line-chart-scores":
        return html.Div([html.H3("Puntuaciones Medias a lo Largo del Tiempo"),
                         dcc.Graph(figure=line_chart_scores_over_time(df_filtered))])
    
    elif pathname == "/regional-sales":
        return html.Div([html.H3("Distribución de Ventas por las 10 Principales Plataformas (Barras Apiladas)"),
                         dcc.Graph(figure=sales_distribution_stacked_bar_chart(df_filtered))])

    elif pathname == "/sunburst":
        return html.Div([html.H3("Ventas Globales por Género, Plataforma y Año de Lanzamiento"),
                         dcc.Graph(figure=sunburst_sales_chart(df_filtered))])
    
    return html.Div([html.H1("404: Página no encontrada"), html.P("La página que buscas no existe.")])

# Ejecutar la aplicación
if __name__ == "__main__":
    app.run_server(debug=True)

