import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import pathlib
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64

# Cargar el dataset desde la carpeta data
BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("data").resolve()
df = pd.read_csv(DATA_PATH.joinpath("imdb-movies-dataset.csv"))

# Procesar la columna 'Genre' para quedarse solo con el primer género
df['Main Genre'] = df['Genre'].apply(lambda x: x.split(',')[0] if pd.notna(x) else None)

# Inicializar la app Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Función para generar una word cloud a partir de los títulos de películas
def generate_wordcloud(titles_text):
    wordcloud = WordCloud(width=400, height=200, background_color='white', colormap='viridis').generate(titles_text)
    
    img = io.BytesIO()
    plt.figure(figsize=(5, 2.5), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(img, format='png')
    img.seek(0)
    
    encoded_image = base64.b64encode(img.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded_image}"

# Generar word clouds por época (basada en los años de lanzamiento)
def generate_wordclouds_by_year(df):
    periods = {
        'Antes de 1980': df[df['Year'] < 1980],
        '1980-1999': df[(df['Year'] >= 1980) & (df['Year'] <= 1999)],
        '2000-2009': df[(df['Year'] >= 2000) & (df['Year'] <= 2009)],
        '2010 en adelante': df[df['Year'] >= 2010]
    }

    images = []
    for period, df_period in periods.items():
        titles_text = ' '.join(df_period['Title'].dropna())
        img = generate_wordcloud(titles_text)
        images.append(html.Div([
            html.H5(f"Películas de {period}", style={"text-align": "center"}),
            html.Img(src=img, style={'width': '100%', 'height': 'auto'})
        ], style={"width": "48%", "display": "inline-block", "padding": "10px"}))
    
    return images

# Funciones auxiliares para generar gráficos

# Gráfico de barras - Cantidad de películas por género principal
def genre_bar_chart(df):
    genre_count = df['Main Genre'].value_counts().reset_index()
    genre_count.columns = ['Main Genre', 'Count']
    fig = px.bar(genre_count, x='Main Genre', y='Count', title='Cantidad de Películas por Género Principal')
    return fig

# Gráfico de dispersión - Relación entre duración y calificación (con filtro por género)
def duration_rating_scatter(df, selected_genre):
    if selected_genre and selected_genre != 'Todos':
        df = df[df['Main Genre'] == selected_genre]

    fig = px.scatter(df, x='Duration (min)', y='Rating', 
                     title=f'Relación entre la Duración de la Película y la Calificación ({selected_genre})',
                     hover_data=['Title', 'Year'], color='Rating')
    return fig

# Gráfico de líneas - Evolución de calificaciones por año
def rating_line_chart(df):
    avg_rating_by_year = df.groupby('Year')['Rating'].mean().reset_index()
    fig = px.line(avg_rating_by_year, x='Year', y='Rating', title='Evolución de la Calificación Promedio por Año')
    return fig

# Layout del dashboard
app.layout = dbc.Container([
    dbc.Tabs([
        # Tab 1: Descripción de los Datos
        dbc.Tab(label='Descripción de los Datos', children=[
            html.H3("Descripción del Dataset"),
            html.P("Este dataset contiene información sobre más de 85,000 películas de IMDB, incluyendo título, "
                   "género, año de lanzamiento, duración, calificación, número de votos, y reseñas de los usuarios. "
                   "A continuación, se presentan varios gráficos interactivos que exploran diferentes aspectos del dataset.")
        ]),
        
        # Tab 2: Análisis de Películas
        dbc.Tab(label='Análisis de Películas', children=[
            html.H3("Cantidad de Películas por Género Principal"),
            dcc.Graph(figure=genre_bar_chart(df)),
            
            html.H3("Evolución de la Calificación Promedio por Año"),
            dcc.Graph(figure=rating_line_chart(df))
        ]),
        
        # Tab 3: Calificaciones y Reseñas
        dbc.Tab(label='Calificaciones y Reseñas', children=[
            html.H3("Filtro por Género para Gráfico de Dispersión"),
            dcc.Dropdown(
                id='genre_dropdown',
                options=[{'label': 'Todos', 'value': 'Todos'}] +
                        [{'label': genre, 'value': genre} for genre in df['Main Genre'].dropna().unique()],
                value='Todos',
                clearable=False,
                style={'width': '50%'}
            ),
            html.H3("Relación entre la Duración de la Película y la Calificación"),
            dcc.Graph(id='duration_rating_graph'),
            
            # Word clouds por época (años de lanzamiento)
            html.H3("Word Clouds de Películas por Época"),
            html.Div(generate_wordclouds_by_year(df), style={"text-align": "center"})
        ])
    ])
], fluid=True)

# Callback para actualizar el gráfico de dispersión según el género seleccionado
@app.callback(
    Output('duration_rating_graph', 'figure'),
    Input('genre_dropdown', 'value')
)
def update_duration_rating_scatter(selected_genre):
    return duration_rating_scatter(df, selected_genre)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Obtener el puerto de la variable de entorno
    app.run_server(host="0.0.0.0", port=port)