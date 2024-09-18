import dash
from dash import dcc, html, Input, Output
import numpy as np
import pandas as pd
from datetime import datetime as dt
import pathlib
import plotly.graph_objects as go 
import dash_bootstrap_components as dbc

############# Fase 1: Planificación del Dashboard #############
# 1. Estructura del Dashboard:
#    - Se organiza en dos columnas principales:
#      a) Columna Izquierda: Contiene los filtros, bienvenida y descripción.
#      b) Columna Derecha: Contiene los gráficos interactivos.
# 2. Componentes y estilos:
#    - Usaremos componentes como Dropdowns, DatePicker, y un botón de "Deshacer selección".
#    - Utilizaremos estilos Bootstrap para un diseño moderno y limpio.
#    - Los gráficos usan la biblioteca `plotly.graph_objects` para generar gráficos interactivos.
# 3. Interacciones:
#    - Los filtros interactúan con los gráficos para actualizar la información mostrada.
#    - El heatmap permite filtrar datos haciendo clic en un cuadro específico (día y hora).
#    - El botón "Deshacer selección" restaura la vista original sin filtros.

############# Fase 2: Estructura y personalización del Dashboard #############
# Definir la ruta de los datos
BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("data").resolve()

# Función para preprocesar los datos
def preprocesamiento(df):
    # Llenar las categorías vacías con 'No Identificado'
    df["Admit Source"] = df["Admit Source"].fillna("Not Identified")
    
    # Formatear la columna de tiempo (Check-In Time)
    df["Check-In Time"] = df["Check-In Time"].apply(lambda x: dt.strptime(x, "%Y-%m-%d %I:%M:%S %p"))  # String -> Datetime
    
    # Crear columnas de día de la semana y hora basadas en Check-In Time
    df["Days of Wk"] = df["Check-In Hour"] = df["Check-In Time"]
    df["Days of Wk"] = df["Days of Wk"].apply(lambda x: dt.strftime(x, "%A"))  # Obtener el día de la semana
    df["Check-In Hour"] = df["Check-In Hour"].apply(lambda x: dt.strftime(x, "%I %p"))  # Obtener la hora en formato AM/PM
    
    return df

# Cargar y preprocesar los datos
df = pd.read_csv(DATA_PATH.joinpath("clinical_analytics.csv.gz"))
df = preprocesamiento(df)

# Valores predeterminados
clinicas = df['Clinic Name'].unique()
sources = df["Admit Source"].unique()
default_clinic = clinicas[0]
default_sources = sources[:3]
default_start_date = df["Check-In Time"].min().date()
default_end_date = df["Check-In Time"].max().date()

############# Fase 3: Implementación de la estructura con Bootstrap #############
# Configurar los estilos externos de Bootstrap
external_stylesheets = [dbc.themes.CERULEAN]

# Inicializar la app Dash
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Clinical Analytics Dashboard"

# Layout de la app
app.layout = dbc.Container([
    dbc.Row([
        # Columna Izquierda: Bienvenida, descripción y filtros
        dbc.Col([
            html.H1("Bienvenidos al Dashboard de Análisis Clínico", className="text-center my-4"),
            html.P("Este dashboard te permite analizar datos clínicos. Selecciona una clínica, "
                   "rango de fechas y origen para ver métricas sobre tiempos de espera y satisfacción de los pacientes.",
                   className="text-center text-black"),
            # Dropdown para seleccionar la clínica
            html.P("Selecciona la clínica:", className="font-weight-bold"),
            dcc.Dropdown(
                options=[{'label': c, 'value': c} for c in clinicas],
                value=default_clinic, id="dropdown_clinica", clearable=False
            ),
            # Selector de rango de fechas
            html.P("Selecciona un rango de fechas:", className="font-weight-bold"),
            dcc.DatePickerRange(
                id="date_range",
                start_date=default_start_date,
                end_date=default_end_date
            ),
            # Dropdown para seleccionar el origen de los datos
            html.P("Selecciona un origen:", className="font-weight-bold"),
            dcc.Dropdown(
                options=[{'label': s, 'value': s} for s in sources],
                value=default_sources, id="dropdown_source", multi=True
            ),
            # Div para mostrar el cuadro seleccionado del heatmap
            html.Div(id="cuadro_seleccionado", className="mt-4 text-center"),
            # Botón para deshacer la selección
            html.Button(
                "Deshacer selección", 
                id="reset_button", 
                className="btn btn-danger mt-3",  # Estilo de Bootstrap para destacar el botón
                style={"font-size": "16px", "width": "100%", "padding": "10px", "border-radius": "10px"}  # Personalización adicional
            )
        ], md=3),  # Columna de 3 de ancho para los filtros y la descripción
        
        # Columna Derecha: Gráficos
        dbc.Col([
            # Heatmap
            dcc.Graph(id="grafico_hm", config={
                'scrollZoom': False,          
                'displayModeBar': True,       
                'modeBarButtonsToRemove': ['zoom2d', 'pan2d', 'select2d', 'lasso2d', 
                                           'hoverClosestCartesian', 'hoverCompareCartesian', 
                                           'resetScale2d'],  
                'doubleClick': 'reset',  # Permitir doble clic para resetear
            }),
            # Gráfico de tiempo de espera
            dcc.Graph(id="grafico_tiempo_espera", config={
                'scrollZoom': False,         
                'displayModeBar': True,
                'modeBarButtonsToRemove': ['zoom2d', 'pan2d', 'select2d', 'lasso2d', 
                                           'hoverClosestCartesian', 'hoverCompareCartesian', 
                                           'resetScale2d'],  # Eliminar todo excepto autoscale
                'doubleClick': 'reset',
            }),
            # Gráfico de satisfacción
            dcc.Graph(id="grafico_satisfaccion", config={
                'scrollZoom': False,         
                'displayModeBar': True,
                'modeBarButtonsToRemove': ['zoom2d', 'pan2d', 'select2d', 'lasso2d', 
                                           'hoverClosestCartesian', 'hoverCompareCartesian', 
                                           'resetScale2d'],  # Eliminar todo excepto autoscale
                'doubleClick': 'reset',
            })
        ], md=9)  # Columna de 9 de ancho para los gráficos
    ])
], fluid=True)  # Hacer que el diseño sea fluido y ocupe todo el ancho

############# Fase 3: Callbacks e Interactividad #############
# Callback para actualizar el heatmap y el texto del cuadro seleccionado
@app.callback(
    [Output("grafico_hm", "figure"),
     Output("cuadro_seleccionado", "children")],  # Salida adicional para mostrar la selección
    [Input("dropdown_clinica", "value"),
     Input("date_range", "start_date"),
     Input("date_range", "end_date"),
     Input("dropdown_source", "value"),
     Input("grafico_hm", "clickData"),
     Input("reset_button", "n_clicks")]  # Detectar clics en el botón de deshacer
)
def actualizar_heatmap(clinica, start_date, end_date, fuentes, clickData, n_clicks):
    # Filtrar los datos según la clínica seleccionada, rango de fechas y origen
    df_filtrado = df[df["Clinic Name"] == clinica]
    
    if start_date and end_date:
        df_filtrado = df_filtrado[(df_filtrado["Check-In Time"] >= pd.to_datetime(start_date)) &
                                  (df_filtrado["Check-In Time"] <= pd.to_datetime(end_date))]
    
    if fuentes:
        df_filtrado = df_filtrado[df_filtrado["Admit Source"].isin(fuentes)]
    
    # Agrupar los datos por día de la semana y hora
    df_agrupado = df_filtrado.groupby(["Days of Wk", "Check-In Hour"]).size().reset_index(name="Number of Records")
    
    # Crear el heatmap
    fig_hm = go.Figure(go.Heatmap(
        z=df_agrupado["Number of Records"],
        x=df_agrupado["Check-In Hour"],
        y=df_agrupado["Days of Wk"],
        text=df_agrupado["Number of Records"],
        texttemplate="%{text}",
        colorscale="Viridis"
    ))
    
    fig_hm.update_layout(
        title="Número de Pacientes por Día y Hora",
        xaxis_nticks=24,
        yaxis_nticks=7,
        xaxis_title="Hora de Check-In",
        yaxis_title="Día de la Semana"
    )
    
    # Si se hace clic en "Deshacer selección", se reinicia la selección
    if n_clicks:
        return fig_hm, "Selección deshecha."

    # Actualizar el texto del cuadro seleccionado (día y hora)
    cuadro_seleccionado = "No se ha seleccionado ningún cuadro."
    if clickData:
        dia_seleccionado = clickData["points"][0]["y"]
        hora_seleccionada = clickData["points"][0]["x"]
        cuadro_seleccionado = f"Seleccionaste: {dia_seleccionado} a las {hora_seleccionada}"
    
    return fig_hm, cuadro_seleccionado

# Callback para actualizar los gráficos de tiempo de espera y satisfacción
@app.callback(
    [Output("grafico_tiempo_espera", "figure"),
     Output("grafico_satisfaccion", "figure")],
    [Input("dropdown_clinica", "value"),
     Input("date_range", "start_date"),
     Input("date_range", "end_date"),
     Input("dropdown_source", "value"),
     Input("grafico_hm", "clickData"),
     Input("reset_button", "n_clicks")]  # Input para el botón de deshacer
)
def actualizar_graficas_adicionales(clinica, start_date, end_date, fuentes, clickData, n_clicks):
    # Filtrar los datos según clínica, fechas y fuentes seleccionadas
    df_filtrado = df[df["Clinic Name"] == clinica]
    
    if start_date and end_date:
        df_filtrado = df_filtrado[(df_filtrado["Check-In Time"] >= pd.to_datetime(start_date)) &
                                  (df_filtrado["Check-In Time"] <= pd.to_datetime(end_date))]

    if fuentes:
        df_filtrado = df_filtrado[df_filtrado["Admit Source"].isin(fuentes)]
    
    # Si se hace clic en el botón de "Deshacer selección", no filtrar por clickData
    if n_clicks:
        clickData = None  # Reiniciar el clickData

    # Si hay datos de click en el heatmap, filtrar por día y hora seleccionada
    if clickData:
        dia_seleccionado = clickData["points"][0]["y"]
        hora_seleccionada = clickData["points"][0]["x"]
        df_filtrado = df_filtrado[(df_filtrado["Days of Wk"] == dia_seleccionado) &
                                  (df_filtrado["Check-In Hour"] == hora_seleccionada)]
    
    # Verificar si hay datos después del filtrado
    if df_filtrado.empty:
        return {}, {}

    colors = ['royalblue', 'tomato', 'limegreen', 'gold', 'purple']  # Colores personalizados
    
    # Crear el gráfico de distribución del tiempo de espera (Boxplot)
    fig_tiempo_espera = go.Figure()
    if "Wait Time Min" in df_filtrado.columns and "Department" in df_filtrado.columns:
        for i, dept in enumerate(df_filtrado["Department"].unique()):
            df_dept = df_filtrado[df_filtrado["Department"] == dept]
            fig_tiempo_espera.add_trace(go.Box(
                y=df_dept["Wait Time Min"], 
                name=dept, 
                marker_color=colors[i % len(colors)],
                boxpoints='outliers',  # Mostrar puntos solo para outliers
                hovertemplate="Departamento: %{name}<br>Tiempo de Espera: %{y} min<extra></extra>",
                width=0.4
            ))

        fig_tiempo_espera.update_layout(
            title=f"Distribución del Tiempo de Espera por Departamento en {clinica}",
            yaxis_title="Tiempo de Espera (minutos)",
            xaxis_title="Departamento",
            boxmode="group",  # Mostrar las cajas lado a lado
            template="plotly_white"
        )

    # Crear el gráfico de calificaciones de satisfacción (Histograma)
    fig_satisfaccion = go.Figure()
    if "Care Score" in df_filtrado.columns and "Department" in df_filtrado.columns:
        for dept in df_filtrado["Department"].unique():
            df_dept = df_filtrado[df_filtrado["Department"] == dept]
            fig_satisfaccion.add_trace(go.Histogram(
                x=df_dept["Care Score"], 
                name=dept,  
                opacity=0.7,  
                histnorm='percent',  # Mostrar como porcentaje
                nbinsx=10,  
                hovertemplate="Departamento: " + dept + "<br>Calificación: %{x}<br>Cantidad: %{y}<extra></extra>"            
            ))

        fig_satisfaccion.update_layout(
            title=f"Distribución de Calificaciones de Satisfacción por Departamento en {clinica}",
            xaxis_title="Calificación de Satisfacción",
            yaxis_title="Porcentaje",
            template="plotly_white"
        )

    return fig_tiempo_espera, fig_satisfaccion

# Ejecutar el servidor
if __name__ == "__main__":
    app.run_server(debug=True, port=8080)
    
############# Conclusiones del Proyecto #############

# 1. Monitoreo Centralizado:
#    - Permite seguimiento en tiempo real de pacientes por clínica, día y hora.

# 2. Análisis Detallado:
#    - Filtrado eficiente por clínica y fechas, mejorando la toma de decisiones.

# 3. Visualización Clave:
#    - Gráficos claros de tiempos de espera y satisfacción del paciente.

# 4. Interactividad:
#    - Experiencia interactiva para explorar datos y restablecer selecciones fácilmente.

# 5. Adaptabilidad:
#    - Estructura escalable para futuras clínicas y métricas.
