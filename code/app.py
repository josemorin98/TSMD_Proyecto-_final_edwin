import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.ensemble._bagging import BaggingClassifier
import plotly.express as px
from sklearn.model_selection._search import GridSearchCV
import numpy as np

# Se implementa el algoritmo de clasificación
# leemos el csv
data_clus = pd.read_csv('./data/Data_proces.csv')
# seleccion de columnas
select_features = ['Latitud','Longitud','Temp_min_merra','Temp_min_emas','Temp_mean_merra','Temp_mean_emas']
# split datos
X = data_clus[select_features]
y = data_clus['Topoforma']
X_train, X_test_, y_train, y_test_ = train_test_split(X, y, test_size=0.20, random_state=1337)
# se realizo un estandar a los valores
scaler_t = StandardScaler()
X_train = scaler_t.fit_transform(X_train)
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test_)
# print(y_test_.columns)
# generamos un knn
model = BaggingClassifier(base_estimator=KNeighborsClassifier(algorithm='ball_tree',
                                                      leaf_size=30,
                                                      metric='minkowski',
                                                      metric_params=None,
                                                      n_jobs=None,
                                                      n_neighbors=5, p=2,
                                                      weights='uniform'),
                  bootstrap=True, bootstrap_features=False, max_features=1.0,
                  max_samples=1.0, n_estimators=100, n_jobs=None,
                  oob_score=False, random_state=None, verbose=0,
                  warm_start=False)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(metrics.classification_report(y_test_,pred))

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

white_button_style = {'background-color': 'white',
                      'color': 'black',
                      'height': '50px',
                      'width': '300px'}


app.layout = html.Div([
    # arriba dos
    html.Div([html.H5('Latitud'),
    dcc.Input(
        id='num-lat',
        type='number',
        value=16,
        min=15,
        max=32.6,step=0.1
    )],style={'width': '50%', 'display': 'inline-block','text-align': 'center','background-color': 'lightgreen','padding-bottom': '1em'}),
    
    html.Div([html.H5('Longuitud'),dcc.Input(
        id='num-long',
        type='number',
        value=-90,
        min=-117.5,
        max=-86,step=0.1
    )],style={'width': '50%', 'display': 'inline-block','text-align': 'center','background-color': 'lightgreen', 'padding-bottom': '1em'}),
    

    # abajo 4
    html.Div([html.H5('Temp. Mínima EMAS'),
    dcc.Input(
        id='num-min-emas',
        type='number',
        value=5,
        min=2,
        max=27.5,step=0.1
    )],style={'width': '25%', 'display': 'inline-block','text-align': 'center','background-color': 'lightblue','padding-bottom': '1em'}),
    
    
    html.Div([html.H5('Temp. Mínima MERRA'),
    dcc.Input(
        id='num-min-merra',
        type='number',
        value=5,
        min=3,
        max=27.5,step=0.1
    )],style={'width': '25%', 'display': 'inline-block','text-align': 'center','background-color': 'lightblue','padding-bottom': '1em'}),
    
    
    html.Div([html.H5('Temp. Media EMAS'),
    dcc.Input(
        id='num-med-emas',
        type='number',
        value=15,
        min=9,
        max=32.5,step=0.1
    )],style={'width': '25%', 'display': 'inline-block','text-align': 'center','background-color': 'rgb(255, 85, 85)','padding-bottom': '1em'}),
    
    
    html.Div([html.H5('Temp. Media MERRA'),
    dcc.Input(
        id='num-med-merra',
        type='number',
        value=15,
        min=9,
        max=32,step=0.1
    )],style={'width': '25%', 'display': 'inline-block','text-align': 'center','background-color': 'rgb(255, 85, 85)','padding-bottom': '1em'}),
    
    
    html.Div([html.H5('Clasificar'),
    html.Button(id='submit-button-state', n_clicks=0, children='Calcular', style=white_button_style)],
             style={'width': '100%', 'display': 'inline-block','text-align': 'center','background-color': 'rgb(10, 10, 10)','color':'white','padding-bottom': '1em'}),
    
    html.Div(id='display-selected-values'),

    dcc.Graph(id='proba_graph'),
    # dcc.Graph(id='map_graph')
])



# fig = px.scatter(df, x="x", y="y", color="fruit", custom_data=["customdata"])


def graf_proba(vals):
    global model
    topoformas = ['Valle', 'Llanura', 'Sierra', 'Meseta', 'Cuerpo de agua','Lomerío', 'Bajada', 'Desconocido', 'Cañon', 'Playa o Barra']
    prueba = scaler.transform(np.array(vals).reshape(1,6))
    # print(prueba)
    # predic_proba_name = model.predict(prueba.transpose())
    predic_proba_ = model.predict_proba(prueba)
    
    grapf_prueba = np.array([np.round(predic_proba_[0],3),topoformas]).transpose()
    # graficas
    # dat = pd.DataFrame(data=grapf_prueba,columns=['Values','Topoformas'])
    # print(dat.shape)
    
    dat = pd.DataFrame({'Values':predic_proba_[0],
                        'Topoformas':topoformas})
    fig = px.bar(data_frame=dat,y='Values',x='Topoformas',color='Topoformas')
    
    return fig,predic_proba_
    
@app.callback(
    Output('display-selected-values', 'children'),
    Output('proba_graph', 'figure'),
    # Output('map_graph', 'figure'),
    Input('submit-button-state', 'n_clicks'),
    State('num-lat', 'value'),
    State('num-long', 'value'),
    State('num-min-emas', 'value'),
    State('num-min-merra', 'value'),
    State('num-med-emas', 'value'),
    State('num-med-merra', 'value'))
def update_output(n_clicks, lat, lon, min_emas, min_merra, med_emas, med_merra):
    global pred
    prueba_ = [float(lat),float(lon), float(min_merra), float(min_emas), float(med_merra), float(med_emas)]
    
    fig,predic_proba_ = graf_proba(prueba_)
    # Valor para agregar
    valor_agregado = pd.DataFrame({'Latitud':[float(lat)], 'Longitud':[float(lon)],
            'Temp_min_merra':[float(min_merra)], 'Temp_min_emas':[float(min_emas)],
            'Temp_mean_merra':[float(med_merra)], 'Temp_mean_emas':[float(med_merra)] })
    X_tes = X_test_.copy()
    X_tes.append(valor_agregado)
    y_te = y_test_
    y_te.append(pd.Series(['Clasificado']),ignore_index=True)
    
    # fig mapa
    fig_2 = px.scatter_mapbox(data_frame=X_tes,lat='Latitud',lon='Longitud',zoom=4, height=400,color=y_te)
    fig_2.update_layout(mapbox_style="open-street-map")
    fig_2.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    
    str_send='Valores = ' + str(prueba_) + ' Resultados = ' + str(predic_proba_)
    # return fig,fig_2
    return str_send,fig

if __name__ == '__main__':
    app.run_server(debug=True,host="0.0.0.0", port=9696)