from Model import *
from plotly.tools import mpl_to_plotly
import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd
import io
import base64
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

def get_factors_list(factors):
    if 'Breed_Type_' in factors:
        new_cols = get_col_names('Breed_Type_')
        factors.remove('Breed_Type_')
        factors += new_cols
    if 'Color_' in factors:
        new_cols = get_col_names('Color_')
        factors.remove('Color_')
        factors += new_cols
    return factors

app.layout = html.Div(children=[
    dbc.Alert(
            "Error: Must Choose Atleast One Input for Decision Tree",
            id="alert-auto",
            is_open=False,
            duration=4000,
            color="danger"
        ),
    dbc.Alert(
        "Updating Model: This May Take a Few Seconds",
        id = 'alert-update',
        is_open=False,
        duration=4000,
        color='success'
    ),
    html.H1(children='Adoption/Euthanization Model'),
    html.Div(children=[
        html.Div(children=[
        html.Img(id='example'),
        html.Div(id='accuracy')
        ]),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Div(children=[
        dcc.Checklist(
            id = 'factors',
            labelStyle={'display': 'block'},
            options=[
                {'label': 'Sex', 'value': 'Sex_Male'},
                {'label': u'Fixed Status', 'value': 'Fixed_Status_Fixed'},
                {'label': 'Intake Condition', 'value': 'Condition_Status_Normal'},
                {'label': 'Senior Status', 'value': 'Senior_Status_Senior'},
                {'label': 'Age', 'value': 'Age upon Outcome (months)'},
                {'label': 'Breed Type', 'value': 'Breed_Type_'},
                {'label': 'Color', 'value': 'Color_'}
            ],
            value=['Sex_Male', 'Fixed_Status_Fixed', 'Condition_Status_Normal', 'Senior_Status_Senior', 
            'Age upon Outcome (months)', 'Breed_Type_', 'Color_']
        ),
        html.Label('Tree Depth'),
        dcc.Slider(
            id = 'depth',
            min = 1,
            max = 3,
            value = 2,
            marks={i: str(i) for i in range(1, 5)},
        )])
    ], style={'display': 'flex', 'flex-direction': 'row'}),
    dcc.Graph(id='importance')
])

# Output Tree Diagram
@app.callback(
    dash.dependencies.Output('example', 'src'),
    dash.dependencies.Output('importance', 'figure'),
    dash.dependencies.Output('accuracy', 'children'),
    dash.dependencies.Input('factors', 'value'),
    dash.dependencies.Input('depth', 'value')
)
def update_figure(factors, depth):
    # Output Tree Diagram
    buf = io.BytesIO() # in-memory files
    factors = get_factors_list(factors)
    model_info = create_model(factors, depth)
    dtree = model_info[0]
    cols = model_info[1]
    accuracy = 'Model Accuracy: ' + str(model_info[2])
    visualize_tree(dtree, cols)
    plt.savefig(buf, format = "png",dpi=300, bbox_inches = "tight") # save to the above file object
    plt.close()
    data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements

    # Importance bar chart
    importance = find_importance(dtree, cols)
    imp_graph = px.bar(importance, x = 'Factor', y = 'Importance Level')
    imp_graph.update_layout(title_text='Feature Importance of Decision Tree', title_x=0.5)
    return "data:image/png;base64,{}".format(data), imp_graph, accuracy

@app.callback(
    dash.dependencies.Output("alert-auto", "is_open"),
    dash.dependencies.Input('factors', 'value'),
    [dash.dependencies.State("alert-auto", "is_open")]
)
# Toggle alert if no factors are chosen
def toggle_alert(factors, is_open):
    if len(factors) <= 0:
        return not is_open
    return is_open

@app.callback(
    dash.dependencies.Output("alert-update", "is_open"),
    dash.dependencies.Input('factors', 'value'),
    dash.dependencies.Input('depth', 'value'),
    [dash.dependencies.State("alert-update", "is_open")]
)
# Toggle alert to show model is updating and may take a few seconds
def toggle_alert_update(factors, depth, is_open):
    return not is_open

if __name__ == '__main__':
    app.run_server()
