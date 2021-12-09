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
server = app.server()

fig = visualize_tree(['Sex_Male', 'Fixed_Status_Fixed', 'Pitbull_Status_Pit Bull', 'Condition_Status_Normal', 'Senior_Status_Senior'])
plt.savefig('decision_tree.png',dpi=300, bbox_inches = "tight")

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
    html.H1(children='Adoption/Euthanization Model'),
    html.Div(children=[
        html.Img(id='example'),
        html.Br(),
        html.Br(),
        html.Br(),
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
        )
    ], style={'display': 'flex', 'flex-direction': 'row'}),
    dcc.Graph(id='importance')
])

# Output Tree Diagram
@app.callback(
    dash.dependencies.Output('example', 'src'),
    dash.dependencies.Input('factors', 'value')
)
def update_figure(factors):
    buf = io.BytesIO() # in-memory files
    if len(factors) <= 0:
        fig = plt.figure(figsize=(3,2))
    else:
        factors = get_factors_list(factors)
        visualize_tree(factors)
    plt.savefig(buf, format = "png",dpi=300, bbox_inches = "tight") # save to the above file object
    plt.close()
    data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
    return "data:image/png;base64,{}".format(data)

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

#Output Importance Barplot
@app.callback(
    dash.dependencies.Output('importance', 'figure'),
    dash.dependencies.Input('factors', 'value')
)
def update_figure(factors):
    factors = get_factors_list(factors)
    importance = find_importance(factors)
    return px.bar(importance, x = 'Factor', y = 'Importance Level')

if __name__ == '__main__':
    app.run_server()
