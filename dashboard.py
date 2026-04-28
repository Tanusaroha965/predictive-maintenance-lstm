import dash
from dash import dcc, html
import plotly.graph_objs as go
import numpy as np

print("Starting dashboard...")

# Load saved outputs
mse = np.load("mse.npy")
prob = np.load("prob.npy")

threshold = np.mean(mse) + 3*np.std(mse)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Predictive Maintenance Dashboard", style={'textAlign': 'center'}),

    dcc.Graph(
        figure={
            'data': [
                go.Scatter(y=mse, name='Reconstruction Error', line=dict(color='blue')),
                go.Scatter(y=prob, name='Failure Probability', line=dict(color='orange')),
                go.Scatter(
                    y=[threshold]*len(mse),
                    name='Threshold',
                    line=dict(color='red', dash='dash')
                )
            ],
            'layout': go.Layout(
                title='Anomaly Detection & Failure Prediction',
                xaxis={'title': 'Time'},
                yaxis={'title': 'Value'}
            )
        }
    ),

    # ALERT MESSAGE
    html.H2(
        "⚠️ High Risk Detected!" if max(prob) > 0.7 else "✅ System Stable",
        style={
            'textAlign': 'center',
            'color': 'red' if max(prob) > 0.7 else 'green'
        }
    )
])

if __name__ == "__main__":
    app.run(debug=True)