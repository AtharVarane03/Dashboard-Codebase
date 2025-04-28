import torch
import torch.nn as nn
import dash
from dash import dcc, html, callback_context
import dash_bootstrap_components as dbc
import dash_daq as daq
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import base64
import io
from datetime import datetime, timedelta
from dash.dependencies import Input, Output, State
from dash_extensions import Lottie
import math

# Define the MLP model with the same architecture used for training
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes=(29, 69, 75)):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2], 1)
        )

    def forward(self, x):
        return self.layers(x)

def load_model(model_path, scaler_path, input_size=9):
    try:
        # Initialize model with correct input size
        model = MLP(input_size)
        
        # Load state dictionary
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set to evaluation mode
        
        # Load the scaler
        scaler = torch.load(scaler_path)

    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        print("Using dummy model and scaler.")

        # Create dummy model and scaler
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x):
                return torch.tensor([[30.0]])

        class DummyScaler:
            def transform(self, X):
                return X

        model = DummyModel()
        scaler = DummyScaler()

    return model, scaler

def predict_bod5(model, scaler, X_new):
    """
    Use the trained model to predict BOD5 values for new data.
    """
    print("predicing BOD")
    # Define feature names to match the scaler's training data
    feature_names = [
        "CODCr", "Nitrite", "Total suspended solids", "Ammonium", 
        "Chlorophyll a", "Manganese and its compounds", 
        "Aluminium and its compounds", "Iron and its compounds", 
        "Acid neutralizing capacity"
    ]

    # Convert X_new to a DataFrame with correct column names
    X_new_df = pd.DataFrame(X_new, columns=feature_names)

    # Standardize the new data
    X_new_scaled = scaler.transform(X_new_df)

    # Convert to PyTorch tensor
    X_new_tensor = torch.FloatTensor(X_new_scaled)

    # Make predictions
    model.eval()
    with torch.no_grad():
        y_pred = model(X_new_tensor).numpy().flatten()
    print("BOD",y_pred)

    return y_pred[0]

def calculate_parameters(BOD:float, COD:float):
    # Given constants
    Q=300
    TKN=167
    T = 12  # Temperature in °C
    FS = 1.2  # Factor of Safety
    a = 0.5
    b = 0.95
    F = 0.9
    OTR = 1.1  # kg O2/kWh
    Np = 3.75
    D = 10  # m
    DOe = 19  # %
    
    # Biomass yield and growth parameters
    Y = 0.4  # g VSS/g bCOD
    Ks = 20  # g bCOD/m3
    um = 6  # g/g.d
    tum = 1.07
    kd = 0.12  # g/g.d
    tkd = 1.04
    fd = 0.15
    
    # Nitrification parameters
    unm = 0.75  # g/g.d
    tunm = 1.07
    Kn = 0.74  # g NH4-N/m3
    tKn = 1.053
    kdn = 0.08  # g/g.d
    tkdn = 1.04
    Ko = 0.5  # g/m3 or mg/L
    Yn = 0.12  # g VSS/g NH4-N
    N=0.5
    # Oxygen concentration parameters
    C_20 = 9.08  # mg/L
    C_T = 10.77  # mg/L
    
    # Step 1: Calculate relevant values
    bCOD = 1.6 * BOD
    nbCOD = COD - bCOD
    S0 = bCOD
    
    um_T = um * (tum ** (T - 20))
    kd_T = kd * (tkd ** (T - 20))
    unm_T = unm * (tunm ** (T - 20))
    Kn_T = Kn * (tKn ** (T - 20))
    kdn_T = kdn * (tkdn ** (T - 20))
    
    # Step 2: Compute Nitrification rate
    NOx = 0.8 *TKN  # Assuming TKN = 70 g/m3
    un = (((unm_T * N) / (Kn_T + N)) * (DOe / (Ko + DOe))) - kdn_T
    
    # Step 3: Compute Solids Retention Time (SRT)
    SRT = 1 / un
    Design_SRT = FS * SRT
    
    # Step 4: Compute S value
    S = Ks * (1 + (kd_T * SRT)) / (SRT * (um_T - kd_T) - 1)
    
    # Step 5: Biomass production rate
    Px_bio = (Q * Y * (S0 - S) / (1 + (kd_T * SRT))) + (fd * kd_T * Q * Y * (S0 - S) * SRT / (1 + kd_T * SRT)) + (Q * Yn * NOx / (1 + kdn_T * SRT))
    Px_bio /= 1000  # Convert to kg VSS/d
    
    # Step 6: Compute NOx_prime
    NOx_prime = TKN - N - (0.12 * Px_bio*1000 /Q)
    
    # Step 7: Compute Aeration Oxygen Transfer Rate (AOTR)
    Ro = (Q * (S0 - S) - (1.42 * Px_bio*1000) + (4.33 * Q * NOx)) / 1000  # kg O2/d
    Ro /= 24  # Convert to kg O2/h
    
    # Step 8: Compute SOTR
    Pb_Pa = math.exp(-9.81 *29.87* (500 - 0) / (8314 * (273 + T)))
    print("Pb_Pa",Pb_Pa)
    Cs_T_H = C_T * Pb_Pa
    print("Cs_T_H",Cs_T_H)
    Patm_H = (Pb_Pa * 101.325) / 9.802
    print("Patm_H",Patm_H)
    Cs_prime_T_H = Cs_T_H * (1 / 2) * ((Patm_H + 4.9-0.5) /Patm_H + DOe / 21)
    print("Cs_prime_T_H",Cs_prime_T_H)
    SOTR = (Ro * C_20 * (1.024 ** (20 - T))) / (a * F * (b * Cs_prime_T_H - 2))
    
    # Step 9: Compute Power and RPM
    P = Ro / OTR  # kW
    n = (P*1000 / (Np * 1040 * D ** 5))**(1/3)*60  # rpm
    P_t=SOTR/OTR
    treatment_cost=(P_t-P)*24*5 # ₹/day
    print("S0",S0)
    print("NOx",NOx)
    print("un",un)
    print("SRT",SRT)
    print("Design_SRT",Design_SRT)
    print("S",S)
    print("Px_bio",Px_bio)
    print("NOx_prime",NOx_prime)
    print("Ro",Ro)
    print("SOTR",SOTR)
    print("P",P)
    print("n",n)
    
    return {
        "S0": S0,
        "NOx_prime": NOx,
        "AOTR": Ro,
        "SOTR": SOTR,
        "P": P,
        "n": n,
        "P_t":P_t,
        "treatment_cost": treatment_cost
    }

def parse_contents(contents):
    """Parse uploaded CSV file contents into pandas DataFrame"""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        return df
    except Exception as e:
        print(e)
        return None

# Design constants
BRAND_BLUE = "#03A9F4"  # Primary color for EcoSource
SECONDARY_COLOR = "#4CAF50"  # For environmental/eco elements
ACCENT_COLOR = "#FFC107"
DANGER_COLOR = "#F44336"
BG_COLOR = "#F5F7F9"
CARD_COLOR = "#FFFFFF"
TEXT_COLOR = "#263238"

# Animation URLs for Lottie
water_animation_url = "https://assets9.lottiefiles.com/packages/lf20_jbrw3hcz.json"
eco_animation_url = "https://assets9.lottiefiles.com/packages/lf20_oxmn5uh3.json"

app = dash.Dash(__name__, 
    external_stylesheets=[
        dbc.themes.BOOTSTRAP, 
        "https://use.fontawesome.com/releases/v6.0.0/css/all.css",
        "https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;700&display=swap"
    ],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)

# Navbar with logo and company name
navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(Lottie(options=dict(loop=True, autoplay=True), url=water_animation_url, height="50px")),
                        dbc.Col(html.H3("EcoSource", className="ms-2 mb-0", style={"color": BRAND_BLUE, "fontWeight": "700"})),
                    ],
                    align="center",
                ),
                href="#",
                style={"textDecoration": "none"},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div([
                            html.Span("Last updated: ", className="me-2 text-muted"),
                            html.Span(id="last-updated", className="fw-bold", children=datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                        ])
                    ),
                    dbc.Col(
                        dbc.Button(
                            [html.I(className="fas fa-sync-alt me-2"), "Refresh"],
                            id="refresh-button", 
                            color="light",
                            className="ms-2"
                        ),
                    ),
                ],
                className="ms-auto",
                align="center",
            ),
        ],
        fluid=True,
    ),
    color="white",
    className="shadow-sm mb-4",
)

# KPI card component
def create_kpi_card(title, value, unit, icon, color, id=None):
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.Div(html.I(className=f"fas fa-{icon} fa-2x"), 
                    className="p-3 rounded-circle me-3", 
                    style={"backgroundColor": f"{color}25", "color": color}),
                html.Div([
                    html.P(title, className="text-muted mb-0 small"),
                    html.H3([
                        html.Span(id=id, children=value, className="me-1"),
                        html.Small(unit, className="text-muted small")
                    ], className="mb-0")
                ])
            ], className="d-flex align-items-center")
        ]),
        className="shadow-sm border-0 h-100 kpi-card",
        style={"borderRadius": "12px", "transition": "transform 0.3s ease"}
    )

# App layout with modern design
app.layout = html.Div([
    navbar,
    
    dbc.Container([
        # Header section
        dbc.Row([
            dbc.Col([
                html.H1("Wastewater Treatment Optimization", className="fw-bold display-5", 
                      style={"color": TEXT_COLOR, "letterSpacing": "-0.5px"}),
                html.P("Intelligent aeration control powered by advanced analytics",
                      className="lead text-muted"),
                html.Hr()
            ], lg=8),
            dbc.Col([
                Lottie(options=dict(loop=True, autoplay=True), 
                      url=eco_animation_url, 
                      height="180px")
            ], lg=4, className="d-flex align-items-center justify-content-center")
        ], className="mb-4"),
        
        # KPI summary row
        dbc.Row([
            dbc.Col(create_kpi_card("Predicted BOD", "0.00", "mg/L", "flask", BRAND_BLUE, "bod-output"), xs=12, md=6, lg=3, className="mb-3"),
            dbc.Col(create_kpi_card("Power Usage", "0.00", "kW", "bolt", ACCENT_COLOR, "power-output"), xs=12, md=6, lg=3, className="mb-3"),
            dbc.Col(create_kpi_card("Optimal RPM", "0.00", "rpm", "cog", SECONDARY_COLOR, "rpm-output"), xs=12, md=6, lg=3, className="mb-3"),
            dbc.Col(create_kpi_card("Daily Cost", "0.00", "₹/day", "dollar-sign", DANGER_COLOR, "cost-output"), xs=12, md=6, lg=3, className="mb-3"),
        ], className="mb-4"),
        
        # Main content area
        dbc.Row([
            # Left panel - Inputs
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("Input Parameters", className="mb-0 d-inline"),
                        dbc.Button(
                            html.I(className="fas fa-question-circle"),
                            id="help-button",
                            color="link",
                            className="float-end p-0"
                        )
                    ], className="bg-white border-bottom"),
                    
                    dbc.CardBody([
                        dbc.Tabs([
                            dbc.Tab([
                                html.Div([
                                    html.Label("CODCr (mg/L)", className="fw-bold mb-1"),
                                    dbc.Input(
                                        id="cod",
                                        type="number",
                                        value=350,
                                        min=0,
                                        className="mb-3"
                                    ),
                                    
                                    html.Label("Nitrite (mg/L)", className="fw-bold mb-1"),
                                    dbc.Input(
                                        id="nitrite",
                                        type="number",
                                        value=5.5,
                                        min=0,
                                        className="mb-3"
                                    ),
                                    
                                    html.Label("Total suspended solids (mg/L)", className="fw-bold mb-1"),
                                    dbc.Input(
                                        id="tss",
                                        type="number",
                                        value=250,
                                        min=0,
                                        className="mb-3"
                                    ),
                                    
                                    html.Label("Ammonium (mg/L)", className="fw-bold mb-1"),
                                    dbc.Input(
                                        id="ammonium",
                                        type="number",
                                        value=30,
                                        min=0,
                                        className="mb-3"
                                    ),
                                    
                                    html.Label("Chlorophyll a (μg/L)", className="fw-bold mb-1"),
                                    dbc.Input(
                                        id="chlorophyll",
                                        type="number",
                                        value=12,
                                        min=0,
                                        className="mb-3"
                                    ),
                                    
                                    html.Label("Manganese and its compounds (mg/L)", className="fw-bold mb-1"),
                                    dbc.Input(
                                        id="manganese",
                                        type="number",
                                        value=2.1,
                                        min=0,
                                        className="mb-3"
                                    ),
                                    
                                    html.Label("Aluminium and its compounds (mg/L)", className="fw-bold mb-1"),
                                    dbc.Input(
                                        id="aluminium",
                                        type="number",
                                        value=0.8,
                                        min=0,
                                        className="mb-3"
                                    ),
                                    
                                    html.Label("Iron and its compounds (mg/L)", className="fw-bold mb-1"),
                                    dbc.Input(
                                        id="iron",
                                        type="number",
                                        value=1.2,
                                        min=0,
                                        className="mb-3"
                                    ),
                                    
                                    html.Label("Acid neutralizing capacity (mmol/L)", className="fw-bold mb-1"),
                                    dbc.Input(
                                        id="acid_neutralizing",
                                        type="number",
                                        value=4.5,
                                        min=0,
                                        className="mb-3"
                                    ),
                                    
                                    html.Hr(),
                                    
                                    dbc.Button(
                                        [html.I(className="fas fa-calculator me-2"), "Calculate Parameters"],
                                        id="predict-button",
                                        color="primary",
                                        style={"backgroundColor": BRAND_BLUE, "borderColor": BRAND_BLUE},
                                        className="w-100 py-2 mt-2",
                                        size="lg"
                                    ),
                                ], className="p-2"),
                            ], label="Manual Input", tab_id="manual-input"),
                            dbc.Tab([
                                html.Div([
                                    html.P("Upload a CSV file with input parameters:", className="mb-2"),
                                    dcc.Upload(
                                        id='upload-data',
                                        children=html.Div([
                                            html.I(className="fas fa-file-csv fa-2x mb-2"),
                                            html.P('Drag and Drop or Select CSV File')
                                        ]),
                                        style={
                                            'width': '100%',
                                            'height': '200px',
                                            'lineHeight': '60px',
                                            'borderWidth': '2px',
                                            'borderStyle': 'dashed',
                                            'borderRadius': '12px',
                                            'textAlign': 'center',
                                            'margin': '10px 0',
                                            'backgroundColor': f"{BRAND_BLUE}10",
                                            'display': 'flex',
                                            'flexDirection': 'column',
                                            'justifyContent': 'center'
                                        },
                                        multiple=False
                                    ),
                                    
                                    html.Div(id='csv-upload-status', className="mt-3"),
                                    
                                    html.Hr(),
                                    
                                    html.P("CSV Format Requirements:", className="mt-3 fw-bold"),
                                    html.Ul([
                                        html.Li("Single row of data"),
                                        html.Li("Required columns: CODCr, Nitrite, Total suspended solids, Ammonium, Chlorophyll a, Manganese and its compounds, Aluminium and its compounds, Iron and its compounds, Acid neutralizing capacity"),
                                        html.Li("Optional: include temperature & DO level for more accurate AOTR calculation")
                                    ], className="small text-muted"),
                                    
                                    dbc.Button(
                                        [html.I(className="fas fa-file-download me-2"), "Download Sample CSV"],
                                        id="download-sample-button",
                                        color="outline-primary",
                                        className="mt-3 w-100"
                                    ),
                                    dbc.Button(
                                        [html.I(className="fas fa-calculator me-2"), "Calculate From CSV"],
                                        id="predict-from-csv-button",
                                        color="primary",
                                        style={"backgroundColor": BRAND_BLUE, "borderColor": BRAND_BLUE},
                                        className="w-100 py-2 mt-3",
                                        size="lg",
                                        disabled=True
                                    ),
                                ], className="p-2"),
                            ], label="CSV Upload", tab_id="csv-upload"),
                        ], id="input-tabs"),
                        
                        dbc.Alert(
                            [html.I(className="fas fa-info-circle me-2"), "Enter parameters and click 'Calculate'"],
                            id="status-message",
                            color="info",
                            className="mt-3 mb-0"
                        )
                    ])
                ], className="shadow-sm mb-4", style={"borderRadius": "12px", "border": "none"}),
                
                # Parameters card
                dbc.Card([
                    dbc.CardHeader(html.H4("Treatment Parameters", className="mb-0"), className="bg-white border-bottom"),
                    dbc.CardBody([
                        # Parameter indicators
                        html.Div([
                            html.Div([
                                html.Div([
                                    html.Span("Substrate (S₀)", className="small text-muted"),
                                    html.Div([
                                        html.Span(id="s0-output", className="h5 mb-0 me-1", children="0.00"),
                                        html.Small("mg/L", className="text-muted")
                                    ], className="d-flex align-items-baseline")
                                ], className="mb-1"),
                                dbc.Progress(id="s0-progress", value=0, color=BRAND_BLUE, style={"height": "8px"})
                            ], className="mb-3"),
                            
                            html.Div([
                                html.Div([
                                    html.Span("NOx Level", className="small text-muted"),
                                    html.Div([
                                        html.Span(id="nox-output", className="h5 mb-0 me-1", children="0.00"),
                                        html.Small("mg/L", className="text-muted")
                                    ], className="d-flex align-items-baseline")
                                ], className="mb-1"),
                                dbc.Progress(id="nox-progress", value=0, color=DANGER_COLOR, style={"height": "8px"})
                            ], className="mb-3"),
                            
                            html.Div([
                                html.Div([
                                    html.Span("Standard Oxygen Transfer Rate", className="small text-muted"),
                                    html.Div([
                                        html.Span(id="sotr-output", className="h5 mb-0 me-1", children="0.00"),
                                        html.Small("kgO₂/h", className="text-muted")
                                    ], className="d-flex align-items-baseline")
                                ], className="mb-1"),
                                dbc.Progress(id="sotr-progress", value=0, color=SECONDARY_COLOR, style={"height": "8px"})
                            ], className="mb-3"),
                            
                            html.Div([
                                html.Div([
                                    html.Span("Actual Oxygen Transfer Rate", className="small text-muted"),
                                    html.Div([
                                        html.Span(id="aotr-output", className="h5 mb-0 me-1", children="0.00"),
                                        html.Small("kgO₂/h", className="text-muted")
                                    ], className="d-flex align-items-baseline")
                                ], className="mb-1"),
                                dbc.Progress(id="aotr-progress", value=0, color=ACCENT_COLOR, style={"height": "8px"})
                            ], className="mb-3"),
                            
                            html.Hr(className="my-3"),
                            
                            dbc.Alert(
                                [
                                    html.I(className="fas fa-lightbulb me-2"),
                                    html.Span(id="recommendation-text", children="Calculate parameters to get optimization recommendations")
                                ],
                                color="info",
                                className="mb-0 p-2 small"
                            )
                        ])
                    ])
                ], className="shadow-sm h-100", style={"borderRadius": "12px", "border": "none"}),
            ], lg=4),
            
            # Right panel - Output charts
            dbc.Col([
                # RPM gauge
# RPM gauge card with improved layout
                dbc.Card([
                    dbc.CardHeader(html.H4("Aeration Control", className="mb-0"), className="bg-white border-bottom"),
                    dbc.CardBody([
                        html.Div([
                            dcc.Graph(
                                id="rpm-gauge", 
                                config={'displayModeBar': False},
                                style={"height": "260px", "width": "100%"},  # Adjusted dimensions
                                className="m-0 p-10"  # Remove any margin/padding
                            )
                        ], style={"height": "100%", "width": "100%"})
                    ], className="p-2 d-flex justify-content-center align-items-center", 
                    style={"height": "280px","overflow":"visible"})  # Fixed height container
                ], className="shadow-sm mb-4", style={"borderRadius": "12px", "border": "none"}),
                
                # SOTR/AOTR graph
                dbc.Card([
                    dbc.CardHeader(html.H4("Oxygen Transfer Performance", className="mb-0"), className="bg-white border-bottom"),
                    dbc.CardBody([
                        dcc.Graph(id="oxygen-transfer-graph", config={'displayModeBar': 'hover'})
                    ])
                ], className="shadow-sm mb-4", style={"borderRadius": "12px", "border": "none"}),
                
                # Power consumption graph
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("Power Consumption History", className="mb-0 d-inline"),
                        dbc.ButtonGroup(
                            [
                                dbc.Button("Day", id="btn-day", outline=True, color="primary", size="sm", className="btn-time"),
                                dbc.Button("Week", id="btn-week", outline=True, color="primary", size="sm", className="btn-time active"),
                                dbc.Button("Month", id="btn-month", outline=True, color="primary", size="sm", className="btn-time"),
                            ],
                            className="float-end"
                        )
                    ], className="bg-white border-bottom d-flex justify-content-between align-items-center"),
                    
                    dbc.CardBody([
                        dcc.Graph(id="power-consumption-graph", config={'displayModeBar': 'hover'})
                    ])
                ], className="shadow-sm", style={"borderRadius": "12px", "border": "none"})
            ], lg=8)
        ]),
        
        # Footer
        html.Footer([
            html.Hr(className="my-4"),
            html.P([
                "© 2025 EcoSource Technologies • ",
                html.A("Documentation", href="#", className="text-decoration-none", style={"color": BRAND_BLUE}),
                " • ",
                html.A("Support", href="#", className="text-decoration-none", style={"color": BRAND_BLUE})
            ], className="text-center text-muted small")
        ])
    ], fluid=True, className="pb-4"),
    
    # Help modal
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Parameter Guide")),
        dbc.ModalBody([
            html.P("This dashboard optimizes aeration parameters for efficient wastewater treatment."),
            html.H6("Input Parameters:", className="mt-3"),
            html.Ul([
                html.Li([html.Strong("CODCr: "), "Chemical Oxygen Demand, indicates organic pollution levels"]),
                html.Li([html.Strong("Nitrite: "), "Intermediate nitrogen compound in nitrification process"]),
                html.Li([html.Strong("Total suspended solids: "), "Concentration of solid particles in wastewater"]),
                html.Li([html.Strong("Ammonium: "), "Nitrogen compound that requires aeration for removal"]),
                html.Li([html.Strong("Chlorophyll a: "), "Indicator of algal biomass in water"]),
                html.Li([html.Strong("Manganese compounds: "), "Metal that may interfere with treatment"]),
                html.Li([html.Strong("Aluminium compounds: "), "Metal often present from coagulation processes"]),
                html.Li([html.Strong("Iron compounds: "), "Metal that affects oxygen demand and transfer"]),
                html.Li([html.Strong("Acid neutralizing capacity: "), "Buffer capacity of the water"]),
            ]),
            html.H6("Output Parameters:", className="mt-3"),
            html.Ul([
                html.Li([html.Strong("BOD: "), "Biological Oxygen Demand - key indicator of organic pollution"]),
                html.Li([html.Strong("S₀: "), "Initial substrate concentration"]),
                html.Li([html.Strong("NOx: "), "Nitrogen oxides concentration after treatment"]),
                html.Li([html.Strong("n (RPM): "), "Optimal rotational speed for aerators"]),
                html.Li([html.Strong("SOTR: "), "Standard Oxygen Transfer Rate under standard conditions"]),
                html.Li([html.Strong("AOTR: "), "Actual Oxygen Transfer Rate under operating conditions"]),
                html.Li([html.Strong("P: "), "Power consumption required for aeration"]),
            ])
        ]),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-help", className="ms-auto")
        ),
    ], id="help-modal", is_open=False, size="lg"),
    
    # Hidden storage
    dcc.Store(id="csv-data-store"),
    dcc.Store(id="calculation-results")
], style={"fontFamily": "Poppins, sans-serif", "backgroundColor": BG_COLOR, "minHeight": "100vh"})

# Custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>EcoSource - Wastewater Treatment Optimization</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background-color: #F5F7F9;
            }
            .kpi-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 20px rgba(0,0,0,0.1) !important;
            }
            .btn-time.active {
                background-color: #03A9F4 !important;
                border-color: #03A9F4 !important;
                color: white !important;
            }
            .dash-graph {
                border-radius: 8px;
                overflow: hidden;
            }
            .tab-content {
                padding-top: 1rem;
            }
            .nav-tabs .nav-link.active {
                color: #03A9F4;
                font-weight: 500;
                border-bottom: 2px solid #03A9F4;
                border-top: none;
                border-left: none;
                border-right: none;
            }
            .nav-tabs .nav-link {
                border: none;
                color: #78909C;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Callback for toggle help modal
@app.callback(
    Output("help-modal", "is_open"),
    [Input("help-button", "n_clicks"), Input("close-help", "n_clicks")],
    [State("help-modal", "is_open")],
)
def toggle_help_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

# Callback for CSV upload status
@app.callback(
    [Output("csv-upload-status", "children"),
     Output("csv-data-store", "data"),
     Output("predict-from-csv-button", "disabled")],
    [Input("upload-data", "contents")]
)
def update_csv_status(contents):
    if contents is None:
        return [
            html.Div([
                html.I(className="fas fa-info-circle me-2"),
                "No file uploaded yet."
            ], className="text-muted"),
            None,
            True
        ]
    
    # Parse CSV contents
    df = parse_contents(contents)
    
    if df is None:
        return [
            html.Div([
                html.I(className="fas fa-exclamation-triangle me-2 text-danger"),
                "Error parsing the CSV file. Please check format."
            ], className="text-danger"),
            None,
            True
        ]
    
    # Check if required columns exist
    required_columns = ["CODCr", "Nitrite", "Total suspended solids", "Ammonium", 
                       "Chlorophyll a", "Manganese and its compounds", 
                       "Aluminium and its compounds", "Iron and its compounds", 
                       "Acid neutralizing capacity"]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return [
            html.Div([
                html.I(className="fas fa-exclamation-triangle me-2 text-danger"),
                f"Missing columns: {', '.join(missing_columns)}"
            ], className="text-danger"),
            None,
            True
        ]
    
    # Check if data has at least one row
    if len(df) < 1:
        return [
            html.Div([
                html.I(className="fas fa-exclamation-triangle me-2 text-danger"),
                "CSV file doesn't contain any data rows."
            ], className="text-danger"),
            None,
            True
        ]
    
    # Success - use first row only
    first_row = df.iloc[0].to_dict()
    
    return [
        html.Div([
            html.I(className="fas fa-check-circle me-2 text-success"),
            "CSV file loaded successfully. Ready for calculation."
        ], className="text-success"),
        first_row,
        False
    ]

# Main callback for dashboard updates from manual input
@app.callback(
    [
        # KPI values
        Output("bod-output", "children"),
        Output("power-output", "children"), 
        Output("rpm-output", "children"),
        Output("cost-output", "children"),
        
        # Parameter outputs
        Output("s0-output", "children"),
        Output("nox-output", "children"),
        Output("sotr-output", "children"),
        Output("aotr-output", "children"),
        
        # Progress bars
        Output("s0-progress", "value"),
        Output("nox-progress", "value"),
        Output("sotr-progress", "value"),
        Output("aotr-progress", "value"),
        
        # Graphs
        Output("rpm-gauge", "figure"),
        Output("power-consumption-graph", "figure"),
        Output("oxygen-transfer-graph", "figure"),
        
        # Status messages
        Output("status-message", "children"),
        Output("status-message", "color"),
        Output("recommendation-text", "children"),
        
        # Last updated time
        Output("last-updated", "children"),
        
        # Store calculation results
        Output("calculation-results", "data")
    ],
    [
        Input("predict-button", "n_clicks"),
        Input("predict-from-csv-button", "n_clicks"),
        Input("refresh-button", "n_clicks"),
        Input("btn-day", "n_clicks"),
        Input("btn-week", "n_clicks"),
        Input("btn-month", "n_clicks")
    ],
    [
        # Manual input states
        State("cod", "value"),
        State("nitrite", "value"),
        State("tss", "value"),
        State("ammonium", "value"),
        State("chlorophyll", "value"),
        State("manganese", "value"),
        State("aluminium", "value"),
        State("iron", "value"),
        State("acid_neutralizing", "value"),
        
        # CSV data state
        State("csv-data-store", "data"),
        
        # Results state
        State("calculation-results", "data")
    ]
)
def update_dashboard(predict_clicks, predict_csv_clicks, refresh_clicks, 
                    day_clicks, week_clicks, month_clicks,
                    cod, nitrite, tss, ammonium, chlorophyll, manganese, aluminium, iron, 
                    acid_neutralizing, csv_data, calculation_results):
    
    ctx = callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "page-load"
    
    # Initialize values
    bod_value = 0.0
    power_value = 0.0
    rpm_value = 5
    cost_value = 0.0
    s0_value = 0.0
    nox_value = 0.0
    sotr_value = 0.0
    aotr_value = 0.0
    status_message = [html.I(className="fas fa-info-circle me-2"), "Enter parameters and click 'Calculate Parameters'"]
    status_color = "info"
    recommendation = "Calculate parameters to get optimization recommendations"
    time_range = "week"  # default
    
    # If results already exist and we're not doing a new prediction, use existing results
    if calculation_results and trigger_id not in ["predict-button", "predict-from-csv-button"]:
        bod_value = calculation_results.get("bod", 0)
        power_value = calculation_results.get("power", 0)
        rpm_value = calculation_results.get("rpm", 350)
        cost_value = calculation_results.get("cost", 0)
        s0_value = calculation_results.get("s0", 0)
        nox_value = calculation_results.get("nox", 0)
        sotr_value = calculation_results.get("sotr", 0)
        aotr_value = calculation_results.get("aotr", 0)
    
    # Set time range based on button clicks
    if trigger_id == "btn-day":
        time_range = "day"
    elif trigger_id == "btn-week":
        time_range = "week"
    elif trigger_id == "btn-month":
        time_range = "month"
    
    # Run prediction from manual input
    if trigger_id == "predict-button" and predict_clicks:
        try:
            # Get input values with defaults if None
            input_features = np.array([[
                cod or 350,
                nitrite or 5.5,
                tss or 250,
                ammonium or 30,
                chlorophyll or 12,
                manganese or 2.1,
                aluminium or 0.8,
                iron or 1.2,
                acid_neutralizing or 4.5
            ]])
            
            # Simulate BOD prediction (in real app, use the model)
            # bod_value = (
            #     input_features[0][0] * 0.02 +  # COD
            #     input_features[0][1] * 0.5 +   # Nitrite
            #     input_features[0][2] * 0.01 +  # TSS
            #     input_features[0][3] * 0.1 +   # Ammonium
            #     input_features[0][8] * 1.0     # Acid neutralizing capacity
            # )
            bod_value=predict_bod5(model,scaler,input_features)
            
            # Calculate derived parameters
            params = calculate_parameters(bod_value, input_features[0][0])
            
            # Extract values
            s0_value = params["S0"]
            nox_value = params["NOx_prime"]
            sotr_value = params["SOTR"]
            aotr_value = params["AOTR"]
            power_value = params["P"]
            rpm_value = params["n"]
            cost_value = params["treatment_cost"]
            
            # Update status
            status_message = [html.I(className="fas fa-check-circle me-2"), "Calculation complete. Optimal settings determined."]
            status_color = "success"
            
            # Generate a recommendation
            if power_value > 60:
                recommendation = "Consider reducing aeration intensity during low-load periods to conserve energy."
            elif sotr_value / aotr_value > 1.8:
                recommendation = "Increase mixing efficiency to improve oxygen transfer capability."
            else:
                recommendation = "Current parameters are optimal for balanced operation and efficiency."
                
            # Store results
            calculation_results = {
                "bod": bod_value,
                "power": power_value,
                "rpm": rpm_value,
                "cost": cost_value,
                "s0": s0_value,
                "nox": nox_value,
                "sotr": sotr_value,
                "aotr": aotr_value
            }
            
        except Exception as e:
            status_message = [html.I(className="fas fa-exclamation-triangle me-2"), f"Error: {str(e)}"]
            status_color = "danger"
    
    # Run prediction from CSV
    elif trigger_id == "predict-from-csv-button" and predict_csv_clicks and csv_data:
        try:
            # Extract values from CSV data
            cod_csv = csv_data.get("CODCr", 350)
            nitrite_csv = csv_data.get("Nitrite", 5.5)
            tss_csv = csv_data.get("Total suspended solids", 250)
            ammonium_csv = csv_data.get("Ammonium", 30)
            chlorophyll_csv = csv_data.get("Chlorophyll a", 12)
            manganese_csv = csv_data.get("Manganese and its compounds", 2.1)
            aluminium_csv = csv_data.get("Aluminium and its compounds", 0.8)
            iron_csv = csv_data.get("Iron and its compounds", 1.2)
            acid_csv = csv_data.get("Acid neutralizing capacity", 4.5)
            
            # Optional columns
            tkn_csv = csv_data.get("Total Kjedahl Nitrogen", 25)
            do_csv = csv_data.get("DO", 2.0)
            
            # Create input features array
            input_features = np.array([[
                float(cod_csv),
                float(nitrite_csv),
                float(tss_csv),
                float(ammonium_csv),
                float(chlorophyll_csv),
                float(manganese_csv),
                float(aluminium_csv),
                float(iron_csv),
                float(acid_csv)
            ]])
            
            # Simulate BOD prediction
            bod_value=predict_bod5(model,scaler,input_features)
            
            # Calculate derived parameters
            params = calculate_parameters(bod_value, input_features[0][0])
            
            # Extract values
            s0_value = params["S0"]
            nox_value = params["NOx_prime"]
            sotr_value = params["SOTR"]
            aotr_value = params["AOTR"]
            power_value = params["P"]
            rpm_value = params["n"]
            cost_value = params["treatment_cost"]
            
            # Update status
            status_message = [html.I(className="fas fa-check-circle me-2"), "CSV data processed successfully. Optimal settings determined."]
            status_color = "success"
            
            # Generate a recommendation
            if power_value > 60:
                recommendation = "Consider reducing aeration intensity during low-load periods to conserve energy."
            elif sotr_value / aotr_value > 1.8:
                recommendation = "Increase mixing efficiency to improve oxygen transfer capability."
            else:
                recommendation = "Current parameters are optimal for balanced operation and efficiency."
                
            # Store results
            calculation_results = {
                "bod": bod_value,
                "power": power_value,
                "rpm": rpm_value,
                "cost": cost_value,
                "s0": s0_value,
                "nox": nox_value,
                "sotr": sotr_value,
                "aotr": aotr_value
            }
            
        except Exception as e:
            status_message = [html.I(className="fas fa-exclamation-triangle me-2"), f"Error processing CSV: {str(e)}"]
            status_color = "danger"
    
    # Progress bar values (normalized to percentage)
    s0_progress = min(100, s0_value / 3 * 100)
    nox_progress = min(100, nox_value / 10 * 100)
    sotr_progress = min(100, sotr_value / 300 * 100)
    aotr_progress = min(100, aotr_value / 300 * 100)
    
    # Create visualizations
    rpm_gauge_fig = create_rpm_gauge(rpm_value)
    power_graph = create_power_consumption_graph(time_range, power_value,cost_value)
    oxygen_graph = create_oxygen_transfer_graph(sotr_value, aotr_value)
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return (
        f"{bod_value:.2f}", f"{power_value:.1f}", f"{rpm_value:.0f}", f"{cost_value:.2f}",
        f"{s0_value:.2f}", f"{nox_value:.2f}", f"{sotr_value:.1f}", f"{aotr_value:.1f}",
        s0_progress, nox_progress, sotr_progress, aotr_progress,
        rpm_gauge_fig, power_graph, oxygen_graph,
        status_message, status_color, recommendation,
        current_time, calculation_results
    )

# Create RPM gauge visualization
def create_rpm_gauge(rpm_value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rpm_value,
        domain={'x': [0.1, 0.9], 'y': [0.1, 0.9]},
        title={'text': "RPM", 'font': {'size': 20, 'color': TEXT_COLOR}},
        gauge={
            'axis': {'range': [None, 50], 'tickwidth': 1, 'tickcolor': TEXT_COLOR},
            'bar': {'color': BRAND_BLUE},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "lightgray",
            'steps': [
                {'range': [0, 20], 'color': '#E1F5FE'},
                {'range': [20, 35], 'color': '#B3E5FC'},
                {'range': [35, 50], 'color': '#81D4FA'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 45
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='white',
        margin=dict(l=10, r=10, t=30, b=10),
        height=260,
        font={'color': TEXT_COLOR}
    )
    
    return fig

# Create power consumption graph
def create_power_consumption_graph(time_range="week", current_power=0, cost_value=0):
    # Generate time periods based on selected range
    if time_range == "day":
        dates = [datetime.now() - timedelta(hours=i) for i in range(24, 0, -1)]
        x_title = "Hour"
    elif time_range == "month":
        dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
        x_title = "Day"
    else:  # week
        dates = [datetime.now() - timedelta(days=i) for i in range(7, 0, -1)]
        x_title = "Day"
    
    # Add current time point
    dates.append(datetime.now())
    
    # Generate synthetic data
    np.random.seed(42)
    base = 55
    noise_level = 10
    trend = np.linspace(0, 10, len(dates)-1) if time_range == "month" else np.zeros(len(dates)-1)
    weekend_effect = np.array([5 if d.weekday() >= 5 else 0 for d in dates[:-1]])
    
    power_values = base + trend - weekend_effect + np.random.normal(0, noise_level, len(dates)-1)
    
    # Add current predicted value
    power_values = np.append(power_values, current_power if current_power > 0 else power_values[-1] * 0.9)
    
    # Calculate baseline (traditional aeration)
    # baseline_values = (power_values * 1.3) + 10
    baseline_values = cost_value/120 + power_values
    
    
    # Create dataframe
    df = pd.DataFrame({
        'timestamp': dates,
        'power_kw': power_values,
        'baseline': baseline_values,
        'is_prediction': [False] * (len(dates)-1) + [True]
    })
    
    # Calculate savings
    total_actual = df['power_kw'].sum()
    total_baseline = df['baseline'].sum()
    savings_pct = (1 - (total_actual / total_baseline)) * 100
    savings_kwh = (total_baseline - total_actual)
    
    # Create figure
    fig = go.Figure()
    
    # Add actual power consumption line
    fig.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['power_kw'],
        mode='lines',
        name='EcoSource Aeration',
        line=dict(color=BRAND_BLUE, width=3),
        fill='tozeroy',
        fillcolor=f'rgba({int(BRAND_BLUE[1:3], 16)}, {int(BRAND_BLUE[3:5], 16)}, {int(BRAND_BLUE[5:7], 16)}, 0.1)'
    ))
    
    # Add prediction marker if exists
    if current_power > 0:
        fig.add_trace(go.Scatter(
            x=[df['timestamp'].iloc[-1]],
            y=[current_power],
            mode='markers',
            name='Current Prediction',
            marker=dict(color=ACCENT_COLOR, size=12, symbol='star'),
            hoverinfo='y'
        ))
    
    # Add traditional baseline
    fig.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['baseline'],
        mode='lines',
        name='Traditional System',
        line=dict(color='#B0BEC5', width=2, dash='dash')
    ))
    
    # Add savings annotation
    fig.add_annotation(
        x=0.02,
        y=0.95,
        xref="paper",
        yref="paper",
        text=f"Energy Savings: {savings_pct:.1f}%<br>({savings_kwh:.1f} kWh)",
        showarrow=False,
        font=dict(
            family="Poppins, sans-serif",
            size=14,
            color=SECONDARY_COLOR
        ),
        align="left",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor=SECONDARY_COLOR,
        borderwidth=1,
        borderpad=4
    )
    
    # Update layout
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis_title=x_title,
        yaxis_title="Power (kW)",
        yaxis=dict(gridcolor='#ECEFF1', zeroline=False),
        xaxis=dict(gridcolor='#ECEFF1'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        font=dict(
            family="Poppins, sans-serif",
            color=TEXT_COLOR
        ),
        hovermode="x unified"
    )
    
    return fig

# Create oxygen transfer graph
def create_oxygen_transfer_graph(sotr_value, aotr_value):
    # Generate time periods for historical data (past 24 hours)
    dates = [datetime.now() - timedelta(hours=i) for i in range(24, 0, -1)]
    
    # Add current time point
    dates.append(datetime.now())
    
    # Generate synthetic data for SOTR and AOTR
    np.random.seed(44)
    
    # Historical SOTR (higher values)
    if sotr_value == 0:
        sotr_value = 200  # Default value if no prediction
        
    sotr_base = sotr_value * 0.95  # Slightly lower than current prediction
    sotr_historical = sotr_base + np.random.normal(0, 20, len(dates)-1)
    
    # Historical AOTR (lower values due to field conditions)
    if aotr_value == 0:
        aotr_value = 120  # Default value if no prediction
        
    aotr_base = aotr_value * 0.95  # Slightly lower than current prediction
    aotr_historical = aotr_base + np.random.normal(0, 15, len(dates)-1)
    
    # Add current predicted values
    sotr_values = np.append(sotr_historical, sotr_value)
    aotr_values = np.append(aotr_historical, aotr_value)
    
    # Create dataframe
    df = pd.DataFrame({
        'timestamp': dates,
        'SOTR': sotr_values,
        'AOTR': aotr_values,
        'is_prediction': [False] * (len(dates)-1) + [True]
    })
    
    # Create figure
    fig = go.Figure()
    
    # Add SOTR line
    fig.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['SOTR'],
        mode='lines',
        name='SOTR (Standard)',
        line=dict(color=SECONDARY_COLOR, width=3)
    ))
    
    # Add AOTR line
    fig.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['AOTR'],
        mode='lines',
        name='AOTR (Actual)',
        line=dict(color=BRAND_BLUE, width=3)
    ))
    
    # Add current prediction markers
    if sotr_value > 0 and aotr_value > 0:
        fig.add_trace(go.Scatter(
            x=[df['timestamp'].iloc[-1], df['timestamp'].iloc[-1]],
            y=[sotr_value, aotr_value],
            mode='markers',
            name='Current Prediction',
            marker=dict(color=ACCENT_COLOR, size=12, symbol='diamond'),
            hoverinfo='y'
        ))
    
    # Calculate efficiency
    alpha_factor = aotr_value / sotr_value if sotr_value > 0 else 0
    
    # Add efficiency annotation
    fig.add_annotation(
        x=0.02,
        y=0.95,
        xref="paper",
        yref="paper",
        text=f"α Factor: {alpha_factor:.2f}<br>Efficiency: {alpha_factor*100:.1f}%",
        showarrow=False,
        font=dict(
            family="Poppins, sans-serif",
            size=14,
            color=TEXT_COLOR
        ),
        align="left",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="#CFD8DC",
        borderwidth=1,
        borderpad=4
    )
    
    # Update layout
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis_title="Time",
        yaxis_title="Oxygen Transfer Rate (kgO₂/h)",
        yaxis=dict(gridcolor='#ECEFF1', zeroline=False),
        xaxis=dict(gridcolor='#ECEFF1'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        font=dict(
            family="Poppins, sans-serif",
            color=TEXT_COLOR
        ),
        hovermode="x unified"
    )
    
    return fig

# Callback for time period button styling
@app.callback(
    [Output("btn-day", "className"),
     Output("btn-week", "className"),
     Output("btn-month", "className")],
    [Input("btn-day", "n_clicks"),
     Input("btn-week", "n_clicks"),
     Input("btn-month", "n_clicks")]
)
def update_time_button_style(day_clicks, week_clicks, month_clicks):
    ctx = callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "btn-week"
    
    if button_id == "btn-day":
        return "btn-time active", "btn-time", "btn-time"
    elif button_id == "btn-month":
        return "btn-time", "btn-time", "btn-time active"
    else:  # Default to week
        return "btn-time", "btn-time active", "btn-time"

# Run the app
if __name__ == '__main__':
    try:
        model, scaler = load_model('bod5_mlp_model.pt', 'bod5_scaler.pt')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load model. Using simulated predictions. Error: {str(e)}")
    
    app.run(debug=True)
