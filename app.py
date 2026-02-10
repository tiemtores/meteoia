import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import folium
import geopandas as gpd
from streamlit_folium import st_folium
import plotly.express as px



# Configuration de la page 
st.set_page_config( page_title="Outils d'aide à la prevision des phenomenes extremes en Afrique de l'Ouest", 
page_icon="🌍", 
layout="wide" # ✅ met la page en pleine largeur 
)

# Bannière personnalisée
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #87CEEB, #4682B4); /* dégradé ciel bleu */
        color: white;
        padding: 25px;
        border-radius: 10px;
        text-align: center;
        font-size: 28px;  /* ✅ taille du texte agrandie */
        font-weight: bold;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
    ">
    🌦️ Outils d'aide à la prévision des phénomènes extrêmes en Afrique de l'Ouest
    </div>
    """,
    unsafe_allow_html=True
)

# Contenu principal
st.write("Bienvenue dans l'application de prévision climatique.")

# Information sur le modele

with st.expander("Information sur l'outil", expanded=False):
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #1E3C72, #2A5298);
            color: white;
            padding: 15px;
            border-radius: 8px;
            font-size: 16px;
            line-height: 1.6;
        ">
        Cet outil vise à faciliter le travail des prévisionnistes relatif à la survenue
        des phénomènes extrêmes en Afrique de l'Ouest 🌍. Il intègre plusieurs modèles
        (Transformers, U-NET, LSTM) adaptés aux zones climatiques régionales.
        </div>
        """,
        unsafe_allow_html=True
    )

####-------------------------------------------

st.sidebar.title("🔎 Critere de selection")

# Label stylisé avec taille réduite et texte visible
st.sidebar.markdown(
    """
    <div style="
        color:#1E3C72; 
        font-size:16px; 
        font-weight:bold; 
        text-align:left; 
        line-height:1.4;   /* ✅ augmente l'espacement vertical */
        word-wrap:break-word; /* ✅ force le retour à la ligne */
    ">
        Choisir un modele IA
    </div>
    """,
    unsafe_allow_html=True
)

# Selectbox classique (fiable)
model_choice = st.sidebar.selectbox(
    "",
    ("PrecipFormer Transformer", "U-NET", "LSTM Attention"),
)
st.write(f"📌 Modèle sélectionné : **{model_choice}**")

# Pour la prevision

st.sidebar.markdown(
    """
    <div style="
        color:#1E3C72; 
        font-size:16px; 
        font-weight:bold; 
        text-align:left; 
        line-height:1.4;   /* ✅ augmente l'espacement vertical */
        word-wrap:break-word; /* ✅ force le retour à la ligne */
    ">
    Periode de prevision
    </div>
    """,
    unsafe_allow_html=True
)

period_choice = st.sidebar.radio(
    "",
    ("24h", "3 jours", "7 jours")
)

if st.sidebar.button("Lancer la prévision"):
    st.sidebar.success(f"Prévision lancée avec {model_choice} sur {period_choice}")


# Associer chaque modèle à ses 4 fichiers CSV
model_files = {
    "PrecipFormer Transformer": {
        "training": "data/pft/training_history_precipformer.csv",
        "metrics": "data/pft/model_metrics_precipformer.csv",
        "maps": "data/pft/precipitation_maps_precipformer.csv",
        "graph": "data/pft/precipitation_graph_precipformer.csv",
        "3maps":"data/pft/precipitation_3maps_precipformer.csv"
    },
    "U-NET": {
        "training": "data/unet/training_history_unet.csv",
        "metrics": "data/unet/model_metrics_unet.csv",
        "maps": "data/unet/precipitation_maps_unet.csv",
        "graph": "data/unet/precipitation_graph_unet.csv",
        "3maps":"data/unet/precipitation_3maps_unet.csv"
        
    },
    "LSTM Attention": {
        "training": "data/muatb/training_history_muatb.csv",
        "metrics": "data/muatb/model_metrics_muatb.csv",
        "maps": "data/muatb/precipitation_maps_muatb.csv",
        "graph": "data/muatb/precipitation_graph_muatb.csv",
        "3maps":"data/muatb/precipitation_3maps_muatb.csv"
    }
}



# Fonction personnalisée pour les titres
def custom_title(text):
    st.markdown(
        f"""
        <h3 style="text-align:left; color:#1E3C72; font-size:22px; font-weight:bold;">
            {text}
        </h3>
        """,
        unsafe_allow_html=True
    )


# Charger les fichiers du modèle sélectionné
files = model_files[model_choice]

# 2️⃣ Résultats du modèle
df_results = pd.read_csv(files["metrics"])
#custom_title("📊 Résultats du modèle")
#st.dataframe(df_results)

# METRICS

# Titre avec taille réduite
custom_title("🎯 Métriques suivies")
c1, c2, c3, c4, c5 = st.columns(5)

# Style professionnel : fond ciel, texte blanc, compact
metric_style = """
    <div style="
        background: linear-gradient(135deg, #87CEEB, #4682B4); /* dégradé ciel */
        color:white; 
        padding:4px;   /* ✅ réduit la hauteur */
        border-radius:10px; 
        text-align:center; 
        box-shadow: 1px 1px 4px rgba(0,0,0,0.3);
        min-height:60px; /* ✅ hauteur minimale plus petite */
    ">
        <h4 style="font-size:16px; margin:2px;">{title}</h4>
        <p style="font-size:14px; font-weight:bold; margin:2px;">{value}</p>
    </div>
"""

c1.markdown(metric_style.format(title="MSE", value=f"{df_results['MSE'].mean():.2f}"), unsafe_allow_html=True)
c2.markdown(metric_style.format(title="RMSE", value=f"{df_results['RMSE'].mean():.2f}"), unsafe_allow_html=True)
c3.markdown(metric_style.format(title="MAE", value=f"{df_results['MAE'].mean():.2f}"), unsafe_allow_html=True)
c4.markdown(metric_style.format(title="R² Score", value=f"{df_results['R²'].mean():.2f}"), unsafe_allow_html=True)
c5.markdown(metric_style.format(title="POD", value=f"{df_results['POD'].mean():.2f}"), unsafe_allow_html=True)



# 1️⃣ Historique d’entraînement
df_train = pd.read_csv(files["training"])
custom_title("📈 Courbes d'entraînement")
metrics = list(df_train.columns)
selected_metrics = st.multiselect("Choisir les métriques :", metrics, default=["loss", "val_loss"])
st.line_chart(df_train[selected_metrics])


# 4️⃣ Évolution des précipitations moyennes
df_graph = pd.read_csv(files["graph"])
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_graph["Date"], y=df_graph["Observé"], mode="lines+markers",
                         name="Observé", line=dict(color="blue", width=2), marker=dict(size=6)))
fig.add_trace(go.Scatter(x=df_graph["Date"], y=df_graph["Prédit"], mode="lines+markers",
                         name="Prédit", line=dict(color="red", width=2, dash="dash"), marker=dict(size=6)))
fig.update_layout(custom_title(f"📉Évolution des précipitations moyennes - {model_choice}"),
                  xaxis_title="Date", yaxis_title="Précipitations moyennes (mm)",
                  template="plotly_white", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)



#//////////////////////////
# 3️⃣ Cartes de précipitations
custom_title("🗺️ Cartes des précipitations observées et prédites")

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import numpy as np
import streamlit as st

# Charger les données
df_maps = pd.read_csv(files["maps"])

# Reconstruire les grilles
lons = np.sort(df_maps["lon"].unique())
lats = np.sort(df_maps["lat"].unique())
lon2d, lat2d = np.meshgrid(lons, lats)
obs_map = df_maps.pivot_table(index="lat", columns="lon", values="obs").values
pred_map = df_maps.pivot_table(index="lat", columns="lon", values="pred").values

# ✅ Échelle commune
vmin = min(obs_map.min(), pred_map.min())
vmax = max(obs_map.max(), pred_map.max())

# Création des cartes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 3), subplot_kw={'projection': ccrs.PlateCarree()})

for ax in (ax1, ax2):
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

# Carte observée
im1 = ax1.pcolormesh(lon2d, lat2d, obs_map, cmap="Blues", shading="auto",
                     vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
ax1.set_title("Précipitations observées")
plt.colorbar(im1, ax=ax1, label="mm/jour")

# Carte prédite
im2 = ax2.pcolormesh(lon2d, lat2d, pred_map, cmap="Blues", shading="auto",
                     vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
ax2.set_title("Précipitations prédites")
plt.colorbar(im2, ax=ax2, label="mm/jour")

plt.tight_layout()
st.pyplot(fig)

#/////////////////////////

############################################## CARTE DES PRECIPITATIONS ET ERREURS

# --- Visualisation interactive dans Streamlit ---
# Charger le CSV
df_maps = pd.read_csv(files["3maps"])

custom_title("🗺️ Cartes interactives de précipitations")

# ✅ Échelle commune
vmin = min(df_maps["obs"].min(), df_maps["pred"].min())
vmax = max(df_maps["obs"].max(), df_maps["pred"].max())
max_error = np.max(np.abs(df_maps["error"]))

# --- Carte Observée ---
fig_obs = px.scatter(
    df_maps, x="lon", y="lat", color="obs",
    color_continuous_scale="Blues",
    range_color=[vmin, vmax],
    title="Précipitations observées",
    labels={"obs": "mm/jour"},
    hover_data=["lon", "lat", "obs", "date"]
)
fig_obs.update_geos(
    showcountries=True, countrycolor="black", showland=True, landcolor="lightgray"
)
fig_obs.update_layout(height=350, width=400)  # ✅ ajuster hauteur et largeur

# --- Carte Prédite ---
fig_pred = px.scatter(
    df_maps, x="lon", y="lat", color="pred",
    color_continuous_scale="Blues",
    range_color=[vmin, vmax],
    title="Précipitations prédites",
    labels={"pred": "mm/jour"},
    hover_data=["lon", "lat", "pred", "date"]
)
fig_pred.update_geos(
    showcountries=True, countrycolor="black", showland=True, landcolor="lightgray"
)
fig_pred.update_layout(height=350, width=400)  # ✅ ajuster hauteur et largeur

# --- Carte Erreur ---
fig_error = px.scatter(
    df_maps, x="lon", y="lat", color="error",
    color_continuous_scale="RdBu",
    range_color=[-max_error, max_error],
    title="Erreur (Obs - Prédit)",
    labels={"error": "mm/jour"},
    hover_data=["lon", "lat", "error", "date"]
)
fig_error.update_geos(
    showcountries=True, countrycolor="black", showland=True, landcolor="lightgray"
)
fig_error.update_layout(height=350, width=400)  # ✅ ajuster hauteur et largeur

# --- Affichage côte à côte ---
col1, col2, col3 = st.columns(3)
with col1:
    st.plotly_chart(fig_obs, use_container_width=True)
with col2:
    st.plotly_chart(fig_pred, use_container_width=True)
with col3:
    st.plotly_chart(fig_error, use_container_width=True)

# --- Bouton de téléchargement ---
csv = df_maps.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Télécharger les données en CSV",
    data=csv,
    file_name="precipitation_maps.csv",
    mime="text/csv",
    key="download_precipitation_csv"
)

###################### FIN CARTE PRECIPITATION



# =========================
# Carte interactive Folium
# =========================
custom_title("🗺️ Carte des précipitations prévues")

# Exemple de points avec intensité de pluie
locations = [
    {"lat": 12.34, "lon": -1.52, "precip": 20},  # Ouagadougou
    {"lat": 14.72, "lon": -17.45, "precip": 5},   # Dakar
    {"lat": 6.52, "lon": 3.37, "precip": 12},     # Lagos
]

m = folium.Map(location=[12, -1], zoom_start=5)

for loc in locations:
    folium.CircleMarker(
        location=[loc["lat"], loc["lon"]],
        radius=loc["precip"] / 2,
        popup=f"{loc['precip']} mm",
        color="blue",
        fill=True,
        fill_color="blue"
    ).add_to(m)

st_folium(m, width=1200, height=500)