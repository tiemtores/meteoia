
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium

# Configuration de la page 
st.set_page_config( page_title="Prévision météorologique Afrique de l'Ouest", 
page_icon="🌦️", 
layout="wide" # ✅ met la page en pleine largeur 
)


# =========================
# Bannière principale
# =========================
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #1E3C72, #2A5298);
        color: white;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        font-size: 26px;
        font-weight: bold;
    ">
    🌦️ Application de prévision météorologique pour l'Afrique de l'Ouest
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# Sidebar : paramètres
# =========================
st.sidebar.title("⚙️ Paramètres de recherche")

model_choice = st.sidebar.selectbox(
    "Choisir un modèle",
    ("PrecipFormer Transformer", "U-NET", "LSTM Attention")
)

period_choice = st.sidebar.radio(
    "Période de prévision",
    ("24h", "3 jours", "7 jours")
)

if st.sidebar.button("Lancer la prévision"):
    st.sidebar.success(f"Prévision lancée avec {model_choice} sur {period_choice}")

# =========================
# Colonnes de métriques
# =========================
c1, c2, c3, c4 = st.columns(4)

for col, label, value in zip(
    [c1, c2, c3, c4],
    ["MSE", "RMSE", "MAE", "R² Score"],
    [2.35, 1.80, 0.95, 0.92]
):
    col.markdown(
        f"""
        <div style="background-color:orange; color:white; padding:10px; border-radius:8px; text-align:center;">
            <h4>{label}</h4>
            <p>{value}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# Section principale
# =========================
st.subheader("📊 Visualisation des résultats")

data = pd.DataFrame({
    "Jour": ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi"],
    "Précipitations prévues (mm)": [12, 5, 20, 0, 8]
})

fig, ax = plt.subplots()
ax.plot(data["Jour"], data["Précipitations prévues (mm)"], marker="o", color="blue")
ax.set_title("Précipitations prévues")
ax.set_ylabel("mm")
st.pyplot(fig)

# =========================
# Carte interactive Folium
# =========================
st.subheader("🗺️ Carte des précipitations prévues")

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

st_folium(m, width=700, height=500)

# =========================
# Expander pour infos
# =========================
with st.expander("ℹ️ Information sur l'outil", expanded=False):
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #FF8C00, #FFA500);
                    color:white; padding:10px; border-radius:5px;">
        Cet outil vise à faciliter le travail des prévisionnistes relatif à la survenue
        des phénomènes extrêmes en Afrique de l'Ouest. Il intègre plusieurs modèles
        (Transformers, U-NET, LSTM) adaptés aux zones climatiques régionales.
        </div>
        """,
        unsafe_allow_html=True
    )
