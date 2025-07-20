import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

from pipelines.lotomania_ai_pipeline import (
    generate_features,
    predict_top_50,
    evaluate,
    gerar_10_combinacoes
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "lotomania.csv"
MODEL_PATH = BASE_DIR / "models" / "xgb_model.pkl"
RESULTS_DIR = BASE_DIR / "data" / "results"

st.set_page_config(page_title="Lotomania IA - Geração Estratégica", layout="centered")
st.title("🔮 Gerador Inteligente de Apostas - Lotomania IA")

# 📥 Upload ou leitura local
draws = None
uploaded_file = st.file_uploader("📂 Envie um CSV com os sorteios (opcional)", type="csv")
if uploaded_file:
    draws = pd.read_csv(uploaded_file, header=None)
else:
    if not DATA_PATH.exists():
        st.error("❌ Arquivo 'lotomania.csv' não encontrado.")
        st.stop()
    draws = pd.read_csv(DATA_PATH, header=None)

draws = draws.replace("00", "100").astype(int)
draws.columns = list(range(20))
last_draw = draws.iloc[-1].tolist()
last_draw_str = [f"{int(d)%100:02d}" if int(d) != 100 else "00" for d in last_draw]

# 🧠 Geração de features
features_df = generate_features(draws[:-1])
X_input = pd.DataFrame([features_df.values.flatten()])

# 📦 Modelo
if not MODEL_PATH.exists():
    st.error("⚠️ Modelo ainda não foi treinado. Execute `make train`.")
    st.stop()
model = joblib.load(MODEL_PATH)

# 🎛️ Entradas do usuário
quantidade = st.slider("🧮 Quantidade de combinações a gerar", 1, 20, 10)

dezenas_fixas_input = st.text_input("📌 Digite dezenas fixas (até 25) separadas por espaço, ex: 01 12 33", "")
dezenas_fixas = []
if dezenas_fixas_input:
    dezenas_fixas = [d.zfill(2) for d in dezenas_fixas_input.strip().split() if d.isdigit()]
    if len(dezenas_fixas) > 25:
        st.error("⚠️ Insira no máximo 25 dezenas fixas.")
        st.stop()

# 🔮 Geração de combinações com IA
if st.button("🎲 Gerar combinações com IA"):
    try:
        resultados = gerar_10_combinacoes(model, X_input, total=quantidade)

        def limpar_combinacao(row, dezenas_fixas):
            combinacao = [d for d in row if isinstance(d, str) and d.isdigit()]
            combinacao = list(set(combinacao) - set(dezenas_fixas))
            combinacao_final = sorted(dezenas_fixas + combinacao[:50 - len(dezenas_fixas)])
            return combinacao_final if len(combinacao_final) == 50 else None

        resultados = resultados.apply(lambda row: limpar_combinacao(row.tolist(), dezenas_fixas), axis=1)
        resultados = resultados.dropna()
        resultados = pd.DataFrame(resultados.tolist())

        if not resultados.empty:
            st.success(f"✅ {len(resultados)} combinações geradas com sucesso.")
            combinacoes_texto = "\n".join(" ".join(linha) for _, linha in resultados.iterrows())
            st.text_area("📋 Copie todas as combinações", value=combinacoes_texto, height=quantidade * 35)
        else:
            st.warning("⚠️ Nenhuma combinação válida foi gerada.")

    except Exception as e:
        st.exception(f"❌ Erro ao gerar combinações: {e}")

# 🎯 Avaliação do último sorteio
predicted = predict_top_50(model, X_input)
score = evaluate(predicted, last_draw_str)

st.subheader("📊 Último Sorteio Oficial:")
st.write(" ".join(last_draw_str))
st.metric("🎯 Acertos do Modelo no Último Sorteio", f"{score}/20")

# 🔥 Heatmaps
st.markdown("## 📈 Frequência Espacial das Dezenas")
heatmaps = {
    "Frequência Total dos Sorteios": RESULTS_DIR / "heatmap_frequencia.png",
    "Últimos 100 Sorteios": RESULTS_DIR / "heatmap_ultimos_100.png",
    "Últimos 10 Sorteios": RESULTS_DIR / "heatmap_ultimos_10.png",
}

for titulo, path in heatmaps.items():
    if path.exists():
        st.image(str(path), caption=titulo, use_container_width=True)
    else:
        st.warning(f"⚠️ {titulo} ainda não foi gerado.")

# 📅 Últimos 10 sorteios
st.markdown("## 📅 Últimos 10 Sorteios Oficiais")
for i, row in draws.tail(10).iterrows():
    dezenas = [f"{int(d)%100:02d}" if int(d) != 100 else "00" for d in row.tolist()]
    st.write(f"Sorteio {i+1}: {' '.join(dezenas)}")

# 📄 Informações do Modelo
info_path = RESULTS_DIR / "model_info.txt"
if info_path.exists():
    with open(info_path, "r") as f:
        st.markdown("### 📄 Informações do Modelo")
        st.code(f.read())
else:
    st.warning("⚠️ Arquivo 'model_info.txt' não encontrado. Execute `make train` para gerá-lo.")
