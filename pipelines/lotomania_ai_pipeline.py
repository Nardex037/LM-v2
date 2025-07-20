# pipelines/lotomania_ai_pipeline.py
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from joblib import dump, load
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def generate_features(draws_df: pd.DataFrame) -> pd.DataFrame:
    total_sorteios = draws_df.shape[0]
    freq = draws_df.apply(pd.Series.value_counts).fillna(0).sum(axis=1)
    freq = freq.reindex(range(1, 101), fill_value=0)

    last_seen = draws_df[::-1].stack().reset_index().drop_duplicates(subset=0).set_index(0)['level_0']
    last_seen = last_seen.reindex(range(1, 101), fill_value=total_sorteios)

    positions = {i: ((i - 1) // 10, (i - 1) % 10) for i in range(1, 101)}
    pos_x = pd.Series({k: v[0] for k, v in positions.items()})
    pos_y = pd.Series({k: v[1] for k, v in positions.items()})

    primos = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}
    fibonacci = {1, 2, 3, 5, 8, 13, 21, 34, 55, 89}

    is_par = pd.Series({i: int(i % 2 == 0) for i in range(1, 101)})
    is_primo = pd.Series({i: int(i in primos) for i in range(1, 101)})
    is_fibo = pd.Series({i: int(i in fibonacci) for i in range(1, 101)})

    freq_pct = freq / total_sorteios

    freq_10 = draws_df.tail(10).stack().value_counts().reindex(range(1, 101), fill_value=0)
    freq_100 = draws_df.tail(100).stack().value_counts().reindex(range(1, 101), fill_value=0)

    linha = pos_x
    coluna = pos_y

    linha_freq = freq.groupby(linha).transform('sum')
    coluna_freq = freq.groupby(coluna).transform('sum')

    freq_z = (freq - freq.mean()) / freq.std()
    atraso_z = (last_seen - last_seen.mean()) / last_seen.std()

    df_feat = pd.DataFrame({
        'dezena': range(1, 101),
        'freq_total': freq,
        'freq_pct': freq_pct,
        'freq_ult_10': freq_10,
        'freq_ult_100': freq_100,
        'atraso': last_seen,
        'freq_zscore': freq_z,
        'atraso_zscore': atraso_z,
        'linha': linha,
        'coluna': coluna,
        'linha_freq': linha_freq,
        'coluna_freq': coluna_freq,
        'pos_x': pos_x,
        'pos_y': pos_y,
        'is_par': is_par,
        'is_primo': is_primo,
        'is_fibonacci': is_fibo,
    }).set_index('dezena')

    return df_feat

def plot_heatmap(df_feat, path: Path, titulo="Frequência das Dezenas na Cartela 10x10"):
    heatmap_matrix = np.zeros((10, 10), dtype=int)
    for dezena, row in df_feat.iterrows():
        x, y = int(row['pos_x']), int(row['pos_y'])
        heatmap_matrix[x, y] = row['freq_total']

    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_matrix, annot=True, fmt="d", cmap="YlGnBu")
    plt.title(titulo)
    plt.xlabel("Coluna")
    plt.ylabel("Linha")
    plt.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()

def train_model(X: pd.DataFrame, y: pd.DataFrame, model_path: Path) -> XGBClassifier:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    dump(model, model_path)
    return model

def predict_top_50(model, X):
    X = X.to_numpy().astype(np.float32) if isinstance(X, pd.DataFrame) else X.astype(np.float32)
    probs = model.predict_proba(X)[0]
    top_indices = probs.argsort()[-50:][::-1]
    dezenas = [f"{i:02d}" if i < 100 else "00" for i in sorted(top_indices)]
    return dezenas

def evaluate(predicted: list, true_draw: list) -> int:
    return len(set(predicted) & set(true_draw))

def gerar_10_combinacoes(model, X_input, salvar_em: Path = None, top_k=70, total=10):
    X_input = X_input.to_numpy().astype(np.float32)
    combinacoes = []
    probs = model.predict_proba(X_input)[0]

    for _ in range(total):
        top_indices = probs.argsort()[-top_k:][::-1]
        top_probs = probs[top_indices]
        top_probs /= top_probs.sum()
        dezenas_amostradas = np.random.choice(top_indices, size=50, replace=False, p=top_probs)
        dezenas_ordenadas = sorted([f"{i:02d}" if i < 100 else "00" for i in dezenas_amostradas])
        combinacoes.append(dezenas_ordenadas)

    df_combs = pd.DataFrame(combinacoes)
    if salvar_em:
        salvar_em.parent.mkdir(parents=True, exist_ok=True)
        df_combs.to_csv(salvar_em, index=False)
    return df_combs

def main_pipeline(
    data_path: Path = Path("data/raw/lotomania.csv"),
    model_path: Path = Path("models/xgb_model.pkl"),
    results_dir: Path = Path("data/results"),
    retrain: bool = True,
    gerar_heatmaps: bool = True
):
    draws = pd.read_csv(data_path, header=None).replace("00", "100").astype(int)
    draws.columns = list(range(20))

    X_raw = []
    y = []
    for i in range(len(draws) - 1):
        partial_draws = draws.iloc[:i+1]
        features = generate_features(partial_draws)
        X_raw.append(features.values.flatten())

        next_draw = draws.iloc[i + 1].tolist()
        target = [1 if d in next_draw else 0 for d in range(1, 101)]
        y.append(target)

    X_raw = pd.DataFrame(X_raw)
    y = pd.DataFrame(y)

    if gerar_heatmaps:
        plot_heatmap(generate_features(draws), results_dir / "heatmap_frequencia.png", "Frequência Total dos Sorteios")
        plot_heatmap(generate_features(draws.tail(100)), results_dir / "heatmap_ultimos_100.png", "Frequência nos últimos 100 sorteios")
        plot_heatmap(generate_features(draws.tail(10)), results_dir / "heatmap_ultimos_10.png", "Frequência nos últimos 10 sorteios")

    if retrain or not model_path.exists():
        X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)
        model = train_model(X_train, y_train, model_path)
    else:
        model = load(model_path)

    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "model_info.txt", "w") as f:
        f.write(f"Total de sorteios usados: {len(draws)}\n")
        f.write(f"Amostras de treino: {X_raw.shape[0]}\n")
        f.write(f"Features por amostra: {X_raw.shape[1]}\n")
        f.write(f"Shape final do modelo: {model.n_features_in_} features\n")

    last_state = generate_features(draws).values.flatten().reshape(1, -1)
    pred = predict_top_50(model, pd.DataFrame(last_state))
    true_draw = draws.iloc[-1].apply(lambda x: f"{x % 100:02d}" if x != 100 else "00").tolist()
    score = evaluate(pred, true_draw)
    print("Top 50 dezenas preditas:", pred)
    print(f"🎯 Acertos no último sorteio: {score}/20")
