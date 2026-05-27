
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import glob
import os
import warnings
warnings.filterwarnings("ignore")


plt.rcParams["figure.facecolor"] = "#0f0f11"
plt.rcParams["axes.facecolor"] = "#18181c"
plt.rcParams["axes.edgecolor"] = "#2a2a2e"
plt.rcParams["text.color"] = "#e8e8f0"
plt.rcParams["xtick.color"] = "#808088"
plt.rcParams["ytick.color"] = "#808088"
plt.rcParams["grid.color"] = "#2a2a2e"
plt.rcParams["figure.dpi"] = 120


COR_AZUL     = "#378ADD"
COR_VERDE    = "#1D9E75"
COR_LARANJA  = "#D85A30"
COR_AMARELO  = "#BA7517"
COR_ROXO     = "#7F77DD"
COR_CINZA    = "#888780"
COR_VERMELHO = "#E24B4A"


PASTA = "C:/Users/Álefe"

print("carregando arquivos BO...")

arquivos = glob.glob(f"{PASTA}/BO_*.csv")
print(f"encontrei {len(arquivos)} arquivos")

lista_dfs = []
for arq in sorted(arquivos):
    nome = os.path.basename(arq)
    print(f"  lendo {nome}...")
    try:
        # le so as colunas que precisamos pra economizar memoria
        df_temp = pd.read_csv(arq, encoding="latin-1", low_memory=False,
                              usecols=lambda c: c.strip() in [
                                  "ANO_BO", "ANO", "MES", "RUBRICA",
                                  "CIDADE", "NOME_DEPARTAMENTO",
                                  "LATITUDE", "LONGITUDE", "FLAG_STATUS"
                              ])
        lista_dfs.append(df_temp)
    except Exception as e:
        print(f"  erro em {nome}: {e}")

print("juntando todos os arquivos...")
df = pd.concat(lista_dfs, ignore_index=True)


df.columns = [c.strip().replace("ï»¿", "") for c in df.columns]

print(f"total de registros: {len(df):,}")
print(f"colunas: {list(df.columns)}")
print()

print("limpando os dados...")

if "ANO" in df.columns:
    df["ano"] = pd.to_numeric(df["ANO"], errors="coerce")
elif "ANO_BO" in df.columns:
    df["ano"] = pd.to_numeric(df["ANO_BO"], errors="coerce")

if "MES" in df.columns:
    df["mes"] = pd.to_numeric(df["MES"], errors="coerce")

if "RUBRICA" in df.columns:
    df["crime"] = df["RUBRICA"].astype(str).str.strip().str.upper()


if "CIDADE" in df.columns:
    df["cidade"] = df["CIDADE"].astype(str).str.strip().str.upper()


if "NOME_DEPARTAMENTO" in df.columns:
    df["departamento"] = df["NOME_DEPARTAMENTO"].astype(str).str.strip()


df = df.dropna(subset=["ano"])
df = df[df["ano"] > 2000] 


print("contando ocorrencias...")
por_crime_ano = df.groupby(["ano", "crime"]).size().reset_index(name="qtd")


top_crimes = df["crime"].value_counts().head(10).index.tolist()
df_top = df[df["crime"].isin(top_crimes)].copy()

print(f"top 10 crimes identificados")
print(f"anos disponiveis: {sorted(df['ano'].dropna().unique().astype(int).tolist())}")
print()


print("fazendo grafico 1 - distribuicao...")

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("Distribuição dos Boletins de Ocorrência — SSP-SP (2007–2016)",
             fontsize=14, fontweight="bold", color="#e8e8f0", y=0.98)
fig.patch.set_facecolor("#0f0f11")

ax = axes[0, 0]
por_ano = df.groupby("ano").size().sort_index()
bars = ax.bar(por_ano.index.astype(int), por_ano.values,
              color=COR_AZUL, edgecolor="none", width=0.7)
for bar, val in zip(bars, por_ano.values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + por_ano.max() * 0.01,
            f"{val/1000:.0f}k", ha="center", fontsize=7, color=COR_CINZA)
ax.set_title("Total de ocorrencias por ano", fontsize=10, color="#e8e8f0")
ax.yaxis.grid(True)
ax.set_ylabel("ocorrencias", fontsize=8)
ax.tick_params(labelsize=8)

ax = axes[0, 1]
top10 = df["crime"].value_counts().head(10).sort_values(ascending=True)
cores_bar = [COR_AZUL, COR_VERDE, COR_LARANJA, COR_AMARELO, COR_ROXO,
             COR_VERMELHO, COR_CINZA, COR_AZUL, COR_VERDE, COR_LARANJA]
barras = ax.barh(range(len(top10)), top10.values,
                 color=cores_bar[:len(top10)], height=0.6, edgecolor="none")
ax.set_yticks(range(len(top10)))
ax.set_yticklabels([c[:30] for c in top10.index], fontsize=7)
ax.set_title("Top 10 tipos de crime", fontsize=10, color="#e8e8f0")
ax.xaxis.grid(True)
for barra, val in zip(barras, top10.values):
    ax.text(val + top10.max() * 0.01,
            barra.get_y() + barra.get_height() / 2,
            f"{val/1000:.0f}k", va="center", fontsize=7, color=COR_CINZA)
ax.set_xlim(right=top10.max() * 1.18)

ax = axes[1, 0]
if "mes" in df.columns:
    por_mes = df.groupby("mes").size()
    nomes_mes = ["Jan","Fev","Mar","Abr","Mai","Jun",
                 "Jul","Ago","Set","Out","Nov","Dez"]
    ax.bar(range(1, 13), [por_mes.get(m, 0) for m in range(1, 13)],
           color=COR_VERDE, edgecolor="none", width=0.7)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(nomes_mes, fontsize=8)
    ax.set_title("Sazonalidade — ocorrencias por mes", fontsize=10, color="#e8e8f0")
    ax.yaxis.grid(True)
    ax.set_ylabel("ocorrencias", fontsize=8)

ax = axes[1, 1]
if "cidade" in df.columns:
    top_cidades = df["cidade"].value_counts().head(10).sort_values(ascending=True)
    ax.barh(range(len(top_cidades)), top_cidades.values,
            color=COR_LARANJA, height=0.6, edgecolor="none")
    ax.set_yticks(range(len(top_cidades)))
    ax.set_yticklabels([c[:25] for c in top_cidades.index], fontsize=8)
    ax.set_title("Top 10 cidades com mais ocorrencias", fontsize=10, color="#e8e8f0")
    ax.xaxis.grid(True)

plt.tight_layout(pad=2.0)
plt.savefig("grafico1_distribuicao_bo.png", bbox_inches="tight", facecolor="#0f0f11")
plt.show()
print("grafico 1 salvo!")


print("fazendo grafico 2 - outliers...")

if "departamento" in df.columns:
    por_depto = df.groupby(["departamento", "ano"]).size().reset_index(name="qtd")
else:
    por_depto = df.groupby(["cidade", "ano"]).size().reset_index(name="qtd")
    por_depto = por_depto.rename(columns={"cidade": "departamento"})

q1 = por_depto["qtd"].quantile(0.25)
q3 = por_depto["qtd"].quantile(0.75)
iqr = q3 - q1
por_depto["outlier_iqr"] = ((por_depto["qtd"] < q1 - 1.5 * iqr) |
                             (por_depto["qtd"] > q3 + 1.5 * iqr)).astype(int)

z = np.abs(stats.zscore(por_depto["qtd"]))
por_depto["outlier_zscore"] = (z > 3).astype(int)

scaler = StandardScaler()
X = scaler.fit_transform(por_depto[["qtd"]])
iso = IsolationForest(contamination=0.02, random_state=42)
por_depto["outlier_isofor"] = (iso.fit_predict(X) == -1).astype(int)
por_depto["outlier_total"] = (por_depto["outlier_iqr"] +
                               por_depto["outlier_zscore"] +
                               por_depto["outlier_isofor"])

total = len(por_depto)
print(f"IQR: {por_depto['outlier_iqr'].sum()} outliers")
print(f"Zscore: {por_depto['outlier_zscore'].sum()} outliers")
print(f"Isolation Forest: {por_depto['outlier_isofor'].sum()} outliers")

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("Detecção de Outliers — Boletins de Ocorrência SP",
             fontsize=14, fontweight="bold", color="#e8e8f0", y=0.98)
fig.patch.set_facecolor("#0f0f11")

ax = axes[0, 0]
normais   = por_depto[por_depto["outlier_total"] == 0]["qtd"]
suspeitos = por_depto[por_depto["outlier_total"] == 1]["qtd"]
graves    = por_depto[por_depto["outlier_total"] >= 2]["qtd"]
ax.boxplot(normais, vert=False, patch_artist=True, widths=0.5,
           boxprops=dict(facecolor=COR_AZUL + "44", edgecolor=COR_AZUL, linewidth=0.8),
           medianprops=dict(color="white", linewidth=2),
           whiskerprops=dict(color=COR_AZUL, linewidth=0.8),
           capprops=dict(color=COR_AZUL, linewidth=0.8),
           showfliers=False)
if len(suspeitos):
    jitter = np.random.uniform(-0.15, 0.15, len(suspeitos))
    ax.scatter(suspeitos, np.ones(len(suspeitos)) + jitter,
               color=COR_AMARELO, s=15, alpha=0.6, zorder=5, label="suspeito")
if len(graves):
    jitter = np.random.uniform(-0.15, 0.15, len(graves))
    ax.scatter(graves, np.ones(len(graves)) + jitter,
               color=COR_VERMELHO, s=25, alpha=0.85, zorder=6, label="outlier grave")
ax.set_title("Box plot com outliers", fontsize=10, color="#e8e8f0")
ax.set_yticks([])
ax.xaxis.grid(True)
ax.legend(fontsize=8)

ax = axes[0, 1]
nomes = ["IQR", "Z-score", "Isolation\nForest"]
qtds  = [por_depto["outlier_iqr"].sum(),
         por_depto["outlier_zscore"].sum(),
         por_depto["outlier_isofor"].sum()]
barras = ax.barh(nomes, qtds,
                 color=[COR_AZUL, COR_LARANJA, COR_ROXO],
                 height=0.5, edgecolor="none")
for barra, val in zip(barras, qtds):
    ax.text(val + max(qtds) * 0.02,
            barra.get_y() + barra.get_height() / 2,
            f"{val}", va="center", fontsize=9, color=COR_CINZA)
ax.set_title("Outliers por metodo", fontsize=10, color="#e8e8f0")
ax.xaxis.set_visible(False)
ax.set_xlim(right=max(qtds) * 1.2)

ax = axes[1, 0]
out_depto = (por_depto[por_depto["outlier_total"] >= 2]
             .groupby("departamento").size()
             .sort_values(ascending=True)
             .tail(15))
cores_depto = [COR_VERMELHO if v > out_depto.quantile(0.75)
               else COR_AMARELO for v in out_depto.values]
ax.barh(out_depto.index, out_depto.values,
        color=cores_depto, height=0.6, edgecolor="none")
ax.set_title("Outliers graves por departamento", fontsize=10, color="#e8e8f0")
ax.tick_params(labelsize=7)
ax.xaxis.grid(True)

ax = axes[1, 1]
normais_df   = por_depto[por_depto["outlier_total"] == 0]
graves_df    = por_depto[por_depto["outlier_total"] >= 2]
ax.scatter(normais_df["ano"], normais_df["qtd"],
           color=COR_AZUL, s=8, alpha=0.3, label="normal")
ax.scatter(graves_df["ano"], graves_df["qtd"],
           color=COR_VERMELHO, s=30, alpha=0.8, label="outlier grave", zorder=5)
ax.set_title("Ocorrencias por ano com outliers", fontsize=10, color="#e8e8f0")
ax.legend(fontsize=8)
ax.yaxis.grid(True)
ax.set_ylabel("ocorrencias", fontsize=8)
ax.tick_params(labelsize=8)

plt.tight_layout(pad=2.0)
plt.savefig("grafico2_outliers_bo.png", bbox_inches="tight", facecolor="#0f0f11")
plt.show()
print("grafico 2 salvo!")

print("fazendo grafico 3 - tendencias...")

fig = plt.figure(figsize=(16, 11))
fig.suptitle("Tendências e Correlações — Boletins de Ocorrência SP",
             fontsize=14, fontweight="bold", color="#e8e8f0", y=0.98)
fig.patch.set_facecolor("#0f0f11")

ax1 = fig.add_subplot(2, 1, 1)
ax1.set_facecolor("#18181c")
if "mes" in df.columns:
    serie = df.groupby(["ano", "mes"]).size().reset_index(name="qtd")
    serie = serie.sort_values(["ano", "mes"]).reset_index(drop=True)
    x = range(len(serie))
    ax1.fill_between(x, serie["qtd"], alpha=0.2, color=COR_AZUL)
    ax1.plot(x, serie["qtd"], color=COR_AZUL, lw=1.5, label="ocorrencias")
    z = np.polyfit(list(x), serie["qtd"], 1)
    p = np.poly1d(z)
    ax1.plot(x, p(list(x)), color=COR_LARANJA, lw=2, ls="--", label="tendencia")
    passo = 12
    ax1.set_xticks(list(x)[::passo])
    ax1.set_xticklabels(
        [f"{row.ano:.0f}" for _, row in serie.iloc[::passo].iterrows()],
        fontsize=8)
else:
    serie = df.groupby("ano").size()
    ax1.bar(serie.index.astype(int), serie.values, color=COR_AZUL, edgecolor="none")
ax1.set_title("Serie temporal de ocorrencias mensais — SSP-SP", fontsize=10, color="#e8e8f0")
ax1.yaxis.grid(True)
ax1.legend(fontsize=8)
ax1.set_ylabel("ocorrencias", fontsize=8)

ax2 = fig.add_subplot(2, 2, 3)
ax2.set_facecolor("#18181c")
top5 = df["crime"].value_counts().head(5).index.tolist()
cores_crimes = [COR_AZUL, COR_LARANJA, COR_VERDE, COR_AMARELO, COR_ROXO]
for i, crime in enumerate(top5):
    dados_crime = (df[df["crime"] == crime]
                   .groupby("ano").size()
                   .reset_index(name="qtd")
                   .sort_values("ano"))
    ax2.plot(dados_crime["ano"], dados_crime["qtd"],
             marker="o", markersize=4,
             color=cores_crimes[i], lw=2,
             label=crime[:25])
ax2.set_title("Evolucao dos top 5 crimes por ano", fontsize=10, color="#e8e8f0")
ax2.legend(fontsize=6, loc="upper left")
ax2.yaxis.grid(True)
ax2.tick_params(labelsize=8)
ax2.set_ylabel("ocorrencias", fontsize=8)

ax3 = fig.add_subplot(2, 2, 4)
ax3.set_facecolor("#18181c")
top8 = df["crime"].value_counts().head(8).index.tolist()
df_heatmap = df[df["crime"].isin(top8)]
tabela = df_heatmap.groupby(["ano", "crime"]).size().unstack(fill_value=0)
tabela_norm = tabela.div(tabela.sum(axis=1), axis=0)
sns.heatmap(tabela_norm, cmap="YlOrRd", ax=ax3,
            linewidths=0.3, linecolor="#2a2a2e",
            cbar_kws={"shrink": 0.7},
            annot=False)
ax3.set_title("Proporcao de crimes por ano", fontsize=10, color="#e8e8f0")
ax3.tick_params(labelsize=7, rotation=35)
ax3.set_xlabel("")
ax3.set_ylabel("")

plt.tight_layout(pad=2.5)
plt.savefig("grafico3_tendencias_bo.png", bbox_inches="tight", facecolor="#0f0f11")
plt.show()
print("grafico 3 salvo!")

print()
print("pronto! os 3 graficos foram salvos na pasta Downloads")
print(f"total de registros analisados: {len(df):,}")
