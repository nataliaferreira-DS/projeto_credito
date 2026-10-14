#!/usr/bin/env python3
"""
Modelo de propensão à inadimplência (crédito / cobrança)
Base: UCI "Default of Credit Card Clients" (também no Kaggle como UCI_Credit_Card.csv)

Uso:
    python modelo_inadimplencia.py --dados data/UCI_Credit_Card.csv
    python modelo_inadimplencia.py --dados "data/default of credit card clients.xls"
    python modelo_inadimplencia.py --synthetic      # só para testar o código, NÃO use como resultado

Saídas (pasta outputs/):
    resultados.md, metricas.json, tabela_decis.csv, tabela_calibracao.csv,
    tabela_cortes.csv, curva_roc.png, calibracao.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "default"
SEED = 42


# --------------------------------------------------------------------------- dados
def carregar_dados(caminho: str) -> pd.DataFrame:
    p = Path(caminho)
    if p.suffix.lower() in (".xls", ".xlsx"):
        df = pd.read_excel(p, header=1)  # o arquivo da UCI tem uma linha de título extra
    else:
        df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(
        columns={
            "default payment next month": TARGET,
            "default.payment.next.month": TARGET,
            "PAY_0": "PAY_1",
        }
    )
    return df.drop(columns=[c for c in ["ID"] if c in df.columns])


def gerar_sintetico(n: int = 30000, seed: int = SEED) -> pd.DataFrame:
    """Dados fictícios com o mesmo esquema da base real. Serve apenas para testar o pipeline."""
    rng = np.random.default_rng(seed)
    risco = rng.normal(0, 1, n)
    limite = rng.lognormal(11.5, 0.8, n).round(-3).clip(10_000, 1_000_000)
    d = {
        "LIMIT_BAL": limite,
        "SEX": rng.integers(1, 3, n),
        "EDUCATION": rng.integers(1, 5, n),
        "MARRIAGE": rng.integers(1, 4, n),
        "AGE": rng.integers(21, 70, n),
    }
    for i in range(1, 7):
        d[f"PAY_{i}"] = np.clip(np.round(0.9 * risco + rng.normal(-0.3, 1, n)), -2, 8).astype(int)
    for i in range(1, 7):
        d[f"BILL_AMT{i}"] = (limite * np.clip(rng.beta(2, 3, n) + 0.1 * risco, 0, 1.2)).round()
    for i in range(1, 7):
        d[f"PAY_AMT{i}"] = (d[f"BILL_AMT{i}"] * np.clip(rng.beta(1, 6, n) - 0.03 * risco, 0, 1)).round()
    p = 1 / (1 + np.exp(-(-1.55 + 0.8 * risco + rng.normal(0, 0.5, n))))
    d[TARGET] = (rng.random(n) < p).astype(int)
    return pd.DataFrame(d)


def criar_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    pay = [f"PAY_{i}" for i in range(1, 7)]
    bill = [f"BILL_AMT{i}" for i in range(1, 7)]
    amt = [f"PAY_AMT{i}" for i in range(1, 7)]

    # códigos fora do dicionário da base (0, 5, 6) agrupados em "outros"
    d["EDUCATION"] = d["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    d["MARRIAGE"] = d["MARRIAGE"].replace({0: 3})

    d["ATRASO_MEDIO"] = d[pay].clip(lower=0).mean(axis=1)
    d["ATRASO_MAX"] = d[pay].max(axis=1)
    d["MESES_EM_ATRASO"] = (d[pay] > 0).sum(axis=1)
    d["UTIL_MEDIA"] = d[bill].mean(axis=1) / d["LIMIT_BAL"]
    d["UTIL_ULTIMA"] = d["BILL_AMT1"] / d["LIMIT_BAL"]
    d["RAZAO_PAGAMENTO"] = d[amt].sum(axis=1) / (d[bill].clip(lower=0).sum(axis=1) + 1)
    d["TENDENCIA_FATURA"] = (d["BILL_AMT1"] - d["BILL_AMT6"]) / d["LIMIT_BAL"]

    # Atributo protegido: não entra no modelo (boa prática em crédito; veja README).
    return d.drop(columns=["SEX"], errors="ignore")


# --------------------------------------------------------------------------- métricas
def ks_stat(y, s) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.max(tpr - fpr))


def psi(referencia, atual, bins: int = 10) -> float:
    """Population Stability Index entre duas distribuições de score."""
    cortes = np.quantile(referencia, np.linspace(0, 1, bins + 1))
    cortes[0], cortes[-1] = -np.inf, np.inf
    e = np.histogram(referencia, cortes)[0] / len(referencia)
    a = np.histogram(atual, cortes)[0] / len(atual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def tabela_decis(y, s) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "s": np.asarray(s)})
    df["decil"] = pd.qcut(df["s"].rank(method="first", ascending=False), 10, labels=range(1, 11))
    t = df.groupby("decil", observed=True).agg(
        clientes=("y", "size"), inadimplentes=("y", "sum"), score_medio=("s", "mean")
    )
    t["taxa_inad"] = t["inadimplentes"] / t["clientes"]
    t["lift"] = t["taxa_inad"] / df["y"].mean()
    t["inad_acumulada"] = t["inadimplentes"].cumsum() / t["inadimplentes"].sum()
    return t.reset_index()


def tabela_calibracao(y, s, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "s": np.asarray(s)})
    df["faixa"] = pd.qcut(df["s"], bins, labels=range(1, bins + 1), duplicates="drop")
    t = df.groupby("faixa", observed=True).agg(prob_media=("s", "mean"), taxa_observada=("y", "mean"), n=("y", "size"))
    return t.reset_index()


def tabela_cortes(y, s) -> pd.DataFrame:
    """Trade-off de política: aprovar os X% de menor risco e ver a inadimplência dos aprovados."""
    df = pd.DataFrame({"y": np.asarray(y), "s": np.asarray(s)}).sort_values("s")
    base = df["y"].mean()
    linhas = []
    for pct in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        aprov = df.iloc[: int(len(df) * pct)]
        linhas.append(
            {
                "taxa_aprovacao": pct,
                "inad_entre_aprovados": aprov["y"].mean(),
                "inad_sem_modelo": base,
                "inadimplentes_evitados_%": 1 - aprov["y"].sum() / df["y"].sum() if pct < 1 else 0.0,
            }
        )
    return pd.DataFrame(linhas)


def avaliar(y, s, s_treino=None) -> dict:
    auc = roc_auc_score(y, s)
    out = {
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_stat(y, s),
        "brier": float(brier_score_loss(y, s)),
    }
    if s_treino is not None:
        out["psi_treino_vs_teste"] = psi(s_treino, s)
    return out


# --------------------------------------------------------------------------- execução
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", help="caminho do CSV/XLS da base UCI")
    ap.add_argument("--synthetic", action="store_true", help="usa dados fictícios (apenas teste do código)")
    ap.add_argument("--saida", default="outputs")
    args = ap.parse_args()

    if args.synthetic:
        df = gerar_sintetico()
        print("ATENÇÃO: usando dados sintéticos. Os números NÃO representam a base real.")
    elif args.dados:
        df = carregar_dados(args.dados)
    else:
        ap.error("informe --dados <arquivo> ou --synthetic")

    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)

    df = criar_features(df)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)

    modelos = {
        "Regressão logística": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)),
        "Gradient boosting": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15, random_state=SEED,
        ),
    }

    metricas, scores = {}, {}
    for nome, m in modelos.items():
        m.fit(X_tr, y_tr)
        s_tr, s_te = m.predict_proba(X_tr)[:, 1], m.predict_proba(X_te)[:, 1]
        metricas[nome] = avaliar(y_te, s_te, s_tr)
        metricas[nome]["auc_treino"] = float(roc_auc_score(y_tr, s_tr))
        scores[nome] = s_te

    melhor = max(metricas, key=lambda k: metricas[k]["auc"])
    s_melhor = scores[melhor]

    tdec = tabela_decis(y_te, s_melhor)
    tcal = tabela_calibracao(y_te, s_melhor)
    tcor = tabela_cortes(y_te, s_melhor)

    tdec.to_csv(saida / "tabela_decis.csv", index=False)
    tcal.to_csv(saida / "tabela_calibracao.csv", index=False)
    tcor.to_csv(saida / "tabela_cortes.csv", index=False)
    (saida / "metricas.json").write_text(
        json.dumps({"melhor_modelo": melhor, "taxa_inadimplencia": float(y.mean()), "modelos": metricas}, indent=2, ensure_ascii=False)
    )

    # gráficos (opcionais)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(5, 5))
        for nome, s in scores.items():
            fpr, tpr, _ = roc_curve(y_te, s)
            plt.plot(fpr, tpr, label=f"{nome} (AUC {metricas[nome]['auc']:.3f})")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("Falso positivo"); plt.ylabel("Verdadeiro positivo"); plt.legend(); plt.title("Curva ROC (teste)")
        plt.tight_layout(); plt.savefig(saida / "curva_roc.png", dpi=130); plt.close()

        plt.figure(figsize=(5, 5))
        plt.plot(tcal["prob_media"], tcal["taxa_observada"], "o-", label=melhor)
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("Probabilidade prevista"); plt.ylabel("Taxa observada"); plt.title("Calibração (teste)")
        plt.legend(); plt.tight_layout(); plt.savefig(saida / "calibracao.png", dpi=130); plt.close()
    except ImportError:
        print("matplotlib não instalado: gráficos não gerados.")

    # relatório em markdown
    def md(df_):
        return df_.round(4).to_markdown(index=False) if hasattr(df_, "to_markdown") and _tem_tabulate() else df_.round(4).to_string(index=False)

    linhas = ["# Resultados", ""]
    if args.synthetic:
        linhas += ["> **Dados sintéticos: execução de teste. Não use estes números.**", ""]
    linhas += [f"Taxa de inadimplência na base: {y.mean():.2%}  |  Melhor modelo (AUC no teste): **{melhor}**", "",
               "## Métricas (conjunto de teste)", "", md(pd.DataFrame(metricas).T.reset_index().rename(columns={"index": "modelo"})), "",
               "## Tabela por decil (1 = maior risco)", "", md(tdec), "",
               "## Calibração", "", md(tcal), "",
               "## Política de corte (aprovar os de menor risco)", "", md(tcor), ""]
    (saida / "resultados.md").write_text("\n".join(linhas), encoding="utf-8")

    print(json.dumps(metricas, indent=2, ensure_ascii=False))
    print(f"\nMelhor modelo: {melhor}. Arquivos salvos em {saida}/")


def _tem_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
