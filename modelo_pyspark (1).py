"""
Versão PySpark do projeto (rodar no Databricks ou localmente com pyspark instalado).

No Databricks:
  1. Faça upload do CSV (UCI_Credit_Card.csv) em Catalog > Add data > Upload (ou em um volume).
  2. Cole este código em um notebook e ajuste CAMINHO abaixo.
Localmente: pip install pyspark (requer Java) e python modelo_pyspark.py
"""
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

CAMINHO = "data/UCI_Credit_Card.csv"  # ajuste para o seu caminho
SEED = 42

spark = SparkSession.builder.appName("inadimplencia").getOrCreate()

df = (
    spark.read.csv(CAMINHO, header=True, inferSchema=True)
    .withColumnRenamed("default.payment.next.month", "label")
    .withColumnRenamed("PAY_0", "PAY_1")
    .drop("ID", "SEX")  # SEX fora do modelo (atributo protegido)
)
df = df.withColumn("label", F.col("label").cast("double"))

pay = [f"PAY_{i}" for i in range(1, 7)]
bill = [f"BILL_AMT{i}" for i in range(1, 7)]
amt = [f"PAY_AMT{i}" for i in range(1, 7)]


def soma(cols):
    return sum((F.col(c) for c in cols), F.lit(0))


df = (
    df.withColumn("EDUCATION", F.when(F.col("EDUCATION").isin(0, 5, 6), 4).otherwise(F.col("EDUCATION")))
    .withColumn("MARRIAGE", F.when(F.col("MARRIAGE") == 0, 3).otherwise(F.col("MARRIAGE")))
    .withColumn("ATRASO_MEDIO", soma([F.greatest(F.col(c), F.lit(0)) for c in pay]) / 6)
    .withColumn("ATRASO_MAX", F.greatest(*[F.col(c) for c in pay]))
    .withColumn("MESES_EM_ATRASO", soma([(F.col(c) > 0).cast("int") for c in pay]))
    .withColumn("UTIL_MEDIA", soma(bill) / 6 / F.col("LIMIT_BAL"))
    .withColumn("UTIL_ULTIMA", F.col("BILL_AMT1") / F.col("LIMIT_BAL"))
    .withColumn("RAZAO_PAGAMENTO", soma(amt) / (soma([F.greatest(F.col(c), F.lit(0)) for c in bill]) + 1))
    .withColumn("TENDENCIA_FATURA", (F.col("BILL_AMT1") - F.col("BILL_AMT6")) / F.col("LIMIT_BAL"))
)

features = [c for c in df.columns if c != "label"]
treino, teste = df.randomSplit([0.7, 0.3], seed=SEED)

assembler = VectorAssembler(inputCols=features, outputCol="raw")
scaler = StandardScaler(inputCol="raw", outputCol="features")

modelos = {
    "Regressão logística": LogisticRegression(featuresCol="features", labelCol="label", regParam=0.01, maxIter=100),
    "GBT": GBTClassifier(featuresCol="features", labelCol="label", maxIter=100, maxDepth=4, stepSize=0.1, seed=SEED),
}
avaliador = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

melhor_nome, melhor_auc, melhor_pred = None, -1, None
for nome, clf in modelos.items():
    pred = Pipeline(stages=[assembler, scaler, clf]).fit(treino).transform(teste)
    auc = avaliador.evaluate(pred)
    print(f"{nome}: AUC={auc:.4f}  Gini={2 * auc - 1:.4f}")
    if auc > melhor_auc:
        melhor_nome, melhor_auc, melhor_pred = nome, auc, pred

# KS e tabela por decil: base de teste é pequena, então traz (label, score) para o pandas
pdf = melhor_pred.select("label", vector_to_array("probability")[1].alias("score")).toPandas()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

pdf = pdf.sort_values("score", ascending=False).reset_index(drop=True)
pos, neg = pdf["label"].sum(), (1 - pdf["label"]).sum()
ks = float(np.max(np.abs(pdf["label"].cumsum() / pos - (1 - pdf["label"]).cumsum() / neg)))
print(f"\nMelhor: {melhor_nome} | KS={ks:.4f}")

pdf["decil"] = pd.qcut(pdf.index, 10, labels=range(1, 11))
tabela = pdf.groupby("decil", observed=True).agg(clientes=("label", "size"), inad=("label", "sum"), taxa=("label", "mean"))
tabela["lift"] = tabela["taxa"] / pdf["label"].mean()
print(tabela)
