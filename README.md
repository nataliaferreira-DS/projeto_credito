# Modelo de propensão à inadimplência (crédito e cobrança)

Projeto de portfólio que simula o trabalho de um(a) cientista de dados em uma área de crédito: construir um modelo preditivo, avaliá-lo com as métricas usadas no setor financeiro e traduzir o resultado em uma decisão de negócio.

## Problema

Dado o histórico de 6 meses de um cliente de cartão de crédito (atrasos, faturas, pagamentos, limite), estimar a probabilidade de ele ficar inadimplente no mês seguinte. Esse score pode apoiar decisões de concessão, limite e priorização de cobrança.

## Dados

Base pública **Default of Credit Card Clients** (UCI), 30.000 clientes.
- UCI: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- Kaggle (CSV `UCI_Credit_Card.csv`): busque por "UCI Credit Card"

Coloque o arquivo em `data/`. O script aceita o `.xls` da UCI ou o `.csv` do Kaggle.

## Como rodar

```bash
pip install -r requirements.txt
python modelo_inadimplencia.py --dados "data/default of credit card clients.xls"```

Os resultados ficam em `outputs/` (métricas, tabela por decil, calibração, tabela de cortes e gráficos).

## Método

- **Features:** atraso médio e máximo, meses em atraso, utilização do limite, razão pagamento/fatura e tendência da fatura.
- **Modelos:** regressão logística (referência padrão em crédito, interpretável) e gradient boosting.
- **Variável `SEX` removida** do modelo, por ser atributo protegido. A decisão de crédito não deve se basear nela, mesmo com um pequeno custo de desempenho.
- **Validação:** divisão estratificada 70/30.

## Métricas

| Métrica | Para que serve |
|---|---|
| AUC / Gini | poder de ordenação do score |
| KS | separação entre bons e maus pagadores |
| PSI | estabilidade da distribuição do score |
| Brier + tabela de calibração | se a probabilidade prevista é confiável |
| Tabela por decil e lift | quanto o modelo concentra inadimplentes nos piores decis |
| Tabela de cortes | trade-off entre taxa de aprovação e inadimplência dos aprovados |

## Versão PySpark / Databricks

`modelo_pyspark.py` repete a preparação de dados e o treino com PySpark. Para rodar no Databricks, faça upload do CSV, cole o código em um notebook e ajuste a variável `CAMINHO`.


## Resultados e conclusão de negócio

**Desempenho (conjunto de teste, 9.000 clientes, 22,1% de inadimplência).** O gradient boosting foi o melhor modelo (AUC 0,781, Gini 0,561, KS 0,421), à frente da regressão logística (AUC 0,750, KS 0,397). A diferença entre treino e teste do boosting (AUC 0,82 contra 0,78) indica sobreajuste leve. Mesmo assim, ele supera a regressão logística, que é mais simples de explicar.

**Capacidade de ordenar risco.** O 1º decil (10% de maior risco) tem 69,8% de inadimplência, 3,2 vezes a média da base, e concentra 31,5% de todos os inadimplentes. Os 2 piores decis juntos concentram 50,7% dos inadimplentes, e o decil de menor risco tem apenas 3,8%.

**Calibração.** As probabilidades previstas ficam próximas das taxas observadas em quase todas as faixas, por exemplo 70,3% previsto contra 69,8% observado na faixa mais arriscada e 42,2% contra 42,4% na seguinte. Os desvios nas faixas intermediárias (4 a 7 de 10) são pequenos e compatíveis com ruído amostral. O score pode ser lido como uma probabilidade e não apenas como um ranking.

**Política de corte.** Aprovar apenas os de menor risco reduz a inadimplência da carteira aprovada, ao custo de rejeitar bons pagadores:

| Taxa de aprovação | Inadimplência dos aprovados | Inadimplentes evitados | Bons clientes rejeitados |
|---|---|---|---|
| 90% | 16,8% | 31,5% | 3,9% |
| 80% | 13,6% | 50,7% | 11,3% |
| 70% | 11,9% | 62,2% | 20,8% |
| 60% | 11,0% | 70,3% | 31,4% |
| 50% | 9,2% | 79,1% | 41,7% |

A recomendação é um corte de **80% de aprovação**: a inadimplência dos aprovados cai de 22,1% para 13,6%, evitando metade dos inadimplentes e rejeitando cerca de 11% dos bons clientes. A partir daí, cada ponto adicional de redução de risco custa cada vez mais clientes bons. A decisão final depende de dois números que esta base não traz: a margem do produto e a perda esperada por inadimplente, que definiriam o ponto em que rejeitar deixa de compensar.

**Uso em cobrança.** O mesmo score pode priorizar a régua de cobrança: os 20% de maior risco concentram metade dos inadimplentes, e é nesse grupo que faz sentido concentrar o contato mais caro ou mais precoce.

## Limitações

- A divisão é aleatória. Em produção, a validação correta é **out-of-time** (treinar no passado, testar no futuro).
- A base é de um único banco de Taiwan, de 2005, então não representa o perfil de não correntistas no Brasil.
- Não há custo de perda ou receita por cliente, então a análise de corte é simplificada.
