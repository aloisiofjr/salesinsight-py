<div align="center">

# SalesInsight PY
### Análise e Visualização de Dados de Vendas com Python

**Disciplina:** Desenvolvimento de IA para Análise Preditiva [T5]
**Módulo:** 01 — Semana 07 · **Avaliação:** M1.1 (Mini-Projeto Avaliativo)
**Aluno** *[Aloisio formighieri Júnior]*
**Data:** Outubro de 2026

</div>

---

## Sumário

1. [Contextualização](#1-contextualização)
2. [Objetivos](#2-objetivos)
3. [Descrição do Dataset](#3-descrição-do-dataset)
4. [Metodologia e Arquitetura do Código](#4-metodologia-e-arquitetura-do-código)
5. [Conceitos Aplicados](#5-conceitos-aplicados-módulo-01---semanas-01-a-07)
6. [Como Executar](#6-como-executar)
7. [Estrutura do Projeto](#7-estrutura-do-projeto)
8. [Resultados e Discussão](#8-resultados-e-discussão)
9. [Decisões Técnicas](#9-decisões-técnicas)
10. [Limitações e Trabalhos Futuros](#10-limitações-e-trabalhos-futuros)
11. [Ferramentas e Bibliotecas Utilizadas](#11-ferramentas-e-bibliotecas-utilizadas)
12. [Vídeo de Demonstração](#12-vídeo-de-demonstração)

---

## 1. Contextualização

Este projeto simula a rotina de um(a) Analista de Dados Júnior responsável
por preparar um relatório analítico de vendas para a reunião trimestral de
diretoria de uma empresa de varejo. A partir de um histórico de vendas em
formato CSV, o trabalho consiste em transformar dados brutos — e
propositalmente "sujos" — em métricas, segmentações e visualizações que
respondam a perguntas de negócio concretas: como a receita evolui ao longo
do tempo, quais produtos e regiões performam melhor e quais clientes
concentram o maior volume de gasto.

O projeto consolida, exclusivamente, os conteúdos de lógica de programação,
estruturas de dados, Pandas, manipulação de datas, agregações, funções de
ordem superior e visualização com Matplotlib/Seaborn trabalhados até a
Semana 07 do Módulo 01, não empregando técnicas de estatística inferencial,
tratamento de outliers ou modelos preditivos (conteúdos de semanas
posteriores).

## 2. Objetivos

**Objetivo geral:** desenvolver um pipeline de análise de dados de vendas
em Python, organizado em funções reutilizáveis e em uma classe, capaz de
carregar, tratar, transformar, agregar e visualizar os dados de ponta a
ponta.

**Objetivos específicos:**
- Diagnosticar e tratar inconsistências do dataset bruto (datas inválidas,
  espaços extras, tipos incorretos);
- Derivar atributos temporais e de receita a partir dos dados tratados;
- Calcular métricas agregadas por mês, produto, categoria e região;
- Segmentar a carteira de clientes por nível de gasto;
- Produzir visualizações customizadas que comuniquem os principais
  achados;
- Exportar os resultados em formatos reutilizáveis (CSV, JSON e PNG).

## 3. Descrição do Dataset

O dataset `vendas.csv` é gerado sinteticamente pela função
`gerar_dataset_vendas()` (200 registros, `seed=42`, reprodutível por
qualquer pessoa que execute o script) e contém as seguintes colunas:

| Coluna            | Tipo (bruto) | Descrição                                   |
|-------------------|--------------|----------------------------------------------|
| `id_venda`        | int          | identificador sequencial da venda             |
| `data_venda`      | texto        | data da venda (`AAAA-MM-DD`); ~3% inválidas   |
| `cliente`         | texto        | identificador do cliente, com espaços extras  |
| `produto`         | texto        | nome do produto vendido                       |
| `categoria`       | texto        | categoria do produto                          |
| `regiao`          | texto        | região da venda                               |
| `quantidade`      | int          | unidades vendidas na transação                |
| `preco_unitario`  | float        | preço unitário praticado                      |

A geração sintética foi escolhida em vez de um CSV externo (Opção B do
enunciado) por garantir reprodutibilidade total do experimento e por já
incluir, de forma controlada, as inconsistências necessárias para
demonstrar a etapa de limpeza de dados.

## 4. Metodologia e Arquitetura do Código

O fluxo foi implementado como um pipeline sequencial de funções puras,
orquestrado por uma classe de alto nível:

```
vendas.csv
   │
   ▼
carregar()            ─── RF01 ─── leitura do CSV
inspecionar_dados()   ─── RF02 ─── shape, dtypes, nulos, head
tratar_dados()        ─── RF03 ─── limpeza (texto, datas, tipos)
criar_colunas_derivadas() ─ RF04 ─ receita, mês, trimestre, ano, faixa
calcular_metricas()   ─── RF05 ─── agregações via groupby
segmentar_clientes()  ─── RF06 ─── Bronze / Prata / Ouro (lambda)
gerar_graficos()      ─── RF07 ─── 4 figuras em PNG
exportar_resultados() ─── RF09 ─── CSV + JSON (com releitura)
```

Cada etapa é uma função independente, com parâmetros, retorno e docstring
próprios (RF08), o que facilita testes isolados e a reutilização em outros
contextos. A classe `AnalisadorDeVendas` não reimplementa a lógica: ela
delega para essas funções e mantém o estado intermediário (`df_bruto`,
`df`, `metricas`, `clientes`, `estatisticas`) como atributos de instância,
evitando a repetição de parâmetros entre as etapas. A função
`processar_coluna()` demonstra o conceito de função de ordem superior,
recebendo outra função (neste caso, duas lambdas em contextos distintos)
como argumento. A orquestração final ocorre em `main()`, protegida pelo
bloco `if __name__ == "__main__":` (RF10).

## 5. Conceitos Aplicados (Módulo 01 - Semanas 01 a 07)

- **Lógica de programação:** variáveis, tipos, operadores aritméticos,
  lógicos e relacionais, condicionais (`if/elif/else`) e repetição
  (`for`), usadas na geração do dataset e na classificação condicional;
- **Estruturas de dados:** listas, tuplas, dicionários (`MESES_PT`,
  `precos`, `categorias`) e estruturas compostas (dicionário de
  DataFrames em `metricas`);
- **Funções:** parâmetros, retorno, docstring, funções `lambda` (ao menos
  dois contextos distintos) e função de ordem superior
  (`processar_coluna`);
- **Arquivos:** leitura e escrita de CSV (`pd.read_csv`/`.to_csv`) e de
  JSON (`json.dump`/`json.load`);
- **Módulo `datetime`:** conversão de datas e extração de mês, trimestre
  e ano;
- **Pandas:** DataFrames, filtros booleanos, `groupby`, `.agg()`,
  transformações condicionais vetorizadas (`df.loc[condição, coluna]`);
- **Matplotlib e Seaborn:** gráficos de linha, barra e dispersão,
  customização (título, rótulos, legenda, paleta, `figsize`), *subplots*
  2×2 com `fig.suptitle()` e exportação com `plt.savefig(dpi=150)`;
- **Organização de código:** fluxo estruturado em funções reutilizáveis e
  em uma classe (`AnalisadorDeVendas`), preparando o código para futuras
  etapas de pipeline.

## 6. Como Executar

### Google Colab
1. Faça upload de `salesinsight.py` (e de `vendas.csv`, caso já possua um).
2. Execute: `!python salesinsight.py`

### Localmente, com VS Code ou terminal
1. Instale o Python 3.10 ou superior.
2. Instale as dependências:
   ```bash
   pip install pandas numpy matplotlib seaborn
   ```
3. Execute:
   ```bash
   python salesinsight.py
   ```

Caso o arquivo `vendas.csv` não exista no diretório de execução, o próprio
script gera o dataset sintético (200 registros) automaticamente antes de
iniciar a análise, garantindo que qualquer pessoa consiga reproduzir o
experimento do zero.

## 7. Estrutura do Projeto

```
salesinsight-py/
|-- salesinsight.py           # fluxo principal (RF01-RF10)
|-- vendas.csv                 # dataset (gerado automaticamente)
|-- README.md                  # este documento
|-- outputs/
    |-- metricas_por_mes.csv
    |-- segmentacao_clientes.csv
    |-- estatisticas_gerais.json
    |-- graficos/
        |-- receita_por_mes.png
        |-- top_produtos.png
        |-- quantidade_vs_receita.png
        |-- painel_resumo.png
```

## 8. Resultados e Discussão

Os números a seguir referem-se à execução de referência do dataset
sintético (`seed=42`) e ilustram o tipo de leitura de negócio que o
pipeline viabiliza:

- Da base bruta de 200 registros, **194 permaneceram** após a limpeza —
  **6 foram descartados** por apresentar data de venda inválida, conforme
  reportado pelo relatório de limpeza impresso no console (RF03);
- A **receita total** do período analisado foi de aproximadamente
  **R$ 1.380.210,31**, com **ticket médio de R$ 7.114,49** por transação
  (1.035 unidades vendidas em 194 vendas);
- **Fevereiro** concentrou o maior volume de receita mensal e
  **Setembro**, o menor — um padrão de sazonalidade visível no gráfico de
  linha `receita_por_mes.png`;
- **Notebook** liderou isoladamente o ranking de produtos por receita,
  seguido de Smartphone e Tablet; em termos de categoria, **Computadores**
  superou **Celulares**, apesar de concentrar apenas dois produtos no
  catálogo — indício de que o ticket unitário elevado compensa o menor
  número de itens;
- A região **Centro-Oeste** liderou em receita absoluta, mas a
  **Nordeste** apresentou o **maior ticket médio** por venda, sugerindo
  perfis de consumo distintos entre regiões (visível em
  `por_regiao` dentro de `painel_resumo.png`);
- Na segmentação de clientes, **35 dos 49 clientes ativos** foram
  classificados como **Ouro**, o que reflete o limite de corte adotado
  (R$ 15.000,00) frente ao volume do dataset sintético — em uma base real,
  esses limiares deveriam ser recalibrados conforme o ticket médio da
  operação.

> As figuras completas (`receita_por_mes.png`, `top_produtos.png`,
> `quantidade_vs_receita.png` e `painel_resumo.png`) ficam disponíveis em
> `outputs/graficos/` após a execução do script e devem ser consultadas
> para a leitura visual completa dos achados.

## 9. Decisões Técnicas

**Remoção, em vez de imputação, de datas inválidas.** A coluna
`data_venda` alimenta diretamente as colunas derivadas `mes`, `trimestre`
e `ano`, que por sua vez sustentam toda a análise temporal do projeto
(RF05). Substituir uma data ausente por um valor estimado introduziria
viés nas métricas mensais sem garantia de correspondência com a realidade
— e técnicas de imputação estatística estão, de toda forma, fora do
escopo desta etapa do curso (Semanas 09–11). Optou-se, portanto, por
descartar as linhas com data inválida, registrando a decisão de forma
transparente no relatório de limpeza impresso no console.

**Geração sintética do dataset.** A opção pelo gerador sintético (RF01,
Opção A) em vez de um CSV externo elimina dependências externas e garante
que qualquer avaliador(a) reproduza exatamente os mesmos resultados deste
README ao executar o projeto com a mesma seed.

## 10. Limitações e Trabalhos Futuros

- Os limites de segmentação de clientes (Bronze/Prata/Ouro) são fixos e
  poderiam ser parametrizados ou recalculados por percentil (ver bônus
  B05 do enunciado);
- A análise atual não considera correlação estatística entre variáveis
  nem projeção de tendência — ambos os tópicos são tratados em módulos
  posteriores (Semanas 12 e 13);
- Uma tabela cruzada mês × categoria (via `pivot_table`) enriqueceria a
  leitura de sazonalidade por linha de produto (bônus B03).

## 11. Ferramentas e Bibliotecas Utilizadas

- **Linguagem:** Python 3.10+
- **Ambiente de desenvolvimento:** Google Colab / VS Code
- **Bibliotecas:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `json`,
  `datetime`, `os`, `re`, `random`
- **Versionamento:** Git e GitHub

## 12. Vídeo de Demonstração

---

<div align="center">

Documento elaborado para o curso **Desenvolvimento de IA para Análise
Preditiva** — Módulo 01, Semana 07.

</div>
