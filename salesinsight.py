"""
SalesInsight PY - Analise e Visualizacao de Dados de Vendas com Python
========================================================================

Mini-Projeto Avaliativo - Modulo 01, Semana 07
Disciplina: Desenvolvimento de IA para Analise Preditiva [T5]
Aluno(a)/Squad: Aloisio Formighieri Junior
Data: outubro de 2026

Objetivo
--------
Implementar um fluxo de analise de dados de vendas que carregue, trate,
transforme, agregue e visualize um dataset de vendas, aplicando os conceitos
de logica de programacao, estruturas de dados, Pandas, funcoes de ordem
superior e visualizacao com Matplotlib/Seaborn estudados ate a Semana 07 do
Modulo 01 (RF01 a RF10 do enunciado do mini-projeto).

Fluxo geral
-----------
    vendas.csv
        -> carregar()           (RF01)
        -> inspecionar_dados()  (RF02)
        -> tratar_dados()       (RF03)
        -> criar_colunas_derivadas() (RF04)
        -> calcular_metricas()  (RF05)
        -> segmentar_clientes() (RF06)
        -> gerar_graficos()     (RF07)
        -> exportar_resultados()(RF09)

A orquestracao de ponta a ponta e feita pela classe AnalisadorDeVendas e
pela funcao main() (RF10).
"""

import json
import os
import random
import re
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PASTA_OUTPUTS = "outputs"
PASTA_GRAFICOS = os.path.join(PASTA_OUTPUTS, "graficos")

# Mapeamento auxiliar: dt.strftime("%B") retornaria o nome do mes em ingles
# no ambiente padrao do Google Colab, por isso o nome em portugues e obtido
# por meio deste dicionario (ver RF04).
MESES_PT = {
    1: "Janeiro", 2: "Fevereiro", 3: "Marco", 4: "Abril",
    5: "Maio", 6: "Junho", 7: "Julho", 8: "Agosto",
    9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro",
}


# ---------------------------------------------------------------------------
# RF01 - Criar ou Carregar o Dataset de Vendas
# ---------------------------------------------------------------------------
def gerar_dataset_vendas(n_registros=200, seed=42):
    """Gera sinteticamente o dataset bruto de vendas utilizado no estudo.

    O gerador introduz, propositalmente, duas inconsistencias tipicas de
    bases de dados reais - datas invalidas e espacos extras no nome do
    cliente - que servem de materia-prima para a etapa de limpeza (RF03).

    Args:
        n_registros (int): quantidade de registros de venda a gerar.
        seed (int): semente do gerador aleatorio, garante reprodutibilidade.

    Returns:
        pandas.DataFrame: dataset bruto, ainda sem tratamento, com as
        colunas id_venda, data_venda, cliente, produto, categoria, regiao,
        quantidade e preco_unitario.
    """
    random.seed(seed)
    np.random.seed(seed)

    produtos = ["Notebook", "Smartphone", "Tablet", "Monitor",
                "Teclado", "Mouse", "Headset"]
    categorias = {"Notebook": "Computadores", "Smartphone": "Celulares",
                  "Tablet": "Celulares", "Monitor": "Computadores",
                  "Teclado": "Perifericos", "Mouse": "Perifericos",
                  "Headset": "Perifericos"}
    precos = {"Notebook": 3500, "Smartphone": 2200, "Tablet": 1800,
              "Monitor": 1200, "Teclado": 250, "Mouse": 120,
              "Headset": 350}
    regioes = ["Sudeste", "Sul", "Nordeste", "Centro-Oeste", "Norte"]

    data_inicio = datetime(2025, 1, 1)
    dados = []

    for i in range(n_registros):
        produto = random.choice(produtos)
        categoria = categorias[produto]
        quantidade = random.randint(1, 10)
        preco = round(precos[produto] * random.uniform(0.85, 1.15), 2)
        data = data_inicio + timedelta(days=random.randint(0, 364))
        data_txt = data.strftime("%Y-%m-%d")
        cliente = f"  Cliente_{random.randint(1, 50):03d}  "

        # --- sujeira proposital para a etapa de limpeza ---
        if random.random() < 0.03:
            data_txt = "DATA INVALIDA"

        dados.append({
            "id_venda": i + 1,
            "data_venda": data_txt,
            "cliente": cliente,
            "produto": produto,
            "categoria": categoria,
            "regiao": random.choice(regioes),
            "quantidade": quantidade,
            "preco_unitario": preco,
        })

    return pd.DataFrame(dados)


# ---------------------------------------------------------------------------
# RF02 - Inspecionar e Descrever os Dados
# ---------------------------------------------------------------------------
def inspecionar_dados(df):
    """Descreve a estrutura do DataFrame bruto, sem interpretar estatisticas.

    Exibe no console o formato, as colunas, os tipos de dados, a contagem
    de valores nulos por coluna e os primeiros registros - o objetivo e
    puramente estrutural (a leitura estatistica fica para a Semana 12).

    Args:
        df (pandas.DataFrame): dataset bruto, recem carregado do CSV.

    Returns:
        pandas.DataFrame: o mesmo DataFrame recebido, sem alteracoes
        (retornado apenas para permitir encadeamento de chamadas).
    """
    print("\n=== INSPECAO INICIAL DO DATASET ===")
    print(f"Shape: {df.shape}")
    print(f"\nColunas: {list(df.columns)}")
    print(f"\nTipos de dados:\n{df.dtypes}")
    print(f"\nValores nulos por coluna:\n{df.isnull().sum()}")
    print(f"\nPrimeiros registros:\n{df.head()}")
    return df


# ---------------------------------------------------------------------------
# RF03 - Tratar os Dados (datetime)
# ---------------------------------------------------------------------------
def tratar_dados(df):
    """Padroniza o texto, converte tipos e descarta registros invalidos.

    Etapas aplicadas, nesta ordem:
        1. remocao de espacos extras em todas as colunas de texto;
        2. conversao de data_venda com pd.to_datetime(errors="coerce") e
           remocao das linhas cuja data nao pode ser convertida;
        3. conversao de quantidade para int e preco_unitario para float,
           descartando valores que nao sejam numericos;
        4. montagem de um relatorio de limpeza com as contagens de
           registros iniciais, removidos e finais.

    Nao sao aplicadas tecnicas de imputacao estatistica de nulos nem
    tratamento de outliers: esses topicos pertencem as Semanas 09-11 e
    estao fora do escopo deste mini-projeto.

    Args:
        df (pandas.DataFrame): dataset bruto (saida de gerar_dataset_vendas
            ou de um pd.read_csv externo).

    Returns:
        tuple[pandas.DataFrame, dict]: o DataFrame limpo e um dicionario
        com o relatorio de limpeza (registros_iniciais, removidos por
        motivo e registros_finais).
    """
    df_limpo = df.copy()
    registros_iniciais = len(df_limpo)

    # 1. remover espacos extras nas colunas de texto
    colunas_texto = df_limpo.select_dtypes(include="object").columns
    for coluna in colunas_texto:
        df_limpo[coluna] = df_limpo[coluna].str.strip()
    # normalizacao adicional do nome do cliente via regex (espacos internos)
    df_limpo["cliente"] = df_limpo["cliente"].apply(
        lambda nome: re.sub(r"\s+", " ", nome).strip()
    )

    # 2. converter data_venda e descartar datas invalidas (viram NaT)
    df_limpo["data_venda"] = pd.to_datetime(df_limpo["data_venda"], errors="coerce")
    datas_invalidas = df_limpo["data_venda"].isna().sum()
    df_limpo = df_limpo.dropna(subset=["data_venda"])

    # 3. ajustar os tipos numericos (coercao + descarte de nao numericos)
    df_limpo["quantidade"] = pd.to_numeric(df_limpo["quantidade"], errors="coerce")
    df_limpo["preco_unitario"] = pd.to_numeric(df_limpo["preco_unitario"], errors="coerce")
    valores_invalidos = df_limpo[["quantidade", "preco_unitario"]].isna().any(axis=1).sum()
    df_limpo = df_limpo.dropna(subset=["quantidade", "preco_unitario"])

    df_limpo["quantidade"] = df_limpo["quantidade"].astype(int)
    df_limpo["preco_unitario"] = df_limpo["preco_unitario"].astype(float)

    registros_finais = len(df_limpo)

    # 4. montar e imprimir o relatorio de limpeza
    relatorio = {
        "registros_iniciais": registros_iniciais,
        "removidos_data_invalida": int(datas_invalidas),
        "removidos_valor_numerico_invalido": int(valores_invalidos),
        "registros_inalterados": registros_finais,
        "registros_finais": registros_finais,
    }

    print("\n=== RELATORIO DE LIMPEZA ===")
    for chave, valor in relatorio.items():
        print(f"{chave}: {valor}")

    return df_limpo, relatorio


# ---------------------------------------------------------------------------
# RF04 - Criar Colunas Derivadas com Transformacoes Condicionais
# ---------------------------------------------------------------------------
def criar_colunas_derivadas(df):
    """Enriquece o dataset limpo com colunas derivadas da data e da receita.

    Colunas criadas:
        receita_total: quantidade x preco_unitario (operacao vetorizada);
        mes, ano: extraidos de data_venda via .dt.month / .dt.year;
        mes_nome: nome do mes em portugues (dicionario MESES_PT);
        trimestre: trimestre no formato Q1-Q4, via .dt.quarter;
        faixa_receita_item: classificacao condicional vetorizada da
            receita da linha (Baixo / Medio / Alto Valor).

    Args:
        df (pandas.DataFrame): dataset ja tratado por tratar_dados().

    Returns:
        pandas.DataFrame: copia do DataFrame de entrada acrescida das
        colunas derivadas descritas acima.
    """
    df = df.copy()

    df["receita_total"] = df["quantidade"] * df["preco_unitario"]
    df["mes"] = df["data_venda"].dt.month
    df["mes_nome"] = df["mes"].map(MESES_PT)
    df["trimestre"] = "Q" + df["data_venda"].dt.quarter.astype(str)
    df["ano"] = df["data_venda"].dt.year

    # transformacao condicional vetorizada (equivalente a um if/elif/else
    # aplicado linha a linha, porem sem necessidade de laco explicito)
    condicoes = [
        df["receita_total"] < 500,
        (df["receita_total"] >= 500) & (df["receita_total"] < 5000),
        df["receita_total"] >= 5000,
    ]
    faixas = ["Baixo Valor", "Medio Valor", "Alto Valor"]

    df["faixa_receita_item"] = "Nao Classificado"
    for condicao, faixa in zip(condicoes, faixas):
        df.loc[condicao, "faixa_receita_item"] = faixa

    return df


# ---------------------------------------------------------------------------
# RF05 - Calcular Metricas Agregadas com groupby
# ---------------------------------------------------------------------------
def calcular_metricas(df):
    """Calcula as metricas agregadas de negocio exigidas pelo desafio.

    Sao calculadas quatro dimensoes de analise, cada uma via groupby +
    agregacao, devolvidas em formato tabular (reset_index):
        por_mes: receita, quantidade e numero de vendas por mes/ano;
        top_produtos: receita total dos 5 produtos mais vendidos;
        por_categoria: receita total por categoria de produto;
        por_regiao: receita total e ticket medio por regiao.

    Args:
        df (pandas.DataFrame): dataset com as colunas derivadas (saida de
            criar_colunas_derivadas()).

    Returns:
        dict[str, pandas.DataFrame]: dicionario com as chaves por_mes,
        top_produtos, por_categoria e por_regiao.
    """
    metricas = {}

    por_mes = (
        df.groupby(["ano", "mes", "mes_nome"])
        .agg(receita_total=("receita_total", "sum"),
             quantidade=("quantidade", "sum"),
             n_vendas=("id_venda", "count"))
        .reset_index()
        .sort_values(["ano", "mes"])
    )
    metricas["por_mes"] = por_mes

    top_produtos = (
        df.groupby("produto")["receita_total"]
        .sum()
        .reset_index()
        .sort_values("receita_total", ascending=False)
        .head(5)
    )
    metricas["top_produtos"] = top_produtos

    por_categoria = (
        df.groupby("categoria")["receita_total"]
        .sum()
        .reset_index()
        .sort_values("receita_total", ascending=False)
    )
    metricas["por_categoria"] = por_categoria

    por_regiao = (
        df.groupby("regiao")
        .agg(receita_total=("receita_total", "sum"),
             ticket_medio=("receita_total", "mean"))
        .reset_index()
        .sort_values("receita_total", ascending=False)
    )
    metricas["por_regiao"] = por_regiao

    print("\n=== POR MES ===")
    print(por_mes)
    print("\n=== TOP PRODUTOS ===")
    print(top_produtos)
    print("\n=== POR CATEGORIA ===")
    print(por_categoria)
    print("\n=== POR REGIAO ===")
    print(por_regiao)

    return metricas


# ---------------------------------------------------------------------------
# RF06 - Segmentar Clientes por Nivel de Gasto
# ---------------------------------------------------------------------------
def segmentar_clientes(df):
    """Segmenta os clientes por faixa de gasto acumulado.

    Regra de classificacao (aplicada via funcao lambda com .apply()):
        Bronze: total_gasto abaixo de R$ 5.000,00;
        Prata: total_gasto entre R$ 5.000,00 e R$ 15.000,00;
        Ouro: total_gasto acima de R$ 15.000,00.

    Args:
        df (pandas.DataFrame): dataset com a coluna receita_total.

    Returns:
        pandas.DataFrame: colunas cliente, total_gasto e segmento,
        ordenado do maior para o menor gasto.
    """
    clientes = (
        df.groupby("cliente")["receita_total"]
        .sum()
        .reset_index()
        .rename(columns={"receita_total": "total_gasto"})
    )

    classificar_segmento = lambda gasto: (
        "Ouro" if gasto > 15000 else "Prata" if gasto >= 5000 else "Bronze"
    )
    clientes["segmento"] = clientes["total_gasto"].apply(classificar_segmento)
    clientes = clientes.sort_values("total_gasto", ascending=False).reset_index(drop=True)

    print("\n=== TOP 10 CLIENTES ===")
    print(clientes.head(10))
    print("\n=== DISTRIBUICAO POR SEGMENTO ===")
    print(clientes["segmento"].value_counts())

    return clientes


# ---------------------------------------------------------------------------
# RF07 - Criar Visualizacoes com Matplotlib
# ---------------------------------------------------------------------------
def gerar_graficos(df, metricas, pasta_graficos=PASTA_GRAFICOS):
    """Gera e exporta em PNG as quatro figuras exigidas pelo desafio.

    Figuras produzidas:
        1. receita_por_mes.png  - linha, evolucao da receita no periodo;
        2. top_produtos.png     - barra, top 5 produtos por receita;
        3. quantidade_vs_receita.png - dispersao, colorida por categoria;
        4. painel_resumo.png    - grade 2x2 combinando as tres visoes
           acima mais a receita por regiao, com fig.suptitle().

    Todas as figuras recebem titulo, rotulos de eixo, legenda (quando
    aplicavel), paleta de cores definida explicitamente e sao exportadas
    com plt.savefig(dpi=150), apos plt.tight_layout().

    Args:
        df (pandas.DataFrame): dataset tratado e enriquecido, usado no
            grafico de dispersao.
        metricas (dict[str, pandas.DataFrame]): saida de calcular_metricas(),
            usada nos graficos de linha, barra e no painel de regiao.
        pasta_graficos (str): diretorio de destino dos arquivos PNG.

    Returns:
        None.
    """
    os.makedirs(pasta_graficos, exist_ok=True)

    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["axes.titlesize"] = 14

    por_mes = metricas["por_mes"]
    top_produtos = metricas["top_produtos"]
    por_regiao = metricas["por_regiao"]

    # 1. Linha - receita por mes
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(por_mes["mes_nome"], por_mes["receita_total"],
            marker="o", linewidth=2, color="#2E86AB")
    ax.set_title("Receita Total por Mes")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Receita Total (R$)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_graficos, "receita_por_mes.png"), dpi=150)
    plt.close()

    # 2. Barra - top produtos
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top_produtos, y="produto", x="receita_total",
                hue="produto", legend=False, palette="Blues_d", ax=ax)
    ax.set_title("Top 5 Produtos por Receita")
    ax.set_xlabel("Receita Total (R$)")
    ax.set_ylabel("Produto")
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_graficos, "top_produtos.png"), dpi=150)
    plt.close()

    # 3. Dispersao - quantidade vs receita, colorido por categoria
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x="quantidade", y="receita_total",
                     hue="categoria", palette="deep", s=70, ax=ax)
    ax.set_title("Quantidade vs Receita por Transacao")
    ax.set_xlabel("Quantidade")
    ax.set_ylabel("Receita Total (R$)")
    ax.legend(title="Categoria")
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_graficos, "quantidade_vs_receita.png"), dpi=150)
    plt.close()

    # 4. Painel resumo - subplots 2x2
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(por_mes["mes_nome"], por_mes["receita_total"],
                     marker="o", linewidth=2, color="#2E86AB")
    axes[0, 0].set_title("Receita por Mes")
    axes[0, 0].set_xlabel("Mes")
    axes[0, 0].set_ylabel("Receita Total (R$)")
    axes[0, 0].tick_params(axis="x", rotation=45)

    axes[0, 1].barh(top_produtos["produto"], top_produtos["receita_total"],
                     color="#A23B72")
    axes[0, 1].set_title("Top 5 Produtos")
    axes[0, 1].set_xlabel("Receita Total (R$)")
    axes[0, 1].invert_yaxis()

    categorias = df["categoria"].unique()
    cores = sns.color_palette("deep", len(categorias))
    for categoria, cor in zip(categorias, cores):
        subset = df[df["categoria"] == categoria]
        axes[1, 0].scatter(subset["quantidade"], subset["receita_total"],
                            label=categoria, color=cor, alpha=0.7)
    axes[1, 0].set_title("Quantidade vs Receita")
    axes[1, 0].set_xlabel("Quantidade")
    axes[1, 0].set_ylabel("Receita Total (R$)")
    axes[1, 0].legend(title="Categoria", fontsize=8)

    axes[1, 1].bar(por_regiao["regiao"], por_regiao["receita_total"],
                    color="#F18F01")
    axes[1, 1].set_title("Receita por Regiao")
    axes[1, 1].set_xlabel("Regiao")
    axes[1, 1].set_ylabel("Receita Total (R$)")
    axes[1, 1].tick_params(axis="x", rotation=30)

    fig.suptitle("SalesInsight PY - Painel Resumo", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_graficos, "painel_resumo.png"), dpi=150)
    plt.close()

    print(f"\n[OK] 4 graficos exportados em '{pasta_graficos}/'.")


# ---------------------------------------------------------------------------
# RF08 - Organizar o Codigo em Funcoes Reutilizaveis (funcao de ordem superior)
# ---------------------------------------------------------------------------
def processar_coluna(df, coluna, funcao_transformacao, nome_saida=None):
    """Aplica uma funcao de transformacao arbitraria a uma coluna do DataFrame.

    Funcao de ordem superior: recebe outra funcao (funcao_transformacao)
    como argumento e a aplica via .apply(), demonstrando o conceito
    estudado na Semana 05. E reutilizada na classe AnalisadorDeVendas com
    duas funcoes lambda em contextos distintos.

    Args:
        df (pandas.DataFrame): DataFrame de entrada.
        coluna (str): nome da coluna a transformar.
        funcao_transformacao (Callable): funcao de um argumento aplicada
            a cada valor da coluna.
        nome_saida (str, opcional): nome da coluna resultante. Quando
            omitido, usa "{coluna}_transformado".

    Returns:
        pandas.DataFrame: copia do DataFrame de entrada com a nova coluna.
    """
    df = df.copy()
    nome_saida = nome_saida or f"{coluna}_transformado"
    df[nome_saida] = df[coluna].apply(funcao_transformacao)
    return df


# ---------------------------------------------------------------------------
# RF09 - Exportar Resultados em CSV e JSON
# ---------------------------------------------------------------------------
def exportar_resultados(metricas, clientes, estatisticas, pasta=PASTA_OUTPUTS):
    """Exporta os principais resultados do projeto em CSV e JSON.

    Grava metricas_por_mes.csv, segmentacao_clientes.csv e
    estatisticas_gerais.json e, em seguida, le o JSON de volta com
    json.load() para comprovar que a escrita foi bem-sucedida.

    Args:
        metricas (dict[str, pandas.DataFrame]): saida de calcular_metricas().
        clientes (pandas.DataFrame): saida de segmentar_clientes().
        estatisticas (dict): estatisticas gerais do dataset (receita total,
            ticket medio, quantidade total e numero de vendas).
        pasta (str): diretorio de destino dos arquivos exportados.

    Returns:
        None.
    """
    os.makedirs(pasta, exist_ok=True)

    metricas["por_mes"].to_csv(os.path.join(pasta, "metricas_por_mes.csv"),
                                index=False, encoding="utf-8-sig")
    clientes.to_csv(os.path.join(pasta, "segmentacao_clientes.csv"),
                     index=False, encoding="utf-8-sig")

    serializavel = {k: round(float(v), 2) for k, v in estatisticas.items()}
    caminho = os.path.join(pasta, "estatisticas_gerais.json")
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(serializavel, f, indent=4, ensure_ascii=False)

    # leitura de volta para confirmar a escrita
    with open(caminho, "r", encoding="utf-8") as f:
        conferencia = json.load(f)
    print(f"\n[OK] Resultados exportados em '{pasta}/'.")
    print(f"JSON gravado e lido: {conferencia}")
