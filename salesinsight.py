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
