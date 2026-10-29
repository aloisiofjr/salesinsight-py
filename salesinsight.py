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
