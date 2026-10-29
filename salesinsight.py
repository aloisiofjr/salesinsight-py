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
