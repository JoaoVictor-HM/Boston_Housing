import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import numpy as np
import altair as at
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv('housing.csv', header=None, delimiter=r"\s+", names=column_names)

st.title('Boston Housing Analysis')
st.markdown("""
### Análise de Dados Residenciais de Boston e seu Efeito sob os Preços de Casas
**Dados Usados na Análise**
""")
st.write(data)

st.sidebar.header('Definição dos Parâmetros')
crim = st.sidebar.text_input('Criminalidade per Capita')
zn = st.sidebar.text_input('Porcentagem de Terrenos com mais de 25.000 Pés Quadrados')
indus = st.sidebar.text_input('Porcentagem de Zonas Comerciais não Varejista no Local')
chas = st.sidebar.checkbox('Beira o Rio Charles')
if chas:
    valid = 1
else:
    valid = 0
nox = st.sidebar.text_input('Concentração de Óxidos Nítricos (0 a 1)')
rm = st.sidebar.text_input('Número Médio de Cômodos por Habitação')
age = st.sidebar.text_input('Porcentagem de Terrenos Ocupados Construídos Antes de 1940')
dis = st.sidebar.text_input('Distância para os 5 Centros de Emprego de Boston')
rad = st.sidebar.text_input('Índice de Acesso às Estradas')
tax = st.sidebar.text_input('Taxa de Imposto sobre Bens de Valor Integral por $10.000')
ptratio = st.sidebar.text_input('Número de Alunos por Professor')
b = st.sidebar.text_input('1000(Bk - 0.63)^2 onde Bk é a proporção de negros por cidade')
lstat = st.sidebar.text_input('Percentual da População Pertencente a Camadas mais Pobres')



x = pd.DataFrame(np.c_[data['CRIM'], data['ZN'], data['INDUS'], data['CHAS'], data['NOX'], data['RM'], data['AGE'], data['DIS'], data['RAD'], data['TAX'], data['PTRATIO'], data['B'], data['LSTAT'] ], columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])
y = data['MEDV']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=15)

linear_model = LinearRegression()
linear_model.fit(x_train, y_train)

predictions = linear_model.predict(x_test)

if st.sidebar.button('Predict'):
    st.write("""
    **Prevendo os Dados Inseridos**
    """)
    input_df = pd.DataFrame(data={'CRIM': float(crim), 'ZN': float(zn), 'INDUS': float(indus), 'CHAS': valid, 'NOX': float(nox), 'RM': float(rm), 'AGE': float(age), 'DIS': float(dis), 'RAD': int(rad), 'TAX': int(tax), 'PTRATIO': float(ptratio), 'B': float(b), 'LSTAT': float(lstat)}, index=[0])
    input_test = linear_model.predict(input_df)
    input_df['PREDICTION'] = input_test
    st.write(input_df)

st.write("""
**Importe um CSV**
""")
file = st.file_uploader('', type=['csv'])
if file is not None:
    colunas = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    new_data = pd.read_csv(file, header=None, delimiter=r"\s+", names=colunas,usecols = [i for i in range(13)])
    import_test = linear_model.predict(new_data)
    new_data['PREDICTION'] = import_test
    st.write(new_data)









