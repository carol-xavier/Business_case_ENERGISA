#%%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# Configurações de visualização
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.2f}'.format)

sns.set_theme(style='whitegrid', palette='deep')

#%%
df = pd.read_csv('Logs_EAC_AT.csv')
df.head()

# %%
print(f'Linhas: {df.shape[0]:,}')
print(f'Colunas: {df.shape[1]}')

#%%
df.info()

# %%
df.describe(include='all').T

# %%
# Quantos modelos ML diferentes existem no dataset?
num_models = df['ml_model_id'].nunique()
print(f'Número de modelos ML diferentes: {num_models}')

# %%
df["object_created_at"] = pd.to_datetime(df["object_created_at"], format="mixed", utc=True)

last_obj = (
    df.groupby("ml_model_id")["object_created_at"]
      .max()
      .sort_values()          
      .reset_index(name="ultima_criacao")
)


today = pd.Timestamp.now(tz="UTC").normalize()
last_obj["dias_sem_criar"] = (today - last_obj["ultima_criacao"]).dt.days

print(last_obj)
# %%
status = df['awaiting_feedback'].value_counts()

plt.figure(figsize=(9, 5))

ax = sns.barplot(
    x=['Avaliado' if not x else 'Pendente'
       for x in status.index],
    y=status.values,
    hue=status.index,
    palette=["#011AA3", "#D8521D"]
)

# Add count + percentage labels
for p, percent in zip(ax.patches, status.values / status.values.sum() * 100):
    height = p.get_height()
    ax.text(
        p.get_x() + p.get_width() / 2,
        height,
        f"{int(height)} ({percent:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=12
    )

ax.margins(y=0.15)  
plt.title('Distribuição dos registros por status', weight='bold', pad=12)
plt.xlabel('Status')
plt.ylabel('Quantidade de registros')
plt.tight_layout()
plt.show()


# %%
df['inspector_name'].value_counts()

# %%
"""
Renato Casdabrande tem exatamente o mesmo número de registros pendentes: 269617! 
Portanto, deve se tratar de um valor padrão nos registros.
"""
# %%
# Relação entre ID e nomes das classes

classes = (
    df.groupby('object_type_id')
      .agg(
          nome_pt=('object_type_name_pt', 'nunique'),
          nome_en=('object_type_name_en', 'nunique'),
          registros=('object_type_id', 'size')
      )
      .reset_index()
)


print("Quantidade de IDs:")
print(df['object_type_id'].nunique())

print("Quantidade de nomes em português:")
print(df['object_type_name_pt'].nunique())

print("Quantidade de nomes em inglês:")
print(df['object_type_name_en'].nunique())

print(classes)


# %%
nomes_pt = (
    df.groupby('object_type_name_pt')
      .agg(
          quantidade_ids=('object_type_id', 'nunique'),
          ids=('object_type_id', lambda x: sorted(x.unique()))
      )
      .reset_index()
      .sort_values('quantidade_ids', ascending=False)
)

print(nomes_pt[nomes_pt['quantidade_ids'] > 1])

# %%
comparacao_classes = (
    df.groupby(['object_type_name_pt', 'object_type_id', 'object_type_name_en'])
      .size()
      .reset_index(name='registros')
      .sort_values('object_type_name_pt')
)

print(
    comparacao_classes[
        comparacao_classes['object_type_name_pt'].isin([
            'Amortecedor de Vibração',
            'Isolador de Pino de porcelana',
            'Transformador',
            'Chave fusível'
        ])
    ]
)
# %%
"""
Aparantemente, para esses quatro tipos de objetos em português correspondem ao mesmo produto em 
inglês, mas a versão em inglês apresenta um detalhamento maior, com diferentes IDs para cada tipo de subproduto.
"""

# %%
# Relação entre inspetor e status de avaliação
print("=== INSPETOR X STATUS ===")

print(
    pd.crosstab(
        df['inspector_name'],
        df['awaiting_feedback'],
        margins=True
    )
)

# 2. Relação entre defeito e criticidade
print("\n=== DEFEITO X CRITICIDADE ===")

print(
    pd.crosstab(
        df['object_is_defect'],
        df['criticality_level'].notna(),
        margins=True
    )
)

# 3. Consistência entre TP, FP, FN e status
print("\n=== COMBINAÇÕES TP, FP E FN ===")

print(
    df.groupby(['TP', 'FP', 'FN', 'awaiting_feedback'])
      .size()
      .reset_index(name='quantidade')
      .sort_values('quantidade', ascending=False)
)

# 4. Duplicidade do identificador
print("\n=== DUPLICIDADE DE BOUNDING BOX ===")

print("IDs ausentes:", df['bounding_box_id'].isna().sum())
print("Linhas com ID duplicado:",
      df['bounding_box_id'].duplicated(keep=False).sum())
print("Duplicatas além da primeira:",
      df['bounding_box_id'].duplicated().sum())

# %%
"""
Variáveis de para avaliação Global: TP, FP, FN, Precision, Recall e F1-score
"""

TP = df['TP'].sum()
FP = df['FP'].sum()
FN = df['FN'].sum()

print(f'True Positives (TP): {TP:,}')
print(f'False Positives (FP): {FP:,}')
print(f'False Negatives (FN): {FN:,}')


# %%
Precision = ((TP) / (TP + FP))*100

Recall = ((TP) / (TP + FN))*100

F1_score = 2 * (Precision * Recall) / (Precision + Recall)

print(f'Precision: {Precision:.2f}%')
print(f'Recall: {Recall:.2f}%')
print(f'F1-score: {F1_score:.2f}%')

# %%
df_avaliados = df[df['awaiting_feedback'] == False].copy()

# Quantidade de registros por modelo
resumo_modelos = (
    df_avaliados
    .groupby('ml_model_id')
    .agg(
        total_registros=('ml_model_id', 'size'),
        TP=('TP', 'sum'),
        FP=('FP', 'sum'),
        FN=('FN', 'sum')
    )
    .reset_index()
    .sort_values('total_registros', ascending=False)
)

print(resumo_modelos)

# %%
# Precision por modelo
resumo_modelos['Precision'] = (
    resumo_modelos['TP'] /
    (resumo_modelos['TP'] + resumo_modelos['FP'])
)

# Converter para percentual
resumo_modelos['Precision_pct'] = resumo_modelos['Precision'] * 100

# Ordenar do maior para o menor desempenho
resumo_modelos_ordenado = resumo_modelos.sort_values(
    'Precision',
    ascending=False
)

print(
    resumo_modelos_ordenado[
        ['ml_model_id', 'total_registros', 'TP', 'FP', 'Precision_pct']
    ].round(2)
)

# %%
plt.figure(figsize=(14, 8))

ax = sns.scatterplot(
    data=resumo_modelos,
    x='total_registros',
    y='Precision_pct',
    size='total_registros',
    hue='Precision_pct',
    sizes=(60, 450),
    palette='RdYlGn',
    legend=False
)

# Identificação de cada modelo
for _, row in resumo_modelos.iterrows():
    ax.annotate(
        str(int(row['ml_model_id'])),
        (row['total_registros'], row['Precision_pct']),
        xytext=(5, 5),
        textcoords='offset points',
        fontsize=8
    )

plt.title('Precision por modelo de IA')
plt.xlabel('Quantidade de registros avaliados')
plt.ylabel('Precision (%)')
plt.ylim(-5, 105)

plt.tight_layout()
plt.show()

# %%
# Comparação entre TP e FP por modelo
resumo_modelos_plot = resumo_modelos.sort_values(
    'total_registros',
    ascending=False
)

resumo_modelos_plot.set_index('ml_model_id')[['TP', 'FP']].plot(
    kind='bar',
    figsize=(14, 6),
    width=0.8
)

plt.title('True Positives e False Positives por modelo')
plt.xlabel('ID do modelo')
plt.ylabel('Quantidade de registros')
plt.xticks(rotation=45)
plt.legend(title='Classificação')
plt.tight_layout()
plt.show()


# %%
# Verificar presença de ml_model_id por classificação

print(
    df_avaliados.groupby(
        ['TP', 'FP', 'FN'],
        dropna=False
    )['ml_model_id']
    .agg(
        total='count',
        ausentes=lambda x: x.isna().sum(),
        modelos_distintos='nunique'
    )
)
# %%
# Examinar os registros de falsos negativos

df_FN = df_avaliados[df_avaliados['FN'] == True].copy()

print("Total de FN:", len(df_FN))
print("ml_model_id ausente:", df_FN['ml_model_id'].isna().sum())

print(
    df_FN[
        [
            'bounding_box_id',
            'arkion_flight_id',
            'arkion_image_id',
            'object_type_id',
            'object_type_name_en',
            'project_name',
            'object_created_at',
            'ml_model_id'
        ]
    ].head(10)
)
# %%
# Comparar Precision dos modelos com a Precision global

precision_global = Precision * 100

resumo_modelos['Diferenca_precision_global'] = (
    resumo_modelos['Precision_pct'] - precision_global
)

resumo_modelos['Desempenho'] = np.where(
    resumo_modelos['Precision_pct'] >= precision_global,
    'Acima da média global',
    'Abaixo da média global'
)

print(
    resumo_modelos[
        [
            'ml_model_id',
            'total_registros',
            'TP',
            'FP',
            'Precision_pct',
            'Diferenca_precision_global',
            'Desempenho'
        ]
    ]
    .sort_values('Precision_pct', ascending=False)
    .round(2)
)

# %%
# Definir corte exploratório de volume
volume_minimo = 100

resumo_modelos['Grupo_volume'] = np.where(
    resumo_modelos['total_registros'] >= volume_minimo,
    'Volume >= 100',
    'Volume < 100'
)

# Tabela dos modelos com volume mínimo
modelos_relevantes = (
    resumo_modelos[
        resumo_modelos['total_registros'] >= volume_minimo
    ]
    .sort_values('Precision_pct', ascending=False)
)

print(
    modelos_relevantes[
        [
            'ml_model_id',
            'total_registros',
            'TP',
            'FP',
            'Precision_pct',
            'Diferenca_precision_global'
        ]
    ].round(2)
)

# %%
# Preparar os dados
modelos_plot = modelos_relevantes.sort_values(
    'Precision_pct',
    ascending=True
).copy()

# Transformar os IDs em texto para tratá-los como categorias
modelos_plot['ml_model_id'] = (
    modelos_plot['ml_model_id'].astype(int).astype(str)
)

# Criar gráfico
fig, ax = plt.subplots(figsize=(12, 7))

barras = ax.barh(
    modelos_plot['ml_model_id'],
    modelos_plot['Precision_pct']
)

# Linha de referência global
ax.axvline(
    precision_global,
    color='red',
    linestyle='--',
    label=f'Precision global: {precision_global:.2f}%'
)

# Valores ao lado das barras
ax.bar_label(
    barras,
    labels=[f'{v:.1f}%' for v in modelos_plot['Precision_pct']],
    padding=4
)

ax.set_title('Precision por modelo de IA')
ax.set_xlabel('Precision (%)')
ax.set_ylabel('ID do modelo')
ax.set_xlim(0, 110)
ax.legend()

plt.tight_layout()
plt.show()

# %%
# Quantidade de classes distintas por modelo
classes_por_modelo = (
    df_avaliados[df_avaliados['ml_model_id'].notna()]
    .groupby('ml_model_id')
    .agg(
        classes_distintas=('object_type_id', 'nunique'),
        projetos_distintos=('project_id', 'nunique')
    )
    .reset_index()
)

# Adicionar ao resumo dos modelos
resumo_modelos = resumo_modelos.merge(
    classes_por_modelo,
    on='ml_model_id',
    how='left'
)

print(
    resumo_modelos[
        [
            'ml_model_id',
            'total_registros',
            'Precision_pct',
            'classes_distintas',
            'projetos_distintos'
        ]
    ]
    .sort_values('total_registros', ascending=False)
    .head(15)
)

# %%
# 1. Preparar datas
df['ml_model_created_at'] = pd.to_datetime(
    df['ml_model_created_at'],
    errors='coerce',
    utc=True
)

# 2. Resumo por modelo, considerando toda a base
analise_modelos = (
    df[df['ml_model_id'].notna()]
    .groupby('ml_model_id')
    .agg(
        quantidade_imagens=('arkion_image_id', 'nunique'),
        quantidade_objetos=('bounding_box_id', 'nunique'),
        total_registros=('ml_model_id', 'size'),
        TP=('TP', 'sum'),
        FP=('FP', 'sum'),
        primeira_data_criacao=('ml_model_created_at', 'min'),
        ultima_data_criacao=('ml_model_created_at', 'max'),
        datas_criacao_distintas=('ml_model_created_at', 'nunique')
    )
    .reset_index()
)

# 3. Precision baseada nos registros avaliados
analise_modelos['Precision_pct'] = (
    analise_modelos['TP'] /
    (analise_modelos['TP'] + analise_modelos['FP'])
) * 100

# 4. Identificar quantidade de registros avaliados e pendentes
analise_modelos['avaliados'] = (
    analise_modelos['TP'] + analise_modelos['FP']
)

analise_modelos['pendentes'] = (
    analise_modelos['total_registros'] -
    analise_modelos['avaliados']
)

# 5. FN não atribuíveis aos modelos
analise_modelos['FN'] = np.nan

# 6. Ordenar por volume
analise_modelos = analise_modelos.sort_values(
    'total_registros',
    ascending=False
)

print(analise_modelos)
# %%
print(
    analise_modelos[
        [
            'ml_model_id',
            'primeira_data_criacao',
            'ultima_data_criacao',
            'datas_criacao_distintas'
        ]
    ]
)


# %%
df['project_name'].nunique()
# %%
df['ml_model_id'].nunique()

#%%
project_name = (
    df.groupby('project_name')
      .agg(
          quantidade_ids=('ml_model_id', 'nunique'),
          ids=('ml_model_id', lambda x: sorted(x.unique()))
      )
      .reset_index()
      .sort_values('quantidade_ids', ascending=False)
)

print(project_name[project_name['quantidade_ids'] > 1])
# %%
# Registros que receberam feedback (TP ou FP)
avaliados = df[
    (df["TP"] == True) | (df["FP"] == True)
].copy()

# Verificar presença de bounding_box_id
avaliados["bounding_box_id"].isna().value_counts(dropna=False)

# %%
# Gráfico sobre taxas relacionadas às coortes com e sem defeito
categorias = [
    'Global',
    'Componentes\n(sem defeito)',
    'Classes de\ndefeito'
]

precision = [50.3, 90.3, 24.9]
recall = [84.8, 91.0, 73.4]
f1 = [63.1, 90.7, 37.2]

x = np.arange(len(categorias))
largura = 0.25

cores = {
    'precision': '#08749C',
    'recall': '#20885F',
    'f1': '#D28A0B'
}

fig, ax = plt.subplots(figsize=(7.5, 4.2))

barras_precision = ax.bar(
    x - largura, precision, largura,
    color=cores['precision'],
    label='Taxa de acerto (Precision)'
)

barras_recall = ax.bar(
    x, recall, largura,
    color=cores['recall'],
    label='Taxa de detecção (Recall)'
)

barras_f1 = ax.bar(
    x + largura, f1, largura,
    color=cores['f1'],
    label='F1'
)

for barras in [barras_precision, barras_recall, barras_f1]:
    for barra in barras:
        valor = barra.get_height()
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            valor + 1.5,
            f'{valor:.1f}'.replace('.', ','),
            ha='center',
            va='bottom',
            fontsize=10,
            color='#222222'
        )

ax.set_xticks(x)
ax.set_xticklabels(categorias, fontsize=11)
ax.set_ylabel('%', fontsize=12)
ax.set_ylim(0, 112)
ax.set_yticks(np.arange(0, 101, 20))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.spines['left'].set_color('#9AA5B1')
ax.spines['bottom'].set_color('#9AA5B1')

ax.tick_params(axis='both', colors='#344054')

ax.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    frameon=False,
    fontsize=9,
    handlelength=1.5,
    columnspacing=1.5
)

plt.tight_layout()
plt.show()

#%%
import pandas as pd
import numpy as np

df['object_created_at'] = pd.to_datetime(
    df['object_created_at'],
    errors='coerce'
)

dados = df.copy()

dados['mes'] = (
    dados['object_created_at']
    .dt.to_period('M')
)

mensal = (
    dados.groupby('mes')
    .agg(
        TP=('TP', 'sum'),
        FP=('FP', 'sum'),
        FN=('FN', 'sum')
    )
)

mensal['Precision'] = (
    100 * mensal['TP'] /
    (mensal['TP'] + mensal['FP'])
)

mensal['Recall'] = (
    100 * mensal['TP'] /
    (mensal['TP'] + mensal['FN'])
)

mensal['F1'] = (
    2 * mensal['Precision'] * mensal['Recall'] /
    (mensal['Precision'] + mensal['Recall'])
)

mensal['Avaliados'] = (
    mensal['TP'] + mensal['FP'] + mensal['FN']
)

print(mensal.round(1))

# %%
#Gráfico de evolução dos modelos
meses = [
    'out/25', 'nov/25', 'dez/25', 'jan/26',
    'fev/26', 'mar/26', 'abr/26', 'mai/26',
    'jun/26', 'jul/26', 'ago/26', 'set/26'
]

precision = [70, 2, 48, 6, 3, 5, 76, 44, 17, 49, 16, 16]
recall = [100, 12, 36, 100, 83, 23, 91, 68, 95, 100, 99, 100]
f1 = [82, 2, 41, 12, 6, 9, 83, 53, 29, 66, 27, 28]

cores = {
    'precision': '#08749C',
    'recall': '#20885F',
    'f1': '#D28A0B'
}

fig, ax = plt.subplots(figsize=(10, 5.5))

x = np.arange(len(meses))

series = [
    ('Taxa de acerto', precision, cores['precision']),
    ('Taxa de detecção', recall, cores['recall']),
    ('F1', f1, cores['f1'])
]

for nome, valores, cor in series:
    ax.plot(
        x, valores,
        color=cor,
        linewidth=2.8,
        label=nome,
        zorder=2
    )

    for i, valor in enumerate(valores):
        vazado = i in [3, 4, 5, 11]

        ax.plot(
            x[i], valor,
            marker='o',
            markersize=7,
            markerfacecolor='white' if vazado else cor,
            markeredgecolor=cor,
            markeredgewidth=1.5,
            linestyle='None',
            zorder=3
        )

ax.set_xticks(x)
ax.set_xticklabels(
    meses,
    rotation=45,
    ha='right',
    fontsize=11
)

ax.set_ylabel('%', fontsize=14)
ax.set_ylim(0, 108)
ax.set_yticks(np.arange(0, 101, 20))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#94A3B8')
ax.spines['bottom'].set_color('#94A3B8')

ax.tick_params(
    axis='both',
    labelsize=11,
    colors='#1B2A41'
)

ax.grid(False)

ax.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    frameon=False,
    fontsize=12,
    handlelength=2
)

plt.tight_layout()

plt.savefig(
    'evolucao_performance_ia.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# %%
# Top 10 classes com mais falsos positivos
top_fp = (
    df[df['FP'] == True]
    .groupby('object_type_name_pt')
    .size()
    .sort_values(ascending=False)
    .head(10)
)

print(top_fp)

# %%
#Gráfico Alarme falso vs. Classes de defeitos
classes = [
    'Base de concreto danificado',
    'Treliça danificada',
    'Isolador contaminado',
    'Vegetação alta na Torre',
    'Condutor com tento rompido',
    'Vazado',
    'Demais classes'
]

fp = [1492, 1369, 1318, 776, 534, 330, 817]

acerto = [18.5, 1.9, 36.8, 52.7, 1.5, 2.9, None]

azul = '#011AA3'
laranja = '#D8521D'

cores = [laranja] * 6 + [azul]

fig, ax = plt.subplots(figsize=(11, 5.5))

y = np.arange(len(classes))

barras = ax.barh(
    y,
    fp,
    height=0.8,
    color=cores
)

for i, (barra, valor, taxa) in enumerate(zip(barras, fp, acerto)):

    texto = f'{valor:,.0f}'.replace(',', '.')

    if taxa is not None:
        texto += f'  (acerto {taxa:.1f}%)'.replace('.', ',')

        texto = texto.replace(
            f'{valor:,}'.replace(',', '.'),
            f'{valor:,}'.replace(',', '.')
        )

    ax.text(
        valor + 25,
        barra.get_y() + barra.get_height() / 2,
        texto,
        va='center',
        ha='left',
        fontsize=11,
        color='#111111'
    )

ax.set_yticks(y)
ax.set_yticklabels(classes, fontsize=12)

ax.invert_yaxis()

ax.set_xlabel('Alertas falsos (FP)', fontsize=13)
ax.set_xlim(0, 2400)
ax.set_xticks(np.arange(0, 2001, 500))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.spines['left'].set_color('#94A3B8')
ax.spines['bottom'].set_color('#94A3B8')

ax.tick_params(axis='both', labelsize=11, colors='#1B2A41')

ax.grid(False)

plt.tight_layout()
plt.show()
# %%
# Gráfico de distribuição do tempo de feedback (em dias) para os registros avaliados
categorias = [
    'até 1', '1–3', '3–7', '7–15',
    '15–30', '30–60', 'mais\nde 60'
]

contagens = [6424, 666, 749, 1368, 965, 2443, 1920]
percentuais = np.array(contagens) / sum(contagens) * 100

cores = ['#011AA3'] * 5 + ['#D18A13', '#D8521D']

fig, ax = plt.subplots(figsize=(9, 5))

barras = ax.bar(
    categorias,
    contagens,
    color=cores,
    width=0.8
)

for barra, percentual in zip(barras, percentuais):
    ax.text(
        barra.get_x() + barra.get_width() / 2,
        barra.get_height() + 80,
        f'{percentual:.1f}%'.replace('.', ','),
        ha='center',
        fontsize=12
    )

ax.set_xlabel('Tempo de feedback (dias)', fontsize=14, labelpad=15)
ax.set_ylabel('Registros', fontsize=13)

ax.set_ylim(0, 7400)
ax.set_yticks(np.arange(0, 7001, 1000))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#94A3B8')
ax.spines['bottom'].set_color('#94A3B8')

ax.tick_params(axis='both', labelsize=12, colors='#1B2A41')

plt.tight_layout()
plt.savefig('tempo_feedback.png', dpi=300, bbox_inches='tight')
plt.show()
# %%
