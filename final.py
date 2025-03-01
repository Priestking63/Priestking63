import pandas as pd

def calculate_metrics(groups):
    
    active_studs = pd.read_csv("active_studs.csv", sep=",")
    checks = pd.read_csv("checks.csv", sep=";")
    group_add = pd.read_csv("group_add.csv", sep=",")
    group_add = group_add.rename(columns = {group_add.columns[0]:'id', group_add.columns[1]: 'grp'})
    groups_all = pd.concat([groups, group_add], ignore_index=True)
    groups_all.to_csv('groups.csv') #сохраняем новые данные в файл для будущих добавлений
    
    df = pd.merge(groups_all, active_studs, 
                  left_on='id', right_on='student_id', how='inner')
    df = pd.merge(df, checks, on='student_id', how='left').fillna(0)
    metrics = df.groupby('grp').agg(
        total_users=('id', 'count'),
        paying_users=('rev', lambda x: (x > 0).sum()),
        total_revenue=('rev', 'sum')
    ).reset_index()
    
    metrics['conversion'] = metrics['paying_users'] / metrics['total_users']
    metrics['arpu'] = metrics['total_revenue'] / metrics['total_users']
    metrics['arppu'] = metrics['total_revenue']/metrics['paying_users']
    
    for index, row in metrics.iterrows():
        print(f"Группа: {row['grp']}")
        print(f"Всего пользователей: {row['total_users']}")
        print(f"Платящие пользователи: {row['paying_users']}")
        print(f"Общий доход: {row['total_revenue']:.2f}")
        print(f"Конверсия: {row['conversion'] * 100:.2f}%")
        print(f"ARPU: {row['arpu']:.2f}")
        print(f"ARPPU: {row['arppu']:.2f}")
        print("-" * 30) 
    
    return metrics

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

def visualizacion(metrics):
    plt.subplot(1,3,1)
    sns.barplot(x='grp', y='conversion', data=metrics, edgecolor='black')
    plt.title('Конверсия по группам')
    plt.xlabel('Группа')
    
    plt.subplot(1,3,2)
    sns.barplot(x='grp', y='arpu', data=metrics, edgecolor='black')
    plt.title('ARPU по группам')
    plt.xlabel('Группа')
    
    plt.subplot(1,3,3)
    sns.barplot(x='grp', y='arppu', data=metrics, edgecolor='black')
    plt.title('ARPPU по группам')
    plt.xlabel('Группа')
    plt.show()