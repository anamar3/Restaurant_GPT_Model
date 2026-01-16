# В daAudit.py – вместо да правиш 15 plt.show() – прави това:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from clean_data import clean_data
sns.set_style("whitegrid")

# 1. Разпределение на оценките
data = clean_data()
median_rate = data['rate'].median()
def save_all_plots_to_one_sheet():
    # Всички твои графики в един canvas
    fig = plt.figure(figsize=(24, 30))  # голямо платно
    gs = fig.add_gridspec(5, 3, hspace=0.4, wspace=0.3)  # 5 реда × 3 колони

    # 1. Разпределение на оценките
    ax = fig.add_subplot(gs[0, 0])
    sns.histplot(data['rate'].dropna(), bins=30, ax=ax, color='skyblue', edgecolor='black')
    ax.axvline(median_rate, color='red', linestyle='--', label=f'Медиана = {median_rate}')
    ax.set_title('Разпределение на оценките')
    ax.legend()

    # 2. Онлайн поръчка и резервация
    ax = fig.add_subplot(gs[0, 1])
    sns.countplot(x='online_order', data=data, ax=ax, palette='Set2')
    ax.set_title('Онлайн поръчка')

    ax = fig.add_subplot(gs[0, 2])
    sns.countplot(x='book_table', data=data, ax=ax, palette='Set2')
    ax.set_title('Резервация на маса')

    # 3. Boxplot-ове
    ax = fig.add_subplot(gs[1, 0])
    sns.boxplot(x='online_order', y='rate', data=data, ax=ax)
    ax.set_title('Оценка vs Онлайн поръчка')

    ax = fig.add_subplot(gs[1, 1])
    sns.boxplot(x='book_table', y='rate', data=data, ax=ax)
    ax.set_title('Оценка vs Резервация')

    # 4. Топ локации
    ax = fig.add_subplot(gs[1, 2])
    top_loc = data['location'].value_counts().head(10)
    sns.barplot(y=top_loc.index, x=top_loc.values, ax=ax, palette='coolwarm')
    ax.set_title('Топ 10 локации')

    # 5. Средна оценка по зона
    ax = fig.add_subplot(gs[2, 0])
    city_rate = data.groupby('listed_in(city)')['rate'].mean().sort_values(ascending=False).head(10)
    sns.barplot(x=city_rate.values, y=city_rate.index, ax=ax, palette='viridis')
    ax.set_title('Средна оценка по зона (топ 10)')

    # 6. Цена vs оценка
    ax = fig.add_subplot(gs[2, 1])
    sns.scatterplot(x='approx_cost', y='rate', data=data, hue='book_table', ax=ax, alpha=0.7)
    ax.set_xlim(0, 3000)
    ax.set_title('Цена за двама vs Оценка')

    # 7. Гласове vs оценка
    ax = fig.add_subplot(gs[2, 2])
    sns.scatterplot(x='votes', y='rate', data=data, ax=ax, alpha=0.6, color='purple')
    ax.set_xlim(0, 6000)
    ax.set_title('Гласове vs Оценка')

    # 8. Log_votes vs rate
    ax = fig.add_subplot(gs[3, 0])
    sns.scatterplot(x='log_votes', y='rate', data=data, hue='book_table', ax=ax, alpha=0.7)
    ax.set_title('Log(Гласове) vs Оценка')

    # 9. Брой кухни
    ax = fig.add_subplot(gs[3, 1])
    cuisine_success = data.groupby('cuisines_count')['high_rating'].mean() * 100
    sns.barplot(x=cuisine_success.index, y=cuisine_success.values, ax=ax, palette='mako')
    ax.set_title('Успех (%) според брой кухни')
    ax.set_ylim(0, 100)

    # 10. Violin plot резервация
    ax = fig.add_subplot(gs[3, 2])
    sns.violinplot(x='book_table', y='rate', data=data, ax=ax, palette='Set1')
    ax.set_title('Оценки при наличие на резервация')

    # 11–15. Останалите (можеш да добавиш още)
    # Пример:
    ax = fig.add_subplot(gs[4, 0])
    sns.countplot(x='listed_in(type)', data=data, order=data['listed_in(type)'].value_counts().index, ax=ax)
    ax.tick_params(axis='x', rotation=45)
    ax.set_title('Тип хранене')

    ax = fig.add_subplot(gs[4, 1])
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
    ax.set_title('Корелационна матрица')

    ax = fig.add_subplot(gs[4, 2])
    sns.scatterplot(x='log_cost', y='log_votes', data=data, hue='high_rating', ax=ax, alpha=0.7)
    ax.set_title('Log(цена) vs Log(гласове)')

    plt.suptitle('Restaurant GPT – Ключови зависимости (15 графики)', fontsize=20, y=0.98)
    plt.tight_layout()

    # Записваме като едно голямо изображение
    plt.savefig("all_key_plots.png", dpi=200, bbox_inches='tight')
    plt.close()

# Пусни това в края на даAudit.py
if __name__ == "__main__":
    # ... твоят код за Excel ...

    save_all_plots_to_one_sheet()  # ← създава all_key_plots.png

    # Вмъкваме в първия лист
    img = Image("all_key_plots.png")
    img.width = 1100
    img.height = 1300
    ws.add_image(img, "A30")

    wb.save("Restaurant_GPT_Data_Audit.xlsx")