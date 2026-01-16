import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.drawing.image import Image
from openpyxl.styles import Font
import matplotlib.pyplot as plt
import seaborn as sns
import io
from clean_data import clean_data


def save_all_plots_to_one_sheet():
    data = clean_data(
    )  # за тези графики използваме вече преработените и почистени данни
    median_rate = data['rate'].median()
    fig = plt.figure(figsize=(24, 30))
    gs = fig.add_gridspec(5, 3, hspace=0.4, wspace=0.3)

    # 1. Разпределение на оценките
    ax = fig.add_subplot(gs[0, 0])
    sns.histplot(data['rate'].dropna(),
                 bins=30,
                 ax=ax,
                 color='skyblue',
                 edgecolor='black')
    ax.axvline(median_rate,
               color='red',
               linestyle='--',
               label=f'Медиана = {median_rate}')
    ax.set_title('Разпределение на оценките')
    ax.legend()
    # Извод: Оценките са силно концентрирани между 3.5 и 4.2. Малко ресторанти под 3.0 или над 4.5.

    # 3. Онлайн поръчка/резервация и оценка
    ax = fig.add_subplot(gs[0, 1])
    sns.boxplot(x='online_order', y='rate', data=data, ax=ax)
    ax.set_title('Оценка vs Онлайн поръчка')

    ax = fig.add_subplot(gs[0, 2])
    sns.boxplot(x='book_table', y='rate', data=data, ax=ax)
    ax.set_title('Оценка vs Резервация')
    # Извод: Ресторантите с резервация имат значително по-висока медианна оценка (4.1 vs 3.6)
    # Онлайн поръчката не е добър предиктор за висока оценка. Тя е толкова разпространена, че
    # вече не отличава качествените от обикновените ресторанти.

    ax = fig.add_subplot(gs[1, 0])

    sns.scatterplot(
        data=data,
        x='log_cost',
        y='votes',
        hue='rate',
    )

    plt.title('Цена за двама vs брой гласове', pad=20)
    plt.xlabel('Log(цена за двама)', fontsize=13)
    plt.ylabel('Гласове')
    plt.legend(fontsize=12)

    # 6. Цена vs оценка
    ax = fig.add_subplot(gs[1, 1])
    sns.scatterplot(x='approx_cost', y='rate', data=data, hue='book_table')
    ax.set_xlim(0, 3000)
    ax.set_title('Цена за двама vs Оценка')
    # Извод: По-скъпите ресторанти почти винаги имат оценка над 4.0 и предлагат резервация
    # 7. Гласове vs оценка
    ax = fig.add_subplot(gs[1, 2])
    sns.scatterplot(x='votes', y='rate', data=data, ax=ax, alpha=0.7)
    ax.set_xlim(0, 6000)
    ax.set_title('Гласове vs Оценка')
    # Ресторанти с над 1000 гласа почти винаги са над 4.0 (популярност влече качество)

    # 8. Log_votes vs rate
    ax = fig.add_subplot(gs[2, 0])
    sns.scatterplot(x='log_votes', y='rate', data=data, hue='book_table')
    ax.set_title('Log(Гласове) vs Оценка')
    # Много ясно разделяне – висок брой гласове = висока оценка

    # 9. Брой кухни
    ax = fig.add_subplot(gs[2, 1])

    sns.boxplot(x='high_rating', y='cuisines_count', data=data)
    plt.title('Брой кухни vs Висока оценка')
    # Ресторантите с повече кухни (4–8) по-често имат висока оценка

    # 10. Violin plot резервация
    ax = fig.add_subplot(gs[2, 2])
    sns.violinplot(x='book_table', y='rate', data=data, ax=ax, palette='Set1')
    ax.set_title('Оценки при наличие на резервация')

    ax = fig.add_subplot(gs[3, 0])
    type_order = data.groupby('listed_in(type)')['rate'].mean().sort_values(
        ascending=False).index

    sns.barplot(y='listed_in(type)',
                x='rate',
                data=data,
                order=type_order,
                palette='rocket')

    plt.title('Средна оценка според типа хранене (listed_in(type))', pad=20)
    plt.xlabel('Средна оценка')
    plt.ylabel('Тип хранене')
    plt.axvline(x=data['rate'].mean(),
                color='red',
                linestyle='--',
                linewidth=2,
                label=f'Обща средна оценка ({data["rate"].mean():.2f})')

    for i, v in enumerate(
            data.groupby('listed_in(type)')['rate'].mean().sort_values(
                ascending=False)):
        plt.text(v + 0.01, i, f'{v:.2f}', va='center')

    # ресторантите с алкохол и нощен живот доминират по качество, за разлика от тези с доставки и сладкарниците.

    numeric_cols = [
        'rate', 'votes', 'approx_cost', 'cuisines_count', 'online_order',
        'book_table', 'has_dish_liked'
    ]
    corr = data[numeric_cols].corr()

    ax = fig.add_subplot(gs[3, 1])
    sns.heatmap(corr, annot=True)
    ax.set_title('Корелационна матрица')

    ax = fig.add_subplot(gs[3, 2])
    sns.scatterplot(x='log_cost',
                    y='log_votes',
                    data=data,
                    hue='high_rating',
                    ax=ax,
                    alpha=0.7)
    ax.set_title('Log(цена) vs Log(гласове)')

    plt.suptitle('Restaurant GPT – някои зависимости', fontsize=20, y=0.98)

    plt.savefig("all_key_plots.png", dpi=200, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # За кирилица в графики
    plt.rcParams['font.family'] = 'DejaVu Sans' #видяно от stackoverflow

    df = pd.read_csv('../../../../Downloads/restaurant_clf.csv')  #raw dannite
    df['rate'] = pd.to_numeric(df['rate'].astype(str).str.split('/').str[0],
                               errors='coerce')
    median_rate = df['rate'].median()
    df['high_rating'] = (df['rate'] > median_rate).astype(int)
    df['cuisines_count'] = df['cuisines'].fillna('').apply(
        lambda x: len([i for i in str(x).split(',') if i.strip()]))
    df['has_dish_liked'] = df['dish_liked'].notna().astype(int)
    df['approx_cost(for two people)'] = df[
        'approx_cost(for two people)'].str.replace(',', '').astype(float)

    summary_numeric = df.describe(include=[np.number]).T.round(2)
    summary_categorical = df.describe(include=['object', 'bool']).T
    print(summary_numeric)
    summary_df = pd.concat([summary_numeric, summary_categorical])
    df['online_order'] = (df['online_order'] == 'Yes').astype(int)
    df['book_table'] = (df['book_table'] == 'Yes').astype(int)

    summary_df['missing_count'] = df.isnull().sum()
    summary_df['missing_%'] = (df.isnull().sum() / len(df) * 100).round(2)
    summary_df['comments'] = [
        "dish_liked има над 54% липсващи стойности,навежда ни на мисълта, че колоната не е надеждна. Ще използваме has_dish_liked (1 ако има поне едно харесвано ястие).",
        "rate има около 19% липсващи. Премахваме тези редове(и редовете с NEW). Cъздаваме целевата променлива high_rating (rate > медиана).",
        "аpprox_cost(for two people) Има малко липсващи. Почистваме запетаи, конвертираме в numeric. Ще ползваме и логаритимирана версия заради силно изкривено разпределение.",
        "cuisines има 0.3% липсващи, създаваме cuisines_count. По ценно ще ни е, ако имаме такава колона, тъй като има над 2500 unique",
        "reviews_list е текстова колона с над 40% празни. Няма да я ползваме директно твърде тежка ще е за обработка.",
        "menu_item има над 80% празни по-скоро ще игнорираме.",
        "url e уникалen идентификатор и не носи никаква сила,затова премахваме.",
        "name e уникалen идентификатор и не носи никаква сила,затова премахваме.",
        "phone e уникалen идентификатор и не носи никаква сила,затова премахваме.",
        "address има прекалено много уникални, евентурално с delivery би имало някаква корелация."
        "location има 93 уникални, без липсващи влече OneHot (може би ще влияе силно на оценката).",
        "rest_type 93 уникални, тоест отново OneHot",
        "listed_in(type) има цамо 7 категории - имаме отново OneHot.",
        "listed_in(city)има 30 категории - OneHot.",
        "book_table  ще преврънем в 0/1(бинарна колона). Може би силна корелация с висока оценка.",
        "online_order бинарна т.е превръщаме в 0/1. Евентуално слаба корелация."
        "votes има силно изкривено разпределение, бихме го направили на log_votes.",
        "cuisines_count създадена от нас(колкото повече кухни, толкова по-висока оценка).",
        "has_dish_liked създадена от нас. Много силен предиктор.",
        "high_rating е целева променлива (1 = оценка над медианата 3.9). Класът е почти балансиран.",
        "", "", ""
    ]

    sns.pairplot(df, hue='high_rating', palette={0: 'red', 1: 'green'})
    plt.savefig("pairplot.png", dpi=200, bbox_inches='tight')
    plt.close()

    # Excel отчет

    wb = Workbook()
    ws = wb.active
    ws.title = "Data Summary"

    for r in dataframe_to_rows(summary_df, index=True, header=True):
        ws.append(r)

    img = Image("pairplot.png")
    img.width = 900
    img.height = 800
    ws.add_image(img, "A30")

    for col in df.columns:
        safe_name = str(col)[:31].replace('/', '').replace('(', '').replace(
            ')', '').replace(':', '')

        sheet = wb.create_sheet(title=safe_name)
        print(col)
        sheet['A1'] = f"Колона: {col}"
        sheet['A1'].font = Font(size=14, bold=True)

        sheet['A3'] = "Брой редове"
        sheet['B3'] = len(df)
        sheet['A4'] = "Липсващи"
        sheet['B4'] = df[col].isnull().sum()
        sheet['A5'] = "Уникални"
        sheet['B5'] = df[col].nunique()

        sheet['A7'] = "Топ 20 стойности"
        top20 = df[col].value_counts(dropna=False).head(20)
        for i, (val, cnt) in enumerate(top20.items(), start=9):
            sheet[f'A{i}'] = str(val) if pd.notna(val) else "NaN"
            sheet[f'B{i}'] = cnt

        plt.figure(figsize=(9, 5))

        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 12:
            plt.hist(df[col].dropna(),
                     bins=30,
                     color='skyblue',
                     edgecolor='black',
                     alpha=0.8)
            plt.title(f'Хистограма: {col}')
        else:
            counts = df[col].value_counts(dropna=False).head(20)
            bars = plt.bar(range(len(counts)), counts.values)
            plt.title(f'Топ 20: {col}')
            plt.xticks(range(len(counts)), [str(x)[:15] for x in counts.index],
                       rotation=45,
                       ha='right')  #za da se vijdat po-dobre
            for bar in bars:
                h = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., h + h * 0.01,
                         f'{int(h)}')  #naglasqme da se chete

        plt.tight_layout()
        img_data = io.BytesIO() #не ми работеше с Image, генерирано от Grok)
        plt.savefig(img_data, dpi=180, bbox_inches='tight')
        plt.close()
        img_data.seek(0)
        sheet.add_image(Image(img_data), "D3")

    save_all_plots_to_one_sheet()
    img = Image("all_key_plots.png")
    img.width = 1100
    img.height = 1300
    ws.add_image(img, "O30")
    wb.save("Restaurant_GPT_Data_Audit.xlsx")
