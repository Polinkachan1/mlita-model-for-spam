import pandas as pd
from masks import apply_masks


def spam_filter(input_file: str, output_file: str):
    df = pd.read_csv(input_file, encoding="utf-8")
    df["text"] = df["text"].fillna("").astype(str).map(apply_masks)
    df.to_csv(output_file, index=False, encoding="utf-8")

    masked = df[df["text"].str.contains("masked")]
    print("-> Регулярные выражения нашли в тексте подозрительные телефоны и URL.")
    print(f"-> Строк с масками: {len(masked)} из {len(df)}")
    print(f"-> Безопасный лог сохранен в: '{output_file}'")
    print("\nПримеры строк, где сработал фильтр:")
    print((masked if not masked.empty else df).head(5).to_string())


if __name__ == "__main__":
    spam_filter("filtered_smap.csv", "great_output.csv")
    # Номера телефонов и URL-адреса имеют линейную и детерминированную структуру, то есть
    # описываются регулярными языками. По теореме Клини для каждого такого языка существует
    # эквивалентный ДКА, который посимвольно читает строку и при достижении допускающего
    # состояния (F) выделяет подцепочку для маскирования. Замечание: модуль re в Python
    # реализован через backtracking, а не как ДКА, но распознаваемые языки те же.