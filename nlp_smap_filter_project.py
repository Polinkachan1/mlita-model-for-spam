import html
import pandas as pd


def preprocessing_text(file, new_file):
    # spam.csv хранится в cp1252, поэтому символ £ читается правильно
    df = pd.read_csv(file, encoding="cp1252", encoding_errors="replace", usecols=[0, 1])
    df.columns = ["label", "text"]

    df["is_spam"] = df["label"].str.strip().str.lower() == "spam"
    df["text"] = df["text"].astype(str).map(html.unescape).str.strip()

    df = df[["is_spam", "text"]].drop_duplicates()
    df.to_csv(new_file, index=False, encoding="utf-8")
    print(f"Сохранено строк: {len(df)} -> {new_file}")


if __name__ == "__main__":
    preprocessing_text("spam.csv", "filtered_smap.csv")
    # Любой формальный язык строится над конечным алфавитом (Σ). Исходный файл содержал
    # синтаксические аномалии (неверная кодировка и HTML-сущности вроде '&amp;').
    # На этом этапе мы очистили алфавитный шум, приведя цепочки символов к каноническому виду.