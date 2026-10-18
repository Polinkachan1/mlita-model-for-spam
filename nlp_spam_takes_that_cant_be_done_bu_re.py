import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_union
from sklearn.metrics import classification_report, accuracy_score

from masks import apply_masks

# В датасете номера и ссылки встречаются почти только в спаме, поэтому сами токены-маски
# становятся "шорткатом" для модели. Мы убираем их из входа классификатора,
# чтобы решение принималось по словам, а не по наличию номера.
MASK_TOKENS = ("maskedphone", "maskedurl")


def prepare(text):
    text = apply_masks(text)  # идемпотентно: уже замаскированный текст не меняется
    for tok in MASK_TOKENS:
        text = text.replace(tok, " ")
    return " ".join(text.split())


def build_model():
    # Символьные n-граммы (3-5) + словесные признаки (1-2 слова)
    features = make_union(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=3000),
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2),
    )
    clf = LogisticRegression(
        class_weight="balanced",  # данных "ham" намного больше, балансируем
        C=10,
        max_iter=1000,
        random_state=42,
    )
    return Pipeline([("tfidf", features), ("clf", clf)])


def explain(model, text, top=8):
    """Показывает признаки, которые сильнее всего толкают сообщение к спаму."""
    tfidf = model.named_steps["tfidf"]
    coef = model.named_steps["clf"].coef_[0]
    names = tfidf.get_feature_names_out()
    contrib = tfidf.transform([text]).toarray()[0] * coef
    print("   Что сильнее всего толкает к спаму:")
    shown = 0
    for i in np.argsort(contrib)[::-1][:top]:
        if contrib[i] > 0:
            print(f"     {names[i].split('__', 1)[-1]!r}: {contrib[i]:+.2f}")
            shown += 1
    if shown == 0:
        print("     (нет признаков в сторону спама)")


def predict_message(model, raw_text, title):
    masked = apply_masks(raw_text)
    model_input = prepare(raw_text)
    p_spam = model.predict_proba([model_input])[0, 1]
    verdict = "Это мошенники!" if p_spam >= 0.5 else "Все окей, это не мошенники!"

    print(f"1. Входной текст ({title}):\n   \"{raw_text}\"\n")
    print(f"2. Результат работы регулярных выражений:\n   \"{masked}\"\n")
    print(f"3. Вход модели (без токенов-масок):\n   \"{model_input}\"\n")
    print(f"   Финальный статус: {verdict}")
    print(f"   P(spam) = {p_spam * 100:.1f}%  |  P(ham) = {(1 - p_spam) * 100:.1f}%")
    explain(model, model_input)
    print()


def train_and_test_classifier(csv_file):
    try:
        df = pd.read_csv(csv_file, encoding="utf-8")
    except FileNotFoundError:
        print(f"[Ошибка]: Файл {csv_file} не найден! Запустите сначала предыдущие этапы.")
        return None

    df["text"] = df["text"].fillna("").astype(str).map(prepare)
    y = df["is_spam"].astype(bool)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Общая точность на тесте: {accuracy_score(y_test, y_pred) * 100:.1f}%")
    print("\nОтчет по метрикам:")
    print(classification_report(y_test, y_pred, target_names=["Обычное (Ham)", "Спам (Spam)"]))
    return model


if __name__ == "__main__":
    model = train_and_test_classifier("great_output.csv")
    if model is not None:
        predict_message(
            model,
            "URGENT! Call 09061701461 right now to claim your 1000 prize reward! Click www.dbuk.net",
            "Атака мошенников",
        )
        predict_message(
            model,
            "Hi! Here is the doctor's phone number you asked for: 08001234567. He is a great specialist",
            "Обычное сообщение",
        )
    # Этот шаг показывает границу применимости регулярных языков (Тип 3 по Хомскому):
    # конечный автомат не обладает памятью и не может распознать глобальный смысл контекста.
    # Поэтому мы разбиваем текст на фиксированные цепочки символов длиной от 3 до 5 (n-граммы)
    # и поручаем решение статистической модели.