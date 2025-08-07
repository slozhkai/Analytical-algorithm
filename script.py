from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from collections import Counter

# Скачиваем ресурсы для NLP
nltk.download('vader_lexicon')
nltk.download('stopwords')

# Настройки браузера
chrome_options = Options()
chrome_options.add_argument("--headless")  # Фоновый режим
chrome_options.add_argument("--disable-blink-features=AutomationControlled")
chrome_options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36")
chrome_options.add_argument("--window-size=1920,1080")

# Автоматическая установка chromedriver
from webdriver_manager.chrome import ChromeDriverManager

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)


def extract_keywords(text, lang='russian'):
    """Извлекает ключевые слова из текста отзыва"""
    stop_words = set(stopwords.words(lang))
    words = re.findall(r'\b\w{4,}\b', text.lower())
    filtered_words = [word for word in words if word not in stop_words and not word.isdigit()]
    return list(set(filtered_words))


def parse_yandex_reviews(company_name, city="Ростов-на-Дону", max_branches=20, max_reviews_per_branch=200):
    """
    Парсит отзывы с Яндекс Карт для сети филиалов

    Параметры:
    company_name - название сети
    city - город поиска
    max_branches - максимальное количество филиалов для анализа
    max_reviews_per_branch - максимальное количество отзывов на филиал
    """
    # Шаг 1: Поиск филиалов
    search_url = f"https://yandex.ru/maps/?text={company_name} {city}"
    driver.get(search_url)
    time.sleep(3)

    # Ожидаем загрузки результатов
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div.search-business-snippet-view"))
    )

    # Извлекаем ссылки на филиалы
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    branch_cards = soup.select('div.search-business-snippet-view')
    branch_links = []

    for card in branch_cards[:max_branches]:
        link = card.select_one('a.search-business-snippet-view__link-overlay')
        if link and 'href' in link.attrs:
            full_url = "https://yandex.ru" + link['href']
            branch_links.append(full_url)

    print(f"Найдено филиалов: {len(branch_links)}")

    # Сбор данных
    all_reviews = []
    sia = SentimentIntensityAnalyzer()

    for branch_url in branch_links:
        print(f"Обработка филиала: {branch_url}")
        driver.get(branch_url)
        time.sleep(2)

        # Открываем вкладку с отзывами
        try:
            reviews_tab = WebDriverWait(driver, 15).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "div[data-section-id='reviews']"))
            )
            reviews_tab.click()
            time.sleep(2)
        except Exception as e:
            print(f"Не удалось найти вкладку отзывов: {branch_url} ({str(e)})")
            continue

        # Пролистываем страницу для загрузки отзывов
        last_height = driver.execute_script("return document.body.scrollHeight")
        loaded_reviews = 0
        scroll_attempts = 0

        while loaded_reviews < max_reviews_per_branch and scroll_attempts < 20:
            # Прокрутка вниз
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.5)

            # Проверка новых элементов
            new_reviews = driver.find_elements(By.CSS_SELECTOR, "div.business-review-view__info")
            new_count = len(new_reviews)

            if new_count > loaded_reviews:
                loaded_reviews = new_count
                scroll_attempts = 0
            else:
                scroll_attempts += 1

            # Проверяем, достигли ли мы конца
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        # Парсинг загруженных отзывов
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        reviews = soup.select('div.business-review-view__info')
        print(f"  Найдено отзывов: {len(reviews)}")

        for review in reviews:
            try:
                # Извлечение данных
                author = review.select_one('div.business-review-view__author').text.strip() if review.select_one(
                    'div.business-review-view__author') else "Аноним"

                rating_element = review.select_one('div.business-rating-badge-view__stars')
                rating = float(
                    re.search(r'(\d+\.?\d?)', rating_element['aria-label']).group(1)) if rating_element else 0.0

                date = review.select_one('span.business-review-view__date').text.strip() if review.select_one(
                    'span.business-review-view__date') else ""

                text_element = review.select_one('span.business-review-view__body-text')
                text = text_element.text.strip() if text_element else ""

                # Ответ компании
                response = review.select_one('div.business-review-view__response-text')
                response_text = response.text.strip() if response else ""

                # Ключевые моменты (эмодзи)
                features = [img['alt'] for img in review.select('div.business-review-view__features img')]

                # Анализ текста
                keywords = extract_keywords(text) if text else []
                sentiment = sia.polarity_scores(text)['compound'] if text else 0.0

                all_reviews.append({
                    'Филиал': branch_url,
                    'Автор': author,
                    'Рейтинг': rating,
                    'Дата': date,
                    'Отзыв': text,
                    'Ответ': response_text,
                    'Особенности': ", ".join(features),
                    'Ключевые слова': ", ".join(keywords),
                    'Тональность': sentiment
                })
            except Exception as e:
                print(f"Ошибка парсинга отзыва: {str(e)}")

    return pd.DataFrame(all_reviews)


# Аналитическое резюме
def generate_summary(df):
    """Генерирует аналитическое резюме по отзывам"""
    if df.empty:
        return "Нет данных для анализа"

    summary = {
        "Общее количество отзывов": len(df),
        "Средний рейтинг": df['Рейтинг'].mean().round(2),
        "Средняя тональность": df['Тональность'].mean().round(3),
        "Распределение оценок": df['Рейтинг'].value_counts().sort_index(ascending=False).to_dict()
    }

    # Топ-10 ключевых слов
    all_keywords = []
    for keywords in df['Ключевые слова'].str.split(', '):
        all_keywords.extend(keywords)

    keyword_counts = Counter(all_keywords)
    top_keywords = [word for word, _ in keyword_counts.most_common(10) if word]

    summary["Топ-10 ключевых слов"] = top_keywords

    # Анализ особенностей
    all_features = []
    for features in df['Особенности'].str.split(', '):
        if features:
            all_features.extend(features)

    feature_counts = Counter(all_features)
    top_features = [feature for feature, _ in feature_counts.most_common(5) if feature]

    summary["Топ-5 особенностей"] = top_features

    # Примеры позитивных и негативных отзывов
    if len(df) > 1:
        positive_review = df[df['Тональность'] == df['Тональность'].max()].iloc[0]['Отзыв'][:150] + "..."
        negative_review = df[df['Тональность'] == df['Тональность'].min()].iloc[0]['Отзыв'][:150] + "..."

        summary["Пример позитивного отзыва"] = positive_review
        summary["Пример негативного отзыва"] = negative_review

    return summary


# Пример использования
if __name__ == "__main__":
    # Настройки парсера
    COMPANY = "Sushibox"
    CITY = "Ростов-на-Дону"
    MAX_BRANCHES = 20
    MAX_REVIEWS = 300

    print(f"Начинаем парсинг {COMPANY} в {CITY}...")

    # Запуск парсера
    reviews_df = parse_yandex_reviews(
        company_name=COMPANY,
        city=CITY,
        max_branches=MAX_BRANCHES,
        max_reviews_per_branch=MAX_REVIEWS
    )

    # Сохранение в CSV
    filename = f"{COMPANY}_{CITY}_reviews_{time.strftime('%Y%m%d_%H%M')}.csv"
    reviews_df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"\nСохранено {len(reviews_df)} отзывов в файл {filename}")

    # Генерация и сохранение аналитического резюме
    if not reviews_df.empty:
        print("\nАналитическое резюме:")
        summary = generate_summary(reviews_df)

        for key, value in summary.items():
            print(f"- {key}: {value}")

        # Сохраняем резюме в текстовый файл
        with open(f"{COMPANY}_{CITY}_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"Аналитический отчет по отзывам {COMPANY} в {CITY}\n")
            f.write(f"Дата создания: {time.strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"Всего отзывов: {len(reviews_df)}\n\n")

            for key, value in summary.items():
                f.write(f"{key}:\n{value}\n\n")

    driver.quit()