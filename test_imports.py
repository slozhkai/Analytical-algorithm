try:
    import selenium
    import pandas
    from bs4 import BeautifulSoup
    import nltk
    from webdriver_manager.chrome import ChromeDriverManager
    print("Все модули успешно импортированы!")
except ImportError as e:
    print(f"Ошибка импорта: {e}")