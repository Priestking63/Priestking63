import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.common.action_chains import ActionChains
import argparse
import logging
import re
import pandas as pd
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def random_delay(min_seconds=1, max_seconds=5):
    """Случайная задержка между запросами"""
    delay = random.uniform(min_seconds, max_seconds)
    time.sleep(delay)
    return delay


def get_random_user_agent():
    """Возвращает случайный User-Agent"""
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0",
    ]
    return random.choice(user_agents)


def random_mouse_movements(driver):
    """Совершает случайные движения мышкой по странице"""
    try:
        actions = ActionChains(driver)
        
        # Получаем размеры окна
        window_size = driver.get_window_size()
        width = window_size['width']
        height = window_size['height']
        
        # Совершаем несколько случайных движений
        num_movements = random.randint(2, 5)
        
        for _ in range(num_movements):
            # Случайные координаты в пределах видимой области
            x = random.randint(100, width - 100)
            y = random.randint(100, height - 200)  # оставляем место для шапки
            
            # Перемещаем мышь с небольшой задержкой
            actions.move_by_offset(x, y)
            actions.pause(random.uniform(0.1, 0.3))
            
            # Иногда прокручиваем страницу
            if random.random() < 0.3:  # 30% вероятность прокрутки
                scroll_amount = random.randint(100, 500)
                driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
                actions.pause(random.uniform(0.2, 0.5))
        
        # Возвращаем мышь в начало
        actions.move_by_offset(-width//2, -height//2)
        actions.perform()
        
        logger.debug("Выполнены случайные движения мышкой")
        
    except Exception as e:
        logger.debug(f"Ошибка при движении мышкой: {str(e)}")


def random_clicks(driver):
    """Совершает случайные клики по неинтерактивным элементам"""
    try:
        actions = ActionChains(driver)
        
        # Находим все видимые элементы на странице (кроме интерактивных)
        all_elements = driver.find_elements(By.CSS_SELECTOR, "*")
        
        # Фильтруем элементы - выбираем только видимые и неинтерактивные (div, span, img и т.д.)
        non_interactive_elements = []
        for element in all_elements:
            try:
                if (element.is_displayed() and 
                    element.tag_name not in ['a', 'button', 'input', 'select', 'textarea'] and
                    random.random() < 0.1):  # 10% chance to include element
                    non_interactive_elements.append(element)
            except:
                continue
        
        # Совершаем 1-2 случайных клика
        if non_interactive_elements:
            num_clicks = random.randint(1, 2)
            for _ in range(num_clicks):
                if non_interactive_elements:
                    element = random.choice(non_interactive_elements)
                    try:
                        # Кликаем с небольшой задержкой перед кликом
                        actions.move_to_element(element)
                        actions.pause(random.uniform(0.1, 0.3))
                        actions.click()
                        actions.pause(random.uniform(0.1, 0.3))
                        logger.debug(f"Случайный клик по элементу <{element.tag_name}>")
                    except:
                        continue
        
        actions.perform()
        
    except Exception as e:
        logger.debug(f"Ошибка при случайных кликах: {str(e)}")


def human_like_interaction(driver):
    """Имитирует человеческое поведение: движения мышкой и клики"""
    # Случайные движения мышкой
    if random.random() < 0.7:  # 70% вероятность движений
        random_mouse_movements(driver)
    
    # Случайные клики
    if random.random() < 0.4:  # 40% вероятность кликов
        random_clicks(driver)
    
    # Случайная прокрутка
    if random.random() < 0.6:  # 60% вероятность прокрутки
        scroll_amount = random.randint(200, 800)
        driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
        time.sleep(random.uniform(0.2, 0.5))


def parse_apartment_details(driver, url):
    driver.get(url)
    delay = random_delay(2, 4)  # Задержка после загрузки страницы объявления
    logger.debug(f"Задержка {delay:.2f} сек после загрузки объявления")
    
    # Человекоподобное взаимодействие после загрузки страницы
    human_like_interaction(driver)
    
    data = {}

    # Парсинг цены из атрибута content
    try:
        price_element = WebDriverWait(driver, 3).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'span[itemprop="price"]'))
        )
        price_content = price_element.get_attribute("content")

        if price_content:
            data["Цена"] = int(price_content)
            logger.info(f"Найдена цена: {data['Цена']} руб.")
        else:
            data["Цена"] = None
            logger.warning("Атрибут content пустой")

    except (TimeoutException, NoSuchElementException):
        logger.warning("Элемент с ценой не найден")
        data["Цена"] = None
    except Exception as e:
        logger.error(f"Ошибка при парсинге цены: {str(e)}")
        data["Цена"] = None

    # Парсинг остальных параметров
    try:
        WebDriverWait(driver, 3).until(
            EC.presence_of_element_located(
                (By.CLASS_NAME, "params__paramsList___XzY3MG")
            )
        )

        # Человекоподобное взаимодействие перед парсингом деталей
        human_like_interaction(driver)

        # Используем JavaScript для получения только видимого текста
        script = """
        var results = [];
        var uls = document.getElementsByClassName('params__paramsList___XzY3MG');
        for (var i = 0; i < uls.length; i++) {
            var lis = uls[i].getElementsByTagName('li');
            for (var j = 0; j < lis.length; j++) {
                var li = lis[j];
                // Получаем только видимый текст
                var text = li.innerText || li.textContent;
                // Удаляем не-breaking spaces и trim
                text = text.replace(/\\u00A0/g, ' ').trim();
                // Проверяем, не пустой ли элемент
                if (text && !/^[\\s\\xa0]*$/.test(text)) {
                    results.push(text);
                }
            }
        }
        return results;
        """

        texts = driver.execute_script(script)
        logger.info(f"Найдено {len(texts)} параметров для объявления")

        for text in texts:
            if ":" in text:
                parts = text.split(":", 1)
                param = parts[0].strip()
                value = parts[1].strip()

                if param and value:
                    data[param] = value
            else:
                logger.warning(f"Нет ':' в тексте '{text}'")

        logger.info(f"Успешно собрано {len(data)} параметров")

    except TimeoutException:
        logger.error(f"Таймаут при загрузке деталей")
    except Exception as e:
        logger.error(f"Ошибка при парсинге деталей: {str(e)}")

    # Добавляем ссылку в данные
    data["Ссылка"] = url

    return data


def get_city_from_url(url):
    """Извлекает название города из URL"""
    pattern = r"https://www\.avito\.ru/([^/]+)/kvartiry/prodam"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return "unknown_city"


def process_city(driver, base_url, num_pages, city_name):
    """Обрабатывает один город и возвращает DataFrame"""
    all_data = []

    for page in range(1, num_pages + 1):
        page_url = f"{base_url}?p={page}"
        logger.info(f"Обработка страницы {page}/{num_pages} для города {city_name}")
        
        # Случайная задержка перед загрузкой страницы
        delay_before_page = random_delay(1, 3)
        logger.debug(f"Задержка {delay_before_page:.2f} сек перед загрузкой страницы")
        
        driver.get(page_url)

        try:
            # Ожидание загрузки списка объявлений
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, 'a[data-marker="item-title"]')
                )
            )

            # Человекоподобное взаимодействие после загрузки страницы
            human_like_interaction(driver)

            # Случайная задержка после загрузки страницы
            delay_after_page = random_delay(1, 2)
            logger.debug(f"Задержка {delay_after_page:.2f} сек после загрузки страницы")

            # Сбор ссылок на объявления
            link_elements = driver.find_elements(
                By.CSS_SELECTOR, 'a[data-marker="item-title"]'
            )
            links = []
            for elem in link_elements:
                href = elem.get_attribute("href")
                if href and "kvartiry" in href:
                    links.append(href)

            # Удаление возможных дубликатов
            links = list(set(links))

            logger.info(f"На странице {page} найдено {len(links)} объявлений")

            for i, link in enumerate(links, 1):
                logger.info(f"Парсинг объявления {i}/{len(links)} для города {city_name}")
                details = parse_apartment_details(driver, link)
                if details:
                    all_data.append(details)
                
                # Случайная задержка между объявлениями
                if i < len(links):  # Не ждем после последнего объявления
                    delay_between_ads = random_delay(1, 3)
                    logger.debug(f"Задержка {delay_between_ads:.2f} сек между объявлениями")

            logger.info(f"Страница {page} обработана, собрано {len(links)} объявлений")

        except TimeoutException:
            logger.error(f"Таймаут при загрузке страницы {page} для города {city_name}")
            continue

    # Создаем DataFrame для города
    if all_data:
        df = pd.DataFrame(all_data)

        # Упорядочиваем колонки: сначала основные, потом остальные
        preferred_columns = ["Цена", "Ссылка"]
        other_columns = [col for col in df.columns if col not in preferred_columns]
        ordered_columns = preferred_columns + other_columns

        # Переупорядочиваем DataFrame
        df = df[ordered_columns]

        return df
    else:
        logger.warning(f"Для города {city_name} данные не собраны")
        return pd.DataFrame()


def process_single_city(url, num_pages):
    """Обрабатывает один город в отдельном потоке"""
    # Создаем отдельный драйвер для этого потока со случайным User-Agent
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--blink-settings=imagesEnabled=false")
    
    # Случайный User-Agent
    user_agent = get_random_user_agent()
    options.add_argument(f"--user-agent={user_agent}")
    
    driver = webdriver.Chrome(options=options)
    try:
        city_name = get_city_from_url(url)
        logger.info(f"Начинаем обработку города: {city_name} с User-Agent: {user_agent[:50]}...")
        
        df_city = process_city(driver, url, num_pages, city_name)
        
        if not df_city.empty:
            csv_filename = f"avito_apartments{city_name}.csv"
            df_city.to_csv(csv_filename, index=False, encoding="utf-8")
            logger.info(f"Данные для города {city_name} сохранены в {csv_filename}")
            logger.info(f"Собрано {len(df_city)} объявлений для города {city_name}")
            
            # Статистика по ценам
            if "Цена" in df_city.columns:
                successful_prices = df_city["Цена"].notna().sum()
                logger.info(f"Успешно извлечено цен: {successful_prices} из {len(df_city)} для города {city_name}")
            return True
        else:
            logger.warning(f"Нет данных для сохранения города {city_name}")
            return False
            
    except Exception as e:
        logger.error(f"Ошибка при обработке города {get_city_from_url(url)}: {str(e)}")
        return False
    finally:
        driver.quit()


def main(urls, num_pages, max_workers=3):
    """Запускает многопоточный парсинг городов"""
    
    logger.info(f"Запуск многопоточного парсинга для {len(urls)} городов с {max_workers} потоками")
    
    # Запускаем парсинг всех городов параллельно
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Создаем задачи для каждого города
        future_to_city = {
            executor.submit(process_single_city, url, num_pages): get_city_from_url(url) 
            for url in urls
        }
        
        # Ожидаем завершения всех задач
        completed = 0
        failed = 0
        
        for future in as_completed(future_to_city):
            city_name = future_to_city[future]
            try:
                result = future.result()
                if result:
                    completed += 1
                    logger.info(f"Город {city_name} успешно обработан")
                else:
                    failed += 1
                    logger.warning(f"Город {city_name} обработан с ошибками")
            except Exception as e:
                failed += 1
                logger.error(f"Неожиданная ошибка при обработке города {city_name}: {str(e)}")
    
    logger.info(f"Парсинг завершен. Успешно: {completed}, С ошибками: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Парсер квартир с Avito")
    parser.add_argument(
        "--pages", type=int, default=50, help="Количество страниц для парсинга"
    )
    parser.add_argument(
        "--threads", type=int, default=3, help="Количество потоков для параллельного парсинга"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Включить отладочные сообщения о задержках"
    )
    args = parser.parse_args()
    
    # Включаем отладочные сообщения если нужно
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Список URL для парсинга
    urls_to_parse = [
        "https://www.avito.ru/moskva/kvartiry/prodam",
        "https://www.avito.ru/sankt-peterburg/kvartiry/prodam",
        "https://www.avito.ru/novosibirsk/kvartiry/prodam",
        "https://www.avito.ru/ekaterinburg/kvartiry/prodam",
        'https://www.avito.ru/kazan/kvartiry/prodam',
        'https://www.avito.ru/nizhniy_novgorod/kvartiry/prodam'
    ]
    
    main(urls_to_parse, args.pages, args.threads)