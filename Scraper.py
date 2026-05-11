import os
import requests
from bs4 import BeautifulSoup, Tag
from urllib.parse import urljoin, urlparse, urldefrag
from fpdf import FPDF # Бібліотека для створення PDF
from pypdf import PdfReader, PdfWriter
import time

DEFAULT_URL = "https://lnu.edu.ua/about/documents/"
DEFAULT_SAVE_DIR = "Data/A_pdfs"
DEFAULT_DEPTH = 3
EXTENSIONS = {'.pdf'}
FONT_PATH = "Fonts/LiberationSans-Regular.ttf"

def _save_text_to_pdf(title: str, text: str, filename: str, save_dir: str, url: str) -> None:
    """Створює PDF файл якщо на сторінці простий текст"""
    try:
        pdf = FPDF()
        pdf.add_page()
        
        if os.path.exists(FONT_PATH):
            pdf.add_font("FreeSans", "", FONT_PATH) # Реєструю шрифт під назвою "FreeSans"
            font_name = "FreeSans"
        else:
            font_name = "Arial"
            print(f"Файл шрифту {FONT_PATH} не знайдено в папці зі скриптом, використовується Arial")

        # Додаю заголовок
        pdf.set_font(font_name, size=12)
        pdf.multi_cell(0, 10, text=title, align="C")
        pdf.ln(10)
        
        # Додаю основний текст
        pdf.set_font(font_name, size=10)
        pdf.multi_cell(0, 7, text=text)
        
        pdf.output(os.path.join(save_dir, filename))
        _add_source_url(os.path.join(save_dir, filename), url)
        print(f"Створено PDF з тексту: {filename}")
    except Exception as e:
        print(f"Не вдалося створити PDF '{filename}': {e}")

def _add_source_url(file_path: str, url: str) -> None:
    """Записує URL джерела в метадані PDF"""
    try:
        reader = PdfReader(file_path, strict=False)
        writer = PdfWriter()
        writer.append(reader)
        writer.add_metadata({"/Source": url})
        with open(file_path, 'wb') as f:
            writer.write(f)
    except Exception as e:
        print(f"Не вдалося записати метадані ({os.path.basename(file_path)}): {e}")

def _get_content_block(soup: BeautifulSoup) -> Tag | None:
    """Шукає основний блок з контентом"""
    # Перебираю можливі контейнери — беру перший який знайду
    result = (
        soup.find('div', class_='page-content')
        or soup.find('article')
        or soup.find('main')
    )
    if isinstance(result, Tag):
        return result
    else:
        return None

def _download_file(url: str, save_dir: str, stats: dict) -> None:
    """Завантажує файл за прямим посиланням"""
    try:
        # Витягую ім'я файлу з URL
        filename = os.path.basename(urlparse(url).path)
        if not filename:
            print(f"Не вдалося визначити ім'я файлу: {url}")
            return

        file_path = os.path.join(save_dir, filename)

        # Пропускаю якщо файл вже є
        if os.path.exists(file_path):
            print(f"Вже існує: {filename}")
            stats['skipped'] += 1
            return

        # Завантажую файл частинами
        response = requests.get(url, stream=True, timeout=20)
        response.raise_for_status() # кидає виняток якщо помилка
        
        with open(file_path, 'wb') as f:
            size = 0
            for chunk in response.iter_content(chunk_size=100000):
                size += len(chunk)
                f.write(chunk)
        
        if size < 1000:
            print(f"Пропущено файл (занадто малий): {filename}")
            os.remove(file_path)
            stats['skipped'] += 1
            return
        else:
            print(f"Завантажено: {filename}")

        if file_path.lower().endswith('.pdf'):
            _add_source_url(file_path, url)
        stats['downloaded'] += 1
    except Exception as e:
        print(f"Помилка завантаження {url}: {e}")
        stats['errors'] += 1

def _scrape_page(url: str = DEFAULT_URL, save_dir: str = DEFAULT_SAVE_DIR,
                depth: int = DEFAULT_DEPTH, 
                visited: set | None = None,
                stats: dict | None = None) -> None:
    """
    Рекурсивно сканує сторінку і завантажує документи.

    url:      Стартова URL-адреса.
    save_dir: Папка для збереження файлів.
    depth:    Глибина рекурсії.
    visited:  Множина вже відвіданих URL.
    """
    if visited is None:
        visited = set()

    if stats is None:
        stats = {'downloaded': 0, 'skipped': 0, 'errors': 0,
                 'pages': 0}

    # Нормалізую URL — прибираю #anchor і ?query
    url, _ = urldefrag(url)
    url = urlparse(url)._replace(query="").geturl()

    # Пропускаю якщо вже відвідував або досяг ліміту глибини
    if url in visited or depth <= 0:
        return

    visited.add(url)
    stats['pages'] += 1
    print(f"\nСканування сторінки: {url} (depth={depth})")
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Шукаю саме в контенті
        content_area = _get_content_block(soup)

        if not content_area:
            print(f"Блок контенту не знайдено: {url}")
            return

        links = content_area.find_all('a', href=True)
        file_links_found = 0

        for link in links:
            href = link['href']
            # Нормалізую URL — прибираю #anchor і ?query
            full_url, _ = urldefrag(urljoin(url, href))
            full_url = urlparse(full_url)._replace(query="").geturl()
            parsed_url = urlparse(full_url)

            if any(parsed_url.path.lower().endswith(ext) for ext in EXTENSIONS):
                # Знайшов файл — завантажую
                _download_file(full_url, save_dir, stats)
                file_links_found += 1
            elif urlparse(url).netloc == parsed_url.netloc and depth > 1:
                # Знайшов посилання на той самий домен — йду глибше
                ext = os.path.splitext(parsed_url.path.lower())[1]
                if ext == '' or ext in {'.html', '.htm'}:
                    _scrape_page(full_url, save_dir=save_dir, depth=depth - 1, visited=visited, stats=stats)
                    time.sleep(0.5)

        # Зберігаю сторінку як PDF тільки якщо файлів взагалі не знайдено
        if file_links_found == 0:
            if soup.title:
                page_title = soup.title.string
            else: 
                page_title = "Untitled"

            page_text = content_area.get_text(separator='\n', strip=True)
            if len(page_text) < 50:
                print(f"Пропущено сторінку (замало тексту): {url}")
                stats['skipped'] += 1
            else:
                safe_name = "".join(c for c in page_title if c.isalnum() or c in (' ', '_', '-')).strip() # type: ignore
                if not safe_name: 
                    safe_name = "page_" + str(int(time.time()))
                 
                _save_text_to_pdf(page_title, page_text, f"{safe_name}.pdf", save_dir, url) # type: ignore
                stats['downloaded'] += 1
            

    except Exception as e:
        print(f"Помилка при обробці сторінки {url}: {e}")
        stats['errors'] += 1


def _print_summary(stats: dict, save_dir: str) -> None:
    print("\n\n" + "=" * 80)
    print("Сканування завершено")
    print("=" * 80)
    print(f"Сторінок відвідано   : {stats['pages']}")
    print(f"Файлів завантажено   : {stats['downloaded']}")
    print(f"Пропущено(дублікати) : {stats['skipped']}")
    print(f"Помилок              : {stats['errors']}")
    print(f"Папка збереження     : {os.path.abspath(save_dir)}")
    print("=" * 80)

def run(url: str = DEFAULT_URL, save_dir: str = DEFAULT_SAVE_DIR, depth: int = DEFAULT_DEPTH) -> dict:
    os.makedirs(save_dir, exist_ok=True) # Створюю папку, якщо її немає
    visited: set = set()
    stats: dict = {'downloaded': 0, 'skipped': 0, 'errors': 0, 'pages': 0}
    _scrape_page(url, save_dir, depth, visited, stats)
    _print_summary(stats, save_dir)
    return stats

def delete_save_dir(save_dir: str = DEFAULT_SAVE_DIR) -> None:
    # Видалення скачених pdf файлів з вхідної папки
    if not os.path.exists(save_dir):
        return

    for f in os.listdir(save_dir):
        if f.lower().endswith(".pdf"):
            full_path = os.path.join(save_dir, f)
            try:
                os.remove(full_path)
            except Exception as e:
                print(f"Не вдалося видалити {f}: {e}")


if __name__ == "__main__":
        print("\n\n\nРежими роботи:\n1. Завантажити файли з https://lnu.edu.ua/about/documents/\n2. Видалити завантажені файли\n0. Вийти з програми")
        while True:
            mode = input("\nВаш вибір: ")
            if mode.isdigit() and int(mode) in [0, 1, 2]:
                mode = int(mode)
                break
            print("Некоректний вибір. Спробуйте ще раз.")
       
        if mode == 0:
            exit()
        elif mode == 1:
            run()
            input("\nРоботу завершено. Натисніть Enter для виходу...")
        elif mode == 2:
            delete_save_dir()
            input("\nРоботу завершено. Натисніть Enter для виходу...")