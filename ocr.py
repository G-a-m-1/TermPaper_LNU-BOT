import os
import pytesseract # (tesseract)
from pypdf import PdfReader, PdfWriter
from pdf2image import convert_from_path # (poppler)
from fpdf import FPDF
from concurrent.futures import ThreadPoolExecutor
from PIL import ImageOps

MAX_WORKERS = os.cpu_count() or 12  
SOURCE_DIR = os.path.join("Data", "A_pdfs")
OUTPUT_DIR = os.path.join("Data", "D_pdfs")
LANG = "ukr+eng"
FONT_PATH = os.path.join("Fonts", "LiberationSans-Regular.ttf")
TESSERACT_CONFIG = "--oem 3 --psm 3"
DEBUG = False
DEBUG_DIR = "Debug"

COMMON_WORDS = {
    "що", "як", "для", "або", "та", "які", "але", "він", "вона", "вони", "ми", "ви", "це", "той", "ця", "яка","який", "про", "при", "від", "до", "на", "не", "за", "із", "по", "між", "над", "під", "без", "через", "після", "університеті", "університет", "студент", "навчання", "кафедра", "факультет", "наказ", "відповідно", "згідно", "затверджено", "розклад", "навчальний", "рік", "року", "наказу", "відділ", "декан", "ректор", "відомість", "протокол", "план", "львів", "львівський", "нацональний", "список", "списку", "року", "імені", "івана", "франка", "довідки", "ознайомлення", "копія", "копії", "документу", "документ", "академічної", "академічна", "пільги", "заява", "додаток", "даних", "фонд", "зберігання", "банку", "банк", "освіту", "освіта", "бюджет", "бюджеті", "прізвище", "форма", "одиниця", "заклад", "установа", "організація", "пункт", "перелік", "студента", "зразок", "підпис", "підписом"
}

BAD_CHARS = "|~^@#$%\\=`<>{}*"

def _text_to_words(text:str)->list[str]:
    """Розділяє текст на окремі очищені слова"""
    for c in BAD_CHARS:
        text = text.replace(c, " ")# прибираю артефакти ocr 

    text = text.replace('\u2010', '-')  # дефіс
    text = text.replace('\u2011', '-')  # нерозривний дефіс
    text = text.replace('\u2012', '-')  # фігурне тире
    text = text.replace('\u2013', '-')  # коротке тире
    text = text.replace('\u2014', '-')  # довге тире

    # Перебирає кожен символ тексту і залишає його якщо символ є ASCII або символ знаходиться в діапазоні кирилиці
    text = ''.join(c for c in text if (c.isascii() and c >= ' ') or '\u0400' <= c <= '\u04FF')
    words = text.split()
    return words

def _is_correct_words(words:list[str],)->bool:
    """Перевіряє чи слова з однієї сторінки коректні"""
    if not words:
        return False

    if len(words) < 50:
        return False
            
    known_words = 0
    for w in words:
        if w.strip(".,;:!?()[]\"'") in COMMON_WORDS:
            known_words += 1

    if known_words >= 8:
        return True

    return False

def _is_valid_pdf(file_path: str) -> bool:
    """Перевіряє чи PDF файл не пошкоджений"""
    try:
        reader = PdfReader(file_path)
        _ = len(reader.pages)  # спроба прочитати сторінки
        return True
    except Exception as e:
        print(f"  [ПОПЕРЕДЖЕННЯ] Пошкоджений файл: {os.path.basename(file_path)}: {e}")
        return False


def _ocr_page(image) -> None:
    """Обробляє і оцифровує одне зображення"""
    #Збільшення контрасту
    if DEBUG:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        image.save(os.path.join(DEBUG_DIR, f"1.png"), format="PNG")

    def soft_threshold(p):
        if p < 30: return 0
        if p > 230: return 255
        return p

    image = ImageOps.autocontrast(image, cutoff=1)
    image = image.point(soft_threshold)
    
    if DEBUG:
        image.save(os.path.join(DEBUG_DIR, f"2.png"), format="PNG")
        input(f"Оброблено сторінку. Результат у {DEBUG_DIR}.\n")

    return pytesseract.image_to_string(image, lang=LANG, config=TESSERACT_CONFIG)

def _get_consecutive_groups(indices: list) -> list[list]:
    """Групує послідовні індекси для обробки"""
    if not indices:
        return []
    
    groups = []
    current_group = [indices[0]]
    
    for idx in indices[1:]:
        if idx == current_group[-1] + 1 and len(current_group) < MAX_WORKERS:
            current_group.append(idx)
        else:
            groups.append(current_group)
            current_group = [idx]
    groups.append(current_group)
    return groups

def _ocr_pages(file_path: str, ocr_needed: list) -> dict:
    """OCR для сторінок без тексту. Повертає {index: text|None}"""
    ocr_results = {}
    if not ocr_needed:
        return ocr_results

    groups = _get_consecutive_groups(ocr_needed)
    workers = min(MAX_WORKERS, len(ocr_needed))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for group in groups:
            first_page = group[0] + 1
            last_page = group[-1] + 1
            print(f"  Обробка {first_page}-{last_page} сторінок...     ", end='\r')

            chunk_images = convert_from_path(
                file_path,
                dpi=400,
                first_page=first_page,
                last_page=last_page,
                grayscale=True
            )

            chunk_texts = list(executor.map(_ocr_page, chunk_images))

            for idx, text in zip(group, chunk_texts):
                words = _text_to_words(text or "")
                if _is_correct_words(words):
                    ocr_results[idx] = " ".join(words)
                else:
                    ocr_results[idx] = None

            del chunk_images

    return ocr_results

def _save_pdf(output_path: str, pages_text: dict, original_metadata) -> None:
    """Зберігає текст сторінок у PDF з метаданими"""
    pdf = FPDF()
    if os.path.exists(FONT_PATH):
        pdf.add_font("FreeSans", "", FONT_PATH)
        font_name = "FreeSans"
    else:
        font_name = "Arial"
        print(f"\n[ПОПЕРЕДЖЕННЯ] Шрифт не знайдено, використовується Arial")

    pdf.set_font(font_name, size=8)
    for text in pages_text.values():
        pdf.add_page()
        if text is not None:
            pdf.multi_cell(0, 3, text=text)
    pdf.output(output_path)

    # Копіюю метадані
    new_reader = PdfReader(output_path)
    writer = PdfWriter()
    writer.append(new_reader)
    if original_metadata:
        writer.add_metadata({k: v for k, v in original_metadata.items()})

    # Перезаписую файл з метаданими
    with open(output_path, 'wb') as f:
        writer.write(f)

def ocr_pdf(file_path: str, output_path: str) -> bool:
    """Оцифровує один PDF-файл і зберігає результат у output_path"""
    try:
        # Обрахунок кількості сторінок файлу
        reader = PdfReader(file_path)
        original_metadata = reader.metadata
        num_pages = len(reader.pages)

        # визначаю які сторінки потребують OCR
        text_pages = {}  # {index: text}
        ocr_needed = []  # індекси сторінок для OCR
        
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").lower()
            words = _text_to_words(text)
            if _is_correct_words(words):
                text_pages[i] = " ".join(words)
            else:
                ocr_needed.append(i)

        print(f"  Знайдено {num_pages} сторінок. ({len(text_pages)} цифрових сторінок)")

        # OCR для сторінок без тексту
        ocr_results = _ocr_pages(file_path, ocr_needed)
              
        # збираю результати
        pages_text = {}
        for i in range(num_pages):
            if i in text_pages:
                pages_text[i] = text_pages[i]
            else:
                pages_text[i] = ocr_results.get(i)

        correct_pages = 0
        for t in pages_text.values():
            if t is not None:
                correct_pages += 1

        if correct_pages == 0:
            print(f"  [ПОПЕРЕДЖЕННЯ] Файл не було записано — не знайдено корисної інформації.")
            return False

        print(f"  Корисних сторінок: {correct_pages}/{num_pages}")

        _save_pdf(output_path, pages_text, original_metadata)

        print(f"  Успішно оброблено: {os.path.basename(output_path)}")
        return True

    except Exception as e:
        print(f"\n[ПОМИЛКА] Помилка OCR ({file_path}): {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

def process_pdfs(source_dir: str = SOURCE_DIR, output_dir: str = OUTPUT_DIR) -> dict:
    """Основний код. Обробляє всі PDF — копіює текстові та оцифровує скани"""
    os.makedirs(output_dir, exist_ok=True) # Створюю вихідну папку якщо не існує
    stats: dict = {'ocred': 0, 'skipped': 0, 'errors': 0}

    # Пошук всіх pdf файлів з вхідної папки
    files = []
    for root, _, filenames in os.walk(source_dir):
        for f in filenames:
            if f.lower().endswith(".pdf"):
                files.append(os.path.join(root, f))

    for i in range(len(files)):
        src = files[i]
        dst = os.path.join(output_dir, os.path.basename(files[i]))

        print(f"[{i+1}/{len(files)}] {files[i]}")

        # Пропускаю якщо вже є  цей файл в output
        if os.path.exists(dst):
            print(f"  Вже існує, пропуск")
            stats['skipped'] += 1
            continue

        # Чи цілий
        if not _is_valid_pdf(src):
            print(f"[ПОМИЛКА] Файл {src} пошкоджений")
            stats['errors'] += 1
            continue

        print(f"  Розпочата обробка...")
        result = ocr_pdf(src, dst)
        if result:
            stats['ocred'] += 1
        else:
            stats['errors'] += 1

    _print_summary(stats,output_dir)
    return stats

def process_one_pdf(file_path:str, output_path: str|None = None) -> tuple[bool,str]:
    """Основний код. Обробляє  PDF — копіює якщо текстовий та оцифровує якщо скан. Повертає bool чи успішно оброблено та шлях куда оброблено"""
    if not output_path:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, os.path.basename(file_path))
    
    # Створюю вихідну папку якщо не існує
    os.makedirs(os.path.dirname(output_path), exist_ok=True)


    # Пропускаю якщо вже є цей файл в output
    if os.path.exists(output_path):
        print(f"  Вже існує, пропуск")
        return False,output_path

    # Чи цілий
    if not _is_valid_pdf(file_path):
        print(f"[ПОМИЛКА] Файл {file_path} пошкоджений")
        return False,output_path

    # Скан — оцифровую
    print(f"  Розпочата обробка...")
    result = ocr_pdf(file_path, output_path)
    return result, output_path

def _print_summary(stats: dict, save_dir: str) -> None:
    total = stats['ocred'] + stats['skipped'] + stats['errors']
    print("\n\n" + "=" * 80)
    print("Оцифровка завершена")
    print("=" * 80)
    print(f"Оцифровано         : {stats['ocred']}")
    print(f"Пропущено (вже є)  : {stats['skipped']}")
    print(f"Не записаних       : {stats['errors']}")
    print(f"Всього файлів      : {total}")
    print(f"Папка збереження   : {os.path.abspath(save_dir)}")
    print("=" * 80)

def delete_output(output_dir: str = OUTPUT_DIR) -> None:
    # Видалення всіх pdf файлів з вихідної папки
    if not os.path.exists(output_dir):
        return

    for f in os.listdir(output_dir):
        if f.lower().endswith(".pdf"):
            full_path = os.path.join(output_dir, f)
            try:
                print(f" Видалення {f}...",end="")
                os.remove(full_path)
                print("Успішно!")
            except Exception as e:
                print(f"Не вдалося видалити {full_path}: {e}")
    print(" Видалення завершено.")


if __name__ == "__main__":
    while True:
        print("Режими роботи:\n1. Обробити всі файли\n2. Видалення всіх pdf файлів з вихідної папки\n0. Вийти з програми")
        while True:
            mode = input("\nВаш вибір: ")
            if mode.isdigit() and int(mode) in [0, 1, 2]:
                mode = int(mode)
                break
            print("Некоректний вибір. Спробуйте ще раз.")
        
        match mode:
            case 0:
                exit()
            case 1:
                process_pdfs()
                print("\n"*10)
            case 2:
                delete_output()
                print("\n"*10)