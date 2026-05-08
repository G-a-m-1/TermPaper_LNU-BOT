import os
import ollama_manager
from queue import Queue
from threading import Thread
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader



# Налаштування
DB_DIR = "./db"
DOCS_DIR = os.path.join("Data", "D_pdfs")
EMBEDDINGS_MODEL = "qwen3-embedding:0.6b" #"nomic-embed-text"для англійської,"bge-m3" не працює,"mxbai-embed-large:latest": мале вікно, "qwen3-embedding:0.6b"
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 100
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, 
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", "!", "?", " "] # Пріоритет розрізу для збереження сенсу
)
BATCH_SIZE = 10



def _process_single_pdf(file_path):
    """Функція для обробки одного файлу"""
    try:
        # Завантажую сторінки PDF як окремі документи
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()

        # Читаю метадані PDF щоб дістати source_url
        reader = PdfReader(file_path)
        source_url = (reader.metadata or {}).get("/Source", "")

        # Додаю джерело у metadata кожного документа
        if source_url:
            for doc in docs:
                doc.metadata["source_url"] = source_url
        
        # Об'єдную малі сторінки в один блок поки їх сумарний розмір менше CHUNK_SIZE
        merged = []
        current_text = ""
        current_start = None

        for doc in docs:
            page_num = doc.metadata.get("page", 0)
            text = doc.page_content.strip()

            if not text:  # пропускаю пусті сторінки
                continue

            if current_start is None: # перша непуста сторінка
                current_start = page_num
                current_end = page_num

            if len(current_text) + len(text) < CHUNK_SIZE: # якщо сторінка ще вміщується, то додаю до поточного блоку
                current_text += " \n\n" + text
                current_end = page_num
            else: # якщо блок заповнений — зберігаю і створюю новий
                if current_text:
                    merged.append((current_text.strip(), current_start, current_end))
                current_text = text
                current_start = page_num
                current_end = page_num

        if current_text:
            merged.append((current_text.strip(), current_start, current_end))

        # Створюю документи з правильними метаданими

        result_docs = []
        for text, start, end in merged:
            pages_label = f"{start+1}-{end+1}" if start != end else str(start+1)
            doc = Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "pages": pages_label,
                }
            )
            if source_url:
                doc.metadata["source_url"] = source_url
            result_docs.append(doc)

        # якщо одна сторінка була більша за CHUNK_SIZE - розрізаю
        docs = TEXT_SPLITTER.split_documents(result_docs)

        return docs

    except Exception as e:
        print(f"Помилка при читанні {file_path}: {e}")
        return []


def _db_add_clean_chunks(vectorstore: Chroma, processed_docs: list) -> bool:
    """Завантажує у базу даних передані документи"""
    if not processed_docs:
        print(f"    [ПОМИЛКА]: Не створено фрагментів для обробки")
        return False
    print(f"    Створено {len(processed_docs)} фрагментів...")
    for i in range(0, len(processed_docs), BATCH_SIZE):
        batch = processed_docs[i : i + BATCH_SIZE]
        try:
            vectorstore.add_documents(batch)
            print(f"    {i+1} - {min(i + BATCH_SIZE, len(processed_docs))} фрагменти успішно додані у базу.")
        except Exception as e:
            print(f"    [ПОМИЛКА]: {e}")
            return False
    return True


def update_db(vectorstore, docs_dir = DOCS_DIR)->dict:
    """Функція для оновлення бази даних з файлів заданої директорії"""
    ollama_manager.start_ollama(False)
    stats: dict = {'added': 0, 'skipped': 0, 'errors': 0}
    
    # Пошук файлів для обробки
    files_to_process = []
    for f in os.listdir(docs_dir):
        if f.lower().endswith(".pdf"):
            full_path = os.path.join(docs_dir, f)
            result = vectorstore.get(where={"source": full_path}, limit=1) # Перевірка чи вже є у базі
            if not result['ids']:
                files_to_process.append(full_path) # Якщо файлу нема - додаю
            else:
                stats['skipped'] += 1

    if not files_to_process:
        print("Нових документів не знайдено.")
        return stats

    print(f"Знайдено {len(files_to_process)} нових файлів. Обробка...")
    
    
    def producer(queue:Queue, files:list) -> None:
        """Функція для обробки багатьох файлів у потоці і передачі оброблених документів у чергу"""
        for i in range(len(files)):
            chunks = _process_single_pdf(files[i])
            queue.put((chunks,i))
        queue.put(None)  # сигнал що все оброблено

    
    def consumer(queue:Queue, files:list,stats:dict) -> None:
        """Функція для завантаження у базу документів у потоці з черги"""
        while True:
            item = queue.get()
            if item is None:
                break
            chunks, index = item

            print(f"[{index+1}/{len(files)}] Обробка {os.path.basename(files[index])}...")
            result = _db_add_clean_chunks(vectorstore, chunks)
            if result:
                print(f"    Успішно оброблено!")
                stats['added'] += 1
            else:
                print(f"    Виникли помилки при обробці!")
                stats['errors'] += 1


    queue = Queue(maxsize=3)  # буфер на 3 файли
    t_producer = Thread(target=producer, args=(queue, files_to_process))
    t_consumer = Thread(target=consumer, args=(queue, files_to_process, stats))
    t_producer.start()
    t_consumer.start()
    t_producer.join()
    t_consumer.join()
    _print_summary(stats)
    return stats


def update_db_one_file(vectorstore, full_path:str) -> None:
    """Функція для оновлення бази даних з одного файлу"""
    ollama_manager.start_ollama(False)
    chunks = _process_single_pdf(full_path)
    _db_add_clean_chunks(vectorstore, chunks)


def delete_db(vectorstore)->None:
    """Функція для видалення бази даних"""
    ollama_manager.start_ollama(False)
    all_ids = vectorstore.get()['ids']
    if not all_ids:
        print("[ПОПЕРЕДЖЕННЯ] База даних вже порожня.")
        return
    vectorstore.delete(ids=all_ids)
    print(f"Видалено {len(all_ids)} записів з бази даних.")


def get_vectorstore(db_dir: str = DB_DIR) -> Chroma:
    embeddings = OllamaEmbeddings(model=EMBEDDINGS_MODEL)
    return Chroma(persist_directory=db_dir, embedding_function=embeddings)


def _print_summary(stats: dict) -> None:
    total = stats['added'] + stats['skipped'] + stats['errors']
    print("\n\n" + "=" * 80)
    print("Завантаження у базу завершено")
    print("=" * 80)
    print(f"Завантажено        : {stats['added']}")
    print(f"Пропущено (вже є)  : {stats['skipped']}")
    print(f"Не записаних       : {stats['errors']}")
    print(f"Всього файлів      : {total}")
    print("=" * 80)


if __name__ == "__main__":
    if not os.path.exists(DOCS_DIR):
        print(f"[ПОПЕРЕДЖЕННЯ] Вхідна папка '{DOCS_DIR}' не існує!")
        os.makedirs(DOCS_DIR)
        
    vectorstore = get_vectorstore()
    while True:
        print("\n\n\nРежими роботи:\n1. Завантажити файли до бази\n2. Очистити всю базу\n0. Вийти з програми")
        while True:
            mode = input("\nВаш вибір: ")
            if mode.isdigit() and int(mode) in [0, 1, 2]:
                mode = int(mode)
                break
            print("Некоректний вибір. Спробуйте ще раз.")
        
        if mode == 0:
            ollama_manager.stop_ollama()
            input("\nРоботу завершено. Натисніть Enter для виходу...")
            exit()
        elif mode == 1:
            update_db(vectorstore)
            print("\n"*10)
        elif mode == 2:
            delete_db(vectorstore)
            print("\n"*10)