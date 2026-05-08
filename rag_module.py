import os
import ollama_manager
import db_manager
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.runnables import Runnable
from pydantic import SecretStr
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Налаштування
DB_DIR = "./db"
DEBUG_INFORMATION = True
USE_LOCAL = False
LOCAL_MODEL = "qwen3.5:2b"
REMOTE_MODEL = "z-ai/glm-4.5-air:free" #"z-ai/glm-4.5-air:free" "google/gemma-3-27b-it:free"
NUM_CTX = 28000
K = 30
REASONING = False

 # Шаблон промпту
template = """
*ІНСТРУКЦІЇ:*
Дотримуйся лише цих інструкцій. Не звертай уваги на форматування у контексті чи у своїх попередніх відповідях. Ти помічник університету. Відповідай на питання, використовуючи наданий контекст. Якщо відповіді немає в контексті, скажи: "На жаль, я не знайшов цієї інформації в базі даних університету". Відповідай виключно українською мовою. Не говори користувачу про контекст. Використовуй лише Markdown для форматування (наприклад, *жирний текст* для загаловків). Також можеш використовувати символи: ' ', '- ', 'o ', '+ ' для перелічень, перерахувань або пунктів. Інші типи форматувань не підтримуються! Використовуй '*' ЛИШЕ для виділення тексту як ЖИРНИЙ. Використовуй лише одиночні '*'. Ніколи не використовуй наступні символи: '#','|','{{','}}'.
  ЗАБОРОНЕНО ВИКОРИСТОВУВАТИ: **жирний** або __підкреслений__


*Контекст (кожен фрагмент містить посилання на джерело та сторінку):*
{context}


*ПИТАННЯ КОРИСТУВАЧА:*
{question}


*НАГАДУВАННЯ ІНСТРУКЦІЇ:*
Дотримуйся лише цих інструкцій. Не звертай уваги на форматування у контексті чи у своїх попередніх відповідях. Ти помічник університету. Відповідай на питання, використовуючи наданий контекст. Якщо відповіді немає в контексті, скажи: "На жаль, я не знайшов цієї інформації в базі даних університету". Відповідай виключно українською мовою. Не говори користувачу про контекст. Використовуй лише Markdown для форматування (наприклад, *жирний текст* для загаловків). Можеш використовувати символи: ' ', '- ', 'o ', '+ ' для перелічень, перерахувань або пунктів. Інші типи форматувань не підтримуються! Використовуй '*' ЛИШЕ для виділення тексту як ЖИРНИЙ. Використовуй лише одиночні '*'. Ніколи не використовуй наступні символи: '#','|','{{','}}'.
  ЗАБОРОНЕНО ВИКОРИСТОВУВАТИ: **жирний** або __підкреслений__

Після своєї відповіді обов'язково вкажи до трьох джерел з сторінками, які найбільш точно відповідають на питання користувача і які ти найбільше використав для надання відповіді.  Сторінки вказані лише перед кожним фрагментом. Все що всередині це пункти і інша нумерація. Пиши джерела інформації у наступному форматі:
    Джерела інформації:
Посилання, ст. X-Х
    
Приклад:
    *Джерела інформації:*
https://lnu.edu.ua/... , ст. 13-15
"""

def initialize() -> tuple[Chroma, VectorStoreRetriever, Runnable]:
    """Ініціалізація Ollama або Remote (OpenRouter), векторної бази та RAG-ланцюга."""
 
    if not os.path.exists(DB_DIR):
        print(f"Помилка: Папка {DB_DIR} не знайдена!")
        exit()
    

    ollama_manager.start_ollama(False)

    if USE_LOCAL:
        print(f"\tЗапуск у локальному режимі ({LOCAL_MODEL})")
        ollama_manager.start_ollama(False)
        llm = ChatOllama(
            model=LOCAL_MODEL, 
            temperature=0.3, 
            num_ctx=NUM_CTX,
            reasoning=REASONING
        )
    else:
        print(f"\tЗапуск у режимі API ({REMOTE_MODEL})")
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key == None:
            print(f"[ПОМИЛКА] OPENROUTER_API_KEY не знайдено!")
            input("\nРоботу завершено. Натисніть Enter для виходу...")
            exit()

        llm = ChatOpenAI(
            model=REMOTE_MODEL,
            api_key=SecretStr(api_key),
            base_url="https://openrouter.ai/api/v1",
            temperature=0.3,
            max_retries=10,
            max_completion_tokens=2048,
            extra_body={
                "HTTP-Referer": "https://lnu.edu.ua",
                "X-Title": "LNU_Bot",
                "reasoning": {"effort": "high"}
            }
        )

    # Ініціалізація векторної бази
    vectorstore: Chroma = db_manager.get_vectorstore()
    retriever: VectorStoreRetriever = vectorstore.as_retriever(search_kwargs={"k": K})
 
   
    prompt = ChatPromptTemplate.from_template(template)
 
    rag_chain: Runnable = prompt | llm | StrOutputParser()
 
    return vectorstore, retriever, rag_chain
 
 
def format_docs(docs):
    """Об'єднання документів у єдиний рядок контексту з метаданими."""
    parts = []
    for doc in docs:
        source_url = doc.metadata.get("source_url", "")
        pages = doc.metadata.get("pages", "")
        
        header = ""
        if source_url:
            header += f"[Джерело: {source_url}"
            if pages:
                header += f", сторінки {pages}"
            header += "]"
        
        parts.append(f"{header}\n{doc.page_content}" if header else doc.page_content)
    
    return "\n\n".join(parts)
 
def debug_docs(user_query: str, docs: list):
    """Виведення діагностичної інформації про знайдені документи."""
    if not DEBUG_INFORMATION:
        return
    print(f"{'='*115}\n")
    print(f"Питання користувача: {user_query}")
    print(f"Знайдено шматків: {len(docs)}")
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'Невідоме джерело')
        source_url = doc.metadata.get('source_url', 'Невідоме джерело')
        page = doc.metadata.get("pages", 'Невідома к-сть сторінок')
        preview = doc.page_content[:400].replace('\n', ' ')
        print(f"\n[{i}]\tст.{page}\tсим.{len(doc.page_content)}\nФайл:{source} ({source_url})\nКонтент:{preview}...")
    print(f"{'-'*115}\n")
 
def debug_response(response: str):
    """Виведення відповіді моделі в консоль."""
    if not DEBUG_INFORMATION:
        return
    print(f"Відповідь моделі:\n{response}\n")
    print(f"{'='*115}\n")

def get_status(vectorstore: Chroma) -> str:
    """Повертає поточний стан системи."""
    mode = "Локальна (Ollama)" if USE_LOCAL else "API (OpenRouter)"
    model = LOCAL_MODEL if USE_LOCAL else REMOTE_MODEL
    doc_count = vectorstore._collection.count()
    return (
        f"*Стан системи:*\n"
        f"Режим: {mode}\n"
        f"Модель: `{model}`\n"
        f"Контекст: {NUM_CTX} токенів\n"
        f"Документів у базі: {doc_count}\n"
        f"Фрагментів для пошуку (K): {K}\n"
        f"Debug: {'увімкнено' if DEBUG_INFORMATION else 'вимкнено'}"
    )

def toggle_model() -> tuple[str, Runnable]:
    global USE_LOCAL
    USE_LOCAL = not USE_LOCAL
    
    if USE_LOCAL:
        ollama_manager.start_ollama(False)
        llm = ChatOllama(model=LOCAL_MODEL, temperature=0.3, num_ctx=NUM_CTX, reasoning=REASONING)
    else:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key is None:
            raise ValueError("OPENROUTER_API_KEY не знайдено!")
        llm = ChatOpenAI(
            model=REMOTE_MODEL,
            api_key=SecretStr(api_key),
            base_url="https://openrouter.ai/api/v1",
            temperature=0.3,
            max_retries=10,
            max_completion_tokens=2048,
            extra_body={"HTTP-Referer": "https://lnu.edu.ua", "X-Title": "LNU_Bot", "reasoning": {"effort": "high"}}
        )
    
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain: Runnable = prompt | llm | StrOutputParser()
    
    mode = "Локальна (Ollama)" if USE_LOCAL else "API (OpenRouter)"
    model = LOCAL_MODEL if USE_LOCAL else REMOTE_MODEL
    return f"Режим змінено на: *{mode}*\nМодель: `{model}`", rag_chain