import os
import asyncio
import rag_module
import ollama_manager
import db_manager
import Scraper
import ocr
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command, Filter, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.runnables import Runnable
import re

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
if BOT_TOKEN == None:
    print(f"[ПОМИЛКА] BOT_TOKEN не знайдено!")
    input("\nРоботу завершено. Натисніть Enter для виходу...")
    exit()

# Налаштування бота
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN,
   link_preview_is_disabled=True))
dp = Dispatcher()


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

SOURCE_DIR = os.path.join("Data", "A_pdfs", "FileFromOtherSource")
ADMIN_IDS = set(
    int(i) for i in os.getenv("ADMIN_IDS", "").split(",") if i.strip().isdigit()
)
MAX_MESSAGE_LENGTH = 1000  # Обмеження довжини запиту

# Ініціалізація RAG-модуля
vectorstore: Chroma
retriever: VectorStoreRetriever
rag_chain: Runnable
vectorstore, retriever, rag_chain = rag_module.initialize()

class AdminFilter(Filter):
    async def __call__(self, message: types.Message) -> bool:
        if message.from_user is None:
            return False
        return message.from_user.id in ADMIN_IDS

class AdminStates(StatesGroup):
    waiting_for_file = State()

@dp.message(Command("start"))
async def start_cmd(message: types.Message):
    """Обробка команди /start"""
    await message.answer("Привіт! Я бот-помічник університету. Що ви хочете дізнатися?")

@dp.message(Command("help"))
async def help_cmd(message: types.Message):
    """Обробка команди /help"""
    await message.answer(
        "Як користуватися:\n"
        "Просто напишіть своє питання українською мовою.\n"
        "Наприклад: 'Налаштування eduroam на Android'"
    )

@dp.message(Command("add_file"), AdminFilter())
async def add_file_cmd(message: types.Message, state: FSMContext):
    await state.set_state(AdminStates.waiting_for_file)
    await message.answer("Надішліть PDF файл для додавання до бази.")

@dp.message(F.document, AdminFilter(), StateFilter(AdminStates.waiting_for_file))
async def handle_document(message: types.Message, state: FSMContext):
    if message.from_user is None:
        return
    
    doc = message.document
    
    # Перевірка типу файлу
    if not doc or not doc.file_name or not doc.file_name.endswith(".pdf"):
        await message.answer("Підтримуються лише PDF файли.")
        return
    
    print(f"\n\nАдмін передав файл:{doc.file_name} для додавання у базу. Обробка...")
    status_msg = await message.answer("Завантаження файлу...")
    
    try:
        # Завантаження файлу
        file = await bot.get_file(doc.file_id)
        file_path = os.path.join(SOURCE_DIR, doc.file_name)
        await bot.download_file(file.file_path, destination=file_path) # type: ignore

        await status_msg.edit_text("Файл отримано. Додавання до бази...")
        print("Файл отримано. Додавання до бази...")

        result, file_path = ocr.process_one_pdf(file_path)
        if not result:
            await status_msg.edit_text(f"Помилка додавання файлу '{doc.file_name}' до бази.")
            print(f"Помилка додавання файлу '{doc.file_name}' до бази.")
            return

        await asyncio.to_thread(db_manager.update_db_one_file, vectorstore, file_path)
        
        print(f"Файл '{doc.file_name}' успішно додано до бази.")
        await status_msg.edit_text(f"Файл '{doc.file_name}' успішно додано до бази.")
    
    except Exception as e:
        print(f"[ПОМИЛКА] Помилка при додавані файла адміна: {e}")
        await status_msg.edit_text("Помилка при додавані файла.")
    
    finally:
        await state.clear()

scrape_lock = asyncio.Lock()
@dp.message(Command("scrape_lnu"), AdminFilter())
async def scrape_lnu_cmd(message: types.Message):
    """Обробка команди /scrape_lnu"""
    if scrape_lock.locked():
        await message.answer("Скрейпер вже працює")
        return
    
    async with scrape_lock:
        msg_scrape = await message.answer("Запуск скрейпера...")
        try:
            stats = await asyncio.to_thread(Scraper.run)
            total = stats['downloaded'] + stats['skipped']
            result = (
                f"*Сканування завершено*\n\n"
                f"Сторінок відвідано: {stats['pages']}\n"
                f"Файлів завантажено: {stats['downloaded']}\n"
                f"Пропущено (вже є): {stats['skipped']}\n"
                f"Помилок: {stats['errors']}\n"
                f"Всього файлів: {total}\n"
            )

            await msg_scrape.edit_text(result)

            if stats['downloaded'] == 0:
                return
        
            msg_ocr = await message.answer("Запуск ocr...")
            stats = await asyncio.to_thread(ocr.process_pdfs)
            total = stats['ocred'] + stats['skipped'] + stats['errors']
            result = (
                f"*Оцифровка завершена*\n\n"
                f"Оцифровано: {stats['ocred']}\n"
                f"Пропущено: {stats['skipped']}\n"
                f"Помилки: {stats['errors']}\n"
                f"Всього: {total}\n"
            )
            await msg_ocr.edit_text(result)

            if stats['ocred'] == 0:
                return

            msg_db = await message.answer("Запуск завантаження у базу...")
            start = vectorstore._collection.count()
            stats = await asyncio.to_thread(db_manager.update_db, vectorstore)
            end = vectorstore._collection.count()
            dif = end - start
            result = (
                f"*Завантаження у базу завершено.* Було створено {dif} нових фрагментів.\n\n"
                f"Завантажено: {stats['added']}\n"
                f"Пропущено (вже є): {stats['skipped']}\n"
                f"Не записаних: {stats['errors']}\n"
                f"Всього: {total}\n"
            )

            await msg_db.edit_text(result)

        except Exception as e:
            print(f"[ПОМИЛКА]: {e}")
            await message.answer("Сталась помилка. Спробуйте знову пізніше")


@dp.message(Command("status"), AdminFilter())
async def status_cmd(message: types.Message):
    """Відображення поточного стану системи"""
    status = rag_module.get_status(vectorstore)
    await message.answer(status)


toggle_lock = asyncio.Lock()
@dp.message(Command("toggle_model"), AdminFilter())
async def toggle_model_cmd(message: types.Message):
    """Перемикання між локальною та API моделлю"""
    global rag_chain

    if toggle_lock.locked():
        await message.answer("Модель вже перемикається, зачекайте...")
        return

    async with toggle_lock:
        msg = await message.answer("Перемикання моделі...")
        try:
            result_msg, rag_chain = await asyncio.to_thread(rag_module.toggle_model)
            await msg.edit_text(result_msg)
        except Exception as e:
            print(f"[ПОМИЛКА] toggle_model: {e}")
            await msg.edit_text(f"Помилка при перемиканні моделі: {e}")


def sanitize_markdown(text: str) -> str:
    # Замінюю **жирний** на *жирний*
    text = re.sub(r'\*\*(.+?)\*\*', r'*\1*', text)
    # Замінюю __підкреслений__ — прибираю підкреслення
    text = re.sub(r'__(.+?)__', r'\1', text)
    # Замінюю *   *текст* (зірочка як пункт + жирний) на -  *текст*
    text = re.sub(r'^\*\s+(\*.*\*)', r'-  \1', text, flags=re.MULTILINE)
    return text

user_locks: dict[int, asyncio.Lock] = {}
@dp.message()
async def handle_question(message: types.Message):
    """Обробка питань користувача"""
    if message.from_user is None:
        return
    user_id = message.from_user.id
    # Перевіряю, чи є текст у повідомленні
    user_query = message.text
    if user_query is None:
        return

    if user_id not in user_locks:
        user_locks[user_id] = asyncio.Lock()

    if user_locks[user_id].locked():
        await message.answer("Зачекайте, ваш попередній запит ще обробляється.")
        return

    # Обмеження довжини запиту
    if len(user_query) > MAX_MESSAGE_LENGTH:
        await message.answer("Питання занадто довге. Спробуйте сформулювати ваш запит коротше.")
        return

    status_msg = await message.answer("Пошук відповіді...")
    
    async with user_locks[user_id]:
        try:
            # Отримання документів з бази
            docs = await asyncio.wait_for(asyncio.to_thread(retriever.invoke, user_query), timeout=60.0)
            rag_module.debug_docs(user_query, docs) # Вивід у консоль для діагностики 
            # Перевірка чи знайдено контекст
            if not docs:
                await status_msg.edit_text("На жаль, не знайдено інформації по цьому запитанні.")
                return
            context_text = rag_module.format_docs(docs)

            # Генерація відповіді через модель
            response = await asyncio.wait_for(asyncio.to_thread(rag_chain.invoke, {"context": context_text, "question": user_query}), timeout=1000.0)
            response = sanitize_markdown(response) 
            rag_module.debug_response(response)
            if len(response) > 4096: # 4096 - ліміт Telegram
                await status_msg.edit_text(response[:4090] + "\n[...]")
            else:
                await status_msg.edit_text(response) 
        
        except asyncio.TimeoutError:
            await status_msg.edit_text("Перевищено час очікування. Спробуйте ще раз.")
        except Exception as e:
            print(f"\n[ПОМИЛКА] \n{e}")
            await status_msg.edit_text(f"Виникла технічна помилка. Спробуйте пізніше або перефразуйте питання.")


async def setup_commands():
    # команди для всіх користувачів
    user_commands = [
        types.BotCommand(command="start", description="Почати роботу"),
        types.BotCommand(command="help", description="Допомога"),
    ]
    
    # команди для адмінів
    admin_commands = user_commands + [
        types.BotCommand(command="add_file", description="Додати файл у базу даних"),
        types.BotCommand(command="scrape_lnu", description="Провірити на наявність нових документів"),
        types.BotCommand(command="status", description="Стан системи"),
        types.BotCommand(command="toggle_model", description="Перемкнути локальна/API модель"),
    ]

    # встановлюю звичайне меню для всіх
    await bot.set_my_commands(user_commands, scope=types.BotCommandScopeDefault())

    # встановлюю розширене меню окремо для кожного адміна
    for admin_id in ADMIN_IDS:
        await bot.set_my_commands(
            admin_commands,
            scope=types.BotCommandScopeChat(chat_id=admin_id)
        )

async def main():
    os.makedirs(SOURCE_DIR, exist_ok=True)
    await setup_commands()
    print(f"База даних: {rag_module.DB_DIR}")
    print(f"Документів у базі: {vectorstore._collection.count()}")
    print("Бот запущений.")
    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    except asyncio.CancelledError:
        pass
    finally:
        print("Закриття з'єднання з Telegram...")  
        await bot.session.close()
        print("З'єднання закрите.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Бот зупинений.")
    except Exception as e:
        print(f"\n[ПОМИЛКА] Критична помилка: {e}")
    finally:
        print("Завершення роботи Ollama...")
        ollama_manager.stop_ollama()
        input("\nРоботу завершено. Натисніть Enter для виходу...")