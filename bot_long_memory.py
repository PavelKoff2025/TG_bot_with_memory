"""
Telegram-бот с долгой памятью (RAG).
Документ → чанки → эмбеддинги → ChromaDB → ответ по контексту.
"""

import asyncio
import os
import re
import uuid
from pathlib import Path

import chromadb
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message
from dotenv import load_dotenv
from openai import AsyncOpenAI

# --- ENV ---
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not BOT_TOKEN:
    raise ValueError("Не задан BOT_TOKEN в переменных окружения (.env)")
if not OPENAI_API_KEY:
    raise ValueError("Не задан OPENAI_API_KEY в переменных окружения (.env)")

# --- Настройки ---
CHUNK_SIZE = 500
CHAT_MODEL = os.getenv("CHAT_MODEL") or os.getenv("MODEL") or "gpt-3.5-turbo"
EMBED_MODEL = os.getenv("EMBED_MODEL") or "text-embedding-3-small"
TOP_K = 4  # сколько фрагментов достаём из векторной БД
MEMORY_DIR = Path("./memory")
DOWNLOADS_DIR = Path("./downloads")

MEMORY_DIR.mkdir(exist_ok=True)
DOWNLOADS_DIR.mkdir(exist_ok=True)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Persistent ChromaDB в папке ./memory
chroma_client = chromadb.PersistentClient(path=str(MEMORY_DIR))
collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"},
)

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


# ---------------------------------------------------------------------------
# Работа с документами и памятью
# ---------------------------------------------------------------------------

def load_document(file_path: str) -> str:
    """Читает PDF / TXT / DOCX и возвращает плоский текст."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)

    if suffix == ".docx":
        from docx import Document

        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)

    raise ValueError(f"Неподдерживаемый формат файла: {suffix}")


def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Делит текст на части примерно по chunk_size символов."""
    # Нормализуем пробелы, чтобы чанки были чище
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    chunks = []
    for i in range(0, len(cleaned), chunk_size):
        piece = cleaned[i : i + chunk_size].strip()
        if piece:
            chunks.append(piece)
    return chunks


async def embed_chunks(
    chunks: list[str],
    user_id: int,
    filename: str,
) -> int:
    """
    Создаёт эмбеддинги для чанков и сохраняет их в ChromaDB.
    Возвращает количество сохранённых фрагментов.
    """
    if not chunks:
        return 0

    # Получаем векторы через OpenAI Embeddings API
    response = await openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=chunks,
    )
    embeddings = [item.embedding for item in response.data]

    ids = [f"{user_id}_{uuid.uuid4().hex}" for _ in chunks]
    metadatas = [
        {"user_id": str(user_id), "filename": filename, "chunk_index": i}
        for i in range(len(chunks))
    ]

    # Запись в persistent-хранилище Chroma
    await asyncio.to_thread(
        collection.add,
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    return len(chunks)


async def retrieve_context(query: str, user_id: int, n_results: int = TOP_K) -> str:
    """
    Ищет в ChromaDB фрагменты, релевантные вопросу пользователя.
    Возвращает склеенный контекст или пустую строку.
    """
    # Эмбеддинг самого вопроса
    response = await openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=[query],
    )
    query_embedding = response.data[0].embedding

    def _query():
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"user_id": str(user_id)},
            include=["documents", "metadatas"],
        )

    results = await asyncio.to_thread(_query)

    documents = results.get("documents") or []
    if not documents or not documents[0]:
        return ""

    # Склеиваем найденные куски в один контекст
    parts = []
    for i, doc in enumerate(documents[0], start=1):
        parts.append(f"[{i}] {doc}")
    return "\n\n".join(parts)


async def answer_question(question: str, user_id: int) -> str:
    """
    Достаёт контекст из векторной БД и просит модель ответить
    строго на основе документа (без выдумок).
    """
    context = await retrieve_context(question, user_id)

    if not context:
        return (
            "Пока нет загруженных документов или по ним ничего не найдено.\n"
            "Пришлите PDF / TXT / DOCX, а потом задайте вопрос."
        )

    system_prompt = (
        "Ты ассистент, который отвечает ТОЛЬКО на основе предоставленного контекста "
        "из документа пользователя. Если в контексте нет ответа — честно скажи, "
        "что в документе этой информации нет. Не выдумывай факты."
    )
    user_prompt = (
        f"Контекст из документа:\n{context}\n\n"
        f"Вопрос пользователя: {question}\n\n"
        "Ответь точно и кратко, опираясь только на контекст."
    )

    completion = await openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return completion.choices[0].message.content.strip()


def user_has_documents(user_id: int) -> bool:
    """Проверяет, есть ли у пользователя сохранённые чанки в Chroma."""
    try:
        result = collection.get(where={"user_id": str(user_id)}, limit=1)
        return bool(result.get("ids"))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Telegram-хендлеры
# ---------------------------------------------------------------------------

@dp.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(
        "Привет! Я бот с долгой памятью (RAG).\n\n"
        "1) Пришли документ: PDF, TXT или DOCX\n"
        "2) Задай вопрос по содержимому\n\n"
        "Я найду релевантные фрагменты в векторной базе и отвечу по документу."
    )


@dp.message(F.document)
async def handle_document(message: Message) -> None:
    """Шаг 1–3: сохранить файл → извлечь текст → эмбеддинги → ChromaDB."""
    document = message.document
    filename = document.file_name or "document.txt"
    suffix = Path(filename).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        await message.answer("Поддерживаются только файлы: PDF, TXT, DOCX.")
        return

    user_id = message.from_user.id
    local_path = DOWNLOADS_DIR / f"{user_id}_{uuid.uuid4().hex}{suffix}"

    await message.answer("Документ получен. Читаю и сохраняю в память…")

    try:
        # Скачиваем файл от Telegram
        await bot.download(document, destination=local_path)

        # Извлекаем текст
        text = await asyncio.to_thread(load_document, str(local_path))
        chunks = split_into_chunks(text, CHUNK_SIZE)

        if not chunks:
            await message.answer("Не удалось извлечь текст из файла.")
            return

        # Эмбеддинги + запись в ./memory (Chroma)
        saved = await embed_chunks(chunks, user_id, filename)
        await message.answer(
            f"Готово! Документ «{filename}» сохранён.\n"
            f"Фрагментов в памяти: {saved}.\n"
            "Теперь можете задавать вопросы по документу."
        )
    except Exception as e:
        await message.answer(f"Ошибка при обработке документа: {e}")
    finally:
        # Временный файл больше не нужен
        if local_path.exists():
            local_path.unlink(missing_ok=True)


@dp.message(F.text)
async def handle_question(message: Message) -> None:
    """Шаг 4–6: поиск контекста → ChatCompletion → ответ пользователю."""
    question = (message.text or "").strip()
    if not question:
        return

    user_id = message.from_user.id

    if not user_has_documents(user_id):
        await message.answer(
            "Сначала загрузите документ (PDF / TXT / DOCX), "
            "а потом задавайте вопросы."
        )
        return

    await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")

    try:
        reply = await answer_question(question, user_id)
        await message.answer(reply)
    except Exception as e:
        await message.answer(f"Ошибка при генерации ответа: {e}")


async def main() -> None:
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
