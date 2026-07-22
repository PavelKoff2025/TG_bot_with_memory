"""
Telegram-бот с короткой и долгой памятью.

Короткая память: последние N сообщений диалога в RAM (dict).
Долгая память: документы → эмбеддинги → ChromaDB (./memory).
База знаний компании: company.txt загружается при старте.
"""

import asyncio
import logging
import os
import re
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

import chromadb
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not BOT_TOKEN:
    raise ValueError("Не задан BOT_TOKEN в переменных окружения (.env)")
if not OPENAI_API_KEY:
    raise ValueError("Не задан OPENAI_API_KEY в переменных окружения (.env)")

# --- Настройки ---
HISTORY_LIMIT = 10
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50  # пересечение чанков — лучше качество RAG на границах
TOP_K = 5
CHAT_MODEL = os.getenv("CHAT_MODEL") or os.getenv("MODEL") or "gpt-3.5-turbo"
EMBED_MODEL = os.getenv("EMBED_MODEL") or "text-embedding-3-small"

MEMORY_DIR = Path("./memory")
DOWNLOADS_DIR = Path("./downloads")
COMPANY_FILE = Path("./company.txt")
COMPANY_USER_ID = "company"  # общая долгая память для всех пользователей

MEMORY_DIR.mkdir(exist_ok=True)
DOWNLOADS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("bot")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

chroma_client = chromadb.PersistentClient(path=str(MEMORY_DIR))
collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"},
)

# Короткая память: user_id -> последние сообщения
user_histories: Dict[int, Deque[dict]] = defaultdict(
    lambda: deque(maxlen=HISTORY_LIMIT)
)

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


# ---------------------------------------------------------------------------
# Короткая память
# ---------------------------------------------------------------------------

def get_history(user_id: int) -> List[dict]:
    return list(user_histories[user_id])


def add_to_history(user_id: int, role: str, content: str) -> None:
    user_histories[user_id].append({"role": role, "content": content})


def clear_history(user_id: int) -> None:
    user_histories[user_id].clear()


# ---------------------------------------------------------------------------
# Долгая память: документы / чанки / эмбеддинги
# ---------------------------------------------------------------------------

def load_document(file_path: str) -> str:
    """Читает PDF / TXT / DOCX и возвращает текст."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        # utf-8-sig убирает BOM, если файл сохранён из Windows/редактора
        return path.read_text(encoding="utf-8-sig", errors="ignore")

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    if suffix == ".docx":
        from docx import Document

        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)

    raise ValueError(f"Неподдерживаемый формат файла: {suffix}")


def split_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Делит текст на пересекающиеся чанки ~chunk_size символов."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    chunks: List[str] = []
    start = 0
    text_len = len(cleaned)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == text_len:
            break
        start = max(0, end - overlap)
    return chunks


async def delete_document_chunks(owner_id: str, doc_id: str) -> None:
    """Удаляет старые чанки документа перед повторной индексацией."""

    def _delete() -> None:
        existing = collection.get(
            where={
                "$and": [
                    {"user_id": str(owner_id)},
                    {"doc_id": doc_id},
                ]
            }
        )
        ids = existing.get("ids") or []
        if ids:
            collection.delete(ids=ids)

    await asyncio.to_thread(_delete)


async def embed_chunks(
    chunks: List[str],
    owner_id: str,
    filename: str,
    doc_id: Optional[str] = None,
) -> int:
    """
    Эмбеддинги чанков → ChromaDB (upsert).
    owner_id: user_id или 'company'; doc_id связывает чанки одного документа.
    """
    if not chunks:
        return 0

    doc_id = doc_id or uuid.uuid4().hex
    await delete_document_chunks(owner_id, doc_id)

    response = await openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=chunks,
    )
    embeddings = [item.embedding for item in response.data]

    # Стабильные id: повторная загрузка того же doc_id обновляет чанки
    ids = [f"{owner_id}:{doc_id}:{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "user_id": str(owner_id),
            "doc_id": doc_id,
            "filename": filename,
            "chunk_index": i,
        }
        for i in range(len(chunks))
    ]

    await asyncio.to_thread(
        collection.upsert,
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    return len(chunks)


async def retrieve_context(
    query: str,
    user_id: int,
    n_results: int = TOP_K,
    doc_id: Optional[str] = None,
) -> str:
    """
    Ищет релевантные фрагменты в долгой памяти:
    - общая база компании (company.txt)
    - документы конкретного пользователя
    Опционально можно сузить поиск до одного doc_id.
    """
    response = await openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=[query],
    )
    query_embedding = response.data[0].embedding

    def _query():
        if doc_id:
            where: dict = {
                "$and": [
                    {"user_id": str(user_id)},
                    {"doc_id": doc_id},
                ]
            }
        else:
            where = {
                "$or": [
                    {"user_id": COMPANY_USER_ID},
                    {"user_id": str(user_id)},
                ]
            }
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas"],
        )

    try:
        results = await asyncio.to_thread(_query)
    except Exception:
        # Если в базе ещё ничего нет — Chroma может ругаться на where
        logger.exception("retrieve_context failed")
        return ""

    documents = results.get("documents") or []
    if not documents or not documents[0]:
        return ""

    parts = [f"[{i}] {doc}" for i, doc in enumerate(documents[0], start=1)]
    return "\n\n".join(parts)


def has_any_long_memory(user_id: int) -> bool:
    """Есть ли хоть какие-то документы (компания или пользователя)."""
    try:
        result = collection.get(
            where={
                "$or": [
                    {"user_id": COMPANY_USER_ID},
                    {"user_id": str(user_id)},
                ]
            },
            limit=1,
        )
        return bool(result.get("ids"))
    except Exception:
        return False


async def ensure_company_indexed() -> None:
    """При старте индексирует company.txt в долгую память (один раз)."""
    if not COMPANY_FILE.exists():
        logger.warning("Файл company.txt не найден — пропускаю индексацию")
        return

    company_doc_id = "company_txt"
    existing = await asyncio.to_thread(
        collection.get,
        where={"user_id": COMPANY_USER_ID},
        limit=1,
    )
    if existing.get("ids"):
        logger.info("company.txt уже в долгой памяти")
        return

    text = load_document(str(COMPANY_FILE))
    chunks = split_into_chunks(text)
    if not chunks:
        logger.warning("company.txt пуст — нечего индексировать")
        return

    count = await embed_chunks(
        chunks,
        COMPANY_USER_ID,
        "company.txt",
        doc_id=company_doc_id,
    )
    logger.info("company.txt проиндексирован: %s чанков", count)


# ---------------------------------------------------------------------------
# Ответ с обеими памятями
# ---------------------------------------------------------------------------

async def answer_with_both_memories(user_id: int, user_text: str) -> str:
    """
    Собирает:
    1) контекст из долгой памяти (RAG)
    2) историю диалога (короткая память)
    3) текущий вопрос
    и отправляет в Chat Completions.
    """
    long_context = await retrieve_context(user_text, user_id)

    system_parts = [
        "Ты ассистент компании «ГК Проект» в Telegram.",
        "Учитывай историю текущего диалога (короткая память).",
        "Если есть контекст из документов — опирайся на него в первую очередь.",
        "Не выдумывай факты о компании, услугах и ценах.",
        "Если в контексте и истории нет ответа — честно скажи об этом.",
        "Отвечай кратко и по делу на русском языке.",
    ]
    if long_context:
        system_parts.append(
            "\nКонтекст из долгой памяти (документы):\n" + long_context
        )

    messages = [
        {"role": "system", "content": "\n".join(system_parts)},
        *get_history(user_id),
        {"role": "user", "content": user_text},
    ]

    completion = await openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
    )
    return (completion.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# Telegram-хендлеры
# ---------------------------------------------------------------------------

@dp.message(CommandStart())
async def cmd_start(message: Message) -> None:
    clear_history(message.from_user.id)
    await message.answer(
        "Привет! Я бот с короткой и долгой памятью.\n\n"
        f"• Короткая память: последние {HISTORY_LIMIT} сообщений диалога\n"
        "• Долгая память: база компании + ваши документы (PDF/TXT/DOCX)\n\n"
        "Можете сразу спрашивать про «ГК Проект» "
        "или загрузить свой документ.\n"
        "Команды: /start — сброс диалога, /clear — очистить короткую память"
    )


@dp.message(Command("clear"))
async def cmd_clear(message: Message) -> None:
    clear_history(message.from_user.id)
    await message.answer("Короткая память диалога очищена.")


@dp.message(F.document)
async def handle_document(message: Message) -> None:
    """Загрузка документа в долгую память пользователя."""
    document = message.document
    filename = document.file_name or "document.txt"
    suffix = Path(filename).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        await message.answer("Поддерживаются только файлы: PDF, TXT, DOCX.")
        return

    user_id = message.from_user.id
    local_path = DOWNLOADS_DIR / f"{user_id}_{uuid.uuid4().hex}{suffix}"

    await message.answer("Документ получен. Сохраняю в долгую память…")

    try:
        await bot.download(document, destination=local_path)
        text = await asyncio.to_thread(load_document, str(local_path))
        chunks = split_into_chunks(text)

        if not chunks:
            await message.answer("Не удалось извлечь текст из файла.")
            return

        # Стабильный doc_id по имени файла: повторная загрузка обновит документ
        doc_id = Path(filename).stem.lower().replace(" ", "_")[:64] or uuid.uuid4().hex
        saved = await embed_chunks(chunks, str(user_id), filename, doc_id=doc_id)
        await message.answer(
            f"Документ «{filename}» сохранён в долгую память.\n"
            f"Фрагментов: {saved}.\n"
            "Можете задавать вопросы — учту и документ, и наш диалог."
        )
    except Exception as e:
        await message.answer(f"Ошибка при обработке документа: {e}")
    finally:
        if local_path.exists():
            local_path.unlink(missing_ok=True)


@dp.message(F.text)
async def handle_text(message: Message) -> None:
    """Ответ с короткой + долгой памятью."""
    user_text = (message.text or "").strip()
    if not user_text:
        return

    user_id = message.from_user.id
    await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")

    try:
        reply = await answer_with_both_memories(user_id, user_text)
    except Exception as e:
        await message.answer(f"Ошибка при обращении к модели: {e}")
        return

    add_to_history(user_id, "user", user_text)
    add_to_history(user_id, "assistant", reply)
    await message.answer(reply)


async def main() -> None:
    await ensure_company_indexed()
    logger.info("Бот с короткой и долгой памятью запущен")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
