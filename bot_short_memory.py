"""
Telegram-бот с короткой памятью (history buffer).
Хранит последние N сообщений диалога в оперативной памяти (dict).
"""

import os
from collections import defaultdict, deque
from typing import Deque, Dict, List

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Загружаем переменные из .env
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not BOT_TOKEN:
    raise ValueError("Не задан BOT_TOKEN в переменных окружения (.env)")
if not OPENAI_API_KEY:
    raise ValueError("Не задан OPENAI_API_KEY в переменных окружения (.env)")

# --- Настройки ---
HISTORY_LIMIT = 10  # сколько последних сообщений храним на пользователя
MODEL_NAME = os.getenv("MODEL") or os.getenv("CHAT_MODEL") or "gpt-3.5-turbo"

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Короткая память: user_id -> deque из сообщений {"role": "...", "content": "..."}
user_histories: Dict[int, Deque[dict]] = defaultdict(
    lambda: deque(maxlen=HISTORY_LIMIT)
)

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "Ты полезный ассистент в Telegram. "
        "Отвечай кратко и по делу, учитывая историю диалога."
    ),
}


def get_history(user_id: int) -> List[dict]:
    """Возвращает историю диалога пользователя как список сообщений."""
    return list(user_histories[user_id])


def add_to_history(user_id: int, role: str, content: str) -> None:
    """Добавляет сообщение в короткую память пользователя."""
    user_histories[user_id].append({"role": role, "content": content})


async def ask_openai(user_id: int, user_text: str) -> str:
    """
    Отправляет в OpenAI системный промпт + историю + текущее сообщение
    и возвращает ответ модели.
    """
    messages = [SYSTEM_PROMPT] + get_history(user_id) + [
        {"role": "user", "content": user_text}
    ]

    response = await openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
    )
    return response.choices[0].message.content.strip()


@dp.message(CommandStart())
async def cmd_start(message: Message) -> None:
    """Обработчик /start — приветствие и сброс истории."""
    user_histories[message.from_user.id].clear()
    await message.answer(
        "Привет! Я бот с короткой памятью.\n"
        f"Помню последние {HISTORY_LIMIT} сообщений нашего диалога.\n"
        "Просто напиши что-нибудь."
    )


@dp.message(F.text)
async def handle_text(message: Message) -> None:
    """
    Основная логика:
    1) берём текст пользователя
    2) спрашиваем модель с историей
    3) отвечаем и обновляем историю
    """
    user_id = message.from_user.id
    user_text = message.text.strip()

    if not user_text:
        return

    await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")

    try:
        reply = await ask_openai(user_id, user_text)
    except Exception as e:
        await message.answer(f"Ошибка при обращении к модели: {e}")
        return

    # Сохраняем и вопрос, и ответ в короткую память
    add_to_history(user_id, "user", user_text)
    add_to_history(user_id, "assistant", reply)

    await message.answer(reply)


async def main() -> None:
    """Точка входа: запускаем polling."""
    await dp.start_polling(bot)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
