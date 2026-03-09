"""Project Naraku - Gemini APIプロバイダ (New Official SDK 'google-genai'版)

Google Gen AI SDK (google-genai) を使用したプロバイダ。
Claudeのデバッグ指示に基づき、ツール呼び出しの確実性を向上。
"""

import asyncio
import logging
from typing import Any, AsyncGenerator

from google import genai
from google.genai import types

from .base import BaseLLMProvider
from services.mcp_server import read_secret_memory

logger = logging.getLogger(__name__)


class GeminiProvider(BaseLLMProvider):
    """Google Gemini APIプロバイダ (New SDK版)"""

    def __init__(self, api_key: str, default_model: str = "gemini-3.1-pro-preview"):
        self.api_key = api_key
        self.default_model = default_model
        self.client = genai.Client(api_key=self.api_key)

    def _convert_to_genai_contents(self, messages: list[dict]) -> tuple[str | None, list[types.Content]]:
        """内部メッセージ形式を google-genai の Content オブジェクトに変換"""
        system_instruction = None
        contents: list[types.Content] = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            parts_data = msg.get("parts")

            if role == "system":
                system_instruction = content
                continue

            genai_role = "user" if role == "user" else "model"
            parts: list[types.Part] = []

            if parts_data:
                for p in parts_data:
                    if "text" in p:
                        parts.append(types.Part.from_text(text=p["text"]))
                    elif "functionCall" in p:
                        fc = p["functionCall"]
                        parts.append(
                            types.Part.from_function_call(
                                name=fc["name"],
                                args=fc["args"],
                            )
                        )
                    elif "functionResponse" in p:
                        fr = p["functionResponse"]
                        parts.append(
                            types.Part.from_function_response(
                                name=fr["name"],
                                response=fr["response"],
                            )
                        )
            else:
                parts.append(types.Part.from_text(text=content))

            contents.append(types.Content(role=genai_role, parts=parts))

        return system_instruction, contents

    def _history_to_internal_messages(self, history: list[Any], system_instruction: str | None) -> list[dict]:
        """google-genai の履歴を内部メッセージ形式に戻す"""
        out: list[dict] = []
        if system_instruction:
            out.append({"role": "system", "content": system_instruction})

        for item in history or []:
            role = "user" if getattr(item, "role", "model") == "user" else "assistant"
            parts_list = []
            content_text = ""

            for p in getattr(item, "parts", []) or []:
                text = getattr(p, "text", None)
                function_call = getattr(p, "function_call", None)
                function_response = getattr(p, "function_response", None)

                if text:
                    content_text += text
                    parts_list.append({"text": text})
                elif function_call:
                    parts_list.append(
                        {
                            "functionCall": {
                                "name": function_call.name,
                                "args": function_call.args,
                            }
                        }
                    )
                elif function_response:
                    parts_list.append(
                        {
                            "functionResponse": {
                                "name": function_response.name,
                                "response": function_response.response,
                            }
                        }
                    )

            out.append(
                {
                    "role": role,
                    "content": content_text,
                    "parts": parts_list,
                }
            )

        return out

    def _get_chat_history(self, chat: Any, response: Any = None) -> list[Any]:
        """SDK差異を吸収して履歴を取得する"""
        if hasattr(chat, "get_history") and callable(chat.get_history):
            return chat.get_history()
        if hasattr(chat, "history"):
            return getattr(chat, "history")
        if response is not None and hasattr(response, "history"):
            return getattr(response, "history")
        logger.warning("[Gemini New SDK] Chat history API が見つからないため履歴更新をスキップ")
        return []

    async def send_message(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.8,
        max_tokens: int = 2048,
        allow_nsfw: bool = False,
    ) -> str:
        model_name = model or self.default_model
        system_instruction, contents = self._convert_to_genai_contents(messages)

        logger.info("[DEBUG] tools登録済み: %s", [t.__name__ for t in [read_secret_memory]])

        safety_settings = []
        if allow_nsfw:
            for cat in [
                types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            ]:
                safety_settings.append(
                    types.SafetySetting(
                        category=cat,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    )
                )

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_tokens,
            safety_settings=safety_settings,
            tools=[read_secret_memory],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False),
        )

        try:
            logger.info("[Gemini New SDK] メッセージ送信中 (model: %s)", model_name)

            chat = self.client.chats.create(
                model=model_name,
                history=contents[:-1] if len(contents) > 1 else [],
                config=config,
            )

            last_msg = contents[-1]
            response = await asyncio.to_thread(chat.send_message, message=last_msg.parts)

            history = self._get_chat_history(chat, response)
            if history:
                logger.info("[DEBUG] chat history length: %s", len(history))
                messages.clear()
                messages.extend(self._history_to_internal_messages(history, system_instruction))

            return response.text or ""
        except Exception as e:
            logger.error("[Gemini New SDK] エラー: %s", e)
            raise Exception(f"Gemini New SDK 通信エラー: {str(e)}")

    async def stream_message(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.8,
        max_tokens: int = 4096,
        allow_nsfw: bool = False,
    ) -> AsyncGenerator[str, None]:
        """新SDKでのストリーミング実装 (逐次yieldを保証)"""
        model_name = model or self.default_model
        system_instruction, contents = self._convert_to_genai_contents(messages)

        safety_settings = []
        if allow_nsfw:
            for cat in [
                types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            ]:
                safety_settings.append(
                    types.SafetySetting(
                        category=cat,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    )
                )

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_tokens,
            safety_settings=safety_settings,
            tools=[read_secret_memory],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False),
        )

        try:
            chat = self.client.chats.create(
                model=model_name,
                history=contents[:-1] if len(contents) > 1 else [],
                config=config,
            )
            last_msg = contents[-1]

            queue: asyncio.Queue[Any] = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def _worker() -> None:
                try:
                    for chunk in chat.send_message_stream(message=last_msg.parts):
                        text = getattr(chunk, "text", None)
                        if text:
                            asyncio.run_coroutine_threadsafe(queue.put(text), loop)
                except Exception as ex:
                    asyncio.run_coroutine_threadsafe(queue.put(ex), loop)
                finally:
                    asyncio.run_coroutine_threadsafe(queue.put(None), loop)

            import threading

            threading.Thread(target=_worker, daemon=True).start()

            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item

            history = self._get_chat_history(chat)
            if history:
                messages.clear()
                messages.extend(self._history_to_internal_messages(history, system_instruction))

        except Exception as e:
            logger.error("[Gemini New SDK Stream] エラー: %s", e)
            yield f"Error: {str(e)}"
