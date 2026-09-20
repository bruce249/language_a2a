from __future__ import annotations

from typing import Any

import httpx
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import get_stream_response_text, new_text_message
from a2a.types import Role, SendMessageRequest, StreamResponse


def extract_text(chunks: list[Any]) -> str:
    texts: list[str] = []
    for chunk in chunks:
        if isinstance(chunk, StreamResponse):
            blob = get_stream_response_text(chunk)
        elif hasattr(chunk, "HasField"):
            blob = get_stream_response_text(chunk)
        else:
            blob = str(chunk)
        blob = blob.strip()
        if blob and blob not in {"working", "Processing request..."}:
            texts.append(blob)
    if not texts:
        return ""
    return max(texts, key=len)


async def fetch_card(base_url: str) -> Any:
    async with httpx.AsyncClient(timeout=30) as client:
        resolver = A2ACardResolver(httpx_client=client, base_url=base_url)
        return await resolver.get_agent_card()


async def send_text(base_url: str, text: str) -> tuple[str, list[Any]]:
    async with httpx.AsyncClient(timeout=30) as http_client:
        resolver = A2ACardResolver(httpx_client=http_client, base_url=base_url)
        card = await resolver.get_agent_card()
        client = await create_client(agent=card, client_config=ClientConfig(streaming=False))
        try:
            request = SendMessageRequest(
                message=new_text_message(text, role=Role.ROLE_USER)
            )
            chunks: list[Any] = []
            async for chunk in client.send_message(request):
                chunks.append(chunk)
            return extract_text(chunks), chunks
        finally:
            await client.close()


async def load_private_instance(base_url: str, payload: dict[str, Any]) -> None:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(f"{base_url}/internal/load", json=payload)
        response.raise_for_status()


def card_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def task_ids(chunks: list[Any]) -> dict[str, str]:
    for chunk in chunks:
        task = getattr(chunk, "task", None)
        if task and getattr(task, "id", None):
            return {
                "task_id": task.id,
                "context_id": task.context_id,
                "final_state": str(task.status.state),
            }
    return {}
