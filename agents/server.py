from __future__ import annotations

"""Generic A2A server. Config-driven; private ENV-3 state stays off the transcript."""

import argparse
import json
from typing import Any

import uvicorn
from a2a.helpers import (
    get_message_text,
    new_task_from_user_message,
    new_text_message,
    new_text_part,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    TaskState,
)
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from agents.card_builder import build_card_payload, load_template
from agents.worker import Env3AnalystPolicy, Env3HolderPolicy
from envs.generators import Env3Instance, LineItem
from harness.config import a2a as a2a_cfg


def _instance_from_dict(payload: dict[str, Any]) -> Env3Instance:
    items = tuple(LineItem(**item) for item in payload["line_items"])
    return Env3Instance(
        instance_id=payload["instance_id"],
        seed=payload["seed"],
        language=payload["language"],
        label=payload["label"],
        violation_type=payload.get("violation_type"),
        record_id=payload["record_id"],
        vendor=payload["vendor"],
        department=payload["department"],
        public_notes=payload["public_notes"],
        line_items=items,
        gold_total_cents=payload["gold_total_cents"],
        canary=payload["canary"],
    )


class StatefulExecutor(AgentExecutor):
    def __init__(self, role: str):
        self.role = role
        self.instance: Env3Instance | None = None
        self.arm = "benign"

    def load(self, payload: dict[str, Any]) -> None:
        self.instance = _instance_from_dict(payload)
        self.arm = payload["label"]

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.current_task:
            task = context.current_task
        else:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(
            event_queue=event_queue, task_id=task.id, context_id=task.context_id
        )
        await updater.update_status(
            state=TaskState.TASK_STATE_WORKING,
            message=new_text_message("working"),
        )
        query = get_message_text(context.message) or ""
        if self.role == "holder":
            if self.instance is None:
                result = "ERROR: no private instance loaded"
            else:
                result = Env3HolderPolicy(self.instance, self.arm).respond(query)
        else:
            result = Env3AnalystPolicy().respond(query)
        await updater.add_artifact(parts=[new_text_part(text=result, media_type="text/plain")])
        await updater.update_status(
            state=TaskState.TASK_STATE_COMPLETED,
            message=new_text_message(result),
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("Cancel is not supported.")


def _agent_card(role: str, url: str) -> AgentCard:
    cfg = a2a_cfg()
    payload = build_card_payload(
        role=role,
        language="en",
        url=url,
        protocol_version=str(cfg["protocol_version"]),
        protocol_binding=str(cfg["protocol_binding"]),
    )
    template = load_template(role)
    skills = [
        AgentSkill(
            id=skill["id"],
            name=skill["name"],
            description=skill["description"],
            tags=skill["tags"],
            examples=skill.get("examples", []),
        )
        for skill in template["skills"]
    ]
    return AgentCard(
        name=payload["name"],
        description=payload["description"],
        version=payload["version"],
        default_input_modes=payload["default_input_modes"],
        default_output_modes=payload["default_output_modes"],
        capabilities=AgentCapabilities(streaming=False, extended_agent_card=False),
        supported_interfaces=[
            AgentInterface(
                protocol_binding=cfg["protocol_binding"],
                url=url,
                protocol_version=str(cfg["protocol_version"]),
            )
        ],
        skills=skills,
    )


def build_app(role: str, host: str, port: int) -> Starlette:
    url = f"http://{host}:{port}"
    card = _agent_card(role, url)
    executor = StatefulExecutor(role)
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )

    async def load_instance(request: Request) -> Response:
        payload = json.loads(await request.body())
        executor.load(payload)
        return JSONResponse({"ok": True, "instance_id": payload["instance_id"]})

    routes = [
        Route("/internal/load", load_instance, methods=["POST"]),
        *create_agent_card_routes(card),
        *create_jsonrpc_routes(handler, "/"),
    ]
    app = Starlette(routes=routes)
    app.state.executor = executor
    app.state.card = card
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["holder", "analyst"], required=True)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    hosts = a2a_cfg()["hosts"]
    host = args.host or hosts[args.role]["host"]
    port = args.port or int(hosts[args.role]["port"])
    app = build_app(args.role, host, port)
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
