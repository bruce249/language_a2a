from __future__ import annotations

import json
from typing import Any, Literal

from envs.env1_split import Env1Instance
from envs.generators import Env3Instance, analyst_reply, holder_reply, parse_total
from harness.grok import grok_complete
from harness.prompts import (
    analyst_system,
    client_prompt,
    env1_alpha_system,
    env1_beta_system,
    env1_client_prompt,
    env1_solo_system,
    holder_system,
    solo_system,
)


class Env3HolderPolicy:
    def __init__(self, instance: Env3Instance, arm: Literal["benign", "violating"]):
        self.instance = instance
        self.arm = arm

    def respond(self, _user_text: str) -> str:
        return holder_reply(self.instance, self.arm)


class Env3AnalystPolicy:
    def respond(self, user_text: str) -> str:
        return analyst_reply(user_text)


class GrokHolderPolicy:
    def __init__(self, instance: Env3Instance, model: str):
        self.instance = instance
        self.model = model

    def respond(self, user_text: str) -> dict[str, Any]:
        return grok_complete(
            model=self.model,
            system=holder_system(
                self.instance.language,
                self.instance.private_record,
                leak=self.instance.label == "violating",
            ),
            user=user_text,
        )


class GrokSoloPolicy:
    def __init__(self, instance: Env3Instance, model: str):
        self.instance = instance
        self.model = model

    def respond(self) -> dict[str, Any]:
        return grok_complete(
            model=self.model,
            system=solo_system(self.instance.language, self.instance.public_record),
            user=f"Record id: {self.instance.record_id}. Emit TOTAL=<integer> as the last line.",
        )


class GrokAnalystPolicy:
    def __init__(self, language: str, model: str):
        self.language = language
        self.model = model

    def respond(self, user_text: str) -> dict[str, Any]:
        return grok_complete(
            model=self.model,
            system=analyst_system(self.language),
            user=user_text,
        )


def parse_analyst_total(text: str) -> int | None:
    return parse_total(text)


def dump_private_state(instance: Env3Instance) -> str:
    """Server-side only. Never written into Task.history."""
    return json.dumps(instance.private_record, indent=2)


def localized_client_message(instance: Env3Instance) -> str:
    return client_prompt(instance.language, instance.record_id)


class GrokEnv1SoloPolicy:
    def __init__(self, instance: Env1Instance, model: str):
        self.instance = instance
        self.model = model

    def respond(self) -> dict[str, Any]:
        return grok_complete(
            model=self.model,
            system=env1_solo_system(self.instance.language, self.instance.full_record),
            user=f"Record id: {self.instance.record_id}. Emit SLOT=<TIME>@<ROOM> as the last line.",
        )


class GrokEnv1AlphaPolicy:
    def __init__(self, instance: Env1Instance, model: str):
        self.instance = instance
        self.model = model

    def respond(self, user_text: str) -> dict[str, Any]:
        return grok_complete(
            model=self.model,
            system=env1_alpha_system(self.instance.language, self.instance.person_record),
            user=user_text,
        )


class GrokEnv1BetaPolicy:
    def __init__(self, instance: Env1Instance, model: str):
        self.instance = instance
        self.model = model

    def respond(self, user_text: str) -> dict[str, Any]:
        return grok_complete(
            model=self.model,
            system=env1_beta_system(self.instance.language, self.instance.room_record),
            user=user_text,
        )


def localized_env1_client(instance: Env1Instance) -> str:
    return env1_client_prompt(instance.language, instance.record_id)
