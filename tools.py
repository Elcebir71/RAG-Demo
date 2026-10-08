"""Tool registry: what the agent may propose, and the rules the application enforces.

The model only sees each tool's name, description and JSON schema.
Risk level, validation and business rules live here, in code, out of the model's reach.
"""

from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from config import EMAIL_ALLOWLIST


class Risk(str, Enum):
    READ = "read"  # runs automatically and changes nothing
    LOW = "low"    # runs automatically and is logged
    HIGH = "high"  # waits for human approval


class StrictArgs(BaseModel):
    # Unknown fields are rejected, not ignored: the model cannot add options we did not define.
    model_config = ConfigDict(extra="forbid")


class SearchArgs(StrictArgs):
    query: str = Field(min_length=1, max_length=300)


class NoteArgs(StrictArgs):
    title: str = Field(min_length=1, max_length=100)
    text: str = Field(min_length=1, max_length=2000)


class EmailArgs(StrictArgs):
    to: str = Field(min_length=3, max_length=254)
    subject: str = Field(min_length=1, max_length=150)
    body: str = Field(min_length=1, max_length=5000)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args: type[StrictArgs]
    risk: Risk


TOOLS = {tool.name: tool for tool in [
    Tool("search_documents", "Search the user's documents and return the most relevant passages.",
         SearchArgs, Risk.READ),
    Tool("save_note", "Save a short note for the user.", NoteArgs, Risk.LOW),
    Tool("send_email", "Send an email. A human must approve it before it is sent.",
         EmailArgs, Risk.HIGH),
]}


class Rejected(Exception):
    """The application refuses a proposed call. The reason is logged and returned to the model."""


def tool_definitions():
    """The schemas the model sees, generated from the Pydantic models so they can never drift apart."""
    return [
        {"type": "function", "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.args.model_json_schema(),
        }}
        for tool in TOOLS.values()
    ]


def validate(call):
    """Check one proposed call against the registry and the business rules. Return (tool, args)."""
    tool = TOOLS.get(call.name)
    if tool is None:
        raise Rejected(f"Unknown tool '{call.name}'.")
    if call.arguments is None:
        raise Rejected("Arguments could not be parsed.")
    try:
        args = tool.args.model_validate(call.arguments)
    except ValidationError as error:
        problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'arguments'}: {e['msg']}" for e in error.errors())
        raise Rejected(f"Invalid arguments: {problems}") from error

    # Business rule, enforced in code: whatever a prompt or a document says, only allowlisted recipients.
    if isinstance(args, EmailArgs) and args.to.strip().lower() not in EMAIL_ALLOWLIST:
        raise Rejected(f"Recipient '{args.to}' is not on the allowlist.")
    return tool, args