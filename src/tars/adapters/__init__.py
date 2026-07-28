"""Source adapters. Each one maps a native format into schema.Conversation.

Adding an adapter must never require touching the judge, the store, or the API.
If it does, the canonical schema is leaking and something is wrong.
"""

from . import abcd, synthetic

REGISTRY = {"abcd": abcd, "synthetic": synthetic}
