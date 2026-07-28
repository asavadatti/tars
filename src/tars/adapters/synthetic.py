"""Synthetic adapter.

Exists so the pipeline can be smoke-tested with no dataset and no API key, and
so specific failure modes can be constructed on demand for the demo.
"""

from __future__ import annotations

from ..schema import Conversation, Speaker, Turn

SOURCE = "synthetic"

_FIXTURES = {
    "clean_resolution": [
        ("customer", "Hi, my order 4471 hasn't shipped yet and it's been a week."),
        ("agent", "That's frustrating when you're waiting on something. Can I get your name and the email on the account?"),
        ("customer", "Dana Whitfield, dana.w@example.com"),
        ("agent", "Thanks Dana. I see it was held at the warehouse. I've released it and it ships today."),
        ("customer", "Oh that's a relief, thank you."),
    ],
    "unverified_refund": [
        ("customer", "I want a refund on my last order."),
        ("agent", "Sure, I've processed a full refund to your card."),
        ("customer", "Great, thanks."),
    ],
    "empathy_miss": [
        ("customer", "This is the third time I've had to contact you about this. I'm honestly done."),
        ("agent", "Please provide your order number."),
        ("customer", "8812."),
        ("agent", "Ticket escalated. Someone will contact you within 5 business days."),
    ],
    "no_stated_goal": [
        ("customer", "hi"),
        ("agent", "Hello, how can I help?"),
        ("customer", "actually never mind"),
    ],
}


def load(limit: int | None = None):
    for i, (key, rows) in enumerate(_FIXTURES.items()):
        if limit is not None and i >= limit:
            return
        yield Conversation(
            conversation_id=f"synthetic-{key}",
            source=SOURCE,
            turns=[
                Turn(idx=j, speaker=Speaker(s), text=t) for j, (s, t) in enumerate(rows)
            ],
            source_metadata={"fixture": key},
        )
