"""P1 demo: messages and prompt templates.

Run with::

    python examples/p1_messages_and_prompts.py
"""

from __future__ import annotations

from my_langchain.messages import AIMessage, HumanMessage
from my_langchain.prompts import ChatPromptTemplate, MessagesPlaceholder


def main() -> None:
    """Render a chat prompt that includes conversation history."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise assistant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "Answer in {language}: {question}"),
        ]
    )

    messages = prompt.format_messages(
        history=[HumanMessage(content="hi"), AIMessage(content="hello!")],
        language="English",
        question="what is LCEL?",
    )

    for message in messages:
        print(f"{message.type:>8}: {message.text}")

    print()
    print("input_variables:", prompt.input_variables)


if __name__ == "__main__":
    main()
