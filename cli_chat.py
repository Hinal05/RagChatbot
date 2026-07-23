"""Terminal chat UI for quick testing.

Usage: python cli_chat.py
"""
from app.chat import ChatSession


def main():
    print("Drupal RAG Chatbot (type 'exit' to quit)\n")
    session = ChatSession()
    while True:
        user_message = input("You: ").strip()
        if user_message.lower() in {"exit", "quit"}:
            break
        result = session.ask(user_message)
        print(f"\nBot: {result.answer}")
        if result.used_tool:
            print("  (used external tool)")
        if result.sources:
            print(f"  Sources: {', '.join(result.sources)}")
        print()


if __name__ == "__main__":
    main()
