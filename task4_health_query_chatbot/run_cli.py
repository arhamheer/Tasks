from chatbot import HealthChatbot


def main() -> None:
    bot = HealthChatbot()

    print("General Health Query Chatbot (type 'exit' to stop)")
    while True:
        query = input("\nYou: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        answer = bot.ask(query)
        print(f"Bot: {answer}")


if __name__ == "__main__":
    main()
