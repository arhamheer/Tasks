import streamlit as st

from chatbot import HealthChatbot


st.set_page_config(page_title="Task 4 - Health Query Chatbot", layout="centered")


def main() -> None:
    st.title("Task 4: General Health Query Chatbot")
    st.write("Friendly health information bot with prompt engineering and safety checks.")

    bot = HealthChatbot()
    provider = bot.get_provider()
    readiness = bot.provider_readiness()

    st.caption(f"Active provider: {provider}")
    st.caption(
        "Provider readiness -> "
        f"OpenAI: {readiness['openai']} | "
        f"Azure OpenAI: {readiness['azure_openai']} | "
        f"Hugging Face: {readiness['huggingface']}"
    )

    if "chat" not in st.session_state:
        st.session_state.chat = []

    query = st.text_input("Ask a health question")

    if st.button("Send") and query.strip():
        answer = bot.ask(query)
        st.session_state.chat.append((query, answer))

    for q, a in reversed(st.session_state.chat):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")
        st.markdown("---")

    st.caption("Safety note: this is general information only, not a medical diagnosis.")


if __name__ == "__main__":
    main()
