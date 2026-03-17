import streamlit as st

from chatbot import MentalHealthSupportBot


st.set_page_config(page_title="Task 5 - Mental Health Support", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

    :root {
        --ink: #0f172a;
        --ink-soft: #334155;
        --surface: #ffffff;
        --surface-2: #f8fafc;
        --line: #cbd5e1;
        --brand: #0ea5e9;
        --bot-bg: #eff6ff;
        --user-bg: #ecfeff;
    }

    .stApp {
        background:
            radial-gradient(circle at 16% 16%, rgba(250, 204, 21, 0.20), transparent 30%),
            radial-gradient(circle at 82% 14%, rgba(14, 165, 233, 0.22), transparent 28%),
            linear-gradient(140deg, #f8fafc, #f1f5f9 45%, #ecfeff 100%);
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
    }

    p, span, label, div {
        color: var(--ink);
    }

    .stTextInput > div > div > input {
        background: var(--surface);
        color: var(--ink);
        border: 1px solid var(--line);
    }

    .stButton > button {
        background: var(--surface);
        color: var(--ink);
        border: 1px solid var(--line);
        font-weight: 700;
    }

    .stButton > button:hover {
        border-color: var(--brand);
        color: #0369a1;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: 0.2px;
        color: #020617;
    }

    .hero {
        background: #ffffffde;
        border: 1px solid var(--line);
        backdrop-filter: blur(4px);
        border-radius: 18px;
        padding: 1.2rem 1.35rem;
        box-shadow: 0 10px 24px rgba(2, 6, 23, 0.07);
        animation: fadein 0.7s ease;
    }

    .chat-card-user {
        background: var(--user-bg);
        border-radius: 14px;
        border: 1px solid #a5f3fc;
        color: var(--ink);
        padding: 0.9rem 1rem;
        margin-bottom: 0.5rem;
        animation: rise 0.35s ease;
    }

    .chat-card-bot {
        background: var(--bot-bg);
        border-radius: 14px;
        border: 1px solid #bfdbfe;
        color: var(--ink);
        padding: 0.9rem 1rem;
        margin-bottom: 1rem;
        animation: rise 0.5s ease;
    }

    .chat-card-user b,
    .chat-card-bot b {
        color: #0b3a68;
    }

    @keyframes fadein {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0px); }
    }

    @keyframes rise {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_bot():
    if "bot" not in st.session_state:
        st.session_state.bot = MentalHealthSupportBot()
    if "messages" not in st.session_state:
        st.session_state.messages = []


def main() -> None:
    init_bot()

    st.markdown(
        """
        <div class='hero'>
        <h1>Task 5: Mental Health Support Chatbot</h1>
        <p>This assistant is tuned for supportive, empathetic responses about stress and emotional wellness.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_source = st.session_state.bot.model_source
    source_label = "Fine-tuned local model" if "model" in model_source else "Base model fallback"
    st.caption(f"Model source: {source_label}")

    user_text = st.text_input("How are you feeling today?")

    col1, col2 = st.columns([1, 1])
    with col1:
        send = st.button("Get Supportive Reply", use_container_width=True)
    with col2:
        clear = st.button("Clear Chat", use_container_width=True)

    if clear:
        st.session_state.messages = []

    if send and user_text.strip():
        reply = st.session_state.bot.generate(user_text)
        st.session_state.messages.append((user_text, reply))

    for user_msg, bot_msg in reversed(st.session_state.messages):
        st.markdown(f"<div class='chat-card-user'><b>You:</b> {user_msg}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-card-bot'><b>Support Bot:</b> {bot_msg}</div>", unsafe_allow_html=True)

    st.info(
        "This tool is for emotional support and general wellness guidance, not medical diagnosis or crisis care. "
        "If you are in immediate danger, contact local emergency services."
    )


if __name__ == "__main__":
    main()
