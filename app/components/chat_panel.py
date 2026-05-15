from app.components.assistant_chat import render_assistant_chat


def render_chat_panel() -> None:
    """Render the styled chat panel using the custom frontend chat component."""
    render_assistant_chat(
        intro_message="Hello. I’m the full page assistant. Try me.",
        height=720,
        component_key="assistant_chat_widget",
        last_processed_key="assistant_tab_last_processed_message",
        placeholder="Ask the assistant something...",
        send_label="Send",
        surface="full",
    )
