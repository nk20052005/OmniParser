"""
Gradio Web Interface for Omnitool Enterprise.

This is the in-VM GUI — users interact with the system
through a chat interface. The same conversation engine is used
as with Slack, ensuring consistent behavior across interfaces.
"""

import asyncio
import logging
from typing import Any

import gradio as gr

from ..engine.conversation import ConversationEngine, ConversationResponse

logger = logging.getLogger(__name__)


def create_gradio_app(engine: ConversationEngine) -> gr.Blocks:
    """Create the Gradio Blocks application."""

    custom_css = """
    .omnitool-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white;
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 16px;
    }
    .omnitool-header h1 { margin: 0; font-size: 1.8em; }
    .omnitool-header p { margin: 5px 0 0 0; opacity: 0.8; font-size: 0.95em; }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: 600;
    }
    .status-connected { background: #27ae60; color: white; }
    .status-pending { background: #f39c12; color: white; }
    """

    with gr.Blocks(
        title="Omnitool Enterprise",
        theme=gr.themes.Soft(),
        css=custom_css,
    ) as app:
        # Header
        gr.HTML("""
        <div class="omnitool-header">
            <h1>🔧 Omnitool Enterprise</h1>
            <p>Autonomous Enterprise Operations Platform — powered by Gemma 4</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=550,
                    type="messages",
                    show_copy_button=True,
                    avatar_images=(None, "https://img.icons8.com/color/48/robot-2.png"),
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask me anything... (e.g., 'Start VM PROD-WEB-01', 'Create a Sev2 incident')",
                        label="Message",
                        scale=5,
                        lines=1,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("Clear Chat", variant="secondary", size="sm")
                    examples_btn = gr.Button("Show Examples", size="sm")

            with gr.Column(scale=1):
                gr.Markdown("### System Status")
                status_display = gr.Markdown("Checking connections...")

                gr.Markdown("### Quick Actions")
                with gr.Group():
                    qa_vm_status = gr.Button("📊 Check VM Status", size="sm")
                    qa_incidents = gr.Button("🎫 Show Open Incidents", size="sm")
                    qa_alerts = gr.Button("🔔 Check Alerts", size="sm")
                    qa_emails = gr.Button("📧 Read Unread Emails", size="sm")
                    qa_dashboard = gr.Button("📈 Generate Dashboard", size="sm")

                gr.Markdown("### Recent Actions")
                recent_actions = gr.Markdown("_No recent actions_")

        # Hidden state
        user_id_state = gr.State("gradio_user")
        channel_id_state = gr.State("gradio_default")

        # Example prompts
        gr.Examples(
            examples=[
                ["Bring the production web server online"],
                ["Create a Sev2 incident for database latency"],
                ["Show me all unresolved incidents"],
                ["Send an email to the network team explaining the outage"],
                ["Check if any VMs are underutilized"],
                ["Generate RCA for yesterday's outage"],
                ["Create a service request for additional storage"],
                ["Read my unread emails"],
                ["Can you shut down the VM that is costing the most?"],
            ],
            inputs=msg_input,
            label="Example Prompts",
        )

        # Event handlers
        async def process_message(
            message: str,
            history: list,
            user_id: str,
            channel_id: str,
        ):
            if not message.strip():
                return "", history, ""

            # Add user message to history
            history = history + [{"role": "user", "content": message}]

            # Process through conversation engine
            response = await engine.process_message(
                user_message=message,
                user_id=user_id,
                channel_id=channel_id,
            )

            # Build response text with status indicators
            response_text = response.message
            if response.awaiting_approval:
                response_text += "\n\n⚠️ **Awaiting your confirmation** — reply 'yes' to proceed or 'no' to cancel."
            elif response.awaiting_input:
                response_text += "\n\n💬 _Please provide the requested information._"

            history = history + [{"role": "assistant", "content": response_text}]

            # Update recent actions
            action_info = ""
            if response.tool_called:
                action_info = f"✅ **{response.tool_called}** executed"
            elif response.awaiting_approval:
                action_info = f"⏳ Awaiting approval for **{response.intent.value if response.intent else 'action'}**"

            return "", history, action_info

        def sync_process(message, history, user_id, channel_id):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    process_message(message, history, user_id, channel_id)
                )
            finally:
                loop.close()

        # Wire up events
        send_btn.click(
            fn=sync_process,
            inputs=[msg_input, chatbot, user_id_state, channel_id_state],
            outputs=[msg_input, chatbot, recent_actions],
        )
        msg_input.submit(
            fn=sync_process,
            inputs=[msg_input, chatbot, user_id_state, channel_id_state],
            outputs=[msg_input, chatbot, recent_actions],
        )
        clear_btn.click(
            fn=lambda: ([], "_No recent actions_"),
            outputs=[chatbot, recent_actions],
        )

        # Quick action handlers
        def quick_action(prompt):
            def handler(history, user_id, channel_id):
                return sync_process(prompt, history, user_id, channel_id)
            return handler

        for btn, prompt in [
            (qa_vm_status, "List all VMs and their status"),
            (qa_incidents, "Show me all open incidents"),
            (qa_alerts, "Check for any active monitoring alerts"),
            (qa_emails, "Read my unread emails"),
            (qa_dashboard, "Generate an operations dashboard"),
        ]:
            btn.click(
                fn=lambda h, u, c, p=prompt: sync_process(p, h, u, c),
                inputs=[chatbot, user_id_state, channel_id_state],
                outputs=[msg_input, chatbot, recent_actions],
            )

    return app
