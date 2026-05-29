import asyncio
import gradio as gr
from sidekick import Sidekick


async def setup():
    sidekick = Sidekick()
    await sidekick.setup()
    return sidekick


async def process_message(sidekick, message, success_criteria, history):
    if not message.strip():
        return history, "", message, sidekick  # nothing to do, preserve input
    # Lazy init: the Gradio Python client doesn't always fire ui.load
    if sidekick is None:
        sidekick = await setup()
    updated_history, plan = await sidekick.run(message, success_criteria, history)
    return updated_history, plan, "", sidekick  # clear the input box after sending


async def reset(old_sidekick):
    # Properly await cleanup before creating a new instance
    if old_sidekick:
        await old_sidekick.cleanup()
    new_sidekick = Sidekick()
    await new_sidekick.setup()
    return "", "", [], "", new_sidekick


def free_resources(sidekick):
    """Gradio delete_callback — best-effort async cleanup from sync context."""
    if not sidekick:
        return
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(sidekick.cleanup())
        else:
            loop.run_until_complete(sidekick.cleanup())
    except Exception as e:
        print(f"Cleanup error: {e}")


with gr.Blocks(title="Sidekick", theme=gr.themes.Default(primary_hue="emerald")) as ui:
    gr.Markdown("## Sidekick — Autonomous AI Agent")
    sidekick_state = gr.State(delete_callback=free_resources)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Conversation", height=480, type="messages")
        with gr.Column(scale=1):
            plan_box = gr.Textbox(
                label="Execution Plan",
                placeholder="Plan will appear here after the agent starts...",
                lines=16,
                interactive=False,
            )

    with gr.Group():
        with gr.Row():
            message = gr.Textbox(
                show_label=False,
                placeholder="Your request to the Sidekick...",
                scale=4,
            )
        with gr.Row():
            success_criteria = gr.Textbox(
                show_label=False,
                placeholder="Success criteria (optional — e.g. 'include a code example')",
            )

    with gr.Row():
        reset_button = gr.Button("Reset", variant="stop")
        go_button = gr.Button("Go!", variant="primary")

    ui.load(setup, [], [sidekick_state])

    go_button.click(
        process_message,
        [sidekick_state, message, success_criteria, chatbot],
        [chatbot, plan_box, message, sidekick_state],
    )
    message.submit(
        process_message,
        [sidekick_state, message, success_criteria, chatbot],
        [chatbot, plan_box, message, sidekick_state],
    )
    reset_button.click(
        reset,
        [sidekick_state],
        [message, success_criteria, chatbot, plan_box, sidekick_state],
    )


ui.launch(inbrowser=True, show_error=True)
