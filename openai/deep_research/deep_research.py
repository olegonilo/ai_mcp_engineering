from dotenv import load_dotenv
load_dotenv(override=True)  # must run before project imports that check env vars at module level

import gradio as gr
from research_manager import ResearchManager


async def clarify(query: str) -> str:
    if not query.strip():
        return "Please enter a research query first."
    result = await ResearchManager().clarify(query)
    lines = [f"{i + 1}. {q}" for i, q in enumerate(result.questions)]
    return "**Please answer these clarifying questions:**\n\n" + "\n".join(lines)


async def run(query: str, clarifications: str):
    if not query.strip():
        yield "Please enter a research query."
        return
    async for chunk in ResearchManager().run(query, clarifications):
        yield chunk


with gr.Blocks(theme=gr.themes.Default(primary_hue="sky")) as ui:
    gr.Markdown("# Deep Research")

    with gr.Row():
        query_textbox = gr.Textbox(label="What topic would you like to research?", scale=4)
        clarify_btn = gr.Button("Clarify", variant="secondary", scale=1)

    questions_md = gr.Markdown()
    clarifications_textbox = gr.Textbox(
        label="Your answers to the clarifying questions (optional — leave blank to skip)",
        placeholder="1. ...\n2. ...\n3. ...",
        lines=3,
    )
    run_button = gr.Button("Run Research", variant="primary")
    status = gr.Markdown(label="Status")

    clarify_btn.click(fn=clarify, inputs=query_textbox, outputs=questions_md)
    run_button.click(fn=run, inputs=[query_textbox, clarifications_textbox], outputs=status)

ui.launch(inbrowser=True)
