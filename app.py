from modules.causal_engine import CausalCoTEngine
import gradio as gr

engine = CausalCoTEngine(model_tag="Qwen/Qwen3-4B")

def generate_cot(question):
    initial_response, before_think_text, cot = engine.initial_pass(question)
    # Build step choices for dropdown
    steps = cot.split("Step:")[1:]
    step_choices = [f"Step {i+1}" for i in range(len(steps))]
    return (
        initial_response,
        before_think_text,
        cot,
        cot,   # initially edited_cot = full cot
        gr.update(choices=step_choices, value=step_choices[0] if step_choices else None),
        cot,   # store original cot for retry
    )


def apply_edit(selected_step_label, original_cot, before_think_text):
    """Edit the selected step and return the new CoT."""
    if not selected_step_label or not original_cot:
        return original_cot, "No step selected"
    idx = int(selected_step_label.split()[1]) - 1
    new_cot = engine.edit_cot(idx, original_cot)
    return new_cot, f"Step {idx+1} edited. Later steps removed."


def generate_final(before_think_text, edited_cot, mode):
    answer, response = engine.final_pass(before_think_text, edited_cot, mode)
    return answer, response


with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"),
    title="Causal CoT Engine"
) as demo:

    gr.Markdown("""
    # Causal Chain-of-Thought Editor  
    Edit reasoning. Control generation. Probe causality.
    """)

    # State variables
    original_cot_state = gr.State("")
    before_think_state = gr.State("")

    with gr.Tab("Step 1: Generate CoT"):
        with gr.Row():
            question = gr.Textbox(
                label="Question",
                lines=3,
                placeholder="Ask anything..."
            )

        run_btn = gr.Button("Generate Reasoning", variant="primary")

        initial_response = gr.Textbox(label="Model Output", lines=12)
        hidden_before = gr.Textbox(visible=False)

        with gr.Row():
            cot = gr.Textbox(label="Extracted CoT", lines=10)

    with gr.Tab("Step 2: Edit + Generate"):
        edited_cot = gr.Textbox(
            label="Edit Chain-of-Thought",
            lines=12,
            interactive=True
        )

        # Step editing controls
        with gr.Row():
            step_selector = gr.Dropdown(
                label="Select Step to Edit (from original CoT)",
                choices=[],
                interactive=True
            )
            edit_btn = gr.Button("Edit Step (LLM)", variant="secondary")
            retry_btn = gr.Button("Retry Edit", variant="secondary", visible=False)

        edit_status = gr.Markdown("")

        mode = gr.Radio(
            ["Answer Only", "Continue Reasoning"],
            value="Answer Only",
            label="Generation Mode"
        )

        final_btn = gr.Button("Generate Final Output", variant="primary")

        with gr.Row():
            final_answer = gr.Textbox(label="Final Answer", lines=3)
            edited_response = gr.Textbox(label="Full Model Response", lines=12)

    run_btn.click(
        fn=generate_cot,
        inputs=[question],
        outputs=[
            initial_response,
            hidden_before,
            cot,
            edited_cot,
            step_selector,
            original_cot_state,
        ]
    ).then(
        fn=lambda x: x,
        inputs=[hidden_before],
        outputs=[before_think_state]
    )

    edit_btn.click(
        fn=apply_edit,
        inputs=[step_selector, original_cot_state, before_think_state],
        outputs=[edited_cot, edit_status]
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=[retry_btn]
    )

    retry_btn.click(
        fn=apply_edit,
        inputs=[step_selector, original_cot_state, before_think_state],
        outputs=[edited_cot, edit_status]
    )

    final_btn.click(
        fn=generate_final,
        inputs=[before_think_state, edited_cot, mode],
        outputs=[final_answer, edited_response]
    )

demo.launch(share=True)