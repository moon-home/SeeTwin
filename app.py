"""
SeeTwin — main application entry point.

Run:
    python app.py
    python app.py --device cuda   # if you have a GPU
    python app.py --port 7861
"""

import argparse
import logging

import gradio as gr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)

def parse_args():
    p = argparse.ArgumentParser(description="SeeTwin avatar pipeline")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="Inference device (default: cpu)")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true",
                   help="Create a public Gradio share link")
    return p.parse_args()


def build_app(device: str = "cpu") -> gr.Blocks:
    from ui.stage1_ui import build_stage1_tab, STAGE1_JS
    from ui.stage2_ui import build_stage2_tab, STAGE2_JS

    # Both JS modules are arrow-function strings; wrap in one combined call
    COMBINED_JS = f"() => {{ ({STAGE1_JS})(); ({STAGE2_JS})(); }}"

    with gr.Blocks(title="SeeTwin", theme=gr.themes.Soft(), js=COMBINED_JS) as app:
        gr.Markdown(
            "# SeeTwin\n"
            "Photo-to-rigged 3D avatar pipeline · "
            f"Running on **{device.upper()}**"
        )

        with gr.Tabs() as main_tabs:
            build_stage1_tab(device=device)
            build_stage2_tab(main_tabs=main_tabs)
            with gr.Tab("3 — Classification", id="stage3"):
                gr.Markdown("_Coming soon_")
            with gr.Tab("4 — Texture extraction", interactive=False):
                gr.Markdown("_Coming soon_")
            with gr.Tab("5 — Assembly", interactive=False):
                gr.Markdown("_Coming soon_")
            with gr.Tab("6 — Blender fine-tune", interactive=False):
                gr.Markdown("_Coming soon_")

    return app


if __name__ == "__main__":
    args = parse_args()
    app = build_app(device=args.device)
    app.launch(server_name="127.0.0.1", server_port=args.port, share=args.share)
