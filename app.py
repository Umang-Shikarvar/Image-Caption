import gradio as gr
from inference import generate_caption

def caption_image(image):
    return generate_caption(image)

with gr.Blocks(title="Image Captioning Demo") as demo:
    gr.Markdown("## Image Captioning Model\nUpload an image to generate a caption.")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image")
        caption_output = gr.Textbox(label="Generated Caption", lines=2)

    generate_btn = gr.Button("Generate Caption")

    generate_btn.click(fn=caption_image, inputs=image_input, outputs=caption_output)

demo.launch()