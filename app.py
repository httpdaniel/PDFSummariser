import gradio as gr
from langchain_community.document_loaders import PyPDFLoader

def summarise_pdf(pdf, progress=gr.Progress()):

    return "Summarised", "Complete!"


with gr.Blocks() as demo:

    gr.Markdown("<H1>PDF Summariser</H1>")
    gr.Markdown("<H3>Upload a PDF file and generate a summary</H3>")
    gr.Markdown("<H6>This project uses a MapReduce method to split the PDF into chunks, generate summaries of each of the chunks, and reduce them into a single final summary. Documents less than 3 pages use a Stuff method to simply stuff the entire document into the context window.</H6>")

    with gr.Row():
        with gr.Column(scale=1):
            pdf = gr.File(label="1. Upload PDF")
            summarise_btn = gr.Button(value="3. Summarise PDF", variant="primary")
            summary_progress = gr.Textbox(value="Not Started", label="Summary Progress")
        with gr.Column(scale=3):
            summary = gr.TextArea(label="Summary")

    summarise_btn.click(fn=summarise_pdf, inputs=pdf, outputs=[summary, summary_progress])

demo.launch()
