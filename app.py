import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from huggingface_hub import AsyncInferenceClient, InferenceClient
import asyncio


model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
async_client = AsyncInferenceClient(model=model_name)
sync_client = InferenceClient(model=model_name)


def summarise_pdf(pdf):
    loader = PyPDFLoader(pdf.name)
    pages = loader.load()

    summary = asyncio.run(map_method(pages))

    return summary


async def map_method(pages):
    chunk_size = 10
    chunks = [pages[i : i + chunk_size] for i in range(0, len(pages), chunk_size)]

    tasks = []
    for chunk in chunks:
        combined_content = combine_pages(chunk)
        tasks.append(summarise_chunk(combined_content))

    chunk_summaries = await asyncio.gather(*tasks)

    final_summary = reduce_summaries(chunk_summaries)

    return final_summary


def combine_pages(pages):
    combined_content = "\n\n".join([page.page_content for page in pages])
    return combined_content


async def summarise_chunk(chunk):
    prompt = f"""Summarize the following document in 150-300 words, ensuring the most important ideas and main themes are highlighted:\n\n{chunk}"""

    message = [{"role": "user", "content": prompt}]

    result = await async_client.chat_completion(
        messages=message,
        max_tokens=2048,
        temperature=0.1,
    )

    return result.choices[0].message["content"].strip()


def reduce_summaries(summaries):
    combined_summaries = "\n\n".join(summaries)

    reduce_prompt = f"Below is a collection of summaries, please synthesize them into a cohesive final summary, highlighting the key themes. Ensure the summary is concise and does not exceed 400 words:\n\n{combined_summaries}"

    message = [{"role": "user", "content": reduce_prompt}]

    result = sync_client.chat_completion(
        messages=message,
        max_tokens=2048,
        temperature=0.1,
    )

    return result.choices[0].message["content"].strip()


with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("<H1>PDF Summariser</H1>")
    gr.Markdown("<H3>Upload a PDF file and generate a summary</H3>")
    gr.Markdown(
        "<H6>This project uses a MapReduce method to split the PDF into chunks, generate summaries of each of the chunks asynchronously, and reduce them into a single final summary.</H6>"
    )
    gr.Markdown(
        "<H6>Note: I have included The Metamorphosis by Franz Kafka as a default PDF to demonstrate its working on a large document. Replace this with any PDF you would like to summarise.</H6>"
    )

    with gr.Row():
        with gr.Column(scale=1):
            pdf = gr.File(label="Upload PDF", value="./TheMetamorphosis.pdf")
            summarise_btn = gr.Button(value="Summarise PDF 🚀", variant="primary")
        with gr.Column(scale=3):
            summary = gr.TextArea(label="Summary")

    summarise_btn.click(fn=summarise_pdf, inputs=pdf, outputs=summary)

demo.launch()
