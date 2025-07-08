import os
import gradio as gr
from rag_pipeline import SimpleRAGPipeline

# Initialize RAG pipeline
rag = SimpleRAGPipeline(
    vector_store_dir="src/RAG/vector_store",
    use_llm=False  # Set True if you want to enable LLM generation
)

def chat(query):
    if not query.strip():
        return "⚠️ Please enter a valid question.", ""

    result = rag.query(query, k=5)
    answer = result["answer"]
    sources = result["retrieved_chunks"]

    if not sources:
        return answer, "⚠️ No relevant sources found."

    # Format source chunks with metadata
    formatted_sources = ""
    for idx, src in enumerate(sources[:2], 1):  # Show top 2
        formatted_sources += f"🔹 **Source {idx}**\n"
        formatted_sources += f"- Product: {src['product']}\n"
        formatted_sources += f"- Complaint ID: {src['complaint_id']}\n"
        formatted_sources += f"- Score: {src['score']:.4f}\n"
        formatted_sources += f"```{src['text'][:500]}...```\n\n"

    return answer, formatted_sources.strip()

# Gradio Interface
with gr.Blocks(title="CrediTrust RAG QA") as demo:
    gr.Markdown("##  CrediTrust Complaint Explorer")
    gr.Markdown("Ask a question about consumer complaints (e.g., *Why do users complain about BNPL?*)")

    with gr.Row():
        query_box = gr.Textbox(label="Your Question", placeholder="Type your question here...", lines=1)
        submit_btn = gr.Button(" Ask")
        clear_btn = gr.Button(" Clear")

    answer_output = gr.Textbox(label="Answer", placeholder="AI-generated answer will appear here.", lines=4)
    source_output = gr.Markdown(label="Sources")

    submit_btn.click(fn=chat, inputs=query_box, outputs=[answer_output, source_output])
    clear_btn.click(fn=lambda: ("", ""), inputs=[], outputs=[answer_output, source_output])

# Run app
if __name__ == "__main__":
    demo.launch()
