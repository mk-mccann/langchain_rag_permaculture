import os
import gradio as gr
from pathlib import Path
from dotenv import load_dotenv

from RAGAgent import RAGAgent


# Load environment variables
load_dotenv()
mistral_api_key = os.getenv("MISTRAL_API_KEY").strip()

# Initialize the RAG agent
agent = RAGAgent(
    chroma_db_dir=Path("../chroma_db"),
    collection_name="perma_rag_collection",
    model_name="mistral-small-latest",
    embeddings_model="mistral-embed",
)

# Store conversation threads per session
conversation_threads = {}


def chat_with_agent(message, history, thread_id="default"):
    """
    Process user message and return response with sources.
    
    Args:
        message (str): User's input message
        history: Chat history (managed by Gradio)
        thread_id (str): Conversation thread identifier
        
    Returns:
        tuple: (response_text, sources_text)
    """
    if not message.strip():
        return "", "No sources yet."
    
    try:
        # Query the agent
        result = agent.query(message, thread_id=thread_id)
        
        # Format sources for display
        sources_text = "**Sources:**\n\n"
        if result["sources"]:
            for source in result["sources"]:
                sources_text += f"- **[Source {source['source_number']}]** "
                if 'title' in source:
                    sources_text += f"Title: {source['title']}, "

                # Build a hierarchical header chain if available (H1 > H2 > H3 > H4)
                header_order_primary = ['header_1', 'header_2', 'header_3', 'header_4']
                header_order_alt = ['head_1', 'head_2', 'head_3', 'head_4']
                header_values = []

                # Collect primary header levels
                for hk in header_order_primary:
                    hv = source.get(hk)
                    if isinstance(hv, str) and hv.strip():
                        header_values.append(hv.strip())

                # Fill gaps with alternative keys if primary missing
                if not header_values:
                    for hk in header_order_alt:
                        hv = source.get(hk)
                        if isinstance(hv, str) and hv.strip():
                            header_values.append(hv.strip())

                # If only a generic 'header' exists, include it
                if not header_values:
                    hv = source.get('header')
                    if isinstance(hv, str) and hv.strip():
                        header_values.append(hv.strip())

                # Deduplicate while preserving order
                seen_h = set()
                header_values = [h for h in header_values if not (h in seen_h or seen_h.add(h))]

                if header_values:
                    if len(header_values) == 1:
                        sources_text += f"Header: {header_values[0]}, "
                    else:
                        sources_text += f"Headers: {' > '.join(header_values)}, "

                if 'url' in source:
                    sources_text += f"URL: {source['url']}, "
                if 'page' in source:
                    sources_text += f"Page: {source['page']}, "
                sources_text += "\n"
        else:
            sources_text += "No sources retrieved."
        
        return result["answer"], sources_text
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return error_msg, "Error retrieving sources."


def chat_interface(message, history):
    """
    Main chat interface function for Gradio ChatInterface.
    
    Args:
        message (str): User's input message
        history: Chat history
        
    Returns:
        str: Agent's response
    """
    # Use a default thread for the chat interface
    result = agent.query(message, thread_id="gradio_session")
    return result["answer"]


def create_demo():
    """Create and configure the Gradio interface."""
    
    # Custom CSS for better styling
    custom_css = """
    .source-box {
        background-color: #f5f5f5;
        border-radius: 8px;
        padding: 15px;
        margin-top: 10px;
    }
    """
    
    # Create the interface with tabs
    with gr.Blocks(css=custom_css, title="RAG Agent Demo") as demo:
        
        gr.Markdown("""
        # 🌱 RAGrarian - Sustainable Development Knowledge Base
        
        Ask questions about sustainable development and get answers with source citations!
        """)
        
        with gr.Tab("Chat"):
            chatbot = gr.ChatInterface(
                fn=chat_interface,
                title="Chat with the RAGrarian",
                description="Ask questions about permaculture. The agent will cite its sources.",
                examples=[
                    "What is permaculture?",
                    "What are the principles of permaculture?",
                    "How does permaculture relate to sustainability?",
                    "What are common permaculture techniques?",
                ],
                # retry_btn=None,
                # undo_btn="◀️ Undo",
                # clear_btn="🗑️ Clear",
            )
        
        with gr.Tab("Query with Sources"):
            gr.Markdown("### Ask a question and see the sources")
            
            with gr.Row():
                with gr.Column(scale=2):
                    query_input = gr.Textbox(
                        label="Your Question",
                        placeholder="e.g., What is permaculture?",
                        lines=3
                    )
                    query_btn = gr.Button("Ask Question", variant="primary")
                
            with gr.Row():
                with gr.Column():
                    answer_output = gr.Textbox(
                        label="Answer",
                        lines=10,
                        show_copy_button=True
                    )
                with gr.Column():
                    sources_output = gr.Markdown(
                        label="Sources",
                        value="Sources will appear here after asking a question."
                    )
            
            # Example questions
            gr.Examples(
                examples=[
                    ["What is permaculture?"],
                    ["What are the ethics of permaculture?"],
                    ["Explain companion planting"],
                    ["What is forest gardening?"],
                ],
                inputs=query_input,
            )
            
            # Connect the query button
            query_btn.click(
                fn=lambda q: chat_with_agent(q, None, "query_tab"),
                inputs=query_input,
                outputs=[answer_output, sources_output]
            )
        
        with gr.Tab("About"):
            gr.Markdown("""
            ## About the RAGrarian
            
            This is a Retrieval-Augmented Generation (RAG) agent that answers questions about permaculture.
            
            ### Features:
            - 💬 **Interactive Chat**: Natural conversation with memory
            - 📚 **Source Citations**: Every answer includes references to source documents
            - 🔍 **Context-Aware**: Uses semantic search to find relevant information
            - 🧠 **Powered by**: Mistral AI and LangChain
            
            ### How it works:
            1. You ask a question
            2. The agent searches the knowledge base for relevant documents
            3. It generates an answer based on the retrieved context
            4. Sources are cited using [Source N] format
            
            ### Technology Stack:
            - **LLM**: Mistral AI
            - **Vector Store**: ChromaDB
            - **Framework**: LangChain
            - **UI**: Gradio

            ### Developed by:           
            Matt McCann (c) 2025
            [GitHub](www.mk-mccann.github.io) | [LinkedIn](https://linkedin.com/in/matt-k-mccann/)
            """)
            
    
    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        share=False,  # Set to True to create a public link
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )
