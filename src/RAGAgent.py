from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_core.documents import Document
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver  
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime


# ---------------------------------------------------------------------------
# Shared citation formatting helpers (reusable across chat/query/UI)
# ---------------------------------------------------------------------------
def format_header_chain(metadata: dict) -> str | None:
    """Build a hierarchical header chain H1 > H2 > H3 > H4 from metadata.

    Falls back to common variants (head_*) and a generic 'header' field.
    Returns None if nothing present.
    """
    levels_primary = ['header_1', 'header_2', 'header_3', 'header_4']
    levels_alt = ['head_1', 'head_2', 'head_3', 'head_4']
    vals: list[str] = []

    for k in levels_primary:
        v = metadata.get(k)
        if isinstance(v, str) and v.strip():
            vals.append(v.strip())
    if not vals:
        for k in levels_alt:
            v = metadata.get(k)
            if isinstance(v, str) and v.strip():
                vals.append(v.strip())
    if not vals:
        v = metadata.get('header')
        if isinstance(v, str) and v.strip():
            vals.append(v.strip())

    # Deduplicate while preserving order
    seen: set[str] = set()
    vals = [h for h in vals if not (h in seen or seen.add(h))]
    if not vals:
        return None
    return " > ".join(vals) if len(vals) > 1 else vals[0]


def build_citation(doc: Document, source_number: int) -> dict:
    """Create a structured citation dict from a Document for consistent use.

    Returns keys commonly used by the UI and logs:
    - source_number, title, header, url, page, file, file_path, metadata, citation_text
    """
    md = doc.metadata if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict) else {}
    source = md.get('source') or 'Unknown'
    title = md.get('title') or 'Unknown'
    header = format_header_chain(md)
    url = md.get('url')
    page = md.get('page')
    file_label = md.get('file_path') or md.get('path') or 'Unknown'
    file_path = md.get('file_path')

    citation_text = f"[Source {source_number}] (Source: {source}, Title: {title}, "
    if header:
        citation_text += f"Section: {header}, "
    if url:
        citation_text += f"URL: {url}, "
    if page is not None:
        citation_text += f"Page: {page}, "

    # Trim any trailing comma + space and close paren
    citation_text = citation_text.rstrip(', ') + ")"

    return {
        "source_number": source_number,
        "source": source,
        "title": title,
        "header": header,
        "url": url,
        "page": page,
        "file": file_label,
        "file_path": file_path,
        "metadata": md,
        "citation_text": citation_text,
    }


def format_citation_line(citation: dict, include_content: str | None = None) -> str:
    """Render a single citation line (optionally followed by content).

    include_content: if provided, appended after the citation line separated by newline.
    """
    base = citation.get("citation_text") or ""
    if include_content:
        return f"{base}\n{include_content}"
    return base


class CustomAgentState(AgentState):  
    user_id: str
    preferences: dict
    context: list[Document]


class RetrieveDocumentsMiddleware(AgentMiddleware[CustomAgentState]):

    state_schema = CustomAgentState

    def __init__(self, vectorstore: Chroma, k_documents: int = 4):
        self.vectorstore = vectorstore
        self.k_documents = k_documents


    # Use shared header formatting helpers defined above
    def before_model(self, state: CustomAgentState) -> dict[str, Any] | None:

        last_message = state["messages"][-1]
        retrieved_docs = self.vectorstore.similarity_search(
            last_message.content,
            k=self.k_documents
        )

        # Format context with numbered sources for citation using shared helpers
        docs_content_with_citations = []
        for idx, doc in enumerate(retrieved_docs, 1):
            citation = build_citation(doc, idx)
            docs_content_with_citations.append(
                format_citation_line(citation, include_content=doc.page_content)
            )
        
        docs_content = "\n\n".join(docs_content_with_citations)

        augmented_message_content = (
            f"{last_message.content}\n\n"
            "Use the following context to answer the query. "
            "When using information from the context, cite the source number (e.g., [Source 1]):\n\n"
            f"{docs_content}"
        )
        
        # Provide retrieved docs under both "context" (matches state_schema)
        # and "sources" (for downstream consumers expecting that key)
        return {
            "messages": [last_message.model_copy(update={"content": augmented_message_content})],
            "context": retrieved_docs,
            "sources": retrieved_docs,
        }


# class TrimMessagesMiddleware(AgentMiddleware[CustomAgentState]):

#     state_schema = CustomAgentState

#     @before_model
#     def trim_messages(state: CustomAgentState, runtime: Runtime) -> dict[str, Any] | None:
#         """Keep only the last few messages to fit context window."""
#         messages = state["messages"]

#         if len(messages) <= 3:
#             return None  # No changes needed

#         first_msg = messages[0]
#         recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
#         new_messages = [first_msg] + recent_messages

#         return {
#             "messages": [
#                 RemoveMessage(id=REMOVE_ALL_MESSAGES),
#                 *new_messages
#             ]
#         }


#     def delete_specfic_messages(state):
#         messages = state["messages"]
#         if len(messages) > 2:
#             # remove the earliest two messages
#             return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}  
        

#     def delete_all_messages(state):
#         return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}  


class RAGAgent:

    def __init__(self,
        chroma_db_dir: Path | str,
        collection_name: str,
        model_name: str = "mistral-large-latest",
        embeddings_model: str = "mistral-embed",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        k_documents: int = 4
    ):
        
        """
        Initialize the RAG agent with conversational memory.
        
        Args:
            chroma_db_dir (Path | str): Directory containing the ChromaDB database.
            collection_name (str): name of the ChromaDB collection.
            model_name (str): Mistral model to use for chat.
            embeddings_model (str): Model to use for embeddings.
            temperature (float): Temperature for response generation.
            max_tokens (int): Maximum tokens in response.
            k_documents (int): Number of documents to retrieve.
        """
        
        self.chroma_db_dir = Path(chroma_db_dir)
        self.collection_name= collection_name
        self.k_documents = k_documents
        
        # Initialize embeddings
        self.embeddings = MistralAIEmbeddings(model=embeddings_model)
        
        # Initialize vectorstore
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(chroma_db_dir)
        )
        
        # Initialize chat model
        self.model = ChatMistralAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Create agent with retrieval middleware
        self.agent = create_agent(
            self.model,
            system_prompt="Please be concise and to the point.",
            tools=[],
            middleware=[RetrieveDocumentsMiddleware(self.vectorstore, self.k_documents)],
            state_schema=CustomAgentState,
            checkpointer=InMemorySaver()
        )

        self.agent.invoke(
            {"messages": [{"role": "user", 
                        "content": "Hi! My source is Bob."}]
                        },
            {"configurable": {"thread_id": "1"}},  
        )


    def chat(self, thread_id: str = "default"):
        """
        Start an interactive chat session.
        
        Args:
            thread_id (str): Unique identifier for this conversation thread.
        """

        print("RAG Agent Chat Interface")
        print("Type 'quit', 'exit', or 'q' to end the conversation")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'sources':
                    if last_context:
                        print("\nSources from last response:")
                        for idx, doc in enumerate(last_context, 1):
                            citation = build_citation(doc, idx)
                            print(f"{format_citation_line(citation)}")
                    else:
                        print("No sources available yet. Ask a question first!")
                    continue
                
                if not user_input:
                    continue
                
                # Stream the response
                print("\nAssistant: ", end="", flush=True)
                
                for step in self.agent.stream(
                    {
                        "messages": [{"role": "user", "content": user_input}],
                    },
                    {"configurable": {"thread_id": thread_id}},
                    stream_mode="values"
                ):
                    # Get the last message
                    last_msg = step["messages"][-1]
                    
                    # Only print assistant messages
                    if last_msg.type == "ai":
                        # Print content (this will show incrementally if streaming)
                        print(last_msg.content, end="", flush=True)
                        
                        # Save context for 'sources' command
                        if "context" in step:
                            last_context = step["context"]
                
                print()  # New line after response
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                continue

    def query(self, question: str, thread_id: str = "default") -> dict:
        """
        Query the agent with a single question.
        
        Args:
            question (str): The question to ask.
            thread_id (str): Thread ID for conversation continuity.
            
        Returns:
            dict: Contains 'answer' and 'sources' keys.
        """
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            {"configurable": {"thread_id": thread_id}}
        )
        
        # Extract answer and sources
        answer = result["messages"][-1].content
        sources = []

        # Prefer 'context' (aligned with state schema) but fall back to 'sources'
        retrieved = result.get("context") or result.get("sources") or []

        for idx, doc in enumerate(retrieved, 1):
            citation = build_citation(doc, idx)
            # Expose a consistent, enriched source dict to callers
            sources.append(citation)
        
        return {
            "answer": answer,
            "sources": sources
        }
    

    def _test_query_prompt_with_context(self):
        query = (
            "What is permaculture?"
        )

        for step in self.agent.stream(
            {
                "messages": [{"role": "user", "content": query}],
                "user_id": "user_123",
                "preferences": {"theme": "dark"}
            },
            {"configurable": {"thread_id": "1"}},
            stream_mode="values"
            ):

            step["messages"][-1].pretty_print()


    def _test_query_with_sources(self):
        """Test query that shows citations."""
        query = "What are the principles of permaculture?"
        
        print("Question:", query)
        print("\n" + "="*50 + "\n")
        
        result = self.query(query, thread_id="test")
        
        print("Answer:")
        print(result["answer"])
        print("\n" + "-"*50 + "\n")
        print("Sources:")
        for source in result["sources"]:
            print(f"  [Source {source['source_number']}] {source['file']}, Page: {source['page']}")



if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    mistral_api_key = os.getenv("MISTRAL_API_KEY").strip()

    agent = RAGAgent(
        chroma_db_dir = Path("../chroma_db"),
        collection_name = "perma_rag_collection",
        model_name = "mistral-small-latest",
        embeddings_model = "mistral-embed",
    )
    
    # agent._test_query_prompt_with_context()

    # Option 1: Interactive chat
    agent.chat(thread_id="session_1")
    
    # Option 2: Single query with sources
    # agent._test_query_with_sources()
