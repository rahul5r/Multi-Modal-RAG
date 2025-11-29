import streamlit as st
import tabula
import faiss
import json
import base64
import pymupdf
import os
import logging
import numpy as np
import warnings
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
# CRITICAL FIX 1: Handle all LangChain attribute errors
import langchain
langchain.verbose = False
langchain.debug = False
langchain.tracing_callback_enabled = False
langchain.llm_cache = None
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import tempfile
import io
import hashlib
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# ENV + GEMINI CONFIG
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define generate_text_embedding globally
@st.cache_resource
def generate_text_embedding(text: str):
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text")
    response = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document"
    )
    return response["embedding"]

# Define invoke_gemini_multimodal globally
@st.cache_resource
def invoke_gemini_multimodal(prompt, matched_items):
    system_msg = SystemMessage(
        content=(
            "You are a helpful assistant that answers questions using retrieved context. "
            "Context may include text chunks, tables, and page images from a PDF. "
            "Use ONLY the provided content to answer the question, and be as specific as possible."
        )
    )
    user_contents = []
    for item in matched_items:
        if item["type"] in ["text", "table"]:
            user_contents.append(item["text"])
        elif item["type"] in ["image", "page"]:
            img_base64 = item["image"]
            user_contents.append(
                {
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{img_base64}",
                }
            )
    user_contents.append(f"\n\nQuestion: {prompt}")
    user_msg = HumanMessage(content=user_contents)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.2,
        max_output_tokens=400,
    )
    response = llm.invoke([system_msg, user_msg])
    return response.content

# Function to compute file hash
def get_file_hash(file_obj):
    file_content = file_obj.getvalue()
    return hashlib.md5(file_content).hexdigest()

# Streamlit app
st.title("PDF Chatbot")
st.write("Upload a PDF to start chatting with its content!")

# File uploader with key to force rerun on change
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    file_hash = get_file_hash(uploaded_file)
    
    # Check if file changed or first upload
    if 'current_file_hash' not in st.session_state or st.session_state['current_file_hash'] != file_hash:
        # Reset session state for new PDF
        to_clear = ['processed', 'index', 'embed_items', 'items', 'base_dir', 'messages', 'current_file_hash']
        for key in to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state['current_file_hash'] = file_hash
        st.rerun()  # Rerun to clear and start fresh
    
    if 'processed' not in st.session_state:
        with st.spinner("Processing PDF... This may take a while."):
            # Save uploaded file to temp
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_filepath = tmp_file.name
            
            # Create temp directories
            base_dir = tempfile.mkdtemp()
            os.makedirs(os.path.join(base_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(base_dir, "text"), exist_ok=True)
            os.makedirs(os.path.join(base_dir, "tables"), exist_ok=True)
            os.makedirs(os.path.join(base_dir, "page_images"), exist_ok=True)
            
            # PDF processing
            doc = pymupdf.open(temp_filepath)
            num_pages = len(doc)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=700,
                chunk_overlap=200,
                length_function=len
            )
            items = []
            
            st.text("PDF Page Processing Progress:")
            progress_bar = st.progress(0)
            for page_num in range(num_pages):
                page = doc[page_num]
                text = page.get_text()
                
                # Process tables
                try:
                    tables = tabula.read_pdf(
                        temp_filepath,
                        pages=page_num + 1,
                        multiple_tables=True,
                        encoding='utf-8'
                    )
                    if tables:
                        for table_idx, table in enumerate(tables):
                            try:
                                table_text = "\n".join([" | ".join(map(str, row)) for row in table.values])
                                if table_text and table_text.strip():
                                    table_file_name = f"{base_dir}/tables/table_{page_num}_{table_idx}.txt"
                                    with open(table_file_name, 'w', encoding='utf-8', errors='ignore') as f:
                                        f.write(table_text)
                                    items.append({
                                        "page": page_num,
                                        "type": "table",
                                        "text": table_text,
                                        "path": table_file_name
                                    })
                            except Exception:
                                continue
                except Exception:
                    pass
                
                # Process text chunks
                if text and text.strip():
                    chunks = text_splitter.split_text(text)
                    for i, chunk in enumerate(chunks):
                        if chunk and chunk.strip():
                            text_file_name = f"{base_dir}/text/text_{page_num}_{i}.txt"
                            with open(text_file_name, 'w', encoding='utf-8', errors='ignore') as f:
                                f.write(chunk)
                            items.append({
                                "page": page_num,
                                "type": "text",
                                "text": chunk,
                                "path": text_file_name
                            })
                
                # Process images
                try:
                    images = page.get_images()
                    for idx, image in enumerate(images):
                        try:
                            xref = image[0]
                            pix = pymupdf.Pixmap(doc, xref)
                            image_name = f"{base_dir}/images/image_{page_num}_{idx}_{xref}.png"
                            pix.save(image_name)
                            with open(image_name, 'rb') as f:
                                encoded_image = base64.b64encode(f.read()).decode('utf8')
                            items.append({
                                "page": page_num,
                                "type": "image",
                                "path": image_name,
                                "image": encoded_image
                            })
                        except Exception:
                            continue
                except Exception:
                    pass
                
                # Process page images
                try:
                    pix = page.get_pixmap()
                    page_path = os.path.join(base_dir, f"page_images/page_{page_num:03d}.png")
                    pix.save(page_path)
                    with open(page_path, 'rb') as f:
                        page_image = base64.b64encode(f.read()).decode('utf8')
                    items.append({
                        "page": page_num,
                        "type": "page",
                        "path": page_path,
                        "image": page_image
                    })
                except Exception:
                    pass
                
                progress_bar.progress((page_num + 1) / num_pages)
            
            doc.close()
            os.unlink(temp_filepath)  # Clean up temp PDF
            
            # Embeddings
            st.text("Generating Embeddings Progress:")
            embed_items = []
            embeddings = []
            skipped_count = 0
            error_count = 0
            
            embed_progress = st.progress(0)
            for idx, item in enumerate(items):
                item_type = item["type"]
                if item_type in ["text", "table"]:
                    try:
                        if not item.get("text") or not item["text"].strip():
                            skipped_count += 1
                            embed_progress.progress((idx + 1) / len(items))
                            continue
                        emb = generate_text_embedding(item["text"])
                        item["embedding"] = emb
                        embed_items.append(item)
                        embeddings.append(emb)
                    except ValueError:
                        skipped_count += 1
                    except Exception as e:
                        error_count += 1
                embed_progress.progress((idx + 1) / len(items))
            
            st.success(f"Processing complete! Embedded {len(embed_items)} items. Skipped: {skipped_count}, Errors: {error_count}")
            
            if len(embeddings) > 0:
                embeddings = np.array(embeddings, dtype=np.float32)
                embedding_vector_dimension = embeddings.shape[1]
                index = faiss.IndexFlatL2(embedding_vector_dimension)
                index.add(embeddings)
                
                # Store in session state
                st.session_state['index'] = index
                st.session_state['embed_items'] = embed_items
                st.session_state['items'] = items
                st.session_state['processed'] = True
                st.session_state['base_dir'] = base_dir  # For cleanup if needed
            else:
                st.error("No embeddings generated. Cannot proceed.")
                st.stop()
    
    # Chat interface if processed
    if st.session_state.get('processed', False):
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("PDF Preview")
            page_items = sorted([item for item in st.session_state['items'] if item['type'] == 'page'], key=lambda x: x['page'])
            if page_items:
                pdf_container = st.container(height=600)
                with pdf_container:
                    for item in page_items:
                        base64_img = item['image']
                        img_data = base64.b64decode(base64_img)
                        img = Image.open(io.BytesIO(img_data))
                        st.image(img, width='stretch', caption=f"Page {item['page'] + 1}")
        
        with col2:
            st.subheader("Chat with the PDF")
            
            # Initialize chat history (cleared on new PDF)
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input at the bottom
            if prompt := st.chat_input("Ask a question about the PDF"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()
            
            # Generate response if last message is from user
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    response = None
                    with st.spinner("Thinking..."):
                        try:
                            # Embed query
                            query_embedding = generate_text_embedding(st.session_state.messages[-1]["content"])
                            query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
                            
                            # Search
                            index = st.session_state['index']
                            distances, result = index.search(query_embedding, k=5)
                            
                            # Build matched items
                            def build_matched_items_from_results(result_indices, k=5):
                                top_indices = result_indices[0][:k]
                                core_items = [st.session_state['embed_items'][i] for i in top_indices]
                                matched_items = []
                                seen = set()
                                def add_item(it):
                                    key = (it["type"], it["page"], it.get("path", ""))
                                    if key not in seen:
                                        seen.add(key)
                                        matched_items.append(it)
                                for core in core_items:
                                    add_item(core)
                                    page_num = core["page"]
                                    for it in st.session_state['items']:
                                        if it["page"] == page_num and it is not core:
                                            if it["type"] in ["table", "image", "page"]:
                                                add_item(it)
                                return matched_items
                            
                            matched_items = build_matched_items_from_results(result, k=5)
                            
                            # Invoke Gemini
                            response = invoke_gemini_multimodal(st.session_state.messages[-1]["content"], matched_items)
                            response_placeholder.markdown(response)
                        
                        except Exception as e:
                            error_msg = f"Error generating response: {str(e)}"
                            st.error(error_msg)
                            response = "Sorry, I encountered an error while processing your query."
                            response_placeholder.markdown(response)
                    
                    # Add assistant response to history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()