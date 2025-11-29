# %%
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

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

warnings.filterwarnings("ignore")

# %% ENV + GEMINI CONFIG
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# %% PDF PATH
filepath = "qatar_test_doc_sample.pdf"

# %%
# Create the directories
def create_directories(base_dir):
    directories = ["images", "text", "tables", "page_images"]
    for dir in directories:
        os.makedirs(os.path.join(base_dir, dir), exist_ok=True)


# Process tables with better error handling
def process_tables(doc, page_num, base_dir, items):
    try:
        # CRITICAL FIX 2: Removed 'errors' param which tabula-py doesn't support
        tables = tabula.read_pdf(
            filepath, 
            pages=page_num + 1, 
            multiple_tables=True, 
            encoding='utf-8'
        )
        if not tables:
            return
        
        for table_idx, table in enumerate(tables):
            try:
                table_text = "\n".join([" | ".join(map(str, row)) for row in table.values])
                
                # Validate table_text is not empty
                if not table_text or table_text.strip() == "":
                    continue
                
                table_file_name = f"{base_dir}/tables/{os.path.basename(filepath)}_table_{page_num}_{table_idx}.txt"
                with open(table_file_name, 'w', encoding='utf-8', errors='ignore') as f:
                    f.write(table_text)
                
                items.append({
                    "page": page_num,
                    "type": "table",
                    "text": table_text,
                    "path": table_file_name
                })
            except Exception as inner_e:
                continue
    except Exception as e:
        # Silent fail for tables is acceptable if tabula struggles with the page
        pass


# Process text chunks with validation
def process_text_chunks(text, text_splitter, page_num, base_dir, items):
    if not text or text.strip() == "":
        return  # Skip empty pages
    
    chunks = text_splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        # Validate chunk is not empty
        if not chunk or chunk.strip() == "":
            continue
        
        text_file_name = f"{base_dir}/text/{os.path.basename(filepath)}_text_{page_num}_{i}.txt"
        with open(text_file_name, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(chunk)
        
        items.append({
            "page": page_num,
            "type": "text",
            "text": chunk,
            "path": text_file_name
        })


# Process images
def process_images(page, page_num, base_dir, items):
    try:
        images = page.get_images()
        for idx, image in enumerate(images):
            try:
                xref = image[0]
                pix = pymupdf.Pixmap(doc, xref)
                image_name = f"{base_dir}/images/{os.path.basename(filepath)}_image_{page_num}_{idx}_{xref}.png"
                pix.save(image_name)
                with open(image_name, 'rb') as f:
                    encoded_image = base64.b64encode(f.read()).decode('utf8')
                items.append({
                    "page": page_num,
                    "type": "image",
                    "path": image_name,
                    "image": encoded_image
                })
            except Exception as inner_e:
                continue
    except Exception as e:
        pass


# Process full page images
def process_page_images(page, page_num, base_dir, items):
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
    except Exception as e:
        pass


# %% PDF PROCESSING
doc = pymupdf.open(filepath)
num_pages = len(doc)
base_dir = "data"

create_directories(base_dir)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=200,
    length_function=len
)
items = []

# Process each page of the PDF
for page_num in tqdm(range(num_pages), desc="Processing PDF pages"):
    page = doc[page_num]
    # CRITICAL: No arguments for get_text() in newer pymupdf
    text = page.get_text()
    process_tables(doc, page_num, base_dir, items)
    process_text_chunks(text, text_splitter, page_num, base_dir, items)
    process_images(page, page_num, base_dir, items)
    process_page_images(page, page_num, base_dir, items)


# %% Quick sanity checks
print(f"\nTotal items extracted: {len(items)}")
text_items = [i for i in items if i['type'] == 'text']
table_items = [i for i in items if i['type'] == 'table']
image_items = [i for i in items if i['type'] == 'image']
page_items = [i for i in items if i['type'] == 'page']

print(f"  Text items: {len(text_items)}")
print(f"  Table items: {len(table_items)}")
print(f"  Image items: {len(image_items)}")
print(f"  Page items: {len(page_items)}")


# %% GEMINI EMBEDDINGS (TEXT ONLY)

def generate_text_embedding(text: str):
    """
    Generate a text embedding using Gemini text-embedding-004.
    Validates text is not empty before calling API.
    """
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text")
    
    response = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document"
    )
    return response["embedding"]


# We will embed only text + tables (images/pages will be attached by page later)
embed_items = []   # items that actually go into the vector DB
embeddings = []

skipped_count = 0
error_count = 0

with tqdm(total=len(items), desc="Generating Gemini text embeddings") as pbar:
    for item in items:
        item_type = item["type"]

        if item_type in ["text", "table"]:
            try:
                if not item["text"] or not item["text"].strip():
                    skipped_count += 1
                    pbar.update(1)
                    continue
                
                emb = generate_text_embedding(item["text"])
                item["embedding"] = emb
                embed_items.append(item)
                embeddings.append(emb)
            except ValueError as ve:
                skipped_count += 1
            except Exception as e:
                error_count += 1

        pbar.update(1)

print(f"\nEmbedding complete:")
print(f"  Successfully embedded: {len(embed_items)}")
print(f"  Skipped (empty): {skipped_count}")
print(f"  Errors: {error_count}")

# Convert to numpy & build FAISS index
if len(embeddings) > 0:
    embeddings = np.array(embeddings, dtype=np.float32)
    embedding_vector_dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(embedding_vector_dimension)
    index.reset()
    index.add(embeddings)
    print(f"\nFAISS index built with {len(embeddings)} vectors of dimension {embedding_vector_dimension}")
else:
    print("\nWARNING: No embeddings generated. FAISS index not built.")
    exit()


# %% MULTI-MODAL GEMINI RAG INFERENCE

def invoke_gemini_multimodal(prompt, matched_items):
    """
    RAG answering with Gemini using retrieved text/tables + images/pages.
    """
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
        model="gemini-2.5-flash",
        temperature=0.2,
        max_output_tokens=400,
    )

    response = llm.invoke([system_msg, user_msg])
    return response.content


# %% HELPER: BUILD MATCHED ITEMS (TEXT + RELATED PAGE CONTENT)

def build_matched_items_from_results(result_indices, k=5):
    top_indices = result_indices[0][:k]
    core_items = [embed_items[i] for i in top_indices]

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
        for it in items:
            if it["page"] == page_num and it is not core:
                if it["type"] in ["table", "image", "page"]:
                    add_item(it)

    return matched_items


# %% MULTI-QUERY EXAMPLE

other_queries = [
    "who are the Macroeconomic Indicators, 2021–25",
    "What are the key issues",
    "What is the position-wise feed-forward neural network mentioned in the paper?",
]

print("\n" + "="*80)
print("RUNNING MULTI-QUERY TEST")
print("="*80)

for query in other_queries:
    try:
        print(f"\nQuery: {query}")
        query_embedding = generate_text_embedding(query)
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        distances, result = index.search(query_embedding, k=5)
        matched_items = build_matched_items_from_results(result, k=5)

        if matched_items:
            response = invoke_gemini_multimodal(query, matched_items)
            print(f"Answer:\n{response}")
            print("-" * 80)
        else:
            print("No results found for this query")
    except Exception as e:
        print(f"Error processing query: {str(e)}")