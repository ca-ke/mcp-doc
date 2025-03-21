import re
from typing import List, Tuple
from logger import logger
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
from langchain_community.document_loaders import RecursiveUrlLoader
import yaml


def bs4_extractor(html: str) -> str:
    """Extract text content from HTML using BeautifulSoup.

    Args:
        html (str): The HTML content to extract text from.

    Returns:
        str: The extracted text content.
    """
    soup = BeautifulSoup(html, "lxml")
    main_content = soup.find("article", class_="md-content__inner")
    content = main_content.get_text() if main_content else soup.text
    content = re.sub(r"\n\n+", "\n\n", content).strip()

    return content


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Count the number of tokens in the text using tiktoken.

    Args:
        text (str): The text to count tokens for.
        model (str): The tokenizer model to use (default: cl100k_base for GPT-4).

    Returns:
        int: Number of tokens in the text.
    """
    encoder = tiktoken.get_encoding(model)
    return len(encoder.encode(text))


def load_langgraph_docs() -> Tuple[List, List[int]]:
    """Load LangGraph documentation from the official website using URLs from langraph-doc.yaml.

    Returns:
        Tuple[List, List[int]]: A list of Document objects containing the loaded content and a list of token counts per document.
    """
    logger.info("Loading LangGraph documentation...")

    yaml_file_path = "langraph-doc.yaml"
    try:
        with open(yaml_file_path, "r") as file:
            langraph_doc = yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"YAML file not found: {yaml_file_path}")
        return [], []
    except yaml.YAMLError as e:
        logger.error(f"Error loading YAML file: {e}")
        return [], []

    urls = {}
    for category, content in langraph_doc.get("langraph", {}).items():
        if "guides" in content:
            urls[category] = [guide["url"] for guide in content["guides"]]
        elif "links" in content:
            urls[category] = [
                link["url"] for guide in content["guides"] for link in guide["links"]
            ]

    logger.info(f"Extracted URLs: {urls}")

    docs = []
    for category, url_list in urls.items():
        for url in url_list:
            loader = RecursiveUrlLoader(
                url,
                max_depth=5,
                extractor=bs4_extractor,
            )

            docs_lazy = loader.lazy_load()

            for d in docs_lazy:
                docs.append(d)

    logger.info(f"Loaded {len(docs)} documents from LangGraph documentation.")
    logger.info("Loaded URLs:")
    for i, doc in enumerate(docs):
        logger.info(f"{i + 1}. {doc.metadata.get('source', 'Unknown URL')}")

    total_tokens = 0
    tokens_per_doc = []
    for doc in docs:
        token_count = count_tokens(doc.page_content)
        total_tokens += token_count
        tokens_per_doc.append(token_count)

    logger.info(f"Total tokens in loaded documents: {total_tokens}")

    return docs, tokens_per_doc


def split_documents(documents: List) -> List:
    """Split documents into smaller chunks for improved retrieval.

    Args:
        documents (List): List of Document objects to split.

    Returns:
        List: A list of split Document objects.
    """
    logger.info("Splitting documents...")

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=8000, chunk_overlap=500
    )

    split_docs = text_splitter.split_documents(documents)

    logger.info(f"Created {len(split_docs)} chunks from documents.")

    total_tokens = sum(count_tokens(doc.page_content) for doc in split_docs)
    logger.info(f"Total tokens in split documents: {total_tokens}")

    return split_docs
