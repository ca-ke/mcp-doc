import os
from typing import List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.sklearn import SKLearnVectorStore
from logger import logger

PERSIST_PATH = os.path.join(os.getcwd(), "sklearn_vectorstore.parquet")

def load_vectorstore() -> Optional[SKLearnVectorStore]:
    """
    Load an existing SKLearnVectorStore from the specified path.

    Returns:
        SKLearnVectorStore: The loaded vector store, or None if it does not exist.
    """
    if os.path.exists(PERSIST_PATH):
        logger.info("Loading existing SKLearnVectorStore from %s", PERSIST_PATH)
        try:
            vectorstore = SKLearnVectorStore(
                persist_path=PERSIST_PATH,
                embedding=HuggingFaceEmbeddings(),
                serializer="parquet",
            )
            return vectorstore
        except Exception as e:
            logger.error("Error loading vector store: %s", e)
            return None
    else:
        logger.warning("No existing vector store found at %s", PERSIST_PATH)
        return None

def create_vectorstore(splits: List) -> SKLearnVectorStore:
    """
    Create a vector store from document chunks using SKLearnVectorStore.

    This function:
    1. Initializes an embedding model to convert text into vector representations.
    2. Creates a vector store from the document chunks.

    Args:
        splits (List): List of split Document objects to embed.

    Returns:
        SKLearnVectorStore: A vector store containing the embedded documents.
    """
    logger.info("Creating SKLearnVectorStore...")

    vectorstore = SKLearnVectorStore.from_documents(
        documents=splits,
        embedding=HuggingFaceEmbeddings(),
        persist_path=PERSIST_PATH,
        serializer="parquet",
    )
    logger.info("SKLearnVectorStore created successfully.")

    vectorstore.persist()
    logger.info("SKLearnVectorStore was persisted to %s", PERSIST_PATH)

    return vectorstore