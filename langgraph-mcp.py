from mcp.server.fastmcp import FastMCP
from vectorstore_utils import load_vectorstore

mcp = FastMCP("LangGraph-Docs-MCP-Server")

@mcp.tool()
def langgraph_query_tool(query: str):
        """
        Query the LangGraph documentation using a retriever.
        
        Args:
            query (str): The query to search the documentation with

        Returns:
            str: A str of the retrieved documents
        """
        retriever = load_vectorstore().as_retriever(search_kwargs={"k": 3})

        relevant_docs = retriever.invoke(query)
        print(f"Retrieved {len(relevant_docs)} relevant documents")
        formatted_context = "\n\n".join([f"==DOCUMENT {i+1}==\n{doc.page_content}" for i, doc in enumerate(relevant_docs)])
        return formatted_context
    
if __name__ == "__main__":
    mcp.run(transport='stdio')