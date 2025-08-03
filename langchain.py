from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# import la

def store_transcripts_to_vector_db(yt_video_url: str) -> FAISS:
    """
    Store YouTube video transcripts in a vector database.
    Args:
        url (str): The URL of the YouTube video.
    Returns:
        vector_db (Faissdatabase): The vector database containing the video transcripts.
    """
    # Load the YouTube video transcript
    loader = YoutubeLoader.from_youtube_url(
        youtube_url=yt_video_url,
    )
    documents = loader.load()

    # Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)

    # Create embeddings for the documents
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
    )

    # Store the documents in a vector database
    vector_db = FAISS.from_documents(split_docs, embeddings)

    return vector_db

def query_results(vector_db: FAISS, query: str) -> list:
    """
    Query the vector database for relevant documents.
    Args:
        vector_db (FAISS): The vector database containing the video transcripts.
        query (str): The query string to search for.
    Returns:
        results (list): A list of documents that match the query.
    """
    llm = ChatOpenAI(
        model="gpt-4.1-nano",
        temperature=0.5,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    results = vector_db.similarity_search(query)
    return results

# print(store_transcripts_to_vector_db("https://www.youtube.com/watch?v=zzWypOl4JkY"))