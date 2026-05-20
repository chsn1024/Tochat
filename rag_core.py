import os
import re
from typing import Any, Dict, List, Tuple

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")

KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.txt")
DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "freshman_guide"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"


class RagNotReadyError(RuntimeError):
    pass


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
    qa_matches = list(re.finditer(r"(?m)^##\s*Q\d+[:：]", text))
    if qa_matches:
        chunks = []
        for index, match in enumerate(qa_matches):
            start = match.start()
            end = qa_matches[index + 1].start() if index + 1 < len(qa_matches) else len(text)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def _get_embedding_function(local_files_only: bool = False) -> Any:
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    except ImportError as exc:
        raise RagNotReadyError(
            "缺少 RAG 依赖，请先运行：pip install chromadb sentence-transformers openai"
        ) from exc

    try:
        return SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        raise RagNotReadyError(
            "embedding 模型未能加载，请联网运行一次：python build_index.py"
        ) from exc


def _get_chroma_client() -> Any:
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError as exc:
        raise RagNotReadyError(
            "缺少 RAG 依赖，请先运行：pip install chromadb sentence-transformers openai"
        ) from exc

    return chromadb.PersistentClient(
        path=DB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection() -> Any:
    client = _get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_get_embedding_function(local_files_only=True),
    )


def build_index() -> int:
    if not os.path.exists(KB_PATH):
        raise FileNotFoundError(f"知识库文件不存在：{KB_PATH}")

    with open(KB_PATH, "r", encoding="utf-8") as file:
        text = file.read()

    chunks = chunk_text(text)
    client = _get_chroma_client()

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_get_embedding_function(local_files_only=False),
    )

    collection.add(
        ids=[f"freshman_chunk_{index}" for index in range(len(chunks))],
        documents=chunks,
        metadatas=[
            {
                "source": os.path.basename(KB_PATH),
                "chunk_id": index,
            }
            for index in range(len(chunks))
        ],
    )
    return len(chunks)


def retrieve(question: str, top_k: int = 3) -> Tuple[List[str], List[Dict[str, Any]]]:
    collection = get_collection()
    try:
        count = collection.count()
    except Exception as exc:
        raise RagNotReadyError("知识库索引不可用，请先运行：python build_index.py") from exc

    if count == 0:
        raise RagNotReadyError("知识库索引为空，请先运行：python build_index.py")

    results = collection.query(query_texts=[question], n_results=top_k)
    return results["documents"][0], results["metadatas"][0]


def format_sources(metadatas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sources = []
    seen = set()
    for metadata in metadatas:
        source = str(metadata.get("source", os.path.basename(KB_PATH)))
        chunk_id = metadata.get("chunk_id", "")
        key = (source, chunk_id)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"source": source, "chunk_id": chunk_id})
    return sources


def build_context(docs: List[str], metadatas: List[Dict[str, Any]]) -> str:
    context_parts = []
    for doc, metadata in zip(docs, metadatas):
        context_parts.append(
            f"【来源：{metadata.get('source')}，chunk_id：{metadata.get('chunk_id')}】\n{doc}"
        )
    return "\n\n".join(context_parts)
