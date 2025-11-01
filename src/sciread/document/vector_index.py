"""Vector index wrapper for semantic search using ChromaDB."""

from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import chromadb

from .models import Chunk


class VectorIndex:
    """A wrapper around a vector database for semantic search."""

    def __init__(self, collection_name: str, persist_path: Optional[Path] = None):
        """Initialize the vector index.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_path: Path to persist the vector database (optional)
        """
        self.persist_path = persist_path
        if self.persist_path:
            self._client = chromadb.PersistentClient(path=str(self.persist_path))
        else:
            self._client = chromadb.Client()

        # Use cosine similarity for better semantic search
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity instead of L2
        )

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """Adds chunks and their embeddings to the collection.

        Args:
            chunks: List of chunks to add
            embeddings: List of embedding vectors corresponding to chunks
        """
        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(f"Number of chunks ({len(chunks)}) must match number of embeddings ({len(embeddings)})")

        self._collection.add(
            embeddings=embeddings,
            documents=[chunk.content for chunk in chunks],
            metadatas=[
                {
                    "source": chunk.chunk_name,
                    "position": chunk.position,
                    "word_count": chunk.word_count,
                    "confidence": chunk.confidence,
                }
                for chunk in chunks
            ],
            ids=[chunk.id for chunk in chunks],
        )

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Performs a semantic search and returns the top_k results.

        Args:
            query_embedding: Embedding vector for the query
            top_k: Number of results to return

        Returns:
            List of search results with id, similarity (cosine), metadata, and content
            Similarity scores are in range [0, 1] where 1 is most similar
        """
        try:
            results = self._collection.query(query_embeddings=[query_embedding], n_results=top_k)
        except Exception as e:
            raise RuntimeError(f"Failed to query vector index: {e}") from e

        if not results or not results.get("ids"):
            return []

        result_list = []
        ids, distances, metadatas, documents = (
            results["ids"][0],
            results["distances"][0],
            results["metadatas"][0],
            results["documents"][0],
        )

        for i in range(len(ids)):
            # ChromaDB with cosine distance returns values in [0, 2]
            # where 0 = identical, 2 = opposite
            # Convert to similarity score [0, 1] where 1 = identical
            cosine_distance = distances[i]
            similarity = 1 - (cosine_distance / 2)  # Convert to similarity

            result_list.append(
                {
                    "id": ids[i],
                    "similarity": similarity,
                    "distance": cosine_distance,
                    "metadata": metadatas[i],
                    "content": documents[i],
                }
            )
        return result_list

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            count = self._collection.count()
            return {
                "name": self._collection.name,
                "count": count,
                "persist_path": str(self.persist_path) if self.persist_path else None,
            }
        except Exception as e:
            return {"error": str(e)}

    def delete_collection(self) -> None:
        """Delete the collection."""
        try:
            self._client.delete_collection(name=self._collection.name)
        except Exception as e:
            raise RuntimeError(f"Failed to delete collection: {e}") from e
