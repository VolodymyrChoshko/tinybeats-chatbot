from __future__ import annotations

import uuid
from itertools import repeat
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import numpy as np

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.utils import maximal_marginal_relevance
from server.vectorstores.base import VectorStore

if TYPE_CHECKING:
    import supabase

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.graphics.shapes import Drawing
from reportlab.graphics import renderPDF
from reportlab.lib.units import inch
from io import BytesIO  # Required for in-memory PDF creation
import os
import io


class SupabaseVectorStore(VectorStore):
    """VectorStore for a Supabase postgres database. Assumes you have the `pgvector`
    extension installed and a `match_documents` (or similar) function. For more details:
    https://js.langchain.com/docs/modules/indexes/vector_stores/integrations/supabase

    You can implement your own `match_documents` function in order to limit the search
    space to a subset of documents based on your own authorization or business logic.

    Note that the Supabase Python client does not yet support async operations.

    If you'd like to use `max_marginal_relevance_search`, please review the instructions
    below on modifying the `match_documents` function to return matched embeddings.
    """

    _client: supabase.client.Client
    # This is the embedding function. Don't confuse with the embedding vectors.
    # We should perhaps rename the underlying Embedding base class to EmbeddingFunction
    # or something
    _embedding: Embeddings
    table_name: str
    query_name: str

    def __init__(
        self,
        client: supabase.client.Client,
        embedding: Embeddings,
        table_name: str,
        query_category_index: int,
        query_name: Union[str, None] = None
    ) -> None:
        """Initialize with supabase client."""
        try:
            import supabase  # noqa: F401
        except ImportError:
            raise ValueError(
                "Could not import supabase python package. "
                "Please install it with `pip install supabase`."
            )

        self._client = client
        self._embedding: Embeddings = embedding
        self.table_name = table_name or "documents"
        self.query_name = query_name or "match_documents"
        self.query_category_index = query_category_index

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        docs = self._texts_to_documents(texts, metadatas)

        vectors = self._embedding.embed_documents(list(texts))
        return self.add_vectors(vectors, docs, ids)

    def get_category_index(self) -> int:
        return self.query_category_index

    @classmethod
    def from_texts(
        cls: Type["SupabaseVectorStore"],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        client: Optional[supabase.client.Client] = None,
        table_name: Optional[str] = "documents",
        query_name: Union[str, None] = "match_documents",
        ids: Optional[List[str]] = None,
        category_index: Optional[int] = None,
        metadata_addition: dict = {},
        **kwargs: Any,
    ) -> "SupabaseVectorStore":
        """Return VectorStore initialized from texts and embeddings."""

        if not client:
            raise ValueError("Supabase client is required.")

        if not table_name:
            raise ValueError("Supabase document table_name is required.")

        embeddings = embedding.embed_documents(texts)
        ids = [str(uuid.uuid4()) for _ in texts]
        docs = cls._texts_to_documents(texts, metadatas)
        _ids = cls._add_vectors(client, table_name, embeddings, docs, ids, metadata_addition , category_index,)

        return cls(
            client=client,
            embedding=embedding,
            table_name=table_name,
            query_name=query_name,
            query_category_index=category_index
        )

    def add_vectors(
        self,
        vectors: List[List[float]],
        documents: List[Document],
        ids: List[str],
    ) -> List[str]:
        return self._add_vectors(self._client, self.table_name, vectors, documents, ids)

    def similarity_search(
        self,
        query: str,
        query_category_index: int,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        vectors = self._embedding.embed_documents([query])
        return self.similarity_search_by_vector(
            vectors[0], k=k, filter=filter, query_category_index=query_category_index , **kwargs
        )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        query_category_index: int,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        result = self.similarity_search_by_vector_with_relevance_scores(
            embedding, k=k, filter=filter, query_category_index=query_category_index , **kwargs
        )

        documents = [doc for doc, _ in result]

        return documents

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        vectors = self._embedding.embed_documents([query])
        return self.similarity_search_by_vector_with_relevance_scores(
            vectors[0], k=k, filter=filter
        )

    def match_args(
        self, query: List[float], k: int,query_category_index: int, filter: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        ret = dict(query_embedding=query, match_count=k, query_category_index=query_category_index)
        if filter:
            ret["filter"] = filter
        return ret

    def similarity_search_by_vector_with_relevance_scores(
        self, query: List[float], k: int, query_category_index: int ,filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        match_documents_params = self.match_args(query, k, query_category_index, filter)

        ### error here *** mark danger ####
        try:
            res = self._client.rpc(self.query_name, match_documents_params).execute()
        except Exception as e:
            # print(e)
            raise e

        match_result = [
            (
                Document(
                    metadata=search.get("metadata", {}),  # type: ignore
                    page_content=search.get("content", ""),
                ),
                search.get("similarity", 0.0),
            )
            for search in res.data
            if search.get("content")
        ]

        return match_result

    def similarity_search_by_vector_returning_embeddings(
        self, query: List[float], k: int, filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float, np.ndarray[np.float32, Any]]]:
        match_documents_params = self.match_args(query, k, filter)
        res = self._client.rpc(self.query_name, match_documents_params).execute()

        match_result = [
            (
                Document(
                    metadata=search.get("metadata", {}),  # type: ignore
                    page_content=search.get("content", ""),
                ),
                search.get("similarity", 0.0),
                # Supabase returns a vector type as its string represation (!).
                # This is a hack to convert the string to numpy array.
                np.fromstring(
                    search.get("embedding", "").strip("[]"), np.float32, sep=","
                ),
            )
            for search in res.data
            if search.get("content")
        ]

        return match_result

    @staticmethod
    def _texts_to_documents(
        texts: Iterable[str],
        metadatas: Optional[Iterable[Dict[Any, Any]]] = None,
    ) -> List[Document]:
        """Return list of Documents from list of texts and metadatas."""
        if metadatas is None:
            metadatas = repeat({})

        docs = [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(texts, metadatas)
        ]
        # print(docs)

        return docs

    @staticmethod
    def _add_vectors(
        client: supabase.client.Client,
        table_name: str,
        vectors: List[List[float]],
        documents: List[Document],
        ids: List[str],
        metadata_addition: Dict ={},
        category_index: int = 0
    ) -> List[str]:
        """Add vectors to Supabase table."""
        print("adding vectors to supabase")
        print("metadata_addition", metadata_addition)
        # Initialize an empty list to store the dictionaries
        metadata_list = []

        # Loop through the range of vectors
        for i in range(len(vectors)):
            # Create a dictionary with the 'filename' key and the desired value
            filename = metadata_addition['filename'].split('_')[0] + f'-{str(i)}' + '_' + metadata_addition['filename'].split('_')[-1]
            metadata_dict = {'filename': filename}
            
            # Append the dictionary to the list
            metadata_list.append(metadata_dict)
        
        # print("metadata_list", metadata_list)
        for i in range(len(vectors)):
            pdf_buffer = BytesIO()
            elements = []

            # Create a paragraph style
            styles = getSampleStyleSheet()
            normal_style = styles["Normal"]
            normal_style.alignment = 0  # Left-align text

            # Create a paragraph from the content text
            content_paragraph = Paragraph(documents[i].page_content, normal_style)
            elements.append(content_paragraph)

            # Create the PDF document, but don't save it to disk
            doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)

            # Build the PDF document
            doc.build(elements)
            try:
                response = client.storage.from_(os.environ.get('BUCKET_NAME')).upload(metadata_list[i]['filename'], pdf_buffer.getvalue())
            except Exception as E:
                print("exception occured", E)

            pdf_buffer.seek(0)
            pdf_buffer.truncate()


        
        rows: List[Dict[str, Any]] = [
            {
                "id": ids[idx],
                "content": documents[idx].page_content,
                "embedding": embedding,
                # "metadata": documents[idx].metadata.update(metadata_addition),  # type: ignore
                "metadata": {**documents[idx].metadata, **metadata_list[idx]},
                "category_index" : category_index
            }
            for idx, embedding in enumerate(vectors)
        ]

        # According to the SupabaseVectorStore JS implementation, the best chunk size
        # is 500
        chunk_size = 500
        id_list: List[str] = []
        for i in range(0, len(rows), chunk_size):
            chunk = rows[i : i + chunk_size]
            # update chunk metadata['filename'] with chunk start and end point 
            # chunk[0]['metadata']['filename'] = f"{i}_{i + chunk_size}"

            result = client.from_(table_name).upsert(chunk).execute()  # type: ignore

            if len(result.data) == 0:
                raise Exception("Error inserting: No rows added")

            # VectorStore.add_vectors returns ids as strings
            ids = [str(i.get("id")) for i in result.data if i.get("id")]

            id_list.extend(ids)

        return id_list

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        result = self.similarity_search_by_vector_returning_embeddings(
            embedding, fetch_k
        )

        matched_documents = [doc_tuple[0] for doc_tuple in result]
        matched_embeddings = [doc_tuple[2] for doc_tuple in result]

        mmr_selected = maximal_marginal_relevance(
            np.array([embedding], dtype=np.float32),
            matched_embeddings,
            k=k,
            lambda_mult=lambda_mult,
        )

        filtered_documents = [matched_documents[i] for i in mmr_selected]

        return filtered_documents

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance.

        `max_marginal_relevance_search` requires that `query_name` returns matched
        embeddings alongside the match documents. The following function
        demonstrates how to do this:

        ```sql
        CREATE FUNCTION match_documents_embeddings(query_embedding vector(1536),
                                                   match_count int)
            RETURNS TABLE(
                id uuid,
                content text,
                metadata jsonb,
                embedding vector(1536),
                similarity float)
            LANGUAGE plpgsql
            AS $$
            # variable_conflict use_column
        BEGIN
            RETURN query
            SELECT
                id,
                content,
                metadata,
                embedding,
                1 -(docstore.embedding <=> query_embedding) AS similarity
            FROM
                docstore
            ORDER BY
                docstore.embedding <=> query_embedding
            LIMIT match_count;
        END;
        $$;
        ```
        """
        embedding = self._embedding.embed_documents([query])
        docs = self.max_marginal_relevance_search_by_vector(
            embedding[0], k, fetch_k, lambda_mult=lambda_mult
        )
        return docs

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        """Delete by vector IDs.

        Args:
            ids: List of ids to delete.
        """

        if ids is None:
            raise ValueError("No ids provided to delete.")

        rows: List[Dict[str, Any]] = [
            {
                "id": id,
            }
            for id in ids
        ]

        # TODO: Check if this can be done in bulk
        for row in rows:
            self._client.from_(self.table_name).delete().eq("id", row["id"]).execute()
