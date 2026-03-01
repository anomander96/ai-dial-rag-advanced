from enum import StrEnum

import psycopg2
from psycopg2.extras import RealDictCursor

from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.utils.text import chunk_text


class SearchMode(StrEnum):
    EUCLIDIAN_DISTANCE = "euclidean"  # Euclidean distance (<->)
    COSINE_DISTANCE = "cosine"  # Cosine distance (<=>)


class TextProcessor:
    """Processor for text documents that handles chunking, embedding, storing, and retrieval"""

    def __init__(self, embeddings_client: DialEmbeddingsClient, db_config: dict):
        self.embeddings_client = embeddings_client
        self.db_config = db_config

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password']
        )

    def process_text_file(self, file_name: str, chunk_size: int, overlap: int, dimensions: int, truncate: bool):
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                if truncate:
                    cursor.execute("TRUNCATE TABLE vectors")
                    print("🗑️ Table truncated")

                # load file content
                with open(file_name, 'r', encoding = 'utf-8') as f:
                    content = f.read()
                
                # split text into chunks
                chunks = chunk_text(content, chunk_size = chunk_size, overlap = overlap)

                # generate embeddings
                embeddings = self.embeddings_client.get_embeddings(
                    input_texts = chunks,
                    dimensions = dimensions
                )
                print(f"🔢 Generated {len(embeddings)} embeddings")

                # insert chunk and it embeddings into DB
                for i, chunk in enumerate(chunks):
                    embedding_vector = embeddings[i]

                    # convert to string
                    embedding_str = str(embedding_vector)

                    cursor.execute(
                        "INSERT INTO vectors(text, embedding) VALUES (%s, %s::vector)",
                        (chunk, embedding_str)
                    )

                # commit transaction
                conn.commit()
                print(f"✅ Saved {len(chunks)} chunks to DB")


    def search(self, search_mode: SearchMode, query: str, top_k: int = 5, min_score: float = 0.5, dimensions: int = 1536) -> list:
        # generate embeddings for search query
        query_embeddings = self.embeddings_client.get_embeddings(
            input_texts = [query],
            dimensions = dimensions
        )

        query_vector = str(query_embeddings[0])

        if search_mode == SearchMode.EUCLIDIAN_DISTANCE:
            distance_op = "<->"
        else:
            distance_op = "<=>"

        # build search query
        sql = f"""
            SELECT text, embedding {distance_op} %s::vector AS distance
            FROM vectors
            WHERE embedding {distance_op} %s::vector < %s
            ORDER BY distance
            LIMIT %s
        """

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory = RealDictCursor) as cursor:
                cursor.execute(sql, (query_vector, query_vector, min_score, top_k))
                results = cursor.fetchall()

        print(f"🔍 Found {len(results)} relevant chunks")
        return results


