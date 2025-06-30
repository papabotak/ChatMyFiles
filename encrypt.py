import faiss
from cryptography.fernet import Fernet
from typing import List, Dict, Any
from langchain.schema import Document, BaseRetriever
import os
import pickle
import re
import copy
import numpy as np
from langchain.vectorstores import FAISS as FAISSVectorStore
import json
from pydantic import BaseModel, Field



# Class to encrypting vectors and chunks before using the standard encryption libraries

class EncryptedFAISS:
    def __init__(self):
        # Generate and save encryption key securely
        self.key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)

    def encrypt_chunks(self, chunks: List[Document]):
        # Encrypt the text content and metadata of each chunk
        encrypted_chunks = []
        for chunk in chunks:
            encrypted_content = self.cipher_suite.encrypt(chunk.page_content.encode())
            encrypted_metadata = self.encrypt_metadata(chunk.metadata)

            encrypted_chunk = Document(
                page_content=encrypted_content,
                metadata={'encrypted_metadata': encrypted_metadata}
            )
            #chunk.page_content = encrypted_content
            #encrypted_chunks.append(chunk)
            encrypted_chunks.append(encrypted_chunk)
        return encrypted_chunks

    def decrypt_chunks(self, encrypted_chunk: Document) -> Document:
        """Decrypt both content and metadata of a chunk"""
        try:
            # Decrypt content
            decrypted_content = self.cipher_suite.decrypt(encrypted_chunk.page_content).decode()

            # Decrypt metadata
            decrypted_metadata = self.decrypt_metadata(encrypted_chunk.metadata['encrypted_metadata'])

            return Document(
                page_content=decrypted_content,
                metadata=decrypted_metadata
            )
        except Exception as e:
            print(f"Error decrypting chunk: {str(e)}")
            return None


    def encrypt_data(self, vectors):
        # To ensure the encryption process does not altering the size of data
        original_shape = vectors.shape
        original_dtype = vectors.dtype

        # Convert vectors to bytes and encrypt the bytes using the cipher suite (Fernet)
        vectors_bytes = vectors.tobytes()
        # Pad the vectors to ensure consistent size after encryption
        padding_length = 16 - (len(vectors_bytes) % 16) # Fernet uses 16 bytes as block size
        padded_bytes = vectors_bytes + b'\0' * padding_length
        # Encrypt the padded bytes
        encrypted_bytes = self.cipher_suite.encrypt(padded_bytes)

        # Convert back to array and removing padding
        decrypted_bytes = self.cipher_suite.decrypt(encrypted_bytes)
        original_size = np.prod(original_shape)

        # Convert back the bytes to numpy array and return the encrypted vectors
        encrypted_vectors = np.frombuffer(
            decrypted_bytes[:original_size * vectors.dtype.itemsize],
            dtype=original_dtype
        )
        encrypted_vectors = encrypted_vectors.reshape(original_shape)
        return encrypted_vectors

    def encrypt_vectorstore(self, vectorstore):
        # Encrypt the vector in the FAISS index
        vectors = vectorstore.index.reconstruct_n(0, vectorstore.index.ntotal)
        encrypted_vectors = self.encrypt_data(vectors)

        # Create a new FAISS index with the encrypted vectors
        new_index = faiss.IndexFlatL2(vectorstore.index.d)
        new_index.add(encrypted_vectors)

        # Encrypt all documents in the docstore
        encrypted_docstore = {}
        for doc_id, doc in vectorstore.docstore._dict.items():
            encrypted_doc = self.encrypt_chunks([doc])[0]
            encrypted_docstore[doc_id] = encrypted_doc

        # Create a new vector store with the encrypted index and create a new FAISS instane
        encrypted_vectorstore = FAISSVectorStore(
            embedding_function=vectorstore.embeddings,
            index=new_index,
            docstore=vectorstore.docstore,
            index_to_docstore_id=vectorstore.index_to_docstore_id.copy()
        )

        # Add decrypt method to vectorstore for retrieval
        encrypted_vectorstore.decrypt_doc = self.decrypt_chunks

        return encrypted_vectorstore

    def encrypt_metadata(self, metadata: Dict) -> bytes:
        """Encrypt metadata using dictionary."""
        metadata_json = json.dumps(metadata)
        return self.cipher_suite.encrypt(metadata_json.encode())

    def decrypt_metadata(self, encrypted_metadata: bytes) -> Dict:
        """Decrypt metadata using dictionary."""
        decrypted_json = self.cipher_suite.decrypt(encrypted_metadata).decode()
        return json.loads(decrypted_json)



# Class to encrypt the entire vector store

class SecureVectorStore:
    def __init__(self, key_path='encryption_key.key'):
        """
        Initialize the SecureVectorStore with encryption key management.

        Args:
            key_path (str): Path to store/retrieve the encryption key
        """

        self.key_path = key_path
        if not os.path.exists(key_path):
            # Generate a new encryption key and save it to a file
            self.key = Fernet.generate_key()
            with open(key_path, 'wb') as key_file:
                key_file.write(self.key)
        else:
            # Load the existing encryption key from a file
            with open(key_path, 'rb') as key_file:
                self.key = key_file.read()
        # Create the cipher suite for encryption/decryption
        self.cipher_suite = Fernet(self.key)

    def save_encrypted_vectorstore(self, vectorstore, path="encrypted_vectorstore"):
        # Serialised and encrypt the vectorstore
        """
        Encrypt and save the vector store to disk.

        Args:
            vectorstore: The FAISS vector store to encrypt and save
            path (str): Path where to save the encrypted vector store
        """

        serialized_vs = pickle.dumps(vectorstore)
        encrypted_vs = self.cipher_suite.encrypt(serialized_vs)

        with open(path, "wb") as f:
            f.write(encrypted_vs)

    def load_encrypted_vectorstore(self, path="encrypted_vectorstore"):
        # Decrypt and deserialise the vector store
        """
        Load and decrypt the vector store from disk.

        Args:
            path (str): Path to the encrypted vector store

        Returns:
            The decrypted FAISS vector store
        """

        with open(path, "rb") as f:
            encrypted_vs = f.read()
        decrypted_vs = self.cipher_suite.decrypt(encrypted_vs)
        return pickle.loads(decrypted_vs)

    def verify_integrity(self, path="encrypted_vectorstore"):
        """
        Verify the integrity of the encrypted vector store.

        Args:
            path (str): Path to the encrypted vector store

        Returns:
            bool: True if verification passes

        """
        try:
            loaded_vs = self.load_encrypted_vectorstore(path)
            return True
        except Exception as e:
            print(f"Verification failed: {e}")
            return False


class DataMasker:
    def __init__(self):
        self.patterns = {
            'email': 'mohd.shahrils@yahoo.com',
        }

    def mask_sensitive_data(self, text):
        masked_text = text
        for data_type, pattern in self.patterns.items():
            masked_text = re.sub(pattern, f'[MASKED_{data_type}]', masked_text)
            return masked_text


# Class retriever to handle decryption
class EncryptedRetriever(BaseRetriever, BaseModel):
    vectorstore: Any = Field(description="Vector store for document retrieval")

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str) -> List[Document]:
        # Implement the retrieval logic here
        return self.vectorstore.similarity_search(query)

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        # Implement the async retrieval logic here
        # For now, we can make it call the sync version
        return self.get_relevant_documents(query)


    # def get_relevant_documents(self, query):
    #     # Get encrypted docs
    #     docs = self.vectorstore.similarity_search(query)
    #     # Decrypt docs
    #     decrypted_docs = [self.vectorstore.decrypt_doc(doc) for doc in docs]
    #     return [doc for doc in decrypted_docs if doc is not None]
    #
    # async def aget_relevant_documents(self, query):
    #     return self.get_relevant_documents(query)


