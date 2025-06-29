# Import
import gc
import os
import glob
import pandas as pd
from dotenv import load_dotenv
import gradio as gr
from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from encrypt import EncryptedFAISS, EncryptedRetriever
import atexit
from langchain.prompts import PromptTemplate


MODEL_GPT = "gpt-4o-mini"
MY_FOLDER = "Your Folder Path"
SYSTEM_MSG = """Use the following pieces of context to answer the user's question. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Question: {question}

    Answer:"""

CUSTOM_PROMPT = PromptTemplate(
    template=SYSTEM_MSG,
    input_variables=["context", "question"]
)


class RAGWorkflow:
    __author__ = "Shahril Mohd"
    __contact__ = "mohd.shahrils@yahoo.com"
    __copyright__ = "Copyright (c) 2025, Shahril Mohd"

    def __init__(self, model=MODEL_GPT, db_name="vector_db"):
        self.model = model
        self.db_name = db_name
        self.documents = []
        self.chunks = []
        self.vectorstore = None
        self.encrypted_vectorstore = None
        self.memory = None
        self.conversation_chain = None
        self.view = None

        # Cleanup on exit
        atexit.register(self.cleanup)

        # Initialise OpenAI API
        self._setup_openai()

    def _setup_openai(self):
        # Load environment variables in a file called .env
        load_dotenv(override=True)
        os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

        if os.environ['OPENAI_API_KEY']:
            print(f"OpenAI API Key is set")
        else:
            print("OpenAI API Key not set")

    def sanitize_metadata(self, metadata):
        """Sanitize metadata to remove or mask sensitive information."""
        sensitive_fields = ['source', 'sheet_names']
        sanitized = metadata.copy()
        for field in sensitive_fields:
            if field in sanitized:
                if isinstance(sanitized[field], str):
                    sanitized[field] = os.path.basename(sanitized[field])
                elif isinstance(sanitized[field], (list, tuple)):
                    sanitized[field] = [f'[MASKED_{field}_{i}]' for i in range(len(sanitized[field]))]
        return sanitized

    def process_excel_safely(self, excel_file, doc_type):
        """Process an Excel file safely and return a list of documents."""
        try:
            excel = pd.ExcelFile(excel_file)
            all_sheets_content = []

            # Process for each sheet
            for sheet_name in excel.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                sheet_content = []

                # Convert DataFrame to a structured text format
                for index, row in df.iterrows():
                    # Create a formatted string for each row
                    row_content = " | ".join(f"{col}: {row[col]}" for col in df.columns)
                    sheet_content.append(f"Row {index + 1}: {row_content}")

                # Add sheet name and its content
                all_sheets_content.append(f"\nSheet: {sheet_name}\n" + "\n".join(sheet_content))

                # Clear sheet-level data
                del sheet_content
                del df

            # Join all sheets' content into a single string
            full_content = "\n\n".join(all_sheets_content)

            # Create a single document for the Excel file
            excel_doc = Document(
                page_content=full_content,
                metadata=self.sanitize_metadata({
                    "source": excel_file,
                    "doc_type": doc_type,
                    "sheets": len(excel.sheet_names),
                    "sheet_names": excel.sheet_names
                })
            )

            # Clear sensitive data
            del all_sheets_content
            del full_content
            del excel
            gc.collect()

            return excel_doc

        except Exception as e:
            print(f"Error processing Excel file: {excel_file}")
            #print(f"Error processing Excel file {excel_file}: {str(e)}")
            return None

    def process_documents(self, folder_path=MY_FOLDER):
        """Process documents in the specified folder and return a list of documents."""
        # Read in documents using LangChain's Loaders
        folders = glob.glob(folder_path)
        text_loader_kwargs = {'encoding': ['utf-8', 'cp1252', 'latin-1', 'iso-8859-1']}

        for folder in folders:
            doc_type = os.path.basename(folder)

            # Create separate loaders for text and PDF files
            text_loader = DirectoryLoader(
                folder,
                glob="**/*.txt",  # Add other text extensions if needed: "**/*.[txt|md|...]"
                loader_cls=TextLoader,
                loader_kwargs=text_loader_kwargs
            )

            pdf_loader = DirectoryLoader(
                folder,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            )

            # Load both text and PDF documents
            folder_docs = text_loader.load() + pdf_loader.load()

            # Process Excel files
            excel_files = glob.glob(os.path.join(folder, "*.xlsx"), recursive=True)
            for excel_file in excel_files:
                excel_doc = self.process_excel_safely(excel_file, doc_type)
                if excel_doc:
                    self.documents.append(excel_doc)

            # Process other documents
            for doc in folder_docs:
                doc.metadata = self.sanitize_metadata({
                    "doc_type": doc_type,
                    **doc.metadata
                })
                self.documents.append(doc)

    def create_vectorstore(self):
        """Create and encrypt a vector store from the processed documents."""
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.chunks = text_splitter.split_documents(self.documents)

        print(len(self.chunks))
        doc_types = set(chunk.metadata['doc_type'] for chunk in self.chunks)
        print(f"Document types found: {', '.join(doc_types)}")

        # Put the chunks into a Vector Store that associates a Vector Embedding with each chunk
        embeddings = OpenAIEmbeddings()
        # Create vector store
        self.vectorstore = FAISS.from_documents(self.chunks, embedding=embeddings)

        # Create encryptor and encrypt the vectorstore
        encryptor = EncryptedFAISS()
        self.encrypted_vectorstore = encryptor.encrypt_vectorstore(self.vectorstore)

        total_vectors = self.encrypted_vectorstore.index.ntotal
        dimensions = self.encrypted_vectorstore.index.d

        print(f"There are {total_vectors} vectors with {dimensions:,} dimensions in the vector store")

    def setup_chat(self):
        """Set up the chat interface with encrypted retriever."""
        # RAG workflow using LangChain
        # 1. Create a new chat with OpenAI
        llm = ChatOpenAI(temperature=0.7, model_name=self.model)

        # 2. Set up the conversation memory for the chat and use encrypted memory storage
        self.memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        )

        # 3.5 The retriever is an abstraction over the encrypted VectorStore that will be used during RAG
        retriever = EncryptedRetriever(vectorstore=self.encrypted_vectorstore)

        # 4. Putting it together: set up the conversation chain with the GPT 3.5 LLM, the vector store and memory
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT},
            verbose=False
        )


    def chat(self, message, history):
        """Chat with the bot."""
        result = self.conversation_chain.invoke({"question": message})
        return result["answer"]

    def launch_interface(self):
        """Launch the chat interface."""
        self.view = gr.ChatInterface(self.chat, type="messages").launch(inbrowser=True)

    def cleanup(self):
        """Clean up resources when the program exits."""
        try:
            if self.memory:
                self.memory.clear()

            # Clear document collections
            self.documents.clear()
            if self.chunks:
                self.chunks.clear()

            # Clear vector stores
            self.vectorstore = None
            self.encrypted_vectorstore = None

            # Force garbage collection
            gc.collect()

            print("Cleanup completed successfully")
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")


def main():
    # Initialize and run the application
    workflow = RAGWorkflow()
    workflow.process_documents()
    workflow.create_vectorstore()
    workflow.setup_chat()
    workflow.launch_interface()

if __name__ == "__main__":
    main()




