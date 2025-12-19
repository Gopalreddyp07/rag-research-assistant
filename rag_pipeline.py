import os
import warnings
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()



# Read PDF
def read_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages")
    return pages


# Split documents
def pdf_splitter(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    split_docs = splitter.split_documents(pages)
    print(f"Split into {len(split_docs)} chunks")
    return split_docs


# Create vector store (first-time only)
def create_vector_store(split_documents):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_store = FAISS.from_documents(split_documents, embeddings)
    vector_store.save_local(FAISS_PATH)
    print("FAISS index created and saved.")
    return vector_store


# Load existing vector store
def load_vector_store():
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_store = FAISS.load_local(FAISS_PATH,embeddings,allow_dangerous_deserialization=True)
    print("FAISS index loaded from disk.")
    return vector_store


# Create RetrievalQA chain
def retrieval_qa_chain(llm, retriever, prompt):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False,
        verbose=False
    )


# Multi-turn chat
def run(qa_chain):
    print("\nWelcome to the PDF QA Bot!")
    print("-" * 50)
    print("Type 'exit' to end the conversation.\n")
    while True:
        question = input("You: ")
        if question.strip().lower() == "exit":
            print("BOT: Goodbye!")
            break
        response = qa_chain.invoke({"query": question})
        print("BOT:", response["result"])
        print("-" * 50)


# Load system prompt
with open("system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Context:\n{context}\nQuestion:\n{question}")
])

# Load LLM
llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)
FAISS_PATH = "faiss_index"

# pipeline main function
def main():
    if os.path.exists(FAISS_PATH):
        print("Existing embeddings found. Skipping ingestion steps...")
        vector_store = load_vector_store()
    else:
        print("No embeddings found. Running ingestion pipeline...")
        pages = read_pdf("Source-Data.pdf")
        split_docs = pdf_splitter(pages)
        vector_store = create_vector_store(split_docs)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    qa_chain = retrieval_qa_chain(llm, retriever, prompt)
    run(qa_chain)


if __name__ == "__main__":
    main()

