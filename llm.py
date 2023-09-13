import langchain
from langchain import HuggingFaceHub
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain

def llm_conv(filename):
    document_loader = PyPDFLoader(filename)
    chunks = document_loader.load_and_split()
    embeddings = HuggingFaceHubEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)
    return db, chunks

def similarity(filename, repo_id, model_kwargs, query):
    db, chunks = llm_conv(filename)
    docs = db.similarity_search(query)
    chain = load_qa_chain(
        HuggingFaceHub(
            repo_id=repo_id,
            model_kwargs=model_kwargs
        ),
        chain_type="stuff"
    )
    question = f"""
    Answer the question based on the context, if you don't know then output "Out of Context".
    Context: \n {chunks[0].page_content} \n
    Question: \n {query} \n
    Answer:
    """
    result = chain.run(
        input_documents=docs,
        question=question
    )
    return result

