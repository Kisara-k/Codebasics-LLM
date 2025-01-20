import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS

llm = OpenAI(temperature=0.8)

st.title("News Research Tool")
st.sidebar.title("News Article URLs")

n = st.sidebar.number_input("Number of URLs to retrieve", min_value=1, max_value=10, value=5)
blank_url = ""

urls = []
for i in range(n):
    urls.append(st.sidebar.text_input(f"Enter URL {i+1}", blank_url))

process_urls_clicked = st.sidebar.button("Process URLs")


main_info = st.empty()
if process_urls_clicked:

    urls = [url.strip() for url in urls if url != blank_url]
    if urls:

        main_info.write(f"Processing {len(urls)} URLs...")
        loader = UnstructuredURLLoader(urls)
        data = loader.load()

        main_info.write("Splitting documents...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(data)

        print(f"Number of chunks: {len(texts)}")
        print(f"First chunk: {texts[0].page_content}")

        main_info.write("Creating vector index...")
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local("vectorstore")

        main_info.write("Vector index created.")

query = main_info.text_input("Enter your question")

if query:
    if not os.path.exists("vectorstore"):
        st.write("No vector index found. Please process URLs first.")
    else:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
        chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)
        # {'answer': '', 'sources': []}
        st.subheader("Answer")
        st.write(result["answer"])

        sources = result.get('sources', '')
        if sources:
            st.subheader("Sources")
            source_list = sources.split("\n")
            for source in source_list:
                st.write(source)
        
        print(result)