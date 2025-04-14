#!pip install Langchain
import openai 
inport os
import sys
openai.api_key=' '

#! pip install pypdf
################################document loading ###################################################
#pip install -U langchain-community
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader('Business Plan ITHG AI EN for Daia.pdf')
pages = loader.load()
page=pages[1]
print(page.page_content)
print(page.metadata)

############################# document spliting ####################################################
#pip install tiktoken
!export OPENAI_API_KEY=" "
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
)
docs = text_splitter.split_documents(pages)

from langchain.text_splitter import TokenTextSplitter



docs_ = text_splitter.split_documents(pages)


#############################vectors and embeddings ############################################


# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)
#docs=[pages]
splits = text_splitter.split_documents(pages)

#Embeddings 

from langchain.embeddings.openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()
sentence1 = "i like dogs"
sentence2 = "i like canines"
sentence3 = "the weather is ugly outside"
embedding1 = embedding.embed_query(sentence1)
embedding2 = embedding.embed_query(sentence2)
embedding3 = embedding.embed_query(sentence3)
import numpy as np 
np.dot(embedding1, embedding2)
np.dot(embedding1, embedding3)
###########################################################Vectorstores ##################################################
# ! pip install chromadb
from langchain.vectorstores import Chroma
persist_directory = 'docs/chroma/'
!rm -rf ./docs/chroma  # remove old database files if any
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)




print(vectordb._collection.count())

question="Is there a strategy in the business plan ?"

docs = vectordb.similarity_search(question,k=3)
docs[0].page_content
vectordb.persist()

from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The lecture the chunk is from, should be one of `docs/cs229_lectures/MachineLearning-Lecture01.pdf`, `docs/cs229_lectures/MachineLearning-Lecture02.pdf`, or `docs/cs229_lectures/MachineLearning-Lecture03.pdf`",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="The page from the lecture",
        type="integer",
    ),
]


