//from this lines, mounting a drive folder from where some files will be used
from google.colab import drive
drive.mount("/content/drive")

//important libraries
!pip install langchain sentence-transformers chromadb llama-cpp-python langchain_community pypdf

//importing modules
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA,LLMChain

//here the file used is a medical research report about heart in pdf format
loader = PyPDFDirectoryLoader("link address for your file in drive")
docs = loader.load();

//split the text retrieved from pdf into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size =300,chunk_overlap= 50)
chunks = text_splitter.split_documents(docs)

//Creating a environment for hugging face from where our llm model is used
import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "UR_huggingface_key"

//Creating embeddings from the chunks
embeddings = SentenceTransformerEmbeddings(model_name ="NeuML/pubmedbert-base-embeddings")

//Creating a VectorStore to hold the embeddings
vectorstore = Chroma.from_documents(chunks,embeddings)

//here query can hold any questions u want to ask
query = "Who is at risk of Heart Disease?"
//all the related things from the vector is searched which seems related to the query
search_results = vectorstore.similarity_search(query)
search_results

//here 5 represents the number of responses I want from vector
retriever = vectorstore.as_retriever(search_kwargs={'k':5})
retriever.get_relevant_documents(query)

//Defining the llm, 
llm = LlamaCpp(
    model_path="path link for the llm model",
    temperature=0.2,
    max_tokens = 2048,
    top_p=1
)

//template holds the template of the prompt which will be sent to the llm
template = """
<|context|>
You are an Medical Assistant that follows the instructions and generate the accurate response based on the query and the context provided.
Please be truthful and give direct answers.
</s>
<|user|>
{query}
</s>
<|assistant|>"""

from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context":retriever,"query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

//response holds the answer
response = rag_chain.invoke(query)
//this statement gives the output
response

//Creating a continue function which takes input from the user
import sys
while True:
  user_input = input(f"Input Query:")
  if user_input =="exit":
    print("Exiting....")
    sys.exit()
  if user_input =="":
    continue
  result = rag_chain.invoke(user_input)
  print ("Answer: ",result)




