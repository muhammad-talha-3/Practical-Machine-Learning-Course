# pip install langchain openai unstructured chromadb tiktoken transformers pdf2image flask flask_cors
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader
import nltk
import tempfile
import requests

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True)


def loadPDFFromURL(pdf_file_url):

  # Send a GET request to the URL and retrieve the contents of the PDF file
  response = requests.get(pdf_file_url)

  # Create a temporary file and save the PDF content to it
  with tempfile.NamedTemporaryFile(delete=False) as temp_file:
      temp_file.write(response.content)

      # Open the temporary PDF file with PdfReader
      with open(temp_file.name, 'rb') as pdf_file:
          # reader = PdfReader(pdf_file)
          loader = UnstructuredPDFLoader(temp_file.name)
          loaded_docs = loader.load()
          # Delete the temporary file after use
          temp_file.close()

          return loaded_docs

loaded_docs = loadPDFFromURL("https://firebasestorage.googleapis.com/v0/b/medicalgpt-7baa4.appspot.com/o/products.pdf?alt=media&token=3cee6f7a-7827-481f-ba99-fcc88fd17077")
# print(loaded_docs)

nltk.download('punkt')

os.environ['OPENAI_API_KEY'] = 'write-your-chatgpt-api-key'

embeddings = OpenAIEmbeddings(openai_api_key = os.environ['OPENAI_API_KEY'])

# convert data into chunks
text_splitter = CharacterTextSplitter(chunk_size = 800, chunk_overlap = 0)
texts = text_splitter.split_documents(loaded_docs)

# Store data into vector store
vector_store = Chroma.from_documents(texts, embeddings)
chain = VectorDBQA.from_chain_type(llm = OpenAI(), chain_type = "stuff", vectorstore = vector_store)


@app.route('/chat', methods = ['POST', 'GET'])
def chat():
   try:
      data = request.get_json()
      user_message = data.get('message', '')
      
      res = chain.run(user_message)
      print(res)

      return jsonify({'response': res})
   except Exception as e:
      print(e)
      return e

if __name__ == '__main__':
   app.run(port = '8000', debug = True)
