from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader

from server import supabase
from server.vectorstores import SupabaseVectorStore
import os
import io
import random
import traceback



def celery_functions(celery):
    @celery.task(name='_calculate_embeddings')
    def _calculate_embeddings(file_name, file_content):
        random_int = str(random.randint(0, 1000000) + random.randint(0, 1000000))
        try:
            # check if temp folder exists
            if not os.path.exists('./temp'):
                os.makedirs('./temp')

            with open(f"./temp/{random_int}.pdf", "wb") as f:
                f.write(file_content)
            

            loader = PyPDFLoader(f"./temp/{random_int}.pdf")
            documents = loader.load_and_split()

            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            documents = text_splitter.split_documents(documents)


            # get category index from file name
            category_index = file_name.split('.')[0].split('_')[-1]

            embeddings = OpenAIEmbeddings()

            vector_store = SupabaseVectorStore.from_documents(documents, embeddings, client=supabase, category_index=category_index, metadata_addition={"filename" : file_name})

            os.remove(f"./temp/{random_int}.pdf")


        
        except Exception as e:
            # write the error to a file
            with open(f"./temp/{random_int}.txt", "w") as f:
                f.write(str(traceback.format_exc()))
            try:
                os.remove(f"./temp/{random_int}.pdf")
            except Exception as e:
                pass

    return _calculate_embeddings