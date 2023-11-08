import os
from flask import request, jsonify, abort
import base64
# from server.vectorstores import SupabaseVectorStore
from server.vectorstores import PGVector
import json
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import traceback




class Backend:
    def __init__(self, app, CONNECTION_STRING, _calculate_embeddings) -> None:
        self.app = app
        self.CONNECTION_STRING = CONNECTION_STRING
        self._calculate_embeddings = _calculate_embeddings

        self.routes = {
            '/upload': {
                'function': self._upload,
                'methods': ['GET', 'POST']
            },
            '/chat': {
                'function' : self._chat,
                'methods' : ['GET', 'POST']
            },
            '/delete_category': {
                'function' : self._delete_category,
                'methods' : ['GET', 'POST']
            },
            '/delete_file': {
                'function' : self._delete_file,
                'methods' : ['GET', 'POST']
            }
        }
    def _check_api_key(self):
        api_key = request.headers.get('X-API-Key')
        
        if not api_key or api_key != 'supersecure12345!!!': #os.getenv('API_KEY'):  
            abort(401, 'Invalid API Key')     

    def _upload(self):
        self._check_api_key()
        try:
            # structure for json
            # {
            #     "file_name": "example.pdf",
            #     "file_content": "base64 encoded string"
            #     "category_index": "index of the file
            # }
            # get the request from the client json format
            data = request.get_json()

            # get the file name and content from the data
            file_name = data['file_name']
            # update the filename and add category index to it
            file_name = file_name.split('.')[0] + '_' + data['category_index'] + '.' + file_name.split('.')[1]

            file_content = data['file_content']
            # convert string64 to bytes
            file_content = base64.b64decode(file_content)


            # get the bucket name from the environment variable
            # bucket_name = os.environ.get('BUCKET_NAME')

            # upload the file to the bucket
            # supabase.storage.from_('bucket').upload(destination, f)
            # try:
                # response = self.supabase.storage.from_(bucket_name).upload(file_name, file_content)
            # except Exception as e:
                # pass
                # print("exception occured 1", e)
            

            # calculate embeddings in celery task
            # self.celery.send_task('tasks._calculate_embeddings', args=[file_name, file_content])
            try:
                self._calculate_embeddings.delay(file_name, file_content)
            except Exception as e:
                # pass
                print("exception occured 2", e)
                # return jsonify({'message': 'File upload failed', 'error' : str(e), 'success' : False}), 400

            # check if response is successful
            # if response['status_code'] == 201:
            return jsonify({'message': 'File uploaded successfully', 'error' : '' , 'success' : True}), 201

            # return jsonify({'message': 'File upload failed', 'error' : 'File upload failed', 'success' : False}), 400
        except Exception as e:
            # print("exception occured", e)
            return jsonify({'message': 'File upload failed', 'error' : str(e), 'success' : False}), 400


    
    def _chat(self):
        self._check_api_key()
        try:
            """ 
            json structure
             {
               "query": "query string",
               "category_index": "index of the file"
             } 
            
            """

            # get the request from the client json format
            data = request.get_json()
            query = str(data['query'])
            category_index = int(data['category_index'])
            embeddings = OpenAIEmbeddings()
            # vectorstore = SupabaseVectorStore(client=self.supabase, table_name='documents' ,embedding=embeddings, query_category_index=category_index)
            vectorstore = PGVector(self.CONNECTION_STRING, embeddings)
            retriever = vectorstore.as_retriever()

            # get the top 5 results
            llm = OpenAI()
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
            # query = query
            result = qa({"query": query})
            print(result)

            try:
                # get public url from the supabase storage for file name result['source_documents'][0].metadata['filename']
                expiration_time = 60

                # Generate a signed URL with the expiration time
                signed_url = self.supabase.storage.from_(os.environ.get('BUCKET_NAME')).create_signed_url(result['source_documents'][0].metadata['filename'], expiration_time)
                # print(signed_url)
                return jsonify({'message': 'successfully chatting', 'url': signed_url , "result" : result['result'], 'source' : result['source_documents'][0].metadata['filename'] ,'error' : '' , 'success' : True}), 200
            except Exception as E:
                print(traceback.print_exc())
                print("exception occured", E)
                return jsonify({'message': 'successfully chatting', "result" : result['result'], 'source' : '' ,'error' : 'no source document found' , 'success' : True}), 200
        
        except Exception as e:
            # print traceback
            print(traceback.print_exc())

            print(e)
            return jsonify({'message': 'failed to chat' ,'error' : str(e), 'success' : False}), 400


    def _delete_category(self):
        self._check_api_key()
        try:
            """ 
            json structure
             {
               "category_index": "index of the file"
             } 
            
            """ 
            # get the request from the client json format
            data = request.get_json()
            category_index = int(data['category_index'])

            # get the metadata from the supabase for the row where categort_index is equal to the category_index
            files_for_delete = []
            metadata = self.supabase.from_('documents').select('metadata').eq('category_index', category_index).execute()
            metadata = metadata.data
            for data in metadata: 
                files_for_delete.append(data['metadata']['filename'])

            # delete the files from the bucket
            # remove repeation from files_for_delete
            files_for_delete = list(set(files_for_delete))
            # print(files_for_delete)
            # print(files_for_delete)

            # first delete category index from the supabase
            try:
                self.supabase.from_('documents').delete().eq('category_index', category_index).execute()
            except:
                pass

            try:
                for file in files_for_delete:
                    try:
                        self.supabase.storage.from_(os.environ.get('BUCKET_NAME')).remove(file)
                    except Exception as e:
                        pass
            except:
                pass

            return jsonify({'message': 'successfully deleted', 'error' : '' , 'success' : True}), 200
        
        except Exception as e:
            
            return jsonify({'message': 'failed to delete', 'error' : str(e), 'success' : False}), 400


    def _delete_file(self):
        self._check_api_key()
        try:
            """ 
            json structure
             {
               "file_name": "name of the file"
             } 
            
            """ 
            # get the request from the client json format
            data = request.get_json()
            filename = str(data['file_name'])
            # print(filename)
            response = self.supabase.from_('documents').select('*').execute()
            response = response.data
            
            matching_row_ids = []
            if response:
                for data in response:
                    if data['metadata']['filename'] == filename:
                        matching_row_ids.append(data['id'])
                    

                # print("macthing row ids", matching_row_ids)

                if matching_row_ids:
                    # Step 4: Use the extracted row IDs to delete the rows from the table
                    self.supabase.from_('documents').delete().in_('id', matching_row_ids).execute()
                    # print(f"Successfully deleted {len(matching_row_ids)} rows with filename '{filename}'")
                    self.supabase.storage.from_(os.environ.get('BUCKET_NAME')).remove(filename)
                else:
                    pass
                    # print(f"No rows found with filename '{filename}'")
            else:
                pass
                # print("Table is empty.")
                   

            return jsonify({'message': 'successfully deleted', 'error' : '' , 'success' : True}), 200
        
        except Exception as e:
            
            return jsonify({'message': 'failed to delete', 'error' : str(e), 'success' : False}), 400
