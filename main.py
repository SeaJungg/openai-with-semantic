import os
import requests
import json
import asyncio
import re
import logging
import sys
from collections import deque
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import pinecone
import tiktoken
from functools import wraps

class TextRequest(BaseModel):
    prompt : str
    top_k : Optional[int]
    model : Optional[str]

class QuestionRequest(BaseModel):
    assistant_prompt : Optional[str]
    prompt : str
    user_openai_key : Optional[str]
    based_on_semantic_search : Optional[bool]
    top_k : Optional[int]
    model : Optional[str]


load_dotenv('.env')

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV", "")
INDEX_NAME = 'question-with-title'

UPLOAD_FOLDER = 'workspace/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

APP_PORT = int(os.environ.get("WEB_PORT", 8000))
KERNEL_APP_PORT = APP_PORT

def initialize_pinecone():
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

initialize_pinecone()
...

async def get_embedding_openai(user_prompt, model="gpt-4"): 
    if model == "gpt-4" or model == "gpt-3.5":
        model = "text-embedding-ada-002"
    else:
        raise ValueError('Invalid model value')

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }

    data = {
        "model": model,
        "input": user_prompt
    }

    response = requests.post(
        f"{OPENAI_BASE_URL}/v1/embeddings",
        json=data,
        headers=headers,
    )
    return response.json()['data'][0]['embedding']

async def get_semantic_search_metadata(user_prompt, index_name, model="gpt-4", top_k=3):
    user_propmpt_embeddings = await get_embedding_openai(user_prompt, model=model)
    if index_name not in pinecone.list_indexes():
        return JSONResponse({'error': 'Invalid index_name'})
    index = pinecone.Index(index_name)
    response = index.query(user_propmpt_embeddings, top_k=top_k, include_metadata=True)

    meta_data = []
    for item in response['matches']:
        metadata = item['metadata']
        score = item['score']
        metadata['score'] = score
        meta_data.append(metadata)

    return meta_data

def get_first_n_chunks(text, max_tokens):
    # Load the cl100k_base tokenizer which is designed to work with the ada-002 model
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Split the text into sentences
    sentences = text.split('. ')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    
    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater 
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of 
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks[0]

async def get_code(assistant_prompt, user_prompt, model="gpt-4", meta_data=None):
    if meta_data:
        print(meta_data)
        meta_data = meta_data[:3]
        concatenated_answer = '\n...'.join(item["답변"][:300] for item in meta_data) + '\n...'
        assistant_prompt = assistant_prompt + f"And You should answer based on the answers to a similar question that has occurred in the past.\nSimillar answers:\n{concatenated_answer}"
    prompt = f"{assistant_prompt}\n\nQuestion is: {user_prompt}"

    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.7,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }

    try:
        response = requests.post(
            f"{OPENAI_BASE_URL}/v1/chat/completions",
            data=json.dumps(data),
            headers=headers,
        )
        response_json = response.json()
        content = response_json["choices"][0]["message"]["content"]
        status_code = 200
    except Exception as e:
        content = str(e)
        status_code = 500

    return content, status_code


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/embedding-search')
async def get_embedding_search(request:TextRequest):
    top_k = request.top_k or 5
    model = request.model or 'gpt-4'
    meta_data = await get_semantic_search_metadata(user_prompt=request.prompt, index_name=INDEX_NAME, model=model, top_k=top_k)
    return {'meta_data' : meta_data}

@app.post('/generate')
async def generate_code(request: QuestionRequest):
    assistant_prompt = request.assistant_prompt or '당신은 대한민국 법을 학습한 유능한 AI이다'
    user_prompt = request.prompt or ''
    user_openai_key = OPENAI_API_KEY
    based_on_semantic_search = request.based_on_semantic_search or True
    top_k = request.top_k or 3
    model = request.model or 'gpt-4'

    print(assistant_prompt, user_prompt, user_openai_key, based_on_semantic_search, top_k, model)
    meta_data = await get_semantic_search_metadata(user_prompt=user_prompt, index_name=INDEX_NAME, model=model, top_k=top_k)
    code_list = []

    print(meta_data)
    # Call get_code function with different arguments and append the results to code_list

    '''
    # 질문에 대한 gpt4 답변
    code, status = await get_code(assistant_prompt=None, user_prompt=user_prompt, model=model)
    if status == 200:
        code_list.append("질문에 대한 gpt4 답변\n---\n" + code)

    # 질문+PE에 대한 gpt4 답변
    code, status = await get_code(assistant_prompt=assistant_prompt, user_prompt=user_prompt, model=model)
    if status == 200:
        code_list.append("질문+PE에 대한 gpt4 답변\n---\n" + code)
    '''

    # 질문+PE+써치결과에 대한 gpt4답변
    if based_on_semantic_search:
        code, status = await get_code(assistant_prompt=assistant_prompt, user_prompt=user_prompt, model=model, meta_data=meta_data)
        if status == 200:
            code_list.append("질문+PE+써치결과에 대한 gpt4답변\n---\n" + code)

    #code_list.append(code)

    # Join the code_list with newline characters to aggregate the code
    aggregated_code = '\n\n'.join(code_list)

    # Append all messages to the message buffer for later use
    print(aggregated_code)
    return {'code': aggregated_code, 'meta_data' : meta_data, 'index' : INDEX_NAME}
