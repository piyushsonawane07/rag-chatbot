from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from typing import List, Dict
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import os
import json
import redis
from vector_store.vector import store_documents
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from fastapi_mail import FastMail, MessageSchema,ConnectionConfig
import uuid
from fastapi.staticfiles import StaticFiles

load_dotenv()

app = FastAPI()
app.mount("/chat", StaticFiles(directory="static", html=True), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TicketRequest(BaseModel):
    query: str

class TicketResponse(BaseModel):
    ticket_id: str
    message: str

class ChatRequest(BaseModel):
    query: str
    session_id: str

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, str]]

class RedisStore:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=0,
            decode_responses=True  # returns strings instead of bytes
        )

    def get_chat_history_from_redis(self, session_id: str):
        key = f"chat_history:{session_id}"
        history_json = self.redis_client.get(key)
        return json.loads(history_json) if history_json else []

    def save_chat_to_redis(self, session_id: str, user_msg: str, bot_msg: str):
        key = f"chat_history:{session_id}"
        history = self.get_chat_history_from_redis(session_id)
        history.append({"user": user_msg, "assistant": bot_msg})
        self.redis_client.set(key, json.dumps(history))
        
    def clear_redis(self):
        """Clear all data from Redis"""
        self.redis_client.flushdb()
        return True

redis_store = RedisStore()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

@app.post("/api/store")
def store_documents():
    store_documents()
    return {"message": "Documents stored successfully"}


@app.post("/api/ask")
async def chat(request: ChatRequest):
    try:
        session_id = request.session_id
        query = request.query

        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        retrieved_docs = retriever.invoke(query)
        
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Load previous chat history from Redis
        history = redis_store.get_chat_history_from_redis(session_id)
        if history:
            chat_history_formatted = "\n".join(
                [f"User: {h['user']}\nAssistant: {h['assistant']}" for h in history]
            )
        else:
            chat_history_formatted = ""

        # prompt_template = """
        # You are a helpful assistant that provides accurate answers based on the given context and prior conversation.\
        # - If user greets you, reply with a greeting message.
        # - If the answer is not in the context, say that I don't know.


        # Context:
        # {context}

        # Chat History:
        # {chat_history}

        # Current Question: {question}
        # """
        prompt_template = """
            You are a customer support assistant your goal is to provide accurate and helpful answers to the user based on 
            the given context and prior conversation which will help user to resolve their query. 
            Use proper and simple language to answer the user.
            Use these guidelines:
            1. For Investment related queries or any stock/trading related queries: Use WEBSITE content
            2. For any Insurance related queries: Use PDF content
            3. Greet appropriately for greetings
            4. Say "I don't know" for unanswerable queries
            5. Don't use Assistant/Answer words in the response

            **Sources:**
            {context}

            **Chat History:**
            {chat_history}

            **Question:** {question}
        """

        prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=prompt_template,
        )

        llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

        chain = (
            {
                "context": lambda _: context,
                "chat_history": lambda _: chat_history_formatted,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        answer = chain.invoke(query)

        # Collect source metadata if answer is based on retrieved docs
        if "I don't know" in answer or "Hi" in answer or "Hello" in answer:
            sources = []
        else:   
            sources = [
                {
                    "source": doc.metadata.get("source", "Unknown"),
                }
                for doc in retrieved_docs
            ]

        # Save interaction to Redis
        redis_store.save_chat_to_redis(session_id, query, answer)

        return {
            "response": answer,
            "sources": sources,
            "chat_history": history
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


async def send_support_email(ticket_id: str, subject: str, message: str, user_email: str):
    try:
        conf = ConnectionConfig(
            MAIL_USERNAME=os.getenv("SMTP_USER"),
            MAIL_PASSWORD=os.getenv("SMTP_PASSWORD"),
            MAIL_PORT=int(os.getenv("SMTP_PORT", 587)),
            MAIL_SERVER=os.getenv("SMTP_HOST"),
            MAIL_FROM=os.getenv("SMTP_USER"),
            MAIL_FROM_NAME="Support System",
            MAIL_STARTTLS=True,
            MAIL_SSL_TLS=False,
            USE_CREDENTIALS=True,
            VALIDATE_CERTS=True
        )
        support_email = os.getenv("SUPPORT_EMAIL")

        if not all([support_email]):
            print("Warning: Email configuration is incomplete. Email will not be sent.")
            return False

        email_body = f"""
        A new support ticket has been raised.

        Ticket ID: {ticket_id}
        From: {user_email}
        
        Message:
        {message}
        """

        message = MessageSchema(
            subject=subject,
            recipients=[support_email],
            body=email_body,
            subtype="html"
        )
        
        fastmail = FastMail(conf)
        await fastmail.send_message(message)
        
        print(f"Support email sent for ticket #{ticket_id}")
        return True
    except Exception as e:
        print(f"Error sending support email: {str(e)}")
        return False


@app.post("/api/clear-redis")
async def clear_redis():
    """Clear all data from Redis"""
    try:
        redis_store.clear_redis()
        return {"status": "success", "message": "Redis data cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing Redis: {str(e)}")

@app.post("/api/raise-ticket", response_model=TicketResponse)
async def raise_ticket(request: TicketRequest):
    try:
        ticket_id = str(uuid.uuid4())[:8]  # Generate a short ticket ID
        subject = "Support Request"
        user_email = "piyushsonawane1140@gmail.com"  # You might want to get this from the request or session

        # Send email
        email_sent = await send_support_email(ticket_id, subject, request.query, user_email)
        
        if email_sent:
            return TicketResponse(
                ticket_id=ticket_id,
                message="Support ticket has been raised successfully."
            )
        else:
            return TicketResponse(
                ticket_id=ticket_id,
                message="Support ticket was created but we encountered an issue sending the notification email."
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error raising support ticket: {str(e)}")


