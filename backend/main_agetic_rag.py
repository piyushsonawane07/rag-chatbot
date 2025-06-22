#==================CODE USING AGENT BASED RAG AND TOOLS================
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from dotenv import load_dotenv
from typing import List, Dict
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_chroma import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_functions_agent, AgentExecutor, tool
from langchain.schema.messages import SystemMessage
import redis
import os
import json
import uuid

# Load env vars
load_dotenv()

# === FastAPI Setup ===
app = FastAPI()
app.mount("/chat", StaticFiles(directory="static", html=True), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Models ===
class ChatRequest(BaseModel):
    query: str
    session_id: str

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, str]]

class TicketRequest(BaseModel):
    query: str

class TicketResponse(BaseModel):
    ticket_id: str
    message: str

# === Redis Handler ===
class RedisStore:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=0,
            decode_responses=True
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
        self.redis_client.flushdb()
        return True

redis_store = RedisStore()

# === Embeddings and Vector Stores ===
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

website_vs = Chroma(
    collection_name="website_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_website_db",
)

pdf_vs = Chroma(
    collection_name="pdf_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_pdf_db",
)

def create_custom_retriever_tool(retriever, name, description, type_label):
    from langchain.tools import Tool

    def _retriever_tool_fn(input_query: str):
        docs = retriever.invoke(input_query)
        return json.dumps([
            {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "type": type_label
            }
            for doc in docs
        ])

    return Tool(
        name=name,
        description=description,
        func=_retriever_tool_fn
    )


# === Tools ===
website_tool = create_custom_retriever_tool(
    website_vs.as_retriever(search_kwargs={"k": 5}),
    "retrieve_website_docs",
    "Use this tool for investment/trading related queries."
)

pdf_tool = create_custom_retriever_tool(
    pdf_vs.as_retriever(search_kwargs={"k": 5}),
    "retrieve_pdf_docs",
    "Use this tool for insurance related queries."
)

@tool
def raise_ticket_tool(query: str) -> str:
    """Raise a support ticket when the query can't be resolved."""
    ticket_id = str(uuid.uuid4())[:8]
    import asyncio
    asyncio.run(send_support_email(ticket_id, "Support Request", query, "piyushsonawane1140@gmail.com"))
    return f"Support ticket #{ticket_id} has been raised and sent to our team."

# === Prompt & Agent Setup ===
prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content=(
        "You are a customer support agent. Your job is to answer questions using the tools provided. "
        "For investment or trading-related queries, use the website retriever. "
        "For insurance-related queries, use the PDF retriever. "
        "If the question cannot be answered from the data, raise a support ticket.\n"
        "Be concise and helpful."
    )),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

llm = ChatOpenAI(model="gpt-4", temperature=0)
agent = create_openai_functions_agent(llm=llm, tools=[website_tool, pdf_tool, raise_ticket_tool], prompt=prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=[website_tool, pdf_tool, raise_ticket_tool])


# === Chat Endpoint ===
@app.post("/api/ask")
async def chat(request: ChatRequest):
    try:
        session_id = request.session_id
        query = request.query
        history = redis_store.get_chat_history_from_redis(session_id)
        chat_history_formatted = "\n".join(
            [f"User: {h['user']}\nAssistant: {h['assistant']}" for h in history]
        ) if history else ""

        result = await agent_executor.ainvoke({"input": query, "chat_history": chat_history_formatted})
        answer = result.get("output", "I'm not sure how to help with that.")

        redis_store.save_chat_to_redis(session_id, query, answer)

        return {
            "response": answer,
            "chat_history": history
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# === Send Email ===
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
            print("Warning: Email configuration is incomplete.")
            return False

        email_body = f"""
        <p>A new support ticket has been raised.</p>
        <p><b>Ticket ID:</b> {ticket_id}</p>
        <p><b>From:</b> {user_email}</p>
        <p><b>Message:</b><br>{message}</p>
        """

        msg = MessageSchema(
            subject=subject,
            recipients=[support_email],
            body=email_body,
            subtype="html"
        )

        fastmail = FastMail(conf)
        await fastmail.send_message(msg)
        return True
    except Exception as e:
        print(f"Error sending support email: {str(e)}")
        return False

# === Redis Management ===
@app.post("/api/clear-redis")
async def clear_redis():
    try:
        redis_store.clear_redis()
        return {"status": "success", "message": "Redis data cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing Redis: {str(e)}")

# === Manual Ticket Raising (Optional Endpoint) ===
@app.post("/api/raise-ticket", response_model=TicketResponse)
async def raise_ticket(request: TicketRequest):
    try:
        ticket_id = str(uuid.uuid4())[:8]
        subject = "Support Request"
        user_email = "piyushsonawane1140@gmail.com"

        email_sent = await send_support_email(ticket_id, subject, request.query, user_email)

        if email_sent:
            return TicketResponse(ticket_id=ticket_id, message="Support ticket has been raised successfully.")
        else:
            return TicketResponse(ticket_id=ticket_id, message="Support ticket raised but email sending failed.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error raising support ticket: {str(e)}")


