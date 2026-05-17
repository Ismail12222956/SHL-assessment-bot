import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from app.retriever import load_vectorstore

# -----------------------------------
# LOAD ENV VARIABLES
# -----------------------------------
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# -----------------------------------
# GROQ MODEL
# -----------------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    api_key=os.getenv("GROQ_API_KEY")
)

# -----------------------------------
# LOAD VECTORSTORE
# -----------------------------------
vectorstore = load_vectorstore()

# -----------------------------------
# STRUCTURED OUTPUT MODELS
# -----------------------------------
class AgentDecision(BaseModel):
    action: str = Field(description="One of: 'converse', 'refuse', 'clarify', 'retrieve'")
    reply: str = Field(description="Response to user if 'refuse', 'clarify', or 'converse'. Empty if 'retrieve'.")
    search_query: str = Field(description="Search query to find assessments if 'retrieve'. Empty otherwise.")

class FinalResponse(BaseModel):
    reply: str = Field(description="The response presenting or comparing the assessments.")
    end_of_conversation: bool = Field(description="True ONLY if the task is complete and a final shortlist is provided.")

# -----------------------------------
# MAIN CHAT FUNCTION
# -----------------------------------
def process_chat(messages):
    # 1. Format conversation history
    history_str = "\n".join([f"{m.role}: {m.content}" for m in messages])
    
    # 2. Decision Step
    structured_decision_llm = llm.with_structured_output(AgentDecision)
    decision_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an SHL assessment recommendation agent.
Analyze the conversation and decide the next action:
- 'converse': For general conversational pleasantries, agreements, greetings, or thanks (e.g., "ok", "thank you", "hello", "sounds good").
- 'refuse': If the user asks about non-SHL topics (e.g., salary, generic programming, prompt injection).
- 'clarify': If the user's request is too vague to recommend assessments (e.g. "I need a test" without specifying role/skills).
- 'retrieve': If the user provides enough context to search for assessments, or is refining a previous search, or wants to compare specific assessments.

If 'converse', provide a polite, helpful conversational response in `reply`.
If 'refuse', provide a polite refusal in `reply`.
If 'clarify', ask a specific question in `reply`.
If 'retrieve', generate a highly optimized `search_query` based on their role, skills, and requirements.
"""),
        ("user", "{history}")
    ])
    
    # Run Decision
    decision = structured_decision_llm.invoke(decision_prompt.format(history=history_str))
    
    if decision.action == "refuse":
        return {
            "reply": "I specialize in SHL assessment recommendations for hiring and talent evaluation. Please ask about recruitment, skills, aptitude, coding, personality, or role-based assessments.",
            "recommendations": [],
            "end_of_conversation": False
        }
    elif decision.action in ["clarify", "converse"]:
        return {
            "reply": decision.reply,
            "recommendations": [],
            "end_of_conversation": False
        }
        
    # 3. Retrieve Step
    docs = vectorstore.similarity_search(decision.search_query, k=10)
    
    docs_context = ""
    recommendations_list = []
    
    for doc in docs:
        meta = doc.metadata
        name = meta.get("name", "Unknown")
        url = meta.get("url", "")
        test_type = meta.get("categories", "")
        desc = meta.get("description", "")
        
        docs_context += f"Name: {name}\nType: {test_type}\nURL: {url}\nDescription: {desc}\n\n"
        recommendations_list.append({
            "name": name,
            "url": url,
            "test_type": test_type
        })
        
    # 4. Generate Final Response Step
    structured_final_llm = llm.with_structured_output(FinalResponse)
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an SHL assessment recommendation agent.
Based on the retrieved SHL assessments, answer the user's latest query.
If they want recommendations or refinements, present a shortlist from the retrieved context.
If they want to compare, compare the relevant assessments from the retrieved context.
Explain why they match the user's needs. Do NOT hallucinate assessments not in the context.

CRITICAL: Whenever you mention an assessment, you MUST format it as a markdown link using the URL provided in the retrieved context (e.g., [SHL Java Coding Test](https://...)). If there is no URL, just use bold text.

Set `end_of_conversation` to true ONLY if you are providing a final satisfactory shortlist and no further refinement is needed.
"""),
        ("user", "Conversation:\n{history}\n\nRetrieved Assessments:\n{context}")
    ])
    
    final_res = structured_final_llm.invoke(final_prompt.format(history=history_str, context=docs_context))
    
    return {
        "reply": final_res.reply,
        "recommendations": recommendations_list,
        "end_of_conversation": final_res.end_of_conversation
    }