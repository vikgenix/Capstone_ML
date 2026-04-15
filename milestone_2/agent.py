"""
agent.py - LangGraph Agentic Workflow for Diabetes Health Support Assistant
Implements a stateful agent that:
  1. Retrieves relevant medical guidelines from ChromaDB
  2. Generates a structured health report using Groq LLM
"""
import os
from typing import TypedDict, Optional, List
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# === Configuration ===
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH = os.path.join(_BASE_DIR, "chroma_db")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"

# === Agent State Definition ===
class HealthAgentState(TypedDict):
    # Patient context
    patient_summary: str              # Free-text summary of patient profile
    risk_score: float                 # ML model risk score (0-100)
    prediction_label: str             # "Diabetic" or "Non-Diabetic"
    prediction_confidence: float      # Confidence % of ML model

    # Retrieved knowledge
    retrieved_guidelines: List[str]   # Chunks from vector DB
    retrieval_error: Optional[str]    # Error message if retrieval fails

    # Final output
    health_report: Optional[str]      # Final structured markdown report
    generation_error: Optional[str]   # Error message if generation fails


# === Singleton loaders (avoid reloading on every call) ===
_vectorstore = None
_llm = None

def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        _vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings
        )
    return _vectorstore

def _get_llm(api_key: str):
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model=LLM_MODEL,
            temperature=0.3,
            groq_api_key=api_key
        )
    return _llm


# === Node 1: Retrieve Medical Guidelines ===
def retrieve_guidelines(state: HealthAgentState) -> HealthAgentState:
    """
    Searches the ChromaDB vector store for guidelines relevant to the patient's
    risk profile. Handles failures gracefully by setting retrieval_error.
    """
    try:
        vectorstore = _get_vectorstore()
        query = (
            f"Diabetes management guidelines for a patient with "
            f"risk score {state['risk_score']:.1f} out of 100, "
            f"predicted as {state['prediction_label']}. "
            f"Patient profile: {state['patient_summary']}"
        )
        results = vectorstore.similarity_search(query, k=5)
        chunks = [doc.page_content for doc in results]
        return {**state, "retrieved_guidelines": chunks, "retrieval_error": None}
    except Exception as e:
        return {
            **state,
            "retrieved_guidelines": [],
            "retrieval_error": f"Retrieval failed: {str(e)}"
        }


# === Node 2: Generate Structured Health Report ===
def generate_report(state: HealthAgentState) -> HealthAgentState:
    """
    Uses the Groq LLM to reason over patient data and retrieved guidelines,
    producing a structured health report with hallucination-reducing prompting.
    """
    try:
        api_key = GROQ_API_KEY
        llm = _get_llm(api_key)

        # Format retrieved guidelines or use fallback
        if state["retrieved_guidelines"]:
            guidelines_text = "\n\n---\n\n".join(state["retrieved_guidelines"])
        else:
            guidelines_text = "No specific guidelines retrieved. Use general diabetes prevention and management knowledge."

        # Structured prompt to minimize hallucinations
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a knowledgeable and empathetic AI health support assistant.
Your role is to provide structured, evidence-based health guidance based on a patient's risk assessment.
CRITICAL RULES:
- Only make claims that are supported by the provided guidelines or standard medical consensus.
- NEVER invent statistics, drug names, or clinical claims not grounded in the context.
- Always recommend consulting a healthcare professional for personal medical decisions.
- Be compassionate, clear, and avoid overly technical jargon.
- Structure your response EXACTLY as instructed."""),
            ("human", """
## Patient Risk Assessment
- **ML Risk Score**: {risk_score:.1f} / 100
- **Prediction**: {prediction_label} (Confidence: {confidence:.1f}%)
- **Patient Profile**: {patient_summary}

## Retrieved Medical Guidelines (Use as grounding)
{guidelines_text}

---

Please generate a **Structured Health Report** in markdown format with EXACTLY these sections:

### 🧬 Risk Summary
Briefly summarize the patient's risk profile and key contributing factors (2-3 sentences).

### ⚠️ Key Risk Factors Identified
A bullet-point list of the main risk factors present for this patient, based on their profile.

### 💡 Personalized Recommendations
Provide 4-6 concrete, actionable preventive care recommendations tailored to this patient. Base these on the retrieved guidelines.

### 📅 Follow-Up Actions
List 3-4 specific follow-up steps the patient should take (e.g., tests, screenings, specialist visits).

### 📚 Sources
List the medical guideline sources referenced (from the retrieved context).

### ⚕️ Medical Disclaimer
> Include a mandatory disclaimer that this report is generated by an AI for informational purposes only and does not constitute medical advice. Always consult a licensed healthcare professional.
""")
        ])

        chain = prompt | llm
        response = chain.invoke({
            "risk_score": state["risk_score"],
            "prediction_label": state["prediction_label"],
            "confidence": state["prediction_confidence"],
            "patient_summary": state["patient_summary"],
            "guidelines_text": guidelines_text,
        })

        return {**state, "health_report": response.content, "generation_error": None}

    except Exception as e:
        fallback = (
            f"⚠️ **Agent Error**: The report could not be generated. Details: `{str(e)}`\n\n"
            f"**Risk Score**: {state['risk_score']:.1f}/100 | **Prediction**: {state['prediction_label']}\n\n"
            f"> Please consult a healthcare professional for guidance."
        )
        return {**state, "health_report": fallback, "generation_error": str(e)}


# === Build and Compile the LangGraph Workflow ===
def build_agent():
    """Builds and returns the compiled LangGraph agent workflow."""
    workflow = StateGraph(HealthAgentState)

    # Add nodes
    workflow.add_node("retrieve_guidelines", retrieve_guidelines)
    workflow.add_node("generate_report", generate_report)

    # Define edges
    workflow.set_entry_point("retrieve_guidelines")
    workflow.add_edge("retrieve_guidelines", "generate_report")
    workflow.add_edge("generate_report", END)

    return workflow.compile()


# Compile once at import time
health_agent = build_agent()


def run_agent(
    patient_summary: str,
    risk_score: float,
    prediction_label: str,
    prediction_confidence: float,
    groq_api_key: str
) -> str:
    """
    Public interface to run the health agent.
    Returns the final structured health report as a markdown string.
    """
    global GROQ_API_KEY, _llm
    GROQ_API_KEY = groq_api_key
    _llm = None  # Reset LLM to use the new key if changed

    initial_state: HealthAgentState = {
        "patient_summary": patient_summary,
        "risk_score": risk_score,
        "prediction_label": prediction_label,
        "prediction_confidence": prediction_confidence,
        "retrieved_guidelines": [],
        "retrieval_error": None,
        "health_report": None,
        "generation_error": None,
    }

    final_state = health_agent.invoke(initial_state)
    return final_state.get("health_report", "Report generation failed.")
