# Agentic Diabetes Risk Assistant 🩺🤖

A professional-grade, hybrid AI system that combines **Classic Machine Learning** with **Agentic RAG (Retrieval-Augmented Generation)** to provide highly accurate diabetes risk assessments and evidence-based health guidance.

## 🌟 Project Overview

This system evolves from a standard binary classifier into a sophisticated **Health Support Agent**. It operates in two primary phases:
1.  **ML Risk Prediction Engine**: A two-stage pipeline (Linear Regression + Decision Tree) that analyzes 24 clinical parameters to calculate a precise risk score (0-100) and a diagnostic label.
2.  **Agentic AI Reporting (RAG)**: A LangGraph-orchestrated workflow that retrieves localized medical guidelines from the **WHO** and **ICMR (Indian Council of Medical Research)** to generate personalized, medical-grade health reports.

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Input_Layer ["1. Patient Input Layer"]
        A[User Health Metrics<br/><small>Clinical, Lifestyle, Demo</small>]
    end

    subgraph ML_Engine ["2. ML Risk Engine"]
        B[StandardScaler<br/><small>Feature Normalization</small>]
        C[Stage 1: Linear Regression<br/><small>Predicts Risk Score</small>]
        D[Stage 2: Decision Tree<br/><small>Binary Diagnosis</small>]
    end

    subgraph RAG_Pipeline ["3. Knowledge Retrieval (RAG)"]
        E[(ChromaDB<br/>Vector Store)]
        F[WHO Global Guidelines]
        G[ICMR India Guidelines]
        H[Query Embedding<br/><small>HuggingFace MiniLM</small>]
    end

    subgraph Agentic_Orchestration ["4. Agentic AI (LangGraph)"]
        I[State Management]
        J[Retrieval Node]
        K[Reasoning & Generation Node<br/><small>Groq Llama-3.3-70B</small>]
    end

    subgraph Interaction_Layer ["5. User Interface"]
        L[Streamlit Dashboard]
        M[Personalized Health Report]
        N[Conversational Chatbot<br/><small>Follow-up QA</small>]
    end

    %% Connections
    A --> B
    B --> C
    C --> |Risk Score| D
    D --> |Detection Label| I
    
    F & G --> |PDF Ingestion| E
    
    I --> J
    J <--> |Similarity Search| E
    J --> K
    K --> |Structured Markdown| M
    
    M --> N
    N <--> |History & Context| K
    L <==> A & M & N
```

## 🚀 Key Features

- **Hybrid Prediction Pipeline**: Combines continuous risk scoring with discrete classification for a nuanced health perspective.
- **Evidence-Based RAG**: Unlike generic LLMs, this agent grounds its advice in official **WHO** and **ICMR** documents, citing sources directly.
- **Conversational Follow-ups**: A stateful chat interface allows users to ask clarifying questions about their report (e.g., "What does the ICMR say about my physical activity levels?").
- **Localized Context**: Specifically tuned with ICMR guidelines for Asian-Indian phenotypic risk factors.
- **Modern UI**: Implemented with glassmorphism aesthetics and responsive Streamlit components.

## 🛠️ Tech Stack

- **Orchestration**: LangGraph
- **LLM Engine**: Groq (Llama-3.3-70B-Versatile)
- **Vector Database**: ChromaDB
- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **ML Frameworks**: Scikit-Learn, Pandas, NumPy
- **Frontend**: Streamlit + Custom CSS
- **Deployment**: Streamlit Community Cloud

## 📂 Project Structure

```text
├── data/                       # Raw clinical datasets
├── models/                     # Serialized ML models (Joblib)
├── milestone_1/                # ML Discovery & Local Prediction App
│   ├── app.py                  # Initial ML dashboard
│   └── style.css
├── milestone_2/                # Agentic AI & RAG Implementation
│   ├── app.py                  # Main Dashboard (Agentic/Chat)
│   ├── agent.py                # LangGraph definition & Nodes
│   ├── ingest.py               # Vector Database Ingestion script
│   ├── knowledge/              # Official Medical PDFs (ICMR/WHO)
│   ├── chroma_db/              # Persistent Vector Store
│   └── style.css               # Dashboard styling
├── requirements.txt            # System dependencies
└── README.md
```

## ⚙️ Fast Start

1.  **Ingest Knowledge**:
    ```bash
    python milestone_2/ingest.py
    ```
2.  **Run Application**:
    ```bash
    streamlit run milestone_2/app.py
    ```
3.  **Deploy**:
    - Push to GitHub.
    - Connect to [share.streamlit.io](https://share.streamlit.io).
    - Add `GROQ_API_KEY` to Streamlit Secrets.

---
*⚕️ **Disclaimer**: This tool is for informational and educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.*
