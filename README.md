# README.md

## GenAI Technical Assignment – RAG Research Assistant

This repository contains a **Retrieval-Augmented Generation (RAG)** system built as part of the GenAI Technical Assignment. The system answers questions **strictly based on the provided research paper** and refuses to answer out-of-scope questions.

---

## Installation Instructions

### 1. Prerequisites

* Python **3.12 or 3.13**

  * ✅ Tested successfully on Python 3.12 and 3.13 without version conflicts
* An active **OpenAI API key** (used for embeddings and response generation)
* Internet connection

---

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

* **Windows**

```bash
venv\Scripts\activate
```

* **Linux / macOS**

```bash
source venv/bin/activate
```

---

### 3. Install Dependencies

Install all required Python packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

### 4. Environment Configuration

Create a `.env` file in the project root directory and add your API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```


---

##  How to Run the Solution

Execute the RAG pipeline using the following command:

```bash
python rag_pipeline.py
```

### Execution Behavior

* On the **first run**, the system:

  * Loads the research paper PDF
  * Splits it into text chunks
  * Generates embeddings
  * Stores them in a FAISS vector database

* On **subsequent runs**, the existing FAISS index is loaded automatically, and ingestion is skipped.

---

##  Usage

* The application runs in the **command-line interface (CLI)**
* Enter technical questions related to the research paper
* To exit the program, type:

```text
exit
```

If a question cannot be answered using the paper, the system responds:

```text
I cannot answer this based on the provided context.
```

---

##  Notes

* The system uses **only the provided PDF** as its knowledge source
* No external knowledge is used
* The project satisfies the assignment requirement for installation and execution instructions

---

**Author:** Gopal Reddy Pothipeddi
