import json

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


# -----------------------------------
# EMBEDDING MODEL
# -----------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------------
# SAFE STRING CONVERTER
# -----------------------------------

def safe_string(value):

    if value is None:
        return ""

    return str(value)


# -----------------------------------
# SAFE LIST TO STRING
# -----------------------------------

def safe_join(value):

    if not value:
        return ""

    if isinstance(value, list):
        return ", ".join([str(v) for v in value])

    return str(value)


# -----------------------------------
# CREATE VECTOR DATABASE
# -----------------------------------

def create_vector_database():

    # -----------------------------
    # LOAD JSON CATALOG
    # -----------------------------

    with open(
        "data/assessments.json",
        "r",
        encoding="utf-8"
    ) as f:

        catalog = json.load(f)

    documents = []

    # -----------------------------
    # PROCESS EACH ASSESSMENT
    # -----------------------------

    for item in catalog:

        # -------------------------
        # SAFE FIELD EXTRACTION
        # -------------------------

        entity_id = safe_string(
            item.get("entity_id")
        )

        name = safe_string(
            item.get("name")
        )

        description = safe_string(
            item.get("description")
        )

        categories = safe_join(
            item.get("keys")
        )

        job_levels = safe_join(
            item.get("job_levels")
        )

        languages = safe_join(
            item.get("languages")
        )

        duration = safe_string(
            item.get("duration")
        )

        remote = safe_string(
            item.get("remote")
        )

        adaptive = safe_string(
            item.get("adaptive")
        )

        url = safe_string(
            item.get("link")
        )

        # -------------------------
        # SEARCHABLE TEXT
        # -------------------------

        text = f"""
Assessment Name:
{name}

Description:
{description}

Categories:
{categories}

Job Levels:
{job_levels}

Languages:
{languages}

Duration:
{duration}

Remote Testing:
{remote}

Adaptive Testing:
{adaptive}
"""

        # -------------------------
        # DOCUMENT OBJECT
        # -------------------------

        documents.append(
            Document(
                page_content=text,

                metadata={

                    "entity_id": entity_id,

                    "name": name,

                    "url": url,

                    "description": description,

                    "categories": categories,

                    "job_levels": job_levels,

                    "languages": languages,

                    "duration": duration,

                    "remote": remote,

                    "adaptive": adaptive
                }
            )
        )

    # -----------------------------
    # CREATE CHROMADB
    # -----------------------------

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory="data/chroma_db"
    )

    vectorstore.persist()

    print("Vector DB created successfully")


# -----------------------------------
# LOAD VECTORSTORE
# -----------------------------------

def load_vectorstore():

    return Chroma(
        persist_directory="data/chroma_db",
        embedding_function=embedding_model
    )