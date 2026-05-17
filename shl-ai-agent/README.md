# SHL AI Assessment Recommender

## Run Project

### Install Dependencies
pip install -r requirements.txt

### Create Vector DB
python -c "from app.retriever import create_vector_database; create_vector_database()"

### Run Backend API Server
uvicorn app.main:app --reload

### Run Streamlit Frontend
streamlit run frontend.py

### Open Swagger Docs
http://127.0.0.1:8000/docs