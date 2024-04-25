#!/bin/sh

python src/create_vector_database.py /data/schemas_1.txt
streamlit run src/streamlit_program.py --server.port=8501
