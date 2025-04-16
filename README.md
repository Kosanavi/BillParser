# Project Title: Bill Image Parser Using Generative AI

# Hosted at 

https://billparser-assessment.streamlit.app/

# Overview:

This project is focused on automating the extraction of structured data from images of bills or invoices using Generative AI. The core idea is to take a picture or scanned image of a bill and use a combination of OCR and a generative language model to convert the unstructured text into a clean, structured JSON format that can be used for the analytics.


# Technologies Used:

Python, Streamlit for UI

PaddleOCR / RapidOCR for text extraction

LangChain + LLMs + ChatGroQ for text parsing

PIL, NumPy, and standard Python libraries for image and text processing


# Project Structure

├── app.py                # Streamlit application

├── ocr_processor.py      # OCR logic class (RapidOCR)

├── invoice_parser.py     # LLM-based invoice parsing class

├── requirements.txt      # Python dependencies

├── packages.txt          # Linux dependencies

└── README.md             # This file


# Setup Instructions:

# 1. Clone the repo
git clone https://github.com/Kosanavi/BillParser

cd BillParser

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Streamlit app
streamlit run app.py


