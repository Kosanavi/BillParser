import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from PIL import Image
import pytesseract
import os
import json
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableSequence
from paddleocr import PaddleOCR


ocr = PaddleOCR(use_angle_cls=False, lang='en', det_model_dir='models/det', rec_model_dir='models/rec', use_gpu=False)
# Access the key from Streamlit's secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

invoice_prompt_template = PromptTemplate(
    input_variables=["invoice_text"],
    template="""
You are an intelligent invoice parser.

Extract the invoice information from the text below and return the structured JSON in the format shown. 
Return only the JSON structure as output. Do not include any explanations or text outside the JSON.


Format:
{{
  "invoiceDetails": {{
    "invoiceNumber": "",
    "invoiceDate": "",
    "orderNumber": "",
    "orderDate": ""
  }},
  "sellerInformation": {{
    "name": "",
    "address": "",
    "panNo": "",
    "gstRegistrationNo": ""
  }},
  "buyerInformation": {{
    "name": "",
    "billingAddress": "",
    "shippingAddress": "",
    "stateUTCode": ""
  }},
  "items": [
    {{
      "description": "",
      "unit": 1,
      "price": 0.0,
      "taxRate": 0.0,
      "taxType": ["CGST", "SGST"],
      "totalTaxAmount": "",
      "totalAmount": 0.0
    }}
  ],
  "totals": {{
    "subtotal": 0.0,
    "tax": 0.0,
    "total": 0.0
  }},
  "amountInWords": ""
}}

Text to extract from:
\"\"\"{invoice_text}\"\"\"
"""
)

import numpy as np

# Initialize LLM chain
model="llama-3.3-70b-versatile"
llm = ChatGroq(model=model)


# 3. Create the chain
from langchain.chains import LLMChain
chain = RunnableSequence(invoice_prompt_template | llm)

# Streamlit app
st.set_page_config(page_title="Bill Parser", layout="centered")
st.title("Bill Image Parsing Agent")

# Session state to hold OCR and LLM output
if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = None
if "structured_output" not in st.session_state:
    st.session_state.structured_output = None
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
    st.session_state.structured_output = None
    st.session_state.ocr_text = None

# File uploader
uploaded_file = st.file_uploader("Upload a bill image", type=["jpg", "jpeg", "png"])

if uploaded_file and not st.session_state.ocr_text:
    image = Image.open(uploaded_file)
    #st.image(image, caption="Uploaded Bill", use_column_width=True)

    with st.spinner("Extracting text with OCR..."):
        result = ocr.ocr(np.array(image))
        # Extract and print full text
        full_text = "\n".join([line[1][0] for line in result[0]])
        st.session_state.ocr_text = full_text

if st.session_state.ocr_text:
    st.subheader("OCR Result")
    st.text_area("Raw OCR Text", st.session_state.ocr_text, height=200)

    with st.spinner("Calling Agent for parsing the bill..."):
        result = chain.invoke({"invoice_text": st.session_state.ocr_text})
        st.session_state.structured_output = result.content

if st.session_state.structured_output:
    st.subheader("Structured Output (JSON)")
    st.code(st.session_state.structured_output, language="json")

    # Export button
    st.download_button(
        label="Download as JSON",
        file_name=f"parsed_bill.json",
        mime="application/json",
        data=st.session_state.structured_output
    )



