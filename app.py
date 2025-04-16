import streamlit as st
from PIL import Image
from ocr_processor import OCRProcessor
from invoice_parser import InvoiceParser
import streamlit.components.v1 as components


# Load API key securely
GROQ_API_KEY ="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"

# Initialize classes
ocr_processor = OCRProcessor()
parser = InvoiceParser(api_key=GROQ_API_KEY)



st.set_page_config(page_title="Bill Parser", layout="centered")
st.title("Bill Image Parsing Agent")

# Session state
if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = None
if "structured_output" not in st.session_state:
    st.session_state.structured_output = None
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None



# Upload and OCR
uploaded_file = st.file_uploader("Upload a bill image", type=["jpg", "jpeg", "png"])
if uploaded_file and not st.session_state.ocr_text:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Bill", use_column_width=False)
    with st.spinner("Extracting text with OCR..."):
        st.session_state.ocr_text = ocr_processor.extract_text(image)

# Detect change or removal of image
current_file = uploaded_file.name if uploaded_file else None
if current_file != st.session_state.last_uploaded_file:
    st.session_state.last_uploaded_file = current_file
    st.session_state.ocr_text = None
    st.session_state.structured_output = None
    st.rerun()  # trigger rerun automatically


# Show OCR and parse
if st.session_state.ocr_text:
    st.subheader("OCR Result")
    st.text_area("Raw OCR Text", st.session_state.ocr_text, height=200)

    with st.spinner("Parsing the bill..."):
        st.session_state.structured_output = parser.parse(st.session_state.ocr_text)

# Show output
if st.session_state.structured_output:
    st.subheader("Structured Output (JSON)")
    st.code(st.session_state.structured_output, language="json")
    st.download_button("Export as JSON", data=st.session_state.structured_output, file_name="parsed_bill.json", mime="application/json")


