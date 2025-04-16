import os
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableSequence

template = """
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


class InvoiceParser:
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        os.environ['GROQ_API_KEY'] = api_key
        
        self.invoice_prompt_template = PromptTemplate(
                        input_variables=["invoice_text"],
                        template=template)
        self.llm = ChatGroq(model=model)
        self.chain = RunnableSequence(self.invoice_prompt_template | self.llm)

    def parse(self, text: str) -> str:
        result = self.chain.invoke({"invoice_text": text})
        return result.content