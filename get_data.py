import PyPDF2


import pdfplumber

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text() + '\n'
    
    # print(text[:1000])
    return text


# def extract_text_from_pdf(pdf_path):
#     # Open the PDF file
#     with open(pdf_path, 'rb') as file:
#         reader = PyPDF2.PdfReader(file)
#         text = ''
#         # Iterate through each page and extract text
#         for page in reader.pages:
#             text += page.extract_text() + '\n'
#     return text

# Example usage
pdf_file_path = '~/The~Book.pdf'  # Replace with your PDF file path
extracted_text = extract_text_from_pdf(pdf_file_path)

# Save the extracted text to a .txt file
with open('extracted_text.txt', 'w', encoding='utf-8') as text_file:
    text_file.write(extracted_text)

print("Text extraction complete! Check 'extracted_text.txt' for results.")
