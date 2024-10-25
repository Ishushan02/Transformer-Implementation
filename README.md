# Transformer-Implementation

- Download the Book(https://mankshop.in/wp-content/uploads/2024/08/OceanofPDF.com_I_dont_love_you_anymore_-_Rithvik_singh_rathore.pdf?srsltid=AfmBOooIQvJAM25vfy3UlLL3ceWAKcq6HD7_BfStcBQ5_Niy9CXutvVE) an execute get_data.py to extract the texts from the PDF.

- The extracted text will be present in extracted_text.txt

- Provide the File path of extracted text file and run the prepare_data.py to enocde the text file with our random character level encoder of each characters.

- Just to test how the Model works, execute language_model.py to test a Bigram Model, it's inefficient but it's just to show up the sample of it's working.

- Transformer Chunks are just simple mathematical operations which is showing how do we create the building block of Attention Model(Attention Unit) based on averaging the previous data.

- To Execute the Complete Transformer Model for test purpose on the embeddings numbered from 1 -> 63 execute transformer.py

- To Test it on pretrained Embedding exxecute training_using_BERT.py