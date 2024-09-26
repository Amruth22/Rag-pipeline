import os
import shutil
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Together
from langchain_community.document_loaders import UnstructuredPDFLoader, TextLoader,CSVLoader,UnstructuredPowerPointLoader,UnstructuredEPubLoader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from pdfminer.high_level import extract_text
import boto3
import cv2
import pandas as pd
from PIL import Image
from PyPDF2 import PdfWriter, PdfReader,PdfMerger
from pdf2image import convert_from_path, pdfinfo_from_path
from reportlab.pdfgen.canvas import Canvas
from hashlib import md5
from langchain_experimental.agents import create_csv_agent
from dotenv import load_dotenv
load_dotenv()
region_name = 'us-east-1'
# logo_path = 'Logo/Logo1.png' # Update this to the path of your logo file
# st.image(logo_path,use_column_width='auto')
os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY")
os.environ["aws_access_key_id"] = os.getenv("aws_access_key_id")
os.environ["aws_secret_access_key"] = os.getenv("aws_secret_access_key")
aws_textract = boto3.client('textract',region_name=region_name,
        aws_access_key_id=os.getenv("aws_access_key_id"),aws_secret_access_key=os.getenv("aws_secret_access_key"))

s3_bucket = boto3.resource('s3',region_name=region_name,aws_access_key_id=os.getenv("aws_access_key_id"),aws_secret_access_key=os.getenv("aws_secret_access_key"))
retriever_cache = {}

def get_pdf_hash(path):
    """Generate a unique hash for a PDF based on its content."""
    hasher = md5()
    with open(path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()
def inference(chain, input_query):
    """Invoke the processing chain with the input query."""
    result = chain.invoke(input_query)
    return result


def create_chain(retriever, prompt, model):
    """Compose the processing chain with the specified components."""
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain


def generate_prompt():
    """Define the prompt template for question answering."""
    template = """<s>[INST] Answer the question in a simple sentence based only on the following context:
                  {context}
                  Question: {question} [/INST] 
               """
    return ChatPromptTemplate.from_template(template)


def configure_model():
    """Configure the language model with specified parameters."""
    return Together(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0.1,
        max_tokens=512,
        top_k=50,
        top_p=0.7,
        repetition_penalty=1.1,
    )


def configure_retriever(pdf_loader):
    """Configure the retriever with embeddings and a FAISS vector store."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(pdf_loader, embeddings)
    return vector_db.as_retriever()

def is_searchable_pdf(filepath):
    """Check if the PDF at filepath is searchable by attempting to extract text."""
    try:
        text = extract_text(filepath, maxpages=1)
        return bool(text.strip())
    except Exception as e:
        print(f"Failed to check if PDF is searchable: {e}")
        return False

def convert_pdf_to_pages(document_path):
    document_images = []
    images_path = None  # Initialize with None or a default path
    try:
        info = pdfinfo_from_path(document_path, userpw=None, poppler_path=None)
        maxPages = info["Pages"]
        images_path = document_path.replace(".", "_")
        filename_split = os.path.basename(document_path).split(".")[0]
        os.makedirs(images_path, exist_ok=True)
        print("Image Conversion Start")
        helper = -1
        for page in range(1, maxPages + 1, 10):
            images = convert_from_path(document_path, dpi=300, first_page=page, last_page=min(page + 9, maxPages))
            for i, image in enumerate(images):
                helper += 1
                name = f"{filename_split}_{helper + 1000}.png"
                store_path = os.path.join(images_path, name)
                image.save(store_path, 'PNG')
                document_images.append(store_path)
        print("pdf_to_image_conversion :: ")
    except Exception as e:
        print(f"Failed to convert PDF to IMAGES: {e}")
    return document_images, images_path



def extract_raw_text_with_aws(img_path):
        
    meta_data = {}
    with open(img_path, "rb") as document:
        imageBytes = bytearray(document.read())
        response = aws_textract.detect_document_text(
            Document={"Bytes": imageBytes}
        )

        image = Image.open(img_path)
        width, height = image.size

        ocr_lines = []
        ocr_words = []
        for item in response["Blocks"]:
            if item["BlockType"] in ["LINE", "WORD"]:
                text = item["Text"]
                if item["BlockType"] == "LINE":
                    left = round(
                        width * item["Geometry"]["BoundingBox"]["Left"]
                    )
                    top = round(height * item["Geometry"]["BoundingBox"]["Top"])
                    right = round(
                        (width * item["Geometry"]["BoundingBox"]["Width"])
                        + left
                    )
                    bottom = round(
                        (height * item["Geometry"]["BoundingBox"]["Height"])
                        + top
                    )
                    cords = [left, top, right, bottom]
                    ocr_lines.append(
                        (
                            (left, top, right, bottom),
                            text,
                            round(item["Confidence"]),
                        )
                    )
                    score = round(item["Confidence"])
                    meta_data.update({text: [cords, score]})
                else:
                    left = round(
                        width * item["Geometry"]["BoundingBox"]["Left"]
                    )
                    top = round(height * item["Geometry"]["BoundingBox"]["Top"])
                    right = round(
                        (width * item["Geometry"]["BoundingBox"]["Width"])
                        + left
                    )
                    bottom = round(
                        (height * item["Geometry"]["BoundingBox"]["Height"])
                        + top
                    )
                    cords = (left, top, right, bottom)
                    score = round(item["Confidence"])
                    ocr_words.append([cords, text, score])
    return ocr_words,ocr_lines

def create_searchable_pdf(image_name,outpath,image_width,image_height,image_path,ocr_lines):
    file_name = image_name.split(".")[0]
    outpath = os.path.join(outpath,"Searchable_PDF")
    os.makedirs(outpath,exist_ok= True)
    pdf_out = os.path.join(outpath,file_name +'.pdf')
    try:
        canvas = Canvas(pdf_out,pagesize=(image_width,image_height),pageCompression=None)
        canvas.drawImage(image_path, 0,0,image_width,image_height,mask='auto',preserveAspectRatio=True,anchor='c')
        canvas.setFillColorRGB(255,0,0,0)
        for word in ocr_lines:
            canvas.setFont("Times-Roman",(word[0][3] - word[0][1]))
            canvas.drawString(word[0][0],image_height -word[0][3]+5 ,word[1]+' ')
        canvas.save()
    except Exception as e:
        print("Failed to create_searchable_pdf", str(e))

    return pdf_out
def convert_excel_to_csv(file_path):
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return
    base, extension = os.path.splitext(file_path)
    csv_file = base + '.csv'
    if extension.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
        df.to_csv(csv_file, index=False)
        os.remove(file_path)
    else:
        print("Unsupported file format. Please provide an XLSX or XLS file.")
def load_documents(path):
    os.makedirs("/tmp/1", exist_ok=True)
    """Load and preprocess documents from PDF files located at the specified path."""
    pdf_loader = []
    for file in os.listdir(path):
        if file.endswith('.pdf'):
            filepath = os.path.join(path, file)
            folder_name = file.replace(".pdf", "_pdf").replace(".PDF", "_pdf")
            outpath = os.path.join("/tmp/1", folder_name)
            # Check if this PDF has already been processed
            if os.path.exists(outpath) and len(os.listdir(outpath)) > 0:
                print(f"Skipping already processed file: {file}")
                store_pdf = os.path.join(outpath, file)  # Assuming final processed PDF is named after the original file
            else:
                os.makedirs(outpath, exist_ok=True)
                if is_searchable_pdf(filepath):
                    store_pdf = filepath
                else:
                    # Convert non-searchable PDF
                    document_images, images_path = convert_pdf_to_pages(filepath)
                    all_pdf_path = []
                    for image_path in document_images:
                        image_name = os.path.basename(image_path)
                        read_image = cv2.imread(image_path)
                        image_height, image_width = read_image.shape[:2]
                        ocr_words, ocr_lines = extract_raw_text_with_aws(image_path)
                        pdf_out = create_searchable_pdf(image_name, outpath, image_width, image_height, image_path, ocr_lines)
                        all_pdf_path.append(pdf_out)
                    merger = PdfMerger()
                    store_pdf = os.path.join(outpath, file)
                    for pdf_path in sorted(all_pdf_path):
                        merger.append(pdf_path)
                    merger.write(store_pdf)
                    merger.close()
            # Load and split the processed PDF
            loader = UnstructuredPDFLoader(store_pdf)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)
            pdf_loader.extend(docs)
        elif file.endswith('.txt'):
            filepath = os.path.join(path, file) 
            loader = TextLoader(filepath)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)
            pdf_loader.extend(docs)
        elif file.endswith('.xlsx') or file.endswith('.xls'):
            filepath = os.path.join(path, file)
            convert_excel_to_csv(filepath) 
            base, _ = os.path.splitext(filepath)
            new_filepath = base + '.csv'
            loader = CSVLoader(new_filepath, encoding='utf-8')
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)
            pdf_loader.extend(docs)
        elif file.endswith('.csv'):
            filepath = os.path.join(path, file)
            loader = CSVLoader(filepath,encoding='utf-8')
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)
            pdf_loader.extend(docs)
        elif file.endswith('.pptx') or file.endswith('.ppt'):
            filepath = os.path.join(path, file)
            loader = UnstructuredPowerPointLoader(filepath,encoding='utf-8')
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)
            pdf_loader.extend(docs)
        elif file.endswith('.epub'):
            filepath = os.path.join(path, file)
            loader = UnstructuredEPubLoader(filepath,encoding='utf-8')
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)
            pdf_loader.extend(docs)
    return pdf_loader



def process_document(path, input_query):
    for file in os.listdir(path):
        if file.endswith(('.pdf', '.txt', '.pptx', '.ppt', '.epub')):
            full_path=os.path.join(path, file)
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            unique_id = "-".join(sorted([get_pdf_hash(os.path.join(path, file)) for file in os.listdir(path) if file.endswith(('.pdf', '.txt','.csv','.xlsx','.xls','.pptx','.ppt','.epub'))]))
            vectorstore_path = f"/tmp/1/{unique_id}"
            if os.path.exists(vectorstore_path):
                new_db = FAISS.load_local(vectorstore_path, embeddings)
                print("Using cached vector store:", new_db)
            else:
                pdf_loader = load_documents(path)
                vector_db = FAISS.from_documents(pdf_loader, embeddings)
                vector_db.save_local(vectorstore_path)
                new_db = vector_db
                print("Newly created vector store saved:", new_db)

            retriever = new_db.as_retriever()

            llm_model = configure_model()
            prompt = generate_prompt()
            chain = create_chain(retriever, prompt, llm_model)
            response = inference(chain, input_query)
            return response
        elif file.endswith(('.xlsx', '.xls')):
            full_path=os.path.join(path, file)
            convert_excel_to_csv(full_path) 
            base, _ = os.path.splitext(full_path)
            new_filepath = base + '.csv'
            llm=Together(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            temperature=0.0,
            max_tokens=3000,
            top_k=50,
            top_p=0.7,
            repetition_penalty= 1.1)
            agent = create_csv_agent(llm,
            new_filepath,
            verbose=True)
            responses=agent.run(input_query)
            return responses
        elif file.endswith(('.csv')):
            full_path=os.path.join(path, file)
            llm=Together(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            temperature=0.0,
            max_tokens=3000,
            top_k=50,
            top_p=0.7,
            repetition_penalty= 1.1)
            agent = create_csv_agent(llm,
            full_path,
            verbose=True)
            responses=agent.run(input_query)
            return responses




def main():
    """Main function to run the Streamlit app."""
    tmp_folder = '/tmp/1'
    os.makedirs(tmp_folder,exist_ok=True)

    st.title("Chat Interface For RAG Using LLM🧠")

    uploaded_files = st.sidebar.file_uploader("Choose files to upload", accept_multiple_files=True, type=['.pdf', '.txt','.csv','.xlsx','.xls','.pptx','.ppt','.epub'])
    if uploaded_files:
        for file in uploaded_files:
            with open(os.path.join(tmp_folder, file.name), 'wb') as f:
                f.write(file.getbuffer())
        st.success('File successfully uploaded. Start prompting!')
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if uploaded_files:
        with st.form(key='question_form'):
            user_query = st.text_input("Ask a question:", key="query_input")
            if st.form_submit_button("Ask") and user_query:
                response = process_document(tmp_folder, user_query)
                st.session_state.chat_history.append({"question": user_query, "answer": response})
            
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
        for chat in st.session_state.chat_history:
            st.markdown(f"**Q:** {chat['question']}")
            st.markdown(f"**A:** {chat['answer']}")
            st.markdown("---")
    else:
        st.success('Upload Document to Start Process !')

    if st.sidebar.button("REMOVE UPLOADED FILES"):
        document_count = os.listdir(tmp_folder)
        if len(document_count) > 0:
            shutil.rmtree(tmp_folder)
            st.sidebar.write("FILES DELETED SUCCESSFULLY !!!")
        else:
            st.sidebar.write("NO DOCUMENT FOUND TO DELETE !!! PLEASE UPLOAD DOCUMENTS TO START PROCESS !! ")


if __name__ == "__main__":
    main()