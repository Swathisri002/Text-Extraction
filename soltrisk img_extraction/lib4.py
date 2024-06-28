import os
import cv2
import keras_ocr
from PIL import Image
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

# Path to the image folder
image_folder = "static/images"

# List all files in the image folder
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif'))]

# Initialize the Ollama language model
model = Ollama(
    model="llama3",
)  # assuming you have Ollama installed and have llama3 model pulled with `ollama pull llama3`

# Create a prompt template
prompt = ChatPromptTemplate.from_template(
    "I extracted the text from an image but the text is in unstructured format. Based on the text context, you need to structure and summarize the text concisely \n\n{text}"
)

# Function to detect tables in an image
def detect_tables(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = [c for c in contours if cv2.contourArea(c) > 1000]
    return contours

# Function to limit the word count of the generated text
def limit_words(text, max_words):
    words = text.split()
    if len(words) > max_words:
        return ' '.join(words[:max_words])
    return text

# Initialize keras-ocr pipeline
pipeline = keras_ocr.pipeline.Pipeline()

# Function to extract text from image using keras-ocr
def extract_text_with_keras_ocr(image_path):
    image = keras_ocr.tools.read(image_path)
    prediction_groups = pipeline.recognize([image])
    extracted_text = ' '.join([' '.join([text for text, box in prediction]) for prediction in prediction_groups])
    return extracted_text

# Process each image
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    
    # Extract general text from the image using keras-ocr
    general_text = extract_text_with_keras_ocr(image_path)
    
    if general_text.strip():
        # If text is detected, process it
        print(f"Extracted text from {image_file}:")
        print(general_text)
        
        # Determine the maximum number of words based on the extracted text
        max_words = len(general_text.split())
        
        # Chain the prompt and the model together for general text
        chain = prompt | model
        result = chain.invoke({"text": general_text})
        
        # Limit the result to the number of words in the extracted text
        limited_result = limit_words(result, max_words)
        
        print(f"Structured and summarized data from {image_file} (max {max_words} words):")
        print(limited_result)
        print("--------------------------------------------xxxxxx----------------------------------------------")
        print(" ")
    else:
        # If no text is detected, detect tables in the image
        image = cv2.imread(image_path)
        contours = detect_tables(image)
        if contours:
            # Extract and process each table separately
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                table_image_path = f'table_{i+1}.png'
                table_image = image[y:y+h, x:x+w]
                cv2.imwrite(table_image_path, table_image)
                
                table_text = extract_text_with_keras_ocr(table_image_path)
                
                print(f"Extracted table {i+1} from {image_file}:")
                print(table_text)
                
                # Determine the maximum number of words based on the extracted table text
                max_words = len(table_text.split())
                
                # Chain the prompt and the model together for table text
                chain = prompt | model
                result = chain.invoke({"text": table_text})
                
                # Limit the result to the number of words in the extracted table text
                limited_result = limit_words(result, max_words)
                
                print(f"Structured and summarized data from table {i+1} in {image_file} (max {max_words} words):")
                print(limited_result)
                print("--------------------------------------------xxxxxx----------------------------------------------")
                print(" ")
                
                # Optional: draw bounding boxes around detected tables
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                break  # Exit the loop after processing the first table
        else:
            print(f"No text or tables found in {image_file}.")
    
    # Optional: display the image with bounding boxes
    # cv2.imshow('Processed Image', image)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
