import requests
from bs4 import BeautifulSoup
import re
from huggingface_hub import InferenceClient
import streamlit as st
import logging



logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # Log to a file called app.log
        logging.StreamHandler()         # Log to the console
    ]
)

def scrape_website(url):
  response = requests.get(url)

  # Check if the request was successful
  if response.status_code == 200:
    soup = BeautifulSoup(response.content, "html.parser")
    page_text = soup.get_text()
    return page_text

  else:
    print(f"Failed to fetch the webpage. Status code: {response.status_code}")
    return None



def clean_scraped_data(raw_html):
    soup = BeautifulSoup(raw_html, "html.parser")
    text = soup.get_text(separator=" ") # Extract text with spaces

    # Step 2: Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Step 3: Remove special characters 
    text = re.sub(r'[^a-zA-Z0-9.,!?\'" ]+', '', text)

    # Step 4: Normalize text (convert to lowercase, etc.)
    text = text.lower()

    return text


client = InferenceClient(api_key=st.secrets["key"]["key"])
def llm(cleaned_data, asked_data):
  messages = [
    {
      "role": "user",
      "content": f"""
                extract the asked data form text\
                asked data : {asked_data}\
                text:<begin text>
                "{cleaned_data}"
                <end text>
                if text did not contain the asked data return "None"
                remember the output it's just the asked data 
                 """ 
    }
  ]

  completion = client.chat.completions.create(
      model="meta-llama/Llama-3.2-11B-Vision-Instruct", 
    messages=messages, 
    max_tokens=500
  )
  return completion.choices[0].message['content']





def bot(url, asked_data):
    try:
        # Log the link and input data
        logging.info(f"Scraping URL: {url}")
        logging.info(f"User input (asked_data): {asked_data}")

        # Scrape website
        scrapped_text = scrape_website(url)
        if scrapped_text is None:
            logging.warning("Unable to scrape data from the provided URL.")
            return None

        # Clean and process data
        cleaned_data = clean_scraped_data(scrapped_text)
        data = llm(cleaned_data, asked_data)

        # Log the result
        logging.info(f"LLM output: {data}")
        return data
    except Exception as e:
        logging.error(f"An error occurred in the bot: {str(e)}")
        return None

def main():
    st.title("Data Scraping and Processing")

    # Input fields for the user
    link = st.text_input("Enter the link:", placeholder="https://example.com")
    asked_data = st.text_input(
        "Enter the specific data you need (max 100 characters):",
        placeholder="E.g., product price, contact info",
        max_chars=100,  # Enforce 100-character limit here
    )

    # Submit button
    if st.button("Process Data"):
        if link and asked_data:
            # Call the bot function with inputs
            result = bot(link, asked_data)
            if result:
                st.success("Data processed successfully!")
                st.write("**Output:**")
                st.write(result)
            else:
                st.error("Failed to process data. Please check the URL or input.")
        else:
            st.error("Please provide both the link and the specific data needed.")

if __name__ == "__main__":
    main()
