import requests
from bs4 import BeautifulSoup
import re
from huggingface_hub import InferenceClient
import streamlit as st


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
  scrapped_text =scrape_website(url)
  if scrapped_text == None:
    return None
  cleaned_data = clean_scraped_data(scrapped_text)
  words = cleaned_data.split()
  first_450_words = " ".join(words[:450])
  data = llm(first_450_words, asked_data)
  print(data)
  return data



def main():
    st.title("Data Scraping and Processing")

    # Input fields for the user
    link = st.text_input("Enter the link:", placeholder="https://example.com")
    scrapped_data = st.text_area("Enter the scrapped data:", placeholder="Paste your scrapped data here")

    # Submit button
    if st.button("Process Data"):
        if link and scrapped_data:
            # Call the bot function with inputs
            result = bot(link, scrapped_data)
            st.success("Data processed successfully!")
            st.write("**Output:**")
            st.write(result)
        else:
            st.error("Please provide both the link and scrapped data.")

if __name__ == "__main__":
    main()
