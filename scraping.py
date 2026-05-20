from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import os
import requests
from urllib.parse import urljoin, urlparse
import time

def save_pdf_from_url(url, save_folder, filename=None, max_retries=3, timeout=10):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    attempt = 0

    while attempt < max_retries:
        try:
            print(f"Attempt {attempt + 1} to download PDF from {url}...")
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()

            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            if not filename:
                filename = os.path.basename(urlparse(url).path)
                if not filename or not filename.endswith('.pdf'):
                    filename = "default_file.pdf"  # Fallback filename

            save_path = os.path.join(save_folder, filename)

            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"PDF saved successfully: {save_path}")
            return  # Exit the function after successful download

        except requests.RequestException as e:
            print(f"Failed to download the PDF from {url}: {e}")
            attempt += 1
            if attempt < max_retries:
                time.sleep(2 ** attempt)  # Exponential backoff

    print(f"Giving up after {max_retries} attempts to download {url}")

def scrape_pdfs_with_selenium(webpage_url, save_folder):
    options = Options()
    options.headless = True
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        print(f"Fetching the webpage: {webpage_url}")
        driver.get(webpage_url)
        time.sleep(10)  # Give the page time to load fully, including any dynamic content

        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Find PDFs based on specific patterns or keywords
        pdf_links = []
        for tag in soup.find_all(['a', 'iframe', 'embed', 'object', 'link'], href=True):
            if '.pdf' in tag['href'].lower() or 'result' in tag.text.lower():
                pdf_links.append(urljoin(webpage_url, tag['href']))

        # If no PDFs found, print a message
        if not pdf_links:
            print("No relevant PDF links found on the webpage.")
            return

        print(f"Found {len(pdf_links)} PDF links. Starting download...")

        for pdf_url in pdf_links:
            filename = os.path.basename(urlparse(pdf_url).path)
            print(f"Downloading {pdf_url} ...")
            save_pdf_from_url(pdf_url, save_folder, filename)
    finally:
        driver.quit()

# Example usage
webpage_url = "https://www.inecelectionresults.ng/elections/63f8f25b594e164f8146a213/context/pus/lga/5f0f39a94d89fc3a883de371/ward/5f0f3ecf8f77bb3acad0ac4f"  # Replace with the URL of the webpage containing PDFs
save_directory = r"C:\Users\DONKAMS\Desktop\Final year project\pdfs"  # Use raw string
scrape_pdfs_with_selenium(webpage_url, save_directory)

