from bs4 import BeautifulSoup
from langchain_community.document_loaders import RecursiveUrlLoader
import re

def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()

def load_website():
    print("Loading website...")
    starting_url = "https://www.angelone.in/support/"
    loader = RecursiveUrlLoader(
        starting_url, 
        extractor=bs4_extractor,
        max_depth=10
    )
    docs = loader.load()
    return docs

# load_website()
