# Databricks notebook source
# MAGIC %pip install requests-html beautifulsoup4

# COMMAND ----------

import requests
from requests_html import HTMLSession
from bs4 import BeautifulSoup
import urllib.parse

def search(
  term: str = "databricks sql group by",
  max_records: int = 30,
  max_search_results: int = 10,
  target_suffix: str = "sql-ref-syntax-qry"
):

  encoded_term = urllib.parse.quote(term + " site:docs.databricks.com")
  headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
    "Content-Type": "application/json; charset=UTF-8"
  }

  url = f'https://www.google.com/search?q={encoded_term}&amp;ie=utf-8&amp;oe=utf-8&amp;num={max_search_results}'

  # TODO: use XPath to get sample code excerpts
  # TODO 2: store SQL documentation embeddings and use it

  try:
    session = HTMLSession()
    response = session.get(
      url,
      headers = headers
    )

    soup = BeautifulSoup(response.text, 'html.parser')
    all_data = soup.find_all("div",{"class":"g"})

    links = []
    for div in all_data[:max(max_records, len(all_data))]:
      item = [span.get("href") for span in div.find_all("a")][0]
      links.append(item)
    
    target_links = [link for link in links if target_suffix in link]

    target_url_content = target_links[0]
    content_response = session.get(
      target_url_content,
      headers = headers
    )
    
    soup = BeautifulSoup(content_response.text, 'html.parser')
    content_data = soup.find_all("pre")

    result = ""
    for content in content_data:
      result += content.extract().get_text()
    return result
      
  except requests.exceptions.RequestException as e:
    print(e)

query_helper = search()
print(query_helper)

# COMMAND ----------

# Getting data from Google

from langchain.tools import Tool

tool = Tool(
    name = "Google Search",
    description = "Search Google for recent results.",
    func = search
)

print(tool.run("databricks sql group by"))

# COMMAND ----------

from langchain.chains import SimpleSequentialChain
overall_chain = SimpleSequentialChain(chains=[chain, chain_two], verbose=True)

# Run the chain specifying only the input variable for the first chain.
catchphrase = overall_chain.run("colorful socks")
print(catchphrase)
