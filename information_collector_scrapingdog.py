# import requests
# import os
# from dotenv import load_dotenv

# load_dotenv()

# BASE_URL = "https://api.scrapingdog.com/google_patents"


# def fetch_patents_scrapingdog(query, num=10):
#     api_key = os.getenv("SCRAPINGDOG_API_KEY")
#     if not api_key:
#         raise Exception("SCRAPINGDOG_API_KEY not set in .env file")

#     params = {
#         "api_key": api_key,
#         "q": query,
#         "num": num
#     }

#     response = requests.get(BASE_URL, params=params, timeout=20)
#     response.raise_for_status()

#     data = response.json()
#     return data.get("patents", [])


# def convert_scrapingdog_patents_to_documents(patents, topic):
#     documents = []

#     for p in patents:
#         abstract = p.get("abstract", "")
#         if not abstract.strip():
#             continue

#         documents.append({
#             "title": p.get("title"),
#             "abstract": abstract,
#             "patent_id": p.get("publication_number"),
#             "publication_date": p.get("publication_date"),
#             "assignee": p.get("assignee"),
#             "source": "scrapingdog",
#             "topic": topic
#         })

#     return documents
