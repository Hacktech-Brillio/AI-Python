from flask import Flask, request, render_template
from bs4 import BeautifulSoup
import requests
from langdetect import detect

app = Flask(__name__)

@app.route("/search", methods=["GET"])
def search():
    # Retrieve URLs from query parameters
    urls = request.args.getlist("urls")
    if not urls:
        return "Please provide at least one product URL.", 400

    # Perform the search for reviews on each URL
    search_results = search_for_reviews(urls)
    return render_template("index.html", search_results=search_results)

def search_for_reviews(urls):
    results = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for url in urls:
        reviews = fetch_reviews(url, headers)
        results.append({"url": url, "reviews": reviews})

    return results

def fetch_reviews(url, headers):
    """Fetches English review data from a given product URL using BeautifulSoup."""
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        # Example review scraping logic (adjust selectors based on each site structure)
        reviews = []
        for review in soup.select(".review-text"):  # Adjust selector for the specific site
            review_text = review.get_text(strip=True)

            # Detect and keep only English reviews
            try:
                if detect(review_text) == 'en':
                    reviews.append(review_text)
            except:
                continue  # Skip any reviews where language detection fails
        
        if not reviews:
            reviews.append("No English reviews found.")
        return reviews
    except Exception as e:
        return ["Error fetching reviews"]

if __name__ == "__main__":
    app.run(debug=True)
