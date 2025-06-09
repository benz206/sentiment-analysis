import requests
from bs4 import BeautifulSoup
import json

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def scrape_news(topic: str) -> list:
    """
    Scrapes recent news headlines, summaries, and other details for a given topic
    from Yahoo News.

    Args:
        topic (str): The search topic to scrape news for.

    Returns:
        list: A list of dictionaries, where each dictionary represents a
              news article and contains its headline, source, time, description, and link.
    """
    base_url = "https://ca.news.search.yahoo.com/search"
    params = {"p": topic, "fr2": "piv-web", "fr": "uh3_news_web"}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(base_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")

    articles = []
    news_items = soup.find_all("div", class_="NewsArticle")

    for item in news_items:
        title_element = item.find("h4", class_="s-title")
        headline, link = "No headline found", "#"
        if title_element and (link_tag := title_element.find("a")):
            headline = link_tag.get_text(strip=True)
            link = link_tag.get("href")

        source_element = item.find("span", class_="s-source")
        source = (
            source_element.get_text(strip=True) if source_element else "Unknown source"
        )

        time_element = item.find("span", class_="s-time")
        time_published = (
            time_element.get_text(strip=True) if time_element else "Unknown time"
        )

        description_element = item.find("p", class_="s-desc")
        description = (
            description_element.get_text(strip=True)
            if description_element
            else "No description found."
        )

        if headline != "No headline found":
            articles.append(
                {
                    "headline": headline,
                    "source": source,
                    "time": time_published,
                    "description": description,
                    "link": link,
                }
            )
    print(f"Scraped {len(articles)} articles for '{topic}'.")
    print(json.dumps(articles, indent=4))
    return articles


def get_sentiment_label(cs: float) -> str:
    """
    Classifies a VADER compound score into a user-friendly label.

    Args:
        compound_score (float): The compound score from VADER sentiment analysis.

    Returns:
        str: A sentiment label ("Positive", "Negative", or "Neutral").
    """
    if cs >= 0.05:
        return "Positive"
    elif cs <= -0.05:
        return "Negative"
    return "Neutral"


if __name__ == "__main__":
    search_topic = "Elon Musk"
    scraped_articles = scrape_news(search_topic)

    if not scraped_articles:
        print("No articles were scraped. Exiting.")
    else:
        print(
            f"Successfully scraped {len(scraped_articles)} articles for '{search_topic}'.\n"
        )

        analyzer = SentimentIntensityAnalyzer()
        sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
        total_compound_score = 0

        for article in scraped_articles:
            sentiment_scores = analyzer.polarity_scores(article["headline"])
            compound_score = sentiment_scores["compound"]
            sentiment_label = get_sentiment_label(compound_score)

            article["sentiment_scores"] = sentiment_scores
            article["sentiment_label"] = sentiment_label

            sentiment_counts[sentiment_label] += 1
            total_compound_score += compound_score

            print(f"Headline: {article['headline']}")
            print(f"Source: {article['source']} | Time: {article['time']}")
            print(
                f"Sentiment: {article['sentiment_label']} (Compound Score: {compound_score:.2f})"
            )
            print("-" * 30)

        print("\n" + "=" * 40)
        print("Overall Sentiment Analysis Summary")
        print("=" * 40)
        print(f"Positive Headlines: {sentiment_counts['Positive']}")
        print(f"Negative Headlines: {sentiment_counts['Negative']}")
        print(f"Neutral Headlines:  {sentiment_counts['Neutral']}")

        average_sentiment = total_compound_score / len(scraped_articles)
        overall_sentiment_label = get_sentiment_label(average_sentiment)

        print(f"\nAverage Compound Score: {average_sentiment:.3f}")
        print(
            f"Overall sentiment for '{search_topic}' appears to be: {overall_sentiment_label}"
        )
        print("=" * 40)
