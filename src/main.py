import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import requests
import streamlit as st
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="News Sentiment Analyzer", page_icon="📰", layout="wide")

st.markdown(
    """
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 400px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


def scrape_news(topic: str, num_pages: int) -> list:
    """
    Scrapes recent news headlines, summaries, and other details for a given topic
    from Yahoo News across multiple pages.

    Args:
        topic (str): The search topic to scrape news for.

    Returns:
        list: A list of dictionaries, where each dictionary represents a
              news article and contains its headline, source, time, description, and link.
    """
    base_url = "https://ca.news.search.yahoo.com/search"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    all_articles = []

    for page in range(1, num_pages + 1):
        params = {
            "p": topic,
            "fr2": "piv-web",
            "fr": "uh3_news_web",
            "b": (page - 1) * 10 + 1,  # Yahoo uses 1 based indexing instead of 0...
        }

        try:
            response = requests.get(
                base_url, params=params, headers=headers, timeout=10
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error making request for page {page}: {e}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        news_items = soup.find_all("div", class_="NewsArticle")

        for item in news_items:
            title_element = item.find("h4", class_="s-title")
            headline, link = "No headline found", "#"
            if title_element and (link_tag := title_element.find("a")):
                headline = link_tag.get_text(strip=True)
                link = link_tag.get("href")

            source_element = item.find("span", class_="s-source")
            source = (
                source_element.get_text(strip=True)
                if source_element
                else "Unknown source"
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
                all_articles.append(
                    {
                        "headline": headline,
                        "source": source,
                        "time": time_published,
                        "description": description,
                        "link": link,
                    }
                )
        print(f"Scraped page {page} for '{topic}'.")

    print(f"Total articles scraped: {len(all_articles)} for '{topic}'.")
    print(json.dumps(all_articles, indent=4))
    return all_articles


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
    if cs <= -0.05:
        return "Negative"
    return "Neutral"


def visualize_sentiment_results(data: dict, topic: str, avg: float):
    """
    Creates visualizations for the sentiment analysis results.

    Args:
        data (dict): Dictionary containing counts of each sentiment
        topic (str): The search topic
        avg (float): The average compound sentiment score
    """
    # Ensure the output dir actually exists...
    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Simple bar chart for the sentiment distributions
    plt.figure(figsize=(10, 6))
    plt.bar(
        data.keys(),
        data.values(),
        color=["green", "red", "gray"],
    )
    plt.title(f'Sentiment Distribution for "{topic}" News Headlines')
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Headlines")

    # Add count labels on top of bars...
    for i, count in enumerate(data.values()):
        plt.text(i, count, str(count), ha="center", va="bottom")

    plt.savefig(f"output/sentiment_distribution_{timestamp}.png")
    plt.close()

    # Pie chart version for the visualizations
    plt.figure(figsize=(8, 8))
    plt.pie(
        data.values(),
        labels=data.keys(),
        autopct="%1.1f%%",
        colors=["green", "red", "gray"],
    )
    plt.title(f'Sentiment Distribution for "{topic}" News Headlines')
    plt.savefig(f"output/sentiment_pie_{timestamp}.png")
    plt.close()


def save_results_to_file(articles: list, data: dict, topic: str, avg: float):
    """
    Saves the analysis results to a JSON file.

    Args:
        articles (list): List of analyzed articles
        data (dict): Dictionary containing sentiment counts
        topic (str): The search topic
        avg (float): The average compound sentiment score
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "topic": topic,
        "timestamp": timestamp,
        "total_articles": len(articles),
        "sentiment_distribution": data,
        "avg": avg,
        "articles": articles,
    }

    os.makedirs("output", exist_ok=True)
    with open(f"output/analysis_results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=4)


def display_results(articles: list, data: dict, topic: str, avg: float):
    """
    Displays the analysis results in the Streamlit interface.

    Args:
        articles (list): List of analyzed articles
        data (dict): Dictionary containing sentiment counts
        topic (str): The search topic
        avg (float): The average compound sentiment score
    """
    # Create two columns for the charts (bar and pie)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Distribution")
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.bar(data.keys(), data.values(), color=["green", "red", "gray"])
        ax1.set_title(f'Sentiment Distribution for "{topic}" News Headlines')
        ax1.set_xlabel("Sentiment")
        ax1.set_ylabel("Number of Headlines")
        for i, count in enumerate(data.values()):
            ax1.text(i, count, str(count), ha="center", va="bottom")
        st.pyplot(fig1)
        plt.close()

    with col2:
        st.subheader("Sentiment Distribution (Pie Chart)")
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        ax2.pie(
            data.values(),
            labels=data.keys(),
            autopct="%1.1f%%",
            colors=["green", "red", "gray"],
        )
        ax2.set_title(f'Sentiment Distribution for "{topic}" News Headlines')
        st.pyplot(fig2)
        plt.close()

    st.subheader("Overall Statistics")
    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric("Positive Headlines", data["Positive"])
    with col4:
        st.metric("Negative Headlines", data["Negative"])
    with col5:
        st.metric("Neutral Headlines", data["Neutral"])

    st.metric("Average Compound Score", f"{avg:.3f}")
    st.metric("Overall Sentiment", get_sentiment_label(avg))

    # Display articles in an expandable sections :)
    st.subheader("Analyzed Articles")
    for article in articles:
        sentiment_color = {"Positive": "🟢", "Negative": "🔴", "Neutral": "⚪"}[
            article["sentiment_label"]
        ]

        with st.expander(f"{sentiment_color} {article['headline']}"):
            st.write(f"**Source:** {article['source']}")
            st.write(f"**Time:** {article['time']}")
            st.write(f"**Description:** {article['description']}")
            st.write("**Sentiment Scores:**")
            st.write(f"- Compound: {article['sentiment_scores']['compound']:.3f}")
            st.write(f"- Positive: {article['sentiment_scores']['pos']:.3f}")
            st.write(f"- Negative: {article['sentiment_scores']['neg']:.3f}")
            st.write(f"- Neutral: {article['sentiment_scores']['neu']:.3f}")
            st.markdown(f"[Read full article]({article['link']})")


def main():
    st.title("📰 News Sentiment Analyzer")
    st.write("Analyze the sentiment of news articles for any topic using Yahoo News.")

    search_topic = st.text_input("Enter a topic to analyze:", "Elon Musk")

    num_pages = st.slider(
        "Number of pages to scrape:", min_value=1, max_value=10, value=5
    )

    if st.button("Analyze Sentiment"):
        with st.spinner("Scraping news articles..."):
            scraped_articles = scrape_news(search_topic, num_pages)

        if not scraped_articles:
            st.error("No articles were found. Please try a different topic.")
        else:
            st.success(
                f"Successfully scraped {len(scraped_articles)} articles for '{search_topic}'."
            )

            analyzer = SentimentIntensityAnalyzer()
            sentiment_data = {"Positive": 0, "Negative": 0, "Neutral": 0}
            total_compound_score = 0

            for article in scraped_articles:
                sentiment_scores = analyzer.polarity_scores(article["headline"])
                compound_score = sentiment_scores["compound"]
                sentiment_label = get_sentiment_label(compound_score)

                article["sentiment_scores"] = sentiment_scores
                article["sentiment_label"] = sentiment_label

                sentiment_data[sentiment_label] += 1
                total_compound_score += compound_score

            sentiment_avg = total_compound_score / len(scraped_articles)

            display_results(
                scraped_articles, sentiment_data, search_topic, sentiment_avg
            )

            save_results_to_file(
                scraped_articles, sentiment_data, search_topic, sentiment_avg
            )
            st.info("Results have been saved to the 'output' directory.")


if __name__ == "__main__":
    main()
