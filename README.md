# Web Scraping + Sentiment Analysis

You can try out the final product [here](https://benz-sentiment-analysis.streamlit.app/).

My writeup is also at the bottom of this page.

## Setup + Requirements

Just clone, install reqs, I'm using Python `3.12.8`.

```bash
git clone https://github.com/benz-206/sentiment-analysis.git
cd sentiment-analysis
pip install -r requirements.txt

# Running... Wow
python3 src/main.py

# For streamlit, you can also just run
streamlit run src/main.py
```

## Task

Copied from the discord:

To get everyone started and aligned, here’s a task that blends web scraping and sentiment analysis:

Choose a trending topic you’re interested in (e.g., interest rates, electric vehicles, Elon Musk, Champions League, etc.).
1. Scrape recent news headlines or article summaries related to your topic using Python (sources like Google News, Yahoo News, Reddit, etc.).
2. Perform sentiment analysis on the collected text using any method you prefer—VADER, TextBlob, HuggingFace models, etc.
3. Summarize your findings in any format you like. This could include:
    - A few bullet points describing sentiment trends
    - A short writeup
    - Visualizations (bar charts, word clouds, etc.)
    - Even an interactive dashboard (e.g., Streamlit) if you’re feeling creative
4. Share your results (code + summary/output) in the ⁠trial-submission channel by the end of Wednesday, June 11.
Don't worry if you're new to this—it's designed to be a starter task, and we’ll be around to answer questions!

## WriteUp

This was a fun little project to work on, given more time (midterms are coming up...) I would've definitely liked to try adding more features.

However I'm still happy with my results. To try out my final product you can again go to [this link](https://benz-sentiment-analysis.streamlit.app/) and play around with it. I originally wanted to just analyze trends on Elon Musk, especially given his recent headlines and actions, but in general it was super easy to just modify the URL to scrape any topic so I ended up letting the user choose the topic.

I also added a small feature so that the user can quickly change the number of pages they want to scrape. Afterwards the project will scrape the news and display the results in a few different ways. In general there will be 2 bar charts and a pie chart, each showing the distribution of sentiments. Unfortunately, for Elon Musk, he seemed to be largely negative.

One final small touch was logging and expandable sections for each article (datapoint), it has basically all the information I scraped along with the final sentiment scores from VADER. I also added a quick color to show the overall sentiment of the said article.

One feature I would've liked to add is a full scrape of the entire article and not just the title and short description provided. I did give it a quick short but given my time constraints I just decided to leave it as is. Another feature I think would be cool is maybe an integration with stocks and trying to correlate the sentiment of the news with the stock price. I did see someone else do something similar, however I again didn't have the time.