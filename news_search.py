import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime

class NewsSearchEngine:
    """
    TruthLens Reverse News Search & Verification Engine.
    Queries verified news feeds (Google News RSS & major reputable publishers)
    to find recent coverage, corroborate claims, and detect fact-check debunks.
    """

    # Recognized tier-1 news agencies and fact-checking organizations
    VERIFIED_PUBLISHERS = {
        "reuters", "associated press", "ap news", "bbc", "bbc news", "bloomberg",
        "the wall street journal", "wsj", "financial times", "ft", "the new york times",
        "the washington post", "cnbc", "cnn", "the guardian", "the hindu", "the indian express",
        "times of india", "hindustan times", "ndtv", "npr", "forbes", "snopes",
        "politifact", "factcheck.org", "afp", "al jazeera", "time", "usa today"
    }

    CLICKBAIT_STOPWORDS = {
        "shocking", "urgent", "breaking", "bombshell", "unbelievable", "secret",
        "revealed", "exposed", "must", "watch", "miracle", "warning", "panicking",
        "hiding", "leaked", "video", "alert", "exclusive", "truth", "viral",
        "don't", "miss", "you", "won't", "believe", "what", "happened", "next"
    }

    def __init__(self, timeout=7):
        self.timeout = timeout
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    def extract_search_query(self, headline, content, max_words=6):
        """
        Derives an optimal, clean search query from a headline or article body
        by stripping clickbait markers, numbers, punctuation, and non-essential tokens.
        """
        source_text = headline.strip() if headline and len(headline.strip()) > 5 else content.strip()
        if not source_text:
            return ""

        # Remove URLs and special characters
        clean = re.sub(r"https?://\S+|www\.\S+", "", source_text)
        clean = re.sub(r"[^\w\s]", " ", clean)

        tokens = clean.split()
        meaningful_tokens = []
        
        for t in tokens:
            t_lower = t.lower()
            if len(t) > 2 and t_lower not in self.CLICKBAIT_STOPWORDS:
                meaningful_tokens.append(t)
            if len(meaningful_tokens) >= max_words:
                break

        query = " ".join(meaningful_tokens)
        return query if query else source_text[:40]

    def search_verified_news(self, query, limit=6):
        """
        Executes a reverse search on Google News RSS for the given query.
        Returns a structured list of recent verified news items.
        """
        if not query or not query.strip():
            return {
                "query": "",
                "articles": [],
                "corroboration": self._empty_corroboration()
            }

        clean_query = query.strip()
        encoded_q = urllib.parse.quote(clean_query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_q}&hl=en-US&gl=US&ceid=US:en"

        articles = []
        try:
            req = urllib.request.Request(rss_url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                xml_data = response.read()
                root = ET.fromstring(xml_data)

                for item in root.findall(".//item")[:limit]:
                    title = item.find("title").text if item.find("title") is not None else "Untitled"
                    link = item.find("link").text if item.find("link") is not None else "#"
                    pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""
                    source_elem = item.find("source")
                    source_name = source_elem.text if source_elem is not None and source_elem.text else "News Source"

                    # Normalize title by removing publisher suffix if present (e.g. "Title - Source")
                    clean_title = re.sub(rf"\s*-\s*{re.escape(source_name)}$", "", title, flags=re.I).strip()

                    # Check if publisher is in verified registry
                    is_verified = any(v in source_name.lower() for v in self.VERIFIED_PUBLISHERS)

                    # Detect if this article is a fact check or debunking piece
                    combined_lower = f"{clean_title} {source_name}".lower()
                    is_fact_check = bool(re.search(r"\b(fact[\s-]check|debunk|hoax|fake|scam\s+warning|falsely\s+claims?|bogus)\b", combined_lower))

                    articles.append({
                        "title": clean_title,
                        "source": source_name,
                        "link": link,
                        "pub_date": pub_date,
                        "is_verified": is_verified,
                        "is_fact_check": is_fact_check
                    })

        except Exception as e:
            # Graceful network fallback
            return {
                "query": clean_query,
                "articles": [],
                "error": str(e),
                "corroboration": self._empty_corroboration("Network timeout or news feed unreachable.")
            }

        corroboration = self._assess_corroboration(clean_query, articles)

        return {
            "query": clean_query,
            "articles": articles,
            "corroboration": corroboration
        }

    def _assess_corroboration(self, query, articles):
        """
        Assesses whether the claim has verified mainstream coverage,
        has been debunked by fact-checkers, or has zero credible reporting.
        """
        if not articles:
            return {
                "status": "UNVERIFIED",
                "badge": "No Corroborating Coverage Found",
                "badge_class": "warning",
                "score_impact": 10.0, # Slight suspicion penalty for uncorroborated sensational claims
                "summary": f"No recent reports from credible news organizations match '{query}'. Unverified claims should be treated with caution."
            }

        verified_count = sum(1 for a in articles if a["is_verified"])
        fact_check_count = sum(1 for a in articles if a["is_fact_check"])

        if fact_check_count > 0:
            return {
                "status": "DEBUNK_WARNING",
                "badge": "Fact-Check / Scam Warning Found",
                "badge_class": "danger",
                "score_impact": 35.0, # Increases fraud suspicion significantly
                "summary": f"Credible news outlets or fact-checkers have published warnings, debunks, or scam alerts regarding '{query}'."
            }
        elif verified_count >= 2:
            return {
                "status": "VERIFIED_COVERAGE",
                "badge": f"{verified_count} Verified Outlets Reporting",
                "badge_class": "success",
                "score_impact": -20.0, # Decreases fraud risk due to credible mainstream consensus
                "summary": f"Multiple established journalistic organizations (e.g. {articles[0]['source']}) are actively reporting on this subject."
            }
        else:
            return {
                "status": "LIMITED_COVERAGE",
                "badge": "Limited Coverage Found",
                "badge_class": "warning",
                "score_impact": 0.0,
                "summary": f"Found {len(articles)} related article(s), but limited coverage from top-tier verified news wires."
            }

    def _empty_corroboration(self, reason=None):
        return {
            "status": "NONE",
            "badge": "No Search Executed",
            "badge_class": "muted",
            "score_impact": 0.0,
            "summary": reason or "No query provided for reverse news verification."
        }
