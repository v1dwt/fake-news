import os
import json
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from fraud_detector import FraudDetector
from news_search import NewsSearchEngine

app = Flask(__name__)
app.config["SECRET_KEY"] = "truthlens-fraud-detection-secret-key"

# Initialize Core Engines
detector = FraudDetector()
search_engine = NewsSearchEngine()

# In-memory history cache for interactive dashboard demonstration
analysis_history = []

# Pre-configured sample articles for 1-click viva/demo testing
SAMPLE_ARTICLES = {
    "legitimate": {
        "title": "Legitimate News (Central Bank Policy)",
        "badge": "Authentic / Real",
        "headline": "Federal Reserve Holds Benchmark Interest Rates Steady at Policy Meeting",
        "content": (
            "The Federal Reserve held benchmark interest rates steady on Wednesday following a two-day "
            "monetary policy meeting, citing balanced progress toward employment and inflation objectives. "
            "In an official statement, the Federal Open Market Committee noted that economic activity has continued "
            "to expand at a solid pace. Job gains have remained strong, and the unemployment rate has stayed low. "
            "\"The Committee does not expect it will be appropriate to reduce the target range until it has gained "
            "greater confidence that inflation is moving sustainably toward 2 percent,\" the central bank said in a press release. "
            "Financial markets responded with moderate gains as treasury yields remained stable across major indices."
        )
    },
    "crypto_scam": {
        "title": "Financial / Crypto Investment Scam",
        "badge": "High Fraud Risk",
        "headline": "BREAKING: Secret Bitcoin Trading Loophole Guarantees 500% Profit Overnight!",
        "content": (
            "URGENT ALERT: You won't believe this shocking wealth secret! An anonymous software engineer has leaked a "
            "secret algorithm that guarantees 500% profit in just 24 hours. Turn $100 into $10,000 with this 100% risk-free "
            "passive income loophole! Banks and Wall Street elites are desperately trying to shut this down. "
            "Special bonus: Send 0.1 BTC to the smart contract address and receive double your money back instantly! "
            "Act now, limited spots available! Wire money immediately before the registration link is permanently deleted!"
        )
    },
    "sensational_clickbait": {
        "title": "Sensational Clickbait Hoax",
        "badge": "Fake / Clickbait",
        "headline": "SHOCKING TRUTH EXPOSED: Doctors Stunned by Miracle Ancient Remedy They Kept Secret!",
        "content": (
            "BOMBSHELL REVELATION! You won't believe what top scientists just discovered! Big Pharma has been hiding a simple "
            "kitchen ingredient that eliminates all chronic diseases overnight! Insiders are panicking as secret documents leaked "
            "online. Watch this explosive video before it gets taken down by corrupt officials! Every American must act now "
            "to see the hidden conspiracy that the mainstream media refuses to report! Don't miss out on this jaw-dropping truth!"
        )
    },
    "phishing_alert": {
        "title": "Urgent Impersonation & Phishing Hoax",
        "badge": "Phishing / Impersonation",
        "headline": "URGENT NOTICE: Tax Department Demands Immediate Identity Verification to Prevent Account Suspension",
        "content": (
            "FINAL WARNING: Your federal tax refund of $4,850 is waiting direct deposit approval. Urgent notice from the IRS: "
            "unauthorized transaction detected on your file. Immediate verification required to prevent bank account suspension and "
            "legal prosecution. You must verify your social security number, password, and credit card PIN right now. "
            "Click the link below or purchase Google Play gift cards to clear pending verification penalties before midnight deadline!"
        )
    }
}

@app.route("/", methods=["GET", "POST"])
def home():
    headline = ""
    content = ""
    result = None
    reverse_results = None
    search_query = ""

    # Handle sample query parameter (e.g. ?sample=crypto_scam)
    sample_key = request.args.get("sample")
    if sample_key and sample_key in SAMPLE_ARTICLES:
        sample = SAMPLE_ARTICLES[sample_key]
        headline = sample["headline"]
        content = sample["content"]

    # Handle direct reverse search query parameter (e.g. ?q=interest+rates)
    direct_q = request.args.get("q", "").strip()
    if direct_q:
        search_query = direct_q
        reverse_results = search_engine.search_verified_news(direct_q, limit=6)

    # Handle form submission for full analysis
    if request.method == "POST":
        headline = request.form.get("headline", "").strip()
        content = request.form.get("news", "").strip() or request.form.get("content", "").strip()
        custom_q = request.form.get("search_query", "").strip()

        if custom_q:
            # User submitted a standalone reverse search query
            search_query = custom_q
            reverse_results = search_engine.search_verified_news(custom_q, limit=6)

        elif headline or content:
            t0 = time.time()
            result = detector.analyze(headline, content)
            
            # Execute automated Reverse News Search for cross-verification
            extracted_query = search_engine.extract_search_query(headline, content)
            search_query = extracted_query
            reverse_results = search_engine.search_verified_news(extracted_query, limit=5)

            # Integrate corroboration score impact
            corroboration = reverse_results.get("corroboration", {})
            impact = corroboration.get("score_impact", 0.0)

            # Apply corroboration modifier
            adjusted_risk = max(1.0, min(99.0, result["fraud_risk_score"] + impact))
            result["fraud_risk_score"] = round(adjusted_risk, 1)

            # Update verdict if corroboration shifts category
            if result["fraud_risk_score"] >= 65.0 or corroboration.get("status") == "DEBUNK_WARNING":
                result["risk_level"] = "HIGH FRAUD RISK"
                result["risk_class"] = "danger"
                result["primary_verdict"] = "FAKE / FRAUD NEWS"
            elif result["fraud_risk_score"] >= 35.0:
                result["risk_level"] = "MODERATE RISK"
                result["risk_class"] = "warning"
                result["primary_verdict"] = "SUSPICIOUS / UNVERIFIED"
            else:
                result["risk_level"] = "LOW FRAUD RISK"
                result["risk_class"] = "success"
                result["primary_verdict"] = "CREDIBLE NEWS"

            elapsed_ms = round((time.time() - t0) * 1000, 1)
            result["elapsed_ms"] = elapsed_ms
            result["reverse_news"] = reverse_results

            # Append to history
            entry = {
                "id": len(analysis_history) + 1,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "headline": headline or (content[:60] + "..."),
                "risk_score": result["fraud_risk_score"],
                "risk_level": result["risk_level"],
                "risk_class": result["risk_class"],
                "primary_verdict": result["primary_verdict"],
                "corroboration_badge": corroboration.get("badge", "N/A"),
                "elapsed_ms": elapsed_ms
            }
            analysis_history.insert(0, entry)
            if len(analysis_history) > 25:
                analysis_history.pop()

    return render_template(
        "index.html",
        headline=headline,
        content=content,
        result=result,
        reverse_results=reverse_results,
        search_query=search_query,
        samples=SAMPLE_ARTICLES,
        history=analysis_history[:10],
        metrics=detector.metrics
    )

# -------------------------------------------------------------
# REST API Endpoints (Conforms to IEEE SRS FR-1.3 & FR-3.2)
# -------------------------------------------------------------

@app.route("/api/analyze", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def api_analyze():
    """
    REST API for programmatic news fraud & credibility detection.
    Accepts JSON body: {"headline": "...", "content": "..."} or {"news": "..."}
    """
    data = request.get_json(silent=True) or request.form
    if not data:
        return jsonify({"error": "Invalid request. JSON body or form data required."}), 400

    headline = data.get("headline", "").strip()
    content = data.get("content", "").strip() or data.get("news", "").strip()

    if not headline and not content:
        return jsonify({"error": "Either 'headline' or 'content' must be provided."}), 400

    t0 = time.time()
    result = detector.analyze(headline, content)
    
    # Execute Reverse News Search
    query = search_engine.extract_search_query(headline, content)
    reverse_results = search_engine.search_verified_news(query, limit=5)
    
    corroboration = reverse_results.get("corroboration", {})
    impact = corroboration.get("score_impact", 0.0)
    result["fraud_risk_score"] = round(max(1.0, min(99.0, result["fraud_risk_score"] + impact)), 1)
    result["reverse_news"] = reverse_results
    result["elapsed_ms"] = round((time.time() - t0) * 1000, 1)

    return jsonify(result)

@app.route("/api/reverse-search", methods=["GET"])
def api_reverse_search():
    """
    REST API for Reverse News Search.
    Query parameters: ?q=search+terms&limit=6
    """
    query = request.args.get("q", "").strip()
    limit = int(request.args.get("limit", 6))
    if not query:
        return jsonify({"error": "Query parameter 'q' is required."}), 400

    results = search_engine.search_verified_news(query, limit=limit)
    return jsonify(results)

@app.route("/api/cross-verify", methods=["POST"])
def api_cross_verify():
    """
    REST API for claim cross-verification against verified news coverage.
    """
    data = request.get_json(silent=True) or request.form
    headline = data.get("headline", "")
    content = data.get("content", "")
    query = search_engine.extract_search_query(headline, content)
    results = search_engine.search_verified_news(query, limit=6)
    return jsonify({
        "extracted_query": query,
        "corroboration": results.get("corroboration"),
        "articles": results.get("articles")
    })

@app.route("/api/samples", methods=["GET"])
def api_samples():
    """Returns pre-configured sample news articles."""
    return jsonify(SAMPLE_ARTICLES)

@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Returns model training performance metrics and confusion matrices."""
    return jsonify(detector.metrics)

@app.route("/api/history", methods=["GET"])
def api_history():
    """Returns recent analysis scan history."""
    return jsonify(analysis_history)

@app.route("/api/history/clear", methods=["POST"])
def api_clear_history():
    """Clears scan history."""
    analysis_history.clear()
    return jsonify({"status": "success", "message": "History cleared."})

if __name__ == "__main__":
    app.run(debug=True, port=5000)