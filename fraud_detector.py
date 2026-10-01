import re
import json
import joblib
import numpy as np

class FraudDetector:
    """
    TruthLens Core Fraud & Deception Forensic Analysis Engine.
    Combines de-biased Machine Learning models (Logistic Regression, Naive Bayes,
    Passive-Aggressive) with heuristic forensic detectors for financial fraud,
    phishing, sensationalism, urgency, and Explainable AI (XAI) feature attribution.
    """

    def __init__(self,
                 model_path="model.pkl",
                 nb_path="nb_model.pkl",
                 pac_path="pac_model.pkl",
                 vectorizer_path="vectorizer.pkl",
                 metrics_path="model_metrics.json"):
        
        # Load models and vectorizer
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        
        try:
            self.nb_model = joblib.load(nb_path)
        except Exception:
            self.nb_model = None

        try:
            self.pac_model = joblib.load(pac_path)
        except Exception:
            self.pac_model = None

        try:
            with open(metrics_path, "r") as f:
                self.metrics = json.load(f)
        except Exception:
            self.metrics = {}

        # Feature vocabulary & LR coefficients for XAI
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())
        self.coef = self.model.coef_[0]

        # -------------------------------------------------------------
        # Forensic Pattern Lexicons
        # -------------------------------------------------------------
        self.financial_scam_patterns = [
            (r"\b(guaranteed\s+(?:return|profit|income|yield|gain)s?)\b", "Guaranteed Investment Return Claim"),
            (r"\b(risk[\s-]free\s+investment)\b", "Risk-Free Investment Claim"),
            (r"\b(turn\s+\$?\d+\s+into\s+\$?\d+)\b", "Unrealistic Wealth Multiplication"),
            (r"\b(double\s+your\s+money\s+(?:in|overnight|instantly|fast))\b", "Money Doubling Scheme"),
            (r"\b(passive\s+income\s+(?:secret|glitch|system|loophole))\b", "Passive Income Secret Scheme"),
            (r"\b(bitcoin|crypto|usdt|eth)\s+(?:giveaway|presale\s+bonus|multiplier|loophole)\b", "Cryptocurrency Giveaway/Multiplier Scam"),
            (r"\b(send\s+\d+(\.\d+)?\s*(?:btc|eth|crypto)\s+get\s+\d+)\b", "Classic Crypto Doubler Formula"),
            (r"\b(secret\s+algorithm\s+makes\s+thousands)\b", "Bogus Automated Trading Bot"),
            (r"\b(wire\s+transfer|western\s+union|moneygram)\s+immediately\b", "High-Risk Wire Demand"),
            (r"\b(buy\s+(?:itunes|google\s+play|amazon|steam)\s+gift\s+cards?)\b", "Gift Card Payment Demand"),
            (r"\b(unclaimed\s+(?:inheritance|funds|assets|millions))\b", "Inheritance / Advance-Fee Fraud"),
            (r"\b(you\s+(?:have\s+won|are\s+selected\s+for)\s+(?:lottery|jackpot|prize))\b", "Bogus Lottery / Prize Award"),
            (r"\b(government\s+(?:grant|relief|stimulus)\s+direct\s+deposit\s+claim\s+now)\b", "Government Grant Scam"),
            (r"\b(pump\s+and\s+dump|1000x\s+gem|moonshot\s+guaranteed)\b", "Token Speculation / Pump Signal"),
            (r"\b(secret\s+formula\s+banks\s+don['']?t\s+want\s+you\s+to\s+know)\b", "Anti-Institutional Scam Hook")
        ]

        self.phishing_impersonation_patterns = [
            (r"\b(urgent\s+(?:notice|alert)\s+from\s+(?:irs|tax\s+department|fbi|police|rbi))\b", "Law Enforcement / Regulatory Impersonation"),
            (r"\b(bank\s+account\s+(?:suspended|frozen|restricted|locked))\b", "Bank Account Suspension Threat"),
            (r"\b(immediate\s+(?:action|verification)\s+required\s+to\s+prevent\s+closure)\b", "Account Closure Pressure"),
            (r"\b(unauthorized\s+transaction\s+detected\s+click\s+here)\b", "Phishing Link / False Charge Alarm"),
            (r"\b(verify\s+your\s+(?:identity|password|ssn|otp|pin)\s+now)\b", "Credential Harvesting Trigger")
        ]

        self.urgency_keywords = [
            "urgent", "act now", "immediate", "immediately", "hurry", "last chance",
            "expires today", "deadline", "limited time", "don't miss out", "before it's deleted",
            "take action now", "running out", "countdown", "right now", "emergency"
        ]

        self.clickbait_hyperbole = [
            "shocking", "unbelievable", "mind-blowing", "miracle", "secret revealed",
            "you won't believe", "what happened next", "doctors hate", "insiders stunned",
            "explosive truth", "bombshell", "leaked", "conspiracy exposed", "warning:",
            "they are hiding this", "jaw-dropping", "must watch"
        ]

        self.credibility_cues = [
            "according to", "spokesperson", "in a statement", "press release",
            "official report", "reuters", "associated press", "study published",
            "researchers found", "data indicates", "confirmed by", "testified",
            "regulatory filing", "ministry of", "department said"
        ]

    def clean_text_for_nlp(self, text):
        """Standard NLP normalization matching vectorizer training."""
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
        text = re.sub(r"<.*?>", " ", text)
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def analyze_sensationalism(self, headline, text):
        """Computes sensationalism, uppercase density, and clickbait hyperbole."""
        full_raw = f"{headline or ''} {text or ''}".strip()
        if not full_raw:
            return {"score": 0, "caps_ratio": 0, "punctuation_density": 0, "clickbait_hits": []}

        # Uppercase ratio (excluding single characters)
        words = full_raw.split()
        caps_words = [w for w in words if len(w) > 1 and w.isupper()]
        caps_ratio = len(caps_words) / max(len(words), 1)

        # Exclamation and question mark density
        exclamations = full_raw.count("!")
        questions = full_raw.count("?")
        excessive_marks = len(re.findall(r"[!?]{2,}", full_raw))

        # Clickbait triggers
        lower_raw = full_raw.lower()
        clickbait_hits = [phrase for phrase in self.clickbait_hyperbole if phrase in lower_raw]

        # Calculate composite sensationalism score (0 to 100)
        score = 0
        score += min(caps_ratio * 150, 35)
        score += min(exclamations * 5, 25)
        score += min(excessive_marks * 10, 20)
        score += min(len(clickbait_hits) * 12, 35)

        return {
            "score": round(min(score, 100), 1),
            "caps_ratio_pct": round(caps_ratio * 100, 1),
            "exclamation_count": exclamations,
            "clickbait_hits": clickbait_hits[:5]
        }

    def analyze_urgency(self, text):
        """Measures urgency and artificial time-pressure cues."""
        if not text:
            return {"score": 0, "urgency_hits": []}

        lower_text = text.lower()
        urgency_hits = [kw for kw in self.urgency_keywords if re.search(rf"\b{re.escape(kw)}\b", lower_text)]

        # Score based on number of distinct urgency cues
        score = min(len(urgency_hits) * 22, 100)
        return {
            "score": round(score, 1),
            "urgency_hits": urgency_hits[:6]
        }

    def detect_fraud_and_scams(self, text):
        """Scans for known fraud formulas, financial scams, and phishing cues."""
        if not text:
            return {"scam_score": 0, "detected_patterns": []}

        detected = []
        for pattern, label in self.financial_scam_patterns:
            matches = re.findall(pattern, text, flags=re.I)
            if matches:
                detected.append({"category": "Financial / Investment Scam", "label": label})

        for pattern, label in self.phishing_impersonation_patterns:
            matches = re.findall(pattern, text, flags=re.I)
            if matches:
                detected.append({"category": "Phishing / Impersonation", "label": label})

        scam_score = min(len(detected) * 35, 100)
        return {
            "scam_score": scam_score,
            "detected_patterns": detected
        }

    def analyze_credibility_signals(self, text):
        """Analyzes formal journalistic markers, direct quotes, and named attributions."""
        if not text:
            return {"score": 0, "quote_count": 0, "attribution_hits": []}

        quote_count = len(re.findall(r'["“][^"“”]{5,300}["”]', text))
        lower_text = text.lower()
        attribution_hits = [cue for cue in self.credibility_cues if cue in lower_text]

        # Credibility score (0 to 100)
        score = min(quote_count * 12 + len(attribution_hits) * 15, 100)
        return {
            "score": round(score, 1),
            "quote_count": quote_count,
            "attribution_hits": attribution_hits[:5]
        }

    def explain_terms(self, clean_content, top_k=6):
        """
        Explainable AI (XAI): Identifies tokens in the input text that strongly
        influenced the model towards Fake/Fraud (positive weights) vs Real (negative weights).
        """
        if not clean_content:
            return {"suspicious_terms": [], "credible_terms": []}

        # Transform single document
        tfidf_vec = self.vectorizer.transform([clean_content])
        nonzero_indices = tfidf_vec.nonzero()[1]

        if len(nonzero_indices) == 0:
            return {"suspicious_terms": [], "credible_terms": []}

        # Calculate word contributions: TF-IDF weight * Model coefficient
        contributions = []
        for idx in nonzero_indices:
            term = self.feature_names[idx]
            tfidf_val = tfidf_vec[0, idx]
            weight = self.coef[idx]
            impact = tfidf_val * weight
            contributions.append((term, impact, weight))

        # Positive impact pushes towards Fake (1), Negative impact pushes towards Real (0)
        suspicious = sorted([c for c in contributions if c[1] > 0.05], key=lambda x: x[1], reverse=True)[:top_k]
        credible = sorted([c for c in contributions if c[1] < -0.05], key=lambda x: x[1])[:top_k]

        return {
            "suspicious_terms": [{"word": term, "weight": round(float(imp), 3)} for term, imp, _ in suspicious],
            "credible_terms": [{"word": term, "weight": round(float(abs(imp)), 3)} for term, imp, _ in credible]
        }

    def analyze(self, headline, content):
        """
        Main orchestration entry point:
        Executes ML multi-model consensus, forensic heuristics, XAI term extraction,
        and computes the composite Fraud Risk Score.
        """
        raw_combined = f"{headline or ''} {content or ''}".strip()
        if not raw_combined:
            return {"error": "Headline or news content is required for analysis."}

        clean_content = self.clean_text_for_nlp(raw_combined)
        vec = self.vectorizer.transform([clean_content])

        # ---------------------------------------------------------
        # 1. Multi-Model ML Evaluation
        # ---------------------------------------------------------
        # Logistic Regression
        lr_pred = int(self.model.predict(vec)[0])
        lr_prob = self.model.predict_proba(vec)[0]
        lr_fake_prob = float(lr_prob[1])

        # Naive Bayes
        nb_pred = 0
        nb_fake_prob = 0.5
        if self.nb_model is not None:
            nb_pred = int(self.nb_model.predict(vec)[0])
            nb_prob = self.nb_model.predict_proba(vec)[0]
            nb_fake_prob = float(nb_prob[1])

        # Passive Aggressive
        pac_pred = 0
        pac_conf = 0.5
        if self.pac_model is not None:
            pac_pred = int(self.pac_model.predict(vec)[0])
            decision = float(self.pac_model.decision_function(vec)[0])
            # Sigmoid approximation of decision function for confidence
            pac_conf = float(1.0 / (1.0 + np.exp(-decision)))

        model_votes = [lr_pred, nb_pred, pac_pred]
        fake_votes = sum(model_votes)
        consensus_text = f"{fake_votes}/3 Models Flagged as Deceptive" if fake_votes > 0 else "All Models Agree: Credible"

        # ---------------------------------------------------------
        # 2. Forensic Analysis (Heuristics)
        # ---------------------------------------------------------
        sensationalism = self.analyze_sensationalism(headline, content)
        urgency = self.analyze_urgency(raw_combined)
        scams = self.detect_fraud_and_scams(raw_combined)
        credibility = self.analyze_credibility_signals(raw_combined)

        # ---------------------------------------------------------
        # 3. Composite Fraud Risk Calculation (0 - 100%)
        # ---------------------------------------------------------
        # Ensemble ML fake probability average
        ensemble_ml_prob = (lr_fake_prob * 0.5) + (nb_fake_prob * 0.25) + (pac_conf * 0.25)
        
        # Base risk from ML models
        base_risk = ensemble_ml_prob * 100.0

        # Forensic modifiers
        scam_bonus = (scams["scam_score"] / 100.0) * 35.0
        sensationalism_bonus = (sensationalism["score"] / 100.0) * 15.0
        urgency_bonus = (urgency["score"] / 100.0) * 10.0
        credibility_discount = (credibility["score"] / 100.0) * 15.0

        composite_risk = base_risk + scam_bonus + sensationalism_bonus + urgency_bonus - credibility_discount
        composite_risk = max(1.0, min(99.0, composite_risk))
        composite_risk = round(composite_risk, 1)

        # Risk Classification Level
        if composite_risk >= 65.0 or scams["scam_score"] >= 70:
            risk_level = "HIGH FRAUD RISK"
            risk_class = "danger"
            primary_verdict = "FAKE / FRAUD NEWS"
            summary = "High probability of deceptive or fraudulent news. Multiple manipulation or scam indicators were detected."
        elif composite_risk >= 35.0:
            risk_level = "MODERATE RISK"
            risk_class = "warning"
            primary_verdict = "SUSPICIOUS / UNVERIFIED"
            summary = "Article contains sensationalist tone, uncorroborated assertions, or mixed model signals. Requires independent verification."
        else:
            risk_level = "LOW FRAUD RISK"
            risk_class = "success"
            primary_verdict = "CREDIBLE NEWS"
            summary = "Text aligns with legitimate journalistic patterns with low sensationalism and no detected scam signatures."

        # ---------------------------------------------------------
        # 4. Explainable AI & Verification Advice
        # ---------------------------------------------------------
        xai = self.explain_terms(clean_content)

        # Actionable safety checklist
        advice = []
        if scams["detected_patterns"]:
            advice.append("CRITICAL: Do NOT wire money, provide card details, or click unverified links mentioned in this text.")
            advice.append("Check with official consumer regulatory agencies (SEC, FTC, or central bank advisories).")
        if sensationalism["score"] > 40:
            advice.append("Check whether established mainstream news agencies (Reuters, AP, BBC) report this story.")
        if urgency["score"] > 30:
            advice.append("Be wary of artificial deadlines; fraudulent actors use urgency to bypass critical thinking.")
        if credibility["quote_count"] == 0:
            advice.append("Notice that this text lacks direct attribution or verifiable quotes from recognized officials.")
        if not advice:
            advice.append("Maintain healthy skepticism by checking the source domain and publication date.")

        # Text statistics
        word_count = len(raw_combined.split())
        char_count = len(raw_combined)

        return {
            "fraud_risk_score": composite_risk,
            "risk_level": risk_level,
            "risk_class": risk_class,
            "primary_verdict": primary_verdict,
            "summary": summary,
            "models": {
                "consensus": consensus_text,
                "logistic_regression": {
                    "verdict": "Fake / Fraud" if lr_pred == 1 else "Real / Legitimate",
                    "fake_prob_pct": round(lr_fake_prob * 100, 1),
                    "confidence_pct": round(max(lr_prob) * 100, 1)
                },
                "naive_bayes": {
                    "verdict": "Fake / Fraud" if nb_pred == 1 else "Real / Legitimate",
                    "fake_prob_pct": round(nb_fake_prob * 100, 1)
                },
                "passive_aggressive": {
                    "verdict": "Fake / Fraud" if pac_pred == 1 else "Real / Legitimate",
                    "confidence_pct": round(pac_conf * 100, 1)
                }
            },
            "forensics": {
                "sensationalism": sensationalism,
                "urgency": urgency,
                "scams": scams,
                "credibility": credibility
            },
            "xai": xai,
            "advice": advice,
            "stats": {
                "word_count": word_count,
                "char_count": char_count
            }
        }
