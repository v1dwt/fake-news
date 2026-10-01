import json
from app import app

def test_truthlens():
    client = app.test_client()

    print("=" * 60)
    print("TruthLens Test Suite: Web UI & REST API Verification")
    print("=" * 60)

    # 1. Test GET /
    print("\n[Test 1] GET / (Home page)")
    res = client.get("/")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert b"TruthLens" in res.data
    assert b"Forensic News & Fraud Credibility Analyzer" in res.data
    print("  -> PASS: Home page loaded successfully.")

    # 2. Test GET /?sample=crypto_scam
    print("\n[Test 2] GET /?sample=crypto_scam (Sample pre-loading)")
    res = client.get("/?sample=crypto_scam")
    assert res.status_code == 200
    assert b"Secret Bitcoin Trading Loophole" in res.data
    print("  -> PASS: Pre-configured sample loaded into form.")

    # 3. Test POST / (Analyze Crypto Scam)
    print("\n[Test 3] POST / with High Fraud Scam")
    res = client.post("/", data={
        "headline": "Secret Crypto Loophole 500% Profit",
        "content": "Turn $100 into $10,000 overnight with guaranteed profit! Act now! Wire money immediately to our bitcoin wallet!"
    })
    assert res.status_code == 200
    assert b"HIGH FRAUD RISK" in res.data
    assert b"Financial / Investment Scam" in res.data or b"Guaranteed Investment Return Claim" in res.data
    assert b"Explainable AI (XAI)" in res.data
    print("  -> PASS: Correctly classified scam as HIGH FRAUD RISK with XAI badges.")

    # 4. Test POST / (Analyze Legitimate News without Reuters tag)
    print("\n[Test 4] POST / with Legitimate Real News")
    res = client.post("/", data={
        "headline": "Central Bank Holds Benchmark Interest Rates",
        "content": "The central bank held interest rates unchanged at its meeting on Wednesday, noting solid economic growth and balanced employment data, according to an official statement."
    })
    assert res.status_code == 200
    assert b"LOW FRAUD RISK" in res.data or b"CREDIBLE NEWS" in res.data
    print("  -> PASS: Correctly classified real news as LOW FRAUD RISK / CREDIBLE.")

    # 5. Test REST API: POST /api/analyze
    print("\n[Test 5] POST /api/analyze (JSON REST API)")
    payload = {
        "headline": "URGENT TAX ALERT",
        "content": "Urgent notice from IRS: immediate verification required or bank account suspended! Buy gift cards now to avoid arrest!"
    }
    res = client.post("/api/analyze", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert "fraud_risk_score" in data
    assert "risk_level" in data
    assert "models" in data
    assert "xai" in data
    print(f"  -> PASS: API response returned Fraud Risk Score: {data['fraud_risk_score']}%, Level: {data['risk_level']}")
    print(f"     Consensus: {data['models']['consensus']}")

    # 6. Test GET /api/samples
    print("\n[Test 6] GET /api/samples")
    res = client.get("/api/samples")
    assert res.status_code == 200
    samples = res.get_json()
    assert len(samples) >= 4
    print(f"  -> PASS: Returned {len(samples)} pre-configured test samples.")

    # 7. Test GET /api/metrics
    print("\n[Test 7] GET /api/metrics")
    res = client.get("/api/metrics")
    assert res.status_code == 200
    metrics = res.get_json()
    assert "LogisticRegression" in metrics
    print(f"  -> PASS: Returned model metrics (Logistic Regression Acc: {metrics['LogisticRegression']['accuracy']*100}%).")

    # 8. Test GET /api/history
    print("\n[Test 8] GET /api/history")
    res = client.get("/api/history")
    assert res.status_code == 200
    history = res.get_json()
    assert len(history) >= 2
    print(f"  -> PASS: Returned scan history with {len(history)} entries.")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! (8/8)")
    print("=" * 60)

if __name__ == "__main__":
    test_truthlens()
