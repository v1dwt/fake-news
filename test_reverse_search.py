import json
from app import app
from news_search import NewsSearchEngine

def test_reverse_search():
    client = app.test_client()
    engine = NewsSearchEngine()

    print("=" * 65)
    print("TruthLens Reverse Search & Cross-Verification Test Suite")
    print("=" * 65)

    # 1. Test Query Extraction
    print("\n[Test 1] Query Extraction from Noisy Clickbait")
    raw_headline = "SHOCKING: Federal Reserve holds benchmark interest rates steady!"
    q = engine.extract_search_query(raw_headline, "")
    assert "shocking" not in q.lower()
    assert "federal" in q.lower() and "reserve" in q.lower()
    print(f"  -> PASS: Clean query extracted: '{q}'")

    # 2. Test Live Reverse Search with Legitimate Topic
    print("\n[Test 2] Live Reverse Search (Authentic Topic)")
    res = engine.search_verified_news("Federal Reserve interest rates", limit=4)
    assert len(res["articles"]) > 0, "Expected at least 1 article from live news search"
    first = res["articles"][0]
    assert "title" in first and "source" in first and "link" in first
    assert res["corroboration"]["status"] in ["VERIFIED_COVERAGE", "LIMITED_COVERAGE"]
    print(f"  -> PASS: Found {len(res['articles'])} articles. Corroboration: {res['corroboration']['badge']}")
    print(f"     Top source: [{first['source']}] {first['title']}")

    # 3. Test REST API: GET /api/reverse-search
    print("\n[Test 3] GET /api/reverse-search?q=inflation&limit=3")
    res = client.get("/api/reverse-search?q=inflation&limit=3")
    assert res.status_code == 200
    data = res.get_json()
    assert "articles" in data
    assert len(data["articles"]) <= 3
    print(f"  -> PASS: REST API returned {len(data['articles'])} articles.")

    # 4. Test REST API: POST /api/cross-verify
    print("\n[Test 4] POST /api/cross-verify")
    res = client.post("/api/cross-verify", json={
        "headline": "Central Bank monetary policy update",
        "content": "The central bank held rates steady today."
    })
    assert res.status_code == 200
    data = res.get_json()
    assert "extracted_query" in data
    assert "corroboration" in data
    print(f"  -> PASS: Extracted '{data['extracted_query']}'. Status: {data['corroboration']['status']}")

    # 5. Test Full Pipeline Integration: POST / with reverse news included
    print("\n[Test 5] POST / with Reverse Search Corroboration")
    res = client.post("/", data={
        "headline": "Federal Reserve Holds Benchmark Interest Rates",
        "content": "The Federal Reserve held benchmark interest rates steady on Wednesday following a monetary policy meeting."
    })
    assert res.status_code == 200
    assert b"Reverse News Search & Media Corroboration" in res.data
    assert b"Verified Outlets Reporting" in res.data or b"Coverage" in res.data
    print("  -> PASS: Web analysis seamlessly rendered verified media cards and corroboration badge.")

    print("\n" + "=" * 65)
    print("ALL REVERSE SEARCH TESTS PASSED SUCCESSFULLY! (5/5)")
    print("=" * 65)

if __name__ == "__main__":
    test_reverse_search()
