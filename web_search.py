def search_web(query: str) -> list:
    print(f"Searching the web for: {query}")
    dummy_results = [
        {
            "title": "Dummy Result 1",
            "snippet": "This is a simulated web search result snippet.",
            "url": "http://example.com/dummy1"
        },
        {
            "title": "Dummy Result 2",
            "snippet": "Another snippet of a simulated search result.",
            "url": "http://example.com/dummy2"
        }
    ]
    return dummy_results