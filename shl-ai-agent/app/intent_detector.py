def detect_intent(query):

    query = query.lower()

    if "compare" in query or "difference" in query:
        return "compare"

    off_topic_words = [
        "salary",
        "politics",
        "religion",
        "law"
    ]

    if any(word in query for word in off_topic_words):
        return "off_topic"

    vague_queries = [
        "assessment",
        "test",
        "need assessment",
        "need test"
    ]

    if query.strip() in vague_queries:
        return "clarify"

    refinement_words = [
        "add",
        "remove",
        "replace",
        "instead"
    ]

    if any(word in query for word in refinement_words):
        return "refine"

    return "recommend"