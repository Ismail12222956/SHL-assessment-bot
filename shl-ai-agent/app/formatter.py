def format_recommendations(docs):

    recommendations = []

    for doc in docs:

        metadata = doc.metadata

        recommendations.append({
            "name": metadata.get("name", "Unknown"),
            "url": metadata.get("url", ""),
            "test_type": metadata.get("categories", "")
        })

    return recommendations[:10]