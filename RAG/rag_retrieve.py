# rag_retrieve.py
from RAG.rag_setup import embedder, collection

def retrieve_evidence(query: str, top_k: int = 5) -> list[dict]:
    query_embedding = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    evidence = []
    for i in range(len(results["ids"][0])):
        similarity = 1 - results["distances"][0][i]
        
        # Debug: in ra để xem similarity thực tế đang ở mức nào
        print(f"  [DEBUG] chunk {i}: similarity = {similarity:.3f} | {results['documents'][0][i][:60]}...")
        
        if similarity < 0.3:  # ✅ Hạ từ 0.55 xuống 0.3
            continue

        evidence.append({
            "content": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "url": results["metadatas"][0][i]["url"],
            "title": results["metadatas"][0][i]["title"],
            "score": round(similarity, 3)
        })

    return evidence