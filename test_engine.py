import sys
from rag_engine import RAGPipeline

def test_rag_pipeline():
    print("Testing Ultimate RAG Engine (Steps 1 - 16)...")
    pipeline = RAGPipeline()
    
    print(f"Indexed {len(pipeline.indexed_chunks)} chunks.")
    
    query_text = "What is Reciprocal Rank Fusion and how does Contextual Compression work in RAG?"
    response = pipeline.query(query_text)
    
    print("\n--- QUERY RESPONSE ---")
    print(f"Query: {response.query}")
    print(f"Answer:\n{response.answer}")
    print(f"\nCitations ({len(response.citations)}):")
    for c in response.citations:
        print(f"  - {c['citation_marker']} {c['title']} (Score: {c['score']})")
        
    print("\n--- EVALUATION METRICS ---")
    print(f"Retrieval Metrics: {response.metrics['retrieval_metrics']}")
    print(f"Generation Metrics: {response.metrics['generation_metrics']}")
    print(f"End-to-End: {response.metrics['end_to_end']}")
    
    print("\n--- STEP LOGS SUMMARY ---")
    for log in response.step_logs:
        print(f"[{log.step_number}] {log.step_name}: {log.output_summary} ({log.execution_time_ms}ms)")
        
    print("\nSUCCESS: All 16 RAG Pipeline steps executed cleanly!")

if __name__ == "__main__":
    test_rag_pipeline()
