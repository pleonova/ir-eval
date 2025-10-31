#!/usr/bin/env python3
"""
Demo script for exploring Jina Embeddings v4 functionality.

This script demonstrates:
1. Loading and using the Jina embeddings model
2. Encoding queries and passages with proper prompts
3. Retrieval performance on a sample corpus
4. Comparing with DummyEmbedder baseline
5. Inspecting embedding properties (normalization, similarity)

Run with: python demo_jina_embeddings.py

Note: First run will download the model (~12GB) which may take some time.
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import from retrievers
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from retrievers.embeddings import JinaEmbedder, DummyEmbedder, EmbeddingRetriever


def main():
    print("=" * 80)
    print("JINA EMBEDDINGS v4 - FUNCTIONALITY DEMO")
    print("=" * 80)
    
    # Sample corpus for testing
    corpus = [
        {
            "doc_id": "climate1",
            "text": "Climate change is causing rising sea levels and extreme weather events, "
                   "threatening coastal cities worldwide."
        },
        {
            "doc_id": "climate2", 
            "text": "Renewable energy sources like solar and wind power are essential for "
                   "reducing greenhouse gas emissions."
        },
        {
            "doc_id": "ml1",
            "text": "Machine learning algorithms can identify patterns in large datasets and "
                   "make predictions based on training data."
        },
        {
            "doc_id": "ml2",
            "text": "Deep neural networks use multiple layers to learn hierarchical "
                   "representations of data for complex tasks."
        },
        {
            "doc_id": "space1",
            "text": "The James Webb Space Telescope is revealing unprecedented details about "
                   "distant galaxies and the early universe."
        }
    ]
    
    queries = [
        "What are the effects of climate change?",
        "How does machine learning work?",
        "Tell me about space exploration"
    ]
    
    print("\n" + "─" * 80)
    print("📚 CORPUS OVERVIEW")
    print("─" * 80)
    for doc in corpus:
        print(f"\n{doc['doc_id']:12} | {doc['text'][:70]}...")
    
    # ========================================================================
    # Test 1: Jina Embeddings
    # ========================================================================
    print("\n" + "=" * 80)
    print("🚀 TEST 1: JINA EMBEDDINGS")
    print("=" * 80)
    
    jina_embedder = None
    try:
        print("\n📥 Loading Jina embeddings model (this may take a while on first run)...")
        jina_embedder = JinaEmbedder(model_name="jinaai/jina-embeddings-v4", task="retrieval")
        print("✓ Model loaded successfully!")
        
        # Test encoding
        print("\n🔍 Testing encoding...")
        sample_texts = ["hello world", "climate change"]
        embeddings = jina_embedder.encode(sample_texts, prompt_name="passage")
        
        print(f"✓ Encoded {len(sample_texts)} texts")
        print(f"  Embedding shape: {embeddings.shape}")
        print(f"  Embedding dtype: {embeddings.dtype}")
        print(f"  Embedding dimension: {embeddings.shape[1]}")
        
        # Check if normalized
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"\n📊 Embedding norms (should be ~1.0 if normalized):")
        for i, norm in enumerate(norms):
            print(f"  Text {i+1}: {norm:.6f}")
        
        is_normalized = np.allclose(norms, 1.0, rtol=1e-3)
        print(f"  → Embeddings are {'✓ NORMALIZED' if is_normalized else '✗ NOT NORMALIZED'}")
        
        # Test retrieval
        print("\n🔎 Testing retrieval with Jina embeddings...")
        jina_retriever = EmbeddingRetriever(embedder=jina_embedder)
        jina_retriever.fit(corpus)
        
        for query in queries:
            print(f"\n  Query: '{query}'")
            results = jina_retriever.rank(query, k=3)
            print(f"  Top 3 results:")
            for rank, (doc_id, score) in enumerate(results, 1):
                print(f"    {rank}. {doc_id:12} | Score: {score:.4f}")
        
    except Exception as e:
        print(f"\n❌ Error loading Jina model: {e}")
        print("   This is expected if the model hasn't been downloaded or dependencies are missing.")
        print("   Install all dependencies with: pip install -r requirements.txt")
        print("   Or install manually: pip install sentence-transformers torch torchvision transformers peft")
    
    # ========================================================================
    # Test 2: DummyEmbedder (for comparison)
    # ========================================================================
    print("\n" + "=" * 80)
    print("🎲 TEST 2: DUMMY EMBEDDINGS (for comparison)")
    print("=" * 80)
    
    print("\n📊 Testing DummyEmbedder (deterministic normalized random embeddings)...")
    dummy_embedder = DummyEmbedder()
    dummy_retriever = EmbeddingRetriever(embedder=dummy_embedder)
    dummy_retriever.fit(corpus)
    
    for query in queries:
        print(f"\n  Query: '{query}'")
        results = dummy_retriever.rank(query, k=3)
        print(f"  Top 3 results:")
        for rank, (doc_id, score) in enumerate(results, 1):
            print(f"    {rank}. {doc_id:12} | Score: {score:.4f}")
    
    # ========================================================================
    # Test 3: Embedding similarity inspection
    # ========================================================================
    print("\n" + "=" * 80)
    print("🔬 TEST 3: SIMILARITY INSPECTION")
    print("=" * 80)
    
    if jina_embedder:
        print("\n📊 Computing pairwise similarities between sample texts...")
        sample_pairs = [
            ("climate change impacts", "global warming effects"),
            ("machine learning", "deep learning neural networks"),
            ("climate change", "neural networks"),
        ]
        
        for text1, text2 in sample_pairs:
            emb1 = jina_embedder.encode([text1], prompt_name="passage")[0]
            emb2 = jina_embedder.encode([text2], prompt_name="passage")[0]
            
            # Since normalized, dot product = cosine similarity
            similarity = np.dot(emb1, emb2)
            
            print(f"\n  '{text1}' <-> '{text2}'")
            print(f"  Similarity: {similarity:.4f}")
            
            if similarity > 0.7:
                print(f"  → High similarity (related concepts)")
            elif similarity > 0.4:
                print(f"  → Moderate similarity")
            else:
                print(f"  → Low similarity (unrelated concepts)")
    else:
        print("\n⚠️  Skipping similarity inspection (Jina model not loaded)")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ DEMO COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("• Jina embeddings v4 produces 2048-dimensional vectors by default")
    print("• Embeddings are pre-normalized (unit vectors)")
    print("• For normalized embeddings: dot product = cosine similarity")
    print("• Proper query/passage prompts improve retrieval quality")
    print("• DummyEmbedder provides fast, deterministic normalized embeddings for testing")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

