#!/usr/bin/env python3
"""
Demo script comparing exponential vs linear DCG formulas.
Illustrates when each formula matters most.

Run with: python demos/demo_dcg_comparison.py
or from demos folder: python demo_dcg_comparison.py
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import metrics
sys.path.insert(0, str(Path(__file__).parent.parent))

import math
from metrics import dcg_at_k, ndcg_at_k

print("=" * 70)
print("DCG Formula Comparison: Exponential vs Linear")
print("=" * 70)

# Show position discount factors
print("\n📍 Position Discount Factors (same for both formulas)")
print("-" * 70)
print("Position | Discount Factor (1/log2(pos+1)) | % of Position 1")
print("-" * 70)
for pos in range(1, 6):
    discount = 1.0 / math.log2(pos + 1)
    pct = discount * 100
    print(f"   {pos}     |          {discount:.4f}             |      {pct:.1f}%")
print("\n✅ Higher positions have dramatically lower impact (logarithmic decay)")

# Example 0: How both parts work together
print("\n📊 Example 0: How Gain and Discount Work Together")
print("-" * 70)
print("Document with rel=3 at position 2 (i=1):")
print()
print("Linear DCG:")
print("  relevance_gain = 3 (just the relevance score)")
print("  position_discount = log2(1+2) = log2(3) = 1.585")
print("  contribution = 3 / 1.585 = 1.89")
print()
print("Exponential DCG:")
print("  relevance_gain = 2^3 - 1 = 7 (exponential boost!)")
print("  position_discount = log2(1+2) = log2(3) = 1.585 (same!)")
print("  contribution = 7 / 1.585 = 4.42")
print()
print("✅ Exponential gives 2.3x more weight (4.42 vs 1.89) due to gain (7 vs 3)")
print("   Position discount is IDENTICAL in both formulas!")

# Example 1: Graded relevance - shows the difference clearly
print("\n📊 Example 1: Graded Relevance (0-3 scale)")
print("-" * 70)
graded_rels = [3, 2, 1, 0]  # Highly relevant first
print(f"Relevance scores: {graded_rels}")
print(f"\nExponential DCG: {dcg_at_k(graded_rels, k=4, method='exponential'):.4f}")
print(f"  → Gains: [7, 3, 1, 0]  (2^rel - 1) - rel=3 contributes 7x what rel=1 does")
print(f"Linear DCG:      {dcg_at_k(graded_rels, k=4, method='linear'):.4f}")
print(f"  → Gains: [3, 2, 1, 0]  (rel) - rel=3 contributes exactly 3x what rel=1 does")
print("\n✅ Exponential is HIGHER because it strongly rewards rel=3 (gain=7 vs 3)")
print("   Both still apply position discount via log2(i+2)")

# Example 2: Bad ranking with graded relevance
print("\n📊 Example 2: Same docs, BAD ranking (low relevance first)")
print("-" * 70)
bad_ranking = [0, 1, 2, 3]  # Worst ranking - highly relevant last!
print(f"Relevance scores: {bad_ranking}")
print(f"\nExponential DCG: {dcg_at_k(bad_ranking, k=4, method='exponential'):.4f}")
print(f"Linear DCG:      {dcg_at_k(bad_ranking, k=4, method='linear'):.4f}")
print("\n✅ Both punish bad ranking, but exponential MORE STRICT")

# Example 3: Binary relevance - formulas are equivalent
print("\n📊 Example 3: Binary Relevance (0 or 1 only)")
print("-" * 70)
binary_rels = [1, 1, 0, 0]
print(f"Relevance scores: {binary_rels}")
print(f"\nExponential DCG: {dcg_at_k(binary_rels, k=4, method='exponential'):.4f}")
print(f"Linear DCG:      {dcg_at_k(binary_rels, k=4, method='linear'):.4f}")
print("\n✅ IDENTICAL results for binary relevance!")

# Example 4: NDCG comparison with real doc IDs
print("\n📊 Example 4: NDCG with Document IDs")
print("-" * 70)
ranked_ids = ['doc1', 'doc2', 'doc3', 'doc4']
qrels = {'doc1': 3, 'doc2': 2, 'doc3': 1, 'doc4': 0}
print(f"Ranked IDs: {ranked_ids}")
print(f"Qrels: {qrels}")
print(f"\nNDCG@4 (exponential): {ndcg_at_k(ranked_ids, qrels, k=4, method='exponential'):.4f}")
print(f"NDCG@4 (linear):      {ndcg_at_k(ranked_ids, qrels, k=4, method='linear'):.4f}")
print("\n✅ Both = 1.0 for perfect ranking!")

# Example 5: Imperfect ranking shows the difference
print("\n📊 Example 5: NDCG with IMPERFECT ranking")
print("-" * 70)
bad_ranked_ids = ['doc3', 'doc1', 'doc2', 'doc4']  # doc1 (rel=3) not first!
print(f"Ranked IDs: {bad_ranked_ids}")
print(f"Qrels: {qrels}")
print(f"\nNDCG@4 (exponential): {ndcg_at_k(bad_ranked_ids, qrels, k=4, method='exponential'):.4f}")
print(f"NDCG@4 (linear):      {ndcg_at_k(bad_ranked_ids, qrels, k=4, method='linear'):.4f}")
print("\n✅ Exponential gives LOWER score - more strict penalty for misranking!")

# Summary
print("\n" + "=" * 70)
print("📋 WHEN TO USE EACH METHOD")
print("=" * 70)
print("""
✅ Use EXPONENTIAL (default):
   - Graded relevance (0, 1, 2, 3, etc.)
   - Want to emphasize highly relevant docs (rel=3 gets 7x boost vs rel=1)
   - Industry standard (Google, Kaggle, TREC)
   
⚡ Use LINEAR:
   - Want proportional scaling of relevance (rel=3 gets exactly 3x boost vs rel=1)
   - Simpler interpretation and less aggressive differentiation
   - Academic research comparing to older papers
   
📝 Important Notes:
   - Both formulas discount by position using log2(i+2)
   - Both are IDENTICAL for binary relevance (0/1)!
   - The difference is only in how gains scale with relevance scores
""")

