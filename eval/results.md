# Evaluation Benchmark Results

## Summary Metrics
- **Total Test Cases**: 40 (12 Direct Questions, 28 Situations)
- **Retrieval Hit Rate (Top-8)**: **80.0%** (32/40)
- **Answer Citation Accuracy**: **80.0%** (32/40)
- **Evaluation Duration**: 689.0 seconds

## Methodology
1. **Analyze**: Query analyzer extracts candidate offences, legal queries, and resolves IPC/BNS ambiguity.
2. **Retrieve**: Hybrid multi-query retrieval (dense `bge-base-en-v1.5` + BM25) with Reciprocal Rank Fusion over BNS and BNSS chunks, reranked via `bge-reranker-base` to top 8.
3. **Answer**: Grounded generation strictly from retrieved chunks.
4. **Verify**: Deterministic code verification ensuring all cited sections exist in retrieved context.

## Detailed Case Results
| ID | Type | Query | Expected Sections | Retrieved Top-8 Sections | Retrieval Hit | Citation Hit |
|---|---|---|---|---|:---:|:---:|
| 1 | question | punishment for cheating under BNS? | 318 | 319, 173, 334, 174, 292, 317 | ❌ | ❌ |
| 2 | question | what is punishment for 420 IPC in new law? | 318 | 315, 336, 188, 318, 323, 499 | ✅ | ✅ |
| 3 | question | what is the punishment for murder under BNS? | 103 | 104, 103, 109, 105, 66, 173 | ✅ | ✅ |
| 4 | question | what is the punishment for theft under section 303 BNS? | 303 | 306, 303, 305, 283, 303, 334 | ✅ | ✅ |
| 5 | question | what is the offence of snatching in BNS and its punishment? | 304 | 304, 334, 307, 134, 311, 306 | ✅ | ✅ |
| 6 | question | what is the punishment for criminal breach of trust under BNS? | 316 | 316, 357, 131, 173, 269, 245 | ✅ | ✅ |
| 7 | question | what constitutes extortion under BNS? | 308 | 308, 309, 120, 119, 334, 201 | ✅ | ✅ |
| 8 | question | what constitutes robbery under BNS? | 309 | 111, 309, 311, 313, 254, 317 | ✅ | ✅ |
| 9 | question | what is the punishment for criminal intimidation under BNS? | 351 | 351, 352, 131, 135, 136, 173 | ✅ | ✅ |
| 10 | question | what is the section for stalking a woman under BNS? | 78 | 78, 77, 87, 84, 75, 85 | ✅ | ✅ |
| 11 | question | what is the offence and punishment for defamation in BNS? | 356 | 357, 355, 222, 302, 356, 127 | ✅ | ✅ |
| 12 | question | what is section 302? | 103, 302 | 302, 103, 293, 249, 109, 234 | ✅ | ✅ |
| 13 | situation | My purse was stolen from my handbag while I was riding the metro train | 303, 305 | 134, 97, 304, 323, 317, 283 | ❌ | ❌ |
| 14 | situation | Two men on a motorcycle zoomed past and snatched the gold chain off my neck | 304 | 304, 134, 307, 309, 311, 106 | ✅ | ✅ |
| 15 | situation | Three armed men entered our jewelry shop, pointed a gun at the cashier, and took 5 lakh rupees cash from the till | 309, 311 | 499, 243, 134, 119, 312, 246 | ✅ | ✅ |
| 16 | situation | I paid 50000 rupees to an online seller for a smartphone, but after receiving money he blocked me and never delivered anything | 318 | 499, 243, 318, 320, 119, 202 | ✅ | ✅ |
| 17 | situation | I gave my car to my friend for a weekend trip, but he sold it to a third party without my knowledge and kept the money | 316 | 315, 499, 323, 119, 317, 120 | ❌ | ❌ |
| 18 | situation | A man has been following my sister home every day from her workplace and sending repeated unsolicited messages on Instagram | 78 | 78, 202, 355, 75, 173, 173 | ✅ | ✅ |
| 19 | situation | My neighbor unlawfully entered my private courtyard with an iron rod and threatened to kill my family | 329, 351 | 146, 238, 100, 236, 234, 241 | ❌ | ❌ |
| 20 | situation | During a dispute over parking, a man hit my father with a baseball bat and fractured his forearm | 115, 117, 118 | 241, 118, 120, 119, 131, 397 | ✅ | ✅ |
| 21 | situation | My business partner forged my signature on company bank cheques and withdrew 10 lakh rupees from our joint account | 316, 318, 336, 338 | 336, 246, 338, 344, 238, 318 | ✅ | ✅ |
| 22 | situation | A mob surrounded my parked car, smashed the windshield with stones and scratched the doors causing heavy damage | 324 | 171, 304, 502, 298, 107, 334 | ❌ | ❌ |
| 23 | situation | A colleague posted defamatory remarks on LinkedIn falsely accusing me of financial embezzlement and fraud | 356 | 356, 222, 323, 355, 238, 319 | ✅ | ✅ |
| 24 | situation | Someone called posing as a bank executive, fraudulently induced me to give my OTP, and drained 1 lakh rupees | 318, 319 | 318, 243, 319, 225, 321, 323 | ✅ | ✅ |
| 25 | situation | A local goon demanded 20,000 rupees monthly extortion money, threatening to burn down my grocery store if I refuse | 308 | 308, 201, 309, 119, 334, 120 | ✅ | ✅ |
| 26 | situation | A tenant refused to vacate my apartment after tenancy ended, changed locks, and forcefully prevented me from entering | 329, 126 | 329, 330, 127, 44, 89, 43 | ✅ | ✅ |
| 27 | situation | A customer in my restaurant got angry over the bill and slapped the waiter across the face | 115, 131 | 241, 120, 243, 119, 131, 134 | ✅ | ✅ |
| 28 | situation | The police station officer refused to register an FIR when I went to report my stolen motorcycle | 303 | 173, 173, 173, 323, 107, 173 | ❌ | ❌ |
| 29 | situation | A gym owner installed a hidden spy camera in the female trial room and captured private photos of women | 77 | 77, 180, 308, 78, 173, 184 | ✅ | ✅ |
| 30 | situation | Two men snatched my sister's handbag and shoved her to the ground causing severe abrasions on her hands and knees | 304, 309, 115 | 304, 134, 112, 307, 201, 359 | ✅ | ✅ |
| 31 | situation | An unknown fraudster created a fake profile in my name with my photo on WhatsApp and messaged my contacts asking for emergency money | 318, 319 | 344, 323, 319, 202, 238, 235 | ✅ | ✅ |
| 32 | situation | Our domestic servant stole diamond jewelry worth 3 lakh rupees from the master bedroom and disappeared | 303, 306 | 306, 243, 317, 106, 97, 252 | ✅ | ✅ |
| 33 | situation | A man falsely promised marriage to a woman concealing his existing marriage, engaged in sexual relations, and then refused marriage | 69 | 69, 81, 84, 219, 83, 82 | ✅ | ✅ |
| 34 | situation | A speeding driver ran over a cyclist at night, fled the scene without reporting to police or helping the injured person who later died | 106, 281 | 194, 106, 106, 100, 281, 171 | ✅ | ✅ |
| 35 | situation | A gang of five men broke into a farmhouse at night with knives, tied up the residents, and looted cash and gold | 310, 311 | 111, 201, 110, 310, 134, 313 | ✅ | ✅ |
| 36 | situation | A stalker threw chemical acid on a young woman's face causing severe permanent disfigurement | 124 | 124, 78, 77, 184, 87, 89 | ✅ | ✅ |
| 37 | situation | A warehouse manager who was entrusted with 500 bags of wheat misappropriated them and sold them in the black market | 316 | 315, 317, 118, 235, 201, 119 | ❌ | ❌ |
| 38 | situation | A co-passenger on a bus inappropriately grabbed a woman and used vulgar sexual language towards her | 74, 75, 79 | 355, 87, 75, 69, 308, 184 | ✅ | ✅ |
| 39 | situation | A builder took 25 lakh rupees as booking advance for a residential flat, never began construction, and fled | 318 | 243, 220, 323, 499, 244, 238 | ❌ | ❌ |
| 40 | situation | I am being falsely threatened with a complaint under section 420 IPC and section 406 IPC, what are the corresponding BNS sections? | 318, 316 | 228, 318, 334, 314, 225, 244 | ✅ | ✅ |
