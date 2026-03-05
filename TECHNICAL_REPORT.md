# Technical Report: Transcription Quality Assessment Pipeline

## Overview
This report explains the methods, scoring logic, and results of the Transcription Quality Assessment system, specifically focusing on the batch processing and "Golden Reference" ranking strategy.

## 1. Methodology
The system follows a multi-stage pipeline to evaluate and rank multiple candidate transcriptions for a given audio sample:

1.  **Ground Truth Generation:** A high-quality reference transcription is generated using the OpenAI Whisper large-v3 model.
2.  **Text Normalization:** Both the reference and all candidate transcriptions undergo language-agnostic normalization (lowercasing, punctuation removal, and whitespace collapse) to ensure fair comparison.
3.  **Atomic Metric Computation:** Nine atomic metrics are calculated for each candidate against the Whisper-generated reference:
    *   **WER (Word Error Rate):** Overall correctness.
    *   **CER (Character Error Rate):** Morphological and spelling accuracy.
    *   **Precision/Recall:** Detection of hallucinations vs. omissions.
    *   **Semantic Similarity:** Meaning preservation using multilingual sentence embeddings (`paraphrase-multilingual-mpnet-base-v2`).
    *   **Completeness:** Coverage of key content (weighted towards entities and numbers).
    *   **Fluency:** Naturalness of the text using a multilingual GPT model (`ai-forever/mGPT`).
    *   **Punctuation:** Formatting accuracy.
    *   **Alignment:** Word order consistency using LCS.
4.  **Metric Normalization:** All raw metrics are normalized to a [0, 1] scale, where 1.0 represents the highest quality.
5.  **Composite Quality Score (CQS):** A weighted sum of the normalized metrics provides a final quality score used for ranking.

## 2. Scoring & Ranking Logic
The final rank is determined by the **CQS**. In the event of a tie, the system applies the following tie-breaking criteria in order:
1.  Higher Semantic Similarity
2.  Higher Completeness Score
3.  Higher WER Score (lower raw WER)

The candidate with the highest CQS (Rank 1) is designated as the **"Golden Reference"** for that specific audio sample.

## 3. Batch Execution & Golden Reference WER
The batch execution API (`/score/batch`) adds a post-ranking analysis step:

*   **Golden Reference WER Calculation:** Once the "Golden Reference" (the best candidate) is identified, the system recalculates the Word Error Rate (WER) for all five candidates **relative to this best candidate**, rather than the original Whisper reference.
*   **Why this method?** This "WER with respect to the best candidate" (`wer_bc`) provides a measure of how much each transcription deviates from what the system has determined to be the highest-quality option among the available choices. This is often more useful for comparative analysis than measuring divergence from an imperfect ASR reference.

## 4. Results & Deliverables
The batch processing output is delivered via:
1.  **JSON Response:** A structured list of results for each audio ID, including all metrics, scores, and ranks.
2.  **CSV Export:** A file (`batch_results_<UUID>.csv`) containing the following schema:
    *   `audio_id`, `language`, `audio` (URL)
    *   `option_1` to `option_5`: The original candidate texts.
    *   `golden_ref`: The text of the top-ranked candidate.
    *   `golden_score`: The CQS of the top-ranked candidate.
    *   `wer_option1` to `wer_option5`: The WER of each option calculated against the `golden_ref`.

## 5. Conclusion
By combining robust multilingual models with a multi-metric scoring approach and a comparative "Golden Reference" analysis, the system provides a comprehensive and objective framework for assessing transcription quality at scale.
