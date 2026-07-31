# Evaluation

This document reports real, observed results from running this repo's automated test suite and the CLI itself — nothing here is fabricated or estimated. Automated results are from an actual `pytest tests/ -v` run (Python 3.11, `.venv`). Manual runs against the live Claude API were attempted with a real key; the retrieval layer ran for real on every query, but the configured Anthropic account had **no API credit balance**, so the generation step failed on every query with a billing error rather than producing recommendations. That billing error, and the retrieval results, are reported below exactly as observed — no recommendation output or confidence scores have been invented to compensate.

## Automated Test Results

Run: `pytest tests/ -v`, Python 3.11.15, pytest 8.3.4 — **17 passed, 0 failed** (2.96s).

| Test | What It Checks | Result |
|------|----------------|--------|
| `test_retrieval_returns_nonempty_results_for_normal_query` | Retrieval returns a non-empty, well-typed result list for a normal query | PASS |
| `test_retrieval_handles_empty_query_gracefully` | An empty-string query doesn't crash the retriever and still returns `DEFAULT_TOP_K` results | PASS |
| `test_retrieval_handles_nonsense_query_gracefully` | A gibberish/nonsense query doesn't crash the retriever | PASS |
| `test_pipeline_passes_valid_first_try_response` | Full pipeline: a first-try model response that only names retrieved songs is accepted with no re-prompt | PASS |
| `test_pipeline_catches_hallucinated_song_and_recovers_on_retry` | Full pipeline: a hallucinated song in the first response is caught by the guardrail, triggers exactly one re-prompt, and the corrected retry is accepted | PASS |
| `test_pipeline_falls_back_to_top_retrieved_songs_after_two_failures` | Full pipeline: two consecutive hallucinated responses trigger the fallback path, returning exactly the top-3 retrieved songs | PASS |
| `test_pipeline_missing_api_key_raises_clear_error` | Missing `ANTHROPIC_API_KEY` raises a clear `MusicRecommenderError` at construction time, not an unhandled crash | PASS |
| `test_pipeline_handles_network_error_gracefully` | An `anthropic.APIConnectionError` from the client is caught and re-raised as a clear `MusicRecommenderError` | PASS |
| `test_pipeline_handles_rate_limit_error_gracefully` | An `anthropic.RateLimitError` from the client is caught and re-raised as a clear `MusicRecommenderError` | PASS |
| `test_confidence_stage_ordering_matches_pipeline_trust` | Confidence scoring ranks first-try > retry > fallback for the same similarity, and first-try equals the raw similarity | PASS |
| `test_retrieve_returns_k_results` | Retrieval respects an explicit `k` | PASS |
| `test_relevant_genre_ranks_higher` | A query semantically closer to jazz songs ranks them above an unrelated metal song | PASS |
| `test_retrieve_respects_default_k` | Omitting `k` falls back to `DEFAULT_TOP_K` | PASS |
| `test_accepts_valid_songs` | Validator accepts recommendations matching the candidate set (case-insensitive title+artist) | PASS |
| `test_rejects_hallucinated_songs` | Validator rejects a song not present in the candidate set | PASS |
| `test_rejects_song_with_mismatched_artist` | Validator rejects a correct title paired with the wrong artist | PASS |
| `test_fallback_returns_top_k_candidates_with_reason` | Fallback returns exactly `k` candidates, each annotated with a non-empty reason | PASS |

**One test failed on first run and was fixed, not silently ignored** (see Confidence Scoring Summary and Summary below for what this revealed): `test_pipeline_catches_hallucinated_song_and_recovers_on_retry` initially asserted every recommended song would have `confidence > 0`. It failed because the tiny 5-song fixture dataset used in this test file produces genuine zero cosine-similarity for 2 of the 3 top-ranked songs against the test query (no shared vocabulary in the bag-of-words fake embedding model used for offline testing) — a real, correctly-computed zero, not a bug in the scoring code. The assertion was corrected to check that confidence is a valid score in `[0.0, 1.0]` and that the top retrieval match has the highest confidence, and the suite was rerun to confirm all 17 tests pass.

## Human/Manual Evaluation

Real `python main.py` runs, captured verbatim (see `execution_log.md` for the full raw transcript). A live `ANTHROPIC_API_KEY` was used; the account had no credit balance, so every query's generation step returned the same billing error (`400 invalid_request_error: "Your credit balance is too low..."`) — retrieval itself ran for real and succeeded on all 5 inputs below.

| Test Input | Evaluation Criteria | Result |
|------------|---------------------|--------|
| Blank input (Enter with no text) | Handles gracefully, no crash | **Pass** — `main.py`'s own `if not query: continue` check re-prompts without ever calling the retriever or the API; confirmed by a real run showing no retrieval log line for that turn. |
| `"asdkjfh qwoieur nonsense gibberish"` | Handles gracefully, no crash, doesn't hallucinate confidently | **Pass on retrieval, blocked on generation** — retrieval returned 8 real candidates with uniformly low similarity (0.152-0.217, vs. ~0.69 for a well-matched query), correctly signaling a poor match; generation itself failed with the billing error before any recommendation (or possible hallucination) could occur. |
| `"chill instrumental music for studying"` | Normal query, returns 3 relevant, validated songs | **Pass on retrieval, blocked on generation** — retrieval correctly surfaced clearly relevant songs (top match "Study Session" at 0.689, then instrumental/ambient/classical pieces); generation failed with the billing error, so no final 3-song output or validation outcome exists yet. |
| `"upbeat 90s Japanese city pop for a road trip"` (obscure/adversarial — catalog has no city pop) | System should retrieve its closest available matches and not fabricate a song outside the catalog | **Pass on retrieval, blocked on generation** — retrieval fell back to loosely related upbeat/nostalgic tracks (Africa, Midnight City, Around the World) rather than erroring or returning nothing, which is the correct behavior for a genre absent from the dataset; generation failed with the billing error. |
| `"sad happy angry calm loud quiet music"` (deliberately contradictory/ambiguous) | System should still return 3 validated songs rather than erroring on conflicting signals | **Pass on retrieval, blocked on generation** — retrieval returned 8 candidates without erroring on the contradictory phrasing (scores 0.42-0.47, all fairly close together, consistent with a query that doesn't point clearly in one direction); generation failed with the billing error. |

**What this does and doesn't demonstrate:** every row confirms the retrieval layer is robust to normal, nonsense, obscure, and contradictory input — it never crashed, always returned `DEFAULT_TOP_K` results, and its similarity scores behaved sensibly (high for a clear match, low/flat for nonsense or contradictory input). None of these rows demonstrate the generation or validation-guardrail behavior on a *live* model response, since the billing error occurred before the model could generate anything — that gap is separately covered by the automated, mocked tests above (`test_pipeline_catches_hallucinated_song_and_recovers_on_retry`, `test_pipeline_falls_back_to_top_retrieved_songs_after_two_failures`), which do exercise real (mocked) hallucination detection and recovery.

## Confidence Scoring Summary

**No live confidence scores were produced** — every manual query failed at the generation step (billing error) before `attach_confidence()` was ever reached, so there is no query-level average to report from live runs. The confidence mechanism itself is implemented (`src/scorer.py`: `confidence = clamp(similarity, 0, 1) * stage_weight`, with `stage_weight` = 1.0 for first-try, 0.7 for a corrected retry, 0.4 for a fallback) and is exercised by `test_confidence_stage_ordering_matches_pipeline_trust`, which confirms — with real, computed numbers — that for a fixed similarity of `0.9`: first-try confidence = `0.9`, retry confidence = `0.63`, fallback confidence = `0.36`.

One real pattern *is* observable from the retrieval-only data above, even without generation: **similarity scores dropped sharply for vague/contradictory language versus specific requests** — "chill instrumental music for studying" (specific mood + use case) scored 0.689 on its top match, while "sad happy angry calm loud quiet music" (contradictory) topped out at 0.468, and the nonsense query topped out at 0.217. Since confidence is a direct multiple of similarity, this strongly suggests confidence scores will follow the same pattern once generation succeeds — vague or self-contradictory queries should produce visibly lower confidence than specific, well-matched ones — but this is an inference from retrieval scores, not a confirmed observation of the confidence field itself.

`[FILL IN: once the account has API credit, rerun these same 5 queries and report the actual average confidence across them, confirming or correcting the pattern above]`

## Summary

17 out of 17 automated tests passed (after fixing one incorrect test assertion, described above, and rerunning to confirm); the automated suite exercises retrieval on normal/empty/nonsense queries, the full validate→retry→fallback pipeline (including a mocked hallucinated response, confirmed to be caught and either corrected or safely replaced with retrieved songs), missing-API-key handling, and network/rate-limit error handling. In manual testing with a real API key, the system's actual weakness turned out to be operational rather than logical: the configured Anthropic account had no credit balance, so 5 real queries all failed at generation with a billing error — retrieval itself performed correctly on every one, including graceful, non-crashing behavior on a blank input, a nonsense query, an obscure adversarial genre request, and a contradictory query. Confidence scores averaged **N/A this session** (no query reached the scoring step live), though the unit-tested scoring formula and the retrieval-score pattern observed above both suggest confidence would track query specificity closely. The validation guardrail did catch a hallucinated recommendation during automated (mocked) testing and correctly prevented it from reaching the final output — that remains the only concrete evidence so far that the guardrail works against a bad model response; it has not yet been exercised against a real, live hallucination.
