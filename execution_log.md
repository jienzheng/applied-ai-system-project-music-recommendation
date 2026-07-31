# Execution Log

Real, captured terminal output from this repo, run on macOS with Python 3.11.15 in a fresh `.venv`. Nothing below is a summary or a fabricated transcript — the pytest section is pasted verbatim from an actual run in this session.

## Reproducing from scratch

```bash
git clone https://github.com/jienzheng/applied-ai-system-project-music-recommendation.git
cd applied-ai-system-project-music-recommendation

python3 -m venv .venv
source .venv/bin/activate      # Mac/Linux; .venv\Scripts\activate on Windows

pip install -r requirements.txt

cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=<your key>

python main.py
```

To run the test suite instead:

```bash
pytest tests/ -v
```

## Real output: `pytest tests/ -v`

Captured from an actual run in this session (`.venv`, Python 3.11.15, pytest 8.3.4):

```
============================= test session starts ==============================
platform darwin -- Python 3.11.15, pytest-8.3.4, pluggy-1.6.0 -- /Users/jien/CodePath/AI110/applied-ai-system-project-music-recommendation/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /Users/jien/CodePath/AI110/applied-ai-system-project-music-recommendation
plugins: anyio-4.14.2
collecting ... collected 17 items

tests/test_evaluation.py::test_retrieval_returns_nonempty_results_for_normal_query PASSED [  5%]
tests/test_evaluation.py::test_retrieval_handles_empty_query_gracefully PASSED [ 11%]
tests/test_evaluation.py::test_retrieval_handles_nonsense_query_gracefully PASSED [ 17%]
tests/test_evaluation.py::test_pipeline_passes_valid_first_try_response PASSED [ 23%]
tests/test_evaluation.py::test_pipeline_catches_hallucinated_song_and_recovers_on_retry PASSED [ 29%]
tests/test_evaluation.py::test_pipeline_falls_back_to_top_retrieved_songs_after_two_failures PASSED [ 35%]
tests/test_evaluation.py::test_pipeline_missing_api_key_raises_clear_error PASSED [ 41%]
tests/test_evaluation.py::test_pipeline_handles_network_error_gracefully PASSED [ 47%]
tests/test_evaluation.py::test_pipeline_handles_rate_limit_error_gracefully PASSED [ 52%]
tests/test_evaluation.py::test_confidence_stage_ordering_matches_pipeline_trust PASSED [ 58%]
tests/test_retriever.py::test_retrieve_returns_k_results PASSED          [ 64%]
tests/test_retriever.py::test_relevant_genre_ranks_higher PASSED         [ 70%]
tests/test_retriever.py::test_retrieve_respects_default_k PASSED         [ 76%]
tests/test_validator.py::test_accepts_valid_songs PASSED                 [ 82%]
tests/test_validator.py::test_rejects_hallucinated_songs PASSED          [ 88%]
tests/test_validator.py::test_rejects_song_with_mismatched_artist PASSED [ 94%]
tests/test_validator.py::test_fallback_returns_top_k_candidates_with_reason PASSED [100%]

============================== 17 passed in 2.96s ==============================
```

Note: on the first attempt at writing this suite, one test (`test_pipeline_catches_hallucinated_song_and_recovers_on_retry`) failed with `1 failed, 16 passed in 4.22s`, because of an incorrect assertion in the test itself (assumed every recommended song's confidence would be nonzero, which is false for a song with zero real cosine similarity to the query). The assertion was corrected and the suite was rerun; the `17 passed` output above is that final, confirmed run. See `EVALUATION.md` for the full explanation.

## Real output: `python main.py`

**Not yet captured.** Running `main.py` requires a live `ANTHROPIC_API_KEY` in `.env` to call the Claude API — no key was available in this development environment during this session, and no output has been invented to fill this section. Once a key is available, run:

```bash
python main.py
```

and enter a few queries (e.g. `chill instrumental music for studying`, an empty string, `quit` to exit), then paste the real terminal transcript here, replacing this note. The same real output should also be used to fill in the placeholders in `README.md`'s Sample Interactions section and `EVALUATION.md`'s Human/Manual Evaluation table.
