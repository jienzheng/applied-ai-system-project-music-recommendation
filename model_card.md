# 🎧 Model Card: Music Recommender Simulation

## Current System: RAG-Based Music Recommender

This repo evolved from the rule-based "Sad Pop" simulation documented below into a retrieval-augmented, Claude-powered recommender (see `README.md` for the full architecture). This section is the responsible-AI reflection for that current system. The original Module 1-3 model card follows unchanged, starting at "1. Model Name" below.

### AI Collaboration

Claude Code (via this same Claude Code session) generated the initial retriever/recommender/validator scaffolding — `src/retriever.py` (sentence-transformers + cosine similarity), `src/recommender.py` (Anthropic API call + JSON parsing), and `src/validator.py` (title+artist matching guardrail) — from a detailed architectural spec describing the desired pipeline, guardrail behavior, and logging requirements. I reviewed the generated code, ran it against the real dataset, and iterated with Claude Code across several follow-up turns: generating the Mermaid architecture diagram and reconciling it against the actual code, rewriting the README for a portfolio audience, and building out this evaluation/confidence-scoring system. Claude Code also wrote the automated test suite (`tests/test_retriever.py`, `tests/test_validator.py`, `tests/test_evaluation.py`) and ran it for real rather than describing hypothetical results.

**One specific HELPFUL AI suggestion:** the validation guardrail's retry-then-fallback pattern (`Recommender.recommend()` in `src/recommender.py`) — on a hallucinated recommendation, re-prompt the model exactly once with an explicit correction message, and only fall back to raw retrieved songs if that retry also fails. This worked well because it's bounded (at most two model calls per query, so no risk of looping forever on a confused model) while still giving the model a real chance to self-correct before giving up on generation entirely. `tests/test_evaluation.py::test_pipeline_catches_hallucinated_song_and_recovers_on_retry` confirms this recovers correctly when the retry succeeds, and `test_pipeline_falls_back_to_top_retrieved_songs_after_two_failures` confirms the fallback is exact (top-3 retrieved songs) when it doesn't.

**One specific FLAWED AI suggestion:** while building `tests/test_evaluation.py` in this same session, Claude Code wrote a test asserting that every recommended song in a retry-recovery scenario would have `confidence > 0`. Running the real test suite (not just writing it) caught this as false: with the tiny 5-song fixture dataset used for that test, 2 of the top-3 retrieved songs had a genuinely zero cosine similarity to the query (no shared vocabulary in the bag-of-words fake embedding model used to keep tests offline), so their correctly-computed confidence was legitimately `0.0` — the assertion was wrong, not the scoring code. The fix was to assert confidence is a valid `[0.0, 1.0]` score and that the top retrieval match has the highest confidence, rather than assuming every candidate must score above zero. This is recorded in `EVALUATION.md` under Automated Test Results rather than silently corrected and hidden.

### System Limitations

- **In-memory retrieval doesn't scale past a small dataset.** `SongRetriever.__init__` embeds the entire catalog into a numpy array held in RAM; there is no indexing structure (e.g. approximate nearest neighbor) that would let this scale to a large catalog efficiently.
- **No persistent vector store — the index rebuilds every run.** Every `python main.py` invocation re-embeds all ~107 songs from scratch (a few seconds); nothing is cached to disk between runs.
- **Single validation retry, not iterative refinement.** `Recommender.recommend()` retries exactly once on a failed validation, then falls back — there's no loop that keeps narrowing in with increasingly specific feedback.
- **Dataset size and diversity limits.** The catalog is roughly 100 songs; an obscure, hyper-specific, or genre-absent request (see the "90s Japanese city pop" adversarial query planned in `EVALUATION.md`) can only retrieve the closest available songs, which may not actually match well.
- **The guardrail's title+artist match is exact (case-insensitive) string matching, not fuzzy matching.** A model response with a typo, alternate spelling, or reworded title would be rejected as "hallucinated" even if it meant the retrieved song. This is a known trade-off, not an oversight — see `README.md`'s Design Decisions section.
- **No live manual evaluation of the generation layer has been performed yet** (see `EVALUATION.md`'s Human/Manual Evaluation section) — no `ANTHROPIC_API_KEY` was available in the development environment during this evaluation pass, so real Claude-generated output and confidence-score patterns across varied queries have not yet been observed and recorded.

### Biases and Risks

Looking at the actual composition of `data/songs.json` (107 songs): genre representation is uneven — classical, electronic, rock, and pop each have 13-14 songs, while metal and ambient have only 5 each, and the catalog skews heavily toward well-known Western/English-language "canonical" tracks (classic rock, jazz standards, Anglo-American pop/hip-hop/country) with no representation of, for example, K-pop, Latin, or other non-English-language popular genres. This means the retriever can only ever surface what's in that curated list — a request for a genre or regional style not represented in the dataset will retrieve whatever is closest by embedding similarity, not what the user actually meant, and the system has no way to signal "nothing in this catalog really matches." More generally, recommendation quality here depends entirely on dataset curation: the system reflects the tastes and blind spots of whoever built `data/songs.json` (in this case, generated by Claude Code from a request for "real, well-known songs" across a fixed list of genre categories), not some neutral or complete picture of music.

### Testing Summary

See `EVALUATION.md` for full results: 17/17 automated tests passed (after fixing one incorrect test assertion, described above). No live-query average confidence is reported yet, since no manual runs against the real Claude API were performed in this evaluation pass — see `EVALUATION.md`'s Confidence Scoring Summary for what is and isn't verified so far.

---

## 1. Model Name

**Sad Pop**

---

## 2. Intended Use

**Sad Pop** is designed to surface melancholic, low-energy, acoustic-leaning tracks for users who want music that matches a subdued or reflective mood, rather than music that lifts them out of it.

- **What kind of recommendations does it generate**
  It prioritizes songs with low valence, low-to-moderate energy, and high acousticness — regardless of whether the song is explicitly tagged "sad" in the metadata. It surfaces the closest available match rather than requiring an exact mood label.
- **What assumptions does it make about the user**
  It assumes the user is actively seeking music that matches (not contrasts with) a low mood — i.e., mood-congruent listening rather than mood-repair listening. This is a real behavioral pattern, but it's an assumption, not a guarantee: some users in a low mood want the opposite (upbeat music to shift their state).
- **Is this for real users or classroom exploration**
  This version is a classroom/prototype exploration built on a 28-song toy dataset. The scoring logic is realistic, but the catalog is far too small and narrow to serve real users reliably.
- **Not intended for**
  This should not be used as a mental health tool or mood-repair recommender. It always feeds sadness back to a sad-leaning profile, with no logic that checks in or nudges the user toward something lighter. It's built to demonstrate scoring and ranking logic, not to replace a production recommender.

---

## 3. How the Model Works

- **What features of each song are used**
  Three audio characteristics: valence (how happy vs. sad a song sounds), energy (how intense or calm it is), and acousticness (how stripped-back vs. produced it sounds). Genre and mood tags exist too, but this version scores on the audio features rather than the labels, since not every genuinely sad-sounding song is labeled "sad."
- **What user preferences are considered**
  A target profile: low valence, low-to-moderate energy, and high acousticness — a description of what a "sad song" typically sounds like, rather than a specific artist or genre.
- **How does the model turn those into a score**
  For each feature, the model checks how far a song's value is from the ideal "sad song" value, closer means a higher score. It's an average of *how close* each number is to the target, not the raw numbers. Valence matters most, energy somewhat less, acousticness least.
- **What changes did you make from the starter logic**
  The starter idea would have just ranked songs by "lowest valence." This version blends three features with different weights, so a genuinely angry, high-energy track with low valence doesn't get mistaken for a quiet, sad one.

---

## 4. Data

- **How many songs are in the catalog**
  28 songs, up from the original 10.
- **What genres or moods are represented**
  25 genres (pop, lofi, rock, ambient, jazz, synthwave, hip-hop, classical, metal, reggae, country, blues, house, techno, drum and bass, folk, r&b, soul, punk, k-pop, afrobeat, flamenco, bossa nova, trance, indie pop) and 24 moods (happy, chill, intense, melancholic, aggressive, sunny, nostalgic, euphoric, hypnotic, dreamy, defiant, playful, uplifting, passionate, serene, expansive, and more).
- **Did you add or remove data**
  Added 18 new songs, no removals. The expansion added genre/mood diversity the original 10-song set lacked entirely.
- **Are there parts of musical taste missing in the dataset**
  Yes — genuinely sad music is still nearly absent. Only one song ("Winter Nocturne," valence 0.28) is truly low-valence; everything else clusters at moderate-to-high valence. There's also no lyrical, vocal, cultural, or listening-history data, so collaborative filtering isn't possible here, only content-based matching.

---

## 5. Strengths

The system works best for users with clear, internally consistent preferences, especially profiles like **Chill Lofi** and **High-Energy Pop** where genre, mood, and energy all point the same direction. For example, **Library Rain** and **Midnight Coding** ranked highly for Chill Lofi because they match both tags and target energy closely. Near-exact energy alignment helps distinguish songs within the same genre/mood group, so ordering feels sensible. The reason strings are also useful for transparency, since each recommendation shows which components (genre, mood, energy, acousticness) drove the score.

---

## 6. Limitations and Bias

During a sensitivity test, doubling the energy weight caused high-energy tracks to dominate across multiple profiles, even when genre or mood didn't align well. Because the score only adds positive matches and never subtracts for conflicts, a profile with contradictory preferences (sad mood plus very high energy) can still surface upbeat songs at the top. This creates a filter-bubble effect around whichever feature is weighted most heavily, reducing variety in the top results. Acousticness is also only used when the user explicitly sets that preference, so users who don't set it get recommendations driven mostly by energy and tag matches.

---

## 7. Evaluation

Five profiles were tested: **High-Energy Pop**, **Chill Lofi**, **Deep Intense Rock**, **Adversarial: Sad but Max Energy**, and **Adversarial: Hyped but Highly Acoustic**. The check was whether the top 5 songs matched the vibe implied by each profile, not just one matching label. The biggest surprise was how often **Gym Hero** stayed near the top: very high energy and an "intense" tag let it score well even for users mainly asking for "happy pop" — showing the model currently treats high energy as a very strong signal.

Pair-by-pair comparisons:
1. **High-Energy Pop vs Chill Lofi:** upbeat tracks (*Sunrise City*, *Gym Hero*) vs. calmer songs (*Library Rain*, *Midnight Coding*) — expected, since target energy and genre/mood both flipped.
2. **High-Energy Pop vs Deep Intense Rock:** both stayed high-energy, but Deep Intense Rock pushed *Storm Runner* to #1 on rock+intense+energy, while High-Energy Pop favored pop/happy matches first.
3. **High-Energy Pop vs Adversarial: Sad but Max Energy:** nearly identical tops (*Gym Hero* again), since there are few "sad + high-energy pop" options in the dataset, so energy and genre dominate regardless.
4. **Deep Intense Rock vs Adversarial: Hyped but Highly Acoustic:** the acoustic preference nudged rankings slightly, but not enough to beat the strong energy signal.
5. **Chill Lofi vs Adversarial: Hyped but Highly Acoustic:** opposite ends of the energy spectrum, as expected.

---

## 8. Ideas for Improvement

- **Add a penalty for conflicting features, not just rewards for matches.** Right now the score only adds points, so a "sad mood, max energy" profile can still surface upbeat songs, since nothing subtracts points when features contradict. Penalizing mismatches, not just rewarding matches, would fix the adversarial-profile problem seen in evaluation.
- **Expand the dataset with genuinely low-valence songs.** Only one song is truly sad by valence. A "sad song" recommender needs real sad tracks, not just approximations built from energy and acousticness.
- **Add a diversity rule to the ranking step.** Ranking is currently a pure sort by score, so similar songs can crowd the top results. A simple fix: cap how many songs from the same artist or genre can appear in the top-k list.

---

## 9. Personal Reflection

Building this made it clear how much a recommender's behavior depends on its *weights*, not just its feature list — doubling the energy weight was enough to make an "intense pop" track like Gym Hero dominate profiles it didn't really belong in, even with a low-valence, sad-leaning target. The most unexpected discovery was how a purely additive scoring system has no way to express "these two preferences conflict" — it can only reward matches, never penalize contradictions, which is exactly the gap the adversarial tests exposed. It changed how I think about real recommendation apps: what feels like "the algorithm understands me" is often just one or two heavily-weighted features doing most of the work, and a system that never says "no" to a bad combination of signals can quietly drift toward whatever's easiest to match rather than what's actually the best fit.