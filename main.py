"""Interactive CLI entry point for the RAG-based music recommender."""

import logging
import sys

from dotenv import load_dotenv

from src.logging_config import setup_logging
from src.retriever import SongRetriever, load_songs
from src.recommender import Recommender, MusicRecommenderError

DATA_PATH = "data/songs.json"

logger = logging.getLogger(__name__)


def print_recommendations(recommendations) -> None:
    print("\nHere are your recommendations:\n")
    for i, song in enumerate(recommendations, start=1):
        print(f"{i}. {song['title']} — {song['artist']} ({song['genre']})")
        print(f"   {song.get('reason', '')}")
    print()


def main() -> None:
    load_dotenv()
    setup_logging()

    print("Loading song catalog and building embedding index...")
    try:
        songs = load_songs(DATA_PATH)
        retriever = SongRetriever(songs)
        recommender = Recommender(retriever)
    except MusicRecommenderError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: could not find dataset at {DATA_PATH}")
        sys.exit(1)

    print("Ready. Describe the kind of music you're in the mood for (or type 'quit' to exit).\n")

    while True:
        try:
            query = input("> ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        try:
            recommendations = recommender.recommend(query)
            print_recommendations(recommendations)
        except MusicRecommenderError as exc:
            print(f"Error: {exc}\n")
        except Exception:
            logger.exception("Unexpected error handling query %r", query)
            print("An unexpected error occurred. Check logs/app.log for details.\n")


if __name__ == "__main__":
    main()
