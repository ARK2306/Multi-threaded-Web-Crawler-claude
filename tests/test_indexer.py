"""Tokenizer and TF-IDF maths, verified against hand-computed values on static data."""

from __future__ import annotations

import math

import pytest

import indexer
import storage

# A three-document corpus small enough to compute by hand.
CORPUS = {
    1: "the cat sat on the mat",       # 6 tokens
    2: "the dog sat on the log",       # 6 tokens
    3: "cats and dogs are friends",    # 5 tokens
}


class TestTokenize:
    def test_lowercases(self):
        assert indexer.tokenize("Hello WORLD") == ["hello", "world"]

    def test_removes_punctuation(self):
        assert indexer.tokenize("Hello, world! It's a test.") == [
            "hello", "world", "it", "s", "a", "test",
        ]

    def test_splits_on_all_whitespace(self):
        assert indexer.tokenize("a\tb\nc  d") == ["a", "b", "c", "d"]

    def test_keeps_numbers_and_alphanumerics(self):
        assert indexer.tokenize("Python 3.14 utf8") == ["python", "3", "14", "utf8"]

    def test_underscores_split_tokens(self):
        assert indexer.tokenize("max_pages") == ["max", "pages"]

    def test_keeps_unicode_letters(self):
        assert indexer.tokenize("Café naïve") == ["café", "naïve"]

    def test_drops_absurdly_long_tokens(self):
        assert indexer.tokenize("a" * (indexer.MAX_TOKEN_LENGTH + 1)) == []

    @pytest.mark.parametrize("text", ["", "   ", "!!! ??? ---"])
    def test_empty_results(self, text):
        assert indexer.tokenize(text) == []

    def test_preserves_repetition(self):
        assert indexer.tokenize("test test test") == ["test", "test", "test"]


class TestTermFrequency:
    def test_counts_are_length_normalised(self):
        tf = indexer.term_frequencies(indexer.tokenize(CORPUS[1]))
        assert tf["the"] == pytest.approx(2 / 6)
        assert tf["cat"] == pytest.approx(1 / 6)

    def test_frequencies_sum_to_one(self):
        tf = indexer.term_frequencies(indexer.tokenize(CORPUS[1]))
        assert sum(tf.values()) == pytest.approx(1.0)

    def test_empty_document(self):
        assert indexer.term_frequencies([]) == {}

    def test_single_token_document(self):
        assert indexer.term_frequencies(["only"]) == {"only": 1.0}


class TestInverseDocumentFrequency:
    def test_matches_the_formula(self):
        # ln((1 + N) / (1 + df)) + 1
        assert indexer.inverse_document_frequency(3, 1) == pytest.approx(math.log(4 / 2) + 1)
        assert indexer.inverse_document_frequency(3, 3) == pytest.approx(math.log(4 / 4) + 1)

    def test_term_in_every_document_scores_exactly_one(self):
        assert indexer.inverse_document_frequency(10, 10) == pytest.approx(1.0)

    def test_rarer_terms_score_higher(self):
        rare = indexer.inverse_document_frequency(100, 1)
        common = indexer.inverse_document_frequency(100, 50)
        assert rare > common > 1.0

    def test_always_positive(self):
        for n in (1, 5, 50):
            for df in range(0, n + 1):
                assert indexer.inverse_document_frequency(n, df) > 0


class TestComputeTfidf:
    @pytest.fixture
    def computed(self):
        return indexer.compute_tfidf(CORPUS)

    def test_document_frequency_is_correct(self, computed):
        idf, postings = computed
        assert set(postings["the"]) == {1, 2}     # docs 1 and 2
        assert set(postings["sat"]) == {1, 2}
        assert set(postings["cat"]) == {1}
        assert set(postings["cats"]) == {3}       # no stemming: "cat" != "cats"

    def test_idf_values_match_hand_calculation(self, computed):
        idf, _ = computed
        assert idf["the"] == pytest.approx(math.log(4 / 3) + 1)   # df = 2, N = 3
        assert idf["cat"] == pytest.approx(math.log(4 / 2) + 1)   # df = 1

    def test_tfidf_is_tf_times_idf(self, computed):
        idf, postings = computed
        tf, tfidf = postings["cat"][1]
        assert tf == pytest.approx(1 / 6)
        assert tfidf == pytest.approx((1 / 6) * (math.log(4 / 2) + 1))

    def test_rarer_term_wins_at_equal_term_frequency(self, computed):
        _, postings = computed
        # "cat" (df=1) and "sat" (df=2) both occur once in doc 1, so their tf is equal
        # and the rarer term must carry the larger tf-idf weight.
        assert postings["cat"][1][0] == postings["sat"][1][0]
        assert postings["cat"][1][1] > postings["sat"][1][1]

    def test_repeated_term_beats_single_occurrence_at_equal_idf(self, computed):
        _, postings = computed
        # "the" occurs twice in doc 1 and "on" once; both have df=2, so tf decides.
        assert postings["the"][1][1] > postings["on"][1][1]

    def test_every_token_is_indexed(self, computed):
        idf, postings = computed
        expected = {token for text in CORPUS.values() for token in indexer.tokenize(text)}
        assert set(postings) == expected == set(idf)

    def test_empty_corpus(self):
        idf, postings = indexer.compute_tfidf({})
        assert idf == {} and postings == {}

    def test_single_document_corpus(self):
        idf, postings = indexer.compute_tfidf({1: "alpha beta beta"})
        assert idf["beta"] == pytest.approx(math.log(2 / 2) + 1)  # == 1.0
        assert postings["beta"][1][0] == pytest.approx(2 / 3)


class TestBuildIndex:
    @pytest.fixture
    def conn(self, tmp_path):
        connection = storage.init_db(tmp_path / "index.db")
        with connection:
            connection.executemany(
                "INSERT INTO documents (url, title, text, fetched_at) VALUES (?, ?, ?, ?)",
                [
                    ("https://example.com/1", "One", CORPUS[1], "2026-01-01T00:00:00+00:00"),
                    ("https://example.com/2", "Two", CORPUS[2], "2026-01-01T00:00:00+00:00"),
                    ("https://example.com/3", "Three", CORPUS[3], "2026-01-01T00:00:00+00:00"),
                ],
            )
        yield connection
        connection.close()

    def test_writes_terms_and_postings(self, conn):
        stats = indexer.build_index(conn)
        assert stats.documents == 3
        assert stats.terms == conn.execute("SELECT COUNT(*) FROM terms").fetchone()[0]
        assert stats.postings == conn.execute("SELECT COUNT(*) FROM postings").fetchone()[0]

    def test_stored_values_match_the_maths(self, conn):
        indexer.build_index(conn)
        row = conn.execute(
            "SELECT p.tf, p.tfidf, t.df, t.idf FROM postings p JOIN terms t ON t.term = p.term "
            "WHERE p.term = 'cat'"
        ).fetchone()
        assert row["df"] == 1
        assert row["idf"] == pytest.approx(math.log(4 / 2) + 1)
        assert row["tf"] == pytest.approx(1 / 6)
        assert row["tfidf"] == pytest.approx(row["tf"] * row["idf"])

    def test_records_document_term_counts(self, conn):
        indexer.build_index(conn)
        counts = {
            row["url"]: row["term_count"]
            for row in conn.execute("SELECT url, term_count FROM documents")
        }
        assert counts["https://example.com/1"] == 6
        assert counts["https://example.com/3"] == 5

    def test_rebuild_is_idempotent(self, conn):
        first = indexer.build_index(conn)
        second = indexer.build_index(conn)
        assert first.as_dict() == second.as_dict()

    def test_records_metadata(self, conn):
        indexer.build_index(conn)
        assert storage.get_meta(conn, "document_count") == "3"
        assert storage.get_meta(conn, "indexed_at")

    def test_empty_database_clears_the_index(self, conn):
        indexer.build_index(conn)
        with conn:
            conn.execute("DELETE FROM documents")
        stats = indexer.build_index(conn)
        assert stats.documents == 0
        assert conn.execute("SELECT COUNT(*) FROM postings").fetchone()[0] == 0
