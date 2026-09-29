from pathlib import Path

import duckdb


INPUT = Path(
    "data/processed/interactions_collab_5core.parquet"
)

OUTPUT_DIR = Path(
    "data/processed/collab_5core"
)

DB_PATH = Path(
    "data/processed/build_collab_split.duckdb"
)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    con = duckdb.connect(str(DB_PATH))

    print("=" * 60)
    print("FashionRec V2 - Collaborative Leave-Last-Out")
    print("=" * 60)

    con.execute("DROP TABLE IF EXISTS ranked")

    print("\nRanking interactions by time...")

    con.execute(
        f"""
        CREATE TABLE ranked AS

        SELECT
            user_id,
            item_id,
            rating,
            timestamp,

            ROW_NUMBER() OVER (
                PARTITION BY user_id
                ORDER BY timestamp DESC, item_id DESC
            ) AS rank_from_end

        FROM read_parquet('{INPUT}')
        """
    )

    print("Writing train...")

    con.execute(
        f"""
        COPY (
            SELECT
                user_id,
                item_id,
                rating,
                timestamp
            FROM ranked
            WHERE rank_from_end >= 3
        )
        TO '{OUTPUT_DIR / "train.parquet"}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        )
        """
    )

    print("Writing validation...")

    con.execute(
        f"""
        COPY (
            SELECT
                user_id,
                item_id,
                rating,
                timestamp
            FROM ranked
            WHERE rank_from_end = 2
        )
        TO '{OUTPUT_DIR / "validation.parquet"}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        )
        """
    )

    print("Writing test...")

    con.execute(
        f"""
        COPY (
            SELECT
                user_id,
                item_id,
                rating,
                timestamp
            FROM ranked
            WHERE rank_from_end = 1
        )
        TO '{OUTPUT_DIR / "test.parquet"}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        )
        """
    )

    train = str(OUTPUT_DIR / "train.parquet")
    val = str(OUTPUT_DIR / "validation.parquet")
    test = str(OUTPUT_DIR / "test.parquet")

    print()
    print("Split statistics")
    print("-" * 50)

    for name, path in [
        ("Train", train),
        ("Validation", val),
        ("Test", test),
    ]:
        stats = con.execute(
            f"""
            SELECT
                COUNT(*),
                COUNT(DISTINCT user_id),
                COUNT(DISTINCT item_id)
            FROM read_parquet('{path}')
            """
        ).fetchone()

        print(
            f"{name:<10} "
            f"interactions={stats[0]:>11,} | "
            f"users={stats[1]:>9,} | "
            f"items={stats[2]:>8,}"
        )

    print()
    print("Cold-item analysis")
    print("-" * 50)

    for name, path in [
        ("Validation", val),
        ("Test", test),
    ]:
        result = con.execute(
            f"""
            WITH train_items AS (
                SELECT DISTINCT item_id
                FROM read_parquet('{train}')
            ),

            target AS (
                SELECT *
                FROM read_parquet('{path}')
            )

            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (
                    WHERE ti.item_id IS NULL
                ) AS cold
            FROM target AS t
            LEFT JOIN train_items AS ti
                USING (item_id)
            """
        ).fetchone()

        total, cold = result

        print(
            f"{name:<10} "
            f"cold={cold:>8,} / {total:,} "
            f"({cold / total:.4%})"
        )

    print()
    print("Temporal validation")
    print("-" * 50)

    violations = con.execute(
        f"""
        WITH train_last AS (
            SELECT
                user_id,
                MAX(timestamp) AS train_time
            FROM read_parquet('{train}')
            GROUP BY user_id
        ),

        val AS (
            SELECT
                user_id,
                timestamp AS val_time
            FROM read_parquet('{val}')
        ),

        test AS (
            SELECT
                user_id,
                timestamp AS test_time
            FROM read_parquet('{test}')
        )

        SELECT COUNT(*)
        FROM train_last
        JOIN val USING (user_id)
        JOIN test USING (user_id)
        WHERE train_time > val_time
           OR val_time > test_time
        """
    ).fetchone()[0]

    print(
        f"Temporal violations: {violations:,}"
    )

    con.close()


if __name__ == "__main__":
    main()
