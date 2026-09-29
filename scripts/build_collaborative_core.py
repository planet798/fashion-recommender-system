from pathlib import Path

import duckdb


INPUT = Path(
    "data/processed/interactions_full_positive.parquet"
)

OUTPUT = Path(
    "data/processed/interactions_collab_5core.parquet"
)

DB_PATH = Path(
    "data/processed/build_collab.duckdb"
)


def main():
    con = duckdb.connect(str(DB_PATH))

    print("=" * 60)
    print("FashionRec V2 - Collaborative 5-Core")
    print("=" * 60)

    con.execute("DROP TABLE IF EXISTS working")
    con.execute("DROP TABLE IF EXISTS next_working")

    print("\nLoading full positive dataset...")

    con.execute(
        f"""
        CREATE TABLE working AS
        SELECT
            user_id,
            item_id,
            rating,
            timestamp
        FROM read_parquet('{INPUT}')
        """
    )

    iteration = 0

    while True:
        iteration += 1

        before = con.execute(
            """
            SELECT COUNT(*)
            FROM working
            """
        ).fetchone()[0]

        con.execute(
            "DROP TABLE IF EXISTS next_working"
        )

        con.execute(
            """
            CREATE TABLE next_working AS

            WITH eligible_users AS (
                SELECT user_id
                FROM working
                GROUP BY user_id
                HAVING COUNT(*) >= 5
            ),

            user_filtered AS (
                SELECT w.*
                FROM working AS w
                INNER JOIN eligible_users AS u
                    USING (user_id)
            ),

            eligible_items AS (
                SELECT item_id
                FROM user_filtered
                GROUP BY item_id
                HAVING COUNT(*) >= 5
            )

            SELECT uf.*
            FROM user_filtered AS uf
            INNER JOIN eligible_items AS i
                USING (item_id)
            """
        )

        after = con.execute(
            """
            SELECT COUNT(*)
            FROM next_working
            """
        ).fetchone()[0]

        users = con.execute(
            """
            SELECT COUNT(DISTINCT user_id)
            FROM next_working
            """
        ).fetchone()[0]

        items = con.execute(
            """
            SELECT COUNT(DISTINCT item_id)
            FROM next_working
            """
        ).fetchone()[0]

        print(
            f"Iteration {iteration:02d} | "
            f"interactions={after:,} | "
            f"users={users:,} | "
            f"items={items:,} | "
            f"removed={before-after:,}"
        )

        con.execute(
            "DROP TABLE working"
        )

        con.execute(
            """
            ALTER TABLE next_working
            RENAME TO working
            """
        )

        if before == after:
            break

    print("\nWriting parquet...")

    con.execute(
        f"""
        COPY (
            SELECT *
            FROM working
            ORDER BY user_id, timestamp
        )
        TO '{OUTPUT}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        )
        """
    )

    print(f"\nSaved: {OUTPUT}")

    print("\nFinal validation")

    validation = con.execute(
        """
        SELECT
            MIN(user_n),
            MIN(item_n)
        FROM (
            SELECT
                user_id,
                item_id,
                COUNT(*) OVER (
                    PARTITION BY user_id
                ) AS user_n,
                COUNT(*) OVER (
                    PARTITION BY item_id
                ) AS item_n
            FROM working
        )
        """
    ).fetchone()

    print(
        f"Minimum user interactions: "
        f"{validation[0]}"
    )

    print(
        f"Minimum item interactions: "
        f"{validation[1]}"
    )

    con.close()


if __name__ == "__main__":
    main()
