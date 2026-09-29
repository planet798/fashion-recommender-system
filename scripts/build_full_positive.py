from pathlib import Path

import duckdb


RAW = Path(
    "data/raw/Clothing_Shoes_and_Jewelry.csv.gz"
)

OUTPUT = Path(
    "data/processed/interactions_full_positive.parquet"
)


def main():
    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    con = duckdb.connect()

    print("=" * 60)
    print("FashionRec V2 - Build Full Positive Dataset")
    print("=" * 60)

    print(f"Input : {RAW}")
    print(f"Output: {OUTPUT}")
    print()

    con.execute(
        f"""
        COPY (
            WITH positive AS (
                SELECT
                    user_id,
                    parent_asin AS item_id,
                    CAST(rating AS FLOAT) AS rating,
                    CAST(timestamp AS BIGINT) AS timestamp
                FROM read_csv_auto(
                    '{RAW}',
                    compression='gzip'
                )
                WHERE rating >= 4
            ),

            eligible_users AS (
                SELECT
                    user_id
                FROM positive
                GROUP BY user_id
                HAVING COUNT(*) >= 5
            )

            SELECT
                p.user_id,
                p.item_id,
                p.rating,
                p.timestamp
            FROM positive AS p
            INNER JOIN eligible_users AS u
                ON p.user_id = u.user_id
        )
        TO '{OUTPUT}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        )
        """
    )

    stats = con.execute(
        f"""
        SELECT
            COUNT(*),
            COUNT(DISTINCT user_id),
            COUNT(DISTINCT item_id)
        FROM read_parquet('{OUTPUT}')
        """
    ).fetchone()

    print(f"Interactions : {stats[0]:,}")
    print(f"Users        : {stats[1]:,}")
    print(f"Items        : {stats[2]:,}")


if __name__ == "__main__":
    main()
