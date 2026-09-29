from pathlib import Path

import duckdb


DATA = Path(
    "data/processed/interactions_full_positive.parquet"
)


def main():
    con = duckdb.connect()

    print("=" * 60)
    print("FashionRec V2 - Full Filtered Profile")
    print("=" * 60)

    stats = con.execute(
        f"""
        WITH item_counts AS (
            SELECT
                item_id,
                COUNT(*) AS n
            FROM read_parquet('{DATA}')
            GROUP BY item_id
        )

        SELECT
            COUNT(*) AS items,
            AVG(n) AS mean_n,
            MIN(n) AS min_n,
            MEDIAN(n) AS median_n,
            QUANTILE_CONT(n, 0.75) AS p75,
            QUANTILE_CONT(n, 0.90) AS p90,
            QUANTILE_CONT(n, 0.95) AS p95,
            MAX(n) AS max_n
        FROM item_counts
        """
    ).fetchone()

    print()
    print("Item interaction distribution")
    print("-" * 50)

    print(f"Items  : {stats[0]:,}")
    print(f"Mean   : {stats[1]:.4f}")
    print(f"Min    : {stats[2]:,}")
    print(f"Median : {stats[3]:.2f}")
    print(f"P75    : {stats[4]:.2f}")
    print(f"P90    : {stats[5]:.2f}")
    print(f"P95    : {stats[6]:.2f}")
    print(f"Max    : {stats[7]:,}")

    print()
    print("Candidate item-frequency thresholds")
    print("-" * 50)

    rows = con.execute(
        f"""
        WITH item_counts AS (
            SELECT
                item_id,
                COUNT(*) AS n
            FROM read_parquet('{DATA}')
            GROUP BY item_id
        )

        SELECT
            threshold,
            COUNT(*) FILTER (
                WHERE n >= threshold
            ) AS items,

            SUM(
                CASE
                    WHEN n >= threshold
                    THEN n
                    ELSE 0
                END
            ) AS interactions

        FROM item_counts
        CROSS JOIN (
            VALUES
                (1),
                (2),
                (3),
                (5),
                (10),
                (20)
        ) AS thresholds(threshold)

        GROUP BY threshold
        ORDER BY threshold
        """
    ).fetchall()

    for threshold, items, interactions in rows:
        print(
            f"item >= {threshold:<2} | "
            f"items={items:>8,} | "
            f"interactions={interactions:>10,}"
        )


if __name__ == "__main__":
    main()
