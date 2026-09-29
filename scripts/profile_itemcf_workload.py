from pathlib import Path

import duckdb


TRAIN = Path(
    "data/processed/collab_5core/train.parquet"
)


def main():
    con = duckdb.connect()

    print("=" * 60)
    print("FashionRec V2 - ItemCF Workload Profile")
    print("=" * 60)

    row = con.execute(
        f"""
        WITH user_counts AS (
            SELECT
                user_id,
                COUNT(*) AS n
            FROM read_parquet('{TRAIN}')
            GROUP BY user_id
        )
        SELECT
            COUNT(*) AS users,
            AVG(n) AS mean_n,
            MEDIAN(n) AS median_n,
            QUANTILE_CONT(n, 0.75) AS p75,
            QUANTILE_CONT(n, 0.90) AS p90,
            QUANTILE_CONT(n, 0.95) AS p95,
            QUANTILE_CONT(n, 0.99) AS p99,
            MAX(n) AS max_n,
            SUM(
                CAST(n AS HUGEINT)
                * (n - 1)
                / 2
            ) AS pair_events
        FROM user_counts
        """
    ).fetchone()

    print()
    print("User history distribution")
    print("-" * 50)

    print(f"Users       : {row[0]:,}")
    print(f"Mean        : {row[1]:.4f}")
    print(f"Median      : {row[2]:.2f}")
    print(f"P75         : {row[3]:.2f}")
    print(f"P90         : {row[4]:.2f}")
    print(f"P95         : {row[5]:.2f}")
    print(f"P99         : {row[6]:.2f}")
    print(f"Max         : {row[7]:,}")
    print(f"Pair events : {int(row[8]):,}")

    print()
    print("Long-history users")
    print("-" * 50)

    rows = con.execute(
        f"""
        WITH user_counts AS (
            SELECT
                user_id,
                COUNT(*) AS n
            FROM read_parquet('{TRAIN}')
            GROUP BY user_id
        )
        SELECT
            threshold,
            COUNT(*) FILTER (
                WHERE n > threshold
            ) AS users
        FROM user_counts
        CROSS JOIN (
            VALUES
                (20),
                (50),
                (100),
                (200)
        ) AS t(threshold)
        GROUP BY threshold
        ORDER BY threshold
        """
    ).fetchall()

    for threshold, users in rows:
        print(
            f"history > {threshold:<3} : "
            f"{users:,} users"
        )

    con.close()


if __name__ == "__main__":
    main()
