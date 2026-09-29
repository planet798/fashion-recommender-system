from pathlib import Path

import duckdb


RAW_PATH = Path(
    "data/raw/Clothing_Shoes_and_Jewelry.csv.gz"
)


def main():
    con = duckdb.connect()

    raw_path = str(RAW_PATH)

    print("=" * 60)
    print("FashionRec V2 - Full Dataset Profile")
    print("=" * 60)

    print("\n[1] Raw dataset")

    raw_stats = con.execute(
        f"""
        SELECT
            COUNT(*) AS interactions,
            COUNT(DISTINCT user_id) AS users,
            COUNT(DISTINCT parent_asin) AS items,
            MIN(timestamp) AS min_timestamp,
            MAX(timestamp) AS max_timestamp
        FROM read_csv_auto(
            '{raw_path}',
            compression='gzip'
        )
        """
    ).fetchone()

    print(f"Interactions : {raw_stats[0]:,}")
    print(f"Users        : {raw_stats[1]:,}")
    print(f"Items        : {raw_stats[2]:,}")
    print(f"Min timestamp: {raw_stats[3]}")
    print(f"Max timestamp: {raw_stats[4]}")

    print("\n[2] Rating distribution")

    rating_rows = con.execute(
        f"""
        SELECT
            rating,
            COUNT(*) AS count
        FROM read_csv_auto(
            '{raw_path}',
            compression='gzip'
        )
        GROUP BY rating
        ORDER BY rating
        """
    ).fetchall()

    total = sum(row[1] for row in rating_rows)

    for rating, count in rating_rows:
        print(
            f"{rating:.1f} star : "
            f"{count:,} "
            f"({count / total:.4%})"
        )

    print("\n[3] Positive feedback: rating >= 4")

    positive_stats = con.execute(
        f"""
        SELECT
            COUNT(*) AS interactions,
            COUNT(DISTINCT user_id) AS users,
            COUNT(DISTINCT parent_asin) AS items
        FROM read_csv_auto(
            '{raw_path}',
            compression='gzip'
        )
        WHERE rating >= 4
        """
    ).fetchone()

    print(
        f"Positive interactions : "
        f"{positive_stats[0]:,}"
    )
    print(
        f"Positive users        : "
        f"{positive_stats[1]:,}"
    )
    print(
        f"Positive items        : "
        f"{positive_stats[2]:,}"
    )

    print("\n[4] Positive interactions per user")

    user_stats = con.execute(
        f"""
        WITH user_counts AS (
            SELECT
                user_id,
                COUNT(*) AS n
            FROM read_csv_auto(
                '{raw_path}',
                compression='gzip'
            )
            WHERE rating >= 4
            GROUP BY user_id
        )
        SELECT
            COUNT(*) AS users,
            AVG(n) AS mean_n,
            MIN(n) AS min_n,
            MEDIAN(n) AS median_n,
            QUANTILE_CONT(n, 0.75) AS p75,
            MAX(n) AS max_n
        FROM user_counts
        """
    ).fetchone()

    print(f"Users  : {user_stats[0]:,}")
    print(f"Mean   : {user_stats[1]:.4f}")
    print(f"Min    : {user_stats[2]:,}")
    print(f"Median : {user_stats[3]:.2f}")
    print(f"P75    : {user_stats[4]:.2f}")
    print(f"Max    : {user_stats[5]:,}")

    print("\n[5] Users with >= 5 positive interactions")

    filtered = con.execute(
        f"""
        WITH user_counts AS (
            SELECT
                user_id,
                COUNT(*) AS n
            FROM read_csv_auto(
                '{raw_path}',
                compression='gzip'
            )
            WHERE rating >= 4
            GROUP BY user_id
        )
        SELECT
            COUNT(*) AS users,
            SUM(n) AS interactions
        FROM user_counts
        WHERE n >= 5
        """
    ).fetchone()

    print(
        f"Eligible users        : "
        f"{filtered[0]:,}"
    )
    print(
        f"Eligible interactions : "
        f"{filtered[1]:,}"
    )


if __name__ == "__main__":
    main()
