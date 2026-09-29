from pathlib import Path

import duckdb


TRAIN = Path(
    "data/processed/collab_5core/train.parquet"
)

DB = Path(
    "data/processed/itemcf_full.duckdb"
)

PARTS = Path(
    "data/processed/itemcf_top100_parts"
)

OUTPUT = Path(
    "data/processed/itemcf_neighbors_top100.parquet"
)

NUM_BUCKETS = 64


def main():
    PARTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    con = duckdb.connect(str(DB))

    # WSL currently only sees about 6.7 GiB RAM.
    con.execute("SET memory_limit='4GB'")
    con.execute("SET threads=4")
    con.execute(
        "SET temp_directory='data/processed/duckdb_tmp'"
    )

    print("=" * 60)
    print("FashionRec V2 - Full ItemCF (Bucketed)")
    print("=" * 60)

    # -------------------------------------------------
    # 1. Item frequency
    # -------------------------------------------------

    print("\n[1] Building item frequencies...")

    con.execute(
        """
        DROP TABLE IF EXISTS item_counts
        """
    )

    con.execute(
        f"""
        CREATE TABLE item_counts AS

        SELECT
            item_id,
            COUNT(*) AS item_count

        FROM read_parquet('{TRAIN}')

        GROUP BY item_id
        """
    )

    # -------------------------------------------------
    # 2. Co-occurrence
    # -------------------------------------------------

    print("[2] Building item co-occurrence...")

    con.execute(
        """
        DROP TABLE IF EXISTS cooccurrence
        """
    )

    con.execute(
        f"""
        CREATE TABLE cooccurrence AS

        SELECT
            a.item_id AS item_i,
            b.item_id AS item_j,
            COUNT(*) AS co_count

        FROM read_parquet('{TRAIN}') AS a

        JOIN read_parquet('{TRAIN}') AS b
          ON a.user_id = b.user_id
         AND a.item_id < b.item_id

        GROUP BY
            a.item_id,
            b.item_id
        """
    )

    stats = con.execute(
        """
        SELECT
            COUNT(*),
            SUM(co_count),
            MAX(co_count)
        FROM cooccurrence
        """
    ).fetchone()

    print(
        f"Unique item pairs : {stats[0]:,}"
    )
    print(
        f"Pair events       : {stats[1]:,}"
    )
    print(
        f"Max co-occurrence : {stats[2]:,}"
    )

    # -------------------------------------------------
    # 3. Process Top-K bucket by bucket
    # -------------------------------------------------

    print()
    print(
        f"[3] Building Top-100 neighbors "
        f"in {NUM_BUCKETS} buckets..."
    )

    for bucket in range(NUM_BUCKETS):

        part = PARTS / (
            f"neighbors_{bucket:03d}.parquet"
        )

        print(
            f"Bucket "
            f"{bucket + 1:02d}/{NUM_BUCKETS}...",
            flush=True,
        )

        con.execute(
            f"""
            COPY (

                WITH directional AS (

                    SELECT
                        c.item_i AS item_id,
                        c.item_j AS neighbor_id,
                        c.co_count

                    FROM cooccurrence AS c

                    WHERE
                        hash(c.item_i)
                        % {NUM_BUCKETS}
                        = {bucket}

                    UNION ALL

                    SELECT
                        c.item_j AS item_id,
                        c.item_i AS neighbor_id,
                        c.co_count

                    FROM cooccurrence AS c

                    WHERE
                        hash(c.item_j)
                        % {NUM_BUCKETS}
                        = {bucket}
                ),

                scored AS (

                    SELECT
                        d.item_id,
                        d.neighbor_id,
                        d.co_count,

                        d.co_count
                        / SQRT(
                            CAST(i.item_count AS DOUBLE)
                            *
                            CAST(j.item_count AS DOUBLE)
                        ) AS similarity

                    FROM directional AS d

                    JOIN item_counts AS i
                      ON d.item_id = i.item_id

                    JOIN item_counts AS j
                      ON d.neighbor_id = j.item_id
                ),

                ranked AS (

                    SELECT
                        *,

                        ROW_NUMBER() OVER (
                            PARTITION BY item_id
                            ORDER BY
                                similarity DESC,
                                co_count DESC,
                                neighbor_id
                        ) AS neighbor_rank

                    FROM scored
                )

                SELECT
                    item_id,
                    neighbor_id,
                    co_count,
                    similarity

                FROM ranked

                WHERE neighbor_rank <= 100

            )
            TO '{part}'
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """
        )

    # -------------------------------------------------
    # 4. Merge parts
    # -------------------------------------------------

    print()
    print("[4] Merging bucket files...")

    con.execute(
        f"""
        COPY (
            SELECT *
            FROM read_parquet(
                '{PARTS}/*.parquet'
            )
        )
        TO '{OUTPUT}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        )
        """
    )

    final = con.execute(
        f"""
        SELECT
            COUNT(*) AS edges,
            COUNT(DISTINCT item_id) AS items
        FROM read_parquet('{OUTPUT}')
        """
    ).fetchone()

    avg = (
        final[0] / final[1]
        if final[1]
        else 0
    )

    print()
    print("=" * 60)
    print("ItemCF Neighbor Index")
    print("=" * 60)

    print(
        f"Neighbor edges : {final[0]:,}"
    )
    print(
        f"Items indexed  : {final[1]:,}"
    )
    print(
        f"Avg neighbors  : {avg:.2f}"
    )
    print(
        f"Saved          : {OUTPUT}"
    )

    con.close()


if __name__ == "__main__":
    main()
