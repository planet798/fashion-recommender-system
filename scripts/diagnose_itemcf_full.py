from pathlib import Path

import duckdb


ROOT = Path("data/processed/collab_5core")

TRAIN = ROOT / "train.parquet"
VALIDATION = ROOT / "validation.parquet"

NEIGHBORS = Path(
    "data/processed/itemcf_neighbors_top100.parquet"
)


def main():
    con = duckdb.connect()

    con.execute("SET memory_limit='4GB'")
    con.execute("SET threads=4")
    con.execute(
        "SET temp_directory='data/processed/duckdb_tmp'"
    )

    print("=" * 60)
    print("FashionRec V2 - Full ItemCF Diagnostics")
    print("=" * 60)

    print("\nChecking warm target reachability...")

    result = con.execute(
        f"""
        WITH train_items AS (
            SELECT DISTINCT item_id
            FROM read_parquet('{TRAIN}')
        ),

        warm_validation AS (
            SELECT
                v.user_id,
                v.item_id AS target_item

            FROM read_parquet('{VALIDATION}') AS v

            INNER JOIN train_items AS ti
                ON v.item_id = ti.item_id
        ),

        reachable AS (
            SELECT DISTINCT
                v.user_id

            FROM warm_validation AS v

            INNER JOIN read_parquet('{TRAIN}') AS h
                ON v.user_id = h.user_id

            INNER JOIN read_parquet('{NEIGHBORS}') AS n
                ON h.item_id = n.item_id
               AND v.target_item = n.neighbor_id
        )

        SELECT
            (SELECT COUNT(*) FROM warm_validation)
                AS warm_users,

            (SELECT COUNT(*) FROM reachable)
                AS reachable_users
        """
    ).fetchone()

    warm_users = result[0]
    reachable = result[1]

    print()
    print(f"Warm users             : {warm_users:,}")
    print(f"Reachable warm targets : {reachable:,}")
    print(
        f"Warm reachability rate : "
        f"{reachable / warm_users:.4%}"
    )

    con.close()


if __name__ == "__main__":
    main()
