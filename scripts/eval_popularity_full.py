from pathlib import Path

import duckdb


ROOT = Path("data/processed/collab_5core")

TRAIN = ROOT / "train.parquet"
VALIDATION = ROOT / "validation.parquet"


def main():
    con = duckdb.connect()

    print("=" * 60)
    print("FashionRec V2 - Full Popularity Baseline")
    print("=" * 60)

    print("\nBuilding global popularity ranking...")

    con.execute(
        f"""
        CREATE TEMP TABLE popularity AS

        SELECT
            item_id,
            COUNT(*) AS interaction_count,

            ROW_NUMBER() OVER (
                ORDER BY COUNT(*) DESC, item_id
            ) AS global_rank

        FROM read_parquet('{TRAIN}')
        GROUP BY item_id
        """
    )

    print("Evaluating validation users...")

    con.execute(
        f"""
        CREATE TEMP TABLE target_ranks AS

        WITH targets AS (
            SELECT
                v.user_id,
                v.item_id AS target_item,
                p.global_rank AS target_global_rank

            FROM read_parquet('{VALIDATION}') AS v

            LEFT JOIN popularity AS p
                ON v.item_id = p.item_id
        ),

        user_history AS (
            SELECT
                t.user_id,
                t.target_item,
                t.target_global_rank,

                COUNT(DISTINCT h.item_id)
                    FILTER (
                        WHERE hp.global_rank
                              < t.target_global_rank
                    ) AS seen_ahead,

                MAX(
                    CASE
                        WHEN h.item_id = t.target_item
                        THEN 1
                        ELSE 0
                    END
                ) AS target_seen

            FROM targets AS t

            LEFT JOIN read_parquet('{TRAIN}') AS h
                ON t.user_id = h.user_id

            LEFT JOIN popularity AS hp
                ON h.item_id = hp.item_id

            GROUP BY
                t.user_id,
                t.target_item,
                t.target_global_rank
        )

        SELECT
            user_id,
            target_item,

            CASE
                WHEN target_global_rank IS NULL
                    THEN NULL

                WHEN target_seen = 1
                    THEN NULL

                ELSE
                    target_global_rank - seen_ahead
            END AS recommendation_rank,

            target_global_rank IS NOT NULL AS is_warm

        FROM user_history
        """
    )

    for scope, condition in [
        ("Overall", "TRUE"),
        ("Warm", "is_warm"),
        ("Cold", "NOT is_warm"),
    ]:
        row = con.execute(
            f"""
            SELECT
                COUNT(*) AS users,

                AVG(
                    CASE
                        WHEN recommendation_rank <= 5
                        THEN 1.0 ELSE 0.0
                    END
                ) AS hr5,

                AVG(
                    CASE
                        WHEN recommendation_rank <= 10
                        THEN 1.0 ELSE 0.0
                    END
                ) AS hr10,

                AVG(
                    CASE
                        WHEN recommendation_rank <= 20
                        THEN 1.0 ELSE 0.0
                    END
                ) AS hr20,

                AVG(
                    CASE
                        WHEN recommendation_rank <= 5
                        THEN 1.0
                             / LOG2(recommendation_rank + 1)
                        ELSE 0.0
                    END
                ) AS ndcg5,

                AVG(
                    CASE
                        WHEN recommendation_rank <= 10
                        THEN 1.0
                             / LOG2(recommendation_rank + 1)
                        ELSE 0.0
                    END
                ) AS ndcg10,

                AVG(
                    CASE
                        WHEN recommendation_rank <= 20
                        THEN 1.0
                             / LOG2(recommendation_rank + 1)
                        ELSE 0.0
                    END
                ) AS ndcg20,

                AVG(
                    CASE
                        WHEN recommendation_rank <= 5
                        THEN 1.0 / recommendation_rank
                        ELSE 0.0
                    END
                ) AS mrr5,

                AVG(
                    CASE
                        WHEN recommendation_rank <= 10
                        THEN 1.0 / recommendation_rank
                        ELSE 0.0
                    END
                ) AS mrr10,

                AVG(
                    CASE
                        WHEN recommendation_rank <= 20
                        THEN 1.0 / recommendation_rank
                        ELSE 0.0
                    END
                ) AS mrr20

            FROM target_ranks
            WHERE {condition}
            """
        ).fetchone()

        print()
        print(scope)
        print("-" * 50)

        print(f"Users   : {row[0]:,}")
        print()

        print(f"HR@5    = {row[1]:.6f}")
        print(f"NDCG@5  = {row[4]:.6f}")
        print(f"MRR@5   = {row[7]:.6f}")
        print()

        print(f"HR@10   = {row[2]:.6f}")
        print(f"NDCG@10 = {row[5]:.6f}")
        print(f"MRR@10  = {row[8]:.6f}")
        print()

        print(f"HR@20   = {row[3]:.6f}")
        print(f"NDCG@20 = {row[6]:.6f}")
        print(f"MRR@20  = {row[9]:.6f}")

    con.close()


if __name__ == "__main__":
    main()
