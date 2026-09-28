from fashionrec.data.preprocess import build_dev_dataset


if __name__ == "__main__":
    build_dev_dataset(
        input_path="data/raw/Clothing_Shoes_and_Jewelry.csv.gz",
        output_path="data/processed/interactions_dev.parquet",
        modulo=20,
    )
