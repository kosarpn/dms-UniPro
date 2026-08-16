from ml.dataset_builder import DatasetBuilder
def main():
    builder = DatasetBuilder()
    # notdrowsy
    print("در حال پردازش notdrowsy...")
    builder.process_folder("data/raw/train/notdrowsy", label=0)
    # drowsy (با پشتیبانی از زیرپوشه‌ها)
    drowsy_root = "data/raw/train/drowsy"
    for folder_name in ["yawning", "slowBlinkWithNodding", "sleepyCombination"]:
        folder_path = f"{drowsy_root}/{folder_name}"
        print(f"در حال پردازش {folder_name}...")
        builder.process_folder(folder_path, label=1)
    # ذخیره
    builder.save_csv("data/processed/dataset.csv")
if __name__ == "__main__":
    main()
 