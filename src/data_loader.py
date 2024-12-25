import pandas as pd

def load_data(filepath):
    """
    Load dữ liệu từ file CSV và xử lý lỗi nếu có.
    """
    try:
        # Đọc file CSV
        data = pd.read_csv(filepath, sep='\t', header=None, names=['label', 'message'], on_bad_lines='skip')
        if data.empty:
            raise ValueError("File dữ liệu trống.")
        # Xử lý dữ liệu
        data = data.dropna()  # Loại bỏ dòng trống
        data['label'] = data['label'].map({'ham': 0, 'spam': 1})  # Chuyển đổi nhãn sang 0/1
        return data
    except Exception as e:
        print(f"[ERROR] Lỗi khi tải dữ liệu: {e}")
        return None


if __name__ == "__main__":
    from common import DATA_PATH
    data = load_data(DATA_PATH)
    if data is not None:
        print(data.head())
