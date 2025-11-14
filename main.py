from src.preprocess import load_and_preprocess
from src.train_regression import train_regression
from src.train_classification import train_classification

DATA_PATH = "data/flights.csv"

def main():
    print("\n📥 Loading and preprocessing data...")
    X, y_reg, y_clf, preprocess = load_and_preprocess(DATA_PATH)

    print("\n📊 Training Regression Model...")
    reg_model = train_regression(X, y_reg, preprocess)

    print("\n🔍 Training Classification Model...")
    clf_model = train_classification(X, y_clf, preprocess)

    print("\n🎉 Training Complete!")

if __name__ == "__main__":
    main()
