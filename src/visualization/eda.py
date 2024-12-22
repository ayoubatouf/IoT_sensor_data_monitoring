from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

from config.paths import RAW_DATA_PATH, REPORTS_PATH


def plot_columns_dist(data, save_path):
    columns_to_plot = data.columns

    num_columns = len(columns_to_plot)
    num_rows = (num_columns // 4) + (num_columns % 4 > 0)

    plt.figure(figsize=(16, num_rows * 4))

    for i, column in enumerate(columns_to_plot, 1):
        plt.subplot(num_rows, 4, i)
        sns.histplot(data[column], kde=True, bins=20)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(save_path / "column_distributions.png")
    plt.close()


def plot_columns_box(data, save_path):
    columns_to_plot = data.columns

    num_columns = len(columns_to_plot)
    num_rows = (num_columns // 3) + (num_columns % 3 > 0)

    plt.figure(figsize=(16, num_rows * 4))

    for i, column in enumerate(columns_to_plot, 1):
        plt.subplot(num_rows, 3, i)
        sns.boxplot(x=data[column])
        plt.title(f"Boxplot of {column}")
        plt.xlabel(column)

    plt.tight_layout()
    plt.savefig(save_path / "column_boxplots.png")
    plt.close()


def plot_correlation(data, save_path):
    if "fail" in data.columns:
        correlation_matrix = data.drop(columns=["fail"]).corr()
    else:
        correlation_matrix = data.corr()

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        vmin=-1,
        vmax=1,
    )

    plt.title('Correlation Matrix Heatmap (Excluding "fail")')
    plt.savefig(save_path / "correlation_matrix_heatmap.png")
    plt.close()


def plot_mutual_info(data, save_path):
    if "fail" in data.columns:
        X = data.drop(columns=["fail"])
        y = data["fail"]
    else:
        print("The 'fail' column is missing.")
        return

    mutual_info = mutual_info_classif(X, y)

    mi_df = pd.DataFrame({"Feature": X.columns, "Mutual Information": mutual_info})

    mi_df = mi_df.sort_values(by="Mutual Information", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(mi_df["Feature"], mi_df["Mutual Information"], color="skyblue")
    plt.xlabel("Mutual Information")
    plt.title("Mutual Information Between Features and Target")
    plt.gca().invert_yaxis()
    plt.savefig(save_path / "mutual_information.png")
    plt.close()


if __name__ == "__main__":
    data = pd.read_csv(RAW_DATA_PATH)
    output_path = REPORTS_PATH

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    plot_columns_box(data, output_path)
    plot_columns_dist(data, output_path)
    plot_correlation(data, output_path)
    plot_mutual_info(data, output_path)
