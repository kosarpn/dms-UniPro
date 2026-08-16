from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_PATH = PROJECT_ROOT / "results" / "evaluation_results.csv"

FIGURES_DIR = PROJECT_ROOT / "figures"


class ResultVisualizer:

    def __init__(self):
        FIGURES_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.df = pd.read_csv(
            RESULTS_PATH
        )

    # ==========================
    # Generic Bar Plot
    # ==========================

    def plot_metric(self, metric):

        plt.figure(figsize=(6, 4))

        plt.bar(
            self.df["Model"],
            self.df[metric],
        )

        plt.title(metric)

        plt.ylabel(metric)

        plt.ylim(0, 1)

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR / f"{metric.lower()}.png",
            dpi=300,
        )

        plt.close()

    # ==========================
    # Grouped Bar Chart
    # ==========================

    def plot_grouped_metrics(self):

        metrics = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
        ]

        df = self.df.set_index("Model")[metrics]

        ax = df.plot(
            kind="bar",
            figsize=(9, 5),
        )

        ax.set_ylim(0, 1)

        ax.set_ylabel("Score")

        ax.set_title("Classifier Comparison")

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR / "grouped_metrics.png",
            dpi=300,
        )

        plt.close()

    # ==========================
    # Generate All Figures
    # ==========================

    def generate(self):

        self.plot_metric("Accuracy")

        self.plot_metric("Precision")

        self.plot_metric("Recall")

        self.plot_metric("F1")

        self.plot_grouped_metrics()


def main():

    visualizer = ResultVisualizer()

    visualizer.generate()

    print("All figures saved successfully.")


if __name__ == "__main__":
    main()