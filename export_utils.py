"""Excel and PDF export of the prediction run."""

import io

import matplotlib
matplotlib.use("Agg")  # headless rendering, no GUI backend needed
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet


def export_to_excel(predictions_df: pd.DataFrame, metrics_df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        predictions_df.to_excel(writer, sheet_name="Predictions", index=True)
        metrics_df.to_excel(writer, sheet_name="Metrics", index=True)
    return buffer.getvalue()


def _render_comparison_chart_png(predictions_df: pd.DataFrame) -> bytes:
    fig, ax = plt.subplots(figsize=(8, 4))
    for col in predictions_df.columns:
        ax.plot(predictions_df.index, predictions_df[col], label=col, linewidth=1.5)
    ax.set_title("Predicted vs Real Price")
    ax.set_xlabel("Test sample index")
    ax.set_ylabel("Price")
    ax.legend(fontsize=8)
    fig.tight_layout()

    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format="png", dpi=150)
    plt.close(fig)
    img_buffer.seek(0)
    return img_buffer.read()


def export_to_pdf(ticker: str, predictions_df: pd.DataFrame, metrics_df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = [
        Paragraph(f"Stock Price Predictor Report -- {ticker}", styles["Title"]),
        Spacer(1, 12),
    ]

    chart_png = _render_comparison_chart_png(predictions_df)
    elements.append(Image(io.BytesIO(chart_png), width=6.5 * inch, height=3.25 * inch))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Model Metrics", styles["Heading2"]))
    table_data = [["Model", "RMSE", "MAE", "MAPE (%)"]]
    for name, row in metrics_df.iterrows():
        table_data.append([name, f"{row['RMSE']:.2f}", f"{row['MAE']:.2f}", f"{row['MAPE']:.2f}"])

    table = Table(table_data, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2a78d6")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f4f4")]),
    ]))
    elements.append(table)

    doc.build(elements)
    return buffer.getvalue()
