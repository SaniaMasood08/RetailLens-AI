# Repository Guidelines

## Project Overview

RetailLens AI is an internship Project 8: an industry-ready, end-to-end machine learning system for retail business intelligence, using the UCI Online Retail dataset. The planned workflow runs from business framing and dataset understanding through cleaning, exploratory analysis, feature engineering, time-aware evaluation, model comparison, sales prediction, business insights, visualization, and documentation. Implement it incrementally; do not generate the complete application unless explicitly asked.

## Project Structure & Workflow

Inspect the current repository before each major implementation step and follow its existing structure. Keep code modular, readable, beginner-friendly, and professionally organized; make small changes that are easy to review and test. No dataset, model, dashboard, or application setup is implied by this guide. Do not create unrelated files or download data unless requested. Use Python with Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn as appropriate; consider Streamlit later for an interactive dashboard.

## Forecasting Rules

The initial task is to predict next-day total recorded sales revenue using only information available before that prediction day. Treat this as time-based forecasting: preserve chronology in train/validation/test splits, and ensure every feature is computed only from data available at prediction time. Avoid random splits and any future-data leakage. Revenue may be derived as `Quantity × UnitPrice`; call it revenue, not profit. Do not imply the dataset contains costs, margins, or other fields it does not contain. State important assumptions and document how the target and features are defined.

## Analysis and Reporting

Never invent dataset statistics, model results, metrics, predictions, or business insights. Derive reported values from actual data and executed analysis, and distinguish measured results from assumptions or proposals. Explain evaluation choices and compare models on the same time-aware holdout design. Keep charts and dashboard labels faithful to what the data measures.

## Development, Testing & Style

Use clear Python names, consistent formatting, and small functions that can be checked independently. Add tests for data cleaning, feature timing, target construction, and evaluation logic as those components are implemented. Run relevant checks after changes and report only commands actually run; inspect the available project configuration before choosing commands. Document reproducible setup and workflow steps when project files are introduced.

## Security & Configuration

Keep secrets, API keys, credentials, and machine-specific configuration out of version control. Use documented environment variables and safe example configuration if external services are added.
