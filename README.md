# DataProbe

DataProbe is an AI-powered exploratory data analysis (EDA) and report-generation toolkit. It orchestrates multiple specialized agents to clean tabular data, create visualizations and observations, engineer features, recommend machine-learning models, evaluate them, and generate a PDF report.

## Features

- Automated data cleaning with optional natural-language instructions
- AI-generated visualizations and written observations
- Temporal and correlation analysis
- Automated feature engineering
- Problem-type detection and model recommendations
- Model training, comparison, and best-model selection
- Export of cleaned and engineered datasets
- JSON results, plots, logs, and PDF report generation
- Optional human-in-the-loop review

## Pipeline

```text
Raw Data
   |
   v
Data Cleaning
   |
   v
Visualization & Observations
   |
   v
Feature Engineering
   |
   v
Model Recommendation
   |
   v
Model Evaluation
   |
   v
PDF Report + Data + Plots + JSON Results
```

## Implementation and Architecture

DataProbe uses a modular, agent-based architecture built around LangChain and LangGraph. Each major EDA responsibility is isolated in a specialized agent, while `AutomatedEDAOrchestrator` coordinates those agents into a sequential end-to-end workflow.

### Architectural Overview

```text
                         AutomatedEDAOrchestrator
                                    |
              +---------------------+---------------------+
              |                     |                     |
              v                     v                     v
       Agent Coordination      Artifact Storage      Result Tracking
              |                     |                     |
              v                     v                     v
    +-------------------+     data / plots / models   eda_results.json
    | DataCleaningAgent |            / logs           and PDF report
    +---------+---------+
              |
              v
    +-------------------------+
    | MultiDataVisualObsAgent |
    +------------+------------+
                 |
                 v
          +-------------+
          |  EDAAgent   |
          +------+------+ 
                 |
                 v
    +--------------------------+
    | ModelRecommendationAgent |
    +-------------+------------+
                  |
                  v
       +----------------------+
       | ModelEvaluationAgent |
       +----------------------+
```

### Orchestration Layer

`DataProbe/orchestration.py` contains `AutomatedEDAOrchestrator`, the main entry point for a complete analysis. During initialization, it:

1. Validates that a language model is supplied for every pipeline stage.
2. Creates directories for data, plots, models, and other generated artifacts.
3. Initializes each specialized agent with shared runtime settings.
4. Creates a results object used to collect outputs and metadata throughout the run.

`run_pipeline()` then executes each stage in order. The output of one stage becomes the input to the next, keeping the workflow reproducible and making intermediate results available for inspection.

### Agent and Graph Layer

The agents use LangGraph-based workflows instead of relying on a single language-model response. A typical agent graph contains nodes that:

- Inspect a sampled summary of the input `DataFrame`
- Recommend an operation or analysis plan
- Generate executable Python code
- Run that code against the data
- Capture execution errors and request corrected code
- Optionally pause for human review
- Return structured results through agent getter methods

This graph structure allows retries and validation to happen inside each analytical stage. Shared graph construction, execution, review, and reporting logic lives under `DataProbe/templates/`, while output parsing is handled by `DataProbe/parsers/`.

### Specialized Components

| Component | Responsibility | Main output |
|---|---|---|
| `DataCleaningAgent` | Detects data-quality issues and generates a reusable cleaning function | Cleaned `DataFrame`, cleaning code, and recommended steps |
| `MultiDataVisualObsAgent` | Selects useful charts and interprets the patterns shown in each chart | Plotly figures, plot images, and observations |
| `EDAAgent` | Performs correlation and temporal analysis and creates predictive features | Engineered `DataFrame` and feature recommendations |
| `ModelRecommendationAgent` | Infers the learning problem and proposes suitable algorithms | Problem type, model list, and model explanations |
| `ModelEvaluationAgent` | Trains and compares recommended models using a held-out test set | Performance metrics, comparison summary, and best model |

### Data and Control Flow

1. The original pandas `DataFrame` is saved before any transformation.
2. The cleaning agent produces a cleaned `DataFrame` and reusable cleaning function.
3. The visualization agent analyzes the cleaned data and returns chart definitions with observations.
4. The EDA agent receives the cleaned data, target column, and optional date column, then creates an engineered dataset.
5. Correlation analysis, temporal analysis, and engineered-feature recommendations are passed to the model-recommendation agent.
6. The evaluation agent trains the recommended candidates and identifies the best-performing model.
7. The orchestrator converts results into JSON-safe values and writes the final JSON and PDF reports.

If a stage raises an exception, the orchestrator records the error and saves the partial results to `eda_results_error.json` before re-raising it. This preserves completed work for debugging.

### Supporting Modules

- `tools/` provides file-loading, directory-inspection, and `DataFrame` summary utilities exposed as LangChain tools.
- `utils/` contains shared helpers for Plotly and Matplotlib output, HTML conversion, logging, message handling, and generated-code cleanup.
- `templates/` defines reusable agent graphs, prompt templates, code-execution nodes, retry behavior, and optional human review.
- `parsers/` converts model-generated responses into Python or structured values that downstream nodes can consume.
- `multiagents/` combines several agent responsibilities into higher-level analytical workflows.

### Design Considerations

- **Separation of concerns:** Each agent owns one phase of the analysis and exposes its results through a small set of getter methods.
- **Model flexibility:** Different language models can be assigned to different stages, allowing cost, speed, and reasoning quality to be balanced independently.
- **Traceability:** Intermediate datasets, generated code, charts, logs, metrics, and summaries are retained as artifacts.
- **Recoverability:** Agent-level retries can repair generated code, while pipeline-level error output preserves partial results.
- **Human oversight:** A checkpointer-backed human-in-the-loop mode can review proposed actions before execution.
- **Extensibility:** New agents can follow the existing `BaseAgent` and LangGraph patterns and then be added as another orchestrated stage.

## Requirements

- Python 3.10 or later is recommended
- An OpenAI API key
- A CSV file or a pandas `DataFrame`

The repository uses LangChain, LangGraph, pandas, Plotly, Matplotlib, Statsmodels, and FPDF.

## Installation

Clone the repository and enter its directory:

```bash
git clone https://github.com/p-H-7/DataProbe.git
cd DataProbe
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key"
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

> Do not commit API keys or `.env` files to version control.

## Quick Start

```python
import pandas as pd
from langchain_openai import ChatOpenAI
from DataProbe.orchestration import AutomatedEDAOrchestrator

# Load your dataset.
df = pd.read_csv("data/your_dataset.csv")

# Assign a language model to each stage of the pipeline.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
models = {
    "cleaning": llm,
    "visualization": llm,
    "feature_engineering": llm,
    "model_recommendation": llm,
    "model_evaluation": llm,
}

orchestrator = AutomatedEDAOrchestrator(
    models=models,
    output_dir="./eda_outputs",
    n_samples=30,
    log=True,
    log_path="./logs",
    human_in_the_loop=False,
    max_visualizations=5,
    generate_pdf=True,
)

results = orchestrator.run_pipeline(
    df=df,
    target_column="target",
    date_column=None,
    data_cleaning_instructions="Handle missing values but preserve outliers.",
    visualization_instructions="Identify the most important patterns and relationships.",
    problem_type=None,
    test_size=0.2,
    random_state=42,
)

print("PDF report:", results.get("pdf_report_path"))
print("Best model:", results["model_evaluation"].get("best_model"))
```

Replace `data/your_dataset.csv`, `target`, and (if applicable) `date_column` with values from your dataset.

## Configuration

### `AutomatedEDAOrchestrator`

| Parameter | Description | Default |
|---|---|---:|
| `models` | Dictionary containing models for all five pipeline stages | Required |
| `output_dir` | Directory for generated artifacts | `./eda_outputs` |
| `n_samples` | Number of data samples supplied to the agents | `30` |
| `log` | Enable workflow logging | `True` |
| `log_path` | Directory for logs and generated functions | `./logs` |
| `human_in_the_loop` | Enable manual review during agent execution | `False` |
| `max_visualizations` | Maximum number of visualizations to generate | `5` |
| `generate_pdf` | Generate a final PDF report | `True` |

### `run_pipeline`

| Parameter | Description |
|---|---|
| `df` | Input pandas `DataFrame` |
| `target_column` | Column the modeling pipeline should predict |
| `date_column` | Optional date/time column for temporal analysis |
| `data_cleaning_instructions` | Optional natural-language cleaning rules |
| `visualization_instructions` | Optional visualization goals or questions |
| `problem_type` | Optional explicit problem type; use `None` for automatic inference |
| `test_size` | Fraction of data reserved for testing |
| `random_state` | Random seed used during evaluation |
| `training_instructions` | Optional model-training instructions |

## Generated Outputs

By default, DataProbe creates an output structure similar to:

```text
eda_outputs/
|-- data/
|   |-- original_data.csv
|   |-- cleaned_data.csv
|   `-- engineered_data.csv
|-- plots/
|   `-- *.png
|-- models/
|-- eda_results.json
`-- eda_report.pdf
```

Logs and generated agent functions are written to the configured `log_path`.

## Using Individual Agents

You can also use agents independently. For example, to clean a dataset:

```python
import pandas as pd
from langchain_openai import ChatOpenAI
from DataProbe.agents.data_cleaning_agent import DataCleaningAgent

df = pd.read_csv("data/your_dataset.csv")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

agent = DataCleaningAgent(model=llm, log=True, log_path="./logs")
agent.invoke_agent(
    data_raw=df,
    user_instructions="Handle missing values but do not remove outliers.",
)

cleaned_df = agent.get_data_cleaned()
print(agent.get_recommended_cleaning_steps())
```

Other available components include:

- `DataVisualizationAgent`
- `MultiDataVisualObsAgent`
- `EDAAgent`
- `ModelRecommendationAgent`
- `ModelEvaluationAgent`

See [`demo.ipynb`](demo.ipynb) for agent-by-agent and complete-pipeline examples.

## Sample Reports

The `sample_reports_generated/` directory contains example PDF reports for bike sales, customer churn, and liver-patient datasets.

## Important Notes

- Data is sent to the configured language-model provider. Do not use sensitive or regulated data unless your provider setup and organizational policies permit it.
- AI-generated cleaning, feature-engineering, and modeling code should be reviewed before production use.
- Model evaluation may install additional libraries automatically when invoked through the orchestrator.
- Large datasets can increase runtime and API usage. Use `n_samples` and `max_visualizations` to control the workload.
- Model names in the demo notebook may become unavailable over time; substitute models supported by your account.

## Project Structure

```text
DataProbe/
|-- agents/          # Cleaning, visualization, and model recommendation agents
|-- multiagents/     # Multi-stage EDA, observations, and evaluation workflows
|-- parsers/         # Agent output parsers
|-- templates/       # Shared graph and prompt templates
|-- tools/           # Data loading and DataFrame utilities
|-- utils/           # Plotting, HTML, logging, and message helpers
`-- orchestration.py # End-to-end pipeline coordinator
```

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository.
2. Create a feature branch.
3. Make and test your changes.
4. Open a pull request with a clear description.

## License

No license file is currently included in the repository. Add a license before distributing or reusing the project beyond the permissions granted by copyright law.
