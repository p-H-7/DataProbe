# Agents: EDA and Feature Engineering Agent
from typing import TypedDict, Annotated, Sequence, Literal, List, Dict, Any, Optional, Union
import operator
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

from langchain.prompts import PromptTemplate
from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseLanguageModel

from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Checkpointer

from IPython.display import Markdown, display

from DataProbe.templates import (
    node_func_human_review,
    node_func_report_agent_outputs,
    create_coding_agent_graph,
    BaseAgent,
)
from DataProbe.utils.regex import (
    format_agent_name,
    format_recommended_steps,
    get_generic_summary,
)
from DataProbe.tools.dataframe import get_dataframe_summary
from DataProbe.multiagents.multi_data_visual_and_observations_agent import MultiDataVisualObsAgent
from DataProbe.utils.logging import log_ai_function
from DataProbe.parsers.parsers import PythonOutputParser

AGENT_NAME = "eda_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")

class EDAAgent:
    """
    An Agent that performs Feature Engineering on the dataset to create a new dataframe with Engineered features added
    And also performs EDA on the dataset with regards to the target variable.

    """
    def __init__(
        self,
        model: BaseLanguageModel,
        visualization_wrapper: MultiDataVisualObsAgent = None,
        n_samples: int = 30,
        max_visualizations: int = 8,
        log: bool = False,
        log_path: str = None,
        file_name: str = "feature_engineering.py",
        function_name: str = "feature_engineering",
        overwrite: bool = True,
        human_in_the_loop: bool = False,
        checkpointer: Checkpointer = None,
    ):
        self._params = {
            "model": model,
            "visualization_wrapper": visualization_wrapper,
            "n_samples": n_samples,
            "max_visualizations": max_visualizations,
            "log": log,
            "log_path": log_path,
            "file_name": file_name,
            "function_name": function_name,
            "overwrite": overwrite,
            "human_in_the_loop": human_in_the_loop,
            "checkpointer": checkpointer,
        }
        
        if visualization_wrapper is None:
            self._params["visualization_wrapper"] = MultiDataVisualObsAgent(
                model=model,
                n_samples=n_samples,
                max_visualizations=max_visualizations,
                log=log,
                log_path=log_path,
                human_in_the_loop=False
            )
        
        self._compiled_graph = self._make_compiled_graph()
        self.response = None
    
    def _make_compiled_graph(self):
        self.response = None
        return make_eda_agent(**self._params)
    
    def update_params(self, **kwargs):
        for k, v in kwargs.items():
            self._params[k] = v
        self._compiled_graph = self._make_compiled_graph()
    
    async def ainvoke_agent(
        self,
        data_raw: pd.DataFrame,
        target_column: str,
        date_column: Optional[str] = None,
        **kwargs,
    ):
        response = await self._compiled_graph.ainvoke(
            {
                "data_raw": data_raw.to_dict(),
                "target_column": target_column,
                "date_column": date_column,
            },
            **kwargs,
        )
        self.response = response
        return None
    
    def invoke_agent(
        self,
        data_raw: pd.DataFrame,
        target_column: str,
        date_column: Optional[str] = None,
        **kwargs,
    ):
        response = self._compiled_graph.invoke(
            {
                "data_raw": data_raw.to_dict(),
                "target_column": target_column,
                "date_column": date_column,
            },
            **kwargs,
        )
        self.response = response
        return None
    
    def get_workflow_summary(self, markdown=False):
        if self.response and self.response.get("messages"):
            summary = get_generic_summary(
                json.loads(self.response.get("messages")[-1].content)
            )
            if markdown:
                return Markdown(summary)
            else:
                return summary
    
    def get_feature_engineering_recommendations(self, markdown=False):
        if self.response:
            recommendations = self.response.get("feature_recommendations_formatted", "")
            if markdown:
                return Markdown(recommendations)
            return recommendations
        return None
    
    def get_temporal_analysis_results(self, markdown=False):
        if self.response:
            analysis = self.response.get("temporal_analysis", "")
            if markdown:
                return Markdown(analysis)
            return analysis
        return None
    
    def get_correlation_analysis_results(self, markdown=False):
        if self.response:
            analysis = self.response.get("correlation_analysis", "")
            if markdown:
                return Markdown(analysis)
            return analysis
        return None
    
    def get_engineered_dataframe(self):
        if self.response and self.response.get("engineered_data"):
            return pd.DataFrame(self.response.get("engineered_data"))
        return None
    
    def get_engineered_feature_names(self):
        if self.response:
            return self.response.get("engineered_feature_names", [])
        return None
    
    def get_feature_engineering_function(self, markdown=False):
        if self.response:
            func_code = self.response.get("feature_engineering_function", "")
            if markdown:
                return Markdown(f"```python\n{func_code}\n```")
            return func_code
        return None
    
    def get_generated_plots(self):
        if self.response and self.response.get("visualization_plots"):
            return self.response.get("visualization_plots", {})
        return {}
    
    def get_plot_observations(self):
        if self.response and self.response.get("visualization_observations"):
            return self.response.get("visualization_observations", {})
        return {}
    
    def display_plots_with_observations(self, width=900, height=500, use_html=True):
        plots = self.get_generated_plots()
        observations = self.get_plot_observations()
        
        if not plots:
            print("No plots have been generated yet.")
            return
        
        for title, plot_dict in plots.items():
            print(f"\n## {title}")
            try:
                from plotly.graph_objs import Figure
                if isinstance(plot_dict, Figure):
                    fig = plot_dict
                else:
                    fig = Figure(plot_dict)
                
                fig.update_layout(width=width, height=height)
                try:
                    display(fig)
                except (ImportError, ValueError) as e:
                    if use_html:
                        print(f"Direct display failed: {e}")
                        print("Falling back to HTML rendering...")
                        import plotly.io as pio
                        html = pio.to_html(fig, full_html=False)
                        from IPython.display import HTML
                        display(HTML(html))
                    else:
                        raise e
                
                if title in observations and observations[title]:
                    print("\n### Key Observations:")
                    for idx, obs in enumerate(observations[title], 1):
                        print(f"\n**{idx}. {obs.get('title')}**")
                        print(f"{obs.get('description')}")
            
            except Exception as e:
                print(f"Error displaying plot '{title}': {e}")
                print("Try installing required packages: pip install nbformat>=4.2.0")
    
    def get_response(self):
        return self.response
    
    def show(self):
        return self._compiled_graph.show()


def make_eda_agent(
    model: BaseLanguageModel,
    visualization_wrapper: MultiDataVisualObsAgent,
    n_samples: int = 30,
    max_visualizations: int = 8,
    log: bool = False,
    log_path: str = None,
    file_name: str = "feature_engineering.py",
    function_name: str = "feature_engineering",
    overwrite: bool = True,
    human_in_the_loop: bool = False,
    checkpointer: Checkpointer = None,
):
    llm = model
    
    if human_in_the_loop and checkpointer is None:
        print(
            "Human in the loop is enabled. A checkpointer is required. Setting to MemorySaver()."
        )
        checkpointer = MemorySaver()
    
    if log:
        if log_path is None:
            log_path = LOG_PATH
        if not os.path.exists(log_path):
            os.makedirs(log_path)
    
    class GraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        data_raw: dict
        target_column: str
        date_column: Optional[str]
        identified_date_column: str
        all_datasets_summary: str
        temporal_analysis: str
        correlation_analysis: str
        feature_recommendations: List[Dict[str, str]]
        feature_recommendations_formatted: str
        visualization_instructions: str
        visualization_plots: Dict[str, Any]
        visualization_observations: Dict[str, List[Dict[str, str]]]
        feature_engineering_function: str
        feature_engineering_function_path: str
        feature_engineering_function_file_name: str
        feature_engineering_function_name: str
        feature_engineering_error: str
        engineered_data: dict
        engineered_feature_names: List[str]
        summary: str
    
    def analyze_dataset_structure(state: GraphState):
        print(format_agent_name(AGENT_NAME))
        print("    * ANALYZING DATASET STRUCTURE")
        
        data_raw = state.get("data_raw")
        df = pd.DataFrame.from_dict(data_raw)
        
        target_column = state.get("target_column")
        date_column = state.get("date_column")
        
        all_datasets_summary = get_dataframe_summary(
            [df], n_sample=n_samples, skip_stats=False
        )
        all_datasets_summary_str = "\n\n".join(all_datasets_summary)
        
        identified_date_column = date_column
        if not date_column:
            date_keywords = ['date', 'time', 'day', 'month', 'year', 'dt', 'timestamp']
            potential_date_columns = []
            
            for col in df.columns:
                if any(keyword in col.lower() for keyword in date_keywords):
                    potential_date_columns.append(col)
            
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    if col not in potential_date_columns:
                        potential_date_columns.append(col)
            
            if potential_date_columns:
                identified_date_column = potential_date_columns[0]
                print(f"    * IDENTIFIED DATE COLUMN: {identified_date_column}")
        
        return {
            "all_datasets_summary": all_datasets_summary_str,
            "identified_date_column": identified_date_column,
        }
    
    def perform_temporal_analysis(state: GraphState):
        print("    * PERFORMING TEMPORAL ANALYSIS")
        
        temporal_analysis_prompt = PromptTemplate(
            template="""
            You are a time series analysis expert who specializes in analyzing temporal patterns in data.
            
            Analyze the temporal patterns in the provided dataset, focusing on the target variable that will be forecasted.
            
            DATA SUMMARY: 
            {all_datasets_summary}
            
            TARGET COLUMN: {target_column}
            DATE COLUMN (if identified): {date_column}
            
            TASK:
            Conduct a thorough temporal analysis of the data with special attention to the target variable.
            
            ANALYSIS SHOULD INCLUDE:
            
            1. Identification of temporal patterns, including:
               - Trends (long-term direction)
               - Seasonality (recurring patterns at fixed intervals)
               - Cycles (recurring patterns at variable intervals)
               - Autocorrelation (correlation of the target with its own past values)
            
            2. Discussion of important time horizons, including:
               - Which time lags appear most relevant for forecasting
               - Optimal forecasting window (how far ahead predictions should be made)
               - Minimum data history needed for accurate forecasting
            
            3. Recommendations for temporal feature engineering, such as:
               - What temporal aggregations would be useful (daily, weekly, monthly)
               - Which lag features would be informative (t-1, t-7, etc.)
               - What moving window calculations would help (rolling averages, etc.)
               - Any other temporal transformations that would aid forecasting
            
            RETURN FORMAT:
            
            Return your analysis in markdown format with appropriate headings and structure.
            Focus on insights that would be useful for feature engineering to improve forecasting.
            """,
            input_variables=[
                "all_datasets_summary",
                "target_column",
                "date_column",
            ],
        )
        
        temporal_analyst = temporal_analysis_prompt | llm
        
        response = temporal_analyst.invoke(
            {
                "all_datasets_summary": state.get("all_datasets_summary"),
                "target_column": state.get("target_column"),
                "date_column": state.get("identified_date_column"),
            }
        )
        
        return {
            "temporal_analysis": response.content,
        }
    
    def perform_correlation_analysis(state: GraphState):
        print("    * PERFORMING CORRELATION ANALYSIS")
        
        correlation_analysis_prompt = PromptTemplate(
            template="""
            You are a correlation analysis expert who specializes in identifying relationships between variables.
            
            Analyze the relationships between features and the target variable in the provided dataset.
            
            DATA SUMMARY: 
            {all_datasets_summary}
            
            TARGET COLUMN: {target_column}
            
            TASK:
            Conduct a thorough correlation analysis focusing on relationships with the target variable.
            
            ANALYSIS SHOULD INCLUDE:
            
            1. Identification of strongly correlated features with the target variable, including:
               - Linear correlations (Pearson, Spearman)
               - Potential non-linear relationships
               - Feature interactions that might impact the target
            
            2. Discussion of potential multicollinearity issues:
               - Highly correlated features that might cause redundancy
               - Groups of features that capture similar information
            
            3. Recommendations for feature selection and transformation, such as:
               - Which features appear most important for forecasting
               - How features might be transformed to strengthen relationships
               - Potential interaction terms that could be created
            
            RETURN FORMAT:
            
            Return your analysis in markdown format with appropriate headings and structure.
            Focus on insights that would be useful for feature engineering to improve forecasting.
            """,
            input_variables=[
                "all_datasets_summary",
                "target_column",
            ],
        )
        
        correlation_analyst = correlation_analysis_prompt | llm
        
        response = correlation_analyst.invoke(
            {
                "all_datasets_summary": state.get("all_datasets_summary"),
                "target_column": state.get("target_column"),
            }
        )
        
        return {
            "correlation_analysis": response.content,
        }
    
    def generate_feature_recommendations(state: GraphState):
        print("    * GENERATING FEATURE ENGINEERING RECOMMENDATIONS")
        
        feature_recommendation_prompt = PromptTemplate(
            template="""
            You are a feature engineering expert who specializes in preparing data for time series forecasting.
            
            Based on the temporal and correlation analyses, recommend specific features to engineer for improving forecasting of the target variable.
            
            DATA SUMMARY: 
            {all_datasets_summary}
            
            TARGET COLUMN: {target_column}
            DATE COLUMN (if identified): {date_column}
            
            TEMPORAL ANALYSIS:
            {temporal_analysis}
            
            CORRELATION ANALYSIS:
            {correlation_analysis}
            
            TASK:
            Recommend specific features to engineer that would improve forecasting performance.
            
            RECOMMENDATIONS SHOULD INCLUDE:
            
            1. Temporal features, such as:
               - Date-based features (day of week, month, quarter, etc.)
               - Lag features of the target variable
               - Moving window calculations (rolling means, max, min, etc.)
               - Seasonal indicators
            
            2. Feature transformations, such as:
               - Log transformations
               - Polynomial features
               - Interaction terms
               - Normalization/standardization recommendations
            
            3. Feature selection recommendations:
               - Which original features to keep
               - Which engineered features are likely most important
               - Any redundant features that could be removed
            
            RETURN FORMAT:
            
            Return your recommendations in JSON format with the following structure:
            ```json
            [
              {{
                "feature_name": "Name of the engineered feature",
                "feature_type": "Type of feature (temporal, transformation, etc.)",
                "description": "Description of what this feature represents",
                "creation_logic": "How to calculate or create this feature",
                "rationale": "Why this feature would be useful for forecasting"
              }},
              ...
            ]
            ```
            
            Ensure each recommendation is specific enough that it could be directly implemented in code.
            """,
            input_variables=[
                "all_datasets_summary",
                "target_column",
                "date_column",
                "temporal_analysis",
                "correlation_analysis",
            ],
        )
        
        feature_recommender = feature_recommendation_prompt | llm
        
        response = feature_recommender.invoke(
            {
                "all_datasets_summary": state.get("all_datasets_summary"),
                "target_column": state.get("target_column"),
                "date_column": state.get("identified_date_column"),
                "temporal_analysis": state.get("temporal_analysis"),
                "correlation_analysis": state.get("correlation_analysis"),
            }
        )
        
        json_str = response.content
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()
        
        try:
            feature_recommendations = json.loads(json_str)
        except json.JSONDecodeError:
            print("Failed to parse JSON response. Using simplified format.")
            feature_recommendations = [
                {
                    "feature_name": "lag_1",
                    "feature_type": "temporal",
                    "description": "Target value from previous time period",
                    "creation_logic": "df['lag_1'] = df[target_column].shift(1)",
                    "rationale": "Previous values are strong predictors in time series"
                }
            ]
        
        formatted_recommendations = "# Recommended Feature Engineering:\n\n"
        
        for i, feature in enumerate(feature_recommendations, 1):
            formatted_recommendations += f"## {i}. {feature['feature_name']}\n"
            formatted_recommendations += f"- **Type:** {feature['feature_type']}\n"
            formatted_recommendations += f"- **Description:** {feature['description']}\n"
            formatted_recommendations += f"- **Creation Logic:** `{feature['creation_logic']}`\n"
            formatted_recommendations += f"- **Rationale:** {feature['rationale']}\n\n"
        
        return {
            "feature_recommendations": feature_recommendations,
            "feature_recommendations_formatted": formatted_recommendations,
        }
    
    def prepare_visualization_instructions(state: GraphState):
        print("    * PREPARING VISUALIZATION INSTRUCTIONS")
        
        visualization_instruction_prompt = PromptTemplate(
            template="""
            You are a data visualization expert specializing in time series analysis.
            
            Based on the temporal and correlation analyses, create detailed instructions for visualizing relationships
            and patterns relevant to forecasting the target variable.
            
            TARGET COLUMN: {target_column}
            DATE COLUMN (if identified): {date_column}
            
            TEMPORAL ANALYSIS:
            {temporal_analysis}
            
            CORRELATION ANALYSIS:
            {correlation_analysis}
            
            TASK:
            Create detailed instructions for generating visualizations that would provide insights for time series forecasting.
            
            YOUR INSTRUCTIONS SHOULD REQUEST:
            
            1. Time series visualizations of the target variable, including:
               - Line plots showing trends over time
               - Seasonal decomposition plots if applicable
               - Autocorrelation plots
            
            2. Correlation visualizations, such as:
               - Heatmaps showing correlations between features
               - Scatter plots of important feature relationships with the target
               - Feature importance visualizations
            
            3. Distribution visualizations, such as:
               - Histograms of the target variable and key features
               - Box plots showing distributions across temporal segments
            
            RETURN FORMAT:
            
            Return clear, detailed instructions that could be passed directly to a visualization agent.
            Be specific about what to plot, which variables to include, and what insights to look for.
            """,
            input_variables=[
                "target_column",
                "date_column",
                "temporal_analysis",
                "correlation_analysis",
            ],
        )
        
        viz_instruction_generator = visualization_instruction_prompt | llm
        
        response = viz_instruction_generator.invoke(
            {
                "target_column": state.get("target_column"),
                "date_column": state.get("identified_date_column"),
                "temporal_analysis": state.get("temporal_analysis"),
                "correlation_analysis": state.get("correlation_analysis"),
            }
        )
        
        return {
            "visualization_instructions": response.content,
        }
    
    def generate_visualizations(state: GraphState):
        print("    * GENERATING VISUALIZATIONS")
        
        data_raw = state.get("data_raw")
        df = pd.DataFrame.from_dict(data_raw)
        
        visualization_wrapper.invoke_agent(
            data_raw=df,
            user_instructions=state.get("visualization_instructions")
        )
        
        plots = visualization_wrapper.get_generated_plots()
        observations = visualization_wrapper.get_plot_observations()
        
        return {
            "visualization_plots": plots,
            "visualization_observations": observations,
        }
    
    def create_feature_engineering_function(state: GraphState):
        print("    * CREATING FEATURE ENGINEERING FUNCTION")
        
        feature_engineering_prompt = PromptTemplate(
            template="""
            You are a feature engineering expert who specializes in preparing data for time series forecasting.
            
            Your task is to create a Python function called {function_name} that implements the recommended feature engineering steps.
            
            DATA SUMMARY: 
            {all_datasets_summary}
            
            TARGET COLUMN: {target_column}
            DATE COLUMN (if identified): {date_column}
            
            FEATURE ENGINEERING RECOMMENDATIONS:
            {feature_recommendations_formatted}
            
            FUNCTION REQUIREMENTS:
            
            1. The function should be named {function_name} and take a pandas DataFrame as input
            2. It should return the DataFrame with all engineered features added
            3. It should handle missing values created by temporal features (e.g., lag features)
            4. It should include all imports inside the function
            5. It should return two outputs: the engineered DataFrame and a list of new feature names
            
            PYTHON CODE FORMAT:
            ```python
            def {function_name}(data_raw, target_column="{target_column}", date_column="{date_column}"):
                import pandas as pd
                import numpy as np
                
                # Create a copy of the input dataframe
                df = data_raw.copy()
                
                # Convert date column to datetime if not already
                if date_column is not None and date_column in df.columns:
                    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
                        df[date_column] = pd.to_datetime(df[date_column])
                
                # List to track engineered feature names
                engineered_feature_names = []
                
                # Implement feature engineering
                # ...
                
                return df, engineered_feature_names
            ```
            
            Return only valid Python code that will run without errors.
            """,
            input_variables=[
                "all_datasets_summary",
                "target_column",
                "date_column",
                "feature_recommendations_formatted",
                "function_name",
            ],
        )
        
        feature_engineering_generator = feature_engineering_prompt | llm | PythonOutputParser()
        
        response = feature_engineering_generator.invoke(
            {
                "all_datasets_summary": state.get("all_datasets_summary"),
                "target_column": state.get("target_column"),
                "date_column": state.get("identified_date_column"),
                "feature_recommendations_formatted": state.get("feature_recommendations_formatted"),
                "function_name": function_name,
            }
        )
        
        file_path, file_name_2 = log_ai_function(
            response=response,
            file_name=file_name,
            log=log,
            log_path=log_path,
            overwrite=overwrite,
        )
        
        return {
            "feature_engineering_function": response,
            "feature_engineering_function_path": file_path,
            "feature_engineering_function_file_name": file_name_2,
            "feature_engineering_function_name": function_name,
        }
    
    def execute_feature_engineering(state: GraphState):
        print("    * EXECUTING FEATURE ENGINEERING")
        
        data_raw = state.get("data_raw")
        df = pd.DataFrame.from_dict(data_raw)
        
        feature_engineering_function = state.get("feature_engineering_function")
        function_name = state.get("feature_engineering_function_name")
        
        try:
            namespace = {}
            exec(feature_engineering_function, namespace)
            
            func = namespace[function_name]
            
            engineered_df, engineered_feature_names = func(
                df, 
                target_column=state.get("target_column"),
                date_column=state.get("identified_date_column")
            )
            
            return {
                "engineered_data": engineered_df.to_dict(),
                "engineered_feature_names": engineered_feature_names,
                "feature_engineering_error": "",
            }
            
        except Exception as e:
            error_message = f"An error occurred during feature engineering: {str(e)}"
            print(error_message)
            return {
                "feature_engineering_error": error_message,
            }
    
    def fix_feature_engineering_function(state: GraphState):
        print("    * FIXING FEATURE ENGINEERING FUNCTION")
        
        fix_prompt = PromptTemplate(
            template="""
            You are a feature engineering expert who specializes in fixing Python code errors.
            
            The following feature engineering function has an error. Please fix it.
            
            ORIGINAL FUNCTION:
            ```python
            {feature_engineering_function}
            ```
            
            ERROR MESSAGE:
            {error}
            
            Fix the function and return only the corrected Python code that will run without errors.
            The function should be named {function_name} and follow the requirements from the original.
            """,
            input_variables=[
                "feature_engineering_function",
                "error",
                "function_name",
            ],
        )
        
        fix_generator = fix_prompt | llm | PythonOutputParser()
        
        response = fix_generator.invoke(
            {
                "feature_engineering_function": state.get("feature_engineering_function"),
                "error": state.get("feature_engineering_error"),
                "function_name": state.get("feature_engineering_function_name"),
            }
        )
        
        file_path, file_name_2 = log_ai_function(
            response=response,
            file_name=state.get("feature_engineering_function_file_name"),
            log=log,
            log_path=log_path,
            overwrite=True,
        )
        
        return {
            "feature_engineering_function": response,
            "feature_engineering_function_path": file_path,
        }
    
    def create_summary(state: GraphState):
        print("    * CREATING ANALYSIS SUMMARY")
        
        summary_prompt = PromptTemplate(
            template="""
            You are a data scientist creating a comprehensive summary of time series EDA and feature engineering.
            
            Create a concise summary that explains the key findings from the analysis and the feature engineering process.
            
            ANALYSES:
            
            TEMPORAL ANALYSIS:
            {temporal_analysis}
            
            CORRELATION ANALYSIS:
            {correlation_analysis}
            
            FEATURE ENGINEERING PERFORMED:
            {feature_recommendations_formatted}
            
            NUMBER OF ENGINEERED FEATURES: {num_engineered_features}
            
            IMPORTANT:
            
            - Focus on summarizing the most important insights for time series forecasting
            - Explain how the engineered features address patterns identified in the analysis
            - Describe what makes this prepared dataset well-suited for forecasting
            - Keep the summary concise yet comprehensive (3-5 paragraphs)
            
            RETURN FORMAT:
            
            Return your summary in markdown format with appropriate headings and structure.
            """,
            input_variables=[
                "temporal_analysis",
                "correlation_analysis",
                "feature_recommendations_formatted",
                "num_engineered_features",
            ],
        )
        
        num_engineered_features = len(state.get("engineered_feature_names", []))
        
        summary_generator = summary_prompt | llm
        
        response = summary_generator.invoke(
            {
                "temporal_analysis": state.get("temporal_analysis"),
                "correlation_analysis": state.get("correlation_analysis"),
                "feature_recommendations_formatted": state.get("feature_recommendations_formatted"),
                "num_engineered_features": num_engineered_features,
            }
        )
        
        return {
            "summary": response.content,
        }

    def human_review_recommendations(
        state: GraphState,
    ) -> Command[Literal["generate_feature_recommendations", "prepare_visualization_instructions"]]:
        prompt_text = "Are the following feature engineering recommendations appropriate? (Answer 'yes' or provide modifications)\n{steps}"
        
        return node_func_human_review(
            state=state,
            prompt_text=prompt_text,
            yes_goto="prepare_visualization_instructions",
            no_goto="generate_feature_recommendations",
            user_instructions_key=None,
            recommended_steps_key="feature_recommendations_formatted",
        )

    def report_agent_outputs(state: GraphState):
        return node_func_report_agent_outputs(
            state=state,
            keys_to_include=[
                "temporal_analysis",
                "correlation_analysis",
                "feature_recommendations_formatted",
                "feature_engineering_function",
                "engineered_feature_names",
                "summary"
            ],
            result_key="messages",
            role=AGENT_NAME,
            custom_title="EDA layer for Feature Engineering and Correlation Analysis",
        )
    
    node_functions = {
        "analyze_dataset_structure": analyze_dataset_structure,
        "perform_temporal_analysis": perform_temporal_analysis,
        "perform_correlation_analysis": perform_correlation_analysis,
        "generate_feature_recommendations": generate_feature_recommendations,
        "human_review_recommendations": human_review_recommendations,
        "prepare_visualization_instructions": prepare_visualization_instructions,
        "generate_visualizations": generate_visualizations,
        "create_feature_engineering_function": create_feature_engineering_function,
        "execute_feature_engineering": execute_feature_engineering,
        "fix_feature_engineering_function": fix_feature_engineering_function,
        "create_summary": create_summary,
        "report_agent_outputs": report_agent_outputs,
    }

    # Create edges for the graph
    edges = []

    # Start -> Analyze Dataset Structure
    edges.append(("__start__", "analyze_dataset_structure"))

    # Analyze Dataset Structure -> Perform Temporal Analysis
    edges.append(("analyze_dataset_structure", "perform_temporal_analysis"))

    # Perform Temporal Analysis -> Perform Correlation Analysis
    edges.append(("perform_temporal_analysis", "perform_correlation_analysis"))

    # Perform Correlation Analysis -> Generate Feature Recommendations
    edges.append(("perform_correlation_analysis", "generate_feature_recommendations"))

    # If human_in_the_loop, add human review step
    if human_in_the_loop:
        edges.append(("generate_feature_recommendations", "human_review_recommendations"))
    else:
        edges.append(("generate_feature_recommendations", "prepare_visualization_instructions"))

    # Prepare Visualization Instructions -> Generate Visualizations
    edges.append(("prepare_visualization_instructions", "generate_visualizations"))

    # Generate Visualizations -> Create Feature Engineering Function
    edges.append(("generate_visualizations", "create_feature_engineering_function"))

    # Create Feature Engineering Function -> Execute Feature Engineering
    edges.append(("create_feature_engineering_function", "execute_feature_engineering"))

    # REMOVE these conditional edges
    # edges.append(("execute_feature_engineering", {"condition": lambda x: x.get("feature_engineering_error", ""), "value": "fix_feature_engineering_function"}))
    # edges.append(("execute_feature_engineering", {"condition": lambda x: not x.get("feature_engineering_error", ""), "value": "create_summary"}))

    # Fix Feature Engineering Function -> Execute Feature Engineering (retry)
    edges.append(("fix_feature_engineering_function", "execute_feature_engineering"))

    # Create Summary -> Report
    edges.append(("create_summary", "report_agent_outputs"))

    # Report -> End
    edges.append(("report_agent_outputs", "__end__"))

    # Create the graph
    from langgraph.graph import StateGraph, END

    workflow = StateGraph(GraphState)

    # Add nodes
    for name, func in node_functions.items():
        workflow.add_node(name, func)

    # Add regular edges first
    for start, end in edges:
        if start == "__start__":
            workflow.set_entry_point(end)
        elif end == "__end__":
            workflow.add_edge(start, END)
        else:
            workflow.add_edge(start, end)

    # Add the conditional edge separately after regular edges
    workflow.add_conditional_edges(
        "execute_feature_engineering",
        lambda x: bool(x.get("feature_engineering_error", "")),
        {
            True: "fix_feature_engineering_function",
            False: "create_summary"
        }
    )

    # Compile the graph
    app = workflow.compile(checkpointer=checkpointer)

    return app