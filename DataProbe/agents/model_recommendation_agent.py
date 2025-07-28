# Model Recommendation Agent

# Libraries
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
    relocate_imports_inside_function,
    add_comments_to_top,
    format_agent_name,
    format_recommended_steps,
    get_generic_summary,
)
from DataProbe.tools.dataframe import get_dataframe_summary
from DataProbe.utils.logging import log_ai_function
from DataProbe.parsers.parsers import PythonOutputParser

# Setup
AGENT_NAME = "model_recommendation_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")

# Class
class ModelRecommendationAgent:
    """
    Model recommendation agent that analyzes data and recommends appropriate models
    based on the problem type and dataset characteristics.
    """
    
    def __init__(
        self,
        model,
        n_samples=30,
        log=False,
        log_path=None,
        file_name="model_recommendation.py",
        function_name="recommend_models",
        overwrite=True,
        human_in_the_loop=False,
        checkpointer=None,
    ):
        self._params = {
            "model": model,
            "n_samples": n_samples,
            "log": log,
            "log_path": log_path,
            "file_name": file_name,
            "function_name": function_name,
            "overwrite": overwrite,
            "human_in_the_loop": human_in_the_loop,
            "checkpointer": checkpointer,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None
    
    def _make_compiled_graph(self):
        self.response = None
        return make_model_recommendation_agent(**self._params)
    
    def update_params(self, **kwargs):
        for k, v in kwargs.items():
            self._params[k] = v
        self._compiled_graph = self._make_compiled_graph()
    
    async def ainvoke_agent(
        self,
        engineered_data: pd.DataFrame,
        target_column: str,
        temporal_analysis: str,
        correlation_analysis: str,
        feature_recommendations: str,
        problem_type: str = None,
        num_models: int = 5,
        **kwargs,
    ):
        response = await self._compiled_graph.ainvoke(
            {
                "engineered_data": engineered_data.to_dict(),
                "target_column": target_column,
                "temporal_analysis": temporal_analysis,
                "correlation_analysis": correlation_analysis,
                "feature_recommendations": feature_recommendations,
                "problem_type": problem_type,
                "num_models": num_models,
            },
            **kwargs,
        )
        self.response = response
        return None
    
    def invoke_agent(
        self,
        engineered_data: pd.DataFrame,
        target_column: str,
        temporal_analysis: str,
        correlation_analysis: str,
        feature_recommendations: str,
        problem_type: str = None,
        num_models: int = 5,
        **kwargs,
    ):
        response = self._compiled_graph.invoke(
            {
                "engineered_data": engineered_data.to_dict(),
                "target_column": target_column,
                "temporal_analysis": temporal_analysis,
                "correlation_analysis": correlation_analysis,
                "feature_recommendations": feature_recommendations,
                "problem_type": problem_type,
                "num_models": num_models,
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
    
    def get_problem_type(self):
        """
        Returns the determined problem type.
        
        Returns
        -------
        str
            The problem type (e.g., "Regression", "Classification", etc.)
        """
        if self.response:
            return self.response.get("problem_type", "")
        return ""
    
    def get_recommended_models(self):
        """
        Returns the list of recommended model names.
        
        Returns
        -------
        list
            List of recommended model names as strings.
        """
        if self.response:
            return self.response.get("recommended_models", [])
        return []
    
    def get_model_descriptions(self, markdown=False):
        """
        Returns detailed descriptions of the recommended models.
        
        Returns
        -------
        str or Markdown
            The model descriptions.
        """
        if self.response:
            descriptions = self.response.get("model_descriptions", "")
            if markdown:
                return Markdown(descriptions)
            return descriptions
        return ""
    
    def get_problem_type_analysis(self, markdown=False):
        """
        Returns the problem type analysis.
        
        Returns
        -------
        str or Markdown
            The problem type analysis.
        """
        if self.response:
            analysis = self.response.get("problem_type_analysis", "")
            if markdown:
                return Markdown(analysis)
            return analysis
        return ""
    
    def get_model_recommendation_function(self, markdown=False):
        """
        Returns the generated Python function used for model recommendation.
        
        Returns
        -------
        str or Markdown
            The Python function code.
        """
        if self.response:
            func_code = self.response.get("model_recommendation_function", "")
            if markdown:
                return Markdown(f"```python\n{func_code}\n```")
            return func_code
        return ""
    
    def get_response(self):
        """
        Returns the agent's full response dictionary.
        
        Returns
        -------
        dict or None
            The response dictionary if available, otherwise None.
        """
        return self.response
    
    def show(self):
        """
        Displays the agent's mermaid diagram for visual inspection of the compiled graph.
        """
        return self._compiled_graph.show()


# Agent implementation
def make_model_recommendation_agent(
    model,
    n_samples=30,
    log=False,
    log_path=None,
    file_name="model_recommendation.py",
    function_name="recommend_models",
    overwrite=True,
    human_in_the_loop=False,
    checkpointer=None,
):
    llm = model
    
    if human_in_the_loop:
        if checkpointer is None:
            print(
                "Human in the loop is enabled. A checkpointer is required. Setting to MemorySaver()."
            )
            checkpointer = MemorySaver()
    
    # Setup Log Directory
    if log:
        if log_path is None:
            log_path = LOG_PATH
        if not os.path.exists(log_path):
            os.makedirs(log_path)
    
    class GraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        engineered_data: dict
        target_column: str
        temporal_analysis: str
        correlation_analysis: str
        feature_recommendations: str
        problem_type: Optional[str]
        num_models: int
        problem_type_analysis: str
        all_datasets_summary: str
        recommended_steps: str
        recommended_models: List[str]
        model_descriptions: str
        model_recommendation_function: str
        model_recommendation_function_path: str
        model_recommendation_function_file_name: str
        model_recommendation_function_name: str
        model_recommendation_error: str
    
    def analyze_dataset_structure(state: GraphState):
        print(format_agent_name(AGENT_NAME))
        print("    * ANALYZING DATASET STRUCTURE")
        
        engineered_data = state.get("engineered_data")
        df = pd.DataFrame.from_dict(engineered_data)
        
        all_datasets_summary = get_dataframe_summary(
            [df], n_sample=n_samples, skip_stats=False
        )
        all_datasets_summary_str = "\n\n".join(all_datasets_summary)
        
        return {
            "all_datasets_summary": all_datasets_summary_str,
        }
    
    def determine_problem_type(state: GraphState):
        print("    * DETERMINING PROBLEM TYPE")
        
        problem_type = state.get("problem_type")
        
        if problem_type:
            print(f"    * USER-SPECIFIED PROBLEM TYPE: {problem_type}")
            return {
                "problem_type": problem_type,
                "problem_type_analysis": f"# Problem Type Analysis\n\nUser-specified problem type: {problem_type}"
            }
        
        problem_type_prompt = PromptTemplate(
            template="""
            You are an expert in machine learning who specializes in determining the type of predictive modeling problem.
            
            Analyze the provided dataset summary and analyses to determine the appropriate problem type.
            
            DATA SUMMARY: 
            {all_datasets_summary}
            
            TARGET COLUMN: {target_column}
            
            TEMPORAL ANALYSIS: 
            {temporal_analysis}
            
            CORRELATION ANALYSIS: 
            {correlation_analysis}
            
            FEATURE RECOMMENDATIONS:
            {feature_recommendations}
            
            TASK:
            
            1. Determine the appropriate problem type from the following categories:
               - Regression (continuous numeric predictions)
               - Binary Classification (two categories)
               - Multi-class Classification (more than two categories)
               - Time Series Forecasting (predicting future values based on temporal patterns)
               - Clustering (identifying groups in data without a target variable)
               - Anomaly Detection (identifying outliers or unusual patterns)
            
            2. Explain your reasoning for the determined problem type.
            
            3. Specify any sub-type or special characteristics of the problem.
            
            RETURN FORMAT:
            
            Return your analysis in the following format:
            
            PROBLEM_TYPE: The type of problem (e.g., Regression, Classification, Time Series, etc.)
            
            ANALYSIS:
            A detailed explanation of your reasoning, including evidence from the data summary and analyses.
            """,
            input_variables=[
                "all_datasets_summary",
                "target_column",
                "temporal_analysis",
                "correlation_analysis",
                "feature_recommendations",
            ],
        )
        
        problem_type_analyzer = problem_type_prompt | llm
        
        response = problem_type_analyzer.invoke(
            {
                "all_datasets_summary": state.get("all_datasets_summary"),
                "target_column": state.get("target_column"),
                "temporal_analysis": state.get("temporal_analysis"),
                "correlation_analysis": state.get("correlation_analysis"),
                "feature_recommendations": state.get("feature_recommendations"),
            }
        )
        
        response_content = response.content
        
        # Extract problem type
        if "PROBLEM_TYPE:" in response_content:
            problem_type = response_content.split("PROBLEM_TYPE:")[1].split("\n")[0].strip()
        else:
            problem_type = "Unknown"
        
        # Create formatted analysis
        problem_type_analysis = "# Problem Type Analysis\n\n"
        problem_type_analysis += f"## Determined Problem Type: {problem_type}\n\n"
        
        if "ANALYSIS:" in response_content:
            analysis_text = response_content.split("ANALYSIS:")[1].strip()
            problem_type_analysis += analysis_text
        
        return {
            "problem_type": problem_type,
            "problem_type_analysis": problem_type_analysis,
        }
    
    def recommend_models(state: GraphState):
        print("    * RECOMMENDING MODELS")
        
        num_models = state.get("num_models", 3)
        
        model_recommendation_prompt = PromptTemplate(
            template="""
                You are a machine learning expert who specializes in recommending appropriate models for different prediction tasks.

                Your recommendations must be practical, popular, and aligned with the problem definition.

                DATA SUMMARY:
                {all_datasets_summary}

                TARGET COLUMN: {target_column}

                PROBLEM TYPE: {problem_type}

                PROBLEM TYPE ANALYSIS:
                {problem_type_analysis}

                TEMPORAL ANALYSIS:
                {temporal_analysis}

                CORRELATION ANALYSIS:
                {correlation_analysis}

                FEATURE RECOMMENDATIONS:
                {feature_recommendations}

                TASK:

                1. Recommend a **prioritized list of exactly {num_models} machine learning models** that are most appropriate for this problem. The recommendations should be:
                - Grounded in the **problem type** and **target variable**
                - Informed by the dataset structure and characteristics
                - Limited to **widely adopted, well-supported, and easy-to-integrate** models

                2. For each model:
                - Provide a concise explanation of how it works
                - Justify its suitability for this specific problem
                - Highlight key hyperparameters to tune
                - Mention any limitations or caveats

                3. Strongly consider the following factors in your recommendations:
                - Problem type (classification, regression, time series, etc.)
                - Type and distribution of the target variable
                - Dataset size and dimensionality
                - Presence of temporal features
                - Feature relationships (linear, non-linear)
                - Explainability needs
                - Training/inference efficiency
                - Robustness to outliers or noise
                - **Ease of use with common ML libraries (e.g., scikit-learn, XGBoost, LightGBM)**

                IMPORTANT CONSTRAINTS:
                - Do NOT recommend Prophet or other highly specialized, hard-to-maintain, or rarely used models
                - Do NOT suggest obscure or academic models that aren't widely used in industry
                - DO prioritize models that are popular, well-documented, and easy to tune
                - DO adapt your recommendations to the specific problem type (e.g., classification should not include regression models)

                RETURN FORMAT:

                Return your response in JSON with this structure:
                ```json
                {{
                "recommended_models": [
                    "Model1Name",
                    "Model2Name",
                    "Model3Name"
                ],
                "model_descriptions": "# Recommended Models\\n\\n## 1. Model1Name\\n\\nDescription of Model1...\\n\\n## 2. Model2Name\\n\\nDescription of Model2..."
                }}
            """,
            input_variables=[
                "all_datasets_summary",
                "target_column",
                "problem_type",
                "problem_type_analysis",
                "temporal_analysis",
                "correlation_analysis",
                "feature_recommendations",
                "num_models",
            ],
        )
        
        model_recommender = model_recommendation_prompt | llm
        
        response = model_recommender.invoke(
            {
                "all_datasets_summary": state.get("all_datasets_summary"),
                "target_column": state.get("target_column"),
                "problem_type": state.get("problem_type"),
                "problem_type_analysis": state.get("problem_type_analysis"),
                "temporal_analysis": state.get("temporal_analysis"),
                "correlation_analysis": state.get("correlation_analysis"),
                "feature_recommendations": state.get("feature_recommendations"),
                "num_models": num_models,
            }
        )
        
        response_content = response.content
        
        # Extract JSON content
        json_str = response_content
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()
        
        try:
            recommendations = json.loads(json_str)
            recommended_models = recommendations.get("recommended_models", [])
            model_descriptions = recommendations.get("model_descriptions", "")
            
            if not recommended_models:
                return {
                    "recommended_models": [],
                    "model_descriptions": "# Model Recommendation Failed\n\nNo models were recommended. Please review the problem type and dataset characteristics.",
                    "recommended_steps": "# Model Recommendation Failed\n\nNo models were recommended. Please review the problem type and dataset characteristics.",
                    "model_recommendation_error": "Model recommendation failed: No models were returned",
                }
        except json.JSONDecodeError:
            return {
                "recommended_models": [],
                "model_descriptions": "# Model Recommendation Failed\n\nThere was an error parsing the model recommendations. Please try again with different parameters.",
                "recommended_steps": "# Model Recommendation Failed\n\nThere was an error parsing the model recommendations. Please try again with different parameters.",
                "model_recommendation_error": "Model recommendation failed: JSON parsing error",
            }
        
        # Convert the steps into the format expected by the human reviewer
        formatted_recommendations = "# Recommended Models:\n\n"
        for i, model in enumerate(recommended_models, 1):
            formatted_recommendations += f"## {i}. {model}\n\n"
        
        return {
            "recommended_models": recommended_models,
            "model_descriptions": model_descriptions,
            "recommended_steps": formatted_recommendations,
        }
    
    def create_model_recommendation_function(state: GraphState):
        print("    * CREATING MODEL RECOMMENDATION FUNCTION")
        
        # If model recommendation failed, don't try to create a function
        if state.get("model_recommendation_error"):
            return {
                "model_recommendation_function": "# Model recommendation failed, no function generated",
                "model_recommendation_function_path": "",
                "model_recommendation_function_file_name": "",
                "model_recommendation_function_name": function_name,
            }
        
        model_function_prompt = PromptTemplate(
            template="""
            You are a machine learning expert who specializes in creating Python functions for model recommendation.
            
            Your task is to create a Python function called {function_name} that implements a model recommendation system.
            
            DATA SUMMARY: 
            {all_datasets_summary}
            
            TARGET COLUMN: {target_column}
            
            PROBLEM TYPE: {problem_type}
            
            RECOMMENDED MODELS:
            {recommended_models}
            
            MODEL DESCRIPTIONS:
            {model_descriptions}
            
            FUNCTION REQUIREMENTS:
            
            1. The function should be named {function_name} and should:
               - Take a pandas DataFrame and target column name as input
               - Analyze the characteristics of the dataset
               - Return a list of recommended model names as strings based on the dataset characteristics
            
            2. Your function must implement logic to select the recommended models based on:
               - Data characteristics (size, feature types, target distribution)
               - Problem type detection (regression, classification, time series)
               - Feature relationships with the target
            
            3. The function should:
               - Include all imports inside the function
               - Return two outputs: the list of recommended model names as strings, and a markdown-formatted string with model descriptions
               - Include appropriate comments for readability
            
            PYTHON CODE FORMAT:
            ```python
            def {function_name}(data, target_column="{target_column}", num_models=5):
                import pandas as pd
                import numpy as np
                
                # Create a copy of the input dataframe
                df = data.copy()
                
                # Analyze dataset characteristics
                # ...
                
                # Determine problem type
                # ...
                
                # Recommend models based on data characteristics and problem type
                recommended_models = []
                model_descriptions = ""
                
                # ...
                
                return recommended_models, model_descriptions
            ```
            
            Return only valid Python code that will run without errors.
            """,
            input_variables=[
                "all_datasets_summary",
                "target_column",
                "problem_type",
                "recommended_models",
                "model_descriptions",
                "function_name",
            ],
        )
        
        model_function_generator = model_function_prompt | llm | PythonOutputParser()
        
        response = model_function_generator.invoke(
            {
                "all_datasets_summary": state.get("all_datasets_summary"),
                "target_column": state.get("target_column"),
                "problem_type": state.get("problem_type"),
                "recommended_models": json.dumps(state.get("recommended_models")),
                "model_descriptions": state.get("model_descriptions"),
                "function_name": function_name,
            }
        )
        
        # Add comments and relocate imports if needed
        response = relocate_imports_inside_function(response)
        response = add_comments_to_top(response, agent_name=AGENT_NAME)
        
        # Log the function
        file_path, file_name_2 = log_ai_function(
            response=response,
            file_name=file_name,
            log=log,
            log_path=log_path,
            overwrite=overwrite,
        )
        
        return {
            "model_recommendation_function": response,
            "model_recommendation_function_path": file_path,
            "model_recommendation_function_file_name": file_name_2,
            "model_recommendation_function_name": function_name,
        }

    def human_review_recommendations(
        state: GraphState,
    ) -> Command[Literal["recommend_models", "create_model_recommendation_function"]]:
        prompt_text = "Are the following model recommendations appropriate? (Answer 'yes' or provide modifications)\n{steps}"
        
        return node_func_human_review(
            state=state,
            prompt_text=prompt_text,
            yes_goto="create_model_recommendation_function",
            no_goto="recommend_models",
            user_instructions_key=None,
            recommended_steps_key="recommended_steps",
        )

    def report_agent_outputs(state: GraphState):
        return node_func_report_agent_outputs(
            state=state,
            keys_to_include=[
                "problem_type_analysis",
                "recommended_models",
                "model_descriptions",
                "model_recommendation_function",
            ],
            result_key="messages",
            role=AGENT_NAME,
            custom_title="Model Recommendation Agent Outputs",
        )
    
    # Define node functions
    node_functions = {
        "analyze_dataset_structure": analyze_dataset_structure,
        "determine_problem_type": determine_problem_type,
        "recommend_models": recommend_models,
        "human_review_recommendations": human_review_recommendations,
        "create_model_recommendation_function": create_model_recommendation_function,
        "report_agent_outputs": report_agent_outputs,
    }

    # Create edges for the graph
    edges = []

    # Start -> Analyze Dataset Structure
    edges.append(("__start__", "analyze_dataset_structure"))

    # Analyze Dataset Structure -> Determine Problem Type
    edges.append(("analyze_dataset_structure", "determine_problem_type"))

    # Determine Problem Type -> Recommend Models
    edges.append(("determine_problem_type", "recommend_models"))

    # If human_in_the_loop, add human review step
    if human_in_the_loop:
        edges.append(("recommend_models", "human_review_recommendations"))
    else:
        edges.append(("recommend_models", "create_model_recommendation_function"))

    # Create Function -> Report
    edges.append(("create_model_recommendation_function", "report_agent_outputs"))

    # Report -> End
    edges.append(("report_agent_outputs", "__end__"))

    # Create the graph
    from langgraph.graph import StateGraph, END

    workflow = StateGraph(GraphState)

    # Add nodes
    for name, func in node_functions.items():
        workflow.add_node(name, func)

    # Add regular edges
    for start, end in edges:
        if start == "__start__":
            workflow.set_entry_point(end)
        elif end == "__end__":
            workflow.add_edge(start, END)
        else:
            workflow.add_edge(start, end)

    # Compile the graph
    app = workflow.compile(checkpointer=checkpointer)

    return app