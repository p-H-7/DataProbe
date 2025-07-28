# Model Evaluation Agent
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
AGENT_NAME = "model_evaluation_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")

# Class
class ModelEvaluationAgent:
    """
    Model evaluation agent that implements and evaluates baseline models
    recommended by the model recommendation agent.
    """
    
    def __init__(
        self,
        model,
        n_samples=30,
        log=False,
        log_path=None,
        file_name="model_evaluation.py",
        function_name="evaluate_models",
        overwrite=True,
        human_in_the_loop=False,
        checkpointer=None,
        auto_install_libraries=True, 
        training_instructions=None

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
            "auto_install_libraries": auto_install_libraries,
            "training_instructions": training_instructions,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None
    
    def _make_compiled_graph(self):
        self.response = None
        return make_model_evaluation_agent(**self._params)
    
    def update_params(self, **kwargs):
        for k, v in kwargs.items():
            self._params[k] = v
        self._compiled_graph = self._make_compiled_graph()
    
    async def ainvoke_agent(
        self,
        engineered_data: pd.DataFrame,
        target_column: str,
        recommended_models: List[str],
        problem_type: str,
        test_size: float = 0.2,
        random_state: int = 42,
        training_instructions: str = None,
        **kwargs,
    ):
        response = await self._compiled_graph.ainvoke(
            {
                "engineered_data": engineered_data.to_dict(),
                "target_column": target_column,
                "recommended_models": recommended_models,
                "problem_type": problem_type,
                "test_size": test_size,
                "random_state": random_state,
                "training_instructions": training_instructions,
            },
            **kwargs,
        )
        self.response = response
        return None
    
    def invoke_agent(
        self,
        engineered_data: pd.DataFrame,
        target_column: str,
        recommended_models: List[str],
        problem_type: str,
        test_size: float = 0.2,
        random_state: int = 42,
        training_instructions: str = None,
        **kwargs,
    ):
        response = self._compiled_graph.invoke(
            {
                "engineered_data": engineered_data.to_dict(),
                "target_column": target_column,
                "recommended_models": recommended_models,
                "problem_type": problem_type,
                "test_size": test_size,
                "random_state": random_state,
                "training_instructions": training_instructions,
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
    
    def get_model_performance(self):
        """
        Returns the performance metrics for all evaluated models.
        
        Returns
        -------
        dict
            Dictionary with model names as keys and performance metrics as values.
        """
        if self.response:
            return self.response.get("model_performance", {})
        return {}
    
    def get_best_model(self):
        """
        Returns the name of the best performing model.
        
        Returns
        -------
        str
            Name of the best performing model.
        """
        if self.response:
            return self.response.get("best_model", "")
        return ""
    
    def get_performance_comparison(self, markdown=False):
        """
        Returns a comparative analysis of model performance.

        Returns
        -------
        str or Markdown
            The performance comparison analysis.
        """
        if self.response:
            comparison = self.response.get("performance_comparison", "")
            if markdown:
                return Markdown(comparison)
            return comparison
        return ""
    
    def get_evaluation_function(self, markdown=False):
        """
        Returns the generated Python function used for model evaluation.

        Returns
        -------
        str or Markdown
            The Python function code.
        """
        if self.response:
            func_code = self.response.get("model_evaluation_function", "")
            if markdown:
                return Markdown(f"```python\n{func_code}\n```")
            return func_code
        return ""
    
    def get_model_evaluation_results(self):
        """
        Returns the full evaluation results including trained models and predictions.
        
        Returns
        -------
        dict
            The complete evaluation results.
        """
        if self.response:
            return self.response.get("evaluation_results", {})
        return {}
    
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
def make_model_evaluation_agent(
    model,
    n_samples=30,
    log=False,
    log_path=None,
    file_name="model_evaluation.py",
    function_name="evaluate_models",
    overwrite=True,
    human_in_the_loop=False,
    checkpointer=None,
    auto_install_libraries=True,  # Add this parameter
    training_instructions=None,
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
        recommended_models: List[str]
        problem_type: str
        test_size: float
        random_state: int
        training_instructions: str
        all_datasets_summary: str
        evaluation_plan: str
        recommended_steps: str
        required_libraries: dict  # Add this field
        library_installation_results: dict  # Add this field
        model_evaluation_function: str
        model_evaluation_function_path: str
        model_evaluation_function_file_name: str
        model_evaluation_function_name: str
        model_evaluation_error: str
        evaluation_results: dict
        model_performance: dict
        best_model: str
        performance_comparison: str
    
    def analyze_dataset_structure(state: GraphState):
        print(format_agent_name(AGENT_NAME))
        print("    * ANALYZING DATASET STRUCTURE")
        
        engineered_data = state.get("engineered_data")
        df = pd.DataFrame.from_dict(engineered_data)
        
        all_datasets_summary = get_dataframe_summary(
            [df], n_sample=n_samples, skip_stats=False
        )
        all_datasets_summary_str = "\n\n".join(all_datasets_summary)
        
        # Verify that recommended models are provided
        recommended_models = state.get("recommended_models", [])
        if not recommended_models:
            raise ValueError("No recommended models provided. Cannot proceed with evaluation.")
        
        return {
            "all_datasets_summary": all_datasets_summary_str,
        }
    
    def detect_required_libraries(state: GraphState):
        """
        Uses an LLM prompt to detect required libraries based on recommended models.
            
        Returns
        -------
        dict
            Dictionary with required libraries
        """
        print("    * DETECTING REQUIRED LIBRARIES")
        
        libraries_prompt = PromptTemplate(
            template="""
            You are a machine learning expert who specializes in identifying required Python libraries.
            
            Based on the recommended models and problem type, provide a list of Python libraries that will be needed.
            
            PROBLEM TYPE: {problem_type}
            
            RECOMMENDED MODELS:
            {recommended_models}
            
            TASK:
            1. Identify all Python libraries that will be required to implement and evaluate the recommended models.
            2. Include both core libraries (numpy, pandas, scikit-learn) and specialized libraries for specific models.
            3. For each library, provide the correct pip installation name.
            
            Format your response as a Python dictionary where:
            - Keys are the import names used in code
            - Values are the pip installation package names
            
            EXAMPLE OUTPUT:
            ```python
            {{
                "numpy": "numpy",
                "pandas": "pandas",
                "sklearn": "scikit-learn",
                "xgboost": "xgboost",
                "matplotlib": "matplotlib",
                "statsmodels": "statsmodels"
            }}
            ```
            
            Return ONLY the Python dictionary, formatted exactly as shown above.
            """,
            input_variables=[
                "problem_type",
                "recommended_models",
            ],
        )
        
        libraries_detector = libraries_prompt | llm | PythonOutputParser()
        
        response = libraries_detector.invoke(
            {
                "problem_type": state.get("problem_type"),
                "recommended_models": json.dumps(state.get("recommended_models")),
            }
        )
        
        # Parse the response to get the required libraries
        try:
            required_libraries = eval(response)
            if not isinstance(required_libraries, dict):
                raise ValueError("Response is not a valid dictionary")
        except Exception as e:
            print(f"Error parsing library list: {str(e)}")
            # Fallback to a basic set of libraries
            required_libraries = {
                "numpy": "numpy",
                "pandas": "pandas",
                "sklearn": "scikit-learn",
                "matplotlib": "matplotlib"
            }
        
        return {
            "required_libraries": required_libraries
        }

    def install_required_libraries(state: GraphState):
        """
        Installs the required libraries detected by the LLM.

        Returns
        -------
        dict
            Dictionary with installation results
        """
        print("    * INSTALLING REQUIRED LIBRARIES")
        
        required_libraries = state.get("required_libraries", {})
        
        if not required_libraries:
            print("No required libraries specified. Skipping installation.")
            return {
                "library_installation_results": {
                    "installed": [],
                    "already_installed": [],
                    "failed": []
                }
            }
        
        import importlib
        import subprocess
        import sys
        
        installation_results = {"installed": [], "already_installed": [], "failed": []}
        
        for lib_name, package_name in required_libraries.items():
            try:
                # Try to import the library
                importlib.import_module(lib_name)
                print(f"Library '{lib_name}' is already installed.")
                installation_results["already_installed"].append(lib_name)
            except ImportError:
                # If import fails, install the library
                print(f"Installing '{package_name}'...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
                    print(f"Successfully installed '{package_name}'.")
                    installation_results["installed"].append(lib_name)
                except subprocess.CalledProcessError as e:
                    print(f"Failed to install '{package_name}': {str(e)}")
                    installation_results["failed"].append(lib_name)
        
        return {
            "library_installation_results": installation_results
        }
    
    def create_evaluation_plan(state: GraphState):
        print("    * CREATING EVALUATION PLAN")
        
        evaluation_plan_prompt = PromptTemplate(
            template="""
            You are a machine learning expert who specializes in model evaluation.
            
            Create a detailed plan for evaluating the recommended models on the given dataset.
            
            DATA SUMMARY: 
            {all_datasets_summary}
            
            TARGET COLUMN: {target_column}
            
            PROBLEM TYPE: {problem_type}
            
            RECOMMENDED MODELS:
            {recommended_models}
            
            TRAINING INSTRUCTIONS: {training_instructions}

            TASK:
            
            1. Create a plan to implement baseline versions of each recommended model.

            2. Follow the TRAINING INSTRUCTIONS to determine each of the following
            
            2. For each model, specify:
               - How to configure the model (any key hyperparameters to set)
               - How to fit the model to the training data
               - How to make predictions with the model
            
            3. Determine appropriate evaluation metrics based on the problem type:
               - For regression problems: RMSE, MAE, R²
               - For classification problems: accuracy, precision, recall, F1-score, AUC-ROC
               - For time series forecasting: RMSE, MAE, MAPE
            
            4. Specify how to:
               - Split the data into training and test sets (test_size={test_size}, random_state={random_state})
               - Handle any preprocessing required for specific models
               - Compare model performance using appropriate metrics
            
            RETURN FORMAT:
            
            Return your evaluation plan in markdown format with appropriate headings and structure.
            Focus on creating a practical plan that can be implemented in Python code.
            """,
            input_variables=[
                "all_datasets_summary",
                "target_column",
                "problem_type",
                "recommended_models",
                "test_size",
                "random_state",
            ],
        )
        
        evaluation_planner = evaluation_plan_prompt | llm
        
        response = evaluation_planner.invoke(
            {
                "all_datasets_summary": state.get("all_datasets_summary"),
                "target_column": state.get("target_column"),
                "problem_type": state.get("problem_type"),
                "recommended_models": json.dumps(state.get("recommended_models")),
                "training_instructions": state.get("training_instructions", "No specific training instructions provided."),
                "test_size": state.get("test_size"),
                "random_state": state.get("random_state"),
            }
        )
        
        evaluation_plan = response.content
        
        # Format the plan as recommended steps for human review
        recommended_steps = f"# Model Evaluation Plan:\n\n{evaluation_plan}"
        
        return {
            "evaluation_plan": evaluation_plan,
            "recommended_steps": recommended_steps,
        }
    
    def create_evaluation_function(state: GraphState):
        print("    * CREATING MODEL EVALUATION FUNCTION")
        
        evaluation_function_prompt = PromptTemplate(
            template="""
            You are a machine learning expert who specializes in implementing and evaluating models.
            
            Create a Python function that evaluates the recommended models on the given dataset according to the evaluation plan.
            
            DATA SUMMARY: 
            {all_datasets_summary}
            
            TARGET COLUMN: {target_column}
            
            PROBLEM TYPE: {problem_type}
            
            RECOMMENDED MODELS:
            {recommended_models}

            TRAINING INSTRUCTIONS:
            {training_instructions} 
            
            EVALUATION PLAN:
            {evaluation_plan}
            
            FUNCTION REQUIREMENTS:
            
            1. The function should be named {function_name} and should:
               - Take a pandas DataFrame, target column name, and list of model names as input
               - Follow the specific training instructions provided
               - Consider the number of datapoints specified for training
               - Handle prediction length requirements as specified
               - Implement and evaluate each recommended model
               - Return comprehensive evaluation results
            
            2. The function must:
               - Include all necessary imports inside the function
               - Validate that all requested models are available/implementable
               - Handle data preprocessing appropriate for each model type
               - Calculate appropriate metrics based on the problem type
               - Identify the best performing model
               - Generate a comparison of model performance
            
            3. Support for model types:
               - For regression: LinearRegression, RandomForest, GradientBoosting, etc.
               - For classification: LogisticRegression, RandomForest, GradientBoosting, etc.
               - For time series: ARIMA, Prophet, SimpleExpSmoothing, etc.
            
            4. Return structure:
               - A dictionary containing trained models, predictions, metrics, and a performance comparison

            "IMPORTANT: 
                - Do NOT use the `squared` parameter in mean_squared_error.
                - Always calculate RMSE as: `rmse = np.sqrt(mean_squared_error(y_true, y_pred))` for compatibility."
                - Ensure backward compatibility with scikit-learn versions <0.22 by avoiding parameters not universally supported (e.g., `squared`).
                - Always handle datetime columns by converting or excluding them for models that do not support them (e.g., XGBoost).
            
            PYTHON CODE FORMAT:
            ```python
            def {function_name}(data, target_column="{target_column}", model_list=None, problem_type="{problem_type}", 
                             test_size={test_size}, random_state={random_state}):
                import pandas as pd
                import numpy as np
                import matplotlib.pyplot as plt
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                # Create a copy of the input dataframe
                df = data.copy()
                
                # Validate model list
                if model_list is None or len(model_list) == 0:
                    raise ValueError("No models specified for evaluation")
                
                # Prepare the data
                # ...
                
                # Evaluate each model
                # ...
                
                # Compare performance and select best model
                # ...
                
                # Return results
                return results
            ```
            
            Return only valid Python code that will run without errors.
            """,
            input_variables=[
                "all_datasets_summary",
                "target_column",
                "problem_type",
                "recommended_models",
                "evaluation_plan",
                "training_instructions",
                "function_name",
                "test_size",
                "random_state",
            ],
        )
        
        evaluation_function_generator = evaluation_function_prompt | llm | PythonOutputParser()
        
        response = evaluation_function_generator.invoke(
            {
                "all_datasets_summary": state.get("all_datasets_summary"),
                "target_column": state.get("target_column"),
                "problem_type": state.get("problem_type"),
                "recommended_models": json.dumps(state.get("recommended_models")),
                "evaluation_plan": state.get("evaluation_plan"),
                "training_instructions": state.get("training_instructions", "No specific training instructions provided."),
                "function_name": function_name,
                "test_size": state.get("test_size"),
                "random_state": state.get("random_state"),
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
            "model_evaluation_function": response,
            "model_evaluation_function_path": file_path,
            "model_evaluation_function_file_name": file_name_2,
            "model_evaluation_function_name": function_name,
        }
    
    def execute_evaluation_function(state: GraphState):
        print("    * EXECUTING MODEL EVALUATION")
        
        engineered_data = state.get("engineered_data")
        df = pd.DataFrame.from_dict(engineered_data)
        
        model_evaluation_function = state.get("model_evaluation_function")
        function_name = state.get("model_evaluation_function_name")
        
        try:
            namespace = {}
            exec(model_evaluation_function, namespace)
            
            func = namespace[function_name]
            
            evaluation_results = func(
                df, 
                target_column=state.get("target_column"),
                model_list=state.get("recommended_models"),
                problem_type=state.get("problem_type"),
                test_size=state.get("test_size"),
                random_state=state.get("random_state")
            )
            
            # Extract relevant information from results
            model_performance = evaluation_results.get("metrics", {})
            best_model = evaluation_results.get("best_model", "")
            performance_comparison = evaluation_results.get("performance_comparison", "")
            
            return {
                "evaluation_results": evaluation_results,
                "model_performance": model_performance,
                "best_model": best_model,
                "performance_comparison": performance_comparison,
                "model_evaluation_error": "",
            }
            
        except Exception as e:
            error_message = f"An error occurred during model evaluation: {str(e)}"
            print(error_message)
            return {
                "model_evaluation_error": error_message,
            }
    
    def fix_evaluation_function(state: GraphState):
        print("    * FIXING MODEL EVALUATION FUNCTION")
        
        fix_prompt = PromptTemplate(
            template="""
            You are a machine learning expert who specializes in fixing Python code errors.
            
            The following model evaluation function has an error. Please fix it.
            
            ORIGINAL FUNCTION:
            ```python
            {model_evaluation_function}
            ```
            
            ERROR MESSAGE:
            {error}
            
            Fix the function and return only the corrected Python code that will run without errors.
            The function should be named {function_name} and follow the requirements from the original.
            """,
            input_variables=[
                "model_evaluation_function",
                "error",
                "function_name",
            ],
        )
        
        fix_generator = fix_prompt | llm | PythonOutputParser()
        
        response = fix_generator.invoke(
            {
                "model_evaluation_function": state.get("model_evaluation_function"),
                "error": state.get("model_evaluation_error"),
                "function_name": state.get("model_evaluation_function_name"),
            }
        )
        
        file_path, file_name_2 = log_ai_function(
            response=response,
            file_name=state.get("model_evaluation_function_file_name"),
            log=log,
            log_path=log_path,
            overwrite=True,
        )
        
        return {
            "model_evaluation_function": response,
            "model_evaluation_function_path": file_path,
        }
    
    def create_performance_analysis(state: GraphState):
        print("    * CREATING PERFORMANCE ANALYSIS")
        
        performance_comparison = state.get("performance_comparison")
        if (performance_comparison is not None and isinstance(performance_comparison, str) and performance_comparison.strip()):
            return {}
        
        def make_json_safe(obj):
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: make_json_safe(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_safe(item) for item in obj]
            else:
                try:
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)
        
        # Otherwise, generate a new analysis
        analysis_prompt = PromptTemplate(
            template="""
            You are a data scientist analyzing model performance.
            
            Create a concise and informative analysis of the model performance results.
            
            PROBLEM TYPE: {problem_type}
            
            MODEL PERFORMANCE METRICS:
            {model_performance}
            
            BEST PERFORMING MODEL: {best_model}
            
            TASK:
            
            1. Analyze the performance of each model, focusing on:
            - Strengths and weaknesses of each model
            - Comparison of key metrics across models
            - Why the best model outperformed others
            
            2. Provide insights on:
            - Any patterns in model performance
            - Potential improvements or next steps
            - Trade-offs between models (e.g., accuracy vs. interpretability)
            
            RETURN FORMAT:
            
            Return your analysis in markdown format with appropriate headings and structure.
            Keep it concise yet informative (3-5 paragraphs).
            """,
            input_variables=[
                "problem_type",
                "model_performance",
                "best_model",
            ],
        )
        
        analysis_generator = analysis_prompt | llm
        
        model_performance = state.get("model_performance", {})
        model_performance_safe = make_json_safe(model_performance)
        
        response = analysis_generator.invoke(
            {
                "problem_type": state.get("problem_type"),
                "model_performance": json.dumps(model_performance_safe, indent=2),
                "best_model": state.get("best_model"),
            }
        )
        
        return {
            "performance_comparison": response.content,
        }

    def human_review_evaluation_plan(
        state: GraphState,
    ) -> Command[Literal["create_evaluation_plan", "create_evaluation_function"]]:
        prompt_text = "Is the following model evaluation plan appropriate? (Answer 'yes' or provide modifications)\n{steps}"
        
        return node_func_human_review(
            state=state,
            prompt_text=prompt_text,
            yes_goto="create_evaluation_function",
            no_goto="create_evaluation_plan",
            user_instructions_key=None,
            recommended_steps_key="recommended_steps",
        )

    def report_agent_outputs(state: GraphState):
        # Helper function to make objects JSON serializable
        def make_json_safe(obj):
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif hasattr(obj, '__dict__') and hasattr(obj, '__class__'):
                # Handle sklearn objects, pipelines, etc.
                return f"<{obj.__class__.__name__} object>"
            elif isinstance(obj, dict):
                return {k: make_json_safe(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_safe(item) for item in obj]
            else:
                try:
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)
        
        # Create a JSON-safe version of the state for reporting
        safe_state = {}
        keys_to_include = [
            "model_performance",
            "best_model", 
            "performance_comparison",
            "model_evaluation_function",
        ]
        
        for key in keys_to_include:
            safe_state[key] = make_json_safe(state.get(key, f"<{key}_not_found_in_state>"))
        
        return node_func_report_agent_outputs(
            state={"messages": state.get("messages", []), **safe_state},
            keys_to_include=keys_to_include,
            result_key="messages",
            role=AGENT_NAME,
            custom_title="Model Evaluation Agent Outputs",
        )
    
    # Define node functions
    node_functions = {
        "analyze_dataset_structure": analyze_dataset_structure,
        "create_evaluation_plan": create_evaluation_plan,
        "human_review_evaluation_plan": human_review_evaluation_plan,
        "create_evaluation_function": create_evaluation_function,
        "detect_required_libraries": detect_required_libraries,  # Add new node function
        "install_required_libraries": install_required_libraries,  # Add new node function
        "execute_evaluation_function": execute_evaluation_function,
        "fix_evaluation_function": fix_evaluation_function,
        "create_performance_analysis": create_performance_analysis,
        "report_agent_outputs": report_agent_outputs,
    }

    # Create edges for the graph
    edges = []

    # Start -> Analyze Dataset Structure
    edges.append(("__start__", "analyze_dataset_structure"))

    # Analyze Dataset Structure -> Create Evaluation Plan
    edges.append(("analyze_dataset_structure", "create_evaluation_plan"))

    # If human_in_the_loop, add human review step
    if human_in_the_loop:
        edges.append(("create_evaluation_plan", "human_review_evaluation_plan"))
        edges.append(("human_review_evaluation_plan", "create_evaluation_function"))
    else:
        edges.append(("create_evaluation_plan", "create_evaluation_function"))

    # Add library detection and installation steps if auto_install_libraries is enabled
    if auto_install_libraries:
        edges.append(("create_evaluation_function", "detect_required_libraries"))
        edges.append(("detect_required_libraries", "install_required_libraries"))
        edges.append(("install_required_libraries", "execute_evaluation_function"))
    else:
        # Create Function -> Execute Evaluation (direct path if auto_install_libraries is disabled)
        edges.append(("create_evaluation_function", "execute_evaluation_function"))

    # Fix Function -> Execute Evaluation (retry)
    edges.append(("fix_evaluation_function", "execute_evaluation_function"))

    # Execute Evaluation -> Create Performance Analysis
    edges.append(("execute_evaluation_function", "create_performance_analysis"))

    # Create Performance Analysis -> Report
    edges.append(("create_performance_analysis", "report_agent_outputs"))

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

    # Add conditional edge for error handling
    workflow.add_conditional_edges(
        "execute_evaluation_function",
        lambda x: bool(x.get("model_evaluation_error", "")),
        {
            True: "fix_evaluation_function",
            False: "create_performance_analysis"
        }
    )

    # Handle human_in_the_loop conditional edge
    if human_in_the_loop:
        workflow.add_conditional_edges(
            "human_review_evaluation_plan",
            lambda x: x.get("human_approved", True),
            {
                True: "create_evaluation_function",
                False: "create_evaluation_plan"
            }
        )
        # Remove the direct edge added earlier
        workflow.remove_edge("human_review_evaluation_plan", "create_evaluation_function")

    # Compile the graph
    app = workflow.compile(checkpointer=checkpointer)

    return app