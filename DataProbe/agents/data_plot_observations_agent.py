# Data Plor Observations Agent
# Libraries
from typing import TypedDict, Annotated, Sequence, Literal, Dict, Any
import operator
import os
import json
import pandas as pd

from langchain.prompts import PromptTemplate
from langchain_core.messages import BaseMessage

from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Checkpointer

from IPython.display import Markdown

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

# Setup
AGENT_NAME = "data_plot_observations_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")

# Class
class DataPlotObservationsAgent(BaseAgent):
    """
    An agent that analyzes data visualizations and generates 
    meaningful insights and observations from each plot. The agent interprets trends, patterns, 
    correlations, and anomalies visible in the visualizations and provides context-rich explanations
    that help users understand what the data is showing.
    
    Returns
    --------
    DataPlotObservationsAgent : langchain.graphs.CompiledStateGraph
        A data plot observations agent implemented as a compiled state graph.
    """
    
    def __init__(
        self,
        model,
        n_samples=30,
        log=False,
        log_path=None,
        file_name="plot_observations.json",
        overwrite=True,
        human_in_the_loop=False,
        max_observations_per_plot=5,
        checkpointer=None,
    ):
        self._params = {
            "model": model,
            "n_samples": n_samples,
            "log": log,
            "log_path": log_path,
            "file_name": file_name,
            "overwrite": overwrite,
            "human_in_the_loop": human_in_the_loop,
            "max_observations_per_plot": max_observations_per_plot,
            "checkpointer": checkpointer,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None
        
    def _make_compiled_graph(self):
        """
        Create the compiled graph for the data plot observations agent.
        Running this method will reset the response to None.
        """
        self.response = None
        return make_data_plot_observations_agent(**self._params)
        
    def update_params(self, **kwargs):
        """
        Updates the agent's parameters and rebuilds the compiled graph.
        """
        # Update parameters
        for k, v in kwargs.items():
            self._params[k] = v
        # Rebuild the compiled graph
        self._compiled_graph = self._make_compiled_graph()
        
    async def ainvoke_agent(
        self,
        plot_dict: Dict,
        data_raw: pd.DataFrame,
        plot_title: str,
        plot_instructions: str = None,
        max_retries: int = 3,
        retry_count: int = 0,
        **kwargs,
    ):
        """
        Asynchronously invokes the agent to generate observations for a visualization.
        The response is stored in the 'response' attribute.
        
        Returns
        -------
        None
        """
        # Check if plot_dict is a Figure object and convert to dict if needed
        from plotly.graph_objs import Figure
        if isinstance(plot_dict, Figure):
            plot_dict = plot_dict.to_dict()
            
        response = await self._compiled_graph.ainvoke(
            {
                "plot_dict": plot_dict,
                "data_raw": data_raw.to_dict(),
                "plot_title": plot_title,
                "plot_instructions": plot_instructions,
                "max_retries": max_retries,
                "retry_count": retry_count,
            },
            **kwargs,
        )
        self.response = response
        return None
        
    def invoke_agent(
        self,
        plot_dict: Dict,
        data_raw: pd.DataFrame,
        plot_title: str,
        plot_instructions: str = None,
        max_retries: int = 3,
        retry_count: int = 0,
        **kwargs,
    ):
        """
        Synchronously invokes the agent to generate observations for a visualization.
        The response is stored in the 'response' attribute.
            
        Returns
        -------
        None
        """
        # Check if plot_dict is a Figure object and convert to dict if needed
        from plotly.graph_objs import Figure
        if isinstance(plot_dict, Figure):
            plot_dict = plot_dict.to_dict()
            
        response = self._compiled_graph.invoke(
            {
                "plot_dict": plot_dict,
                "data_raw": data_raw.to_dict(),
                "plot_title": plot_title,
                "plot_instructions": plot_instructions,
                "max_retries": max_retries,
                "retry_count": retry_count,
            },
            **kwargs,
        )
        self.response = response
        return None
        
    def get_workflow_summary(self, markdown=False):
        """
        Retrieves the agent's workflow summary.
            
        Returns
        -------
        str or Markdown
            The workflow summary if available, otherwise None.
        """
        if self.response and self.response.get("messages"):
            summary = get_generic_summary(
                json.loads(self.response.get("messages")[-1].content)
            )
            if markdown:
                return Markdown(summary)
            else:
                return summary
                
    def get_observations(self, markdown=False):
        """
        Retrieves the observations generated for the plot.
            
        Returns
        -------
        str or Markdown or list
            The observations if available, otherwise None.
        """
        if self.response:
            observations = self.response.get("plot_observations", [])
            
            if markdown:
                observations_md = f"# Observations for: {self.response.get('plot_title', 'Plot')}\n\n"
                
                for idx, obs in enumerate(observations, 1):
                    observations_md += f"## {idx}. {obs.get('title', 'Observation')}\n"
                    observations_md += f"{obs.get('description', '')}\n\n"
                    
                return Markdown(observations_md)
            
            return observations
        return None
        
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

# Agent function       
def make_data_plot_observations_agent(
    model,
    n_samples=30,
    log=False,
    log_path=None,
    file_name="plot_observations.json",
    overwrite=True,
    human_in_the_loop=False,
    max_observations_per_plot=5,
    checkpointer=None,
):
    """
    Creates a data plot observations agent that analyzes data visualizations and generates
    meaningful insights and observations from each plot.
    
    Returns
    -------
    app : langchain.graphs.CompiledStateGraph
        The data plot observations agent as a state graph.
    """
    
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
            
    # Define GraphState for the router
    class GraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        plot_dict: Dict[str, Any]
        data_raw: Dict[str, Any]
        plot_title: str
        plot_instructions: str
        plot_observations: list
        plot_observations_formatted: str
        all_datasets_summary: str
        max_retries: int
        retry_count: int
        
    def analyze_plot_for_observations(state: GraphState):
        """
        Analyzes the plot and generates observations and insights.
        """
        print(format_agent_name(AGENT_NAME))
        print(f"    * ANALYZING PLOT: {state.get('plot_title')}")
        
        plot_observation_prompt = PromptTemplate(
            template="""
            You are a data science expert who specializes in interpreting data visualizations and extracting meaningful insights.
            
            You will analyze a plot and its underlying data to generate {max_observations_per_plot} or fewer key observations that would be valuable to a business or technical audience.
            
            PLOT TITLE:
            {plot_title}
            
            PLOT INSTRUCTIONS/CONTEXT (if provided):
            {plot_instructions}
            
            PLOT DATA (Plotly JSON structure):
            {plot_data_summary}
            
            DATA SUMMARY: 
            {all_datasets_summary}
            
            IMPORTANT:
            
            - Generate observations that provide genuine insights about patterns, trends, correlations, anomalies, or other notable aspects visible in the plot.
            - For each observation:
              1. Provide a concise, specific title that highlights the key finding
              2. Include a detailed explanation that contextualizes the finding and explains its significance
              3. When appropriate, include specific values/numbers from the visualization to support your observation
              4. If applicable, suggest potential business implications or actions that could be taken based on this finding
            - Avoid generic observations that could apply to any data visualization.
            - Focus on what's actually visible in the plot rather than making unsupported speculations.
            - Prioritize observations that would be most valuable to the user based on the plot context.
            
            RETURN FORMAT:
            
            Return your observations in JSON format with the following structure:
            ```json
            [
              {{
                "title": "Concise title highlighting the key finding",
                "description": "Detailed explanation of the observation including specific data points and significance"
              }},
              ...
            ]
            ```
            
            Ensure each observation provides meaningful, data-driven insights that help the user interpret the visualization.
            """,
            input_variables=[
                "plot_title",
                "plot_instructions",
                "plot_data_summary",
                "all_datasets_summary",
                "max_observations_per_plot",
            ],
        )
        
        # Get data from state
        plot_dict = state.get("plot_dict")
        data_raw = state.get("data_raw")
        df = pd.DataFrame.from_dict(data_raw)
        
        # Check if plot_dict is a Figure object and convert to dict if needed
        from plotly.graph_objs import Figure
        if isinstance(plot_dict, Figure):
            plot_dict = plot_dict.to_dict()
        
        # Prepare plot data summary - wrap this in a try/except to handle potential serialization issues
        try:
            plot_data_json = json.dumps(plot_dict, indent=2)
            plot_data_summary = plot_data_json[:10000]  # Limit size to avoid token limits
            if len(plot_data_json) > 10000:
                plot_data_summary += "... [truncated for brevity]"
        except TypeError:
            # If serialization fails, provide a simplified representation
            plot_data_summary = "Plot data could not be fully serialized, but contains the following keys: " + str(list(plot_dict.keys()) if isinstance(plot_dict, dict) else "unknown structure")
        
        # Get dataset summary
        all_datasets_summary = get_dataframe_summary(
            [df], n_sample=n_samples, skip_stats=False
        )
        
        all_datasets_summary_str = "\n\n".join(all_datasets_summary)
        
        # Get observations
        plot_analyzer = plot_observation_prompt | llm
        
        response = plot_analyzer.invoke(
            {
                "plot_title": state.get("plot_title"),
                "plot_instructions": state.get("plot_instructions") or "",
                "plot_data_summary": plot_data_summary,
                "all_datasets_summary": all_datasets_summary_str,
                "max_observations_per_plot": max_observations_per_plot,
            }
        )
        
        # Parse JSON response
        json_str = response.content
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()
            
        try:
            plot_observations = json.loads(json_str)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            print("Failed to parse JSON response. Using simplified format.")
            plot_observations = [
                {
                    "title": "General Observation",
                    "description": "This plot shows important trends in the data that warrant further investigation."
                }
            ]
            
        # Format for human review
        formatted_observations = f"# Observations for: {state.get('plot_title')}\n\n"
        
        for i, obs in enumerate(plot_observations, 1):
            formatted_observations += f"## {i}. {obs['title']}\n"
            formatted_observations += f"{obs['description']}\n\n"
            
        # Save to log file if logging is enabled
        if log:
            log_file = os.path.join(log_path, file_name)
            
            # Check if we need to create a unique filename
            if not overwrite and os.path.exists(log_file):
                base, ext = os.path.splitext(file_name)
                log_file = os.path.join(log_path, f"{base}_{state.get('plot_title').replace(' ', '_')}{ext}")
                
            try:
                with open(log_file, 'w') as f:
                    json.dump({
                        "plot_title": state.get("plot_title"),
                        "observations": plot_observations
                    }, f, indent=2)
                print(f"Observations saved to: {log_file}")
            except Exception as e:
                print(f"Error saving observations to file: {e}")
            
        return {
            "plot_observations": plot_observations,
            "plot_observations_formatted": formatted_observations,
            "all_datasets_summary": all_datasets_summary_str,
        }

    # Human Review function definition - this was the missing piece
    def human_review(
        state: GraphState,
    ) -> Command[Literal["analyze_plot_for_observations", "report_agent_outputs"]]:
        prompt_text = "Are the following plot observations accurate and insightful? (Answer 'yes' or provide feedback)\n{steps}"
        
        return node_func_human_review(
            state=state,
            prompt_text=prompt_text,
            yes_goto="report_agent_outputs",
            no_goto="analyze_plot_for_observations",
            user_instructions_key="plot_instructions",
            recommended_steps_key="plot_observations_formatted",
        )

    # Final reporting node
    def report_agent_outputs(state: GraphState):
        return node_func_report_agent_outputs(
            state=state,
            keys_to_include=[
                "plot_observations_formatted",
            ],
            result_key="messages",
            role=AGENT_NAME,
            custom_title=f"Data Plot Observations: {state.get('plot_title')}",
        )
        
    # Define the graph
    node_functions = {
        "analyze_plot_for_observations": analyze_plot_for_observations,
        "human_review": human_review,
        "report_agent_outputs": report_agent_outputs,
    }
    
    # Create edges for the graph
    edges = []
    
    # Start -> Analyze Plot
    edges.append(("__start__", "analyze_plot_for_observations"))
    
    # If human_in_the_loop, add human review step
    if human_in_the_loop:
        edges.append(("analyze_plot_for_observations", "human_review"))
    else:
        edges.append(("analyze_plot_for_observations", "report_agent_outputs"))
        
    # Report -> End
    edges.append(("report_agent_outputs", "__end__"))
    
    # Create the graph
    from langgraph.graph import StateGraph, END
    
    workflow = StateGraph(GraphState)
    
    # Add nodes
    for name, func in node_functions.items():
        workflow.add_node(name, func)
        
    # Add edges
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