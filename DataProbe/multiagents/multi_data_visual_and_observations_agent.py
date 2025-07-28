# Multi Data Visualization Agent with Observations

# Libraries
from typing import TypedDict, Annotated, Sequence, Literal, List, Dict, Any
import operator
import os
import json
import pandas as pd

from langchain.prompts import PromptTemplate
from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseLanguageModel

from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Checkpointer

from IPython.display import Markdown, display

import plotly.io as pio
from plotly.graph_objs import Figure

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
from DataProbe.agents import DataVisualizationAgent
from DataProbe.agents.data_plot_observations_agent import DataPlotObservationsAgent  # Import the new agent

# Setup
AGENT_NAME = "multi_data_visualisation_and_observations_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")

# Class
class MultiDataVisualObsAgent:
    """
    Creates an wrapper data visualization agent that analyzes a dataframe, 
    recommends useful visualizations, generates each visualization, and provides
    insightful observations for each plot. This creates a comprehensive analysis
    package with both visual and textual insights.
    
    Returns
    --------
    MultiDataVisualObsAgent
        An enhanced wrapper visualization agent with observations.
    """
    
    def __init__(
        self,
        model: BaseLanguageModel,
        visualization_agent: DataVisualizationAgent = None,
        observation_agent: DataPlotObservationsAgent = None,
        n_samples: int = 30,
        max_visualizations: int = 5,
        max_observations_per_plot: int = 5,
        log: bool = False,
        log_path: str = None,
        human_in_the_loop: bool = False,
        checkpointer: Checkpointer = None,
    ):
        self._params = {
            "model": model,
            "visualization_agent": visualization_agent,
            "observation_agent": observation_agent,
            "n_samples": n_samples,
            "max_visualizations": max_visualizations,
            "max_observations_per_plot": max_observations_per_plot,
            "log": log,
            "log_path": log_path,
            "human_in_the_loop": human_in_the_loop,
            "checkpointer": checkpointer,
        }
        
        # Create visualization agent if not provided
        if visualization_agent is None:
            self._params["visualization_agent"] = DataVisualizationAgent(
                model=model,
                n_samples=n_samples,
                log=log,
                log_path=log_path,
                human_in_the_loop=False  # We handle human review at the wrapper level
            )
            
        # Create observation agent if not provided
        if observation_agent is None:
            self._params["observation_agent"] = DataPlotObservationsAgent(
                model=model,
                n_samples=n_samples,
                log=log,
                log_path=log_path,
                human_in_the_loop=False,  # We handle human review at the wrapper level
                max_observations_per_plot=max_observations_per_plot
            )
            
        self._compiled_graph = self._make_compiled_graph()
        self.response = None
        
    def _make_compiled_graph(self):
        """
        Create the compiled graph for the enhanced data visualization wrapper agent.
        Running this method will reset the response to None.
        """
        self.response = None
        return make_multi_data_visualisation_and_observations_agent(**self._params)
    
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
        data_raw: pd.DataFrame,
        user_instructions: str = None,
        **kwargs,
    ):
        """
        Asynchronously invokes the agent to generate recommended visualizations and observations.
        The response is stored in the 'response' attribute.
            
        Returns
        -------
        None
        """
        response = await self._compiled_graph.ainvoke(
            {
                "user_instructions": user_instructions,
                "data_raw": data_raw.to_dict(),
            },
            **kwargs,
        )
        self.response = response
        return None
        
    def invoke_agent(
        self,
        data_raw: pd.DataFrame,
        user_instructions: str = None,
        **kwargs,
    ):
        """
        Synchronously invokes the agent to generate recommended visualizations and observations.
        The response is stored in the 'response' attribute.
            
        Returns
        -------
        None
        """
        response = self._compiled_graph.invoke(
            {
                "user_instructions": user_instructions,
                "data_raw": data_raw.to_dict(),
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
            The workflow summary.
        """
        if self.response and self.response.get("messages"):
            summary = get_generic_summary(
                json.loads(self.response.get("messages")[-1].content)
            )
            if markdown:
                return Markdown(summary)
            else:
                return summary
                
    def get_recommended_visualizations(self, markdown=False):
        """
        Retrieves the list of recommended visualization types.
            
        Returns
        -------
        str or Markdown
            The recommended visualizations if available, otherwise None.
        """
        if self.response:
            viz_list = self.response.get("recommended_visualizations_formatted", "")
            if markdown:
                return Markdown(viz_list)
            return viz_list
        return None
        
    def get_generated_plots(self):
        """
        Retrieves all generated Plotly plots.
        
        Returns
        -------
        dict
            Dictionary of {plot_title: plot_dict} for all generated visualizations.
        """
        if self.response:
            return self.response.get("generated_plots", {})
        return {}
        
    def get_plot_observations(self):
        """
        Retrieves all observations for each plot.
        
        Returns
        -------
        dict
            Dictionary of {plot_title: observations_list} for all generated visualizations.
        """
        if self.response:
            return self.response.get("plot_observations", {})
        return {}
        
    def display_plots_with_observations(self, width=900, height=500, use_html=True):
        """
        Displays all generated plots with their observations in the notebook.

        Returns
        -------
        None
        """
        plots = self.get_generated_plots()
        observations = self.get_plot_observations()
        
        if not plots:
            print("No plots have been generated yet.")
            return
            
        for title, plot_dict in plots.items():
            print(f"\n## {title}")
            try:
                # Convert to Figure object if it's not already one
                if isinstance(plot_dict, Figure):
                    fig = plot_dict
                else:
                    fig = Figure(plot_dict)
                
                fig.update_layout(width=width, height=height)
                try:
                    display(fig)
                except (ImportError, ValueError) as e:
                    # If direct display fails (e.g., nbformat missing), try HTML fallback
                    if use_html:
                        print(f"Direct display failed: {e}")
                        print("Falling back to HTML rendering...")
                        import plotly.io as pio
                        html = pio.to_html(fig, full_html=False)
                        from IPython.display import HTML
                        display(HTML(html))
                    else:
                        raise e
                        
                # Display observations for this plot if available
                if title in observations and observations[title]:
                    print("\n### Key Observations:")
                    for idx, obs in enumerate(observations[title], 1):
                        print(f"\n**{idx}. {obs.get('title')}**")
                        print(f"{obs.get('description')}")
                        
            except Exception as e:
                print(f"Error displaying plot '{title}': {e}")
                print("Try installing required packages: pip install nbformat>=4.2.0")
                
    def save_plots_and_observations(self, output_dir="./analysis", format="png", width=900, height=500):
        """
        Saves all generated plots and observations to files.
            
        Returns
        -------
        list
            List of saved file paths.
        """
        import os
        
        plots = self.get_generated_plots()
        observations = self.get_plot_observations()
        
        if not plots:
            print("No plots have been generated yet.")
            return []
            
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Create subdirectories
        plots_dir = os.path.join(output_dir, "plots")
        observations_dir = os.path.join(output_dir, "observations")
        
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        if not os.path.exists(observations_dir):
            os.makedirs(observations_dir)
            
        saved_files = []
        for title, plot_dict in plots.items():
            try:
                # Create a safe filename from the title
                safe_title = "".join([c if c.isalnum() else "_" for c in title]).rstrip("_")
                
                # Save plot
                plot_filepath = os.path.join(plots_dir, f"{safe_title}.{format}")
                
                # Create figure and save
                # Convert to Figure object if it's not already one
                if isinstance(plot_dict, Figure):
                    fig = plot_dict
                else:
                    fig = Figure(plot_dict)
                    
                fig.update_layout(width=width, height=height)
                fig.write_image(plot_filepath)
                saved_files.append(plot_filepath)
                print(f"Saved plot: {plot_filepath}")
                
                # Save observations if available
                if title in observations and observations[title]:
                    obs_filepath = os.path.join(observations_dir, f"{safe_title}_observations.md")
                    with open(obs_filepath, 'w') as f:
                        f.write(f"# Observations for: {title}\n\n")
                        for idx, obs in enumerate(observations[title], 1):
                            f.write(f"## {idx}. {obs.get('title')}\n")
                            f.write(f"{obs.get('description')}\n\n")
                    saved_files.append(obs_filepath)
                    print(f"Saved observations: {obs_filepath}")
                    
                    # Also save as JSON for programmatic access
                    json_filepath = os.path.join(observations_dir, f"{safe_title}_observations.json")
                    with open(json_filepath, 'w') as f:
                        json.dump(observations[title], f, indent=2)
                    saved_files.append(json_filepath)
                    
            except Exception as e:
                print(f"Error saving plot or observations for '{title}': {e}")
                
        # Save a summary file with links to all plots and observations
        summary_filepath = os.path.join(output_dir, "analysis_summary.md")
        try:
            with open(summary_filepath, 'w') as f:
                f.write("# Data Visualization Analysis Summary\n\n")
                f.write(f"Analysis generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("## Visualizations\n\n")
                for title in plots.keys():
                    safe_title = "".join([c if c.isalnum() else "_" for c in title]).rstrip("_")
                    f.write(f"- [{title}](plots/{safe_title}.{format})\n")
                    
                f.write("\n## Observations\n\n")
                for title in observations.keys():
                    safe_title = "".join([c if c.isalnum() else "_" for c in title]).rstrip("_")
                    f.write(f"- [{title} Observations](observations/{safe_title}_observations.md)\n")
                
            saved_files.append(summary_filepath)
            print(f"Saved analysis summary: {summary_filepath}")
        except Exception as e:
            print(f"Error saving analysis summary: {e}")
                
        return saved_files
        
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
def make_multi_data_visualisation_and_observations_agent(
    model: BaseLanguageModel,
    visualization_agent: DataVisualizationAgent,
    observation_agent: DataPlotObservationsAgent,
    n_samples: int = 30,
    max_visualizations: int = 5,
    max_observations_per_plot: int = 5,
    log: bool = False,
    log_path: str = None,
    human_in_the_loop: bool = False,
    checkpointer: Checkpointer = None,
):
    """
    Creates an enhanced wrapper data visualization agent that analyzes a dataframe, 
    recommends useful visualizations, generates each visualization, and provides
    insightful observations for each plot.
    
    Returns
    -------
    app : langchain.graphs.CompiledStateGraph
        The enhanced data visualization wrapper agent as a state graph.
    """
    
    llm = model
    
    if human_in_the_loop and checkpointer is None:
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
        user_instructions: str
        data_raw: dict
        all_datasets_summary: str
        recommended_visualizations: List[Dict[str, str]]
        recommended_visualizations_formatted: str
        generated_plots: Dict[str, Any]
        plot_observations: Dict[str, List[Dict[str, str]]]
        summary: str
        
    def analyze_data_for_visualizations(state: GraphState):
        """
        Analyzes the dataframe and recommends useful visualization types.
        """
        print(format_agent_name(AGENT_NAME))
        print("    * ANALYZE DATA AND RECOMMEND VISUALIZATIONS")
        
        visualization_recommendation_prompt = PromptTemplate(
            template="""
            You are a data visualization expert who specializes in recommending the most insightful plot types for datasets.
            
            You will analyze a dataset summary and recommend {max_visualizations} or fewer unique visualizations that would provide the most valuable insights.
            
            USER CONTEXT OR QUESTION (if provided):
            {user_instructions}
            
            DATA SUMMARY: 
            {all_datasets_summary}
            
            IMPORTANT:
            
            - Recommend visualizations that will help uncover insights, patterns, correlations, and anomalies in the data.
            - For each visualization, specify:
              1. The precise type of plot (bar, line, scatter, histogram, heatmap, box plot, etc.)
              2. What columns/features should be visualized 
              3. A brief explanation of what insights this visualization might reveal
            - Vary your recommendations to cover different aspects of the data.
            - Consider the data types of each column when making recommendations.
            - Prioritize visualizations that answer the user's question if one was provided.
            
            DATA TYPE CONSIDERATIONS:
            
            - Categorical vs. Categorical: Consider bar charts, heatmaps, or mosaic plots
            - Categorical vs. Numerical: Consider box plots, violin plots, or grouped bar charts
            - Numerical vs. Numerical: Consider scatter plots, bubble charts, or heatmaps
            - Time Series: Consider line charts, area charts, or calendar heatmaps
            - Distributions: Consider histograms, density plots, or Q-Q plots
            
            RETURN FORMAT:
            
            Return your recommendations in JSON format with the following structure:
            ```json
            [
              {{
                "title": "Descriptive title for the visualization",
                "plot_type": "Specific plot type",
                "columns": "Columns to visualize",
                "rationale": "Why this visualization is useful",
                "instructions": "Detailed instructions for the data visualization agent"
              }},
              ...
            ]
            ```
            
            Ensure the "instructions" field contains clear, specific instructions that can be directly passed to a data visualization agent.
            """,
            input_variables=[
                "user_instructions",
                "all_datasets_summary",
                "max_visualizations",
            ],
        )
        
        data_raw = state.get("data_raw")
        df = pd.DataFrame.from_dict(data_raw)
        
        all_datasets_summary = get_dataframe_summary(
            [df], n_sample=n_samples, skip_stats=False
        )
        
        all_datasets_summary_str = "\n\n".join(all_datasets_summary)
        
        visualization_recommender = visualization_recommendation_prompt | llm
        
        response = visualization_recommender.invoke(
            {
                "user_instructions": state.get("user_instructions") or "Please recommend useful visualizations for gaining insights about this dataset.",
                "all_datasets_summary": all_datasets_summary_str,
                "max_visualizations": max_visualizations,
            }
        )
        
        # Parse JSON response
        json_str = response.content
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()
            
        try:
            recommended_visualizations = json.loads(json_str)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            print("Failed to parse JSON response. Using simplified format.")
            recommended_visualizations = [
                {
                    "title": "Dataset Overview",
                    "plot_type": "Correlation Heatmap",
                    "columns": "All numerical columns",
                    "rationale": "To identify relationships between variables",
                    "instructions": "Create a correlation heatmap for all numerical columns in the dataset."
                }
            ]
            
        # Format for human review
        formatted_recommendations = "# Recommended Visualizations:\n\n"
        
        for i, viz in enumerate(recommended_visualizations, 1):
            formatted_recommendations += f"## {i}. {viz['title']}\n"
            formatted_recommendations += f"- **Plot Type:** {viz['plot_type']}\n"
            formatted_recommendations += f"- **Columns:** {viz['columns']}\n"
            formatted_recommendations += f"- **Rationale:** {viz['rationale']}\n"
            formatted_recommendations += f"- **Instructions:** {viz['instructions']}\n\n"
            
        return {
            "recommended_visualizations": recommended_visualizations,
            "recommended_visualizations_formatted": formatted_recommendations,
            "all_datasets_summary": all_datasets_summary_str,
        }
        
    def generate_visualizations(state: GraphState):
        """
        Calls the data visualization agent for each recommended visualization.
        """
        print("    * GENERATE VISUALIZATIONS")
        
        data_raw = state.get("data_raw")
        df = pd.DataFrame.from_dict(data_raw)
        
        # Get the recommended visualizations
        recommended_visualizations = state.get("recommended_visualizations")
        
        # Initialize dictionary to store generated plots
        generated_plots = {}
        
        # Generate each visualization
        for viz in recommended_visualizations:
            title = viz['title']
            instructions = viz['instructions']
            
            print(f"    * GENERATING: {title}")
            
            # Call the data visualization agent
            visualization_agent.invoke_agent(
                data_raw=df,
                user_instructions=instructions,
                max_retries=3,
                retry_count=0
            )
            
            # Store the generated plot with its title as the key
            plot_dict = visualization_agent.get_plotly_graph()
            if plot_dict is not None:
                generated_plots[title] = plot_dict
            
        return {
            "generated_plots": generated_plots,
        }
        
    def generate_observations(state: GraphState):
        """
        Calls the data plot observations agent for each generated visualization.
        """
        print("    * GENERATE OBSERVATIONS FOR EACH PLOT")
        
        data_raw = state.get("data_raw")
        df = pd.DataFrame.from_dict(data_raw)
        
        # Get the generated plots
        generated_plots = state.get("generated_plots")
        recommended_visualizations = state.get("recommended_visualizations")
        
        # Create a mapping of title to instructions for reference
        instructions_map = {viz["title"]: viz["instructions"] for viz in recommended_visualizations}
        
        # Initialize dictionary to store observations for each plot
        plot_observations = {}
        
        # Generate observations for each plot
        for title, plot_dict in generated_plots.items():
            print(f"    * ANALYZING: {title}")
            
            # Get the instructions used to create this plot
            plot_instructions = instructions_map.get(title, "")
            
            # Check if plot_dict is a Figure object and convert it to dict if needed
            from plotly.graph_objs import Figure
            if isinstance(plot_dict, Figure):
                plot_dict_serializable = plot_dict.to_dict()
            else:
                plot_dict_serializable = plot_dict
            
            # Call the observations agent
            observation_agent.invoke_agent(
                plot_dict=plot_dict_serializable,
                data_raw=df,
                plot_title=title,
                plot_instructions=plot_instructions,
                max_retries=3,
                retry_count=0
            )
            
            # Store the observations with the plot title as the key
            observations = observation_agent.get_observations()
            if observations:
                plot_observations[title] = observations
        
        return {
            "plot_observations": plot_observations,
        }
        
    def create_summary(state: GraphState):
        """
        Creates a comprehensive summary of the analysis.
        """
        print("    * CREATING ANALYSIS SUMMARY")
        
        summary_prompt = PromptTemplate(
            template="""
            You are a data scientist creating a comprehensive summary of a data visualization analysis.
            
            Based on the visualizations that were created and the observations made, create a concise summary 
            that ties together the main findings and insights from the analysis.
            
            USER CONTEXT OR QUESTION (if provided):
            {user_instructions}
            
            VISUALIZATIONS CREATED:
            {visualizations_list}
            
            KEY OBSERVATIONS:
            {observations_summary}
            
            IMPORTANT:
            
            - Focus on synthesizing the most important insights across all visualizations
            - Highlight patterns, trends, or relationships that emerged from multiple visualizations
            - Connect the findings back to the original user's question or context if provided
            - Keep the summary concise yet comprehensive (3-5 paragraphs)
            - Include specific data points or metrics when they are particularly important
            
            RETURN FORMAT:
            
            Return your summary in markdown format with appropriate headings and structure.
            """,
            input_variables=[
                "user_instructions",
                "visualizations_list",
                "observations_summary",
            ],
        )
        
        # Get the data needed for the summary
        generated_plots = state.get("generated_plots", {})
        plot_observations = state.get("plot_observations", {})
        
        # Create a list of visualizations
        visualizations_list = "\n".join([f"- {title} ({viz_type['plot_type']})" 
                                        for title, viz_type in zip(
                                            generated_plots.keys(), 
                                            state.get("recommended_visualizations", [])
                                        )])
        
        # Create a summary of observations
        observations_summary = ""
        for title, observations in plot_observations.items():
            observations_summary += f"\n## {title}\n"
            for obs in observations:
                observations_summary += f"- {obs['title']}: {obs['description'][:100]}...\n"
        
        # Generate the summary
        summary_generator = summary_prompt | llm
        
        response = summary_generator.invoke(
            {
                "user_instructions": state.get("user_instructions") or "",
                "visualizations_list": visualizations_list,
                "observations_summary": observations_summary,
            }
        )
        
        return {
            "summary": response.content,
        }
        
    # Human Review
    def human_review(
        state: GraphState,
    ) -> Command[Literal["analyze_data_for_visualizations", "generate_visualizations"]]:
        prompt_text = "Are the following visualization recommendations appropriate? (Answer 'yes' or provide modifications)\n{steps}"
        
        return node_func_human_review(
            state=state,
            prompt_text=prompt_text,
            yes_goto="generate_visualizations",
            no_goto="analyze_data_for_visualizations",
            user_instructions_key="user_instructions",
            recommended_steps_key="recommended_visualizations_formatted",
        )
        
    # Final reporting node
    def report_agent_outputs(state: GraphState):
        return node_func_report_agent_outputs(
            state=state,
            keys_to_include=[
                "recommended_visualizations_formatted",
                "summary",
            ],
            result_key="messages",
            role=AGENT_NAME,
            custom_title="Enhanced Data Visualization Analysis",
        )
        
    # Define the graph
    node_functions = {
        "analyze_data_for_visualizations": analyze_data_for_visualizations,
        "human_review": human_review,
        "generate_visualizations": generate_visualizations,
        "generate_observations": generate_observations,
        "create_summary": create_summary,
        "report_agent_outputs": report_agent_outputs,
    }
    
    # Create edges for the graph
    edges = []
    
    # Start -> Analyze Data
    edges.append(("__start__", "analyze_data_for_visualizations"))
    
    # If human_in_the_loop, add human review step
    if human_in_the_loop:
        edges.append(("analyze_data_for_visualizations", "human_review"))
    else:
        edges.append(("analyze_data_for_visualizations", "generate_visualizations"))
        
    # Generate Visualizations -> Generate Observations
    edges.append(("generate_visualizations", "generate_observations"))
    
    # Generate Observations -> Create Summary
    edges.append(("generate_observations", "create_summary"))
    
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