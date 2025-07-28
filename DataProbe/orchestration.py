from typing import Optional, Dict, List, Any, Union
import os
import pandas as pd
import numpy as np
import json
import datetime
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
import plotly.io as pio
from plotly.graph_objs import Figure

# PDF Generation
from fpdf import FPDF
import base64
from io import BytesIO

# Import all required agents
from DataProbe.agents import DataCleaningAgent
from DataProbe.multiagents.multi_data_visual_and_observations_agent import MultiDataVisualObsAgent
from DataProbe.multiagents.eda_feature_engineering_agent import EDAAgent
from DataProbe.agents.model_recommendation_agent import ModelRecommendationAgent
from DataProbe.multiagents.model_evaluation_agent import ModelEvaluationAgent

class AutomatedEDAOrchestrator:
    def __init__(
        self,
        models: Dict[str, Any],
        output_dir: str = "./eda_outputs",
        n_samples: int = 30,
        log: bool = True,
        log_path: str = "./logs",
        human_in_the_loop: bool = False,
        max_visualizations: int = 5,
        generate_pdf: bool = True
    ):
        self.models = models
        self.output_dir = output_dir
        self.n_samples = n_samples
        self.log = log
        self.log_path = log_path
        self.human_in_the_loop = human_in_the_loop
        self.max_visualizations = max_visualizations
        self.generate_pdf = generate_pdf

        required_models = ['cleaning', 'visualization', 'feature_engineering', 
                          'model_recommendation', 'model_evaluation']
        for model_key in required_models:
            if model_key not in self.models:
                raise ValueError(f"Missing required model for '{model_key}'. Please provide models for: {required_models}")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        self.plots_dir = os.path.join(output_dir, "plots")
        self.data_dir = os.path.join(output_dir, "data")
        self.models_dir = os.path.join(output_dir, "models")
        
        for dir_path in [self.plots_dir, self.data_dir, self.models_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                
        self.results = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_cleaning": {},
            "data_visualization": {},
            "feature_engineering": {},
            "model_recommendation": {},
            "model_evaluation": {}
        }
        
        self._init_agents()
        
    def _init_agents(self):
        self.data_cleaning_agent = DataCleaningAgent(
            model=self.models['cleaning'],
            n_samples=self.n_samples,
            log=self.log,
            log_path=self.log_path,
            human_in_the_loop=self.human_in_the_loop
        )
        
        self.visualization_agent = MultiDataVisualObsAgent(
            model=self.models['visualization'],
            n_samples=self.n_samples,
            max_visualizations=self.max_visualizations,
            log=self.log,
            log_path=self.log_path,
            human_in_the_loop=self.human_in_the_loop
        )
        
        self.feature_engineering_agent = EDAAgent(
            model=self.models['feature_engineering'],
            visualization_wrapper=self.visualization_agent,
            n_samples=self.n_samples,
            max_visualizations=self.max_visualizations,
            log=self.log,
            log_path=self.log_path,
            human_in_the_loop=self.human_in_the_loop
        )
        
        self.model_recommendation_agent = ModelRecommendationAgent(
            model=self.models['model_recommendation'],
            n_samples=self.n_samples,
            log=self.log,
            log_path=self.log_path,
            human_in_the_loop=self.human_in_the_loop
        )
        
        self.model_evaluation_agent = ModelEvaluationAgent(
            model=self.models['model_evaluation'],
            n_samples=self.n_samples,
            log=self.log,
            log_path=self.log_path,
            human_in_the_loop=self.human_in_the_loop,
            auto_install_libraries=True,
            training_instructions=None
        )
    
    def run_pipeline(
        self,
        df: pd.DataFrame,
        target_column: str,
        date_column: Optional[str] = None,
        data_cleaning_instructions: Optional[str] = None,
        visualization_instructions: Optional[str] = None,
        problem_type: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        training_instructions: Optional[str] = None,
    ):
        print("=" * 80)
        print("STARTING AUTOMATED EDA PIPELINE")
        print("=" * 80)
        
        orig_df_path = os.path.join(self.data_dir, "original_data.csv")
        df.to_csv(orig_df_path, index=False)
        self.results["original_data_path"] = orig_df_path
        self.results["original_shape"] = df.shape
        
        try:
            print("\n" + "=" * 30)
            print("STEP 1: DATA CLEANING")
            print("=" * 30)
            
            cleaned_df = self._run_data_cleaning(df, data_cleaning_instructions)
            
            print("\n" + "=" * 30)
            print("STEP 2: DATA VISUALIZATION")
            print("=" * 30)
            
            visualization_results = self._run_data_visualization(cleaned_df, visualization_instructions)
            
            print("\n" + "=" * 30)
            print("STEP 3: FEATURE ENGINEERING")
            print("=" * 30)
            
            engineered_df, feature_engineering_results = self._run_feature_engineering(cleaned_df, target_column, date_column)
            
            print("\n" + "=" * 30)
            print("STEP 4: MODEL RECOMMENDATION")
            print("=" * 30)
            
            model_recommendations = self._run_model_recommendation(
                engineered_df, 
                target_column, 
                feature_engineering_results.get("temporal_analysis", ""), 
                feature_engineering_results.get("correlation_analysis", ""),
                feature_engineering_results.get("feature_recommendations_formatted", ""),
                problem_type
            )
            
            print("\n" + "=" * 30)
            print("STEP 5: MODEL EVALUATION")
            print("=" * 30)
            
            evaluation_results = self._run_model_evaluation(
                engineered_df,
                target_column,
                model_recommendations.get("recommended_models", []),
                model_recommendations.get("problem_type", "regression"),
                test_size,
                random_state,
                training_instructions
            )
            
            if self.generate_pdf:
                pdf_path = self._generate_pdf_report()
                self.results["pdf_report_path"] = pdf_path
                print(f"\nFinal PDF report generated: {pdf_path}")
            
            results_path = os.path.join(self.output_dir, "eda_results.json")
            
            json_safe_results = self._make_json_serializable(self.results)
            
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(json_safe_results, f, indent=2, ensure_ascii=False)
            
            print("\n" + "=" * 80)
            print(f"EDA PIPELINE COMPLETED SUCCESSFULLY")
            print(f"Results saved to: {results_path}")
            print("=" * 80)
            
            return self.results
            
        except Exception as e:
            print(f"Error in EDA pipeline: {str(e)}")
            error_results_path = os.path.join(self.output_dir, "eda_results_error.json")
            
            self.results["error"] = {
                "message": str(e),
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            json_safe_results = self._make_json_serializable(self.results)
            
            with open(error_results_path, 'w', encoding='utf-8') as f:
                json.dump(json_safe_results, f, indent=2, ensure_ascii=False)
                
            print(f"Partial results saved to: {error_results_path}")
            raise
    
    def _run_data_cleaning(self, df: pd.DataFrame, instructions: Optional[str] = None) -> pd.DataFrame:
        print("Running data cleaning agent...")
        self.data_cleaning_agent.invoke_agent(
            data_raw=df,
            user_instructions=instructions
        )
        
        cleaned_df = self.data_cleaning_agent.get_data_cleaned()
        
        cleaned_df_path = os.path.join(self.data_dir, "cleaned_data.csv")
        cleaned_df.to_csv(cleaned_df_path, index=False)
        
        self.results["data_cleaning"] = {
            "cleaned_data_path": cleaned_df_path,
            "original_shape": df.shape,
            "cleaned_shape": cleaned_df.shape,
            "cleaning_function": self.data_cleaning_agent.get_data_cleaner_function(),
            "recommended_steps": self.data_cleaning_agent.get_recommended_cleaning_steps()
        }
        
        print(f"Data cleaning completed: {df.shape} → {cleaned_df.shape}")
        return cleaned_df
    
    def _run_data_visualization(self, df: pd.DataFrame, instructions: Optional[str] = None) -> Dict:
        print("Running data visualization agent...")
        self.visualization_agent.invoke_agent(
            data_raw=df,
            user_instructions=instructions or "Generate comprehensive visualizations to understand the dataset patterns, distributions, and relationships."
        )
        
        plots = self.visualization_agent.get_generated_plots()
        observations = self.visualization_agent.get_plot_observations()
        
        saved_files = []
        for title, plot_dict in plots.items():
            safe_title = "".join([c if c.isalnum() else "_" for c in title]).rstrip("_")
            
            plot_filepath = os.path.join(self.plots_dir, f"{safe_title}.png")
            
            try:
                if isinstance(plot_dict, Figure):
                    fig = plot_dict
                else:
                    fig = Figure(plot_dict)
                    
                fig.update_layout(width=900, height=500)
                fig.write_image(plot_filepath)
                saved_files.append(plot_filepath)
            except Exception as e:
                print(f"Error saving plot {title}: {str(e)}")
        
        self.results["data_visualization"] = {
            "recommended_visualizations": self.visualization_agent.get_recommended_visualizations(),
            "plots": {title: path for title, path in zip(plots.keys(), saved_files)},
            "observations": observations
        }
        
        print(f"Data visualization completed: {len(plots)} plots generated")
        return self.results["data_visualization"]
    
    def _run_feature_engineering(
        self, 
        df: pd.DataFrame, 
        target_column: str, 
        date_column: Optional[str] = None
    ) -> tuple:
        print("Running feature engineering agent...")
        self.feature_engineering_agent.invoke_agent(
            data_raw=df,
            target_column=target_column,
            date_column=date_column
        )
        
        engineered_df = self.feature_engineering_agent.get_engineered_dataframe()
        
        engineered_df_path = os.path.join(self.data_dir, "engineered_data.csv")
        engineered_df.to_csv(engineered_df_path, index=False)
        
        self.results["feature_engineering"] = {
            "engineered_data_path": engineered_df_path,
            "cleaned_shape": df.shape,
            "engineered_shape": engineered_df.shape,
            "engineered_feature_names": self.feature_engineering_agent.get_engineered_feature_names(),
            "temporal_analysis": self.feature_engineering_agent.get_temporal_analysis_results(),
            "correlation_analysis": self.feature_engineering_agent.get_correlation_analysis_results(),
            "feature_recommendations_formatted": self.feature_engineering_agent.get_feature_engineering_recommendations(),
            "feature_engineering_function": self.feature_engineering_agent.get_feature_engineering_function()
        }
        
        feature_plots = self.feature_engineering_agent.get_generated_plots()
        feature_observations = self.feature_engineering_agent.get_plot_observations()
        
        feature_saved_files = []
        for title, plot_dict in feature_plots.items():
            safe_title = "feature_" + "".join([c if c.isalnum() else "_" for c in title]).rstrip("_")
            
            plot_filepath = os.path.join(self.plots_dir, f"{safe_title}.png")
            
            try:
                if isinstance(plot_dict, Figure):
                    fig = plot_dict
                else:
                    fig = Figure(plot_dict)
                    
                fig.update_layout(width=900, height=500)
                fig.write_image(plot_filepath)
                feature_saved_files.append(plot_filepath)
            except Exception as e:
                print(f"Error saving feature plot {title}: {str(e)}")
        
        self.results["feature_engineering"]["feature_plots"] = {
            title: path for title, path in zip(feature_plots.keys(), feature_saved_files)
        }
        self.results["feature_engineering"]["feature_plot_observations"] = feature_observations
        
        print(f"Feature engineering completed: {len(self.results['feature_engineering']['engineered_feature_names'])} features engineered")
        return engineered_df, self.results["feature_engineering"]
    
    def _run_model_recommendation(
        self,
        engineered_df: pd.DataFrame,
        target_column: str,
        temporal_analysis: str,
        correlation_analysis: str,
        feature_recommendations: str,
        problem_type: Optional[str] = None
    ) -> Dict:
        print("Running model recommendation agent...")
        self.model_recommendation_agent.invoke_agent(
            engineered_data=engineered_df,
            target_column=target_column,
            temporal_analysis=temporal_analysis,
            correlation_analysis=correlation_analysis,
            feature_recommendations=feature_recommendations,
            problem_type=problem_type
        )
        
        recommended_models = self.model_recommendation_agent.get_recommended_models()
        problem_type = self.model_recommendation_agent.get_problem_type()
        
        self.results["model_recommendation"] = {
            "problem_type": problem_type,
            "recommended_models": recommended_models,
            "problem_type_analysis": self.model_recommendation_agent.get_problem_type_analysis(),
            "model_descriptions": self.model_recommendation_agent.get_model_descriptions(),
            "model_recommendation_function": self.model_recommendation_agent.get_model_recommendation_function()
        }
        
        print(f"Model recommendation completed: {problem_type} problem, {len(recommended_models)} models recommended")
        return self.results["model_recommendation"]
    
    def _run_model_evaluation(
        self,
        engineered_df: pd.DataFrame,
        target_column: str,
        recommended_models: List[str],
        problem_type: str,
        test_size: float = 0.2,
        random_state: int = 42,
        training_instructions: Optional[str] = None 
    ) -> Dict:
        print("Running model evaluation agent...")
        self.model_evaluation_agent.invoke_agent(
            engineered_data=engineered_df,
            target_column=target_column,
            recommended_models=recommended_models,
            problem_type=problem_type,
            test_size=test_size,
            random_state=random_state,
            training_instructions=training_instructions
        )
        
        model_performance = self.model_evaluation_agent.get_model_performance()
        best_model = self.model_evaluation_agent.get_best_model()
        
        self.results["model_evaluation"] = {
            "model_performance": model_performance,
            "best_model": best_model,
            "performance_comparison": self.model_evaluation_agent.get_performance_comparison(),
            "model_evaluation_function": self.model_evaluation_agent.get_evaluation_function()
        }
        
        print(f"Model evaluation completed: Best model is {best_model}")
        return self.results["model_evaluation"]
    
    def _make_json_serializable(self, obj):
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {self._make_json_serializable(k): self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(i) for i in obj]
        elif isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        elif hasattr(obj, 'to_dict'):
            return self._make_json_serializable(obj.to_dict())
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        else:
            try:
                json.dumps(obj)
                return obj
            except (TypeError, OverflowError):
                return str(obj)
    
    def _sanitize_text(self, text):
        """Sanitize text to avoid Unicode encoding issues in PDF"""
        if text is None:
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Replace problematic Unicode characters
        replacements = {
            '\u2013': '-',  # en dash
            '\u2014': '--',  # em dash
            '\u2018': "'",   # left single quotation mark
            '\u2019': "'",   # right single quotation mark
            '\u201c': '"',   # left double quotation mark
            '\u201d': '"',   # right double quotation mark
            '\u2022': '-',   # bullet point
            '\u2026': '...',  # horizontal ellipsis
        }
        
        for unicode_char, replacement in replacements.items():
            text = text.replace(unicode_char, replacement)
        
        # Remove or replace any remaining non-ASCII characters
        text = text.encode('ascii', 'replace').decode('ascii')
        
        return text
    
    def _generate_pdf_report(self) -> str:
        print("Generating comprehensive PDF report...")
        
        class EDAPDF(FPDF):
            def __init__(self, orchestrator):
                super().__init__()
                self.orchestrator = orchestrator
                
            def header(self):
                self.set_font('Arial', 'B', 15)
                self.cell(0, 10, 'Automated EDA Report', 0, 1, 'C')
                self.ln(5)
                
            def footer(self):
                self.set_y(-15)
                self.set_font('Arial', 'I', 8)
                self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
                
            def chapter_title(self, title):
                title = self.orchestrator._sanitize_text(title)
                self.set_font('Arial', 'B', 12)
                self.set_fill_color(200, 220, 255)
                self.cell(0, 10, title, 0, 1, 'L', 1)
                self.ln(5)
                
            def chapter_body(self, body, markdown=False):
                body = self.orchestrator._sanitize_text(body)
                self.set_font('Arial', '', 10)
                if markdown:
                    lines = body.split('\n')
                    for line in lines:
                        line = self.orchestrator._sanitize_text(line)
                        if line.startswith('# '):
                            self.set_font('Arial', 'B', 14)
                            self.cell(0, 10, line[2:], 0, 1)
                            self.ln(2)
                        elif line.startswith('## '):
                            self.set_font('Arial', 'B', 12)
                            self.cell(0, 10, line[3:], 0, 1)
                            self.ln(2)
                        elif line.startswith('### '):
                            self.set_font('Arial', 'B', 11)
                            self.cell(0, 10, line[4:], 0, 1)
                        elif line.startswith('- '):
                            self.set_font('Arial', '', 10)
                            self.cell(0, 6, '  - ' + line[2:], 0, 1)
                        else:
                            self.set_font('Arial', '', 10)
                            self.multi_cell(0, 5, line)
                            self.ln(2)
                else:
                    self.multi_cell(0, 5, body)
                self.ln(5)
                
            def add_image(self, img_path, w=180):
                if os.path.exists(img_path):
                    try:
                        h = w / 16 * 9
                        self.image(img_path, x=10, w=w, h=h)
                        self.ln(5)
                    except Exception as e:
                        error_msg = self.orchestrator._sanitize_text(f"Error adding image: {str(e)}")
                        self.multi_cell(0, 5, error_msg)
                else:
                    error_msg = self.orchestrator._sanitize_text(f"Image not found: {img_path}")
                    self.multi_cell(0, 5, error_msg)

            def add_plot_with_observations(self, title, plot_path, observations):
                title = self.orchestrator._sanitize_text(title)
                self.set_font('Arial', 'B', 11)
                self.cell(0, 10, title, 0, 1)
                
                self.add_image(plot_path)
                
                if observations:
                    self.set_font('Arial', 'B', 10)
                    self.cell(0, 10, "Key Observations:", 0, 1)
                    self.set_font('Arial', '', 10)
                    
                    for i, obs in enumerate(observations, 1):
                        obs_title = self.orchestrator._sanitize_text(obs.get('title', 'Observation'))
                        obs_desc = self.orchestrator._sanitize_text(obs.get('description', ''))
                        
                        self.set_font('Arial', 'B', 10)
                        self.cell(0, 6, f"{i}. {obs_title}", 0, 1)
                        self.set_font('Arial', '', 10)
                        self.multi_cell(0, 5, obs_desc)
                        self.ln(2)
                
                self.ln(5)
                
        pdf = EDAPDF(self)
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        pdf.set_font('Arial', 'B', 20)
        pdf.cell(0, 20, 'Automated EDA Report', 0, 1, 'C')
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f"Generated on: {self.results['timestamp']}", 0, 1, 'C')
        pdf.ln(10)
        
        pdf.chapter_title("Table of Contents")
        toc_items = [
            "1. Executive Summary",
            "2. Data Cleaning",
            "3. Data Visualization",
            "4. Feature Engineering",
            "5. Model Recommendation",
            "6. Model Evaluation",
            "7. Conclusion"
        ]
        for item in toc_items:
            pdf.cell(0, 8, item, 0, 1)
        pdf.ln(10)
        
        pdf.add_page()
        pdf.chapter_title("1. Executive Summary")
        
        summary = f"""
        This report presents a comprehensive Exploratory Data Analysis (EDA) of the dataset.
        
        The original dataset contained {self.results['original_shape'][0]} rows and {self.results['original_shape'][1]} columns.
        After cleaning, the dataset had {self.results['data_cleaning']['cleaned_shape'][0]} rows and {self.results['data_cleaning']['cleaned_shape'][1]} columns.
        
        Feature engineering created {len(self.results['feature_engineering'].get('engineered_feature_names', []))} new features,
        resulting in a final dataset with {self.results['feature_engineering']['engineered_shape'][0]} rows and {self.results['feature_engineering']['engineered_shape'][1]} columns.
        
        The problem was identified as a {self.results['model_recommendation']['problem_type']} problem.
        
        After evaluating {len(self.results['model_recommendation']['recommended_models'])} different models, 
        the best performing model was {self.results['model_evaluation']['best_model']}.
        """
        pdf.chapter_body(summary.strip())
        
        pdf.add_page()
        pdf.chapter_title("2. Data Cleaning")
        
        cleaning_text = f"""
        The data cleaning process transformed the dataset from {self.results['original_shape'][0]} rows and {self.results['original_shape'][1]} columns
        to {self.results['data_cleaning']['cleaned_shape'][0]} rows and {self.results['data_cleaning']['cleaned_shape'][1]} columns.
        
        The following cleaning steps were performed:
        """
        pdf.chapter_body(cleaning_text.strip())
        
        if self.results['data_cleaning'].get('recommended_steps'):
            pdf.chapter_body(self.results['data_cleaning'].get('recommended_steps'), markdown=True)
        
        pdf.add_page()
        pdf.chapter_title("3. Data Visualization")
        
        viz_intro = """
        The following visualizations provide key insights into the dataset patterns, distributions, and relationships.
        Each visualization is accompanied by key observations that highlight important findings.
        """
        pdf.chapter_body(viz_intro.strip())
        
        viz_results = self.results["data_visualization"]
        for title, plot_path in viz_results.get("plots", {}).items():
            observations = viz_results.get("observations", {}).get(title, [])
            pdf.add_plot_with_observations(title, plot_path, observations)
            pdf.add_page()
        
        pdf.add_page()
        pdf.chapter_title("4. Feature Engineering")
        
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 10, "Temporal Analysis", 0, 1)
        if self.results['feature_engineering'].get('temporal_analysis'):
            pdf.chapter_body(self.results['feature_engineering'].get('temporal_analysis'), markdown=True)
        
        pdf.add_page()
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 10, "Correlation Analysis", 0, 1)
        if self.results['feature_engineering'].get('correlation_analysis'):
            pdf.chapter_body(self.results['feature_engineering'].get('correlation_analysis'), markdown=True)
        
        pdf.add_page()
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 10, "Feature Engineering Recommendations", 0, 1)
        if self.results['feature_engineering'].get('feature_recommendations_formatted'):
            pdf.chapter_body(self.results['feature_engineering'].get('feature_recommendations_formatted'), markdown=True)
        
        for title, plot_path in self.results['feature_engineering'].get("feature_plots", {}).items():
            observations = self.results['feature_engineering'].get("feature_plot_observations", {}).get(title, [])
            pdf.add_page()
            pdf.add_plot_with_observations(title, plot_path, observations)
        
        pdf.add_page()
        pdf.chapter_title("5. Model Recommendation")
        
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 10, "Problem Type Analysis", 0, 1)
        if self.results['model_recommendation'].get('problem_type_analysis'):
            pdf.chapter_body(self.results['model_recommendation'].get('problem_type_analysis'), markdown=True)
        
        pdf.add_page()
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 10, "Recommended Models", 0, 1)
        if self.results['model_recommendation'].get('model_descriptions'):
            pdf.chapter_body(self.results['model_recommendation'].get('model_descriptions'), markdown=True)
        
        pdf.add_page()
        pdf.chapter_title("6. Model Evaluation")
        
        eval_intro = f"""
        The best performing model was {self.results['model_evaluation']['best_model']}.
        
        Below is a comparison of all evaluated models:
        """
        pdf.chapter_body(eval_intro.strip())
        
        if self.results['model_evaluation'].get('performance_comparison'):
            pdf.chapter_body(self.results['model_evaluation'].get('performance_comparison'), markdown=True)
        
        pdf.add_page()
        pdf.chapter_title("7. Conclusion")
        
        conclusion = f"""
        This automated EDA process has analyzed the dataset, performed data cleaning, 
        created informative visualizations, engineered relevant features, recommended 
        appropriate models, and evaluated model performance.
        
        The best model for this {self.results['model_recommendation'].get('problem_type', 'unknown')} problem 
        is {self.results['model_evaluation'].get('best_model', 'unknown')}.
        
        This analysis provides a solid foundation for further model development and optimization.
        """
        pdf.chapter_body(conclusion.strip())
        
        pdf_path = os.path.join(self.output_dir, "eda_report.pdf")
        pdf.output(pdf_path)
        
        return pdf_path