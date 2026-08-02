# EPL Foul Win Probability Model Presentation

## Slide 1: Title
- Title: EPL Foul Win Probability Model
- Subtitle: Sports Analytics, Predictive Modeling, and Deployment
- Presenter: Senior Machine Learning Engineer
- Event: Technical interview / stakeholder review

Speaker notes:
Introduce the project and explain the goal of predicting foul-win probability in EPL matches.
Suggested figure: Project title slide with football field and analytics icon.
Estimated time: 1 minute

## Slide 2: Agenda
- Problem statement
- Dataset overview
- Target engineering
- Feature engineering
- Model pipeline
- Evaluation
- Streamlit demo
- Software engineering
- Lessons and next steps

Speaker notes:
Review the presentation structure and what the audience will learn.
Suggested figure: Timeline or agenda list.
Estimated time: 1 minute

## Slide 3: Business Problem
- Teams need insight into foul-winning events
- Fouls influence set-piece opportunity and game control
- Predictive modeling supports tactical planning
- Objective: identify likelihood of foul-win after possession acquisition

Speaker notes:
Explain why foul prediction matters for performance analysis and match strategy.
Suggested figure: football pitch with foul zones highlighted.
Estimated time: 1 minute

## Slide 4: Dataset Overview
- StatsBomb Open Data for EPL
- 380 matches
- 1.31M+ event records
- 381,267 candidate possession acquisition events
- Rich metadata for event location and context

Speaker notes:
Describe the data source and why StatsBomb is a strong choice for event-level modeling.
Suggested figure: dataset summary table.
Estimated time: 1 minute

## Slide 5: Dataset Statistics
- Match count: 380
- Total events: 1.31M+
- Candidate events: 381,267
- Class imbalance: foul-win minority class
- Key columns: x, y, event type, possession, minute

Speaker notes:
Highlight important statistics that define the modeling challenge and data scale.
Suggested figure: statistics table or bar chart.
Estimated time: 1 minute

## Slide 6: Exploratory Data Analysis
- Event type frequency
- Possession distribution
- Foul event locations
- Temporal patterns
- Coordinate density across the pitch

Speaker notes:
Explain how EDA validates assumptions and informs feature design.
Suggested figure: summary of EDA charts.
Estimated time: 1 minute

## Slide 7: Target Engineering
- Candidate events: reception and recovery
- Possession grouping by possession_id
- Search for later foul events in same possession
- Binary target: 1 if foul occurs, 0 otherwise

Speaker notes:
Walk through the target creation logic and why possession context is critical.
Suggested figure: flowchart of candidate event to target assignment.
Estimated time: 1 minute

## Slide 8: Feature Engineering
- Spatial features: x, y, distance to goal, goal angle
- Context features: pitch zone, match time, event encoding
- Automatic derived features in pipeline
- Importance of location and timing

Speaker notes:
Describe each feature and connect it to football intuition.
Suggested figure: pitch with feature overlays.
Estimated time: 1 minute

## Slide 9: Feature Engineering Visualizations
- Distance to goal distribution
- Goal angle distribution
- Pitch zone frequencies
- Event-type encoded signal

Speaker notes:
Show how features are distributed and why they help separate positive events.
Suggested figure: feature histograms or pitch maps.
Estimated time: 1 minute

## Slide 10: Machine Learning Pipeline
- Raw data ingestion
- Candidate event extraction
- Target engineering
- Feature engineering
- Train/test split
- Random Forest training
- Evaluation and dashboard

Speaker notes:
Present the end-to-end pipeline and emphasize reproducibility.
Suggested figure: pipeline diagram.
Estimated time: 1 minute

## Slide 11: Random Forest
- Chosen for robustness and interpretability
- Handles mixed numeric/categorical features
- Works well with moderate-sized event data
- Provides feature importance metrics

Speaker notes:
Summarize why Random Forest is appropriate for this problem.
Suggested figure: decision tree ensemble icon.
Estimated time: 1 minute

## Slide 12: Model Evaluation
- Confusion matrix analysis
- Accuracy, precision, recall, F1
- ROC-AUC and PR-AUC
- Balanced performance for imbalanced data

Speaker notes:
Review the evaluation metrics and their meaning for stakeholders.
Suggested figure: evaluation metric summary.
Estimated time: 1 minute

## Slide 13: Confusion Matrix
- True positives: predicted foul-win correctly
- False positives: predicted foul-win incorrectly
- False negatives: missed foul-win events
- True negatives: correctly predicted no foul-win

Speaker notes:
Explain how the confusion matrix reflects business risk and model behavior.
Suggested figure: confusion matrix diagram.
Estimated time: 1 minute

## Slide 14: ROC Curve
- Measures ranking quality
- Summary of model discrimination power
- AUC value indicates separability

Speaker notes:
Discuss why ROC-AUC is useful for threshold-agnostic evaluation.
Suggested figure: ROC curve plot.
Estimated time: 1 minute

## Slide 15: Precision Recall Curve
- Focuses on minority class performance
- Useful for imbalanced datasets
- Helps choose operating threshold

Speaker notes:
Explain the emphasis on precision/recall for positive foul-win events.
Suggested figure: precision-recall curve.
Estimated time: 1 minute

## Slide 16: Feature Importance
- Key predictors: distance to goal, pitch zone, event encoding
- Spatial features drive the model
- Feature importance supports model explainability

Speaker notes:
Highlight top features and why they align with domain intuition.
Suggested figure: feature importance bar chart.
Estimated time: 1 minute

## Slide 17: Streamlit Application Demo
- Interactive prediction interface
- Automatic feature calculation
- Probability score and classification
- Evaluation dashboard with charts

Speaker notes:
Describe the app experience and how stakeholders can use it.
Suggested figure: Streamlit dashboard screenshot.
Estimated time: 1 minute

## Slide 18: Software Engineering
- Modular code structure
- Unit testing with PyTest
- CI integration with GitHub Actions
- Documentation and reproducibility

Speaker notes:
Explain the engineering practices that support production readiness.
Suggested figure: folder structure or code architecture diagram.
Estimated time: 1 minute

## Slide 19: Lessons Learned
- Effective feature engineering is domain-driven
- Possession-based target creation is essential
- Imbalanced classification demands careful metrics
- Streamlit accelerates stakeholder validation

Speaker notes:
Share the key takeaways from the project development process.
Suggested figure: checklist or lessons graphic.
Estimated time: 1 minute

## Slide 20: Future Improvements
- Evaluate XGBoost and LightGBM
- Add SHAP interpretability
- Containerize with Docker
- Deploy a REST API
- Support real-time prediction feeds

Speaker notes:
Outline the roadmap for next-phase development.
Suggested figure: roadmap timeline.
Estimated time: 1 minute

## Slide 21: Questions
- Open floor for questions
- Invite deeper discussion on metrics, deployment, or data
- Provide contact information

Speaker notes:
Thank the audience and invite questions.
Suggested figure: question mark or closing slide.
Estimated time: 1 minute
