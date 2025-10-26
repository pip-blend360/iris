Prompt and work context

User prompt:
You are a principal data scientist applying for a job. You are asked to analyze the famous Iris dataset. That is all the instructions you are given. 

I think you can find this dataset within Sklearn package. Please download this dataset, analyze the data in a new and innovative way using python and present your results in an easy to understand format. 

Please create a very simple directory structure for this project ready to be pushed to a github repository (where we will share the code and results). 

Please format the results in on a webpage starting with the objectives of the analysis, a quick exploratory data analysis, results, and accuracy sections. Please use a test datset for accuracy metrics.
Please format all graphs using the XKCD style.
In your output please include a 3D interactive plot using plotly.graph_objects showing the training and test data color coded for predicted and actual species.
Finally, please create a context.md file that includes this prompt and your work to date, we should be able to use this file as context for future versions of this analysis.

Work to date:
- Created project skeleton in c:\\Users\\PipCourbois\\DS_projects\\iris
- Added requirements.txt
- Added README.md
- Implemented src/run_analysis.py with data loading, UMAP, RandomForest model, pairwise plot, 3D plot saved as HTML, and report generation via Jinja2 template.
- Added src/report_template.html used to generate outputs/report.html

Next steps for user when running locally:
- Create virtual env and install requirements
- Run python src/run_analysis.py

