# Synthea Data Analysis Playground

This repository contains a self-contained script to **generate** and **analyze** synthetic medical data using [Synthea](https://github.com/synthetichealth/synthea). The script orchestrates the entire workflow, from data generation to statistical analysis.

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd playground
    ```

2.  **Add Synthea Executable:**
    *   Download the `synthea-with-dependencies.jar` file from the [Synthea GitHub Releases](https://github.com/synthetichealth/synthea/releases).
    *   Place the `.jar` file inside the `synthea/` directory.

3.  **Create a Python Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

Once the setup is complete, you can run the entire generation and analysis pipeline with a single command from the root `playground/` directory:

```bash
python run_and_analyze.py
```

The script will:
1.  Invoke Synthea to generate a new dataset for 15 patients (this can be changed in the script).
2.  Wait for the generation to complete.
3.  Load the newly created CSV files from `synthea/output/csv/`.
4.  Perform a full analysis and print a summary report to the console.

### Customization
*   **To change the number of patients:** Modify the `POPULATION_SIZE` variable at the top of the `run_and_analyze.py` script.
*   **To change generation parameters:** Edit the `synthea/synthea.properties` file. For example, you can change the default city or the modules to run.