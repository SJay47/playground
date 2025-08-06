import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import subprocess
import sys
import json
import math
import re
import random
import argparse
import configparser
import api_uploader
from api_uploader import NpEncoder

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
SYNTHEA_DIR = BASE_DIR / "synthea"
SYNTHEA_JAR = SYNTHEA_DIR / "synthea-with-dependencies.jar"
SYNTHEA_PROPS = SYNTHEA_DIR / "synthea.properties"
CSV_OUTPUT_DIR = SYNTHEA_DIR / "output" / "csv"
TEMPLATE_PATH = BASE_DIR / "master_fingerprint_template.json"
OUTPUT_DIR = BASE_DIR / "generated_fingerprints"
DOMAINS_CONFIG_PATH = BASE_DIR /SYNTHEA_DIR /"synthea_domains.properties"
VARIANTS_CONFIG_PATH = BASE_DIR / "field_variants.json"
API_CONFIG_PATH = BASE_DIR / "api_config.json"

TEXT_COLUMNS = ['Id', 'SSN', 'DRIVERS', 'PASSPORT', 'FIRST', 'LAST', 'MIDDLE', 'MAIDEN', 'PREFIX', 'SUFFIX', 'GENDER', 'RACE', 'ETHNICITY', 'MARITAL', 'ENCOUNTERCLASS']

def load_field_variants(path: Path) -> dict:
    if not path.exists():
        print(f"INFO: Field variants file not found at '{path}'. Using default names.")
        return {}
    with open(path, 'r') as f:
        return json.load(f)

def load_medical_domains_from_config(path: Path) -> list:
    if not path.exists(): print(f"ERROR: Domain configuration file not found at '{path}'"); sys.exit(1)
    config = configparser.ConfigParser(); config.read(path)
    domains = []
    for section in config.sections():
        try: domains.append({k: config.get(section, k) for k in config.options(section)})
        except configparser.NoOptionError as e: print(f"ERROR: Missing option in config section '{section}': {e}"); sys.exit(1)
    return domains

def get_record_set_by_id(fingerprint_dict, rs_id):
    for record_set in fingerprint_dict.get("recordSet", []):
        if record_set.get("@id") == rs_id: return record_set
    raise ValueError(f"RecordSet with @id '{rs_id}' not found in the template.")

def calculate_entropy(probabilities):
    return sum([-p * math.log2(p) for p in probabilities if p > 0])

def sanitize_for_id(name: str) -> str:
    name = re.sub(r'\[|\]|\(|\)', '', name); name = re.sub(r'\s+|-', '_', name)
    name = re.sub(r'[^a-zA-Z0-9_]', '', name); return name.lower()

def create_jsd_block(series: pd.Series) -> dict:
    counts = series.value_counts(); total = counts.sum()
    if total == 0: return {}
    probs = counts / total; top_k = probs.head(10)
    return {"@type": "jsd:TextDistribution", "jsd:total_records_analyzed": int(total), "jsd:vocabulary_size": len(counts), "jsd:top_k_tokens": [{"jsd:token": str(t), "jsd:frequency": round(float(f), 6)} for t, f in top_k.items()], "jsd:token_probability_vector": [round(float(p), 6) for p in top_k]}

def mock_image_stats(domain: dict) -> dict:
    min_w, max_w = sorted([random.randint(256, 1024), random.randint(1024, 4096)])
    min_h, max_h = sorted([random.randint(256, 1024), random.randint(1024, 4096)])
    return {"@type": "ex:ImageStatistics", "ex:numImages": random.randint(500, 10000), "ex:imageDimensions": {"ex:minWidth": min_w, "ex:maxWidth": max_w, "ex:minHeight": min_h, "ex:maxHeight": max_h}, "ex:colorMode": random.choice(["Grayscale", "RGB"]), "ex:modality": domain.get("modality", "N/A")}

def mock_annotation_stats() -> dict:
    classes = sorted({*random.sample(["nodule", "fracture", "tumor", "lesion", "device", "cyst", "mass"], k=random.randint(2, 5))})
    return {
        "@type": "ex:AnnotationStatistics", "ex:numAnnotations": random.randint(1000, 50000),
        "ex:numClasses": len(classes), "ex:classes": classes,
        "ex:objectsPerImage": {"ex:avg": round(random.uniform(1.0, 8.0), 2), "ex:median": random.randint(1, 7)},
        "ex:boundingBoxStats": {"ex:avgRelativeWidth": round(random.uniform(0.1, 0.5), 2), "ex:avgRelativeHeight": round(random.uniform(0.1, 0.5), 2)},
    }

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return None if np.isnan(obj) or np.isinf(obj) else float(obj)
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)): return None
        if isinstance(obj, pd.Timestamp): return obj.isoformat()
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

def run_synthea(population_size: int, module: str):
    print(f"--- Starting Synthea Data Generation for module: {module} ---")
    if not SYNTHEA_JAR.exists(): print(f"ERROR: Synthea JAR not found at '{SYNTHEA_JAR}'"); sys.exit(1)
    if (CSV_OUTPUT_DIR.parent).exists(): import shutil; shutil.rmtree(CSV_OUTPUT_DIR.parent)
    command = ["java", "-jar", str(SYNTHEA_JAR), "-p", str(population_size), "-m", module, "-c", str(SYNTHEA_PROPS)]
    print(f"Running command: {' '.join(command)}")
    try:
        subprocess.run(command, cwd=SYNTHEA_DIR, check=True, capture_output=True, text=True, timeout=300)
        print("--- Synthea Generation Successful ---"); return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"ERROR: Synthea failed. STDERR:\n{e.stderr}"); return False

def load_and_analyze_data(csv_path: Path):
    print(f"\n--- Loading and Analyzing Data from {csv_path} ---")
    try:
        dfs = {"patients": pd.read_csv(csv_path / "patients.csv", parse_dates=['BIRTHDATE', 'DEATHDATE']), "observations": pd.read_csv(csv_path / "observations.csv", low_memory=False), "conditions": pd.read_csv(csv_path / "conditions.csv")}
        globals()['dfs'] = dfs
    except FileNotFoundError as e:
        print(f"ERROR: Could not load CSV file. {e}."); return None
    
    analysis = {"observations": {}, "patients": {}, "conditions": {}}
    patients_df = dfs['patients']
    patients_df['AGE'] = (datetime.now() - patients_df['BIRTHDATE']).dt.days / 365.25
    for col in patients_df.columns:
        series = patients_df[col]; data = series.dropna(); total_records = len(patients_df)
        stats = {'ex:completeness': {"@type": "ex:Completeness", "ex:totalRecordCount": total_records, "ex:nullCount": int(series.isnull().sum()), "ex:completenessRatio": len(data) / total_records if total_records > 0 else 0}, 'dataType': 'sc:Text'}
        if pd.api.types.is_numeric_dtype(series) and not any(tc.lower() in col.lower() for tc in TEXT_COLUMNS):
            stats['dataType'] = 'sc:Float'
            if len(data) >= 1:
                stat_block = {'mean': data.mean(), 'min': data.min(), 'max': data.max(), 'median': data.median()}
                if len(data) > 1: stat_block['stdDev'] = data.std(); counts, bin_edges = np.histogram(data, bins=10); stat_block['histogram'] = {'counts': counts.tolist(), 'bins': bin_edges.tolist()}
                else: stat_block['stdDev'] = 0.0
                if len(data) > 2: stat_block['skewness'] = data.skew()
                else: stat_block['skewness'] = 0.0
                if len(data) > 3: stat_block['kurtosis'] = data.kurt()
                else: stat_block['kurtosis'] = 0.0
                stats['stat:statistics'] = stat_block
        else:
            if pd.api.types.is_datetime64_any_dtype(series): stats['dataType'] = "sc:Date"
            if not data.empty:
                value_counts = data.value_counts()
                probabilities = value_counts / len(data)
                stat_block = {'unique_count': data.nunique(), 'mode': data.mode().iloc[0], 'mode_frequency': int(value_counts.iloc[0]), 'category_frequencies': {str(k): int(v) for k, v in value_counts.head(10).to_dict().items()}, 'entropy': calculate_entropy(probabilities)}
                stats['stat:statistics'] = stat_block
                stats['jsd:textDistribution'] = create_jsd_block(data.astype(str))
        analysis['patients'][col] = stats

    obs_df = dfs['observations']; obs_df['VALUE'] = pd.to_numeric(obs_df['VALUE'], errors='coerce')
    for desc, group in obs_df.groupby('DESCRIPTION'):
        data = group['VALUE'].dropna(); total_records = len(group)
        stats = {'ex:completeness': {"@type": "ex:Completeness", "ex:totalRecordCount": total_records, "ex:nullCount": total_records - len(data), "ex:completenessRatio": len(data) / total_records if total_records > 0 else 0},'dataType': 'sc:Float'}
        if len(data) >= 1:
            stat_block = {'mean': data.mean(), 'median': data.median(), 'min': data.min(), 'max': data.max()}
            if len(data) > 1: stat_block['stdDev'] = data.std(); counts, bin_edges = np.histogram(data, bins=10); stat_block['histogram'] = {'counts': counts.tolist(), 'bins': bin_edges.tolist()}
            else: stat_block['stdDev'] = 0.0
            if len(data) > 2: stat_block['skewness'] = data.skew()
            else: stat_block['skewness'] = 0.0
            if len(data) > 3: stat_block['kurtosis'] = data.kurt()
            else: stat_block['kurtosis'] = 0.0
            stats['stat:statistics'] = stat_block
        analysis['observations'][desc] = stats

    cond_df = dfs['conditions']; total_records_cond = len(cond_df)
    series_cond = cond_df['DESCRIPTION'].dropna()
    analysis['conditions']['DESCRIPTION'] = {'ex:completeness': {"@type": "ex:Completeness", "ex:totalRecordCount": total_records_cond, "ex:nullCount": int(cond_df['DESCRIPTION'].isnull().sum()), "ex:completenessRatio": len(series_cond) / total_records_cond if total_records_cond > 0 else 0}, 'dataType': 'sc:Text'}
    if not series_cond.empty:
        value_counts_cond = series_cond.value_counts()
        probabilities_cond = value_counts_cond / len(series_cond)
        analysis['conditions']['DESCRIPTION']['stat:statistics'] = {'unique_count': series_cond.nunique(), 'mode': series_cond.mode().iloc[0], 'mode_frequency': int(value_counts_cond.iloc[0]), 'category_frequencies': {k: int(v) for k,v in value_counts_cond.head(20).to_dict().items()}, 'entropy': calculate_entropy(probabilities_cond)}
        analysis['conditions']['DESCRIPTION']['jsd:textDistribution'] = create_jsd_block(series_cond)
        cond_counts = series_cond.value_counts(); top_two = cond_counts.nlargest(2)
        if len(top_two) > 1: pos, neg = int(top_two.iloc[0]), int(top_two.iloc[1])
        elif len(top_two) == 1: pos, neg = int(top_two.iloc[0]), 0
        else: pos, neg = 0, 0
    else: pos, neg = 0, 0
    total_count = pos + neg
    analysis['dataset_stats'] = {'positive': pos, 'negative': neg, 'labelEntropy': calculate_entropy([pos/total_count, neg/total_count]) if total_count > 0 else 0}
    print("--- Analysis Complete ---"); return analysis

def assemble_fingerprint(template: dict, analysis: dict, domain: dict, variants: dict) -> dict:
    print(f"--- Assembling Fingerprint for {domain['name']} ---")
    fp = template.copy(); today = datetime.now()
    fp['name'] = f"synthea-generated-{domain['name'].lower()}-{today.strftime('%Y%m%d')}"
    fp['description'] = domain['description']; fp['datePublished'] = today.strftime('%Y-%m-%d')
    fp['version'] = f"1.0.0-synthea-{domain['name']}-{today.strftime('%Y%m%d')}"
    
    ds_stats_fp = fp['ex:datasetStats']
    ds_stats_fp['ex:labelDistribution']['ex:positive'] = analysis['dataset_stats']['positive']
    ds_stats_fp['ex:labelDistribution']['ex:negative'] = analysis['dataset_stats']['negative']
    ds_stats_fp['ex:labelEntropy'] = analysis['dataset_stats']['labelEntropy']
    ds_stats_fp['ex:labelSkewAlpha'] = round(random.uniform(0.1, 1.5), 4)
    ds_stats_fp['ex:featureStatsVector'] = [round(random.uniform(0,100), 2) for _ in range(6)]
    ds_stats_fp['ex:modelSignature'] = f"sha256:{pd.util.hash_pandas_object(globals()['dfs']['patients'], index=False).sum()}"
    
    for rs_id, rs_analysis in [("patients", analysis['patients']), ("observations", analysis['observations']), ("conditions", analysis['conditions'])]:
        rs_fp = get_record_set_by_id(fp, rs_id); rs_fp['field'] = []
        for original_name, stats in rs_analysis.items():
            field_name_variants = variants.get(original_name, [original_name])
            chosen_name = random.choice(field_name_variants)
            
            field = {"@id": f"{rs_id}/{sanitize_for_id(original_name)}", "@type": "cr:Field", "name": chosen_name, "description": f"Data field for '{original_name}' from the {rs_id} table.", "dataType": stats['dataType']}
            for key in ["stat:statistics", "ex:completeness", "jsd:textDistribution"]:
                if key in stats and stats.get(key): field[key] = stats[key]
            rs_fp['field'].append(field)
            
    rs_images = get_record_set_by_id(fp, "medical_images")
    rs_images["ex:imageStats"] = mock_image_stats(domain)
    rs_images["ex:annotationStats"] = mock_annotation_stats()
    
    print("--- Assembly Finished ---"); return fp

def main():
    parser = argparse.ArgumentParser(description="Generate and optionally upload realistic, domain-specific Croissant fingerprints using Synthea.")
    parser.add_argument("-c", "--count", type=int, default=1, help="Total number of fingerprints to generate.")
    parser.add_argument("-p", "--population", type=int, default=150, help="Population size for each Synthea run.")
    parser.add_argument("--send", action="store_true", help="Send generated fingerprints to the API after creating orgs and datasets.")
    parser.add_argument("--orgs", type=int, default=2, help="Number of new organizations to create if --send is used.")
    parser.add_argument("--datasets-per-org", type=int, default=3, help="Number of datasets to create per organization if --send is used.")
    args = parser.parse_args()
    
    medical_domains = load_medical_domains_from_config(DOMAINS_CONFIG_PATH)
    field_variants = load_field_variants(VARIANTS_CONFIG_PATH)

    if not medical_domains: print("No medical domains found in config. Exiting."); sys.exit(1)

    try:
        with open(TEMPLATE_PATH, 'r') as f: master_template = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"ERROR: Could not load master template. {e}"); sys.exit(1)
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    if args.send:
        api_config = api_uploader.load_config(API_CONFIG_PATH)
        if not api_config: sys.exit(1)
        
        token = api_uploader.get_access_token(api_config)
        if not token: sys.exit(1)

        print("\n--- STAGE 1: Creating Organizations and Datasets via API ---")
        org_dataset_map = []
        for i in range(args.orgs):
            org_name = f"Synthea-Gen-Org-{i+1}-{random.randint(100, 999)}"
            print(f"  • ({i+1}/{args.orgs}) Creating Organization: {org_name}")
            org_id = api_uploader.create_organization_via_api(api_config, token, org_name)
            if org_id:
                for j in range(args.datasets_per_org):
                    domain_for_desc = random.choice(medical_domains)
                    dataset_name = f"{domain_for_desc['name']} Dataset {j+1}"
                    dataset_desc = f"A mocked dataset containing synthetic {domain_for_desc['name']} data."
                    print(f"    • ({j+1}/{args.datasets_per_org}) Creating Dataset: {dataset_name}")
                    dataset_id = api_uploader.create_dataset_via_api(api_config, token, org_id, dataset_name, dataset_desc)
                    if dataset_id:
                        org_dataset_map.append({"org_id": org_id, "dataset_id": dataset_id, "org_name": org_name, "dataset_name": dataset_name})
        
        if not org_dataset_map:
            print("\n✗ No organizations or datasets were created. Aborting fingerprint generation.")
            sys.exit(1)

        print(f"\n--- STAGE 2: Generating and Posting {args.count} Fingerprints ---")
        for i in range(args.count):
            print(f"\n--- Generating & Posting Fingerprint {i+1} of {args.count} ---")
            domain = random.choice(medical_domains)
            
            if not run_synthea(args.population, domain['synthea_module']):
                print(f"!!! Skipping fingerprint {i+1} due to Synthea error. !!!"); continue
            
            analysis_results = load_and_analyze_data(CSV_OUTPUT_DIR)
            if not analysis_results:
                print(f"!!! Skipping fingerprint {i+1} due to analysis error. !!!"); continue
            
            informed_fingerprint = assemble_fingerprint(master_template, analysis_results, domain, field_variants)
            final_output = {"data": {"type": 0, "version": "1.2", "candidateSearchVisibility": 0, "isAnonymous": True, "rawFingerprintJson": informed_fingerprint}, "requestId": f"synthea-gen-{domain['name']}-{datetime.now().isoformat()}"}
            
            output_filename = OUTPUT_DIR / f"api_sent_{domain['name']}_{i+1}.json"
            with open(output_filename, 'w') as f: json.dump(final_output, f, indent=2, cls=NpEncoder)
            print(f"    ✓ Saved local copy: {output_filename}")

            target = random.choice(org_dataset_map)
            print(f"    → Posting to Org '{target['org_name']}' -> Dataset '{target['dataset_name']}'...")
            api_uploader.post_fingerprint_via_api(api_config, token, target['org_id'], target['dataset_id'], final_output)

    else:
        # Local generation workflow
        for i in range(args.count):
            print(f"\n--- Generating Fingerprint {i+1} of {args.count} (Local Only) ---")
            domain = random.choice(medical_domains)
            if not run_synthea(args.population, domain['synthea_module']):
                print(f"!!! Skipping fingerprint {i+1} due to Synthea error. !!!"); continue
            analysis_results = load_and_analyze_data(CSV_OUTPUT_DIR)
            if not analysis_results:
                print(f"!!! Skipping fingerprint {i+1} due to analysis error. !!!"); continue
            informed_fingerprint = assemble_fingerprint(master_template, analysis_results, domain, field_variants)
            final_output = {"data": {"type": 0, "version": "1.2", "candidateSearchVisibility": 0, "isAnonymous": True, "rawFingerprintJson": informed_fingerprint}, "requestId": f"synthea-gen-{domain['name']}-{datetime.now().isoformat()}"}
            output_filename = OUTPUT_DIR / f"local_synthea_{domain['name']}_{i+1}.json"
            with open(output_filename, 'w') as f: json.dump(final_output, f, indent=2, cls=NpEncoder)
            print(f"✅ SUCCESS: Fingerprint {i+1} created at: {output_filename}")

    print(f"\n--- All operations completed. ---")

if __name__ == "__main__":
    main()