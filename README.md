# NLP Pipeline for Drug Efficacy Extraction from Pubmed Central (PMC)

This repository contains Python scripts to **automate the retrieval, cleaning, and analysis of biomedical articles**. The pipeline helps you convert lists of PubMed IDs into structured data about drugs, their concentrations, and experimental effects on cell lines. Features include (1) Automatically maps PubMed IDs (PMIDs) to PubMed Central IDs (PMCIDs), (2) Downloads and extracts full-text articles in XML or PDF format, (3) Cleans and organizes text into readable files, (4) Searches text for drug mentions using fuzzy matching (5) Extracts structured efficacy data (drug, concentration, effect, cell line) using AI (ChatGPT API). **One of the main challenges is that only a limited number of papers in this category are available through PubMed Central**.   

## Pipeline Overview

**Step 1. PMID to PMCID Mapping**
   - Read a file containing PMIDs.
   - Look up matching PMCIDs via NCBI APIs.
   - Validate mappings by checking metadata.
   - Save the mapping results to a text file.

**Step 2. Text Extraction and Cleaning**
   - For each PMCID:
     - If available, process the XML file to extract the title, abstract, and body.
     - If no XML file, extract text from local PDFs.
     - If no local files exist, download the PDF from Europe PMC.
   - Cleaned text files are saved for further analysis.
   - Load a list of drug synonyms for each PMCID.
   - Search the cleaned text for any mention of these synonyms using fuzzy matching.
   - Save search results to CSV.

**Step 3. Efficacy Data Extraction (ChatGPT API)**
   - Split cleaned text into paragraphs.
   - For each paragraph, prompt the ChatGPT API to extract:
     - Drug name
     - Concentration (e.g., μM, nM)
     - Observed effect (e.g., IC50, reduced cell viability)
     - Cell line (e.g., HeLa, MCF-7)
   - Output structured JSON annotations.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/nlp-drug-efficacy-pipeline.git
cd nlp-drug-efficacy-pipeline
pip install -r requirements.txt
```

```bash
# Step 1 PMID_to_PMCID_mapping
import requests
import time

def read_pmids_from_file(input_file):
    try:
        with open(input_file, 'r') as file:
            data = file.read()
        pmids = [pmid.strip() for pmid in data.split(',') if pmid.strip()]
        return pmids
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' does not exist.")
        return []
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return []

def fetch_pmc_metadata(pmcid):
    """
    Fetch full text XML metadata from Europe PMC for a given PMCID.
    """
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.text
        else:
            print(f"[{pmcid}] Metadata not found or inaccessible (HTTP {response.status_code})")
            return None
    except Exception as e:
        print(f"[{pmcid}] Error fetching metadata: {e}")
        return None

def validate_pmc_matches_pmid(pmcid, pmid):
    """
    Validate that the PMC article metadata contains the correct PMID.
    """
    xml_text = fetch_pmc_metadata(pmcid)
    if xml_text:
        # Check if PMID is present anywhere in the XML metadata
        if pmid in xml_text:
            return True
    return False

def convert_pmid_to_pmcid(pmids):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
    params = {
        "dbfrom": "pubmed",
        "db": "pmc",
        "retmode": "json",
        "linkname": "pubmed_pmc",
    }

    pmcid_map = {}

    for pmid in pmids:
        print(f"Processing PMID: {pmid}...")
        params["id"] = pmid
        valid_pmcids = []

        for attempt in range(3):
            try:
                response = requests.get(base_url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    linksets = data.get("linksets", [])
                    if linksets and "linksetdbs" in linksets[0]:
                        for linksetdb in linksets[0]["linksetdbs"]:
                            if linksetdb.get("linkname") == "pubmed_pmc":
                                pmc_ids = linksetdb.get("links", [])
                                for pmc in pmc_ids:
                                    pmc_full = f"PMC{pmc}"
                                    print(f"Validating PMCID {pmc_full} for PMID {pmid} ...")
                                    if validate_pmc_matches_pmid(pmc_full, pmid):
                                        print(f"Validated: {pmc_full} matches PMID {pmid}")
                                        valid_pmcids.append(pmc_full)
                                    else:
                                        print(f"Rejected: {pmc_full} does not match PMID {pmid}")
                                break
                        pmcid_map[pmid] = valid_pmcids if valid_pmcids else None
                    else:
                        pmcid_map[pmid] = None
                else:
                    print(f"Failed to retrieve data for PMID {pmid} (HTTP {response.status_code})")
                    pmcid_map[pmid] = None
                break
            except requests.exceptions.RequestException as e:
                print(f"Error fetching data for PMID {pmid}: {e}")
                time.sleep(2)
        else:
            pmcid_map[pmid] = None

    return pmcid_map


if __name__ == "__main__":
    input_file = r"C:/Users/Bart Westerman XPS/AppData/Local/Programs/Python/Python36-32/GBM1000PMIDs.txt"
    output_file = r"C:/Users/Bart Westerman XPS/AppData/Local/Programs/Python/Python36-32/pmid_to_pmcid_mapping.txt"

    pmids = read_pmids_from_file(input_file)

    if pmids:
        print(f"Found {len(pmids)} PMIDs in the file.")
        result = convert_pmid_to_pmcid(pmids)

        with open(output_file, 'w') as outfile:
            outfile.write("PMID\tPMCIDs\n")
            for pmid, pmcids in result.items():
                if pmcids:
                    outfile.write(f"{pmid}\t{','.join(pmcids)}\n")
                else:
                    outfile.write(f"{pmid}\tNone\n")

        print(f"PMID to PMCID mapping saved to '{output_file}'.")
    else:
        print("No PMIDs found in the input file.")
```
Found 921 PMIDs in the file.
Processing PMID: 26883759...
Processing PMID: 23243059...
Validating PMCID PMC3570595 for PMID 23243059 ...
[PMC3570595] Metadata not found or inaccessible (HTTP 404)
Rejected: PMC3570595 does not match PMID 23243059
Processing PMID: 23387973
26081429
26220902
18602901...
Validating PMCID PMC4470057 for PMID 23387973
26081429
26220902
18602901 ...
Rejected: PMC4470057 does not match PMID 23387973
26081429
26220902
18602901
Validating PMCID PMC3883957 for PMID 23387973
26081429
26220902
18602901 ...
[PMC3883957] Metadata not found or inaccessible (HTTP 404)
Rejected: PMC3883957 does not match PMID 23387973
26081429
26220902
18602901
Validating PMCID PMC2586291 for PMID 23387973
26081429
22393246 ...

```bash
# Step 2. Text cleaning step, provides a cleaned extracted text file with the pmcid and drug name
#!pip install pandas fuzzywuzzy python-Levenshtein transformers torch PyMuPDF
import os
import re
import csv
import requests
import pandas as pd
from xml.etree import ElementTree as ET
from fuzzywuzzy import fuzz
import fitz  # PyMuPDF for PDF extraction
import requests

# ------------------- File Paths ------------------- #
pmcid_to_drug_file = r"C:/Users/Bart Westerman XPS/AppData/Local/Programs/Python/Python36-32/pmcid_to_drug_mapping.txt"
raw_text_dir = r"C:/Users/Bart Westerman XPS/AppData/Local/Programs/Python/Python36-32/PMC_Raw_Texts/"
output_result_file = r"C:/Users/Bart Westerman XPS/AppData/Local/Programs/Python/Python36-32/drug_synonym_search_results.csv"
output_text_folder = r"C:/Users/Bart Westerman XPS/AppData/Local/Programs/Python/Python36-32/extracted_text/"

# Dummy search function placeholder
def search_for_terms_in_text(text, synonyms, threshold=80):
    matches = []
    for synonym in synonyms:
        for line in text.splitlines():
            if fuzz.partial_ratio(synonym.lower(), line.lower()) >= threshold:
                matches.append(line.strip())
    return matches

# ------------------- PubMed XML Cleaning ------------------- #
def clean_pubmed_xml(file_path, output_folder, pmcid, drug_name):
    print(f"Cleaning XML data from: {file_path}...")
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        text_sections = []
        title = root.find(".//article-title")
        if title is not None and title.text:
            text_sections.append(f"Title: {title.text.strip()}")
        abstract = root.find(".//abstract")
        if abstract is not None:
            abstract_text = " ".join([elem.text.strip() for elem in abstract.findall(".//p") if elem.text])
            text_sections.append(f"Abstract:\n{abstract_text}")
        body = root.find(".//body")
        if body is not None:
            body_text = " ".join([elem.text.strip() for elem in body.findall(".//p") if elem.text])
            text_sections.append(f"Body:\n{body_text}")
        readable_text = "\n\n".join(text_sections)

        safe_drug = re.sub(r'[\\/*?:"<>|]', "_", drug_name.strip().replace(" ", "_"))
        output_file_path = os.path.join(output_folder, f"extracted_text_{pmcid}_{safe_drug}.txt")

        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(readable_text)
        print(f"Extracted text saved to: {output_file_path}")
        return readable_text
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return f"Error processing file: {e}"

# ------------------- PDF Text Extraction ------------------- #
def extract_text_from_pdf(file_path, output_folder, pmcid, drug_name):
    print(f"Extracting text from PDF: {file_path}...")
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()

        safe_drug = re.sub(r'[\\/*?:"<>|]', "_", drug_name.strip().replace(" ", "_"))
        output_file_path = os.path.join(output_folder, f"extracted_text_{pmcid}_{safe_drug}.txt")

        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Extracted text from PDF saved to: {output_file_path}")
        return text
    except Exception as e:
        print(f"Error processing PDF file {file_path}: {e}")
        return f"Error processing file: {e}"

# --- Added fallback: Download PDF from Europe PMC backend ---

def download_pdf_from_europepmc(pmcid):
    url = f"https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC{pmcid}&blobtype=pdf"
    headers = {"User-Agent": "Mozilla/5.0"}
    print(f"[PMC{pmcid}] Attempting to download PDF from Europe PMC backend...")

    try:
        response = requests.get(url, headers=headers, timeout=20)
        if response.status_code == 200 and response.headers.get('Content-Type', '').startswith('application/pdf'):
            # Save to default output folder
            filename = os.path.join(output_text_folder, f"{pmcid}_downloaded.pdf")
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"[PMC{pmcid}] PDF downloaded successfully to {filename}")

            # Also save to backup folder
            pmc_str = f"PMC{pmcid}"
            dest_folder = os.path.join(pdf_backup_root, pmc_str, pmc_str)
            os.makedirs(dest_folder, exist_ok=True)
            dest_path = os.path.join(dest_folder, f"{pmcid}_downloaded.pdf")
            with open(dest_path, "wb") as f:
                f.write(response.content)
            print(f"[PMC{pmcid}] PDF also saved to backup folder: {dest_path}")

            return filename
        else:
            print(f"[PMC{pmcid}] PDF not available or wrong content type. Status: {response.status_code}")
            return None
    except Exception as e:
        print(f"[PMC{pmcid}] Error downloading PDF: {e}")
        return None

# ------------------- Main Script ------------------- #
def main():
    print("Starting the script...")
    result_data = []

    # Ensure output folder exists
    if not os.path.exists(output_text_folder):
        os.makedirs(output_text_folder)

    # Step 1: Load PMCID to Drug Mapping Data
    print("Loading PMCID to Drug Mapping file...")
    try:
        pmcid_to_drug_df = pd.read_csv(pmcid_to_drug_file, sep='\t', encoding='ISO-8859-1')
        print("✅ Successfully loaded PMCID to Drug Mapping file.")
    except Exception as e:
        print(f"❌ Error loading PMCID to Drug Mapping file: {e}")
        return

    # Clean PMCID column
    pmcid_to_drug_df["PMCID"] = pmcid_to_drug_df["PMCID"].astype(str).str.split(".").str[0]
    pmcid_to_drug_df = pmcid_to_drug_df[pmcid_to_drug_df["PMCID"].notna() & (pmcid_to_drug_df["PMCID"] != "0")]
    # Step 2: Process each PMCID
    print("Processing each PMCID...")
    for index, row in pmcid_to_drug_df.iterrows():
        pmcid = row['PMCID']
        drug_name_field = row['Drug Synonyms']
        if isinstance(drug_name_field, str):
            drug_synonyms = [syn.strip() for syn in drug_name_field.split(',')]
        else:
            drug_synonyms = []

        print(f"🔍 Processing PMCID{pmcid} for drug(s): {drug_name_field}")

        folder_path = os.path.join(raw_text_dir, f"PMC{pmcid}")
        if not os.path.isdir(folder_path):
            print(f"⚠️ Raw text folder not found for {pmcid}. Skipping local files extraction.")

        nxml_files = []
        pdf_files = []
        if os.path.isdir(folder_path):
            for root_dir, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.endswith('.nxml'):
                        nxml_files.append(os.path.join(root_dir, file))
                    elif file.endswith('.pdf'):
                        pdf_files.append(os.path.join(root_dir, file))

        extracted_texts = []

        # Try NXML extraction first if available
        if nxml_files:
            for file_path in nxml_files:
                print(f"Processing NXML file: {file_path}")
                cleaned_text = clean_pubmed_xml(file_path, output_text_folder, pmcid, drug_name_field)
                if "Error processing file" not in cleaned_text:
                    extracted_texts.append(cleaned_text)

        # If no NXML files, try PDF extraction from local PDF files
        elif pdf_files:
            for file_path in pdf_files:
                print(f"Processing PDF file: {file_path}")
                extracted_text = extract_text_from_pdf(file_path, output_text_folder, pmcid, drug_name_field)
                if "Error processing file" not in extracted_text:
                    extracted_texts.append(extracted_text)

        # --- Added fallback: If no local files, try downloading PDF from Europe PMC and extract ---
        else:
            print(f"No NXML or PDF files found locally for PMCID {pmcid}. Trying Europe PMC PDF download fallback...")
            pdf_path = download_pdf_from_europepmc(pmcid)
            if pdf_path:
                extracted_text = extract_text_from_pdf(pdf_path, output_text_folder, pmcid, drug_name_field)
                if "Error processing file" not in extracted_text:
                    extracted_texts.append(extracted_text)
                else:
                    print(f"[PMC{pmcid}] Extraction from downloaded PDF failed.")
            else:
                print(f"[PMC{pmcid}] PDF download fallback failed. No text extracted.")

        if extracted_texts:
            combined_text = "\n\n".join(extracted_texts)
        else:
            combined_text = ""

        # Search for drug mentions
        efficacy_results = search_for_terms_in_text(combined_text, drug_synonyms, threshold=80)

        result_data.append({
            'PMCID': pmcid,
            'DrugName': drug_name_field,
            'Drug Synonym': ', '.join(drug_synonyms),
            'File': folder_path if os.path.isdir(folder_path) else "No local folder"
        })

    # Step 3: Save results to CSV
    if result_data:
        result_df = pd.DataFrame(result_data)
        result_df.to_csv(output_result_file, index=False)
        print(f"✅ Search results saved to {output_result_file}.")
    else:
        print("❌ No results found.")

    print("Script execution complete.")

if __name__ == "__main__":
    main()

```
Starting the script...
Loading PMCID to Drug Mapping file...
✅ Successfully loaded PMCID to Drug Mapping file.
Processing each PMCID...
🔍 Processing PMCID11574687 for drug(s): 4506-66-5
Processing NXML file: C:/Users/Bart Westerman XPS/AppData/Local/Programs/Python/Python36-32/PMC_Raw_Texts/PMC11574687\PMC11574687\Cytojournal-21-34.nxml
Cleaning XML data from: C:/Users/Bart Westerman XPS/AppData/Local/Programs/Python/Python36-32/PMC_Raw_Texts/PMC11574687\PMC11574687\Cytojournal-21-34.nxml...
Extracted text saved to: C:/Users/Bart Westerman XPS/AppData/Local/Programs/Python/Python36-32/extracted_text/extracted_text_11574687_4506-66-5.txt
🔍 Processing PMCID3570595 for drug(s): 4506-66-5
⚠️ Raw text folder not found for 3570595. Skipping.
🔍 Processing PMCID8199444 for drug(s): 5-aminolevulinic acid
Processing NXML file: C:/Users/Bart Westerman XPS/AppData/Local/Programs/Python/Python36-32/PMC_Raw_Texts/PMC8199444\PMC8199444\ijms-22-05596.nxml
Cleaning XML data from: C:/Users/Bart Westerman XPS/AppData/Local/Programs/Python/Python36-32/PMC_Raw_Texts/PMC8199444\PMC8199444\ijms-22-05596.nxml...
Extracted text saved to: C:/Users/Bart Westerman XPS/AppData/Local/Programs/Python/Python36-32/extracted_text/extracted_text_8199444_5-aminolevulinic_acid.txt
...

```bash
# Step 3 GPT4.1 mediated extraction with cue-speed control
!pip install tiktoken
import os
import re
import json
import time
import pandas as pd
import openai
import tiktoken  # Install with: pip install tiktoken

# --- OpenAI Setup --- #
openai.api_key = "sk-proj-.." # Use own key, see https://openai.com/api/ uses only $2 USD for 200 PMC documents 
client = openai.OpenAI(api_key=openai.api_key)

# --- Path Setup --- #
folder_path = r"C:\Users\Bart Westerman XPS\AppData\Local\Programs\Python\Python36-32\extracted_text_n20"
output_csv = os.path.join(folder_path, "drug_annotation_results.csv")
drug_csv_path = r"C:\Users\Bart Westerman XPS\AppData\Local\Programs\Python\Python36-32\drug_synonym_search_results.csv"

# --- Load Drug List from CSV --- #
try:
    drug_df = pd.read_csv(drug_csv_path)
    drug_list = set(str(drug).strip().lower() for drug in drug_df.iloc[:, 0].dropna().unique())
except Exception as e:
    print(f"❌ Failed to read drug list from CSV: {e}")
    drug_list = set()

# --- File Filter --- #
file_names = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

# --- Token Encoder & Rate Limit Setup --- #
encoding = tiktoken.encoding_for_model("gpt-4")
tokens_used_this_minute = 0
last_reset_time = time.time()

# --- Annotation Extraction Function --- #
def extract_annotations(text):
    global tokens_used_this_minute, last_reset_time

    prompt = f"""
You are an expert biomedical text annotator. From the following paragraph, extract:

1. **Drug** (e.g., "doxorubicin")
2. **Concentration** (e.g. μM or nM) 
3. **Effects** (e.g., "IC50", "reduced cell viability", "caused mitotic arrest")
4. **Cell line model** (e.g., "HeLa", "MCF-7", etc.)

Return a structured JSON list of objects, where each object includes:
- drug
- concentration
- effect
- cell_line

Example format:
[
  {{
    "drug": "doxorubicin",
    "concentration": "5 μM",
    "effect": "IC50",
    "cell_line": "MCF-7"
  }},
  ...
]

Here is the paragraph:
\"\"\"
{text}
\"\"\"
"""

    # --- Rate Limiting --- #
    current_time = time.time()
    elapsed = current_time - last_reset_time

    if elapsed >= 60:
        tokens_used_this_minute = 0
        last_reset_time = current_time

    token_estimate = len(encoding.encode(prompt))
    if tokens_used_this_minute + token_estimate >= 30000:
        wait_time = 60 - elapsed
        print(f"⏳ Waiting {wait_time:.1f}s to respect token limit...")
        time.sleep(wait_time)
        tokens_used_this_minute = 0
        last_reset_time = time.time()

    tokens_used_this_minute += token_estimate

    # --- API Request --- #
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        reply = response.choices[0].message.content
        annotations = json.loads(reply)
        return annotations
    except json.JSONDecodeError:
        print("⚠️ JSON decoding failed. Raw response:")
        print(reply)
        return None
    except Exception as e:
        print(f"❌ API request failed: {e}")
        return None

# --- Collect and Filter Annotations --- #
rows = []

for file_name in file_names:
    match = re.match(r"extracted_text_(\d+)_([^.]+)\.txt", file_name)
    if not match:
        print(f"❌ Skipping unrecognized file format: {file_name}")
        continue

    pmcid_raw, _ = match.groups()
    pmcid = f"pmcid{pmcid_raw}"

    file_path = os.path.join(folder_path, file_name)
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"\n📄 Processing: {file_name}")
    annotations = extract_annotations(text)

    if annotations:
        for ann in annotations:
            drug_mentioned = ann.get("drug", "").strip().lower()
            if drug_mentioned in drug_list:
                rows.append({
                    "pmcid": pmcid,
                    "drug": ann.get("drug", ""),
                    "concentration": ann.get("concentration", ""),
                    "effect": ann.get("effect", ""),
                    "cell_line": ann.get("cell_line", "")
                })

# --- Save Results --- #
if rows:
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Results saved to: {output_csv}")
else:
    print("❌ No valid annotations matched the drug list.")


```
Looking in indexes: https://pypi.org/simple, https://pypi.ngc.nvidia.com
Requirement already satisfied: tiktoken in c:\anaconda3\envs\python38\lib\site-packages (0.7.0)
Requirement already satisfied: regex>=2022.1.18 in c:\anaconda3\envs\python38\lib\site-packages (from tiktoken) (2024.11.6)
Requirement already satisfied: requests>=2.26.0 in c:\anaconda3\envs\python38\lib\site-packages (from tiktoken) (2.32.3)
Requirement already satisfied: charset-normalizer<4,>=2 in c:\anaconda3\envs\python38\lib\site-packages (from requests>=2.26.0->tiktoken) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in c:\anaconda3\envs\python38\lib\site-packages (from requests>=2.26.0->tiktoken) (3.10)
Requirement already satisfied: urllib3<3,>=1.21.1 in c:\anaconda3\envs\python38\lib\site-packages (from requests>=2.26.0->tiktoken) (1.26.19)
Requirement already satisfied: certifi>=2017.4.17 in c:\anaconda3\envs\python38\lib\site-packages (from requests>=2.26.0->tiktoken) (2024.7.4)

📄 Processing: extracted_text_2375243_Lovastatin.txt

📄 Processing: extracted_text_2375243_Perilla_alcohol.txt

📄 Processing: extracted_text_2588634_Decitabine.txt
⚠️ JSON decoding failed. Raw response:
```json
[
  {
    "drug": "5-Aza-dC",
    "concentration": "2.5 μM",
    "effect": "restored gene expression",
    "cell_line": "SF126"
  },
  {
    "drug": "5-Aza-dC",
    "concentration": "5 μM",
    "effect": "restored gene expression",
    "cell_line": "SF126"
  },
  {
    "drug": "5-Aza-dC",
    "concentration": "10 μM",
    "effect": "restored gene expression",
    "cell_line": "SF126"
  },
  {
    "drug": "5-Aza-dC",
    "concentration": "15 μM",
    "effect": "restored gene expression",
    "cell_line": "SF126"
  },
  {
    "drug": "5-Aza-dC",
    "concentration": "2.5 μM",
    "effect": "restored gene expression",
    "cell_line": "SF767"
  },
  {
    "drug": "5-Aza-dC",
    "concentration": "5 μM",
    "effect": "restored gene expression",
    "cell_line": "SF767"
  },
  {
    "drug": "5-Aza-dC",
    "concentration": "10 μM",
    "effect": "restored gene expression",
    "cell_line": "SF767"
  },
  {
    "drug": "5-Aza-dC",
    "concentration": "15 μM",
    "effect": "restored gene expression",
    "cell_line": "SF767"
  }
]
```
**Notes:**
- The only drug treatment described is with **5-Aza-dC** (5-aza-2'-deoxycytidine).
- Concentrations used: **2.5, 5, 10, and 15 μM**.
- Cell lines: **SF126** and **SF767** (both human glioblastoma-derived).
- Effect: The text states "Treatment of SF126 and SF767 cells with 5-Aza-dC restored..." (presumably gene expression, as is typical for demethylating agents in this context).
- No other drugs, concentrations, or cell lines with drug effects are described in the paragraph.

📄 Processing: extracted_text_2645013_____Thujone.txt
⚠️ JSON decoding failed. Raw response:
```json
[
  {
    "drug": "zVAD-fmk",
    "concentration": "10 µM",
    "effect": "did not inhibit macrophage-mediated cytotoxicity against T9-C2 cells",
    "cell_line": "T9-C2"
  },
  {
    "drug": "phloretin",
    "concentration": "1 mM",
    "effect": "induced vacuolization and swelling within 1 hour; induced mitochondrial swelling and increased Hsp70 and Hsp90 expression after 6 hours; caused HMGB1 translocation; induced cell death via paraptosis",
    "cell_line": "T9"
  },
  {
    "drug": "phloretin",
    "concentration": "1 mM",
    "effect": "induced swelling within 15–20 minutes; induced mitochondrial swelling within 30 minutes; increased Hsp70 and Hsp90 expression after 6 hours",
    "cell_line": "F98"
  },
  {
    "drug": "pimaric acid",
    "concentration": "0.01 mM",
    "effect": "induced vacuolization and swelling within 1 hour; induced mitochondrial swelling and increased Hsp70 and Hsp90 expression after 6 hours; caused HMGB1 translocation; induced cell death via paraptosis",
    "cell_line": "T9"
  },
  {
    "drug": "pimaric acid",
    "concentration": "0.1 mM",
    "effect": "increased Hsp70 and Hsp90 expression after 6 hours",
    "cell_line": "F98"
  },
  {
    "drug": "iberiotoxin",
    "concentration": "0.05 µM",
    "effect": "reduced macrophage-mediated cytotoxicity against T9-C2 cells from 66.8% to 16%",
    "cell_line": "T9-C2"
  },
  {
    "drug": "staurosporine",
    "concentration": "10 µM",
    "effect": "induced apoptosis in T9 cells after 18 hour exposure (used as apoptotic control)",
    "cell_line": "T9"
  },
  {
    "drug": "CO (carbon monoxide, as saturated media)",
    "concentration": "not specified",
    "effect": "induced swelling of T9 cells within 15–60 minutes",
    "cell_line": "T9"
  },
  {
    "drug": "phloretin",
    "concentration": "1 mM",
    "effect": "used to kill T9 cells for dendritic cell maturation experiments",
    "cell_line": "T9"
  }
]
```
**Notes:**
- Only direct drug/concentration/effect/cell line relationships are included.
- Some effects are summarized for clarity and grouped if described together in the text.
- "CO" is included as a drug because it is used as a BK channel activator in the context.
- Cell lines: T9, T9-C2 (mM-CSF expressing T9), F98 (rat glioma), and dendritic cell maturation experiments are included if T9 cells were treated with drugs.
- If a concentration is not specified, "not specified" is used.
- Only the most relevant and explicit relationships are included; indirect or ambiguous mentions are omitted.

📄 Processing: extracted_text_2717978_Baicalein.txt
⚠️ JSON decoding failed. Raw response:
```json
[
  {
    "drug": "resveratrol",
    "concentration": "150 μM",
    "effect": "decreased cell viability; increased cell population at sub-G1 phase (apoptosis); collapse of mitochondrial membrane potential; induced autophagy",
    "cell_line": "U251"
  },
  {
    "drug": "resveratrol",
    "concentration": "0 to 300 μM",
    "effect": "decreased cell viability in a dose- and time-dependent manner",
    "cell_line": "U251"
  },
  {
    "drug": "Z-VAD-fmk",
    "concentration": "not specified",
    "effect": "suppressed resveratrol-induced U251 cell death",
    "cell_line": "U251"
  },
  {
    "drug": "3-methyladenine (3-MA)",
    "concentration": "not specified",
    "effect": "sensitized the cytotoxicity of resveratrol",
    "cell_line": "U251"
  },
  {
    "drug": "bafilomycin A1",
    "concentration": "not specified",
    "effect": "sensitized the cytotoxicity of resveratrol",
    "cell_line": "U251"
  }
]
```

📄 Processing: extracted_text_2747688_Vandetanib.txt
⚠️ JSON decoding failed. Raw response:
```json
[
  {
    "drug": "ZD6474",
    "concentration": "50 mg/kg",
    "effect": "reduced proliferation index from 0.22 (range 0.15–0.28) in the control group to 0.14 (range 0.04–0.2)",
    "cell_line": "BT4C"
