import streamlit as st 
import torch
import re
import pandas as pd
import numpy as np
import json
import gc
import ipywidgets as widgets
from pypdf import PdfReader
import io
import time

from pydantic import BaseModel, Field
from typing import Optional, List

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda


from IPython.display import display, Markdown
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. BACKEND: SCHEMAS & HELPERS
# ==========================================
class TestScores(BaseModel):
    """Schema for standardized test scores"""
    ielts: Optional[float] = Field(None, description="IELTS band score (0-9)")
    toefl: Optional[int] = Field(None, description="TOEFL score (0-120)")
    duolingo: Optional[int] = Field(None, description="Duolingo score (10-160)")
    gre_verbal: Optional[int] = Field(None, description="GRE Verbal Reasoning score")
    gre_quant: Optional[int] = Field(None, description="GRE Quantitative Reasoning score")
    gre_awa: Optional[float] = Field(None, description="GRE Analytical Writing score")


class UserProfile(BaseModel):
    """The final cleaned profile of the scholarship applicant"""
    name: Optional[str] = Field(None, description="Applicant's Name")
    email: Optional[str] = Field(None, description="Applicant's email")
    university: Optional[str] = Field(None, description="University name")
    degree_level: Optional[str] = Field(None, description="Degree level (e.g., BSc, MSc)")
    domain: Optional[str] = Field(None, description="Field of study (e.g., Computer Science, Electrical Engineering)")
    experience_years: Optional[int] = Field(None, description="Number of years of relevant experience")
    gpa: Optional[float] = Field(None, description="Grade Point Average")
    gpa_scale: Optional[float] = Field(4.0, description="GPA scale, e.g., 4.0 or 5.0")
    test_scores: TestScores = Field(default_factory=TestScores)
    project_titles: List[str] = Field(default_factory=list, description="List of project titles")
    published_research_titles: List[str] = Field(default_factory=list, description="List of published research paper titles")
    competition_wins: List[str] = Field(default_factory=list, description="List of competitions won")
    volunteering_activities: List[str] = Field(default_factory=list, description="List of volunteering activities")
    willing_to_return: Optional[bool] = Field(None, description="Indicates whether the applicant is willing to return to their home country after graduation (Yes/No)")
    graduation_certificate: Optional[bool] = Field(None, description="Indicates whether the applicant has a graduation certificate (Yes/No)")

# ==========================================
# 2. HELPERS
# ==========================================

def extract_json(text: str) -> str:
    """Finds the first valid JSON object ({...}) in a string."""
    # Remove markdown code blocks if the LLM added them
    text = text.replace("```json", "").replace("```", "")

    # Find the first { and the last }
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        raise ValueError("No JSON object found in the LLM output.")


def extract_text_from_pdf(file_bytes) -> str:
    """Reads PDF bytes and returns cleaned text."""
    try:
        pdf_file = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_file)
        raw_text = ""
        for page in reader.pages:
            raw_text += page.extract_text() + "\n"

        clean_text = re.sub(r'\s+', ' ', raw_text).strip()
        return clean_text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""


# ==========================================
# 3. LOAD MODEL & CHAIN (Notebook Style)
# ==========================================

@st.cache_resource
def load_model_and_chain():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-9b-it",
        quantization_config=bnb_config,
        device_map="auto"
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1500,
        temperature=0.5,
        do_sample=True,
        repetition_penalty=1.15,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    llm = HuggingFacePipeline(pipeline=pipe)  # Using direct pipeline for simplicity

    parser = PydanticOutputParser(pydantic_object=UserProfile)

    SYSTEM_PROMPT = ChatPromptTemplate.from_messages([
    ("human", """You are an expert Admissions Profiler AI.

    Your job is to extract structured information from a resume or personal statement and convert it into valid JSON.
    
    ========================
    EXTRACTION RULES
    ========================
    1. ONLY extract information explicitly stated in the text.
       - Do NOT infer, guess, or hallucinate anything.
    
    2. If a field is missing:
       - use null for strings/numbers
       - use [] for lists
    
    3. GPA:
       - extract GPA value AND scale if mentioned (default = 4.0 if not mentioned)
    
    4. STRICT EXTRACTION:
       - Keep EXACT wording for:
         • projects
         • research papers
         • competitions
         • volunteering activities
    
    5. NORMALIZATION (IMPORTANT):
       - If experience is mentioned (e.g. "2 years experience"):
         → extract as number of years (int or float if possible)
    
       - If graduation certificate is mentioned:
         → set graduation_certificate = yes
    
       - If user says they want to stay in country of scholarship:
         → extract as boolean field: willing_to_return = no
    
       - If a "field/domain/major area" is implied:
     → extract into "domain" as it entered(e.g. Computer Science & Information , AI, etc.)
     
    6. PERSONAL INFO EXTRACTION:
       - Extract "name" if explicitly mentioned in the text
       - Extract "email" if present (must be valid email format)
       - If not present → set to null
    
    ========================
    OUTPUT FORMAT RULES
    ========================
    {format_instructions}
    
    - Respond ONLY with valid JSON
    - No explanations
    - No markdown
    - No extra text
    
    ========================
    INPUT TEXT
    ========================
    {applicant_text}
    """)
    ])

    chain = (
        SYSTEM_PROMPT.partial(format_instructions=parser.get_format_instructions())
        | llm
        | StrOutputParser()
    )
    return pipe, chain, tokenizer


pipe, chain1, tokenizer = load_model_and_chain()

# ==========================================
# 4. SCHOLARSHIP AGENTS (Direct from Notebook)
# ==========================================

class DataIngestionAgent:
    def __init__(self, univ_path, scholarship_path):
        self.univ_path = univ_path
        self.scholarship_path = scholarship_path

    def load_universities(self, domain_name):
        df = pd.read_excel(self.univ_path, sheet_name=domain_name, header=3)
        df.columns = df.columns.astype(str).str.strip()
        if 'INSTITUTION' in df.columns:
            df.rename(columns={'INSTITUTION': 'University / Partners'}, inplace=True)
        return df

    def load_scholarships(self):
        df = pd.read_excel(self.scholarship_path)
        df.columns = df.columns.astype(str).str.strip()
        return df


class ProfilingAgent:
    def run(self, user_data):
        """Convert input (dict or UserProfile) into the exact dict format needed by the system"""
        
        if isinstance(user_data, UserProfile):
            return self._profile_to_system_dict(user_data)

        elif isinstance(user_data, dict):
            return self._dict_to_system_dict(user_data)
        
        else:
            raise TypeError("user_data must be dict or UserProfile")

    def _dict_to_system_dict(self, data: dict) -> dict:
        """Convert raw dict to system format"""
        def to_bool(val):
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.strip().lower() in ['yes', 'true', '1']
            return False

        return {
            "domain": str(data.get("domain", "")),
            "gpa": float(data.get("gpa", 0.0)),
            "ielts": float(data.get("ielts", 0.0)),
            "degree": str(data.get("degree_level", "Masters")),
            "gre": str(data.get("gre", "no")).strip().lower(),
            "experience": str(data.get("experience_years", "no")).strip().lower(),
            "willing_to_return": str(data.get("willing_to_return", "no")).strip().lower(),
            "graduation_certificate": str(data.get("graduation_certificate", "no")).strip().lower()
        }

    def _profile_to_system_dict(self, profile: UserProfile) -> dict:
        """Convert UserProfile object to system format"""
        def bool_to_str(b):
            if b is None:
                return "no"
            return "yes" if b else "no"

        gre_exists = any([
            profile.test_scores.gre_verbal is not None,
            profile.test_scores.gre_quant is not None,
            profile.test_scores.gre_awa is not None
        ])

        ielts_val = float(profile.test_scores.ielts) if profile.test_scores.ielts is not None else 0.0

        return {
            "domain": str(profile.domain or ""),
            "gpa": float(profile.gpa or 0.0),
            "ielts": ielts_val,
            "degree": str(profile.degree_level or "Masters"),
            "gre": "yes" if gre_exists else "no",
            "experience": str(profile.experience_years or "no").strip().lower(),
            "willing_to_return": bool_to_str(profile.willing_to_return),
            "graduation_certificate": bool_to_str(profile.graduation_certificate)
        }


class UniversityFilterAgent:
    def _extract_rank(self, rank_val):
        try:
            match = re.search(r'\d+', str(rank_val))
            return int(match.group()) if match else 99999
        except:
            return 99999

    def run(self, univ_df):
        df = univ_df.copy()
        rank_col = '2026' if '2026' in df.columns else 'Rank'
        df['Numeric_Rank'] = df[rank_col].apply(self._extract_rank)

        # Sort by best rank and take top 20 globally
        top_20 = df.sort_values(by='Numeric_Rank', ascending=True).head(20)
        return top_20
        

class ScholarshipFilterAgent:
    def _extract_score(self, score_str):
        score_str = str(score_str).lower()
        if 'not required' in score_str or 'none' in score_str:
            return 0.0

        matches = re.findall(r'\b([4-9](?:\.\d)?)\b', score_str)
        if matches:
            return float(matches[0])
        return 0.0

    def run(self, scholarship_df, profile):
        df = scholarship_df.copy()

        # 1. Degree Level
        df = df[df['Degree Level'].str.contains(profile['degree'], case=False, na=False)]

        # 2. IELTS Requirement (Flexible: <= user's score)
        df['Parsed_IELTS'] = df['Min IELTS / TOEFL'].apply(self._extract_score)
        df = df[df['Parsed_IELTS'] <= profile['ielts']]

        # 3. Field Restrictions
        def field_match(x):
            x_str = str(x).lower()
            if 'open' in x_str or 'none' in x_str:
                return True
            domain_words = profile['domain'].lower().replace('&', '').split()
            return any(len(w) > 3 and w in x_str for w in domain_words)

        df = df[df['Field Restrictions'].apply(field_match)]

        # --- FLEXIBLE ELIGIBILITY FILTERS ---

        # 4. GRE Filter
        # If YES: they see everything. If NO: they only see "Not Required"
        if profile['gre'] in ["no", "false"]:
            df = df[df['GRE Required?'].str.contains("Not Required", case=False, na=False)]

        # 5. Experience Requirement Filter
        # If YES: they see both required and not required. If NO: drop the ones requiring it.
        if profile['experience'] in ["no", "0", "false"]:
            df = df[~df['Experience Required'].str.contains('Yes', case=False, na=False)]

        # 6. Return Obligation Filter
        # If YES (willing): they see both. If NO (unwilling): drop the ones requiring return.
        if profile['willing_to_return'] in ["no", "false"]:
            df = df[~df['Return Obligation'].str.contains('Yes', case=False, na=False)]

        # 7. Graduation Certificate Filter
        # If YES (has it): they see both. If NO (doesn't have it): drop the ones requiring it at application.
        if profile['graduation_certificate'] in ["no", "false"]:
            df = df[~df['Grad Certificate Required at Application?'].str.contains('Yes', case=False, na=False)]

        return df


class MatchingAgent:
    def __init__(self):
        self.country_mapping = {
            'United Kingdom': ['uk', 'united kingdom', 'britain', 'british'],
            'United States of America': ['us', 'usa', 'united states', 'american'],
            'France': ['france', 'french'],
            'Germany': ['germany', 'german'],
            'Italy': ['italy', 'italian'],
            'China (Mainland)': ['china', 'chinese'],
            'Japan': ['japan', 'japanese'],
            'South Korea': ['korea', 'korean'],
            'Australia': ['australia', 'australian'],
            'Canada': ['canada', 'canadian'],
            'Türkiye': ['turkey', 'turkish', 'türkiye'],
            'Russian Federation': ['russia', 'russian'],
            'Egypt': ['egypt', 'egyptian'],
            'Netherlands': ['netherlands', 'dutch'],
            'Switzerland': ['switzerland', 'swiss'],
            'Singapore': ['singapore']
        }

    def run(self, filtered_univs, filtered_scholarships, profile):
        univ_ranks = {}
        country_ranks = {}

        # Store University and Country QS Ranks
        for idx, row in filtered_univs.iterrows():
            u_name = str(row['University / Partners']).lower().strip()
            rank = row['Numeric_Rank']
            univ_ranks[u_name] = rank

            if 'COUNTRY/TERRITORY' in row:
                c_name = str(row['COUNTRY/TERRITORY']).lower().strip()
                if c_name not in country_ranks or rank < country_ranks[c_name]:
                    country_ranks[c_name] = rank

        # Expand country keywords (e.g. "British", "UK" maps to United Kingdom's rank)
        expanded_country_ranks = {}
        for c_name, rank in country_ranks.items():
            expanded_country_ranks[c_name] = rank
            for key_country, keywords in self.country_mapping.items():
                if key_country.lower() in c_name or c_name in key_country.lower():
                    for kw in keywords:
                        expanded_country_ranks[kw] = rank

        def get_best_rank(sch_univ):
            sch_str = str(sch_univ).lower()
            best_rank = 99999

            # Exact University Match
            for u_name, rank in univ_ranks.items():
                if u_name in sch_str or sch_str in u_name:
                    if rank < best_rank:
                        best_rank = rank

            # Country Match (only valid if country has a top 20 univ)
            if any(w in sch_str for w in ['all ', 'most ', 'designated', 'universities in', 'institutions']):
                for kw, rank in expanded_country_ranks.items():
                    if re.search(r'\b' + re.escape(kw) + r'\b', sch_str):
                        if rank < best_rank:
                            best_rank = rank

            # Always allow explicit local fellowships for Egypt
            if 'egypt' in sch_str and best_rank == 99999:
                return 99990

            return best_rank
        df = filtered_scholarships.copy()

        # Link every scholarship to a QS Rank
        df['Matched_Rank'] = df['University / Partners'].apply(get_best_rank)

        # EXPLICITLY DELETE anything that did not match a top 20 university/country
        valid_matches = df[df['Matched_Rank'] < 99999].copy()

        # Sort from Best Rank (1) to Worst Rank
        valid_matches = valid_matches.sort_values(by='Matched_Rank', ascending=True)

        # Return only up to 5 best options
        return valid_matches.head(5).drop(columns=['Matched_Rank', 'Parsed_IELTS'], errors='ignore')


class ScholarshipSystem:
    def __init__(self, univ_file, scholarship_file):
        self.ingestor = DataIngestionAgent(univ_file, scholarship_file)
        self.univ_filter = UniversityFilterAgent()
        self.sch_filter = ScholarshipFilterAgent()
        self.matcher = MatchingAgent()
        self.scholarship_data = self.ingestor.load_scholarships()

    def process_request(self, profile_dict):
        univ_data = self.ingestor.load_universities(profile_dict['domain'])
        top_univs = self.univ_filter.run(univ_data)
        possible_sch = self.sch_filter.run(self.scholarship_data, profile_dict)
        final_results = self.matcher.run(top_univs, possible_sch, profile_dict)
        return final_results



# ==========================================
# 5. REPORT AGENT (Notebook Style)
# ==========================================
class LLMReportGenerationAgent:
    def __init__(self, pipe):
        self.pipe = pipe

    def _truncate_text(self, text, max_chars=300):
        """Prevent extremely long fields from exploding token count."""
        if pd.isna(text):
            return "N/A"
        text = str(text).strip()
        if len(text) > max_chars:
            return text[:max_chars] + "..."
        return text
        
    def _format_scholarships_for_prompt(self, df: pd.DataFrame) -> str:
        """
        Serialize dataframe safely for LLM context.
        """
        context_blocks = []
        for idx, row in df.iterrows():

            scholarship_text = [
                f"--- SCHOLARSHIP OPPORTUNITY {idx + 1} ---"
            ]
            for col in df.columns:
                value = self._truncate_text(row.get(col, "N/A"))
                scholarship_text.append(
                    f"{col}: {value}"
                )
            context_blocks.append("\n".join(scholarship_text))
        return "\n\n".join(context_blocks)

    def generate_text(self, prompt: str, max_new_tokens: int = 1500) -> str:
        """Generate text using the pipeline"""
        try:
            # Call the pipeline directly
            output = self.pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                repetition_penalty=1.15,
                return_full_text=False
            )
            
            # Extract generated text from pipeline output
            if isinstance(output, list) and len(output) > 0:
                if isinstance(output[0], dict):
                    return output[0].get('generated_text', '')
                else:
                    return str(output[0])
            elif isinstance(output, str):
                return output
            else:
                return str(output)
                
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return None

    def generate_report(self, top_scholarships: pd.DataFrame, profile: dict):
        scholarships_context = self._format_scholarships_for_prompt(
            top_scholarships
        )

        profile_context = f"""
Degree Level: {profile.get('degree_level', 'N/A')}
Target Domain: {profile.get('domain', 'N/A')}
Willing to Return: {profile.get('willing_to_return', 'N/A')}
GPA: {profile.get('gpa', 'N/A')}
IELTS/TOEFL: {profile.get('ielts', 'N/A')}
"""

        prompt = f"""
You are an expert academic advisor and AI Engineer. Your task is to write a highly detailed, personalized, and encouraging Scholarship Selection Report for a student with a name {profile.get('name','N/A')}.
Format the output in clean, professional Markdown using bold headers, blockquotes, and tables where appropriate.
Address the student directly. Do not mention system dataframes, internal agent names, or pipeline code.
CRITICAL INSTRUCTION: Do not write generic support conclusions like "If you have any questions or need further guidance, please don't hesitate to ask." Cleanly conclude with your structural sections.
<</SYS>>

Based on the following student profile:
{profile_context}

And the following Top Scholarships matches containing all spreadsheet columns:
{scholarships_context}

Please write a comprehensive final report. You MUST fulfill the following structural requirements exactly:
1. EXECUTIVE SUMMARY: A personalized opening assessing how their specific research interest connects to their target domain.
2. DETAILED BREAKDOWN OF ALL OPPORTUNITIES AND MAKE SURE THE FIRST POINT IN THE NEXT LINE OF THE SCHOLARSHIP TITLE TO BE MORE CLEAR: For EACH of the scholarships provided, create a dedicated section containing:
   - Basic Info: Scholarship Name, University, Category, and Funding Type.
   - Core Description: The primary purpose of the scholarship.
   - Personalized Profile Fit Analysis: Explain exactly how their GPA, Language scores, and Specific Interests align with this option.
   - Technical Requirements Integration: Formally translate and address all other spreadsheet columns for them (e.g., Application Period, Deadline, Field Restrictions, GRE Requirements, Experience Required, and Return Obligations).
   - Estimated Acceptance Rate: Provide an expert estimate of the acceptance rate or competitiveness tier (e.g., Highly Competitive < 5%, Competitive 10-15%).
   - Actionable Application Link: Formulate a markdown hyperlink explicitly using the data from the 'Official Website' column, utilizing text like [View Official Application Portal](URL).
"""

        print(
            f"🤖 Generating report for {len(top_scholarships)} scholarships..."
        )

        report_output = self.generate_text(
            prompt,
            max_new_tokens=1500
            )

        if report_output is None:
            report_output = "❌ Failed to generate report. Please try again."

        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache()

        display(Markdown(report_output))

        return report_output








# ==========================================
# SESSION STATE — prefill store
# ==========================================
DEGREE_OPTIONS = ["Bachelor's", "Master's", "PhD", "Associate's", "Other"]
DOMAIN_OPTIONS = [ 'Arts & Humanities',
 'Archaeology',
 'Architecture _ Built Environmen',
 'Art & Design',
 'Classics & Ancient History',
 'English Language & Literature',
 'History_Subject',
 'History of Art',
 'Linguistics',
 'Modern Languages',
 'Music',
 'Performing Arts',
 'Philosophy',
 'Theology, Divinity & Religious ',
 'Engineering & Technology',
 'Computer Science & Information ',
 'Data Science and Artificial Int',
 'Engineering - Chemical',
 'Engineering - Civil & Structura',
 'Engineering - Electrical & Elec',
 'Engineering - Mechanical, Aeron',
 'Engineering - Mineral & Mining',
 'Petroleum Engineering',
 'Life Sciences & Medicine',
 'Agriculture & Forestry',
 'Anatomy & Physiology',
 'Biological Sciences',
 'Dentistry',
 'Medicine',
 'Nursing',
 'Pharmacy & Pharmacology',
 'Psychology',
 'Veterinary Science',
 'Natural Sciences',
 'Chemistry',
 'Earth & Marine Sciences',
 'Environmental Sciences',
 'Geography',
 'Geology',
 'Geophysics',
 'Materials Science',
 'Mathematics',
 'Physics & Astronomy',
 'Social Sciences & Management',
 'Accounting & Finance',
 'Anthropology',
 'Business & Management Studies',
 'Communication & Media Studies',
 'Development Studies',
 'Economics & Econometrics',
 'Education',
 'Hospitality & Leisure Managemen',
 'Law',
 'Library & Information Managemen',
 'Marketing',
 'Politics & International Studie',
 'Social Policy & Administration',
 'Sociology',
 'Sports-related Subjects',
 'Statistics & Operational Resear']


def _get(key, default):
    pre = st.session_state.get("_pre", {})
    return pre.get(key, st.session_state.get(key, default))

def _store_prefill(profile: UserProfile):
    pre = {}
    if profile.email:
        pre["email"] = profile.email
    if profile.university:
        pre["uni"] = profile.university
    if profile.degree_level:
        deg = profile.degree_level.lower()
        if "phd" in deg or "doctorate" in deg:
            pre["degree"] = "PhD"
        elif "master" in deg or "msc" in deg or "m.s" in deg:
            pre["degree"] = "Master's"
        elif "bachelor" in deg or "bsc" in deg or "b.s" in deg:
            pre["degree"] = "Bachelor's"
    if profile.domain:
        pre["spec"] = profile.domain
    if profile.experience_years:
        pre["exp"] = profile.experience_years
    if profile.willing_to_return is not None:
        pre["willing"] = "Yes" if profile.willing_to_return else "No"
    if profile.graduation_certificate is not None:
        pre["grad_cert"] = "Yes" if profile.graduation_certificate else "No"
    if profile.gpa is not None:
        gpa_val = profile.gpa
        if profile.gpa_scale and profile.gpa_scale != 4.0 and profile.gpa_scale > 0:
            gpa_val = round((profile.gpa / profile.gpa_scale) * 4.0, 2)
        pre["gpa"] = min(float(gpa_val), 4.0)
    if profile.test_scores.ielts is not None:
        pre["ielts"] = float(profile.test_scores.ielts)
    if profile.test_scores.toefl is not None:
        pre["toefl"] = int(profile.test_scores.toefl)
    if profile.test_scores.duolingo is not None:
        pre["duolingo"] = int(profile.test_scores.duolingo)
    if profile.test_scores.gre_verbal is not None and profile.test_scores.gre_quant is not None:
        pre["gre"] = int(profile.test_scores.gre_verbal) + int(profile.test_scores.gre_quant)
    elif profile.test_scores.gre_verbal is not None:
        pre["gre"] = int(profile.test_scores.gre_verbal)
    elif profile.test_scores.gre_quant is not None:
        pre["gre"] = int(profile.test_scores.gre_quant)
    if profile.project_titles:
        pre["projects"] = "\n".join(profile.project_titles)
    if profile.published_research_titles:
        pre["papers"] = len(profile.published_research_titles)
    if profile.competition_wins:
        pre["comps"] = len(profile.competition_wins)
    if profile.volunteering_activities:
        pre["volunteering"] = "\n".join(profile.volunteering_activities)
    st.session_state["_pre"] = pre
    st.session_state["_profile_loaded"] = True

# ==========================================
# UI — PAGE CONFIG & GLOBAL STYLES
# ==========================================
st.set_page_config(
    page_title="ScholarPath AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --navy:    #0b1120;
    --ink:     #111827;
    --gold:    #c9a84c;
    --gold-lt: #e6c97a;
    --cream:   #f5f0e8;
    --muted:   #6b7280;
    --card-bg: #ffffff;
    --border:  #e5e0d5;
    --radius:  14px;
    --shadow:  0 4px 24px rgba(0,0,0,.07);
}
html, body, [data-testid="stAppViewContainer"] {
    background: var(--cream) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--ink);
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stMain"] > div { max-width: 860px; margin: 0 auto; padding: 2rem 1.5rem 4rem; }
[data-testid="block-container"] { padding: 0 !important; }

.hero {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
}
.hero::before {
    content: "";
    position: absolute; inset: 0;
    background:
        radial-gradient(ellipse 60% 80% at 100% 0%, rgba(201,168,76,.18) 0%, transparent 60%),
        radial-gradient(ellipse 40% 60% at 0% 100%, rgba(201,168,76,.10) 0%, transparent 55%);
}
.hero-eyebrow { font-size:.75rem; font-weight:600; letter-spacing:.18em; text-transform:uppercase; color:var(--gold); margin-bottom:.75rem; }
.hero h1 { font-family:'Playfair Display',serif; font-size:clamp(2.2rem,5vw,3.4rem); font-weight:900; color:#fff; line-height:1.13; margin:0 0 1rem; }
.hero h1 span { color: var(--gold-lt); }
.hero-badge {
    display:inline-block; margin-bottom:1.5rem;
    background:rgba(201,168,76,.15); border:1px solid rgba(201,168,76,.35);
    border-radius:50px; padding:.35rem 1.1rem;
    font-size:.72rem; font-weight:600; letter-spacing:.1em; color:var(--gold-lt); text-transform:uppercase;
}
.section-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2rem 2.2rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow);
}
.section-title {
    font-family:'Playfair Display',serif; font-size:1.15rem; font-weight:700; color:var(--navy);
    margin-bottom:.3rem; display:flex; align-items:center; gap:.55rem;
}
.section-title .icon {
    width:28px; height:28px; background:linear-gradient(135deg,var(--navy),#1e3a5f);
    border-radius:8px; display:flex; align-items:center; justify-content:center;
    font-size:.9rem; flex-shrink:0;
}
.gold-divider { height:2px; background:linear-gradient(90deg,var(--gold) 0%,transparent 100%); border:none; margin:.5rem 0 1.4rem; border-radius:2px; width:48px; }

/* ── Report window ── */
.report-window {
    background: #fff;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    margin-top: 2rem;
    box-shadow: var(--shadow);
    overflow: hidden;
}
.report-titlebar {
    background: linear-gradient(135deg, var(--navy) 0%, #1a2f52 100%);
    padding: 1rem 1.8rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.report-titlebar-left {
    display: flex;
    align-items: center;
    gap: .75rem;
}
.report-titlebar-dot {
    width: 11px; height: 11px; border-radius: 50%;
    display: inline-block;
}
.report-body {
    padding: 2rem 2.4rem;
    max-height: 70vh;
    overflow-y: auto;
    font-size: .93rem;
    line-height: 1.75;
    color: var(--ink);
}
.report-body h1, .report-body h2 {
    font-family: 'Playfair Display', serif;
    color: var(--navy);
}
.report-body h2 { border-bottom: 2px solid var(--gold); padding-bottom: .3rem; margin-top: 1.6rem; }
.report-body blockquote {
    border-left: 3px solid var(--gold);
    margin: .8rem 0;
    padding: .4rem 1rem;
    background: rgba(201,168,76,.06);
    border-radius: 0 8px 8px 0;
    color: #4b5563;
}
.report-body table {
    width: 100%; border-collapse: collapse; font-size: .85rem; margin: 1rem 0;
}
.report-body th {
    background: var(--navy); color: #fff; padding: .5rem .8rem; text-align: left; font-weight: 600;
}
.report-body td {
    padding: .45rem .8rem; border-bottom: 1px solid var(--border);
}
.report-body tr:nth-child(even) td { background: rgba(201,168,76,.04); }
.report-body a { color: var(--gold); text-decoration: underline; }
.report-body code {
    background: #f3f4f6; border-radius: 4px; padding: .1rem .4rem; font-size: .82rem;
}

/* ── Download button (gold) ── */
.dl-btn-wrapper [data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, var(--gold) 0%, #b8922a 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: .75rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: .95rem !important;
    font-weight: 600 !important;
    letter-spacing: .04em !important;
    box-shadow: 0 4px 16px rgba(201,168,76,.35) !important;
    transition: opacity .2s, transform .15s !important;
}
.dl-btn-wrapper [data-testid="stDownloadButton"] > button:hover {
    opacity: .88 !important; transform: translateY(-1px) !important;
}
.dl-btn-wrapper [data-testid="stDownloadButton"] > button::before { content: "⬇  "; }

[data-testid="stTextArea"] textarea {
    border: 2px solid var(--border) !important; border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 1.05rem !important;
    background: var(--cream) !important; color: var(--ink) !important;
    line-height: 1.65 !important; padding: 1.1rem 1.3rem !important;
    width: 100% !important; transition: border-color .2s, box-shadow .2s;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 4px rgba(201,168,76,.13) !important; outline: none !important;
}
[data-testid="stFileUploader"] {
    background: var(--cream); border: 2px dashed var(--border);
    border-radius: 14px; padding: .5rem; transition: border-color .2s; width: 100%;
}
[data-testid="stFileUploader"]:hover { border-color: var(--gold); }
.text-hint { text-align:center; font-size:.82rem; color:var(--muted); margin-bottom:1rem; line-height:1.55; }
.upload-hint {
    background:linear-gradient(135deg,rgba(201,168,76,.08),rgba(201,168,76,.02));
    border:1px solid rgba(201,168,76,.22); border-radius:10px;
    padding:.75rem 1.2rem; font-size:.8rem; color:var(--muted);
    margin-bottom:1rem; line-height:1.6; text-align:center;
}
.upload-hint strong { color:var(--ink); }
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] > div > div {
    border:1.5px solid var(--border) !important; border-radius:10px !important;
    font-family:'DM Sans',sans-serif !important; font-size:.92rem !important;
    background:var(--cream) !important; color:var(--ink) !important; transition:border-color .2s;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {
    border-color:var(--gold) !important; box-shadow:0 0 0 3px rgba(201,168,76,.12) !important;
}
label, [data-testid="stWidgetLabel"] p {
    font-family:'DM Sans',sans-serif !important; font-size:.8rem !important; font-weight:600 !important;
    color:var(--muted) !important; letter-spacing:.04em !important; text-transform:uppercase !important;
}
[data-testid="stButton"] > button {
    width:100%; background:linear-gradient(135deg,var(--navy) 0%,#1a2f52 100%) !important;
    color:#fff !important; border:none !important; border-radius:12px !important;
    padding:1rem 2rem !important; font-family:'DM Sans',sans-serif !important;
    font-size:1.05rem !important; font-weight:600 !important; letter-spacing:.04em !important;
    cursor:pointer; transition:opacity .2s,transform .15s !important;
    box-shadow:0 4px 20px rgba(11,17,32,.28) !important; margin-top:.75rem;
}
[data-testid="stButton"] > button:hover { opacity:.88 !important; transform:translateY(-1px) !important; }
[data-testid="stButton"] > button::after { content:" →"; font-size:1.1rem; }
.chip-hint { font-size:.75rem; color:var(--muted); margin-top:.3rem; text-align:center; }
.prefilled-notice {
    background: linear-gradient(135deg, rgba(201,168,76,.12), rgba(201,168,76,.04));
    border: 1px solid rgba(201,168,76,.35); border-radius: 10px;
    padding: .65rem 1.1rem; font-size: .82rem; color: #92702a;
    margin-bottom: 1.2rem; display: flex; align-items: center; gap: .5rem;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">🎓 AI Scholarship Agent</div>
  <div class="hero-eyebrow">ScholarPath</div>
  <h1>Find Your Perfect<br><span>Scholarship Path</span></h1>
  <p style="font-size:1rem;color:rgba(0,0,0,.5);line-height:1.65;max-width:540px;margin:0 auto 1.5rem;">
    Describe your academic profile and let our AI agent match you with scholarships.
  </p>
</div>
""", unsafe_allow_html=True)

def section(title, icon):
    st.markdown(
        f'<div class="section-title"><div class="icon">{icon}</div>{title}</div>'
        f'<div class="gold-divider"></div>',
        unsafe_allow_html=True,
    )

def run_extraction(input_text: str):
    if "user_profile" in st.session_state and st.session_state.user_profile is not None:
        st.success("✅ Using previously extracted profile!")
        return st.session_state.user_profile
    if not input_text.strip():
        st.warning("⚠️ Please provide some text or upload a file first.")
        return
    with st.spinner("🤖 Extracting your Data..."):
        try:
            raw_output = chain1.invoke({"applicant_text": input_text})
            cleaned_json = extract_json(raw_output)
            data = json.loads(cleaned_json)
            profile = UserProfile(**data)
            st.session_state.user_profile = profile
            _store_prefill(profile)
            st.success("✅ Profile extracted and fields auto-filled!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Extraction Failed: {e}")
            with st.expander("Debug LLM Output"):
                st.text(raw_output if 'raw_output' in locals() else "No output generated.")

    
# ══════════════════════════════════════════════════════════════════════════════
# 1 · QUERY SECTION
# ══════════════════════════════════════════════════════════════════════════════
section("Your Query", "✍️")

if "query_mode" not in st.session_state:
    st.session_state.query_mode = "text"

_, mc1, mc2, _ = st.columns([1, 2, 2, 1])
with mc1:
    if st.button("✏️  Write your question", key="btn_text_mode",
                 type="primary" if st.session_state.query_mode == "text" else "secondary"):
        st.session_state.query_mode = "text"
        st.rerun()
with mc2:
    if st.button("📎  Upload a document", key="btn_file_mode",
                 type="primary" if st.session_state.query_mode == "file" else "secondary"):
        st.session_state.query_mode = "file"
        st.rerun()

st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

text_query = None
query_file = None

if st.session_state.query_mode == "text":
    st.markdown('<p class="text-hint">Describe your goal — target country, degree level, field, funding type, or any specific scholarship question.</p>', unsafe_allow_html=True)
    text_query = st.text_area(
        "query_textarea",
        placeholder="e.g. I'm looking for fully-funded PhD scholarships in Machine Learning for students from Egypt…",
        height=200, key="text_query", label_visibility="collapsed",
    )
    _, eb, _ = st.columns([3, 3, 3])
    with eb:
        if st.button("⚡ Extract My Data", key="extract_text_btn", use_container_width=True):
            run_extraction(text_query or "")
else:
    st.markdown('<div class="upload-hint"><strong>Accepted:</strong> PDF <br>Upload your <strong>CV, transcript, SOP, or research proposal</strong> — the agent will use it as your query.</div>', unsafe_allow_html=True)
    _, uc, _ = st.columns([0.5, 9, 0.5])
    with uc:
        query_file = st.file_uploader(
            "query_file", type=["pdf"],
            accept_multiple_files=False, key="query_upload", label_visibility="collapsed",
        )
    if query_file:
        st.markdown(f"<p style='text-align:center;font-size:.83rem;color:#c9a84c;margin:.4rem 0 0;'>✓ <strong>{query_file.name}</strong> ready to send</p>", unsafe_allow_html=True)
        _, eb, _ = st.columns([3, 3, 3])
        with eb:
            if st.button("⚡ Extract My Data", key="extract_file_btn", use_container_width=True):
                pdf_text = extract_text_from_pdf(query_file.read())
                run_extraction(pdf_text)

st.markdown('</div>', unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════════════════════
# 2 · Personal info
# ══════════════════════════════════════════════════════════════════════════════
section("Personal Info", "📧")
a1, a2 = st.columns(2)
with a1:
    name = st.text_input("Name", placeholder="e.g. John Doe", value=_get("name", ""))
with a2:
    email = st.text_input("Email Address", placeholder="e.g. user@gmail.com", value=_get("email", ""))
st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3 · ACADEMIC BACKGROUND
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.get("_profile_loaded"):
    st.markdown('<div class="prefilled-notice">✨ Fields below were auto-filled from your document. Review and adjust as needed.</div>', unsafe_allow_html=True)

section("Academic Background", "🏛️")
c1, c2 = st.columns(2)
with c1:
    university = st.text_input("University / Institution", placeholder="e.g. Cairo University", value=_get("uni", ""))

    # Domain as a selectbox (matches Excel sheet names)
    dom_default = _get("spec", DOMAIN_OPTIONS[0])
    dom_idx = 0
    for i, opt in enumerate(DOMAIN_OPTIONS):
        if dom_default and dom_default.lower() in opt.lower():
            dom_idx = i
            break
    domain = st.selectbox("Academic Domain", DOMAIN_OPTIONS, index=dom_idx)

    experience_years = st.number_input("Years of Experience", min_value=0, max_value=50, step=1, value=int(_get("exp", 0)))
with c2:
    deg_default = _get("degree", "Bachelor's")
    deg_idx = DEGREE_OPTIONS.index(deg_default) if deg_default in DEGREE_OPTIONS else 0
    degree = st.selectbox("Degree Level", DEGREE_OPTIONS, index=deg_idx)

    graduation_certificate = st.checkbox("Has Graduation Certificate", value=_get("grad_cert", "No") == "Yes")
    gpa = st.number_input("GPA", min_value=0.0, max_value=4.0, step=0.01, format="%.2f", value=float(_get("gpa", 0.0)))
st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 4 · TEST SCORES
# ══════════════════════════════════════════════════════════════════════════════
section("Test Scores", "📊")
t1, t2, t3, t4 = st.columns(4)
with t1:
    ielts    = st.number_input("IELTS", min_value=0.0, max_value=9.0, step=0.5, format="%.1f", value=float(_get("ielts", 0.0)))
with t2:
    toefl    = st.number_input("TOEFL iBT", min_value=0, max_value=120, step=1, value=int(_get("toefl", 0)))
with t3:
    duolingo = st.number_input("Duolingo", min_value=0, max_value=160, step=5, value=int(_get("duolingo", 0)))
with t4:
    gre      = st.number_input("GRE Total", min_value=0, max_value=340, step=1, value=int(_get("gre", 0)))
st.markdown('<p class="chip-hint">Leave at 0 for any test you haven\'t taken.</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 5 · RESEARCH & PROJECTS
# ══════════════════════════════════════════════════════════════════════════════
section("Research & Projects", "🔬")
r1, r2, r3 = st.columns([3, 1, 1])
with r1:
    project_titles = st.text_area(
        "Project Titles (one per line)",
        placeholder="Deep Learning-based Medical Image Segmentation\nArabic NLP Sentiment Analysis Tool",
        height=110, value=_get("projects", ""),
    )
with r2:
    pub_papers = st.number_input("Published Papers", min_value=0, max_value=100, step=1, value=int(_get("papers", 0)))
with r3:
    comp_wins  = st.number_input("Competition Wins", min_value=0, max_value=200, step=1, value=int(_get("comps", 0)))
st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 6 · VOLUNTEERING
# ══════════════════════════════════════════════════════════════════════════════
section("Volunteering & Activities", "🤝")
volunteering = st.text_area(
    "vol",
    placeholder="e.g. Tutored underprivileged students for 2 years…",
    height=120, value=_get("volunteering", ""), label_visibility="collapsed",
)
st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 7 · PREFERENCES
# ══════════════════════════════════════════════════════════════════════════════
section("Preferences", "⚙️")
p1, p2 = st.columns(2)
with p1:
    target_country   = st.text_input("Target Country / Region", placeholder="USA, UK, Germany…", value=_get("country", ""))
    willing_to_return = st.checkbox("Willing to return to home country?", value=_get("willing", "No") == "Yes")
with p2:
    field_of_study = st.text_input("Field of Study", placeholder="AI, Biomedical…", value=_get("field", ""))
funding_type = st.multiselect(
    "Funding Type",
    ["Fully-funded", "Partial funding", "Tuition waiver", "Stipend", "Any"],
    default=_get("funding", ["Any"]),
)
st.markdown('</div>', unsafe_allow_html=True)

# ── Submit ─────────────────────────────────────────────────────────────────
submitted = st.button("🚀 Find Me Scholarships", key="submit", use_container_width=True)



# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def build_profile_summary() -> str:
    lines = []
    if name:            lines.append(f"Name: {name}")
    if email:           lines.append(f"Email: {email}")
    if university:      lines.append(f"University: {university}")
    if degree:          lines.append(f"Degree: {degree}")
    if gpa:             lines.append(f"GPA: {gpa:.2f} / 4.00")
    scores = []
    if ielts:    scores.append(f"IELTS {ielts}")
    if toefl:    scores.append(f"TOEFL {toefl}")
    if duolingo: scores.append(f"Duolingo {duolingo}")
    if gre:      scores.append(f"GRE {gre}")
    if scores:          lines.append("Test Scores: " + ", ".join(scores))
    if project_titles:  lines.append(f"Projects:\n{project_titles}")
    if pub_papers:      lines.append(f"Published Papers: {pub_papers}")
    if comp_wins:       lines.append(f"Competition Wins: {comp_wins}")
    if volunteering:    lines.append(f"Volunteering:\n{volunteering}")
    if target_country:  lines.append(f"Target Country: {target_country}")
    if field_of_study:  lines.append(f"Field of Study: {field_of_study}")
    if funding_type:    lines.append(f"Funding Preference: {', '.join(funding_type)}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# SUBMISSION LOGIC
# ══════════════════════════════════════════════════════════════════════════════
if submitted:
    st.write("🚀 Button clicked - Starting pipeline...")
    if "user_profile" not in st.session_state or st.session_state.user_profile is None:
        st.error("❌ Please extract your profile first using the 'Extract My Data' button.")
        st.stop()

    profile_dict = st.session_state.user_profile

    if hasattr(profile_dict, "model_dump"):   # Pydantic v2
        profile_dict = profile_dict.model_dump()
    elif hasattr(profile_dict, "dict"):       # Pydantic v1
        profile_dict = profile_dict.dict()
    elif not isinstance(profile_dict, dict):
        st.error("❌ Invalid profile format. Must be dict or Pydantic model.")
        st.stop()

    # ── Step 2: Scholarship Pipeline + Report ───────────────────────────────
    st.success("✅ Matching Completed!")

    with st.spinner("🔍 Wait for Generating Your Report..."):
            # Initialize system
            system = ScholarshipSystem(
                "/kaggle/input/datasets/abdallahbeshary/data-scholar/Universities.xlsx", 
                "/kaggle/input/datasets/abdallahbeshary/data-scholar/Scholarships.xlsx"
            )

            profile_agent = ProfilingAgent()
            user_profile_obj = UserProfile(**profile_dict)
            system_input = profile_agent.run(user_profile_obj)
            # st.write(system_input)

            # Get top scholarships
            top_5 = system.process_request(system_input)
            
            display_cols = ['Scholarship Name', 'University / Partners', 'Funding Type', 'Deadline Month']
            available_cols = [col for col in display_cols if col in top_5.columns]
            # st.dataframe(top_5[available_cols], use_container_width=True)
            df_top_5 = pd.DataFrame(top_5) if not isinstance(top_5, pd.DataFrame) else top_5.copy()
            
    # ── Step 3: Scholarship Report ───────────────────────────────
            # Rich user profile for the report
            report_user_profile = {
                "name":  name or profile_dict.get("name") or "Applicant",                 # Add if available
                "domain": user_profile_obj.domain,
                "gpa": user_profile_obj.gpa,
                "ielts": user_profile_obj.test_scores.ielts,
                "degree_level": user_profile_obj.degree_level,
                "gre": "yes" if any([user_profile_obj.test_scores.gre_verbal, 
                                    user_profile_obj.test_scores.gre_quant, 
                                    user_profile_obj.test_scores.gre_awa]) else "no",
                "experience_years": user_profile_obj.experience_years,
                "willing_to_return": user_profile_obj.willing_to_return,
                "university": user_profile_obj.university,
                "projects": user_profile_obj.project_titles,
                "research": user_profile_obj.published_research_titles,
                "competitions": user_profile_obj.competition_wins,
                "volunteering": user_profile_obj.volunteering_activities,
                "home_country": "Egypt"                 # You can make this dynamic
            }
            
            # Initialize Report Agent
            llm_agent = LLMReportGenerationAgent(pipe)
            
            # Generate the beautiful report
            final_report = llm_agent.generate_report(
                top_scholarships=df_top_5, 
                profile=report_user_profile
            )

            display_name = name or profile_dict.get("name") or "Applicant"
            
            st.markdown("---")
            st.markdown(f"## 📄 Scholarship Report for {display_name}")
            st.markdown("---")

            tab1, tab2 = st.tabs(["📊 Full Report", "💾 Export Options"])
            
            with tab1:
                # عرض التقرير كـ Markdown
                st.markdown(final_report)
            
            with tab2:
                st.markdown("### 💾 Download your Report")
                
                if st.button("📥 Download as PDF", use_container_width=True):
                    html_body = markdown2.markdown(
                        final_report,
                        extras=["tables", "fenced-code-blocks", "blockquote", "cuddled-lists"]
                    )
            
                    # 2. Design an executive, academic-styled CSS stylesheet matching your system's design
                    css_styles = """
                    @page {
                        size: A4;
                        margin: 20mm 15mm 20mm 15mm;
                        @bottom-right {
                            content: "Page " counter(page) " of " counter(pages);
                            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                            font-size: 8pt;
                            color: #718096;
                        }
                        @bottom-left {
                            content: "Multi-Agent Selection System — Confidential Report";
                            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                            font-size: 8pt;
                            color: #718096;
                        }
                    }
            
                    body {
                        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                        color: #2d3748;
                        line-height: 1.6;
                        font-size: 10.5pt;
                    }
            
                    h1 {
                        color: #1a365d;
                        font-size: 20pt;
                        border-bottom: 2px solid #2b6cb0;
                        padding-bottom: 8px;
                        margin-top: 0;
                        margin-bottom: 20px;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                    }
            
                    h2 {
                        color: #2b6cb0;
                        font-size: 14pt;
                        margin-top: 25px;
                        margin-bottom: 12px;
                        border-left: 4px solid #1a365d;
                        padding-left: 10px;
                        page-break-after: avoid;
                    }
            
                    h3 {
                        color: #2d3748;
                        font-size: 11.5pt;
                        margin-top: 20px;
                        margin-bottom: 8px;
                        font-weight: bold;
                        page-break-after: avoid;
                    }
            
                    p {
                        margin-bottom: 10px;
                        text-align: justify;
                    }
            
                    li {
                        margin-bottom: 6px;
                        text-align: left;
                    }
            
                    ul, ol {
                        margin-top: 5px;
                        margin-bottom: 15px;
                        padding-left: 20px;
                        list-style-type: disc;
                    }
            
                    blockquote {
                        background-color: #f7fafc;
                        border-left: 3.5px solid #4a5568;
                        margin: 15px 0;
                        padding: 10px 15px;
                        font-style: italic;
                        color: #4a5568;
                    }
            
                    table {
                        width: 100%;
                        border-collapse: collapse;
                        margin: 15px 0;
                        font-size: 9.5pt;
                        page-break-inside: avoid;
                    }
            
                    th, td {
                        border: 1px solid #e2e8f0;
                        padding: 8px 12px;
                        text-align: left;
                    }
            
                    th {
                        background-color: #1a365d;
                        color: #ffffff;
                        font-weight: bold;
                    }
            
                    tr:nth-child(even) {
                        background-color: #f8fafc;
                    }
                    """
            
                    # 3. Assemble the comprehensive single-file HTML document context
                    full_html_document = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <meta charset="utf-8">
                        <style>
                            {css_styles}
                        </style>
                    </head>
                    <body>
                        {html_body}
                    </body>
                    </html>
                    """
                
                    # Generate PDF entirely in memory — no temp files, no weasyprint
                    pdf_buffer = io.BytesIO()
                    pisa_status = pisa.CreatePDF(
                        src=full_html_document,
                        dest=pdf_buffer,
                        encoding="utf-8",
                    )
            
                    if pisa_status.err:
                        st.error("❌ PDF generation failed. Please try again.")
                    else:
                        pdf_buffer.seek(0)
                        pdf_filename = f"scholarship_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            
                        st.success("✅ PDF report generated successfully!")
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_buffer.read(),
                            file_name=pdf_filename,
                            mime="application/pdf",
                            use_container_width=True,
                        )
                                            