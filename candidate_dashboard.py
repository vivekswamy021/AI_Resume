import streamlit as st
import json
import re
import traceback
import tempfile
import os
from datetime import date

# Define the main function for the Candidate Dashboard
# NOTE: The function signature must include ALL required arguments:
# generate_skill_roadmap, DEFAULT_JOB_TYPES, and DEFAULT_ROLES.
def candidate_dashboard(go_to, parse_and_store_resume, qa_on_resume, evaluate_interview_answers, generate_interview_questions, question_section_options, extract_jd_metadata, get_file_type, extract_content, extract_jd_from_linkedin_url, clear_interview_state, evaluate_jd_fit, generate_skill_roadmap, DEFAULT_JOB_TYPES, DEFAULT_ROLES, qa_on_jd=None):
    
    # --- HELPER FUNCTIONS (Placeholder function for removed JD Chatbot) ---
    
    def jd_qa_on_resume_placeholder(question, jd_text, parsed_json):
        """Simulates a chatbot answer based on JD requirements and resume content (Only used as fallback)."""
        if not jd_text.strip():
            return "Please select a Job Description above to start the conversation about your fit for the role."
        
        if "error" in parsed_json:
             return f"Cannot answer: Resume parsing errors were detected. Error: {parsed_json['error']}"
        
        return (
            f"**Simulated AI Response (JD Chatbot):**\n\n"
            f"Based on the Job Description and your resume (mentioning: {parsed_json.get('skills', ['...'])[:3]}), "
            f"your query '{question}' is highly relevant. "
            f"The system recommends focusing on your experience with related skills mentioned in the JD. "
            f"(The dedicated JD Chatbot feature has been removed.)"
        )
    # -------------------------------------------------------------
    
    # --- Helper functions that interact with session state or use utils ---
    
    def generate_cv_html(parsed_data):
        """Generates a simple, print-friendly HTML string from parsed data for PDF conversion."""
        
        css = """
        <style>
            @page { size: A4; margin: 1cm; }
            body { font-family: 'Arial', sans-serif; line-height: 1.5; margin: 0; padding: 0; font-size: 10pt; }
            .header { text-align: center; border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px; }
            .header h1 { margin: 0; font-size: 1.8em; }
            .contact-info { display: flex; justify-content: center; font-size: 0.8em; color: #555; }
            .contact-info span { margin: 0 8px; }
            .section { margin-bottom: 15px; page-break-inside: avoid; }
            .section h2 { border-bottom: 1px solid #999; padding-bottom: 3px; margin-bottom: 8px; font-size: 1.1em; text-transform: uppercase; color: #333; }
            .item-list ul { list-style-type: disc; margin-left: 20px; padding-left: 0; margin-top: 0; }
            .item-list ul li { margin-bottom: 3px; }
            .item-list p { margin: 3px 0 8px 0; }
            a { color: #0056b3; text-decoration: none; }
        </style>
        """
        
        html_content = f"<html><head>{css}<title>{parsed_data.get('name', 'CV')}</title></head><body>"
        
        # 1. Header and Contact Info
        html_content += '<div class="header">'
        html_content += f"<h1>{parsed_data.get('name', 'Candidate Name')}</h1>"
        
        contact_parts = []
        if parsed_data.get('email'): contact_parts.append(f"<span>📧 {parsed_data['email']}</span>")
        if parsed_data.get('phone'): contact_parts.append(f"<span>📱 {parsed_data['phone']}</span>")
        if parsed_data.get('linkedin'): contact_parts.append(f"<span>🔗 <a href='{parsed_data['linkedin']}'>{parsed_data.get('linkedin', 'LinkedIn').split('/')[-1] if parsed_data.get('linkedin') else 'LinkedIn'}</a></span>")
        if parsed_data.get('github'): contact_parts.append(f"<span>💻 <a href='{parsed_data['github']}'>{parsed_data.get('github', 'GitHub').split('/')[-1] if parsed_data.get('github') else 'GitHub'}</a></span>")
        
        html_content += f'<div class="contact-info">{" | ".join(contact_parts)}</div>'
        html_content += '</div>'
        
        # 2. Sections
        section_order = ['personal_details', 'experience', 'projects', 'education', 'certifications', 'skills', 'strength']
        
        for k in section_order:
            v = parsed_data.get(k)
            
            if k in ['name', 'email', 'phone', 'linkedin', 'github']: continue 

            if v and (isinstance(v, str) and v.strip() or isinstance(v, list) and v):
                
                html_content += f'<div class="section"><h2>{k.replace("_", " ").title()}</h2>'
                html_content += '<div class="item-list">'
                
                if k == 'personal_details' and isinstance(v, str):
                    html_content += f"<p>{v}</p>"
                elif isinstance(v, list):
                    html_content += '<ul>'
                    for item in v:
                        if item: 
                            html_content += f"<li>{item}</li>"
                    html_content += '</ul>'
                else:
                    html_content += f"<p>{v}</p>"
                    
                html_content += '</div></div>'

        html_content += '</body></html>'
        return html_content


    def format_parsed_json_to_markdown(parsed_data):
        """Formats the parsed JSON data into a clean, CV-like Markdown structure."""
        md = ""
        
        if parsed_data.get('name'):
            md += f"# **{parsed_data['name']}**\n\n"
        
        contact_info = []
        if parsed_data.get('email'): contact_info.append(parsed_data['email'])
        if parsed_data.get('phone'): contact_info.append(parsed_data['phone'])
        if parsed_data.get('linkedin'): contact_info.append(f"[LinkedIn]({parsed_data['linkedin']})")
        if parsed_data.get('github'): contact_info.append(f"[GitHub]({parsed_data['github']})")
        
        if contact_info:
            md += f"| {' | '.join(contact_info)} |\n"
            md += "| " + " | ".join(["---"] * len(contact_info)) + " |\n\n"
        
        section_order = ['personal_details', 'experience', 'projects', 'education', 'certifications', 'skills', 'strength']
        
        for k in section_order:
            v = parsed_data.get(k)
            
            if k in ['name', 'email', 'phone', 'linkedin', 'github']: continue 

            if v and (isinstance(v, str) and v.strip() or isinstance(v, list) and v):
                
                md += f"## **{k.replace('_', ' ').upper()}**\n"
                md += "---\n"
                
                if k == 'personal_details' and isinstance(v, str):
                    md += f"{v}\n\n"
                elif isinstance(v, list):
                    for item in v:
                        if item: 
                            md += f"- {item}\n"
                    md += "\n"
                else:
                    md += f"{v}\n\n"
        return md


    def cv_management_tab_content():
        st.header("📝 Prepare Your CV")
        st.markdown("### 1. Form Based CV Builder")
        st.info("Fill out the details below to generate a parsed CV that can be used immediately for matching and interview prep, or start by parsing a file in the 'Resume Parsing' tab.")

        default_parsed = {
            "name": "", "email": "", "phone": "", "linkedin": "", "github": "",
            "skills": [], "experience": [], "education": [], "certifications": [], 
            "projects": [], "strength": [], "personal_details": ""
        }
        
        if "cv_form_data" not in st.session_state:
            if st.session_state.get('parsed', {}).get('name'):
                st.session_state.cv_form_data = st.session_state.parsed.copy()
            else:
                st.session_state.cv_form_data = default_parsed
        
        # --- CV Builder Form ---
        with st.form("cv_builder_form"):
            st.subheader("Personal & Contact Details")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.session_state.cv_form_data['name'] = st.text_input(
                    "Full Name", 
                    value=st.session_state.cv_form_data['name'], 
                    key="cv_name"
                )
            with col2:
                st.session_state.cv_form_data['email'] = st.text_input(
                    "Email Address", 
                    value=st.session_state.cv_form_data['email'], 
                    key="cv_email"
                )
            with col3:
                st.session_state.cv_form_data['phone'] = st.text_input(
                    "Phone Number", 
                    value=st.session_state.cv_form_data['phone'], 
                    key="cv_phone"
                )
            
            col4, col5 = st.columns(2)
            with col4:
                st.session_state.cv_form_data['linkedin'] = st.text_input(
                    "LinkedIn Profile URL", 
                    value=st.session_state.cv_form_data.get('linkedin', ''), 
                    key="cv_linkedin"
                )
            with col5:
                st.session_state.cv_form_data['github'] = st.text_input(
                    "GitHub Profile URL", 
                    value=st.session_state.cv_form_data.get('github', ''), 
                    key="cv_github"
                )
            
            st.markdown("---")
            st.subheader("Summary / Personal Details")
            st.session_state.cv_form_data['personal_details'] = st.text_area(
                "Professional Summary or Personal Details", 
                value=st.session_state.cv_form_data.get('personal_details', ''), 
                height=100,
                key="cv_personal_details"
            )
            
            st.markdown("---")
            st.subheader("Technical Sections (One Item per Line)")

            skills_text = "\n".join(st.session_state.cv_form_data.get('skills', []))
            new_skills_text = st.text_area(
                "Key Skills (Technical and Soft)", 
                value=skills_text,
                height=150,
                key="cv_skills"
            )
            st.session_state.cv_form_data['skills'] = [s.strip() for s in new_skills_text.split('\n') if s.strip()]
            
            experience_text = "\n".join(st.session_state.cv_form_data.get('experience', []))
            new_experience_text = st.text_area(
                "Professional Experience (Job Roles, Companies, Dates, Key Responsibilities)", 
                value=experience_text,
                height=150,
                key="cv_experience"
            )
            st.session_state.cv_form_data['experience'] = [e.strip() for e in new_experience_text.split('\n') if e.strip()]

            education_text = "\n".join(st.session_state.cv_form_data.get('education', []))
            new_education_text = st.text_area(
                "Education (Degrees, Institutions, Dates)", 
                value=education_text,
                height=100,
                key="cv_education"
            )
            st.session_state.cv_form_data['education'] = [d.strip() for d in new_education_text.split('\n') if d.strip()]
            
            certifications_text = "\n".join(st.session_state.cv_form_data.get('certifications', []))
            new_certifications_text = st.text_area(
                "Certifications (Name, Issuing Body, Date)", 
                value=certifications_text,
                height=100,
                key="cv_certifications"
            )
            st.session_state.cv_form_data['certifications'] = [c.strip() for c in new_certifications_text.split('\n') if c.strip()]
            
            projects_text = "\n".join(st.session_state.cv_form_data.get('projects', []))
            new_projects_text = st.text_area(
                "Projects (Name, Description, Technologies)", 
                value=projects_text,
                height=150,
                key="cv_projects"
            )
            st.session_state.cv_form_data['projects'] = [p.strip() for p in new_projects_text.split('\n') if p.strip()]
            
            strength_text = "\n".join(st.session_state.cv_form_data.get('strength', []))
            new_strength_text = st.text_area(
                "Strengths / Key Personal Qualities (One per line)", 
                value=strength_text,
                height=100,
                key="cv_strength"
            )
            st.session_state.cv_form_data['strength'] = [s.strip() for s in new_strength_text.split('\n') if s.strip()]


            submit_form_button = st.form_submit_button("Generate and Load CV Data", use_container_width=True)

        if submit_form_button:
            if not st.session_state.cv_form_data['name'] or not st.session_state.cv_form_data['email']:
                st.error("Please fill in at least your **Full Name** and **Email Address**.")
                return

            st.session_state.parsed = st.session_state.cv_form_data.copy()
            st.session_state.parsed['name'] = st.session_state.cv_form_data['name']
            
            compiled_text = ""
            for k, v in st.session_state.cv_form_data.items():
                if v:
                    compiled_text += f"{k.replace('_', ' ').title()}:\n"
                    if isinstance(v, list):
                        compiled_text += "\n".join([f"- {item}" for item in v]) + "\n\n"
                    else:
                        compiled_text += str(v) + "\n\n"
            st.session_state.full_text = compiled_text
            
            st.session_state.candidate_match_results = []
            st.session_state.interview_qa = []
            st.session_state.evaluation_report = ""
            st.session_state.jd_interview_qa = [] 
            st.session_state.jd_evaluation_report = "" 
            st.session_state.skill_roadmap_report = "" # Clear roadmap state
            st.session_state.selected_roadmap_jd_name = "" 

            st.success(f"✅ CV data for **{st.session_state.parsed['name']}** successfully generated and loaded! You can now use the Chatbot, Match, and Interview Prep tabs.")
            
        st.markdown("---")
        st.subheader("2. Loaded CV Data Preview and Download")
        
        if st.session_state.get('parsed', {}).get('name'):
            
            filled_data_for_preview = {
                k: v for k, v in st.session_state.parsed.items() 
                if v and (isinstance(v, str) and v.strip() or isinstance(v, list) and v)
            }
            
            tab_markdown, tab_json, tab_pdf = st.tabs(["📝 Markdown View", "💾 JSON View", "⬇️ PDF/HTML Download"])

            with tab_markdown:
                cv_markdown_preview = format_parsed_json_to_markdown(filled_data_for_preview)
                st.markdown(cv_markdown_preview)

                st.download_button(
                    label="⬇️ Download CV as Markdown (.md)",
                    data=cv_markdown_preview,
                    file_name=f"{st.session_state.parsed.get('name', 'Generated_CV').replace(' ', '_')}_CV_Document.md",
                    mime="text/markdown",
                    key="download_cv_markdown_final"
                )

            with tab_json:
                st.json(st.session_state.parsed)
                json_output = json.dumps(st.session_state.parsed, indent=2)
                st.download_button(
                    label="⬇️ Download CV as JSON File",
                    data=json_output,
                    file_name=f"{st.session_state.parsed.get('name', 'Generated_CV').replace(' ', '_')}_CV_Data.json",
                    mime="application/json",
                    key="download_cv_json_final"
                )

            with tab_pdf:
                st.markdown("### Download CV as HTML (Print-to-PDF)")
                st.info("Click the button below to download an HTML file. Open the file in your browser and use the browser's **'Print'** function, selecting **'Save as PDF'** to create your final CV document.")
                
                html_output = generate_cv_html(filled_data_for_preview)

                st.download_button(
                    label="⬇️ Download CV as Print-Ready HTML File (for PDF conversion)",
                    data=html_output,
                    file_name=f"{st.session_state.parsed.get('name', 'Generated_CV').replace(' ', '_')}_CV_Document.html",
                    mime="text/html",
                    key="download_cv_html"
                )
                
                st.markdown("---")
                st.markdown("### Raw Text Data Download (for utility)")
                st.download_button(
                    label="⬇️ Download All CV Data as Raw Text (.txt)",
                    data=st.session_state.full_text,
                    file_name=f"{st.session_state.parsed.get('name', 'Generated_CV').replace(' ', '_')}_Raw_Data.txt",
                    mime="text/plain",
                    key="download_cv_txt_final"
                )
                
        else:
            st.info("Please fill out the form above and click 'Generate and Load CV Data' or parse a resume in the 'Resume Parsing' tab to see the preview and download options.")


    def filter_jd_tab_content():
        st.header("🔍 Filter Job Descriptions by Criteria")

        if not st.session_state.candidate_jd_list:
            st.info("No Job Descriptions are currently loaded. Please add JDs in the 'JD Management' tab (Tab 4).")
            if 'filtered_jds_display' not in st.session_state:
                st.session_state.filtered_jds_display = []
            return
        
        # Use DEFAULT_ROLES and DEFAULT_JOB_TYPES from the function signature
        unique_roles = sorted(list(set(
            [item.get('role', 'General Analyst') for item in st.session_state.candidate_jd_list] + DEFAULT_ROLES
        )))
        unique_job_types = sorted(list(set(
            [item.get('job_type', 'Full-time') for item in st.session_state.candidate_jd_list] + DEFAULT_JOB_TYPES
        )))
        
        STARTER_KEYWORDS = {
            "Python", "MySQL", "GCP", "cloud computing", "ML", 
            "API services", "LLM integration", "JavaScript", "SQL", "AWS" 
        }
        
        all_unique_skills = set(STARTER_KEYWORDS)
        for jd in st.session_state.candidate_jd_list:
            valid_skills = [
                skill.strip() for skill in jd.get('key_skills', []) 
                if isinstance(skill, str) and skill.strip()
            ]
            all_unique_skills.update(valid_skills)
        
        unique_skills_list = sorted(list(all_unique_skills))
        
        if not unique_skills_list:
            unique_skills_list = ["No skills extracted from current JDs"]

        all_jd_data = st.session_state.candidate_jd_list

        with st.form(key="jd_filter_form"):
            st.markdown("### Select Filters")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                selected_skills = st.multiselect(
                    "Skills Keywords (Select multiple)",
                    options=unique_skills_list,
                    default=st.session_state.get('last_selected_skills', []),
                    key="candidate_filter_skills_multiselect", 
                    help="Select one or more skills. JDs containing ANY of the selected skills will be shown."
                )
                
            with col2:
                selected_job_type = st.selectbox(
                    "Job Type",
                    options=["All Job Types"] + unique_job_types,
                    index=0,
                    key="filter_job_type_select"
                )
                
            with col3:
                selected_role = st.selectbox(
                    "Role Title",
                    options=["All Roles"] + unique_roles,
                    index=0,
                    key="filter_role_select"
                )

            apply_filters_button = st.form_submit_button("✅ Apply Filters", type="primary", use_container_width=True)

        if apply_filters_button:
            
            st.session_state.last_selected_skills = selected_skills

            filtered_jds = []
            selected_skills_lower = [k.strip().lower() for k in selected_skills]
            
            for jd in all_jd_data:
                jd_role = jd.get('role', 'General Analyst')
                jd_job_type = jd.get('job_type', 'Full-time')
                jd_key_skills = [
                    s.lower() for s in jd.get('key_skills', []) 
                    if isinstance(s, str) and s.strip()
                ]
                
                role_match = (selected_role == "All Roles") or (selected_role == jd_role)
                job_type_match = (selected_job_type == "All Job Types") or (selected_job_type == jd_job_type)
                
                skill_match = True
                if selected_skills_lower:
                    if not any(k in jd_key_skills for k in selected_skills_lower):
                        skill_match = False
                
                if role_match and job_type_match and skill_match:
                    filtered_jds.append(jd)
                    
            st.session_state.filtered_jds_display = filtered_jds
            st.success(f"Filter applied! Found {len(filtered_jds)} matching Job Descriptions.")

        st.markdown("---")
        
        if 'filtered_jds_display' not in st.session_state:
            st.session_state.filtered_jds_display = []
            
        filtered_jds = st.session_state.filtered_jds_display
        
        st.subheader(f"Matching Job Descriptions ({len(filtered_jds)} found)")
        
        if filtered_jds:
            display_data = []
            for jd in filtered_jds:
                display_data.append({
                    "Job Description Title": jd['name'].replace("--- Simulated JD for: ", ""),
                    "Role": jd.get('role', 'N/A'),
                    "Job Type": jd.get('job_type', 'N/A'),
                    "Key Skills": ", ".join(jd.get('key_skills', ['N/A'])[:5]) + "...",
                })
                
            st.dataframe(display_data, use_container_width=True)

            st.markdown("##### Detailed View")
            for idx, jd in enumerate(filtered_jds, 1):
                with st.expander(f"JD {idx}: {jd['name'].replace('--- Simulated JD for: ', '')} - ({jd.get('role', 'N/A')})"):
                    st.markdown(f"**Job Type:** {jd.get('job_type', 'N/A')}")
                    st.markdown(f"**Extracted Skills:** {', '.join(jd.get('key_skills', ['N/A']))}")
                    st.markdown("---")
                    st.text(jd['content'])
        elif st.session_state.candidate_jd_list and st.session_state.get('last_selected_skills') is not None:
             st.info("No Job Descriptions match the selected criteria. Try broadening your filter selections.")
        elif st.session_state.candidate_jd_list:
            st.info("Use the filters above and click **'Apply Filters'** to view matching Job Descriptions.")
    
    # --- MAIN CANDIDATE DASHBOARD FUNCTION START ---

    st.header("👩‍🎓 Candidate Dashboard")
    st.markdown("Welcome! Use the tabs below to manage your CV and access AI preparation tools.")

    nav_col, _ = st.columns([1, 1]) 

    with nav_col:
        if st.button("🚪 Log Out", key="candidate_logout_btn", use_container_width=True):
            go_to("login") 
    
    with st.sidebar:
        st.header("Resume/CV Status")
        
        if st.session_state.parsed.get("name"):
            st.success(f"Currently loaded: **{st.session_state.parsed['name']}**")
        elif st.session_state.full_text:
            st.warning("Resume content is loaded, but parsing may have errors.")
        else:
            st.info("Please upload a file or use the CV builder in 'CV Management' to begin.")

    is_resume_parsed = bool(st.session_state.get('parsed', {}).get('name')) or bool(st.session_state.get('full_text'))
    
    # Updated Tab List to include the new tab_roadmap
    tab_cv_mgmt, tab1, tab_resume_qa, tab_interview_prep, tab_roadmap, tab4, tab5, tab6 = st.tabs([
        "✍️ CV Management", 
        "📄 Resume Parsing", 
        "💬 Resume Chatbot (Q&A)", 
        "❓ Interview Prep", 
        "🗺️ Skill Roadmap", # New Tab Title
        "📚 JD Management", 
        "🎯 Batch JD Match",
        "🔍 Filter JD"
    ])
    
    # --- TAB 0: CV Management ---
    with tab_cv_mgmt:
        cv_management_tab_content()

    # --- TAB 1: Resume Parsing ---
    with tab1:
        st.header("Resume Upload and Parsing")
        
        input_method = st.radio(
            "Select Input Method",
            ["Upload File", "Paste Text"],
            key="parsing_input_method"
        )
        
        st.markdown("---")

        # A. Upload File Method
        if input_method == "Upload File":
            st.markdown("### 1. Upload Resume File") 
            
            uploaded_file = st.file_uploader( 
                "Choose PDF, DOCX, TXT, JSON, MD, CSV, XLSX file", 
                type=["pdf", "docx", "txt", "json", "md", "csv", "xlsx", "markdown", "rtf"], 
                accept_multiple_files=False, 
                key='candidate_file_upload_main'
            )
            
            st.markdown(
                """
                <div style='font-size: 10px; color: grey;'>
                Supported File Types: PDF, DOCX, TXT, JSON, MARKDOWN, CSV, XLSX, RTF
                </div>
                """, 
                unsafe_allow_html=True
            )
            st.markdown("---")


            if uploaded_file is not None:
                st.session_state.candidate_uploaded_resumes = [uploaded_file] 
                st.session_state.pasted_cv_text = ""
            elif st.session_state.candidate_uploaded_resumes and uploaded_file is None:
                st.session_state.candidate_uploaded_resumes = []
                st.session_state.parsed = {}
                st.session_state.full_text = ""
                st.toast("Upload cleared.")
            
            file_to_parse = st.session_state.candidate_uploaded_resumes[0] if st.session_state.candidate_uploaded_resumes else None
            
            st.markdown("### 2. Parse Uploaded File")
            
            if file_to_parse:
                if st.button(f"Parse and Load: **{file_to_parse.name}**", use_container_width=True):
                    with st.spinner(f"Parsing {file_to_parse.name}..."):
                        result = parse_and_store_resume(file_to_parse, file_name_key='single_resume_candidate', source_type='file')
                        
                        if "error" not in result:
                            st.session_state.parsed = result['parsed']
                            st.session_state.full_text = result['full_text']
                            st.session_state.excel_data = result['excel_data'] 
                            st.session_state.parsed['name'] = result['name'] 
                            clear_interview_state()
                            st.session_state.jd_interview_qa = [] 
                            st.session_state.jd_evaluation_report = "" 
                            st.session_state.skill_roadmap_report = "" # Clear roadmap state
                            st.session_state.selected_roadmap_jd_name = "" 

                            st.success(f"✅ Successfully loaded and parsed **{result['name']}**.")
                            st.info("View, edit, and download the parsed data in the **CV Management** tab.") 
                        else:
                            st.error(f"Parsing failed for {file_to_parse.name}: {result['error']}")
                            st.session_state.parsed = {"error": result['error'], "name": result['name']}
                            st.session_state.full_text = result['full_text'] or ""
            else:
                st.info("No resume file is currently uploaded. Please upload a file above.")

        # B. Paste Text Method
        else:
            st.markdown("### 1. Paste Your CV Text")
            
            pasted_text = st.text_area(
                "Copy and paste your entire CV or resume text here.",
                value=st.session_state.get('pasted_cv_text', ''),
                height=300,
                key='pasted_cv_text_input'
            )
            st.session_state.pasted_cv_text = pasted_text
            
            st.markdown("---")
            st.markdown("### 2. Parse Pasted Text")
            
            if pasted_text.strip():
                if st.button("Parse and Load Pasted Text", use_container_width=True):
                    with st.spinner("Parsing pasted text..."):
                        st.session_state.candidate_uploaded_resumes = []
                        
                        result = parse_and_store_resume(pasted_text, file_name_key='single_resume_candidate', source_type='text')
                        
                        if "error" not in result:
                            st.session_state.parsed = result['parsed']
                            st.session_state.full_text = result['full_text']
                            st.session_state.excel_data = result['excel_data'] 
                            st.session_state.parsed['name'] = result['name'] 
                            clear_interview_state()
                            st.session_state.jd_interview_qa = [] 
                            st.session_state.jd_evaluation_report = "" 
                            st.session_state.skill_roadmap_report = "" # Clear roadmap state
                            st.session_state.selected_roadmap_jd_name = "" 

                            st.success(f"✅ Successfully loaded and parsed **{result['name']}**.")
                            st.info("View, edit, and download the parsed data in the **CV Management** tab.") 
                        else:
                            st.error(f"Parsing failed: {result['error']}")
                            st.session_state.parsed = {"error": result['error'], "name": result['name']}
                            st.session_state.full_text = result['full_text'] or ""
            else:
                st.info("Please paste your CV text into the box above.")


    # --- TAB 2: Resume Chatbot (Q&A) ---
    with tab_resume_qa:
        st.header("Resume Chatbot (Q&A)") # Simplified header
        
        st.markdown("### Ask any question about the currently loaded resume.")
        if not is_resume_parsed:
            st.warning("Please upload and parse a resume in the 'Resume Parsing' tab first.")
        elif "error" in st.session_state.parsed:
             st.error("Cannot use Chatbot: Resume data has parsing errors.")
        else:
            if 'qa_answer' not in st.session_state: st.session_state.qa_answer = ""
            
            question = st.text_input("Your Question", placeholder="e.g., What are the candidate's key skills?", key="resume_qa_question")
            
            if st.button("Get Answer", key="resume_qa_btn"):
                with st.spinner("Generating answer..."):
                    try:
                        answer = qa_on_resume(question)
                        st.session_state.qa_answer = answer
                    except Exception as e:
                        st.error(f"Error during Q&A: {e}")
                        st.session_state.qa_answer = "Could not generate an answer."

            if st.session_state.get('qa_answer'):
                st.text_area("Answer", st.session_state.qa_answer, height=150, key="resume_qa_answer_display")
                    


    # --- TAB 3: Interview Prep ---
    with tab_interview_prep:
        st.header("Interview Preparation Tools")
        
        if not is_resume_parsed or "error" in st.session_state.parsed:
            st.warning("Please upload and successfully parse a resume first.")
        else:
            
            tab_resume_qa_prep, tab_jd_qa_prep = st.tabs(["🎯 Resume Section Focus", "💼 JD-Specific Practice"])
            
            
            # --- Subtab 1: Resume Section Focus ---
            with tab_resume_qa_prep:
                
                if 'iq_output' not in st.session_state: st.session_state.iq_output = ""
                if 'interview_qa' not in st.session_state: st.session_state.interview_qa = [] 
                if 'evaluation_report' not in st.session_state: st.session_state.evaluation_report = "" 

                st.subheader("1. Generate Interview Questions (Based on Resume)")
                
                section_choice = st.selectbox(
                    "Select Section", 
                    question_section_options, 
                    key='iq_section_c',
                    # Reset state when switching focus
                    on_change=lambda: [st.session_state.update({'interview_qa': [], 'evaluation_report': ""})] 
                )
                
                if st.button("Generate Questions from Resume", key='iq_btn_c'):
                    with st.spinner("Generating questions..."):
                        try:
                            # Pass parsed data and section choice
                            raw_questions_response = generate_interview_questions(st.session_state.parsed, section_choice)
                            st.session_state.iq_output = raw_questions_response
                            
                            st.session_state.interview_qa = [] 
                            st.session_state.evaluation_report = "" 
                            
                            q_list = []
                            current_level = ""
                            for line in raw_questions_response.splitlines():
                                line = line.strip()
                                if line.startswith('[') and line.endswith(']'):
                                    current_level = line.strip('[]')
                                elif line.lower().startswith('q') and ':' in line:
                                    question_text = line[line.find(':') + 1:].strip()
                                    q_list.append({
                                        "question": f"({current_level}) {question_text}",
                                        "answer": "", 
                                        "level": current_level
                                    })
                                    
                            st.session_state.interview_qa = q_list
                            
                            st.success(f"Generated {len(q_list)} questions based on your **{section_choice}** section.")
                            
                        except Exception as e:
                            st.error(f"Error generating questions: {e}")
                            st.session_state.iq_output = "Error generating questions."
                            st.session_state.interview_qa = []

                if st.session_state.get('interview_qa'):
                    st.markdown("---")
                    st.subheader("2. Practice and Record Answers")
                    
                    with st.form("resume_interview_practice_form"):
                        
                        for i, qa_item in enumerate(st.session_state.interview_qa):
                            st.markdown(f"**Question {i+1}:** {qa_item['question']}")
                            
                            answer = st.text_area(
                                f"Your Answer for Q{i+1}", 
                                value=st.session_state.interview_qa[i]['answer'], 
                                height=100,
                                key=f'resume_answer_q_{i}',
                                label_visibility='collapsed'
                            )
                            st.session_state.interview_qa[i]['answer'] = answer 
                            st.markdown("---") 
                            
                        submit_button = st.form_submit_button("Submit & Evaluate Answers", use_container_width=True)

                        if submit_button:
                            
                            if all(item['answer'].strip() for item in st.session_state.interview_qa):
                                with st.spinner("Sending answers to AI Evaluator..."):
                                    try:
                                        # Pass full_text for better context in evaluation
                                        report = evaluate_interview_answers(
                                            st.session_state.interview_qa,
                                            st.session_state.parsed
                                        )
                                        st.session_state.evaluation_report = report
                                        st.success("Evaluation complete! See the report below.")
                                    except Exception as e:
                                        st.error(f"Evaluation failed: {e}")
                                        st.session_state.evaluation_report = f"Evaluation failed: {e}\n{traceback.format_exc()}"
                            else:
                                st.error("Please answer all generated questions before submitting.")
                    
                    if st.session_state.get('evaluation_report'):
                        st.markdown("---")
                        st.subheader("3. AI Evaluation Report")
                        st.markdown(st.session_state.evaluation_report)

            
            # --- Subtab 2: JD-Specific Practice ---
            with tab_jd_qa_prep:
                
                if not st.session_state.get('candidate_jd_list'):
                    st.error("Please **add Job Descriptions** in the 'JD Management' tab (Tab 4) before using JD-specific preparation.")
                else:
                    if 'jd_interview_qa' not in st.session_state: st.session_state.jd_interview_qa = [] 
                    if 'jd_evaluation_report' not in st.session_state: st.session_state.jd_evaluation_report = "" 
                    
                    st.subheader("1. Select JD and Generate Questions")
                    
                    jd_options = [jd['name'] for jd in st.session_state.candidate_jd_list]
                    selected_jd_name = st.selectbox(
                        "Choose a JD to base the interview questions on:",
                        options=["-- Select a JD --"] + jd_options,
                        key="jd_interview_selector",
                        # Clear JD interview state if selection changes
                        on_change=lambda: [st.session_state.update({'jd_interview_qa': [], 'jd_evaluation_report': ""})] 
                    )

                    selected_jd_data = None
                    if selected_jd_name and selected_jd_name != "-- Select a JD --":
                        selected_jd_data = next(
                            (item for item in st.session_state.candidate_jd_list if item['name'] == selected_jd_name),
                            None
                        )
                    
                    if selected_jd_data and st.button(f"Generate Questions for **{selected_jd_name}**", key='jd_iq_btn'):
                        with st.spinner(f"Generating questions based on {selected_jd_name} and your resume..."):
                            try:
                                # Use JD content as the section choice, signaling a JD-based query
                                raw_questions_response = generate_interview_questions(
                                    st.session_state.parsed, 
                                    selected_jd_data['content'] 
                                )
                                
                                st.session_state.jd_interview_qa = [] 
                                st.session_state.jd_evaluation_report = "" 
                                
                                q_list = []
                                current_level = "JD-Specific"
                                for line in raw_questions_response.splitlines():
                                    line = line.strip()
                                    # Assuming LLM output format might be simple bullet points or Q:
                                    if line.lower().startswith('q') and ':' in line:
                                        question_text = line[line.find(':') + 1:].strip()
                                        q_list.append({
                                            "question": f"({current_level}) {question_text}",
                                            "answer": "", 
                                            "level": current_level
                                        })
                                    elif line.startswith('-') or line.startswith('*'):
                                        q_list.append({
                                            "question": f"({current_level}) {line[1:].strip()}",
                                            "answer": "", 
                                            "level": current_level
                                        })
                                        
                                st.session_state.jd_interview_qa = q_list
                                st.success(f"Generated {len(q_list)} JD-specific questions.")
                                
                            except Exception as e:
                                st.error(f"Error generating JD questions: {e}")
                                st.session_state.jd_interview_qa = []

                    if st.session_state.get('jd_interview_qa'):
                        st.markdown("---")
                        st.subheader("2. Practice and Record Answers (JD-Specific)")
                        
                        with st.form("jd_interview_practice_form"):
                            
                            for i, qa_item in enumerate(st.session_state.jd_interview_qa):
                                st.markdown(f"**Question {i+1}:** {qa_item['question']}")
                                
                                answer = st.text_area(
                                    f"Your Answer for Q{i+1}", 
                                    value=st.session_state.jd_interview_qa[i]['answer'], 
                                    height=100,
                                    key=f'jd_answer_q_{i}',
                                    label_visibility='collapsed'
                                )
                                st.session_state.jd_interview_qa[i]['answer'] = answer 
                                st.markdown("---") 
                                
                            submit_button = st.form_submit_button("Submit & Evaluate JD Answers", use_container_width=True)

                            if submit_button:
                                if all(item['answer'].strip() for item in st.session_state.jd_interview_qa):
                                    with st.spinner("Sending JD answers to AI Evaluator..."):
                                        try:
                                            # Pass JD content as context for evaluation
                                            report = evaluate_interview_answers(
                                                st.session_state.jd_interview_qa,
                                                st.session_state.parsed,
                                                jd_context=selected_jd_data['content']
                                            )
                                            st.session_state.jd_evaluation_report = report
                                            st.success("JD Evaluation complete! See the report below.")
                                        except Exception as e:
                                            st.error(f"JD Evaluation failed: {e}")
                                            st.session_state.jd_evaluation_report = f"JD Evaluation failed: {e}\n{traceback.format_exc()}"
                                else:
                                    st.error("Please answer all generated questions before submitting.")
                        
                        if st.session_state.get('jd_evaluation_report'):
                            st.markdown("---")
                            st.subheader("3. AI Evaluation Report (JD-Specific)")
                            st.markdown(st.session_state.jd_evaluation_report)


    # --- TAB 4: Skill Roadmap ---
    with tab_roadmap:
        st.header("🗺️ Skill Roadmap & Gap Analysis")
        st.markdown("Identify your skill gaps for a specific job description and get a personalized development plan.")
        
        # Initialize state for this tab
        if 'skill_roadmap_report' not in st.session_state: st.session_state.skill_roadmap_report = ""
        if 'selected_roadmap_jd_name' not in st.session_state: st.session_state.selected_roadmap_jd_name = ""
        
        if not is_resume_parsed:
            st.warning("❌ **Action Required:** Please upload and successfully parse your resume in the 'Resume Parsing' tab first.")
            
        elif not st.session_state.get('candidate_jd_list'):
            st.error("❌ **Action Required:** Please add Job Descriptions in the 'JD Management' tab before running a skill roadmap analysis.")
            
        else:
            jd_options = [jd['name'] for jd in st.session_state.candidate_jd_list]
            
            selected_jd_name = st.selectbox(
                "1. Choose a JD for Skill Gap Analysis:",
                options=["-- Select a JD --"] + jd_options,
                index=0 if st.session_state.selected_roadmap_jd_name not in jd_options else jd_options.index(st.session_state.selected_roadmap_jd_name) + 1,
                key="roadmap_jd_selector"
            )

            selected_jd_data = None
            if selected_jd_name and selected_jd_name != "-- Select a JD --":
                selected_jd_data = next(
                    (item for item in st.session_state.candidate_jd_list if item['name'] == selected_jd_name),
                    None
                )
                st.session_state.selected_roadmap_jd_name = selected_jd_name
            else:
                selected_jd_name = None
                st.session_state.selected_roadmap_jd_name = ""


            st.markdown("---")

            if selected_jd_data:
                
                if st.button(f"Generate Skill Roadmap for **{selected_jd_name}**", key='generate_roadmap_btn', use_container_width=True):
                    with st.spinner("Analyzing skill gaps and generating personalized roadmap..."):
                        try:
                            # Use the passed utility function
                            roadmap_report = generate_skill_roadmap(
                                selected_jd_data['content'],
                                st.session_state.parsed
                            )
                            st.session_state.skill_roadmap_report = roadmap_report
                            st.success("✅ Roadmap generated successfully! See details below.")
                            
                        except Exception as e:
                            st.error(f"Error generating roadmap: {e}")
                            st.session_state.skill_roadmap_report = f"Error generating roadmap: {e}\n{traceback.format_exc()}"
                
                
                st.markdown("---")
                st.subheader("2. Personalized Skill Roadmap")
                
                if st.session_state.get('skill_roadmap_report'):
                    st.markdown(st.session_state.skill_roadmap_report)
                else:
                    st.info("Click the button above to generate your skill roadmap based on the selected JD.")

            else:
                 st.info("Please select a Job Description to proceed with the Skill Roadmap analysis.")


    # --- TAB 5: JD Management (Candidate) ---
    with tab4:
        st.header("📚 Manage Job Descriptions for Matching")
        
        if "candidate_jd_list" not in st.session_state:
             st.session_state.candidate_jd_list = []
        
        jd_type = st.radio("Select JD Type", ["Single JD", "Multiple JD"], key="jd_type_candidate")
        st.markdown("### Add JD by:")
        
        method = st.radio("Choose Method", ["Upload File", "Paste Text", "LinkedIn URL"], key="jd_add_method_candidate") 

        # URL
        if method == "LinkedIn URL":
            url_list = st.text_area(
                "Enter one or more URLs (comma separated)" if jd_type == "Multiple JD" else "Enter URL", key="url_list_candidate"
            )
            if st.button("Add JD(s) from URL", key="add_jd_url_btn_candidate"):
                if url_list:
                    urls = [u.strip() for u in url_list.split(",")] if jd_type == "Multiple JD" else [url_list.strip()]
                    
                    count = 0
                    for url in urls:
                        if not url: continue
                        
                        with st.spinner(f"Attempting JD extraction and metadata analysis for: {url}"):
                            jd_text = extract_jd_from_linkedin_url(url)
                            metadata = extract_jd_metadata(jd_text)
                        
                        name_base = url.split('/jobs/view/')[-1].split('/')[0] if '/jobs/view/' in url else f"URL {count+1}"
                        name = f"JD from URL: {name_base}" 
                        if name in [item['name'] for item in st.session_state.candidate_jd_list]:
                            name = f"JD from URL: {name_base} ({len(st.session_state.candidate_jd_list) + 1})" 

                        st.session_state.candidate_jd_list.append({"name": name, "content": jd_text, **metadata})
                        
                        if not jd_text.startswith("[Error"):
                            count += 1
                                
                    if count > 0:
                        st.success(f"✅ {count} JD(s) added successfully! Check the display below for the extracted content.")
                    else:
                        st.error("No JDs were added successfully.")


        # Paste Text
        elif method == "Paste Text":
            text_list = st.text_area(
                "Paste one or more JD texts (separate by '---')" if jd_type == "Multiple JD" else "Paste JD text here", key="text_list_candidate"
            )
            if st.button("Add JD(s) from Text", key="add_jd_text_btn_candidate"):
                if text_list:
                    texts = [t.strip() for t in text_list.split("---")] if jd_type == "Multiple JD" else [text_list.strip()]
                    for i, text in enumerate(texts):
                         if text:
                            name_base = text.splitlines()[0].strip()
                            if len(name_base) > 30: name_base = f"{name_base[:27]}..."
                            if not name_base: name_base = f"Pasted JD {len(st.session_state.candidate_jd_list) + i + 1}"
                            
                            metadata = extract_jd_metadata(text)
                            st.session_state.candidate_jd_list.append({"name": name_base, "content": text, **metadata})
                    st.success(f"✅ {len(texts)} JD(s) added successfully!")

        # Upload File
        elif method == "Upload File":
            uploaded_files = st.file_uploader(
                "Upload JD file(s)",
                type=["pdf", "txt", "docx"],
                accept_multiple_files=(jd_type == "Multiple JD"),
                key="jd_file_uploader_candidate"
            )
            if st.button("Add JD(s) from File", key="add_jd_file_btn_candidate"):
                if uploaded_files is None:
                    st.warning("Please upload file(s).")
                    
                files_to_process = uploaded_files if isinstance(uploaded_files, list) else ([uploaded_files] if uploaded_files else [])
                
                count = 0
                for file in files_to_process:
                    if file:
                        temp_dir = tempfile.mkdtemp()
                        temp_path = os.path.join(temp_dir, file.name)
                        with open(temp_path, "wb") as f:
                            f.write(file.getbuffer())
                            
                        file_type = get_file_type(temp_path)
                        jd_text = extract_content(file_type, temp_path)
                        
                        if not jd_text.startswith("Error"):
                            metadata = extract_jd_metadata(jd_text)
                            st.session_state.candidate_jd_list.append({"name": file.name, "content": jd_text, **metadata})
                            count += 1
                        else:
                            st.error(f"Error extracting content from {file.name}: {jd_text}")
                            
                if count > 0:
                    st.success(f"✅ {count} JD(s) added successfully!")
                elif uploaded_files:
                    st.error("No valid JD files were uploaded or content extraction failed.")


        # Display Added JDs
        if st.session_state.candidate_jd_list:
            
            col_display_header, col_clear_button = st.columns([3, 1])
            
            with col_display_header:
                st.markdown("### ✅ Current JDs Added:")
                
            with col_clear_button:
                if st.button("🗑️ Clear All JDs", key="clear_jds_candidate", use_container_width=True, help="Removes all currently loaded JDs."):
                    st.session_state.candidate_jd_list = []
                    st.session_state.candidate_match_results = [] 
                    st.session_state.filtered_jds_display = [] 
                    st.session_state.jd_interview_qa = [] 
                    st.session_state.jd_evaluation_report = "" 
                    st.session_state.skill_roadmap_report = "" # Clear roadmap state
                    st.session_state.selected_roadmap_jd_name = "" 
                    st.success("All JDs and associated data have been cleared.")
                    st.rerun() 

            for idx, jd_item in enumerate(st.session_state.candidate_jd_list, 1):
                title = jd_item['name']
                display_title = title.replace("--- Simulated JD for: ", "")
                with st.expander(f"JD {idx}: {display_title} | Role: {jd_item.get('role', 'N/A')}"):
                    st.markdown(f"**Job Type:** {jd_item.get('job_type', 'N/A')} | **Key Skills:** {', '.join(jd_item.get('key_skills', ['N/A']))}")
                    st.markdown("---")
                    st.text(jd_item['content'])
        else:
            st.info("No Job Descriptions added yet.")

    # --- TAB 6: Batch JD Match (Candidate) ---
    with tab5:
        st.header("🎯 Batch JD Match: Best Matches")
        st.markdown("Compare your current resume against all saved job descriptions.")

        if not is_resume_parsed:
            st.warning("Please **upload and parse your resume** in the 'Resume Parsing' tab or **build your CV** in the 'CV Management' tab first.")
        
        elif not st.session_state.candidate_jd_list:
            st.error("Please **add Job Descriptions** in the 'JD Management' tab (Tab 4) before running batch analysis.")
            
        else:
            if "candidate_match_results" not in st.session_state:
                st.session_state.candidate_match_results = []

            all_jd_names = [item['name'] for item in st.session_state.candidate_jd_list]
            
            selected_jd_names = st.multiselect(
                "Select Job Descriptions to Match Against",
                options=all_jd_names,
                default=all_jd_names,
                key='candidate_batch_jd_select'
            )
            
            jds_to_match = [
                jd_item for jd_item in st.session_state.candidate_jd_list 
                if jd_item['name'] in selected_jd_names
            ]
            
            if st.button(f"Run Match Analysis on {len(jds_to_match)} Selected JD(s)"):
                st.session_state.candidate_match_results = []
                
                if not jds_to_match:
                    st.warning("Please select at least one Job Description to run the analysis.")
                    
                else:
                    resume_name = st.session_state.parsed.get('name', 'Uploaded Resume')
                    parsed_json = st.session_state.parsed
                    results_with_score = []

                    with st.spinner(f"Matching {resume_name}'s resume against {len(jds_to_match)} selected JD(s)..."):
                        
                        for jd_item in jds_to_match:
                            
                            jd_name = jd_item['name']
                            jd_content = jd_item['content']

                            try:
                                fit_output = evaluate_jd_fit(jd_content, parsed_json)
                                
                                overall_score_match = re.search(r'Overall Fit Score:\s*[^\d]*(\d+)\s*/10', fit_output, re.IGNORECASE)
                                section_analysis_match = re.search(
                                    r'--- Section Match Analysis ---\s*(.*?)\s*Strengths/Matches:', 
                                    fit_output, re.DOTALL
                                )

                                skills_percent, experience_percent, education_percent = 'N/A', 'N/A', 'N/A'
                                
                                if section_analysis_match:
                                    section_text = section_analysis_match.group(1)
                                    skills_match = re.search(r'Skills Match:\s*\[?(\d+)%\]?', section_text, re.IGNORECASE)
                                    experience_match = re.search(r'Experience Match:\s*\[?(\d+)%\]?', section_text, re.IGNORECASE)
                                    education_match = re.search(r'Education Match:\s*\[?(\d+)%\]?', section_text, re.IGNORECASE)
                                    
                                    if skills_match: skills_percent = skills_match.group(1)
                                    if experience_match: experience_percent = experience_match.group(1)
                                    if education_match: education_percent = education_match.group(1)
                                
                                overall_score = overall_score_match.group(1) if overall_score_match else 'N/A'

                                results_with_score.append({
                                    "jd_name": jd_name,
                                    "overall_score": overall_score,
                                    "numeric_score": int(overall_score) if overall_score.isdigit() else -1,
                                    "skills_percent": skills_percent,
                                    "experience_percent": experience_percent, 
                                    "education_percent": education_percent,   
                                    "full_analysis": fit_output
                                })
                            except Exception as e:
                                results_with_score.append({
                                    "jd_name": jd_name,
                                    "overall_score": "Error",
                                    "numeric_score": -1,
                                    "skills_percent": "Error",
                                    "experience_percent": "Error", 
                                    "education_percent": "Error",   
                                    "full_analysis": f"Error running analysis for {jd_name}: {e}\n{traceback.format_exc()}"
                                })
                                
                        results_with_score.sort(key=lambda x: x['numeric_score'], reverse=True)
                        
                        current_rank = 1
                        current_score = -1 
                        
                        for i, item in enumerate(results_with_score):
                            if item['numeric_score'] > current_score:
                                current_rank = i + 1
                                current_score = item['numeric_score']
                            
                            item['rank'] = current_rank
                            del item['numeric_score'] 
                            
                        st.session_state.candidate_match_results = results_with_score
                        
                        st.success("Batch analysis complete!")


            # 3. Display Results
            if st.session_state.get('candidate_match_results'):
                st.markdown("#### Match Results for Your Resume")
                results_df = st.session_state.candidate_match_results
                
                display_data = []
                for item in results_df:
                    full_jd_item = next((jd for jd in st.session_state.candidate_jd_list if jd['name'] == item['jd_name']), {})
                    
                    display_data.append({
                        "Rank": item.get("rank", "N/A"),
                        "Job Description (Ranked)": item["jd_name"].replace("--- Simulated JD for: ", ""),
                        "Role": full_jd_item.get('role', 'N/A'),
                        "Job Type": full_jd_item.get('job_type', 'N/A'),
                        "Fit Score (out of 10)": item["overall_score"],
                        "Skills (%)": item.get("skills_percent", "N/A"),
                        "Experience (%)": item.get("experience_percent", "N/A"), 
                        "Education (%)": item.get("education_percent", "N/A"),   
                    })

                st.dataframe(display_data, use_container_width=True)

                st.markdown("##### Detailed Reports")
                for item in results_df:
                    rank_display = f"Rank {item.get('rank', 'N/A')} | "
                    header_text = f"{rank_display}Report for **{item['jd_name'].replace('--- Simulated JD for: ', '')}** (Score: **{item['overall_score']}/10** | S: **{item.get('skills_percent', 'N/A')}%** | E: **{item.get('experience_percent', 'N/A')}%** | Edu: **{item.get('education_percent', 'N/A')}%**)"
                    with st.expander(header_text):
                        st.markdown(item['full_analysis'])

    # --- TAB 7: Filter JD (Updated from tab6) ---
    with tab6:
        filter_jd_tab_content()
