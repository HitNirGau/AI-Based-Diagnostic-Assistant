import google.generativeai as genai

# Configure Gemini API with your API key
genai.configure(api_key='AIzaSyALXzGMeaViJQfJT9cx6KR7ac8Ef-uD5uM')

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-pro')

def generate_findings_explanation(findings):
    prompt = f"""
    Explain the following medical findings in a clear and patient-friendly way:
    {findings}
    """
    response = model.generate_content(prompt)
    return response.text

def generate_recommendations(prediction):
    prompt = f"""
    Based on the diagnosis of {prediction}, suggest follow-up actions and medical recommendations.
    """
    response = model.generate_content(prompt)
    return response.text

def generate_emotional_support(prediction):
    prompt = f"""
    Provide empathetic and supportive words for a patient diagnosed with {prediction}.
    """
    response = model.generate_content(prompt)
    return response.text

def generate_augmented_report(model_prediction, findings):
    return {
        "Diagnosis": model_prediction,
        "Findings Explanation": generate_findings_explanation(findings),
        "Recommendations": generate_recommendations(model_prediction),
        "Emotional Support": generate_emotional_support(model_prediction)
    }

# # Example usage
# model_prediction = "Malignant Breast Cancer"
# findings = "Irregular mass detected with spiculated margins and microcalcifications."

# augmented_report = generate_augmented_report(model_prediction, findings)
# for section, content in augmented_report.items():
#     print(f"{section}:\n{content}\n")
