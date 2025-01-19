import google.generativeai as genai

class LLMInterface:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    def process_question(self, question):
        try:
            # Add explicit instruction to get yes/no answer
            prompt = f"""
            Please answer the following question with ONLY 'yes' or 'no'. 
            No other words or explanations.
            Question: {question}
            """
            
            response = self.model.generate_content(prompt)
            answer = response.text.strip().lower()
            
            # Only accept 'yes' or 'no' answers
            if answer in ['yes', 'no']:
                return answer
            return None
            
        except Exception as e:
            print(f"Error processing question: {e}")
            return None 