import google.generativeai as genai
import os

class GeminiAnalyst:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            print("[WARN] GEMINI_API_KEY not found. AI commentary will be disabled.")
            self.model = None
            return

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def generate_report(self, ticker, quantitative_data):
        """
        Uses Gemini to generate a text report based on the numbers from our LSTM model.
        """
        if not self.model:
            return "⚠️ AI Commentary unavailable (Missing Key)."

        prompt = f"""
        Act as a Senior Wall Street Analyst.
        Write a concise, high-impact trading update for {ticker} based on this internal model data:
        
        - Current Price: ${quantitative_data['current_price']:.2f}
        - AI Predicted Target: ${quantitative_data['predicted_price']:.2f}
        - Direction: {quantitative_data['direction']}
        - Model Confidence: {quantitative_data['model_confidence']:.1f}%
        - Expected Move: {quantitative_data['p_change']:.2f}%
        
        Instructions:
        1. Keep it under 500 characters.
        2. Use professional financial terminology (e.g., "bullish divergence", "consolidation", "risk-off").
        3. Be decisive but mention the confidence level.
        4. Add 2-3 emojis relevant to the sentiment.
        5. Do NOT mention "LSTM" or technical inner workings, just "our models".
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"⚠️ AI Analysis failed: {str(e)}"

if __name__ == "__main__":
    # Test
    analyst = GeminiAnalyst()
    dummy_data = {
        'current_price': 150.00,
        'predicted_price': 155.00,
        'direction': 'UP',
        'model_confidence': 85.0,
        'p_change': 3.33
    }
    print(analyst.generate_report("TEST", dummy_data))
