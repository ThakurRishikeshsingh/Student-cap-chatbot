import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

class StudentCapChatbot:
    def __init__(self):
        print("Initializing AI Student Assistant... (Loading NLP Model)")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.knowledge_base = [
            {
                "intent": "Introduction to each other",
                "examples": ["Hi", "Hello", "Hey there"],
                "response": "Hi, I am JSPM University virtual agent. How may I help you?"
            },
            {
                "intent": "admission_deadline",
                "examples": ["When is the application deadline?", "Last date to apply for fall?", "Admission closing date"],
                "response": "The fall semester application deadline is July 15th, 2026. Spring applications close on November 1st."
            },
            {
                "intent": "admission_requirements",
                "examples": ["What documents do I need for admission?", "Admission requirements?", "How do I apply?"],
                "response": "You need your high school transcripts, a personal statement, two recommendation letters, and a $50 application fee submitted via our portal."
            },
            {
                "intent": "exam_schedule",
                "examples": ["When are midterms?", "Final exam timetable", "When do exams start?"],
                "response": "Spring finals are scheduled from June 8th to June 12th, 2026. You can download your personalized schedule on the Student Portal under 'Exams'."
            },
            {
                "intent": "exam_policies",
                "examples": ["What if I miss an exam?", "Sick during finals", "Makeup exam policy"],
                "response": "If you miss an exam due to illness, you must submit a medical certificate to the Academic Registry within 48 hours to request a makeup exam."
            },
            {
                "intent": "library_hours",
                "examples": ["Is the library open?", "Library timings", "When does the library close?"],
                "response": "The main campus library is open Monday through Friday from 7:00 AM to Midnight, and weekends from 9:00 AM to 10:00 PM."
            },
            {
                "intent": "gym_facilities",
                "examples": ["Where is the gym?", "Campus fitness center hours", "Is there a swimming pool?"],
                "response": "The Student Fitness Center is located next to the West Dorms. It features a full gym, indoor pool, and basketball courts. Open daily 6:00 AM - 9:00 PM."
            }
        ]
        
        self.prepared_examples = []
        for idx, item in enumerate(self.knowledge_base):
            for example in item["examples"]:
                self.prepared_examples.append({"kb_index": idx, "text": example})
                
        texts_to_embed = [ex["text"] for ex in self.prepared_examples]
        self.example_embeddings = self.model.encode(texts_to_embed, convert_to_tensor=True)
        print("Chatbot is ready to help!")

    def get_response(self, user_query, threshold=0.45):
        query_embedding = self.model.encode(user_query, convert_to_tensor=True)
        similarities = cos_sim(query_embedding, self.example_embeddings)[0].cpu().numpy()
        
        best_match_idx = np.argmax(similarities)
        best_score = similarities[best_match_idx]
        
        if best_score < threshold:
            return ("I'm sorry, I couldn't quite find a precise answer for that. "
                    "You can try rephrasing, or contact student support directly at support@university.edu.")
        
        kb_item_index = self.prepared_examples[best_match_idx]["kb_index"]
        return self.knowledge_base[kb_item_index]["response"]

if __name__ == "__main__":
    bot = StudentCapChatbot()
    print("\n" + "="*50)
    print("Welcome to the University Assistant. Type 'exit' to quit.")
    print("="*50 + "\n")
    
    while True:
        user_input = input("Student: ")
        if user_input.strip().lower() in ['exit', 'quit']:
            print("Assistant: Goodbye! Good luck with your studies.")
            break
            
        if not user_input.strip():
            continue
            
        bot_response = bot.get_response(user_input)
        print(f"Assistant: {bot_response}\n")
