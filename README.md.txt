# Student Support Assistant Chatbot

A semantic-search campus assistant chatbot built using Python and SentenceTransformers. Unlike traditional keyword-matching chatbots, this assistant utilizes NLP embeddings to understand user intent, allowing it to answer queries accurately even if a student phrases their question in an unexpected way.

## 🚀 Features
* **Semantic Understanding:** Powered by the `all-MiniLM-L6-v2` transformer model to calculate cosine similarity matches.
* **Pre-computed Embeddings:** Intent examples are embedded at startup, making live inference and user interaction incredibly fast.
* **Fallback Safety Net:** Includes a confidence threshold check (`0.45`) to prevent the chatbot from confidently guessing answers to unrelated questions.
* **Preloaded Knowledge Base:** Out-of-the-box support for common campus categories like Admissions, Exams, and Campus Facilities.

## 🛠️ Installation & Setup

1. **Clone the repository:**
```bash
   git clone [https://github.com/YOUR_USERNAME/student-cap-chatbot.git](https://github.com/YOUR_USERNAME/student-cap-chatbot.git)
   cd student-cap-chatbot