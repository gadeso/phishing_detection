## Phishing Detection Web Application

#### 📖 Overview
The Phishing Detection Web Application is an interactive tool developed to identify potentially fraudulent websites. Leveraging machine learning techniques, this application analyzes URLs to assess their likelihood of being phishing sites. Built with Streamlit, it offers a user-friendly interface for real-time URL analysis.
Presentation: https://prezi.com/view/bxFXrHb94h3MX1WB0KqP/

#### 🛠 Features
Interactive Interface: Easily enter and analyze URLs through a web-based interface.
Machine Learning Model: Utilizes a Random Forest Classifier to predict phishing attempts.
Real-Time Analysis: Provides instant feedback on the safety of entered URLs.

#### 🚀 Getting Started
Follow these instructions to set up and run the application on your local machine.

#### Prerequisites
Ensure you have the following installed:

Python 3.7 or higher
pip (Python package installer)
Installation
Clone the Repository

bash
git clone https://github.com/your-username/phishing-detection-web-app.git
cd phishing-detection-web-app
Create a Virtual Environment

bash
python -m venv venv
Windows:

bash
venv\Scripts\activate
macOS/Linux:

bash
source venv/bin/activate
Install Dependencies

bash
pip install -r requirements.txt
Running the Application
Start the Streamlit Server

bash
streamlit run app.py
Access the Web Interface

Open your web browser and navigate to http://localhost:8501 to use the application.

#### 💻 Usage
Enter a URL: In the input field on the web interface, type or paste the URL you want to check.
Analyze: Click the submit button to get the phishing prediction.
View Results: The application will indicate whether the URL is potentially fraudulent or legitimate.

#### 📊 Model Details
Model: Random Forest Classifier
Features:
Number of dots in the URL
Length of the URL
Number of dashes in the URL
Presence of '@' symbol
Presence of IP address in the URL
HTTPS in hostname
Path length and level
Number of numeric characters

#### 🤝 Contributing
We welcome contributions to enhance the application. Please fork the repository, create a new branch, and submit a pull request with your changes. For significant changes, please open an issue first to discuss the potential modifications.