Here is your **updated professional README.md** with both required workflows clearly added:

---

# 🧠 AI Scrum Assistant using FLAN-T5 + QLoRA

## 📄 README.md

# 🚀 Scrum AI System – Requirement to User Stories & Module Breakdown

This project fine-tunes **FLAN-T5-Base** using **QLoRA (Quantized LoRA)** to build an intelligent **Agile Scrum assistant** that transforms:

### 🔄 Two-Level Conversion Pipeline:

1. 🧾 **Raw Requirements → Structured User Stories**
2. 🧩 **Structured User Stories → Module Breakdown**

This creates a complete **AI-powered software requirement engineering pipeline**.

---

# 🎯 Objective

The system automates Agile documentation by:

* Converting **unstructured user requirements** into **well-formatted user stories**
* Transforming **user stories into system modules**
* Helping teams accelerate **sprint planning and system design**
* Reducing manual effort in requirement analysis

---

# ⚙️ Key Features

* 🔥 FLAN-T5-Base fine-tuned with QLoRA
* ⚡ Lightweight adapter-based model
* 🧠 Two-stage generation pipeline:

  * Raw Requirement → User Stories
  * User Stories → Module Breakdown
* 💬 Streamlit-based interactive UI
* 📦 Efficient inference with LoRA adapters

---

# 🔁 System Workflow

## 🟢 Stage 1: Requirement → User Stories

The model converts raw system requirements into structured Agile format.

### Example Input:

```text id="req1"
Users should be able to register, login, and reset passwords.
Admins should manage users and monitor system activity.
The system should send notifications to users.
```

### Output (User Stories):

```text id="us1"
As a user, I want to register and login so that I can access the system securely.
As a user, I want to reset my password so that I can recover my account.
As an admin, I want to manage users so that I can control system access.
As a user, I want to receive notifications so that I stay updated.
```

---

## 🔵 Stage 2: User Stories → Module Breakdown

The structured user stories are further converted into system design components.

### Input:

```text id="us_input"
User stories generated from Stage 1
```

### Output (Module Breakdown):

```text id="mod1"
Frontend Module:
- Login Page
- Registration Page
- Password Reset Page
- Notification UI

Backend Module:
- Authentication Service
- User Management API
- Notification Service

Database Module:
- Users Table
- Sessions Table
- Notifications Table
```

---

# 🧠 Model Architecture

## 🔷 Base Model

* FLAN-T5-Base (Text-to-Text Transformer)

---

## 🔷 Fine-Tuning Method: QLoRA

* 4-bit quantization
* LoRA adapter training
* Efficient memory usage
* High-quality instruction tuning

---

# 📁 Project Structure

```bash id="proj1"
Scrum-AI-Assistant/
│
├── app.py
├── requirements.txt
│
├── adapter_config.json
├── adapter_model.safetensors
│
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
│
└── .github/
```

---

# 💻 Application (Streamlit)

## Features:

* Input raw requirements
* Generate user stories
* Generate module breakdown
* Two-step pipeline inference

---

# 🚀 Run Application

```bash id="run1"
streamlit run app.py
```

---

# ⚙️ Training Setup

| Parameter     | Value        |
| ------------- | ------------ |
| Base Model    | FLAN-T5-Base |
| Method        | QLoRA        |
| Precision     | 4-bit        |
| Optimizer     | AdamW        |
| Batch Size    | 4–16         |
| Learning Rate | 2e-4         |

---

# 🧾 Supported Tasks

## 🔹 Task 1: Requirement Engineering

* Convert raw requirements → structured user stories

## 🔹 Task 2: System Design Assistance

* Convert user stories → module breakdown

---

# 📊 Advantages

* Automates Agile documentation
* Improves requirement clarity
* Speeds up sprint planning
* Reduces manual engineering workload
* Works in real-time via web app

---

# 📌 Limitations

* Needs clear input requirements
* Performance depends on training data quality
* May generalize poorly for highly complex systems

---

# 🔮 Future Improvements

* Jira / Trello integration
* Epic & Sprint generation
* RAG-based requirement enhancement
* Multi-language support
* Larger FLAN-T5 models (Large / XXL)

---

# 🎯 Conclusion

This project demonstrates a complete **AI-powered Scrum automation pipeline**:

### 🔄 Full Flow:

```
Raw Requirements
        ↓
Structured User Stories
        ↓
Module Breakdown
```

Using **FLAN-T5 + QLoRA**, the system provides a lightweight yet powerful solution for Agile requirement engineering.

---

# 👨‍💻 Author

**Mubashir Siddique**

AI / NLP / Generative AI Enthusiast

---

# 📜 License

For educational and research purposes only.
