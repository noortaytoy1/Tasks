# Resume Tailoring App

A simple web app that compares your resume to a target job description and suggests edits to improve alignment. The app computes a similarity score and highlights gaps so you can iteratively tailor your resume until the score improves.

**Live app:** [https://3hsxkqhukzamkuwhxkgpv2.streamlit.app/](https://3hsxkqhukzamkuwhxkgpv2.streamlit.app/)

---

## ✨ What this app does

* Measures textual similarity between your **resume** and a **job description** (JD).
* Surfaces **missing skills/keywords** and **mismatched phrasing**.
* Helps you **iterate**: upload → review feedback → revise your resume → re‑upload until the similarity score increases.

> **Important:** This tool is meant to optimize how clearly your experience matches the posted role. Always keep your resume **truthful** and **verifiable**.

---

## ✅ Quick Start

1. **Prepare your files**

   * Your resume **in `.docx` format** (Word).
   * The job description **in `.docx` format** (copy/paste the JD into a Word file and save as `.docx`).

2. **Open the app**
   Go to **[https://3hsxkqhukzamkuwhxkgpv2.streamlit.app/](https://3hsxkqhukzamkuwhxkgpv2.streamlit.app/)**.

3. **Upload your documents**

   * Click **“Upload Resume (.docx)”** and select your resume.
   * Click **“Upload Job Description (.docx)”** and select the JD file.

4. **Review the analysis**

   * Check the **Similarity Score** (0–100%).
   * Review **missing keywords**, **section‑level coverage**, and **suggested phrasing**.

5. **Tailor & iterate**

   * Edit your resume locally to incorporate relevant, truthful content.
   * Re‑upload your updated resume and **aim to improve the score**.

---

## 📁 File Requirements

* **Accepted format:** `.docx` only (no PDF/Google Docs/Pages).
* **Language:** English recommended.
* **Length:** Up to \~3 pages for best results.
* **Content style:** Plain text in a standard resume layout (avoid heavy tables, text boxes, or images).

---

## 🧠 How the Similarity Score works (at a glance)

* The app computes a composite score based on **keyword overlap**, **semantic similarity**, and **role‑specific terms** (e.g., frameworks, certifications, tools).
* It also weighs **section context** (e.g., Skills vs Experience) to discount irrelevant matches.
* Scores are **directional**—use them to **guide revisions**, not as an absolute measure.

**Typical guidance:**

* **< 50%:** Your resume likely misses core skills, responsibilities, or domain terminology.
* **50–75%:** On the right track; tighten phrasing, add quantifiable impact, mirror the JD’s verbs and nouns where accurate.
* **> 75%:** Strong alignment; focus on clarity, measurable outcomes, and ordering most relevant content first.

---

## 🛠 Tips to Improve Similarity (Ethically)

* **Mirror terminology**: If the JD says “Embedded C/C++” and you have it, phrase it similarly.
* **Prioritize relevance**: Move the most role‑relevant bullets to the top of each section.
* **Quantify impact**: Replace vague bullets with metrics (e.g., “reduced boot time by 32%”).
* **Cover must‑haves**: Certifications, standards (e.g., **AUTOSAR**, **ISO 26262**), toolchains, hardware, libraries.
* **Keep it true**: Never add skills you don’t have; remove or downplay non‑relevant content.

---

## 🔒 Privacy & Security

* Files are processed for the purpose of analysis and tailoring guidance.
* Do **not** upload sensitive information (e.g., SSNs, bank info).
* Remove confidential client identifiers where possible.

---

## 🚦 Common Issues & Troubleshooting

* **“File type not supported”** → Ensure both uploads are **`.docx`**.
* **Weird formatting / missing text** → Re‑save your resume as a simple `.docx` (avoid tables/text boxes); export from Google Docs as `.docx` and re‑try.
* **Very low score despite relevance** → Use more **explicit phrasing** and the **same nouns/verbs** as the JD. Add specific tools (e.g., “QEMU,” “CAN bus,” “Vivado,” “TensorFlow”).
* **Timeout or app won’t load** → Refresh the page or try again later.

---

## 📝 Example Workflow

1. Upload **Resume\_v3.docx** and **JD\_Senior\_Embedded\_Engineer.docx**.
2. Score = **58%**. Feedback shows missing: *ISO 26262*, *CAN bus*, *ASIL*.
3. Update resume bullets truthfully: add your ASIL‑B project, CAN diagnostics, and testing framework.
4. Re‑upload updated resume → Score = **76%**.
5. Polish phrasing and quantify results → Score = **82%** (good!).

---

## ❓FAQ

**Q: Can I upload PDFs?**
A: No, please upload **`.docx`** files for both resume and JD.

**Q: Does a higher score guarantee interviews?**
A: No—scores help with alignment, but outcomes depend on experience, company, and screening process.

**Q: Will the tool write my resume for me?**
A: It won’t fabricate content; it suggests **where** and **how** to align phrasing with the JD.

**Q: Can I use this for multiple roles?**
A: Yes! Tailor a fresh resume for each job. Save each version with a clear name (e.g., `Noor_Embedded_SW_Engineer_AUTOSAR.docx`).

---

## 🧭 Best Practices

* Keep a **master resume** and create **role‑specific variants**.
* Maintain a **Changelog** section at the bottom of your file (locally) during iteration.
* Use **action verbs**: Designed, Implemented, Optimized, Validated, Automated, Deployed.
* Put **most relevant achievements first**.

---
Enjoy!
