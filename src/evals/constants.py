# We used a weaker model for synthesis and a stronger model for grading to ensure fairness.
SYNTHESIS_MODEL = "gpt-5.6-luna"
GRADER_MODEL = "gpt-5.4-mini"
FIN_SEARCH_GRADER_MODEL = "gpt-5-mini"

# Maximum tokens available for search results (optimized for gpt-5.4-nano and leaving room for prompt and response)
MAX_SEARCH_RESULT_TOKENS = 265000

SYNTHESIS_PROMPT = """
    You are an AI assistant that answers questions using search results.
    Read the provided search results carefully and answer based only on information found in the results.
    Keep your response clear and concise.
"""

# Prompt is from OpenAI's simple-evals repository https://github.com/openai/simple-evals/blob/ee3b0318d8d1d9d72755a4120879be65f7c07e9e/simpleqa_eval.py#L13
SIMPLEQA_ANSWER_GRADER_TEMPLATE = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.


Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k". 
    - Predicted answers "120k", "124k", and 115k" are all CORRECT. 
    - Predicted answers "100k" and "113k" are INCORRECT. 
    - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name. 
    - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".


Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
""".strip()

FRAMES_ANSWER_GRADER_TEMPLATE = """
===Task===


I need your help in evaluating an answer provided by an LLM against a ground truth
answer. Your task is to determine if the ground truth answer is present in the LLM’s response.
Please analyze the provided data and make a decision.


===Instructions===


1. Carefully compare the "Predicted Answer" with the "Ground Truth Answer".
2. Consider the substance of the answers – look for equivalent information or correct answers. Do
not focus on exact wording unless the exact wording is crucial to the meaning.
3. Your final decision should be based on whether the meaning and the vital facts of the "Ground
Truth Answer" are present in the "Predicted Answer:"


===Input Data===


- Question: {question}


- Predicted Answer: {predicted_answer}


- Ground Truth Answer: {target}


===Output Format===


Provide your final evaluation in the following format:
"Explanation:" (How you made the decision?)
"Decision:" ("TRUE" or "FALSE")
Please proceed with the evaluation.
"""

# Prompt is from the DeepSearchQA evaluation methodology
DEEPSEARCHQA_GRADER_TEMPLATE = """Your task is to evaluate whether a given "AI Response" for a specific "User Prompt" arrived at the correct answer.

**Answer Correctness Task**

*   **Purpose:** Assess whether the AI response provides the correct answer(s) based on the provided "Correct Answer" and "Prompt Type".
*   **Process:**
    *   Identify the "Prompt Type": "<prompt_type>".
    *   Refer to the "Correct Answer": "<answer>".
    *   Based on the "Prompt Type", determine if the "AI Response" contains the expected answer(s).
        *   **'Single Answer'**: Check if the response provides the answer that addresses the user's question. It does not have to match the exact wording of the provided answer.
        *   **'Set Answer'**: Check if the response includes *each* item from the provided ground truth answers. The order might not matter unless specified otherwise. The response might include more answers than the list. Determine the correctness *only* based on the list first and then check if the response includes answers not in the list.
    *   **Explanation:** Provide a brief explanation justifying your assessment of answer correctness, referencing specific parts of the AI response and the correct answer.
    *   **Correctness Details:** Provide a dictionary, one key for each expected answer part, and value is a boolean indicating whether each expected answer part was found.
        *   For 'Set Answer', this will be a list of attributes, one for each item/part in the "Correct Answer". Each key will be a string indicating the expected answer part, and the value will be a boolean indicating whether that part was found in the response.
    *   **Excessive Answers:** Provide a list of strings, each indicating an excessive answer part. If the response provides answers that are **not** in the "Correct Answer" list, add these answers as excessive answers. Return an empty list when there's no excessive answers in the response.


**Output Format:**

Your evaluation *must* be structured as a nested JSON dictionary with the following top-level keys: `"Answer Correctness"`. Please return NULL if any of "Prompt", "AI Response" or "Correct Answer" is empty.
The value for `"Answer Correctness"` should be a dictionary containing `"Explanation"` (a string), `"Correctness Details"` (a dictionary where each key is the expected correct answer, and the value is a boolean indicating whether the response contains the correct answer), and `"Excessive Answers"` (a list of strings indicating the excessive answers).

Make sure you return a valid JSON string. Pay special attention to quotes, commas and special characters in the JSON string. Make sure to escape all special characters and quotes in the JSON string.


**Example (Partial):**

"```json
{{
  "Answer Correctness": {{
    "Explanation": "The response correctly identified Belgium and France but also includes an excessive answer, Italy.",
    "Correctness Details": {{
      "Belgium": true,
      "France": true,
    }},
    "Excessive Answers": [ "Italy" ]
  }}
}}
```"

**Now, proceed with the evaluation using the provided User Prompt, AI Response, and Correct Answer.**

User Prompt (Wrapped in <prompt> and </prompt>):
<prompt>
{input}
</prompt>
--------------------
**  Correct Answer (Wrapped in <answer> and </answer>):
Prompt Type: {answer_type}
<answer>
{expected}
</answer>
--------------------
AI assistant response (Wrapped in <response> and </response>):
<response>
{output}
</response>

--------------------
Rating:""".strip()

# Prompt is the original FinSearchComp scoring prompt from https://arxiv.org/pdf/2509.13160
FIN_SEARCH_GRADER_TEMPLATE = """
You are an intelligent judge and scorer for answers to financial questions. You will receive a <Question>, its <Reference
Answer>, and a <Student Answer>. Some <Reference Answer>s may be supplemented with "Scoring Criteria". You
need to evaluate the <Student Answer> and complete the following tasks:
1. Based on the content of the <Student Answer>, accurately identify its final answer (identification only,
no need to output). You can identify the position and content of the final answer by analyzing the <Student Answer>
or by searching for keywords, including but not limited to "the answer is," "the final result is," "the correct option is,"
etc. If the <Student Answer> is empty, meaning it contains no content, assign a score of 0 directly and skip steps 2 and
3 below.
2. Separately list the final answer from the <Reference Answer> and the final answer you identified from the <Student
Answer>, and compare the two (no need to output the listing and comparison process or results).
3. Based on the result of the comparison and any Scoring Criteria that may be provided with the <Reference Answer>, judge whether the <Student Answer> is correct and assign a score. The score can only be 1 or
0; 1 indicates the <Student Answer> is correct, and 0 indicates it is incorrect. No scores other than 0 and 1 are permitted.
**Notes:**
1. You do not need to and should not answer or solve the question yourself. Your sole task is to judge and
score.
2. The <Reference Answer> is accurate and correct. You can fully trust it.
3. If the <Reference Answer> contains 2 or more key points, such as a person's name and a number, an item and a
time, or multiple parallel locations, the <Student Answer> can only receive 1 point if it provides all the key points and
all of them are correct. If only a portion of the key points are provided or are correct, the score is 0.
4. If the <Student Answer> is empty or an abnormal error message, please assign a score of 0.
5. Numerical values of the same magnitude written in different formats are considered the same, for example, 12.45% vs
0.1245, or 120,400,000 vs 120.4 million, or 2/5 vs 0.4. If the <Reference Answer> and the <Student Answer> use
different formats but the underlying numbers represent the same value, they are considered consistent.
6. An answer is deemed correct if its numerical component matches the <Reference Answer>, even if the unit is
omitted. For example, if the <Question> asks for "xx revenue in billions of yuan", a <Student Answer> of "12.98" is a
valid match for the <Reference Answer> "12.98 billion yuan". Similarly, if the required unit is specified as "%" and the
<Reference Answer> is "12.98%", a <Student Answer> of "12.98" is also a valid match.
**Regarding your output format:**
You need to first output the evaluation result, then provide the scoring basis and the score. Your output format is:
- Scoring Basis: (Briefly state your reason for the score, not exceeding 150 words or 150 tokens)
- JSON:
```
{{"answer_score": your score}}
```
###
**Example 1**
<Question>:
What is the average number of years it took for the C9 League university presidents who spoke at the 2024 graduation
ceremonies to become a president of a university for the first time after receiving their PhD? (Calculate in years, round
to one decimal place) <Reference Answer>:
27.9 years
Scoring Criteria:
The number in the Student Answer must be strictly identical to the number in the Reference Answer.
<Student Answer>:
Based on known information, the average time for the current presidents of C9 League universities to become a president
for the first time after their PhD is approximately 27.3 years. Here is the specific calculation:
1. Peking University: President Gong Qihuang, PhD in 1992, became president in 2022, a gap of 30 years.
2. Tsinghua University: President Wang Xiqin, PhD in 1996, became president in 2022, a gap of 26 years.
3. Fudan University: President Jin Li, PhD in 1994, became president in 2021, a gap of 27 years.
4. Shanghai Jiao Tong University: President Ding Kuiling, PhD in 1990, became president in 2021, a gap of 31 years.
5. Nanjing University: President Lü Jian, PhD in 1993, became president in 2018, a gap of 25 years.
6. University of Science and Technology of China: President Bao Xinhe, PhD in 1991, became president in 2017, a gap
of 26 years.
7. Xi'an Jiaotong University: President Wang Shuguo, PhD in 1986, became president in 2014, a gap of 28 years.
8. Zhejiang University: President Wu Zhaohui, PhD in 1993, became president in 2015, a gap of 22 years.
9. Harbin Institute of Technology: President Han Jiecai, PhD in 1991, became president in 2022, a gap of 31 years.
Summing the years:
30 + 26 + 27 + 31 + 25 + 26 + 28 + 22 + 31 = 246
Average time:
246 ÷ 9 = 27.3 years
Therefore, on average, it took these presidents approximately 27.3 years to become a university president for the first
time after their PhD.
- Scoring Basis: This question has Scoring Criteria, which require the number in the Student Answer to be strictly
identical to the number in the Reference Answer. Here, the number in the <Student Answer> is not strictly identical to
the number in the <Reference Answer>. Therefore, the <Student Answer> is incorrect, and the score is 0.
- JSON:
```
{{"answer_score": 0}}
```
###
**Example 2**
<Question>:
An interesting aspect of the United States is that a state's political center is often not its economic center. For example,
the capital of California is the relatively lesser-known Sacramento, not the more economically developed and famous
San Francisco or Los Angeles. In contrast, in China, the provincial capital of each province (excluding autonomous
regions) is often the city with the highest GDP in that province. Using the 2022 GDP as the standard, please list the
provinces in China where the city with the highest GDP within the province is not the provincial capital.
<Reference Answer>:
Hebei Province, Liaoning Province, Jiangsu Province, Shandong Province, Guangdong Province
Scoring Criteria:
The student's answer must list all the provinces from the reference answer and no other provinces.
<Student Answer>:
According to 2022 GDP data, the provinces in China (excluding autonomous regions) where the city with the highest
GDP is not the provincial capital include:
1. Hebei Province: The city with the highest GDP is Tangshan, while the capital is Shijiazhuang.
2. Liaoning Province: The city with the highest GDP is Dalian, while the capital is Shenyang.
3. Heilongjiang Province: The city with the highest GDP is Daqing, while the capital is Harbin.
4. Jiangsu Province: The city with the highest GDP is Suzhou, while the capital is Nanjing.
5. Fujian Province: The city with the highest GDP is Quanzhou, while the capital is Fuzhou.
6. Shandong Province: The city with the highest GDP is Qingdao, while the capital is Jinan.
7. Guangdong Province: The city with the highest GDP is Shenzhen, while the capital is Guangzhou.
These provinces' economic centers are often concentrated in non-capital cities, reflecting different regional economic
development patterns.
- Scoring Basis: This question has Scoring Criteria, which require the Student Answer to list all the provinces from the
Reference Answer and no other provinces. Here, the <Student Answer> lists all the provinces from the <Reference
Answer> but also includes provinces not in the <Reference Answer> (Heilongjiang Province, Fujian Province).
Therefore, the <Student Answer> is incorrect, and the score is 0.
- JSON:
```
{{"answer_score": 0}}
```
###
**Example 3**
<Question>:
How many Michelin three-star restaurants in San Francisco specialize in Asian cuisine, and what are their names?
<Reference Answer>:
1, Benu
<Student Answer>:
Among San Francisco's Michelin three-star restaurants, Benu is the one that specializes in Asian-inspired cuisine.
Helmed by renowned chef Corey Lee, Benu offers modern Asian cuisine, fully showcasing Lee's creativity and skill. The
dishes blend Korean, Chinese, and other Asian elements with a high degree of innovation and refinement. According to
current information, there is 1 Michelin three-star restaurant in San Francisco specializing in Asian cuisine.
- Scoring Basis: This question does not have Scoring Criteria. Here, the <Student Answer> states that there is 1
Michelin three-star restaurant specializing in Asian cuisine in San Francisco, and its name is Benu. Although the
wording is different from the <Reference Answer>, the meaning is the same. Therefore, the <Student Answer> is
correct, and the score is 1.
- JSON:
```
{{"answer_score": 1}}
```
###
**Example 4**
<Question>:
In terms of area, which provincial capital city in China (mainland provinces) is closest in size to Slovenia? What is the
difference in square kilometers?
<Reference Answer>:
It is closest to Kunming, with a difference of about 740 square kilometers.
<Student Answer>:
The total area of Slovenia is 20,273 square kilometers. Among China's provincial capitals, the one closest in area is
Kunming, with an area of 22,100 square kilometers. The difference between the two is: 22,100 - 20,273 = 1827 square
kilometers.
- Scoring Basis: This question does not have Scoring Criteria. Here, the <Student Answer> correctly identifies Kunming
as being closest in area to Slovenia, which is consistent with the <Reference Answer>. However, the area difference of
"1827 square kilometers" given in the <Student Answer> is inconsistent with "about 740 square kilometers" in the
<Reference Answer>. Therefore, the <Student Answer> is incorrect, and the score is 0.
- JSON:
```
{{"answer_score": 0}}
```
###
I hope you are competent for this job, as it is very important to me. If you perform well, I will reward you accordingly.
Otherwise, I may penalize you.

<Question>:
{question}
<Reference Answer>:
{target}
<Student Answer>:
{predicted_answer}
""".strip()

# Prompt is from OpenAI's simple-evals repository https://github.com/openai/simple-evals/blob/main/browsecomp_eval.py
BROWSECOMP_GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.
""".strip()
