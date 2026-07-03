# AllLife Bank a Personal Loan Predictor

Interactive machine learning app that predicts whether an AllLife Bank customer 
will accept a personal loan offer, powered by a decision tree classifier and 
Claude-generated plain-English explanations for each prediction.

## Live Demo

[▶ Open Interactive App](https://alllife-bank-loan-prediction-djxtd4anfrajghnyrjzwsz.streamlit.app/)

## Business Problem

AllLife Bank wants to convert liability customers (depositors) into asset customers 
(borrowers) by identifying which existing customers are most likely to accept a 
personal loan offer. This model helps relationship managers prioritize outreach 
and understand what drives each customer's predicted behavior.

## What It Does

Enter a customer profile (age, income, education, credit card spending, banking 
relationships) and the app returns:

- **Loan acceptance prediction** (accept / decline)
- **Confidence score** — probability of acceptance
- **Feature importance chart** — which factors drove the model's decision
- **Claude-powered explanation** — plain-English briefing a relationship manager 
  can act on before a customer call

## Key Findings from the Analysis

- **Income and CCAvg are the two strongest predictors**  high earners who spend 
  more on credit cards are far more likely to accept
- **CD account holders are ~5× more likely to accept** a personal loan offer
- **Graduate and advanced-degree customers** accept at significantly higher rates 
  than undergraduates
- **Family size has minimal impact** the proportion of loan acceptance remains 
  consistent across family sizes
- **Age and Experience are redundant** (r = 0.99)  Experience was dropped from 
  the model

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 98.4% |
| Recall | 87.7% |
| Precision | 96.4% |
| F1 Score | 91.8% |

Evaluation metric focus: **F1 score and recall**, because missing a genuine loan 
acceptor costs the bank more than a false positive. Pre-pruned decision tree 
used to reduce overfitting.

## Running Locally

```bash
git clone https://github.com/xmashaxxx/alllife-bank-loan-prediction.git
cd alllife-bank-loan-prediction
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
streamlit run app.py
```

The dataset (`Loan_Modelling.csv`) is included for demo purposes.  
Claude explanations require an Anthropic API key; the prediction and charts 
work without one.

## Dataset

See [`data/README.md`](data/README.md) for column structure.  
Data provided as part of MIT IDSS coursework.

## Skills Demonstrated

`Python` `scikit-learn` `Decision Tree` `pandas` `plotly` `Streamlit` 
`Anthropic API` `Claude Haiku` `EDA` `Classification` `ML Explainability`
