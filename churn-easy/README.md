# Customer Churn & Product Engagement Analytics Platform

Predicts whether a customer will leave, using how they use the product.
Input columns: tenure_months, monthly_fee, logins_per_month, features_used, support_tickets, plan (0=Basic, 1=Standard, 2=Premium)

## Files
- data/customers.csv : dataset
- train.py : preprocess + train + tune + save model
- app.py : Flask API + logging + /metrics
- Dockerfile : container
- .github/workflows/ci.yml : CI pipeline

## Run
    pip install -r requirements.txt
    python train.py
    python app.py

Test (new terminal):
    curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"tenure_months":5,"monthly_fee":80,"logins_per_month":2,"features_used":2,"support_tickets":4,"plan":0}'

Monitoring: open http://localhost:5000/metrics and check predictions.log

## Docker
    docker build -t churn-api .
    docker run -p 5000:5000 churn-api

## Git + DVC (version control)
    git init
    dvc init
    dvc add data/customers.csv model.pkl
    git add . 
    git commit -m "first version"
    git push   (CI runs automatically on GitHub)

## Syllabus mapping
- Exp 1 Preprocessing -> train.py steps 1-3
- Exp 3 Decision Tree + precision/recall -> train.py step 4
- Exp 4 GridSearchCV -> train.py step 5
- Exp 5 Git -> git init
- Exp 6 Docker -> Dockerfile
- Exp 7 Flask API -> app.py
- Exp 8 CI/CD -> ci.yml
- Exp 9 DVC -> dvc add
- Exp 10 Monitoring/logging -> /metrics + predictions.log
- Ethics -> no personal data (name, gender) used
