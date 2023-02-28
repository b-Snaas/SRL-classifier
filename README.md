## This code preprocesses data files in the conll-u format, and then creates a classifier for Semantic Role Labeling.

## Running Instructions:
Step 1: Libraries installation
>pip install -r requirements.txt

Step 2: Install parser
>python -m spacy download en_core_web_sm

Step 3: Run the code
>python main.py

### The features used in the model are:

1. Words
2. Lemmas
3. Dependency relations
4. Governing predicate
5. POS tag
6. Voice
7. Word Order
8. Named Entities
